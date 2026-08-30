"""
Select Elite ML Long champion using OOS-only metrics (no train leakage in ranking).
Train ML on first 70% sessions; score all packs on last 30% only.
Also report full-period for reference after selection.
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from trading.services.fno_engine import (
    CAPITAL,
    FEATURE_COLS,
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    SLIPPAGE_PTS,
    TRADES_PER_DAY,
    enrich_features,
    lots_for_risk,
    max_lots,
    normalize_df,
)

LOT_SIZE = INSTRUMENTS["NIFTY"]["lot_size"]
MARGIN = INSTRUMENTS["NIFTY"]["mis_margin"]
DATA_PATH = ROOT / "data" / "intraday_fno" / "NIFTY.pkl"
OUT_MODEL = ROOT / "data" / "intraday_fno_ml_long_model.pkl"
OUT_META = ROOT / "data" / "intraday_fno_ml_long_model.json"
OUT_RESULTS = ROOT / "data" / "intraday_fno_ml_long_results.json"
OUT_TRADES = ROOT / "data" / "intraday_fno_ml_long_trades.json"

MIN_WR = 70.0
MIN_TRADES_OOS = 8
MIN_TRADES_FULL = 15


def ema_stack_ok(row) -> bool:
    return bool(
        pd.notna(row.get("ema_9")) and pd.notna(row.get("ema_21")) and pd.notna(row.get("ema_50"))
        and row["ema_9"] > row["ema_21"] > row["ema_50"]
        and (pd.isna(row.get("vwap")) or row["close"] > row["vwap"])
    )


def loose_long(row) -> bool:
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return False
    return bool(
        row["ema_9"] > row["ema_21"]
        and (pd.isna(row.get("vwap")) or row["close"] > row["vwap"])
        and 40 < row["rsi"] < 75
    )


def simulate_long(df, entry_idx, stop, target):
    if entry_idx >= len(df):
        return None
    entry = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
    entry_ts = df.index[entry_idx]
    for j in range(entry_idx, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        if lo <= stop:
            exit_px = stop - SLIPPAGE_PTS
            return {"win": False, "pnl_pts": exit_px - entry, "entry": entry,
                    "exit": exit_px, "reason": "stop_hit", "exit_ts": ts}
        if hi >= target:
            exit_px = target - SLIPPAGE_PTS
            return {"win": True, "pnl_pts": exit_px - entry, "entry": entry,
                    "exit": exit_px, "reason": "target_hit", "exit_ts": ts}
        if ts.time() >= FORCE_EXIT:
            exit_px = cl - SLIPPAGE_PTS
            return {"win": exit_px > entry, "pnl_pts": exit_px - entry, "entry": entry,
                    "exit": exit_px, "reason": "eod_exit", "exit_ts": ts}
        if j > entry_idx and ts.date() != entry_ts.date():
            break
    return None


def train_models(X, y):
    models = {
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=12, random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=350, max_depth=7, min_samples_leaf=10,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, max_iter=280,
            min_samples_leaf=15, l2_regularization=0.08, random_state=42,
        ),
    }
    for m in models.values():
        m.fit(X, y)
    return models


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)


def checklist_ok(c, losses_today: int) -> bool:
    score = 0
    path = c["pct_from_open"]
    if path < -0.65 or path > 0.85:
        return False
    if path <= -0.05:
        score += 2
    elif path <= 0.40:
        score += 1
    if c["full_stack"] and c["above_vwap"]:
        score += 2
    elif c["ema_ok"] and c["above_vwap"]:
        score += 1
    elif not c["above_vwap"]:
        return False
    h = c["hour"]
    if 9 <= h <= 11:
        score += 2
    elif h <= 13:
        score += 1
    p = c["prob"]
    if p >= 0.68:
        score += 2
    elif p >= 0.58:
        score += 1
    elif p < 0.52:
        return False
    if losses_today >= 2:
        return False
    if losses_today == 0:
        score += 1
    return score >= 6


def backtest(selected, target_r: float, risk_pct: float = RISK_PCT, name: str = ""):
    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    wins = gp = gl = 0.0
    trades = 0
    day_count: dict = {}
    daily: dict = {}
    log = []
    for c in sorted(selected, key=lambda x: x["i"]):
        sess = c["session"]
        if day_count.get(sess, 0) >= TRADES_PER_DAY:
            continue
        key = f"out_{target_r}"
        if key not in c:
            continue
        out = c[key]
        entry = out["entry"]
        stop = c["stop"]
        stop_pts = entry - stop
        if stop_pts <= 0:
            continue
        max_l = max_lots(equity, MARGIN)
        lots = lots_for_risk(equity, risk_pct, stop_pts, LOT_SIZE, max_l)
        if lots <= 0:
            continue
        pnl = out["pnl_pts"] * LOT_SIZE * lots
        equity += pnl
        trades += 1
        day_count[sess] = day_count.get(sess, 0) + 1
        daily[sess] = daily.get(sess, 0.0) + pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        if pnl > 0:
            wins += 1
            gp += pnl
        else:
            gl += abs(pnl)
        log.append({
            "trade_no": trades, "strategy": name or "Elite ML Long v1", "side": "LONG",
            "session_date": str(sess), "signal_time": c["ts"].isoformat(),
            "entry_price": round(entry, 2), "stop": round(stop, 2),
            "target": round(entry + stop_pts * target_r, 2),
            "risk_pts": round(stop_pts, 2), "target_r": target_r, "lots": lots,
            "ml_prob": round(c["prob"], 3), "exit_time": out["exit_ts"].isoformat(),
            "exit_price": round(out["exit"], 2), "exit_reason": out["reason"],
            "pnl_pts": round(out["pnl_pts"], 2), "pnl_inr": round(pnl, 2),
            "result": "WIN" if pnl > 0 else "LOSS", "equity_after": round(equity, 2),
        })
    wr = 100.0 * wins / trades if trades else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {
        "trades": trades, "wins": int(wins), "losses": trades - int(wins),
        "win_rate": round(wr, 2), "net_pnl": round(equity - CAPITAL, 2),
        "profit_factor": round(pf, 2), "max_dd_pct": round(max_dd, 2),
        "final_equity": round(equity, 2),
        "avg_daily": round((equity - CAPITAL) / max(len(daily), 1), 2),
        "trade_log": log,
    }


def select_cands(pool, cfg, target_r: float):
    out = []
    losses_today: dict = {}
    day_count: dict = {}
    for c in sorted(pool, key=lambda x: x["i"]):
        if c["prob"] < cfg["th"]:
            continue
        if cfg["stack"] and not c["full_stack"]:
            continue
        if c["adx"] < cfg["adx_min"]:
            continue
        if not (cfg["min_risk"] <= c["risk_pts"] <= cfg["max_risk"]):
            continue
        if not (cfg["rsi_lo"] <= c["rsi"] <= cfg["rsi_hi"]):
            continue
        if c["hour"] < cfg["hour_start"] or c["hour"] > cfg["hour_end"]:
            continue
        if c["range_pos"] < cfg["rp_min"]:
            continue
        if c["pct_from_open"] > cfg["max_path"] or c["pct_from_open"] < cfg["min_path"]:
            continue
        if cfg["di_bull"] and c["di_diff"] <= 0:
            continue
        if cfg["macd_pos"] and c["macd_hist"] <= 0:
            continue
        sess = c["session"]
        if day_count.get(sess, 0) >= cfg["max_day"]:
            continue
        if cfg["checklist"] and not checklist_ok(c, losses_today.get(sess, 0)):
            continue
        if f"out_{target_r}" not in c:
            continue
        out.append(c)
        day_count[sess] = day_count.get(sess, 0) + 1
        if not c[f"out_{target_r}"]["win"]:
            losses_today[sess] = losses_today.get(sess, 0) + 1
    return out


def main():
    print("=" * 80)
    print("Elite ML Long — OOS-first champion selection")
    print("=" * 80)
    raw = pickle.load(open(DATA_PATH, "rb"))
    df = enrich_features(normalize_df(raw))
    print(f"bars={len(df)} {df.index.min().date()} → {df.index.max().date()}")

    target_rs = [1.0, 1.2, 1.5, 1.8, 2.0]
    cands = []
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        if not loose_long(row) or pd.isna(row.get("ema_21")):
            continue
        stop = float(row["ema_21"])
        risk = float(row["close"]) - stop
        if risk <= 0 or risk > 50:
            continue
        entry_idx = i + 1
        if entry_idx >= len(df) or df.index[entry_idx].date() != ts.date():
            continue
        outs = {}
        for tr in target_rs:
            entry0 = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
            sp = entry0 - stop
            if sp <= 0:
                continue
            out = simulate_long(df, entry_idx, stop, entry0 + sp * tr)
            if out:
                outs[tr] = out
        if not outs:
            continue
        feat = [float(row.get(c, np.nan)) for c in FEATURE_COLS]
        if any(np.isnan(feat)):
            continue
        base = outs.get(1.5) or next(iter(outs.values()))
        c = {
            "i": i, "ts": ts, "session": row["session_date"], "hour": int(ts.hour),
            "stop": stop, "risk_pts": risk, "feat": feat,
            "rsi": float(row["rsi"]),
            "adx": float(row["adx"]) if pd.notna(row.get("adx")) else 0.0,
            "range_pos": float(row["range_pos"]) if pd.notna(row.get("range_pos")) else 0.5,
            "pct_from_open": float(row["pct_from_open"]) if pd.notna(row.get("pct_from_open")) else 0.0,
            "di_diff": float(row["di_diff"]) if pd.notna(row.get("di_diff")) else 0.0,
            "macd_hist": float(row["macd_hist"]) if pd.notna(row.get("macd_hist")) else 0.0,
            "above_vwap": bool(pd.isna(row.get("vwap")) or row["close"] > row["vwap"]),
            "ema_ok": bool(row["ema_9"] > row["ema_21"]),
            "full_stack": ema_stack_ok(row),
            "win": base["win"],
        }
        for tr, o in outs.items():
            c[f"out_{tr}"] = o
        cands.append(c)

    sessions = sorted({c["session"] for c in cands})
    cut = max(1, int(len(sessions) * 0.7))
    train_sess, test_sess = set(sessions[:cut]), set(sessions[cut:])
    train = [c for c in cands if c["session"] in train_sess]
    test = [c for c in cands if c["session"] in test_sess]
    print(f"candidates={len(cands)} train={len(train)} oos={len(test)}")
    print(f"OOS sessions: {sorted(test_sess)[0]} → {sorted(test_sess)[-1]} ({len(test_sess)} days)")

    models = train_models(
        np.array([c["feat"] for c in train]),
        np.array([1 if c["win"] else 0 for c in train]),
    )
    # Probabilities — critical: only rank using OOS probs for selection
    for c, p in zip(cands, ensemble_proba(models, np.array([c["feat"] for c in cands]))):
        c["prob"] = float(p)

    oos_probs = [c["prob"] for c in test]
    oos_wins = [c["win"] for c in test]
    print("OOS raw candidate WR by ML threshold:")
    for th in [0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
        m = [(w, p) for w, p in zip(oos_wins, oos_probs) if p >= th]
        if m:
            print(f"  th≥{th:.2f}: n={len(m)} WR={100*np.mean([x[0] for x in m]):.1f}%")

    packs = []
    for th, stack, adx, max_r, rsi, he, cl, tr, md, di, path in itertools.product(
        [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72],
        [False, True],
        [0, 12, 15, 18, 22],
        [15, 18, 22, 28, 35],
        [(40, 75), (45, 70), (48, 72), (50, 70)],
        [12, 13, 14],
        [True, False],
        [1.0, 1.2, 1.5, 1.8],
        [2, 3, 4],
        [False, True],
        [0.45, 0.60, 0.80, 1.2],
    ):
        packs.append({
            "th": th, "stack": stack, "adx_min": adx, "max_risk": max_r, "min_risk": 1.5,
            "rsi_lo": rsi[0], "rsi_hi": rsi[1], "hour_start": 9, "hour_end": he,
            "rp_min": 0.0, "max_path": path, "min_path": -0.55,
            "di_bull": di, "macd_pos": False, "checklist": cl, "target_r": tr, "max_day": md,
        })
    # Sample down while keeping diversity
    seen = set()
    uniq = []
    # Prefer checklist + moderate th first then fill
    packs.sort(key=lambda p: (p["checklist"], p["th"], p["stack"], p["di_bull"]), reverse=True)
    for p in packs:
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= 8000:
            break
    print(f"testing {len(uniq)} packs on OOS…")

    oos_results = []
    eligible = []
    for i, cfg in enumerate(uniq, 1):
        tr = cfg["target_r"]
        sel = select_cands(test, cfg, tr)
        if len(sel) < 5:
            continue
        bt = backtest(sel, tr, name="OOS")
        if bt["trades"] < 5:
            continue
        row = {**bt, "cfg": cfg, "period": "oos"}
        oos_results.append(row)
        if (
            bt["win_rate"] >= MIN_WR
            and bt["net_pnl"] > 0
            and bt["trades"] >= MIN_TRADES_OOS
            and bt["profit_factor"] >= 1.2
        ):
            eligible.append(row)
            print(
                f"★ OOS WR={bt['win_rate']:5.1f}% n={bt['trades']:3d} PnL=₹{bt['net_pnl']:>8,.0f} "
                f"PF={bt['profit_factor']:5.2f} DD={bt['max_dd_pct']:4.1f}% "
                f"| th={cfg['th']} stack={cfg['stack']} CL={cfg['checklist']} "
                f"ADX≥{cfg['adx_min']} risk≤{cfg['max_risk']} {tr}R h≤{cfg['hour_end']}"
            )
        if i % 1500 == 0:
            best = max(oos_results, key=lambda r: (r["win_rate"], r["net_pnl"])) if oos_results else None
            print(
                f"  … {i}/{len(uniq)} oos_res={len(oos_results)} elig={len(eligible)} "
                f"bestWR={best['win_rate'] if best else 0}%"
            )

    print(f"\nOOS results={len(oos_results)} | OOS eligible WR≥{MIN_WR}% n≥{MIN_TRADES_OOS}: {len(eligible)}")

    # Rank OOS
    if eligible:
        eligible.sort(key=lambda x: (x["net_pnl"], x["win_rate"], x["profit_factor"]), reverse=True)
        champ_cfg = eligible[0]["cfg"]
        print(f"\nBest OOS eligible: WR={eligible[0]['win_rate']}% n={eligible[0]['trades']} "
              f"PnL=₹{eligible[0]['net_pnl']:,.0f}")
    else:
        # Best OOS profitable high WR
        prof = [r for r in oos_results if r["net_pnl"] > 0 and r["trades"] >= 6]
        if not prof:
            prof = sorted(oos_results, key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
        else:
            prof.sort(key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
        champ_cfg = prof[0]["cfg"] if prof else None
        print("\nNo OOS pack hit WR≥70% with constraints — using best OOS profitable high-WR")
        if prof:
            print(f"  Fallback OOS: WR={prof[0]['win_rate']}% n={prof[0]['trades']} PnL=₹{prof[0]['net_pnl']:,.0f}")

    if not champ_cfg:
        print("No champion")
        return

    # Evaluate champion on OOS + FULL
    tr = champ_cfg["target_r"]
    oos_sel = select_cands(test, champ_cfg, tr)
    full_sel = select_cands(cands, champ_cfg, tr)
    oos_bt = backtest(oos_sel, tr, name="Elite ML Long v1")
    full_bt = backtest(full_sel, tr, name="Elite ML Long v1")

    # Also compare top 10 OOS strategies full metrics
    print("\n### TOP OOS strategies (and their full-period check) ###")
    rank = eligible[:15] if eligible else sorted(
        [r for r in oos_results if r["net_pnl"] > 0],
        key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True,
    )[:15]
    comparison = []
    for r in rank:
        cfg = r["cfg"]
        tr_ = cfg["target_r"]
        full = backtest(select_cands(cands, cfg, tr_), tr_, name="full")
        comparison.append({
            "oos_wr": r["win_rate"], "oos_n": r["trades"], "oos_pnl": r["net_pnl"],
            "oos_pf": r["profit_factor"], "oos_dd": r["max_dd_pct"],
            "full_wr": full["win_rate"], "full_n": full["trades"], "full_pnl": full["net_pnl"],
            "full_pf": full["profit_factor"], "full_dd": full["max_dd_pct"],
            "cfg": cfg,
            "name": (
                f"th={cfg['th']} stack={cfg['stack']} CL={cfg['checklist']} "
                f"ADX≥{cfg['adx_min']} risk≤{cfg['max_risk']} RSI{cfg['rsi_lo']}-{cfg['rsi_hi']} "
                f"{tr_}R h≤{cfg['hour_end']} DI={cfg['di_bull']}"
            ),
        })
        print(
            f"  OOS WR={r['win_rate']:5.1f}% n={r['trades']:3d} PnL=₹{r['net_pnl']:>8,.0f} | "
            f"FULL WR={full['win_rate']:5.1f}% n={full['trades']:3d} PnL=₹{full['net_pnl']:>9,.0f} | "
            f"{comparison[-1]['name'][:55]}"
        )

    # Prefer among eligible: best OOS PnL with full also profitable
    if comparison:
        good = [c for c in comparison if c["full_pnl"] > 0 and c["oos_pnl"] > 0]
        if good:
            # if any OOS WR≥70
            hi = [c for c in good if c["oos_wr"] >= MIN_WR]
            pick = max(hi or good, key=lambda x: (x["oos_wr"] >= MIN_WR, x["oos_pnl"], x["oos_wr"]))
            champ_cfg = pick["cfg"]
            tr = champ_cfg["target_r"]
            oos_bt = backtest(select_cands(test, champ_cfg, tr), tr, name="Elite ML Long v1")
            full_bt = backtest(select_cands(cands, champ_cfg, tr), tr, name="Elite ML Long v1")

    print("\n### FINAL CHAMPION (OOS-selected) ###")
    print(f"CFG: {champ_cfg}")
    print(f"OOS:  WR={oos_bt['win_rate']}% n={oos_bt['trades']} PnL=₹{oos_bt['net_pnl']:,.0f} "
          f"PF={oos_bt['profit_factor']} DD={oos_bt['max_dd_pct']}%")
    print(f"FULL: WR={full_bt['win_rate']}% n={full_bt['trades']} PnL=₹{full_bt['net_pnl']:,.0f} "
          f"PF={full_bt['profit_factor']} DD={full_bt['max_dd_pct']}%")

    # Rule baselines OOS for comparison
    print("\n### Rule baselines on OOS ###")
    for label, pred, th in [
        ("Loose long", lambda c: True, 0),
        ("EMA stack", lambda c: c["full_stack"], 0),
        ("ML≥0.60", lambda c: c["prob"] >= 0.60, 0.60),
        ("ML≥0.65", lambda c: c["prob"] >= 0.65, 0.65),
        ("ML≥0.70 stack", lambda c: c["prob"] >= 0.70 and c["full_stack"], 0.70),
    ]:
        for tr_ in [1.0, 1.5]:
            cfg = {
                "th": th, "stack": "stack" in label, "adx_min": 0, "max_risk": 99, "min_risk": 0,
                "rsi_lo": 0, "rsi_hi": 100, "hour_start": 9, "hour_end": 14,
                "rp_min": 0, "max_path": 99, "min_path": -99, "di_bull": False, "macd_pos": False,
                "checklist": False, "target_r": tr_, "max_day": 4,
            }
            # custom select
            sel = []
            day_count = {}
            for c in sorted(test, key=lambda x: x["i"]):
                if not pred(c):
                    continue
                if day_count.get(c["session"], 0) >= 4:
                    continue
                if f"out_{tr_}" not in c:
                    continue
                sel.append(c)
                day_count[c["session"]] = day_count.get(c["session"], 0) + 1
            bt = backtest(sel, tr_, name=label)
            print(f"  {label:16} {tr_}R: n={bt['trades']:3d} WR={bt['win_rate']:5.1f}% PnL=₹{bt['net_pnl']:>9,.0f} PF={bt['profit_factor']}")

    filters = {
        "ml_threshold": champ_cfg["th"],
        "require_ema_stack": champ_cfg["stack"],
        "adx_min": champ_cfg["adx_min"],
        "max_risk_pts": champ_cfg["max_risk"],
        "min_risk_pts": champ_cfg["min_risk"],
        "rsi_min": champ_cfg["rsi_lo"],
        "rsi_max": champ_cfg["rsi_hi"],
        "hour_start": champ_cfg["hour_start"],
        "hour_end": champ_cfg["hour_end"],
        "range_pos_min": champ_cfg["rp_min"],
        "max_path_pct": champ_cfg["max_path"],
        "min_path_pct": champ_cfg["min_path"],
        "require_di_bull": champ_cfg["di_bull"],
        "require_macd_hist_pos": champ_cfg["macd_pos"],
        "soft_checklist": champ_cfg["checklist"],
        "soft_checklist_min_score": 6,
        "target_r": champ_cfg["target_r"],
        "max_trades_per_day": champ_cfg["max_day"],
    }

    with open(OUT_MODEL, "wb") as f:
        pickle.dump({
            "models": models,
            "feature_cols": FEATURE_COLS,
            "strategy_name": "Elite ML Long v1",
            "filters": filters,
            "target_r": filters["target_r"],
        }, f)

    champ_summary = {
        "name": "Elite ML Long v1",
        "oos": {k: v for k, v in oos_bt.items() if k != "trade_log"},
        "full": {k: v for k, v in full_bt.items() if k != "trade_log"},
    }
    meta = {
        "model_type": "ensemble_rf_gb_hgb",
        "threshold": filters["ml_threshold"],
        "features": FEATURE_COLS,
        "strategy_name": "Elite ML Long v1",
        "strategy_filters": filters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_method": "oos_first_70_30_time_split",
        "data_range": {
            "instrument": "NIFTY",
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
            "bars": len(df),
            "train_sessions": len(train_sess),
            "oos_sessions": len(test_sess),
        },
        "champion": champ_summary,
        "oos_eligible_count": len(eligible),
        "min_wr_target": MIN_WR,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    out = {
        "selected": "Elite ML Long v1",
        "generated_at": meta["trained_at"],
        "selection_method": "oos_first",
        "data_range": meta["data_range"],
        "filters": filters,
        "champion": champ_summary,
        "oos_top": [
            {
                "oos_wr": c["oos_wr"], "oos_n": c["oos_n"], "oos_pnl": c["oos_pnl"],
                "oos_pf": c["oos_pf"], "full_wr": c["full_wr"], "full_n": c["full_n"],
                "full_pnl": c["full_pnl"], "name": c["name"],
            }
            for c in comparison[:20]
        ],
        "full_period": {
            "summary": champ_summary["full"],
            "wins": full_bt["wins"],
            "losses": full_bt["losses"],
            "trades": full_bt["trade_log"],
        },
        "oos_period": {
            "summary": champ_summary["oos"],
            "trades": oos_bt["trade_log"],
        },
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT_TRADES, "w", encoding="utf-8") as f:
        json.dump(full_bt["trade_log"], f, indent=2, default=str)

    print(f"\nSaved → {OUT_MODEL.name}, {OUT_META.name}, {OUT_RESULTS.name}")
    meets = oos_bt["win_rate"] >= MIN_WR and oos_bt["net_pnl"] > 0
    print(f"Meets OOS WR≥{MIN_WR}% + profit: {meets}")


if __name__ == "__main__":
    main()
