"""
Fast Elite ML Long search — precompute probs once, grid filters in-memory.
Goal: profitable long F&O strategy with WR ≥ 70%.
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
MIN_TRADES = 15  # ~3.5 months of 5m data — 15 is still meaningful


def ema_stack_ok(row) -> bool:
    return bool(
        pd.notna(row.get("ema_9"))
        and pd.notna(row.get("ema_21"))
        and pd.notna(row.get("ema_50"))
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
            return {
                "win": False, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "stop_hit", "exit_ts": ts,
            }
        if hi >= target:
            exit_px = target - SLIPPAGE_PTS
            return {
                "win": True, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "target_hit", "exit_ts": ts,
            }
        if ts.time() >= FORCE_EXIT:
            exit_px = cl - SLIPPAGE_PTS
            return {
                "win": exit_px > entry, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "eod_exit", "exit_ts": ts,
            }
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
            "trade_no": trades,
            "strategy": name or "Elite ML Long v1",
            "side": "LONG",
            "session_date": str(sess),
            "signal_time": c["ts"].isoformat(),
            "entry_price": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(entry + stop_pts * target_r, 2),
            "risk_pts": round(stop_pts, 2),
            "target_r": target_r,
            "lots": lots,
            "ml_prob": round(c["prob"], 3),
            "exit_time": out["exit_ts"].isoformat(),
            "exit_price": round(out["exit"], 2),
            "exit_reason": out["reason"],
            "pnl_pts": round(out["pnl_pts"], 2),
            "pnl_inr": round(pnl, 2),
            "result": "WIN" if pnl > 0 else "LOSS",
            "equity_after": round(equity, 2),
        })
    wr = 100.0 * wins / trades if trades else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {
        "trades": trades,
        "wins": int(wins),
        "losses": trades - int(wins),
        "win_rate": round(wr, 2),
        "net_pnl": round(equity - CAPITAL, 2),
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd, 2),
        "final_equity": round(equity, 2),
        "avg_daily": round((equity - CAPITAL) / max(len(daily), 1), 2),
        "trade_log": log,
    }


def select_cands(pool, cfg, target_r: float):
    """Filter + sequential checklist/loss/day caps."""
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
        if c["pct_from_open"] > cfg["max_path"]:
            continue
        if c["pct_from_open"] < cfg["min_path"]:
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
        key = f"out_{target_r}"
        if key not in c:
            continue
        out.append(c)
        day_count[sess] = day_count.get(sess, 0) + 1
        if not c[key]["win"]:
            losses_today[sess] = losses_today.get(sess, 0) + 1
    return out


def main():
    print("=" * 80)
    print("FAST Elite ML Long — balanced grid")
    print("=" * 80)
    raw = pickle.load(open(DATA_PATH, "rb"))
    df = enrich_features(normalize_df(raw))
    print(f"bars={len(df)}  {df.index.min().date()} → {df.index.max().date()}")

    target_rs = [1.0, 1.2, 1.5, 1.8, 2.0]
    print("Building candidates…")
    cands = []
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        if not loose_long(row):
            continue
        if pd.isna(row.get("ema_21")):
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
            stop_pts = entry0 - stop
            if stop_pts <= 0:
                continue
            true_target = entry0 + stop_pts * tr
            out = simulate_long(df, entry_idx, stop, true_target)
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
            "win": base["win"], "pnl_pts": base["pnl_pts"],
        }
        for tr, o in outs.items():
            c[f"out_{tr}"] = o
        cands.append(c)

    print(f"candidates={len(cands)} rawWR={100 * np.mean([c['win'] for c in cands]):.1f}%")

    sessions = sorted({c["session"] for c in cands})
    cut = max(1, int(len(sessions) * 0.7))
    train_sess, test_sess = set(sessions[:cut]), set(sessions[cut:])
    train = [c for c in cands if c["session"] in train_sess]
    test = [c for c in cands if c["session"] in test_sess]
    print(f"train={len(train)} oos={len(test)} sessions {len(train_sess)}/{len(test_sess)}")

    X_tr = np.array([c["feat"] for c in train])
    y_tr = np.array([1 if c["win"] else 0 for c in train])
    print("Training ensemble…")
    models = train_models(X_tr, y_tr)
    probs = ensemble_proba(models, np.array([c["feat"] for c in cands]))
    for c, p in zip(cands, probs):
        c["prob"] = float(p)

    # Prob calibration stats
    for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        m = [c for c in cands if c["prob"] >= th]
        if m:
            print(f"  prob≥{th:.2f}: n={len(m)} WR={100*np.mean([c['win'] for c in m]):.1f}%")

    # Balanced grid (not only ultra-tight)
    grid = {
        "th": [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75],
        "stack": [False, True],
        "adx_min": [0, 12, 15, 18, 22],
        "max_risk": [18, 22, 28, 35],
        "min_risk": [1.5, 3.0],
        "rsi": [(40, 75), (45, 70), (48, 72), (50, 70), (50, 75)],
        "hour_end": [13, 14],
        "rp_min": [0.0, 0.25, 0.40],
        "max_path": [0.55, 0.75, 1.2],
        "min_path": [-0.5, -0.2],
        "di_bull": [False, True],
        "macd_pos": [False, True],
        "checklist": [False, True],
        "target_r": [1.0, 1.2, 1.5, 1.8],
        "max_day": [2, 3, 4],
    }

    # Cartesian is huge — build targeted layers
    packs = []

    # Layer A: high WR hunters (tighter)
    for th, stack, adx, max_r, rsi, he, cl, tr, md in itertools.product(
        [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75],
        [True, False],
        [0, 15, 18, 22],
        [18, 22, 28],
        [(45, 70), (48, 72), (50, 70), (50, 75)],
        [13, 14],
        [True, False],
        [1.0, 1.2, 1.5],
        [2, 3, 4],
    ):
        packs.append({
            "th": th, "stack": stack, "adx_min": adx, "max_risk": max_r, "min_risk": 1.5,
            "rsi_lo": rsi[0], "rsi_hi": rsi[1], "hour_start": 9, "hour_end": he,
            "rp_min": 0.0, "max_path": 0.75, "min_path": -0.5,
            "di_bull": False, "macd_pos": False, "checklist": cl, "target_r": tr, "max_day": md,
        })

    # Layer B: quality structure
    for th, adx, di, macd, path, tr, cl in itertools.product(
        [0.58, 0.62, 0.65, 0.70],
        [12, 18],
        [True, False],
        [True, False],
        [0.55, 0.70],
        [1.2, 1.5],
        [True, False],
    ):
        packs.append({
            "th": th, "stack": True, "adx_min": adx, "max_risk": 25, "min_risk": 2.0,
            "rsi_lo": 48, "rsi_hi": 72, "hour_start": 9, "hour_end": 13,
            "rp_min": 0.30, "max_path": path, "min_path": -0.35,
            "di_bull": di, "macd_pos": macd, "checklist": cl, "target_r": tr, "max_day": 3,
        })

    # Layer C: volume / looser for PnL
    for th, tr, max_r, he in itertools.product(
        [0.52, 0.55, 0.58, 0.60],
        [1.0, 1.5, 2.0],
        [22, 30, 40],
        [13, 14],
    ):
        packs.append({
            "th": th, "stack": False, "adx_min": 0, "max_risk": max_r, "min_risk": 1.5,
            "rsi_lo": 40, "rsi_hi": 75, "hour_start": 9, "hour_end": he,
            "rp_min": 0.0, "max_path": 1.2, "min_path": -0.6,
            "di_bull": False, "macd_pos": False, "checklist": False, "target_r": tr, "max_day": 4,
        })
        packs.append({
            "th": th, "stack": False, "adx_min": 0, "max_risk": max_r, "min_risk": 1.5,
            "rsi_lo": 40, "rsi_hi": 75, "hour_start": 9, "hour_end": he,
            "rp_min": 0.0, "max_path": 0.70, "min_path": -0.5,
            "di_bull": False, "macd_pos": False, "checklist": True, "target_r": tr, "max_day": 4,
        })

    # Dedup
    seen = set()
    uniq = []
    for p in packs:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    print(f"packs={len(uniq)}")

    results = []
    eligible = []
    for i, cfg in enumerate(uniq, 1):
        tr = cfg["target_r"]
        selected = select_cands(cands, cfg, tr)
        if len(selected) < 8:
            continue
        name = (
            f"MLLong th={cfg['th']} stack={cfg['stack']} ADX≥{cfg['adx_min']} "
            f"risk≤{cfg['max_risk']} RSI{cfg['rsi_lo']}-{cfg['rsi_hi']} "
            f"h≤{cfg['hour_end']} CL={cfg['checklist']} DI={cfg['di_bull']} {tr}R"
        )
        bt = backtest(selected, tr, name=name)
        if bt["trades"] < 8:
            continue
        row = {**bt, "name": name, "cfg": cfg, "period": "full"}
        results.append(row)
        if (
            bt["win_rate"] >= MIN_WR
            and bt["net_pnl"] > 0
            and bt["trades"] >= MIN_TRADES
            and bt["profit_factor"] >= 1.25
        ):
            eligible.append(row)
            print(
                f"★ WR={bt['win_rate']:5.1f}% n={bt['trades']:3d} PnL=₹{bt['net_pnl']:>9,.0f} "
                f"PF={bt['profit_factor']:5.2f} DD={bt['max_dd_pct']:4.1f}% | {name[:72]}"
            )
        if i % 800 == 0:
            top = max(results, key=lambda r: (r["win_rate"], r["net_pnl"])) if results else None
            print(
                f"  … {i}/{len(uniq)} res={len(results)} elig={len(eligible)} "
                f"topWR={top['win_rate'] if top else 0}% n={top['trades'] if top else 0}"
            )

    print(f"\nTotal results={len(results)} | eligible WR≥{MIN_WR}% n≥{MIN_TRADES}: {len(eligible)}")

    # Also rule-only baselines stored
    for mode_name, pred in [
        ("Rule EMA stack", lambda c: c["full_stack"]),
        ("Rule loose long", lambda c: True),
        ("Rule high ML≥0.65", lambda c: c["prob"] >= 0.65),
        ("Rule high ML≥0.70 stack", lambda c: c["prob"] >= 0.70 and c["full_stack"]),
    ]:
        for tr in [1.0, 1.5]:
            pool = [c for c in cands if pred(c) and f"out_{tr}" in c]
            # simple sequential day cap
            cfg = {
                "th": 0.0, "stack": False, "adx_min": 0, "max_risk": 99, "min_risk": 0,
                "rsi_lo": 0, "rsi_hi": 100, "hour_start": 9, "hour_end": 14,
                "rp_min": 0, "max_path": 99, "min_path": -99,
                "di_bull": False, "macd_pos": False, "checklist": False,
                "target_r": tr, "max_day": 4,
            }
            # manual filter
            sel = []
            day_count = {}
            for c in sorted(pool, key=lambda x: x["i"]):
                if not pred(c):
                    continue
                sess = c["session"]
                if day_count.get(sess, 0) >= 4:
                    continue
                sel.append(c)
                day_count[sess] = day_count.get(sess, 0) + 1
            bt = backtest(sel, tr, name=f"{mode_name} {tr}R")
            results.append({**bt, "name": f"{mode_name} {tr}R", "cfg": cfg, "period": "full"})
            print(
                f"  {mode_name} {tr}R: n={bt['trades']} WR={bt['win_rate']}% "
                f"PnL=₹{bt['net_pnl']:,.0f} PF={bt['profit_factor']}"
            )

    # OOS for top eligible / top WR
    print("\n### OOS validation ###")
    eligible.sort(key=lambda x: (x["net_pnl"], x["win_rate"]), reverse=True)
    rank_pool = eligible[:20] if eligible else sorted(
        [r for r in results if r["net_pnl"] > 0 and r["trades"] >= 12],
        key=lambda x: (x["win_rate"], x["net_pnl"]),
        reverse=True,
    )[:20]

    oos_ok = []
    for row in rank_pool:
        cfg = row["cfg"]
        tr = cfg["target_r"]
        # need th etc — for rule baselines skip oos detailed
        if "th" not in cfg:
            continue
        sel = select_cands(test, cfg, tr)
        bt = backtest(sel, tr, name="OOS " + row["name"][:40])
        print(
            f"  OOS WR={bt['win_rate']:5.1f}% n={bt['trades']:3d} PnL=₹{bt['net_pnl']:>8,.0f} "
            f"| full WR={row['win_rate']}% PnL=₹{row['net_pnl']:,.0f} | {row['name'][:50]}"
        )
        if bt["trades"] >= 4 and bt["net_pnl"] > 0:
            oos_ok.append({**row, "oos": bt})

    # Champion
    if eligible:
        if oos_ok:
            # among eligible with OOS profit, pick best full PnL with OOS WR≥55
            good = [x for x in oos_ok if x["oos"]["win_rate"] >= 55]
            champ = max(good or oos_ok, key=lambda x: (x["win_rate"], x["net_pnl"]))
        else:
            champ = eligible[0]
    else:
        prof = [r for r in results if r["net_pnl"] > 0 and r["trades"] >= MIN_TRADES]
        prof.sort(key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
        champ = prof[0] if prof else max(results, key=lambda x: x.get("win_rate", 0))

    print("\n" + "=" * 80)
    print("### CHAMPION ###")
    print(champ["name"])
    print(
        f"Full: WR={champ['win_rate']}% n={champ['trades']} PnL=₹{champ['net_pnl']:,.0f} "
        f"PF={champ['profit_factor']} DD={champ['max_dd_pct']}%"
    )
    if "oos" in champ:
        o = champ["oos"]
        print(f"OOS:  WR={o['win_rate']}% n={o['trades']} PnL=₹{o['net_pnl']:,.0f} PF={o['profit_factor']}")
    print("CFG:", champ.get("cfg"))

    print(f"\n### TOP 25 by WR (profitable n≥{MIN_TRADES}) ###")
    board = [r for r in results if r["net_pnl"] > 0 and r["trades"] >= MIN_TRADES]
    board.sort(key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
    for i, r in enumerate(board[:25], 1):
        mark = " ◀ CHAMP" if r["name"] == champ["name"] else ""
        print(
            f"{i:2d}. WR={r['win_rate']:5.1f}% n={r['trades']:3d} PnL=₹{r['net_pnl']:>9,.0f} "
            f"PF={r['profit_factor']:5.2f} DD={r['max_dd_pct']:4.1f}% | {r['name'][:62]}{mark}"
        )

    print("\n### TOP 15 by PnL (WR≥60% n≥15) ###")
    board2 = [r for r in results if r["win_rate"] >= 60 and r["net_pnl"] > 0 and r["trades"] >= 15]
    board2.sort(key=lambda x: x["net_pnl"], reverse=True)
    for i, r in enumerate(board2[:15], 1):
        print(
            f"{i:2d}. PnL=₹{r['net_pnl']:>9,.0f} WR={r['win_rate']:5.1f}% n={r['trades']:3d} "
            f"PF={r['profit_factor']:5.2f} | {r['name'][:58]}"
        )

    # Save
    cfg = champ.get("cfg") or {}
    filters = {
        "ml_threshold": cfg.get("th", 0.65),
        "require_ema_stack": cfg.get("stack", True),
        "adx_min": cfg.get("adx_min", 15),
        "max_risk_pts": cfg.get("max_risk", 22),
        "min_risk_pts": cfg.get("min_risk", 1.5),
        "rsi_min": cfg.get("rsi_lo", 45),
        "rsi_max": cfg.get("rsi_hi", 70),
        "hour_start": cfg.get("hour_start", 9),
        "hour_end": cfg.get("hour_end", 14),
        "range_pos_min": cfg.get("rp_min", 0.0),
        "max_path_pct": cfg.get("max_path", 0.75),
        "min_path_pct": cfg.get("min_path", -0.5),
        "require_di_bull": cfg.get("di_bull", False),
        "require_macd_hist_pos": cfg.get("macd_pos", False),
        "soft_checklist": cfg.get("checklist", True),
        "soft_checklist_min_score": 6,
        "target_r": cfg.get("target_r", 1.5),
        "max_trades_per_day": cfg.get("max_day", 4),
    }
    payload = {
        "models": models,
        "feature_cols": FEATURE_COLS,
        "strategy_name": "Elite ML Long v1",
        "filters": filters,
        "target_r": filters["target_r"],
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(payload, f)

    meta = {
        "model_type": "ensemble_rf_gb_hgb",
        "threshold": filters["ml_threshold"],
        "features": FEATURE_COLS,
        "strategy_name": "Elite ML Long v1",
        "strategy_filters": filters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_range": {
            "instrument": "NIFTY",
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
            "bars": len(df),
        },
        "champion": {k: v for k, v in champ.items() if k not in ("trade_log", "cfg", "oos")},
        "eligible_count": len(eligible),
        "min_wr_target": MIN_WR,
        "min_trades": MIN_TRADES,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    out = {
        "selected": "Elite ML Long v1",
        "generated_at": meta["trained_at"],
        "data_range": meta["data_range"],
        "filters": filters,
        "champion": meta["champion"],
        "eligible_top": [
            {k: v for k, v in r.items() if k not in ("trade_log", "cfg", "oos")}
            for r in eligible[:30]
        ],
        "top_by_wr": [
            {k: v for k, v in r.items() if k not in ("trade_log", "cfg", "oos")}
            for r in board[:40]
        ],
        "top_by_pnl_wr60": [
            {k: v for k, v in r.items() if k not in ("trade_log", "cfg", "oos")}
            for r in board2[:20]
        ],
        "full_period": {
            "summary": meta["champion"],
            "wins": champ["wins"],
            "losses": champ["losses"],
            "trades": champ.get("trade_log", []),
        },
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT_TRADES, "w", encoding="utf-8") as f:
        json.dump(champ.get("trade_log", []), f, indent=2, default=str)

    print(f"\nSaved model/results under data/intraday_fno_ml_long_*")
    print(f"Eligible packs: {len(eligible)}")


if __name__ == "__main__":
    main()
