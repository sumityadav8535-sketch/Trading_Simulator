"""
Historical / holdout preview for LIVE Elite ML Short v2.

A) ~1 year ago, random 2 months — Yahoo has no 5m that far back; try 1h with
   exact live filters (often 0 trades: 22pt stop cap is 5m-calibrated).
B) Same 1h window with risk cap scaled for hourly ranges (NOT live — informational).
C) Random 2-month slice of available 5m NIFTY pickle (true live setup).

Does not modify live F&O result files.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.intraday_fno_ml_enhance import (  # noqa: E402
    FEATURE_COLS,
    TARGET_R,
    backtest_signals,
    build_dataset,
    enrich_features,
    model_proba,
    train_models,
)
from scripts.intraday_fno_optimize import FilterConfig, build_filtered_signals  # noqa: E402
from scripts.intraday_fno_search import normalize_df  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
OUT = ROOT / "data" / "_historical_window_backtest_preview.json"
PKL = ROOT / "data" / "intraday_fno" / "NIFTY.pkl"

LIVE = FilterConfig(
    name="Elite ML Short v2 LIVE",
    ml_th=0.58,
    max_risk_pts=22,
    adx_min=0.0,
    range_pos_min=0.0,
    require_ema_stack=False,
)


def pick_window_1y(months: int = 2, seed: int | None = None) -> tuple[date, date]:
    rng = random.Random(seed)
    today = datetime.now(IST).date()
    end_lo = today - timedelta(days=int(365 * 1.25))
    end_hi = today - timedelta(days=int(365 * 0.90))
    floor = today - timedelta(days=680)
    end_lo = max(end_lo, floor + timedelta(days=100))
    if end_hi <= end_lo:
        end_hi = end_lo + timedelta(days=40)
    end = end_lo + timedelta(days=rng.randint(0, (end_hi - end_lo).days))
    start = end - timedelta(days=int(30.4 * months))
    return start, end


def fetch_1h(start: date, end: date) -> pd.DataFrame:
    raw = yf.download(
        "^NSEI",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1h",
        progress=False,
        auto_adjust=False,
    )
    return normalize_df(raw)


def median_risk(dset: pd.DataFrame) -> float:
    if dset.empty:
        return 22.0
    r = ((dset["stop"] - dset["target"]) / (1 + TARGET_R)).astype(float)
    return float(r.median())


def train_and_score(df: pd.DataFrame, train_before: date | None = None):
    df_e = enrich_features(df)
    dset = build_dataset(df_e, use_loose=True).dropna(subset=FEATURE_COLS)
    if dset.empty:
        return df_e, dset, None, "no candidates"

    if train_before is not None:
        pre = dset[dset["session_date"] < train_before]
    else:
        dates = sorted(dset["session_date"].unique())
        cut = max(int(len(dates) * 0.7), 1)
        pre = dset[dset["session_date"].isin(set(dates[:cut]))]

    if len(pre) < 12 or pre["label"].nunique() < 2:
        pre = dset.iloc[: max(int(len(dset) * 0.7), 12)]
    if len(pre) < 10 or pre["label"].nunique() < 2:
        return df_e, dset, None, "insufficient train labels"

    rf = train_models(pre[FEATURE_COLS].values, pre["label"].values)["random_forest"]
    dset = dset.copy()
    dset["prob"] = model_proba(rf, dset[FEATURE_COLS].values)
    return df_e, dset, rf, None


def bt(df, dset, cfg: FilterConfig, tag: str):
    if dset is None or dset.empty:
        return {"trades": 0, "note": "no dataset"}
    sigs = build_filtered_signals(df, dset, dset["prob"].values, cfg)
    if sigs is None or sigs.empty:
        # diagnostics
        probs = dset["prob"]
        risks = ((dset["stop"] - dset["target"]) / (1 + TARGET_R)).astype(float)
        return {
            "trades": 0,
            "note": "no signals passed filters",
            "diag": {
                "candidates": int(len(dset)),
                "pct_ml_ge_th": float((probs >= cfg.ml_th).mean()),
                "pct_risk_le_cap": float((risks <= cfg.max_risk_pts).mean()),
                "median_risk": float(risks.median()),
                "mean_prob": float(probs.mean()),
                "max_prob": float(probs.max()),
            },
        }
    stats, trades = backtest_signals(df, sigs, f"{cfg.name} {tag}", tag, record_trades=True)
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    return {
        "trades": stats.trades,
        "win_rate": stats.win_rate,
        "wins": stats.trades - losses,
        "losses": losses,
        "avg_daily": stats.avg_daily,
        "net_pnl": stats.net_pnl,
        "profit_factor": stats.profit_factor,
        "max_dd_pct": stats.max_dd_pct,
        "trade_log": trades,
    }


def print_block(title: str, res: dict):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)
    if res.get("trades", 0) == 0:
        print("  Trades: 0")
        if res.get("note"):
            print(f"  Note: {res['note']}")
        if res.get("diag"):
            d = res["diag"]
            print(
                f"  Diag: candidates={d['candidates']} | "
                f"% ML≥th={d['pct_ml_ge_th']*100:.1f}% | "
                f"% risk≤cap={d['pct_risk_le_cap']*100:.1f}% | "
                f"median risk={d['median_risk']:.1f} | "
                f"prob mean/max={d['mean_prob']:.3f}/{d['max_prob']:.3f}"
            )
        return
    print(f"  Trades:   {res['trades']}")
    print(f"  Win rate: {res['win_rate']}%")
    print(f"  W / L:    {res['wins']} / {res['losses']}")
    print(f"  ₹/day:    {res['avg_daily']:,.0f}")
    print(f"  Net P&L:  ₹{res['net_pnl']:,.0f}")
    print(f"  PF:       {res['profit_factor']}")
    print(f"  Max DD:   {res['max_dd_pct']}%")
    if res.get("trade_log"):
        print("  Trades detail:")
        for t in res["trade_log"]:
            print(
                f"    #{t['trade_no']} {t['session_date']} {str(t['signal_time'])[11:16]} "
                f"{t['result']:4s} ml={t.get('ml_prob')} risk={t.get('risk_pts')} "
                f"{t.get('exit_reason')} ₹{t.get('pnl_inr'):,.0f}"
            )


def main():
    seed = random.randint(1, 99_999)
    win_start, win_end = pick_window_1y(months=2, seed=seed)
    ctx_start = win_start - timedelta(days=120)

    print("=" * 72)
    print("HISTORICAL CHECK — LIVE Elite ML Short v2 (preview, no live writes)")
    print("=" * 72)
    print(f"Seed: {seed}")
    print(f"Random ~1y-ago window: {win_start} → {win_end}")
    print("Live filters: ml≥0.58 | max_risk≤22 | no extra filters")
    print()

    # --- A/B: 1h historical ---
    print("Fetching 1h Nifty for train context + window...")
    df_1h = fetch_1h(ctx_start, win_end)
    print(f"  1h bars: {len(df_1h)} | {df_1h.index.min() if len(df_1h) else '—'} → {df_1h.index.max() if len(df_1h) else '—'}")

    # confirm 5m empty
    raw5 = yf.download(
        "^NSEI",
        start=win_start.isoformat(),
        end=(win_end + timedelta(days=1)).isoformat(),
        interval="5m",
        progress=False,
        auto_adjust=False,
    )
    print(f"  5m bars for window: {0 if raw5 is None or raw5.empty else len(raw5)} (Yahoo limit ~60d)")

    results = {
        "preview_only": True,
        "seed": seed,
        "window": {"start": win_start.isoformat(), "end": win_end.isoformat()},
        "live_filters": {
            "ml_threshold": LIVE.ml_th,
            "max_risk_pts": LIVE.max_risk_pts,
        },
    }

    if df_1h.empty:
        print("ERROR: no 1h data")
        results["error"] = "no 1h data"
    else:
        df_e, dset, rf, err = train_and_score(df_1h, train_before=win_start)
        if err:
            print(f"  Train issue: {err}")
            results["1h_error"] = err
        else:
            med_r = median_risk(dset)
            print(f"  Candidates: {len(dset)} | median risk pts≈{med_r:.1f}")

            df_win = df_e[(df_e.index.date >= win_start) & (df_e.index.date <= win_end)]
            dset_win = dset[
                (dset["session_date"] >= win_start) & (dset["session_date"] <= win_end)
            ].copy()

            res_live = bt(df_win, dset_win, LIVE, "1h_window_LIVE")
            print_block(
                f"A) 1h window {win_start}→{win_end} — EXACT live filters (max_risk=22)",
                res_live,
            )
            results["A_1h_exact_live"] = {k: v for k, v in res_live.items() if k != "trade_log"}
            results["A_trades"] = res_live.get("trade_log", [])

            # Scaled risk: map 5m 22pt cap to 1h using median risk ratio vs typical 5m ~16
            scale = med_r / 16.0 if med_r > 0 else 3.0
            scaled_cap = round(22.0 * scale, 1)
            cfg_scaled = FilterConfig(
                name="Elite SCALED-for-1h (NOT live)",
                ml_th=0.58,
                max_risk_pts=scaled_cap,
            )
            # ML often stuck near 0.5 on thin 1h — if no trade, also try ml 0.50 for info
            res_scaled = bt(df_win, dset_win, cfg_scaled, "1h_window_scaled")
            if res_scaled.get("trades", 0) == 0:
                cfg_soft = FilterConfig(
                    name="Elite SCALED+softML (NOT live)",
                    ml_th=0.50,
                    max_risk_pts=scaled_cap,
                )
                res_scaled = bt(df_win, dset_win, cfg_soft, "1h_soft")
                res_scaled["note"] = (
                    res_scaled.get("note", "")
                    + f" | used soft ML 0.50 + scaled risk {scaled_cap} (NOT live)"
                ).strip(" |")
            else:
                res_scaled["note"] = f"scaled max_risk={scaled_cap} from median 1h risk (NOT live)"

            print_block(
                f"B) 1h window — risk scaled for hourly bars (NOT the live setup) "
                f"cap≈{scaled_cap}",
                res_scaled,
            )
            results["B_1h_scaled_not_live"] = {
                k: v for k, v in res_scaled.items() if k != "trade_log"
            }
            results["B_scaled_max_risk"] = scaled_cap
            results["B_trades"] = res_scaled.get("trade_log", [])

    # --- C: true 5m from local pickle, random 2 months ---
    print()
    print("C) Random 2-month slice of LOCAL 5m NIFTY pickle (true live setup)...")
    if not PKL.exists():
        print("  No NIFTY.pkl found.")
        results["C_error"] = "no pickle"
    else:
        hist = pd.read_pickle(PKL)
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("Asia/Kolkata")
        else:
            hist.index = hist.index.tz_convert("Asia/Kolkata")
        d0 = hist.index.min().date()
        d1 = hist.index.max().date()
        span = (d1 - d0).days
        print(f"  Pickle range: {d0} → {d1} ({span} days, {len(hist)} bars)")
        if span < 40:
            win_a, win_b = d0, d1
        else:
            rng = random.Random(seed + 7)
            # random 60-calendar-day window inside pickle
            latest_start = d1 - timedelta(days=60)
            if latest_start <= d0:
                win_a, win_b = d0, d1
            else:
                win_a = d0 + timedelta(days=rng.randint(0, (latest_start - d0).days))
                win_b = win_a + timedelta(days=60)
                if win_b > d1:
                    win_b = d1
                    win_a = win_b - timedelta(days=60)

        # Train on all pickle data before window start when possible
        df_e, dset, rf, err = train_and_score(hist, train_before=win_a)
        if err:
            df_e, dset, rf, err = train_and_score(hist, train_before=None)
        if err:
            print(f"  Failed: {err}")
            results["C_error"] = err
        else:
            df_w = df_e[(df_e.index.date >= win_a) & (df_e.index.date <= win_b)]
            ds_w = dset[(dset["session_date"] >= win_a) & (dset["session_date"] <= win_b)].copy()
            res5 = bt(df_w, ds_w, LIVE, "5m_local_window")
            print_block(
                f"C) 5m local window {win_a}→{win_b} — EXACT live filters",
                res5,
            )
            results["C_5m_local"] = {
                "window": {"start": win_a.isoformat(), "end": win_b.isoformat()},
                **{k: v for k, v in res5.items() if k != "trade_log"},
            }
            results["C_trades"] = res5.get("trade_log", [])

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Saved: {OUT}")
    print("Live F&O files NOT modified.")


if __name__ == "__main__":
    main()
