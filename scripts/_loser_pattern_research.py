"""Why last-1y Stage 2.0 Q>=60 losers failed, and which skip rules would have helped."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import django
import numpy as np
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
django.setup()

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import (
    EntryFilters,
    MA_COND_ABOVE,
    _preload_frames,
    fill_performance_metrics,
    run_stage_v2_backtest,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.stage_engine import detect_weekly_stage
from stage_analysis_v2.services.tech_filters import enrich_daily_tech, snapshot_tech
from trading.constants import NIFTY50_SYMBOL
from trading.models import Stock
from trading.services.market_data import get_universe_symbols, load_price_dataframe

OUT = ROOT / "data" / "loser_pattern_research.json"
START = date(2025, 8, 28)
END = date(2026, 8, 28)

# User-pasted shared-book losers (signal date).
USER_LOSERS = [
    ("PFC", "2026-07-31"),
    ("NYKAA", "2026-02-06"),
    ("AUBANK", "2026-03-20"),
    ("NHPC", "2025-09-19"),
    ("AXISBANK", "2026-01-30"),
    ("GODREJCP", "2025-09-05"),
    ("SBIN", "2026-08-07"),
    ("JSWSTEEL", "2026-01-30"),
    ("HINDUNILVR", "2025-08-29"),
    ("BSE", "2025-09-05"),
    ("AXISBANK", "2026-07-17"),
    ("APLAPOLLO", "2026-01-30"),
    ("RELIANCE", "2025-11-14"),
    ("ICICIGI", "2025-11-14"),
    ("BAJAJFINSV", "2025-09-19"),
    ("EXIDEIND", "2025-09-26"),
    ("DABUR", "2026-01-02"),
    ("ASHOKLEY", "2026-04-10"),
    ("TIINDIA", "2026-08-21"),
    ("SIEMENS", "2026-04-24"),
    ("CHOLAFIN", "2026-08-14"),
    ("NMDC", "2026-07-31"),
    ("TVSMOTOR", "2026-08-14"),
    ("NTPC", "2026-06-19"),
    ("GMRAIRPORT", "2026-02-06"),
    ("POWERGRID", "2026-07-03"),
    ("YESBANK", "2025-10-10"),
]

DEFENSIVE = {
    "HINDUNILVR", "GODREJCP", "DABUR", "MARICO", "NESTLEIND", "COLPAL",
    "BRITANNIA", "ITC", "TATACONSUM", "UNITDSPR",
}
PSU_YIELD = {
    "NHPC", "NTPC", "POWERGRID", "SJVN", "PFC", "RECLTD", "IRFC", "NMDC",
    "COALINDIA", "SAIL", "GAIL", "ONGC", "IOC", "BPCL", "HINDPETRO", "HUDCO",
}


def _ret(df: pd.DataFrame, ts: pd.Timestamp, days: int) -> float | None:
    hist = df.loc[:ts]
    if len(hist) < 3:
        return None
    last = float(hist["close"].iloc[-1])
    past = hist.loc[: ts - pd.Timedelta(days=days)]
    if past.empty:
        return None
    first = float(past["close"].iloc[-1])
    if first <= 0:
        return None
    return round((last / first - 1) * 100, 2)


def _nifty_stage(weekly: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    w = weekly.loc[weekly.index <= ts]
    if len(w) < 38:
        return None
    try:
        stage, _, _ = detect_weekly_stage(w)
        return int(stage)
    except ValueError:
        return None


def _summarize(bt) -> dict:
    fill_performance_metrics(bt)
    wins = [t for t in bt.trades if t.pnl > 0]
    losses = [t for t in bt.trades if t.pnl <= 0]
    return {
        "signals": bt.stage2_entries,
        "trades": bt.total_trades,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": bt.win_rate,
        "total_return_pct": bt.total_return_pct,
        "max_dd_pct": bt.max_drawdown_pct,
        "profit_factor": bt.profit_factor,
        "avg_rr": bt.avg_rr,
        "skipped_cash": bt.signals_skipped_cash,
        "final_capital": round(float(bt.final_cash), 0),
        "capital_compare": bt.capital_compare,
    }


def snapshot_trade(t, daily: pd.DataFrame, nifty: pd.DataFrame, nifty_w: pd.DataFrame, sector: str) -> dict:
    sig_ts = pd.Timestamp(t.signal_date)
    entry_ts = pd.Timestamp(t.entry_date)
    exit_ts = pd.Timestamp(t.exit_date)
    d = daily.copy()
    d["sma50"] = d["close"].rolling(50).mean()
    d["sma150"] = d["close"].rolling(150).mean()
    d["sma200"] = d["close"].rolling(200).mean()
    d["high_252"] = d["high"].rolling(252, min_periods=60).max()
    tech = enrich_daily_tech(d, include_supertrend=True)
    snap = snapshot_tech(tech, sig_ts)
    hist = d.loc[:sig_ts]
    close = float(hist["close"].iloc[-1]) if not hist.empty else float(t.entry_price)
    sma150 = float(hist["sma150"].iloc[-1]) if not hist.empty and pd.notna(hist["sma150"].iloc[-1]) else None
    sma200 = float(hist["sma200"].iloc[-1]) if not hist.empty and pd.notna(hist["sma200"].iloc[-1]) else None
    sma50 = float(hist["sma50"].iloc[-1]) if not hist.empty and pd.notna(hist["sma50"].iloc[-1]) else None
    hi252 = float(hist["high_252"].iloc[-1]) if not hist.empty and pd.notna(hist["high_252"].iloc[-1]) else None
    vs_sma150 = round((close / sma150 - 1) * 100, 2) if sma150 else None
    vs_high = round((close / hi252 - 1) * 100, 2) if hi252 else None
    stop = float(t.stop_loss or 0)
    stop_pct = round((float(t.entry_price) - stop) / float(t.entry_price) * 100, 2) if t.entry_price and stop else None
    atr_pct = round(snap.atr / close * 100, 2) if snap.ok and close else None
    stop_vs_atr = round(stop_pct / atr_pct, 2) if stop_pct and atr_pct else None
    nifty_to_exit = None
    if not nifty.empty:
        n_entry = nifty.loc[:entry_ts]
        n_exit = nifty.loc[:exit_ts]
        if not n_entry.empty and not n_exit.empty:
            a = float(n_entry["close"].iloc[-1])
            b = float(n_exit["close"].iloc[-1])
            if a:
                nifty_to_exit = round((b / a - 1) * 100, 2)
    return {
        "symbol": t.symbol,
        "signal_date": t.signal_date,
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "pnl": round(float(t.pnl), 0),
        "pnl_pct": t.pnl_pct,
        "rr": t.rr_achieved,
        "days": t.days_held,
        "exit_reason": t.exit_reason,
        "quality": int(t.quality_score or 0),
        "rs": round(float(t.rs_rating or 0), 1),
        "win": t.pnl > 0,
        "crumb": abs(float(t.pnl)) < 100,
        "sector": sector or "",
        "defensive": t.symbol in DEFENSIVE,
        "psu_yield": t.symbol in PSU_YIELD,
        "rsi": round(snap.rsi, 1) if snap.ok else None,
        "ext_ema20_pct": snap.price_vs_ema20_pct if snap.ok else None,
        "not_extended": snap.not_extended if snap.ok else None,
        "ema_stack": snap.ema_stack_bull if snap.ok else None,
        "st_bull": snap.supertrend_bull if snap.ok else None,
        "vol_ratio": snap.vol_ratio if snap.ok else None,
        "atr_pct": atr_pct,
        "above_sma50": bool(sma50 and close > sma50),
        "above_sma150": bool(sma150 and close > sma150),
        "above_sma200": bool(sma200 and close > sma200),
        "vs_sma150_pct": vs_sma150,
        "vs_52w_high_pct": vs_high,
        "ret_20d": _ret(d, sig_ts, 20),
        "ret_60d": _ret(d, sig_ts, 60),
        "nifty_ret_20d": _ret(nifty, sig_ts, 20),
        "nifty_ret_60d": _ret(nifty, sig_ts, 60),
        "nifty_stage": _nifty_stage(nifty_w, sig_ts),
        "nifty_while_held": nifty_to_exit,
        "stop_pct": stop_pct,
        "stop_vs_atr": stop_vs_atr,
        "user_loser": (t.symbol, t.signal_date) in {(s, d) for s, d in USER_LOSERS},
    }


def _median(vals):
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return round(float(np.median(clean)), 2)


def _mean(vals):
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return round(float(np.mean(clean)), 2)


def _pct_true(rows, key):
    vals = [bool(r.get(key)) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100, 1)


def cohort_stats(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "pnl": round(sum(r["pnl"] for r in rows), 0),
        "median_pnl": _median([r["pnl"] for r in rows]),
        "median_rr": _median([r["rr"] for r in rows]),
        "median_rs": _median([r["rs"] for r in rows]),
        "median_quality": _median([r["quality"] for r in rows]),
        "median_rsi": _median([r["rsi"] for r in rows]),
        "median_ext_ema20": _median([r["ext_ema20_pct"] for r in rows]),
        "median_vs_sma150": _median([r["vs_sma150_pct"] for r in rows]),
        "median_vs_52w_high": _median([r["vs_52w_high_pct"] for r in rows]),
        "median_stop_pct": _median([r["stop_pct"] for r in rows]),
        "median_stop_vs_atr": _median([r["stop_vs_atr"] for r in rows]),
        "median_nifty_20d": _median([r["nifty_ret_20d"] for r in rows]),
        "median_nifty_held": _median([r["nifty_while_held"] for r in rows]),
        "pct_rs_lt_70": round(sum(1 for r in rows if r["rs"] < 70) / len(rows) * 100, 1) if rows else None,
        "pct_below_sma150": round(sum(1 for r in rows if r["above_sma150"] is False) / len(rows) * 100, 1) if rows else None,
        "pct_extended": round(sum(1 for r in rows if r.get("not_extended") is False) / len(rows) * 100, 1) if rows else None,
        "pct_defensive": round(sum(1 for r in rows if r["defensive"]) / len(rows) * 100, 1) if rows else None,
        "pct_psu": round(sum(1 for r in rows if r["psu_yield"]) / len(rows) * 100, 1) if rows else None,
        "pct_nifty_stage_34": round(
            sum(1 for r in rows if r.get("nifty_stage") in (3, 4)) / len(rows) * 100, 1
        ) if rows else None,
        "pct_nifty_20d_down": round(
            sum(1 for r in rows if (r.get("nifty_ret_20d") or 0) < 0) / len(rows) * 100, 1
        ) if rows else None,
        "exit_reasons": dict(Counter(r["exit_reason"] for r in rows)),
    }


def eval_skip(rows: list[dict], name: str, pred) -> dict:
    skipped = [r for r in rows if pred(r)]
    kept = [r for r in rows if not pred(r)]
    skip_w = [r for r in skipped if r["win"]]
    skip_l = [r for r in skipped if not r["win"]]
    kept_w = [r for r in kept if r["win"]]
    n = len(kept)
    pnl_kept = sum(r["pnl"] for r in kept)
    pnl_skip_l = sum(r["pnl"] for r in skip_l)
    pnl_skip_w = sum(r["pnl"] for r in skip_w)
    return {
        "rule": name,
        "skipped": len(skipped),
        "skipped_losers": len(skip_l),
        "skipped_winners": len(skip_w),
        "loser_pnl_avoided": round(pnl_skip_l, 0),
        "winner_pnl_given_up": round(pnl_skip_w, 0),
        "net_if_skipped": round(-(pnl_skip_l + pnl_skip_w), 0),
        "kept_trades": n,
        "kept_win_rate": round(len(kept_w) / n * 100, 1) if n else None,
        "kept_pnl": round(pnl_kept, 0),
        "examples_skipped_losers": [
            f"{r['symbol']} {r['signal_date']} {r['pnl']:+.0f} rs={r['rs']} q={r['quality']}"
            for r in sorted(skip_l, key=lambda x: x["pnl"])[:8]
        ],
    }


def main():
    symbols = get_universe_symbols("nifty200")
    print(f"Running Stage 2.0 Q>=60 {START}→{END} n={len(symbols)} take-all primary…")
    bt = run_stage_v2_backtest(
        symbols=symbols,
        start_date=START,
        end_date=END,
        capital=1_000_000,
        min_quality_score=60,
        tech_filter="daily_mtf",
        shared_capital=False,
    )
    fill_performance_metrics(bt)
    baseline = _summarize(bt)
    print(
        f"Take-all: {bt.total_trades} trades WR {bt.win_rate}% ret {bt.total_return_pct}%  "
        f"shared book: {bt.capital_compare.get('shared', {})}"
    )

    print("Snapshotting entry features…")
    frames = _preload_frames(list({t.symbol for t in bt.trades}))
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_w = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    sectors = {s.symbol: s.sector for s in Stock.objects.filter(symbol__in=frames.keys())}

    rows = []
    for t in bt.trades:
        daily = frames.get(t.symbol)
        if daily is None:
            daily = load_price_dataframe(t.symbol)
        rows.append(snapshot_trade(t, daily, nifty, nifty_w, sectors.get(t.symbol, "")))

    wins = [r for r in rows if r["win"]]
    losses = [r for r in rows if not r["win"]]
    user_rows = [r for r in rows if r["user_loser"]]
    real_losses = [r for r in losses if not r["crumb"]]

    print(f"Matched user losers in this book: {len(user_rows)} / {len(USER_LOSERS)}")

    # Cluster stage-exits by week
    stage_exits = [r for r in losses if r["exit_reason"] == "stage_exit"]
    by_exit_week = defaultdict(list)
    for r in stage_exits:
        by_exit_week[str(pd.Timestamp(r["exit_date"]).to_period("W-FRI"))].append(r["symbol"])
    clustered = {
        week: names
        for week, names in sorted(by_exit_week.items(), key=lambda kv: -len(kv[1]))
        if len(names) >= 2
    }

    rules = [
        ("RS < 70", lambda r: r["rs"] < 70),
        ("Below SMA150", lambda r: r["above_sma150"] is False),
        ("RS<70 OR below SMA150", lambda r: r["rs"] < 70 or r["above_sma150"] is False),
        ("Nifty weekly Stage 3/4", lambda r: r.get("nifty_stage") in (3, 4)),
        ("Nifty 20d < 0", lambda r: (r.get("nifty_ret_20d") or 0) < 0),
        ("Nifty 20d < -3%", lambda r: (r.get("nifty_ret_20d") or 0) < -3),
        ("Extended >8% EMA20", lambda r: r.get("not_extended") is False),
        ("Within 5% of 52w high", lambda r: r.get("vs_52w_high_pct") is not None and r["vs_52w_high_pct"] > -5),
        ("RSI >= 70", lambda r: (r.get("rsi") or 0) >= 70),
        ("Quality < 70", lambda r: r["quality"] < 70),
        ("Defensive FMCG", lambda r: r["defensive"]),
        ("PSU / yield names", lambda r: r["psu_yield"]),
        ("Stop < 1.5 ATR (gap risk)", lambda r: r.get("stop_vs_atr") is not None and r["stop_vs_atr"] < 1.5),
        ("No EMA stack", lambda r: r.get("ema_stack") is False),
        ("Supertrend not bull", lambda r: r.get("st_bull") is False),
        ("RS<70 + Nifty 20d<0", lambda r: r["rs"] < 70 and (r.get("nifty_ret_20d") or 0) < 0),
        (
            "Pack: RS<70 or <SMA150 or Nifty S3/4",
            lambda r: r["rs"] < 70 or r["above_sma150"] is False or r.get("nifty_stage") in (3, 4),
        ),
        (
            "User-style: RS<70 or <SMA150 or defensive/PSU",
            lambda r: r["rs"] < 70 or r["above_sma150"] is False or r["defensive"] or r["psu_yield"],
        ),
    ]
    skip_table = [eval_skip(rows, name, pred) for name, pred in rules]
    skip_table.sort(key=lambda x: x["net_if_skipped"], reverse=True)

    print("\n=== Skip-rule ranking on take-all Q>=60 (net = losers avoided minus winners given up) ===")
    for s in skip_table[:10]:
        print(
            f"{s['rule']:<42} skip L{s['skipped_losers']}/W{s['skipped_winners']}  "
            f"net {s['net_if_skipped']:+.0f}  kept WR {s['kept_win_rate']}% pnl {s['kept_pnl']:+.0f}"
        )

    print("\nRe-running top structural filters as full engines…")
    variants = {}
    variants["Q60 RS70 + SMA150"] = _summarize(run_stage_v2_backtest(
        symbols=symbols, start_date=START, end_date=END, capital=1_000_000,
        min_quality_score=60, min_rs_rating=70, tech_filter="daily_mtf",
        entry_filters=EntryFilters(ma_condition=MA_COND_ABOVE, ma_period=150, ma_type="sma"),
        shared_capital=False,
    ))
    variants["Q60 + Nifty Stage 1/2 only"] = _summarize(run_stage_v2_backtest(
        symbols=symbols, start_date=START, end_date=END, capital=1_000_000,
        min_quality_score=60, tech_filter="daily_mtf", market_filter=True,
        shared_capital=False,
    ))
    variants["Q60 + not extended"] = _summarize(run_stage_v2_backtest(
        symbols=symbols, start_date=START, end_date=END, capital=1_000_000,
        min_quality_score=60, tech_filter="daily_mtf_not_ext",
        shared_capital=False,
    ))
    variants["Q70"] = _summarize(run_stage_v2_backtest(
        symbols=symbols, start_date=START, end_date=END, capital=1_000_000,
        min_quality_score=70, tech_filter="daily_mtf",
        shared_capital=False,
    ))
    for k, v in variants.items():
        print(
            f"{k:<32} trades {v['trades']} WR {v['win_rate']}% ret {v['total_return_pct']}% "
            f"DD {v['max_dd_pct']}% PF {v['profit_factor']}"
        )

    payload = {
        "window": {"start": str(START), "end": str(END), "min_quality": 60},
        "baseline_take_all": baseline,
        "cohorts": {
            "winners": cohort_stats(wins),
            "losers": cohort_stats(losses),
            "real_losers_not_crumb": cohort_stats(real_losses),
            "user_losers": cohort_stats(user_rows),
        },
        "user_loser_rows": user_rows,
        "all_losers": sorted(losses, key=lambda r: r["pnl"]),
        "stage_exit_clusters": clustered,
        "skip_rules": skip_table,
        "engine_variants": variants,
        "unmatched_user_losers": [
            f"{s} {d}" for s, d in USER_LOSERS
            if not any(r["symbol"] == s and r["signal_date"] == d for r in rows)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
