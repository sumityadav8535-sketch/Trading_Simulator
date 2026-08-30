"""Second-pass gap filters + data-quality check on chronic gappers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_n200_search import add_indicators, nifty200_symbols  # noqa: E402
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts.intraday_gap_hunt import collect_events, make_signals, months_from_trades, prepare, simulate_open  # noqa: E402
from scripts._gap_filter_search import (  # noqa: E402
    chronic_gappers,
    enrich_events,
    index_gaps,
    outcome_scan,
)
from trading.services.market_data import load_price_dataframe  # noqa: E402

SL_ATR = 0.6
STORY = dict(risk_pct=5.0, max_pos=4, top_k=2, max_deploy=0.35)
WIN = dict(risk_pct=10.0, max_pos=5, top_k=3, max_deploy=0.40)


def keys(df):
    return df[["symbol", "ts"]].drop_duplicates()


def run(stocks, events, name, mode, gmin, size, filt=None, score_col=None):
    sigs = make_signals(events, mode, gmin, 0.15, SL_ATR, "fill")
    if filt is not None and not sigs.empty:
        sigs = sigs.merge(keys(filt), on=["symbol", "ts"], how="inner")
    if score_col and filt is not None and not sigs.empty:
        extra = filt[["symbol", "ts", score_col]].drop_duplicates(["symbol", "ts"])
        sigs = sigs.merge(extra, on=["symbol", "ts"], how="left")
        sigs["score"] = pd.to_numeric(sigs[score_col], errors="coerce").fillna(sigs["score"])
    res, tdf = simulate_open(stocks, sigs, name, **size)
    months = months_from_trades(tdf)
    ms = " | ".join(f"{m['month'][-2:]} ₹{m['pnl']:.0f} {m['wr']:.0f}%" for m in months) if months else ""
    print(
        f"{res.total_return_pct:7.1f}% WR {res.win_rate:5.1f} PF {res.profit_factor:5.2f} "
        f"n={res.trades:3d} DD {res.max_dd_pct:5.1f} OOS {res.oos_pnl:8.0f}  {name}"
    )
    if ms:
        print(f"         {ms}")
    return res


def dq_check(sym: str, stocks: dict):
    df5 = stocks[sym]
    daily = load_price_dataframe(sym)
    first = df5.groupby("session").head(1)
    print(f"\n=== {sym} 5m-open vs official daily ===")
    n = 0
    big = 0
    rows = []
    for ts, row in first.iterrows():
        sess = row["session"]
        o5 = float(row["open"])
        # official daily same-day open and prior close
        d = daily.copy()
        d.index = pd.to_datetime(d.index).date
        if sess not in d.index:
            continue
        o_off = float(d.loc[sess, "open"])
        # prior daily close
        loc = list(d.index).index(sess)
        if loc == 0:
            continue
        pdc = float(d.iloc[loc - 1]["close"])
        gap5 = o5 / pdc - 1
        gap_off = o_off / pdc - 1
        n += 1
        if abs(gap5) >= 0.03:
            big += 1
        rows.append((sess, o5, o_off, pdc, gap5 * 100, gap_off * 100, o5 / o_off - 1))
    print(f"  days={n}  5m-gap≥3% {big} ({100*big/max(n,1):.0f}%)")
    diffs = [r[6] * 100 for r in rows]
    print(f"  5m open vs official open: mean {np.mean(diffs):+.2f}%  median {np.median(diffs):+.2f}%  "
          f"|diff|>1% {(np.abs(diffs)>1).mean()*100:.0f}%  |diff|>3% {(np.abs(diffs)>3).mean()*100:.0f}%")
    print("  sample days with |5m gap|>=3%:")
    shown = 0
    for sess, o5, o_off, pdc, g5, go, dlt in rows:
        if abs(g5) < 3:
            continue
        print(f"    {sess}  pdc {pdc:.2f}  5m_open {o5:.2f} ({g5:+.1f}%)  official_open {o_off:.2f} ({go:+.1f}%)  5m/off {dlt*100:+.2f}%")
        shown += 1
        if shown >= 8:
            break


def main():
    symbols = nifty200_symbols()
    cache = load_frames(symbols)
    stocks = prepare(cache)
    idx = cache.get("_INDEX")
    if idx is not None and not idx.empty:
        idx = add_indicators(idx)
    events = collect_events(stocks)
    ev = enrich_events(events, stocks, index_gaps(idx) if idx is not None else {})
    ev = outcome_scan(ev, stocks)
    chronic = chronic_gappers(events)
    ev["chronic"] = ev["symbol"].isin(chronic)
    print("chronic", sorted(chronic))

    for sym in sorted(chronic):
        if sym in stocks:
            dq_check(sym, stocks)

    down = ev[(ev["gap"] <= -0.02) & (ev["gap"] >= -0.15)].copy()
    both3 = ev[(ev["gap"].abs() >= 0.03) & (ev["gap"].abs() <= 0.15)].copy()
    d1 = ev[(ev["gap"] <= -0.01) & (ev["gap"] >= -0.15)].copy()

    rsi14 = pd.to_numeric(down["d_rsi14"], errors="coerce")
    rsi2 = pd.to_numeric(down["d_rsi2"], errors="coerce")
    di_p = pd.to_numeric(down["d_di_plus"], errors="coerce")
    di_m = pd.to_numeric(down["d_di_minus"], errors="coerce")
    idxg = pd.to_numeric(down["idx_gap"], errors="coerce")
    gatr = pd.to_numeric(down["gap_vs_atr"], errors="coerce")
    rsi5 = pd.to_numeric(down["rsi5_prev"], errors="coerce")

    print("\n===== targeted sized tests =====")
    run(stocks, events, "BASE down≥2%", "down_bounce", 0.02, STORY)
    run(stocks, events, "RSI14 50-65 (healthy)", "down_bounce", 0.02, STORY,
        down[rsi14.between(50, 65)])
    run(stocks, events, "RSI14 45-65", "down_bounce", 0.02, STORY,
        down[rsi14.between(45, 65)])
    run(stocks, events, "+DI > -DI", "down_bounce", 0.02, STORY, down[di_p > di_m])
    run(stocks, events, "-DI > +DI (knife)", "down_bounce", 0.02, STORY, down[di_m > di_p])
    run(stocks, events, "htf_up + +DI>-DI", "down_bounce", 0.02, STORY,
        down[down["d_htf_up"].fillna(False).astype(bool) & (di_p > di_m)])
    run(stocks, events, "idx -1.0 to -0.3%", "down_bounce", 0.02, STORY,
        down[idxg.between(-0.01, -0.003)])
    run(stocks, events, "RSI2 5-20", "down_bounce", 0.02, STORY, down[rsi2.between(5, 20)])
    run(stocks, events, "RSI5 40-50", "down_bounce", 0.02, STORY, down[rsi5.between(40, 50)])
    run(stocks, events, "+DI + prev red", "down_bounce", 0.02, STORY,
        down[(di_p > di_m) & (pd.to_numeric(down["d_day_ret"], errors="coerce") < 0)])
    run(stocks, events, "+DI + RSI14 45-65", "down_bounce", 0.02, STORY,
        down[(di_p > di_m) & rsi14.between(45, 65)])
    run(stocks, events, "above SMA20 + RSI14 45-65", "down_bounce", 0.02, STORY,
        down[down["d_above_sma20"].fillna(False).astype(bool) & rsi14.between(45, 65)])
    run(stocks, events, "gap/ATR 1.2-2.5 + +DI", "down_bounce", 0.02, STORY,
        down[gatr.between(1.2, 2.5) & (di_p > di_m)])
    # score most oversold RSI2 first
    down = down.copy()
    down["score_rsi2"] = 100 - pd.to_numeric(down["d_rsi2"], errors="coerce").fillna(50)
    run(stocks, events, "rank by most-oversold RSI2", "down_bounce", 0.02, STORY, down, score_col="score_rsi2")
    down["score_di"] = di_p - di_m
    run(stocks, events, "rank by +DI-(-DI)", "down_bounce", 0.02, STORY, down, score_col="score_di")

    print("\n-- both_fill skip chronic combos --")
    b = both3[~both3["chronic"]].copy()
    run(stocks, events, "BASE both≥3%", "both_fill", 0.03, WIN)
    run(stocks, events, "skip chronic", "both_fill", 0.03, WIN, b)
    brsi = pd.to_numeric(b["d_rsi14"], errors="coerce")
    brsi2 = pd.to_numeric(b["d_rsi2"], errors="coerce")
    bdi_p = pd.to_numeric(b["d_di_plus"], errors="coerce")
    bdi_m = pd.to_numeric(b["d_di_minus"], errors="coerce")
    long_ok = (b["gap"] < 0) & (bdi_p > bdi_m)
    short_ok = (b["gap"] > 0) & (brsi > 55)
    run(stocks, events, "chron + L:+DI  S:RSI>55", "both_fill", 0.03, WIN, b[long_ok | short_ok])
    long_ok = (b["gap"] < 0) & brsi.between(45, 65)
    short_ok = (b["gap"] > 0) & (brsi >= 55)
    run(stocks, events, "chron + L:RSI45-65 S:RSI≥55", "both_fill", 0.03, WIN, b[long_ok | short_ok])
    long_ok = (b["gap"] < 0)  # all longs
    short_ok = (b["gap"] > 0) & (~b["d_htf_up"].fillna(False).astype(bool))
    run(stocks, events, "chron + shorts only if NOT htf_up", "both_fill", 0.03, WIN, b[long_ok | short_ok])
    long_ok = (b["gap"] < 0) & (brsi2 < 20)
    short_ok = (b["gap"] > 0)  # keep all shorts except chronic already dropped
    run(stocks, events, "chron + longs only RSI2<20", "both_fill", 0.03, WIN, b[long_ok | short_ok])
    # cap gap at 8%
    run(stocks, events, "chron + gap≤8%", "both_fill", 0.03, WIN, b[b["gap"].abs() <= 0.08])
    # index not strongly against the fade
    bidx = pd.to_numeric(b["idx_gap"], errors="coerce")
    long_ok = (b["gap"] < 0) & (bidx <= 0.003)
    short_ok = (b["gap"] > 0) & (bidx >= -0.003)
    run(stocks, events, "chron + fade with index", "both_fill", 0.03, WIN, b[long_ok | short_ok])

    print("\n-- down ≥1% quality --")
    d1rsi = pd.to_numeric(d1["d_rsi14"], errors="coerce")
    d1rsi2 = pd.to_numeric(d1["d_rsi2"], errors="coerce")
    d1di_p = pd.to_numeric(d1["d_di_plus"], errors="coerce")
    d1di_m = pd.to_numeric(d1["d_di_minus"], errors="coerce")
    run(stocks, events, "BASE down≥1%", "down_bounce", 0.01, STORY)
    run(stocks, events, "≥1% +DI>-DI", "down_bounce", 0.01, STORY, d1[d1di_p > d1di_m])
    run(stocks, events, "≥1% RSI14 50-65", "down_bounce", 0.01, STORY, d1[d1rsi.between(50, 65)])
    run(stocks, events, "≥1% RSI2<20 + SMA200", "down_bounce", 0.01, STORY,
        d1[d1["d_above_sma200"].fillna(False).astype(bool) & (d1rsi2 < 20)])
    run(stocks, events, "≥1% +DI + RSI14 45-65", "down_bounce", 0.01, STORY,
        d1[(d1di_p > d1di_m) & d1rsi.between(45, 65)])
    run(stocks, events, "≥1% SMA20 + RSI14 45-65", "down_bounce", 0.01, STORY,
        d1[d1["d_above_sma20"].fillna(False).astype(bool) & d1rsi.between(45, 65)])


if __name__ == "__main__":
    main()
