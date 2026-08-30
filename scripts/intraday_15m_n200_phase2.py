"""
Phase 2/3: diagnose + hunt fades / quality filters on 15m Nifty 200.
Reuses cached 15m frames from phase 1.
"""
from __future__ import annotations

import json
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

from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    COST,
    DATA_15M,
    FORCE_EXIT,
    NO_ENTRY_AFTER,
    TARGET_LO,
    _base_cols,
    _can_trade_mask,
    add_indicators,
    attach_daily_bias,
    build_exec_index,
    collect_signals,
    load_universe,
    nifty200_symbols,
    simulate,
)
from dataclasses import asdict
from datetime import time as dtime

from scripts.intraday_15m_n200_search import SimResult  # noqa: E402

OUT = ROOT / "data" / "intraday_15m_n200_phase2.json"


def load_prepared():
    cache = load_universe(nifty200_symbols(), skip_download=True)
    stocks = {}
    for sym, df in cache.items():
        if sym.startswith("_"):
            continue
        en = add_indicators(df)
        en = attach_daily_bias(en, sym)
        if len(en) >= 50:
            stocks[sym] = en
    idx = cache.get("_INDEX", pd.DataFrame())
    if not idx.empty:
        idx = add_indicators(idx)
    return stocks, idx


def diagnose_index(idx: pd.DataFrame):
    if idx.empty:
        print("No index")
        return
    daily = idx.groupby("session").agg(o=("open", "first"), c=("close", "last"))
    daily["ret"] = daily["c"] / daily["o"] - 1
    print("\n=== Index session stats ===")
    print(f"days={len(daily)}  up={(daily['ret']>0).mean()*100:.1f}%  "
          f"avg={daily['ret'].mean()*100:.2f}%  "
          f"cum={(daily['c'].iloc[-1]/daily['o'].iloc[0]-1)*100:.1f}%")
    print(daily["ret"].describe())


# ── new signal families ──────────────────────────────────────────


def sig_orb_fade(df, target_r=1.0, ext=0.006):
    """Fade an opening-drive extreme back toward VWAP / mid of ORB."""
    m = _can_trade_mask(df, 1)
    t = pd.Series(df.index.time, index=df.index)
    m &= t <= dtime(12, 0)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    # stretched above ORB
    short = m & (df["close"] > df["orb15_high"] * (1 + ext)) & (rng >= 0.004) & (df["rsi"] > 65)
    short &= df["close_loc"] >= 0.7
    stop = df["high"].rolling(2).max() * 1.001
    mid = (df["orb15_high"] + df["orb15_low"]) / 2
    tgt = np.maximum(df["vwap"], mid)
    score = (df["rsi"] - 60) * df["rel_vol"].fillna(1)
    frames = [_base_cols(df, short, "short", stop, tgt, score)]
    # stretched below ORB
    lng = m & (df["close"] < df["orb15_low"] * (1 - ext)) & (rng >= 0.004) & (df["rsi"] < 35)
    lng &= df["close_loc"] <= 0.3
    lstop = df["low"].rolling(2).min() * 0.999
    ltgt = np.minimum(df["vwap"], mid)
    frames.append(_base_cols(df, lng, "long", lstop, ltgt, (40 - df["rsi"]) * df["rel_vol"].fillna(1)))
    return pd.concat(frames, ignore_index=True)


def sig_vwap_fade(df, ext=0.008):
    m = _can_trade_mask(df, 2)
    t = pd.Series(df.index.time, index=df.index)
    m &= t <= dtime(13, 30)
    dist = (df["close"] - df["vwap"]) / df["vwap"]
    short = m & (dist > ext) & (df["rsi"] > 68) & (df["close_loc"] >= 0.65)
    stop = df["high"].rolling(2).max()
    tgt = df["vwap"]
    score = dist * 100 * df["rsi"] / 70
    frames = [_base_cols(df, short, "short", stop, tgt, score)]
    lng = m & (dist < -ext) & (df["rsi"] < 32) & (df["close_loc"] <= 0.35)
    lstop = df["low"].rolling(2).min()
    frames.append(_base_cols(df, lng, "long", lstop, df["vwap"], (-dist) * 100))
    return pd.concat(frames, ignore_index=True)


def sig_failed_orb(df, target_r=1.2):
    """Break ORB then close back inside — fade the failure."""
    m = _can_trade_mask(df, 2)
    failed_up = m & (df["high"] > df["orb15_high"]) & (df["close"] < df["orb15_high"]) & (df["close"] > df["orb15_low"])
    failed_up &= df["rel_vol"] >= 1.1
    stop = df["high"]
    tgt = df["close"] - (stop - df["close"]) * target_r
    frames = [_base_cols(df, failed_up, "short", stop, tgt, df["rel_vol"].fillna(1))]
    failed_dn = m & (df["low"] < df["orb15_low"]) & (df["close"] > df["orb15_low"]) & (df["close"] < df["orb15_high"])
    failed_dn &= df["rel_vol"] >= 1.1
    lstop = df["low"]
    ltgt = df["close"] + (df["close"] - lstop) * target_r
    frames.append(_base_cols(df, failed_dn, "long", lstop, ltgt, df["rel_vol"].fillna(1)))
    return pd.concat(frames, ignore_index=True)


def sig_atr_momentum(df, atr_stop=0.7, atr_tgt=1.2, htf=True):
    m = _can_trade_mask(df, 2)
    long = (
        m & (df["ema9"] > df["ema21"]) & (df["close"] > df["vwap"])
        & (df["close_loc"] >= 0.7) & (df["rel_vol"] >= 1.3)
        & (df["adx"] >= 20) & (df["rsi"].between(52, 70))
        & (df["close"] > 80)
    )
    if htf:
        long &= df["htf_up"]
    stop = df["close"] - df["atr"] * atr_stop
    tgt = df["close"] + df["atr"] * atr_tgt
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    sh = (
        m & (df["ema9"] < df["ema21"]) & (df["close"] < df["vwap"])
        & (df["close_loc"] <= 0.3) & (df["rel_vol"] >= 1.3)
        & (df["adx"] >= 20) & (df["rsi"].between(30, 48))
    )
    if htf:
        sh &= df["htf_dn"]
    sstop = df["close"] + df["atr"] * atr_stop
    stgt = df["close"] - df["atr"] * atr_tgt
    frames.append(_base_cols(df, sh, "short", sstop, stgt, score))
    return pd.concat(frames, ignore_index=True)


def sig_quality_orb(df, target_r=1.4, vol_min=1.4):
    """ORB only if HTF up, above VWAP, strong close, liquid, not first 15m."""
    m = _can_trade_mask(df, 1)
    t = pd.Series(df.index.time, index=df.index)
    m &= t <= dtime(12, 30)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    long = (
        m & df["htf_up"] & (df["close"] > df["orb15_high"]) & (df["prev_close"] <= df["orb15_high"])
        & (df["close"] > df["vwap"]) & (df["rel_vol"] >= vol_min)
        & (rng.between(0.003, 0.025)) & (df["close_loc"] >= 0.7)
        & (df["close"] > 100) & (df["vol_sma"] > 50_000)
        & (df["adx"] >= 16)
    )
    stop = np.maximum(df["orb15_low"], df["close"] - df["atr"] * 1.2)
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(0) * df["adx"].fillna(10) * (1 + df["day_chg"].clip(lower=0) * 15)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_rs_break(df, target_r=1.3):
    """Session leader: up strongly vs open, break session high, volume."""
    m = _can_trade_mask(df, 2)
    t = pd.Series(df.index.time, index=df.index)
    m &= (t >= dtime(9, 45)) & (t <= dtime(13, 0))
    prior_hod = df["high"].groupby(df["session"]).cummax().shift(1)
    long = (
        m & (df["day_chg"] > 0.008) & (df["close"] > prior_hod)
        & (df["close"] > df["vwap"]) & (df["rel_vol"] >= 1.4)
        & df["htf_up"] & (df["close"] > 80)
    )
    stop = np.minimum(df["low"], df["vwap"])
    tgt = df["close"] + (df["close"] - stop) * target_r
    score = df["day_chg"] * 100 * df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_ema_ext_fade(df, atr_mult=1.6):
    m = _can_trade_mask(df, 3)
    ext = (df["close"] - df["ema21"]) / df["atr"].replace(0, np.nan)
    short = m & (ext > atr_mult) & (df["rsi"] > 70)
    stop = df["close"] + df["atr"] * 0.6
    tgt = df["ema21"]
    frames = [_base_cols(df, short, "short", stop, tgt, ext)]
    lng = m & (ext < -atr_mult) & (df["rsi"] < 30)
    lstop = df["close"] - df["atr"] * 0.6
    frames.append(_base_cols(df, lng, "long", lstop, df["ema21"], -ext))
    return pd.concat(frames, ignore_index=True)


def sig_lunch_trend(df, target_r=1.5):
    """After 11:00, trade in the direction of the morning if still aligned with VWAP."""
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 6) & (t >= dtime(11, 0)) & (t <= dtime(13, 30))
    long = (
        m & (df["day_chg"] > 0.006) & (df["close"] > df["vwap"])
        & (df["ema9"] > df["ema21"]) & (df["rel_vol"] >= 1.1)
        & (df["close_loc"] >= 0.6) & df["htf_up"]
    )
    stop = np.minimum(df["low"].rolling(3).min(), df["vwap"])
    tgt = df["close"] + (df["close"] - stop) * target_r
    frames = [_base_cols(df, long, "long", stop, tgt, df["day_chg"] * 100)]
    sh = (
        m & (df["day_chg"] < -0.006) & (df["close"] < df["vwap"])
        & (df["ema9"] < df["ema21"]) & (df["rel_vol"] >= 1.1)
        & (df["close_loc"] <= 0.4) & df["htf_dn"]
    )
    sstop = np.maximum(df["high"].rolling(3).max(), df["vwap"])
    stgt = df["close"] - (sstop - df["close"]) * target_r
    frames.append(_base_cols(df, sh, "short", sstop, stgt, -df["day_chg"] * 100))
    return pd.concat(frames, ignore_index=True)


def raw_edge(exec_idx, sigs, label, n=80):
    """Unconstrained next-bar 1-position expectancy (first n*10 signals)."""
    if sigs.empty:
        print(f"  {label}: 0 signals")
        return
    res, tdf = simulate(exec_idx, sigs, label, risk_pct=1.5, max_pos=8, top_k=3,
                        daily_lock=0, daily_halt=0)
    print(f"  {label:28s} sigs={len(sigs):5d} trd={res.trades:4d} WR={res.win_rate:5.1f} "
          f"PF={res.profit_factor:5.2f} pnl={res.net_pnl:8.0f} avg/d={res.avg_daily_pnl:7.0f} "
          f"OOS={res.oos_avg_daily:7.0f} DD={res.max_dd_pct:5.1f}")
    return res, tdf


def main():
    print("Loading prepared 15m Nifty 200...")
    stocks, idx = load_prepared()
    print(f"stocks={len(stocks)}")
    diagnose_index(idx)
    exec_idx = build_exec_index(stocks)
    print(f"calendar={len(exec_idx['calendar'])}")

    families = [
        ("ORB fade", sig_orb_fade, {}),
        ("VWAP fade 0.8%", sig_vwap_fade, {"ext": 0.008}),
        ("VWAP fade 1.2%", sig_vwap_fade, {"ext": 0.012}),
        ("Failed ORB 1.2R", sig_failed_orb, {"target_r": 1.2}),
        ("Failed ORB 0.9R", sig_failed_orb, {"target_r": 0.9}),
        ("ATR mom 0.7/1.2", sig_atr_momentum, {"atr_stop": 0.7, "atr_tgt": 1.2, "htf": True}),
        ("ATR mom 0.5/1.0", sig_atr_momentum, {"atr_stop": 0.5, "atr_tgt": 1.0, "htf": True}),
        ("Quality ORB 1.4R", sig_quality_orb, {"target_r": 1.4, "vol_min": 1.4}),
        ("Quality ORB 1.2R v1.2", sig_quality_orb, {"target_r": 1.2, "vol_min": 1.2}),
        ("RS break 1.3R", sig_rs_break, {"target_r": 1.3}),
        ("EMA ext fade 1.6", sig_ema_ext_fade, {"atr_mult": 1.6}),
        ("EMA ext fade 2.0", sig_ema_ext_fade, {"atr_mult": 2.0}),
        ("Lunch trend 1.5R", sig_lunch_trend, {"target_r": 1.5}),
        ("Lunch trend 1.2R", sig_lunch_trend, {"target_r": 1.2}),
    ]

    print("\n=== Phase 2 family scan (r2% p6 k2 no lock) ===")
    book = []
    for name, fn, kw in families:
        sigs = collect_signals(stocks, fn, **kw)
        res, tdf = simulate(exec_idx, sigs, name, risk_pct=2.0, max_pos=6, top_k=2)
        res.params = {"family": name, **kw}
        book.append((res, tdf, sigs))
        print(f"  {name:24s} sigs={len(sigs):5d} trd={res.trades:4d} WR={res.win_rate:5.1f} "
              f"PF={res.profit_factor:5.2f} pnl={res.net_pnl:8.0f} avg/d={res.avg_daily_pnl:7.0f} "
              f"OOS={res.oos_avg_daily:7.0f} ge2k={res.days_ge_2k}/{res.trading_days} DD={res.max_dd_pct:4.1f}")

    book.sort(key=lambda x: (x[0].oos_avg_daily if x[0].oos_days >= 5 else x[0].avg_daily_pnl, x[0].avg_daily_pnl), reverse=True)
    print("\nTop phase-2:")
    for res, _, _ in book[:8]:
        print(f"  {res.name}: avg {res.avg_daily_pnl} OOS {res.oos_avg_daily} WR {res.win_rate} PF {res.profit_factor}")

    # Grid the best 4 families
    top_names = [b[0].name for b in book[:4]]
    print("\n=== Phase 2b grid on top families ===")
    grid_book = []
    fn_map = {n: (fn, kw) for n, fn, kw in families}
    for tname in top_names:
        fn, kw = fn_map[tname]
        sigs = collect_signals(stocks, fn, **kw)
        for risk in (1.5, 2.0, 3.0):
            for pos in (3, 5, 8):
                for topk in (1, 2):
                    for lock, halt in ((0, 0), (3000, 2000), (5000, 2500)):
                        label = f"{tname} r{risk} p{pos} k{topk} L{lock}"
                        res, tdf = simulate(exec_idx, sigs, label, risk_pct=risk, max_pos=pos,
                                            top_k=topk, daily_lock=lock, daily_halt=halt)
                        res.params = {"family": tname, "risk": risk, "max_pos": pos, "top_k": topk,
                                      "lock": lock, "halt": halt}
                        grid_book.append((res, tdf))

    grid_book.sort(key=lambda x: (x[0].oos_avg_daily if x[0].oos_days >= 5 else -9999, x[0].avg_daily_pnl), reverse=True)
    print("\nTop grid:")
    for res, _ in grid_book[:15]:
        print(f"  {res.name:42s} trd={res.trades:4d} WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} "
              f"avg/d={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} ge2k={res.days_ge_2k}/{res.trading_days} "
              f"DD={res.max_dd_pct:4.1f} worst={res.worst_day:7.0f}")

    hits = [r for r, _ in grid_book if r.avg_daily_pnl >= TARGET_LO and r.oos_avg_daily >= 800]
    payload = {
        "phase2_top": [asdict(r) for r, _, _ in book[:10]],
        "grid_top": [asdict(r) for r, _ in grid_book[:20]],
        "hits": [asdict(r) for r in hits[:5]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {OUT}  hits={len(hits)}")


if __name__ == "__main__":
    main()
