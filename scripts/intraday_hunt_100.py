"""
Broader 15m indicator hunt aimed at 100% / 2 months.

No look-ahead (signal close → next open), costs on, SL-first.
Loads cached Nifty 200 15m pickles.

    python scripts/intraday_hunt_100.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_hunt import (  # noqa: E402
    load_frames,
    months_from_trades,
    prepare,
    simulate_sized,
)
from scripts.intraday_15m_n200_phase3 import attach_index, sig_vwap_fade_v2  # noqa: E402
from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    _base_cols,
    _can_trade_mask,
    collect_signals,
    nifty200_symbols,
)

OUT = ROOT / "data" / "intraday_hunt_100.json"
TRADES = ROOT / "data" / "intraday_hunt_100_trades.json"


def _liq(df: pd.DataFrame) -> pd.Series:
    return (df["close"] > 60) & (df["vol_sma"] > 25_000) & (df["atr"] / df["close"] >= 0.002)


def add_extra(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    v = df["volume"].astype(float)
    sess = df["session"]

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_mid"] = mid
    df["bb_up"] = mid + 2.0 * sd
    df["bb_dn"] = mid - 2.0 * sd
    bw = (df["bb_up"] - df["bb_dn"]).replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_dn"]) / bw
    df["bb_bw"] = bw / mid.replace(0, np.nan)
    df["bb_bw_lo"] = df["bb_bw"] <= df["bb_bw"].rolling(40).quantile(0.2)

    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    rng14 = (hi14 - lo14).replace(0, np.nan)
    df["stoch_k"] = 100.0 * (c - lo14) / rng14
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_k_prev"] = df["stoch_k"].shift(1)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    tp = (h + l + c) / 3.0
    sma20 = tp.rolling(20).mean()
    mad = (tp - sma20).abs().rolling(20).mean()
    df["cci"] = (tp - sma20) / (0.015 * mad.replace(0, np.nan))
    df["cci_prev"] = df["cci"].shift(1)

    df["willr"] = -100.0 * (hi14 - c) / rng14
    df["don_hi"] = h.rolling(20).max().shift(1)
    df["don_lo"] = l.rolling(20).min().shift(1)

    df["kel_mid"] = c.ewm(span=20, adjust=False).mean()
    df["kel_up"] = df["kel_mid"] + 1.5 * df["atr"]
    df["kel_dn"] = df["kel_mid"] - 1.5 * df["atr"]
    df["squeeze"] = (df["bb_up"] < df["kel_up"]) & (df["bb_dn"] > df["kel_dn"])
    df["squeeze_prev"] = df["squeeze"].shift(1)

    dist = c - df["vwap"]
    exp_std = dist.groupby(sess).transform(lambda s: s.expanding(min_periods=5).std())
    df["vwap_z"] = dist / exp_std.replace(0, np.nan)

    df["pp"] = (df["pdh"] + df["pdl"] + df["pdc"]) / 3.0
    df["r1"] = 2.0 * df["pp"] - df["pdl"]
    df["s1"] = 2.0 * df["pp"] - df["pdh"]
    df["r2"] = df["pp"] + (df["pdh"] - df["pdl"])
    df["s2"] = df["pp"] - (df["pdh"] - df["pdl"])

    if "idx_day" in df.columns:
        df["rs"] = df["day_chg"] - df["idx_day"]
    else:
        df["rs"] = 0.0

    body = (c - df["open"]).abs()
    df["bull_engulf"] = (
        (df["open"].shift(1) > c.shift(1))
        & (c > df["open"])
        & (df["open"] <= c.shift(1))
        & (c >= df["open"].shift(1))
        & (body > body.shift(1))
    )
    df["bear_engulf"] = (
        (c.shift(1) > df["open"].shift(1))
        & (df["open"] > c)
        & (df["open"] >= c.shift(1))
        & (c <= df["open"].shift(1))
        & (body > body.shift(1))
    )
    return df


def sig_bb_fade(df, z=None, rsi_hi=68, rsi_lo=32, adx_max=28):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & _liq(df) & (t >= dtime(10, 0)) & (t <= dtime(14, 0))
    m &= df["adx"] <= adx_max
    if "idx_day" in df.columns:
        m &= df["idx_day"].abs() <= 0.008
    short = m & (df["close"] > df["bb_up"]) & (df["rsi"] >= rsi_hi)
    lng = m & (df["close"] < df["bb_dn"]) & (df["rsi"] <= rsi_lo)
    sstop = df["close"] + df["atr"] * 0.45
    lstop = df["close"] - df["atr"] * 0.45
    return pd.concat([
        _base_cols(df, short, "short", sstop, df["bb_mid"], df["rsi"]),
        _base_cols(df, lng, "long", lstop, df["bb_mid"], 100 - df["rsi"]),
    ], ignore_index=True)


def sig_bb_squeeze_break(df, target_r=1.5):
    m = _can_trade_mask(df, 3) & _liq(df)
    long = m & df["squeeze_prev"].fillna(False) & ~df["squeeze"].fillna(False)
    long &= (df["close"] > df["bb_up"]) & (df["close"] > df["vwap"]) & (df["rel_vol"] >= 1.2) & df["dst_up"]
    stop = np.minimum(df["low"], df["kel_mid"])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    return _base_cols(df, long, "long", stop, df["close"] + risk * target_r, df["rel_vol"].fillna(1))


def sig_stoch_cross(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = m & (df["stoch_k_prev"] < 20) & (df["stoch_k"] >= 20) & (df["close"] > df["vwap"]) & (df["close_loc"] >= 0.55)
    if htf:
        long &= df["htf_up"]
    short = m & (df["stoch_k_prev"] > 80) & (df["stoch_k"] <= 80) & (df["close"] < df["vwap"])
    stop_l = df["low"].rolling(4).min()
    stop_s = df["high"].rolling(4).max()
    rl = (df["close"] - stop_l).clip(lower=df["atr"] * 0.3)
    rs = (stop_s - df["close"]).clip(lower=df["atr"] * 0.3)
    return pd.concat([
        _base_cols(df, long, "long", stop_l, df["close"] + rl * target_r, 50 - df["stoch_k_prev"]),
        _base_cols(df, short, "short", stop_s, df["close"] - rs * target_r, df["stoch_k_prev"] - 50),
    ], ignore_index=True)


def sig_macd_cross(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 3) & _liq(df)
    long = m & (df["macd_hist_prev"] <= 0) & (df["macd_hist"] > 0) & (df["close"] > df["vwap"]) & (df["ema9"] > df["ema21"])
    if htf:
        long &= df["htf_up"]
    short = m & (df["macd_hist_prev"] >= 0) & (df["macd_hist"] < 0) & (df["close"] < df["vwap"])
    sl = np.minimum(df["low"], df["ema21"])
    ss = np.maximum(df["high"], df["ema21"])
    return pd.concat([
        _base_cols(df, long, "long", sl, df["close"] + (df["close"] - sl).clip(lower=df["atr"] * 0.3) * target_r, df["macd_hist"]),
        _base_cols(df, short, "short", ss, df["close"] - (ss - df["close"]).clip(lower=df["atr"] * 0.3) * target_r, -df["macd_hist"]),
    ], ignore_index=True)


def sig_cci_reclaim(df, target_r=1.5):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = m & (df["cci_prev"] < -100) & (df["cci"] > -50) & (df["close"] > df["vwap"]) & df["htf_up"]
    short = m & (df["cci_prev"] > 100) & (df["cci"] < 50) & (df["close"] < df["vwap"])
    sl = df["low"].rolling(5).min()
    ss = df["high"].rolling(5).max()
    return pd.concat([
        _base_cols(df, long, "long", sl, df["close"] + (df["close"] - sl).clip(lower=df["atr"] * 0.3) * target_r, df["cci"]),
        _base_cols(df, short, "short", ss, df["close"] - (ss - df["close"]).clip(lower=df["atr"] * 0.3) * target_r, -df["cci"]),
    ], ignore_index=True)


def sig_willr_fade(df):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & _liq(df) & (t >= dtime(10, 0)) & (df["adx"] <= 26)
    short = m & (df["willr"] >= -15) & (df["close"] > df["vwap"] * 1.012)
    lng = m & (df["willr"] <= -85) & (df["close"] < df["vwap"] * 0.988)
    return pd.concat([
        _base_cols(df, short, "short", df["close"] + df["atr"] * 0.45, df["vwap"], -df["willr"]),
        _base_cols(df, lng, "long", df["close"] - df["atr"] * 0.45, df["vwap"], 100 + df["willr"]),
    ], ignore_index=True)


def sig_donchian(df, target_r=1.5):
    m = _can_trade_mask(df, 4) & _liq(df)
    long = m & (df["close"] > df["don_hi"]) & (df["prev_close"] <= df["don_hi"]) & (df["rel_vol"] >= 1.2) & df["dst_up"] & (df["close"] > df["vwap"])
    stop = df["don_lo"]
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.4)
    return _base_cols(df, long, "long", stop, df["close"] + risk * target_r, df["rel_vol"].fillna(1))


def sig_vwap_z_fade(df, zmin=1.8, zmax=4.0, adx_max=26):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 3) & _liq(df) & (t >= dtime(10, 0)) & (t <= dtime(14, 0))
    m &= df["adx"] <= adx_max
    if "idx_day" in df.columns:
        m &= df["idx_day"].abs() <= 0.008
    z = df["vwap_z"]
    short = m & z.between(zmin, zmax) & (df["rsi"] >= 65)
    lng = m & z.between(-zmax, -zmin) & (df["rsi"] <= 35)
    return pd.concat([
        _base_cols(df, short, "short", df["close"] + df["atr"] * 0.45, df["vwap"], z),
        _base_cols(df, lng, "long", df["close"] - df["atr"] * 0.45, df["vwap"], -z),
    ], ignore_index=True)


def sig_pivot_fade(df):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & _liq(df) & (t >= dtime(10, 0)) & (df["adx"] <= 28)
    short = m & (df["high"] >= df["r1"]) & (df["close"] < df["r1"]) & (df["rsi"] >= 62)
    lng = m & (df["low"] <= df["s1"]) & (df["close"] > df["s1"]) & (df["rsi"] <= 38)
    return pd.concat([
        _base_cols(df, short, "short", df["close"] + df["atr"] * 0.45, df["pp"], df["rsi"]),
        _base_cols(df, lng, "long", df["close"] - df["atr"] * 0.45, df["pp"], 100 - df["rsi"]),
    ], ignore_index=True)


def sig_rs_break(df, target_r=1.4):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = (
        m & (df["rs"] >= 0.012) & (df["close"] > df["orb15_high"])
        & (df["close"] > df["vwap"]) & (df["rel_vol"] >= 1.2) & df["htf_up"]
        & (df["prev_close"] <= df["orb15_high"])
    )
    stop = np.minimum(df["orb15_low"], df["low"])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    return _base_cols(df, long, "long", stop, df["close"] + risk * target_r, df["rs"] * 100)


def sig_engulf(df, target_r=1.5):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = m & df["bull_engulf"] & (df["close"] > df["vwap"]) & df["htf_up"] & (df["rel_vol"] >= 1.1)
    short = m & df["bear_engulf"] & (df["close"] < df["vwap"])
    sl = df["low"].shift(1)
    ss = df["high"].shift(1)
    return pd.concat([
        _base_cols(df, long, "long", sl, df["close"] + (df["close"] - sl).clip(lower=df["atr"] * 0.3) * target_r, df["rel_vol"].fillna(1)),
        _base_cols(df, short, "short", ss, df["close"] - (ss - df["close"]).clip(lower=df["atr"] * 0.3) * target_r, df["rel_vol"].fillna(1)),
    ], ignore_index=True)


def sig_keltner_fade(df):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & _liq(df) & (t >= dtime(10, 0)) & (df["adx"] <= 26)
    short = m & (df["close"] > df["kel_up"]) & (df["rsi"] >= 68)
    lng = m & (df["close"] < df["kel_dn"]) & (df["rsi"] <= 32)
    return pd.concat([
        _base_cols(df, short, "short", df["close"] + df["atr"] * 0.45, df["kel_mid"], df["rsi"]),
        _base_cols(df, lng, "long", df["close"] - df["atr"] * 0.45, df["kel_mid"], 100 - df["rsi"]),
    ], ignore_index=True)


def sig_vwap_band_fade(df, k=1.6):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & _liq(df) & (t >= dtime(10, 0)) & (t <= dtime(14, 0)) & (df["adx"] <= 26)
    if "idx_day" in df.columns:
        m &= df["idx_day"].abs() <= 0.01
    short = m & (df["close"] > df["vwap"] + k * df["atr"]) & (df["rsi"] >= 66)
    lng = m & (df["close"] < df["vwap"] - k * df["atr"]) & (df["rsi"] <= 34)
    return pd.concat([
        _base_cols(df, short, "short", df["close"] + df["atr"] * 0.45, df["vwap"], df["rsi"]),
        _base_cols(df, lng, "long", df["close"] - df["atr"] * 0.45, df["vwap"], 100 - df["rsi"]),
    ], ignore_index=True)


def family_specs():
    return [
        ("VWAP fade 1.8% VWAP", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "vwap", "idx_abs_max": 0.008}),
        ("VWAP fade 1.8% ATR", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}),
        ("VWAP fade 1.5% VWAP", sig_vwap_fade_v2, {"ext_min": 0.015, "adx_max": 26, "target_mode": "vwap", "idx_abs_max": 0.008}),
        ("VWAP fade 2.2% VWAP", sig_vwap_fade_v2, {"ext_min": 0.022, "adx_max": 28, "target_mode": "vwap", "idx_abs_max": 0.01}),
        ("VWAP z 2.0 fade", sig_vwap_z_fade, {"zmin": 2.0, "zmax": 4.5, "adx_max": 26}),
        ("VWAP z 1.6 fade", sig_vwap_z_fade, {"zmin": 1.6, "zmax": 4.0, "adx_max": 28}),
        ("VWAP ATR-band 1.6", sig_vwap_band_fade, {"k": 1.6}),
        ("VWAP ATR-band 2.0", sig_vwap_band_fade, {"k": 2.0}),
        ("BB fade", sig_bb_fade, {}),
        ("BB squeeze break", sig_bb_squeeze_break, {"target_r": 1.5}),
        ("Stoch 20/80 cross", sig_stoch_cross, {"target_r": 1.5, "htf": True}),
        ("MACD hist cross", sig_macd_cross, {"target_r": 1.5, "htf": True}),
        ("CCI reclaim", sig_cci_reclaim, {"target_r": 1.5}),
        ("Williams fade", sig_willr_fade, {}),
        ("Donchian 20 break", sig_donchian, {"target_r": 1.5}),
        ("Keltner fade", sig_keltner_fade, {}),
        ("Pivot R1/S1 fade", sig_pivot_fade, {}),
        ("RS + ORB break", sig_rs_break, {"target_r": 1.4}),
        ("Engulfing", sig_engulf, {"target_r": 1.5}),
    ]


def size_combos():
    combos = []
    for risk in (10.0, 15.0, 20.0, 25.0):
        for max_pos, top_k, deploy in ((4, 1, 0.45), (5, 2, 0.60), (6, 2, 0.80)):
            combos.append(dict(
                mode="risk", risk_pct=risk, alloc_pct=0.0, leverage=5.0,
                max_pos=max_pos, top_k=top_k, max_deploy=deploy,
                daily_lock=0.0, daily_halt=0.0,
            ))
    return combos


def rank_key(res):
    oos = res.oos_pnl if res.oos_days >= 5 else -1e9
    return (
        1 if res.total_return_pct >= 100 and oos > 0 else 0,
        1 if oos > 0 else 0,
        res.total_return_pct,
        res.profit_factor,
        -res.max_dd_pct,
    )


def main():
    symbols = nifty200_symbols()
    print(f"15m extra-indicator hunt | 100% target | {len(symbols)} names")
    cache = load_frames(symbols)
    n_stocks = sum(1 for k in cache if not k.startswith("_"))
    print(f"Loaded {n_stocks} 15m frames")
    stocks, idx = prepare(cache)
    stocks = attach_index(stocks, idx)
    print("Extra indicators (BB/MACD/stoch/z/pivots)…")
    for i, (sym, df) in enumerate(stocks.items(), 1):
        stocks[sym] = add_extra(df)
        if i % 50 == 0:
            print(f"  {i}/{len(stocks)}", flush=True)
    from scripts.intraday_15m_n200_search import build_exec_index

    exec_idx = build_exec_index(stocks)
    print(f"Prepared {len(stocks)}  calendar={len(exec_idx['calendar'])}")

    specs = family_specs()
    print(f"\nExtracting {len(specs)} families…")
    sig_cache = {}
    for name, fn, kw in specs:
        sigs = collect_signals(stocks, fn, **kw)
        sig_cache[name] = sigs
        print(f"  {name:28s} {len(sigs):5d}", flush=True)

    combos = size_combos()
    total = len(specs) * len(combos)
    print(f"\nSimulating {total} configs (risk 10–25%)…")
    book = []
    n = 0
    for name, fn, kw in specs:
        sigs = sig_cache[name]
        if sigs is None or sigs.empty:
            continue
        for sz in combos:
            n += 1
            label = f"{name} | r{sz['risk_pct']:g} p{sz['max_pos']} k{sz['top_k']}"
            res, tdf = simulate_sized(exec_idx, sigs, label, **sz)
            res.params.update(kw)
            res.params["family"] = name
            months = months_from_trades(tdf)
            book.append((res, tdf, months))
            if n % 25 == 0 or n == total:
                print(f"  {n}/{total}  last {res.total_return_pct:7.1f}%  {label[:60]}", flush=True)

    ranked = sorted(book, key=lambda x: rank_key(x[0]), reverse=True)
    print("\n===== TOP 25 =====")
    print(f"{'name':<52} {'n':>4} {'WR':>5} {'PF':>5} {'ret%':>7} {'OOS':>8} {'DD':>5}")
    for res, _, _ in ranked[:25]:
        print(
            f"{res.name[:52]:<52} {res.trades:4d} {res.win_rate:5.1f} {res.profit_factor:5.2f} "
            f"{res.total_return_pct:7.1f} {res.oos_pnl:8.0f} {res.max_dd_pct:5.1f}"
        )

    hits = [
        x for x in ranked
        if x[0].total_return_pct >= 100 and x[0].oos_pnl > 0 and x[0].trades >= 15
    ]
    wres, wtdf, wmonths = hits[0] if hits else ranked[0]
    payload = {
        "capital": CAPITAL,
        "timeframe": "15m",
        "hit_100": bool(hits),
        "n_configs": total,
        "winner": {**asdict(wres), "months": wmonths},
        "hits": [{**asdict(r), "months": m} for r, _, m in hits[:8]],
        "top": [{**asdict(r), "months": m} for r, _, m in ranked[:20]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if wtdf is not None and not wtdf.empty:
        wtdf.to_json(TRADES, orient="records", date_format="iso")
    print(f"\nHit 100%? {bool(hits)}  n_hits={len(hits)}")
    print(f"Winner {wres.name}")
    print(f"  {wres.total_return_pct:.1f}%  PnL ₹{wres.net_pnl:,.0f}  WR {wres.win_rate:.1f}  PF {wres.profit_factor:.2f}  DD {wres.max_dd_pct:.1f}%")
    print(f"  months {wmonths}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
