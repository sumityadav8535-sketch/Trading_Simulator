"""
Nifty 200 · 5-minute strategy hunt.

₹1,00,000 capital, 10% equity risk per trade, 5× MIS cap.
Signal on 5m close, enter next bar open, costs on, SL-first same bar.

Usage:
    python scripts/intraday_5m_hunt.py
    python scripts/intraday_5m_hunt.py --skip-download
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_hunt import (  # noqa: E402
    attach_daily_st,
    attach_supertrend,
    months_from_trades,
    sig_nr7_break,
    sig_st_ema_mom,
    sig_st_flip,
    sig_st_pdh,
    sig_st_pullback,
    simulate_sized,
)
from scripts.intraday_15m_n200_phase3 import attach_index, sig_orb_fade_v2, sig_vwap_fade_v2  # noqa: E402
from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    DATA_5M,
    MARKET_OPEN,
    _base_cols,
    _can_trade_mask,
    _normalize,
    add_indicators,
    attach_daily_bias,
    build_exec_index,
    collect_signals,
    nifty200_symbols,
    sig_ema_momentum,
    sig_gap_and_go,
    sig_hod_break,
    sig_orb15,
    sig_pdh_break,
    sig_vwap_reclaim,
)
from trading.constants import NIFTY200_PROXY  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402
import scripts.intraday_15m_hunt as hunt15  # noqa: E402

OUT = ROOT / "data" / "intraday_5m_hunt.json"
TRADES = ROOT / "data" / "intraday_5m_hunt_trades.json"
INDEX_PKL = DATA_5M / "_NIFTY200_INDEX.pkl"

RISK_PCT = 10.0
LEVERAGE = 5.0
BATCH = 15


def _liq5(df: pd.DataFrame) -> pd.Series:
    return (df["close"] > 60) & (df["vol_sma"] > 8_000) & (df["atr"] / df["close"] >= 0.001)


hunt15._liq = _liq5


def download_5m_universe(symbols: list[str]) -> int:
    DATA_5M.mkdir(parents=True, exist_ok=True)
    saved = 0
    tickers = [yfinance_ticker(s) for s in symbols]
    t2s = dict(zip(tickers, symbols))
    print(f"Downloading 5m 60d for {len(tickers)} symbols…")
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        print(f"  batch {i // BATCH + 1}/{(len(tickers) + BATCH - 1) // BATCH} ({batch[0]}…)", flush=True)
        try:
            raw = yf.download(
                batch,
                interval="5m",
                period="60d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"    fail: {exc}")
            time.sleep(1.2)
            continue
        for ticker in batch:
            df = _normalize(raw, ticker)
            if df.empty or len(df) < 80:
                continue
            t = pd.Series(df.index.tz_convert("Asia/Kolkata").time, index=df.index)
            df = df.loc[(t >= MARKET_OPEN) & (t <= dtime(15, 30))]
            if df.empty:
                continue
            df.to_pickle(DATA_5M / f"{t2s[ticker]}.pkl")
            saved += 1
        time.sleep(0.5)

    print("  index ^CNX200 5m…", flush=True)
    try:
        raw = yf.download(NIFTY200_PROXY, interval="5m", period="60d", progress=False, auto_adjust=False)
        idx = _normalize(raw, NIFTY200_PROXY)
        if not idx.empty:
            idx.to_pickle(INDEX_PKL)
            saved += 1
    except Exception as exc:
        print("    index fail", exc)
    return saved


def load_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = DATA_5M / f"{sym}.pkl"
        if not path.exists():
            continue
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if df is None or df.empty or len(df) < 80:
            continue
        cache[sym] = df
    if INDEX_PKL.exists():
        cache["_INDEX"] = pd.read_pickle(INDEX_PKL)
    return cache


def fix_orb_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Opening ranges in clock time, no look-ahead on later bars if min_bar is respected."""
    sess = df["session"]
    first = df.groupby(sess).head(1)
    df["orb5_high"] = sess.map(first.set_index("session")["high"])
    df["orb5_low"] = sess.map(first.set_index("session")["low"])
    first3 = df.groupby(sess).head(3)
    df["orb15_high"] = sess.map(first3.groupby("session")["high"].max())
    df["orb15_low"] = sess.map(first3.groupby("session")["low"].min())
    first6 = df.groupby(sess).head(6)
    df["orb30_high"] = sess.map(first6.groupby("session")["high"].max())
    df["orb30_low"] = sess.map(first6.groupby("session")["low"].min())
    return df


def prepare(cache: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    stocks = {}
    items = [(k, v) for k, v in cache.items() if not k.startswith("_")]
    for i, (sym, df) in enumerate(items, 1):
        en = add_indicators(df)
        en = fix_orb_5m(en)
        en = attach_daily_bias(en, sym)
        en = attach_supertrend(en)
        en = attach_daily_st(en, sym)
        if len(en) >= 80:
            stocks[sym] = en
        if i % 40 == 0:
            print(f"  prepared {i}/{len(items)}", flush=True)
    idx = cache.get("_INDEX", pd.DataFrame())
    if not idx.empty:
        idx = add_indicators(idx)
        idx = fix_orb_5m(idx)
        idx["idx_day"] = idx["close"] / idx.groupby("session")["open"].transform("first") - 1.0
        idx = attach_supertrend(idx)
    return stocks, idx


def sig_orb5(df, target_r=1.5, vol_min=1.2, htf=False, daily_st=False, shorts=False):
    m = _can_trade_mask(df, 1) & _liq5(df)
    rng = (df["orb5_high"] - df["orb5_low"]) / df["close"]
    long = (
        m
        & (df["close"] > df["orb5_high"])
        & (df["prev_close"] <= df["orb5_high"])
        & (df["rel_vol"] >= vol_min)
        & (rng >= 0.0012)
        & (df["close"] > df["vwap"])
    )
    if htf:
        long &= df["htf_up"]
    if daily_st:
        long &= df["dst_up"]
    stop = df["orb5_low"]
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.4)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    if shorts:
        sh = (
            m
            & (df["close"] < df["orb5_low"])
            & (df["prev_close"] >= df["orb5_low"])
            & (df["rel_vol"] >= vol_min)
            & (rng >= 0.0012)
        )
        if daily_st:
            sh &= df["dst_dn"]
        sstop = df["orb5_high"]
        frames.append(_base_cols(df, sh, "short", sstop, df["close"] - (sstop - df["close"]).clip(lower=df["atr"] * 0.4) * target_r, score))
    return pd.concat(frames, ignore_index=True)


def sig_st_orb_5m(df, key="st10_30", target_r=1.5, vol_min=1.15, daily_st=True):
    m = _can_trade_mask(df, 3) & _liq5(df)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    long = (
        m
        & (df[f"{key}_dir"] == 1)
        & (df["close"] > df["orb15_high"])
        & (df["prev_close"] <= df["orb15_high"])
        & (df["rel_vol"] >= vol_min)
        & (rng >= 0.002)
        & (df["close"] > df["vwap"])
    )
    if daily_st:
        long &= df["dst_up"]
    stop = np.minimum(df["orb15_low"], df[key])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.4)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_orb15_5m(df, target_r=1.5, vol_min=1.2, htf=True, daily_st=False):
    m = _can_trade_mask(df, 3) & _liq5(df)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    long = (
        m
        & (df["close"] > df["orb15_high"])
        & (df["prev_close"] <= df["orb15_high"])
        & (df["rel_vol"] >= vol_min)
        & (rng >= 0.002)
        & (df["close"] > df["vwap"])
    )
    if htf:
        long &= df["htf_up"]
    if daily_st:
        long &= df["dst_up"]
    stop = df["orb15_low"]
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.4)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def family_specs():
    r = 1.5
    specs = [
        (f"ST10/3 flip {r}R VWAP", sig_st_flip, {"key": "st10_30", "target_r": r, "vwap": True, "daily_st": True}),
        (f"ST10/3 flip both {r}R", sig_st_flip, {"key": "st10_30", "target_r": r, "vwap": True, "shorts": True, "daily_st": True}),
        (f"ST10/3 pull {r}R VWAP dST", sig_st_pullback, {"key": "st10_30", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 16}),
        (f"ST7/3 pull {r}R VWAP dST", sig_st_pullback, {"key": "st7_30", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 16}),
        (f"ST10/2 pull {r}R VWAP dST", sig_st_pullback, {"key": "st10_20", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 18}),
        (f"ST+ORB15 {r}R", sig_st_orb_5m, {"key": "st10_30", "target_r": r, "daily_st": True}),
        (f"ST+PDH {r}R", sig_st_pdh, {"key": "st10_30", "target_r": r}),
        (f"ST+EMA mom {r}R", sig_st_ema_mom, {"key": "st10_30", "target_r": r}),
        (f"NR7 {r}R dST", sig_nr7_break, {"target_r": r, "daily_st": True}),
        (f"ORB5 {r}R dST", sig_orb5, {"target_r": r, "daily_st": True, "htf": True}),
        (f"ORB15 {r}R HTF", sig_orb15_5m, {"target_r": r, "htf": True, "daily_st": True}),
        (f"PDH {r}R HTF", sig_pdh_break, {"target_r": r, "vol_min": 1.2, "htf": True}),
        (f"HOD {r}R HTF", sig_hod_break, {"target_r": r, "htf": True}),
        (f"GapGo {r}R", sig_gap_and_go, {"target_r": r, "htf": True}),
        (f"VWAP reclaim {r}R", sig_vwap_reclaim, {"target_r": r, "htf": True}),
        (f"EMA mom {r}R", sig_ema_momentum, {"target_r": r, "htf": True}),
        ("ST10/3 pull both 1.5R", sig_st_pullback, {"key": "st10_30", "target_r": 1.5, "vwap": True, "daily_st": True, "shorts": True, "adx_min": 16}),
        ("VWAP fade 1.8% ATR", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}),
        ("VWAP fade 1.5% ATR", sig_vwap_fade_v2, {"ext_min": 0.015, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}),
        ("VWAP fade 1.2% ATR", sig_vwap_fade_v2, {"ext_min": 0.012, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}),
        ("VWAP fade 1.8% VWAP", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "vwap", "idx_abs_max": 0.008}),
        ("VWAP fade 1.5% VWAP", sig_vwap_fade_v2, {"ext_min": 0.015, "adx_max": 26, "target_mode": "vwap", "idx_abs_max": 0.008}),
        ("ORB fade 0.7%", sig_orb_fade_v2, {"ext": 0.007, "adx_max": 26}),
        ("ORB5 both 1.5R", sig_orb5, {"target_r": 1.5, "daily_st": True, "shorts": True}),
    ]
    return specs


def size_combos():
    # 10% risk; max_deploy chosen so max_pos * deploy * 5x stays near full MIS.
    return [
        dict(mode="risk", risk_pct=RISK_PCT, alloc_pct=0.0, leverage=LEVERAGE, max_pos=3, top_k=1, max_deploy=0.33, daily_lock=0.0, daily_halt=0.0),
        dict(mode="risk", risk_pct=RISK_PCT, alloc_pct=0.0, leverage=LEVERAGE, max_pos=4, top_k=2, max_deploy=0.25, daily_lock=0.0, daily_halt=0.0),
        dict(mode="risk", risk_pct=RISK_PCT, alloc_pct=0.0, leverage=LEVERAGE, max_pos=5, top_k=2, max_deploy=0.40, daily_lock=0.0, daily_halt=0.0),
        dict(mode="risk", risk_pct=RISK_PCT, alloc_pct=0.0, leverage=LEVERAGE, max_pos=3, top_k=1, max_deploy=0.33, daily_lock=8000.0, daily_halt=10000.0),
    ]


def rank_key(res) -> tuple:
    oos = res.oos_pnl if res.oos_days >= 5 else res.net_pnl * 0.3
    return (
        1 if res.oos_pnl > 0 and res.oos_days >= 5 else 0,
        oos,
        res.net_pnl,
        res.profit_factor,
        -res.max_dd_pct,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    symbols = nifty200_symbols()
    print(f"Nifty 200 5m hunt | ₹{CAPITAL:,.0f} | 10% risk/trade | {len(symbols)} names")
    if not args.skip_download:
        n = download_5m_universe(symbols)
        print(f"Saved {n} 5m frames")

    cache = load_frames(symbols)
    n_stocks = sum(1 for k in cache if not k.startswith("_"))
    print(f"Loaded {n_stocks} stock frames")
    if n_stocks < 30:
        print("Not enough 5m data.")
        return

    sample = next(df for k, df in cache.items() if not k.startswith("_"))
    print(f"Sample window {sample.index.min()} → {sample.index.max()}  bars={len(sample)}")

    print("Indicators + Supertrend on 5m…")
    stocks, idx = prepare(cache)
    stocks = attach_index(stocks, idx)
    print(f"Prepared {len(stocks)} stocks")
    exec_idx = build_exec_index(stocks)
    print(f"Calendar bars={len(exec_idx['calendar'])}")

    specs = family_specs()
    print(f"\nExtracting {len(specs)} families…")
    sig_cache = {}
    for name, fn, kw in specs:
        sigs = collect_signals(stocks, fn, **kw)
        sig_cache[name] = sigs
        print(f"  {name:34s} {len(sigs):5d}", flush=True)

    combos = size_combos()
    total = len(specs) * len(combos)
    print(f"\nSimulating {total} configs at 10% risk…")
    book = []
    n = 0
    for name, fn, kw in specs:
        sigs = sig_cache[name]
        if sigs is None or sigs.empty:
            continue
        for sz in combos:
            n += 1
            label = f"{name} | r{sz['risk_pct']:g} p{sz['max_pos']} k{sz['top_k']} d{sz['max_deploy']}"
            if sz["daily_halt"]:
                label += " halt"
            res, tdf = simulate_sized(exec_idx, sigs, label, **sz)
            res.params.update(kw)
            res.params["family"] = name
            months = months_from_trades(tdf)
            book.append((res, tdf, months))
            if n % 10 == 0 or n == total:
                print(f"  {n}/{total}  last {res.total_return_pct:7.1f}%  {res.name[:56]}", flush=True)

    ranked = sorted(book, key=lambda x: rank_key(x[0]), reverse=True)
    print("\n===== TOP 20 (OOS-positive preferred) =====")
    print(f"{'name':<56} {'n':>4} {'WR':>5} {'PF':>5} {'ret%':>7} {'PnL':>9} {'OOS':>8} {'DD':>5}")
    for res, tdf, months in ranked[:20]:
        print(
            f"{res.name[:56]:<56} {res.trades:4d} {res.win_rate:5.1f} {res.profit_factor:5.2f} "
            f"{res.total_return_pct:7.1f} {res.net_pnl:9.0f} {res.oos_pnl:8.0f} {res.max_dd_pct:5.1f}"
        )

    hits = [x for x in ranked if x[0].total_return_pct >= 100 and x[0].oos_pnl > 0 and x[0].trades >= 20]
    wres, wtdf, wmonths = hits[0] if hits else ranked[0]
    consistent = None
    for res, tdf, months in ranked:
        if res.win_rate >= 50 and res.oos_pnl > 0 and res.max_dd_pct <= 20 and res.trades >= 20:
            consistent = (res, tdf, months)
            break

    payload = {
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "leverage": LEVERAGE,
        "universe": "nifty200",
        "timeframe": "5m",
        "window": {
            "start": str(sample.index.min()),
            "end": str(sample.index.max()),
            "stocks": n_stocks,
            "bars": len(exec_idx["calendar"]),
        },
        "hit_100": bool(hits),
        "winner": {**asdict(wres), "months": wmonths},
        "consistent": {**asdict(consistent[0]), "months": consistent[2]} if consistent else None,
        "top": [{**asdict(r), "months": m} for r, _, m in ranked[:15]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if wtdf is not None and not wtdf.empty:
        wtdf.to_json(TRADES, orient="records", date_format="iso")
    print(f"\nWinner: {wres.name}")
    print(f"  return {wres.total_return_pct:.1f}%  PnL ₹{wres.net_pnl:,.0f}  WR {wres.win_rate:.1f}%  PF {wres.profit_factor:.2f}")
    print(f"  OOS ₹{wres.oos_pnl:,.0f}  DD {wres.max_dd_pct:.1f}%  trades {wres.trades}")
    print(f"  months {wmonths}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
