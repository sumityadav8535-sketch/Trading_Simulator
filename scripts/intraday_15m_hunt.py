"""
Nifty 200 · 15-minute strategy hunt.

₹1,00,000 capital, 5% of equity allocated per trade (MIS 5× → 25% notional),
signal on 15m close, enter next bar open, costs on, SL-first same bar.

Families: Supertrend flip/pullback, ST+VWAP/ORB/PDH, classic breakouts,
NR7, VWAP fade (prior winner). Last ~60 trading days from Yahoo 15m.

Usage:
    python scripts/intraday_15m_hunt.py
    python scripts/intraday_15m_hunt.py --skip-download
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

from scripts.intraday_15m_n200_phase3 import attach_index, sig_orb_fade_v2, sig_vwap_fade_v2  # noqa: E402
from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    DATA_15M,
    MARKET_OPEN,
    _base_cols,
    _can_trade_mask,
    _normalize,
    add_indicators,
    attach_daily_bias,
    build_exec_index,
    collect_signals,
    nifty200_symbols,
    simulate as simulate_base,
    sig_ema_momentum,
    sig_gap_and_go,
    sig_hod_break,
    sig_orb15,
    sig_pdh_break,
    sig_vwap_reclaim,
)
from stage_analysis_v2.services.supertrend_swing import supertrend_np  # noqa: E402
from trading.constants import NIFTY200_PROXY  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402

OUT = ROOT / "data" / "intraday_15m_hunt.json"
TRADES = ROOT / "data" / "intraday_15m_hunt_trades.json"

ALLOC_PCT = 5.0
LEVERAGE = 5.0
BATCH = 20


def download_15m_universe(symbols: list[str], force: bool = False) -> int:
    DATA_15M.mkdir(parents=True, exist_ok=True)
    saved = 0
    tickers = [yfinance_ticker(s) for s in symbols]
    t2s = dict(zip(tickers, symbols))
    print(f"Downloading 15m 60d for {len(tickers)} symbols…")
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        print(f"  batch {i // BATCH + 1}/{(len(tickers) + BATCH - 1) // BATCH} ({batch[0]}…)", flush=True)
        try:
            raw = yf.download(
                batch,
                interval="15m",
                period="60d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"    fail: {exc}")
            time.sleep(1.0)
            continue
        for ticker in batch:
            df = _normalize(raw, ticker)
            if df.empty or len(df) < 40:
                continue
            t = pd.Series(df.index.tz_convert("Asia/Kolkata").time, index=df.index)
            df = df.loc[(t >= MARKET_OPEN) & (t <= dtime(15, 30))]
            if df.empty:
                continue
            path = DATA_15M / f"{t2s[ticker]}.pkl"
            df.to_pickle(path)
            saved += 1
        time.sleep(0.45)

    print("  index ^CNX200…", flush=True)
    try:
        raw = yf.download(NIFTY200_PROXY, interval="15m", period="60d", progress=False, auto_adjust=False)
        idx = _normalize(raw, NIFTY200_PROXY)
        if not idx.empty:
            idx.to_pickle(DATA_15M / "_NIFTY200_INDEX.pkl")
            saved += 1
    except Exception as exc:
        print("    index fail", exc)
    return saved


def load_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = DATA_15M / f"{sym}.pkl"
        if not path.exists():
            continue
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if df is None or df.empty or len(df) < 40:
            continue
        cache[sym] = df
    idx_path = DATA_15M / "_NIFTY200_INDEX.pkl"
    if idx_path.exists():
        cache["_INDEX"] = pd.read_pickle(idx_path)
    return cache


def attach_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    for period, mult in ((7, 3.0), (10, 3.0), (10, 2.0), (14, 3.0)):
        st, direction = supertrend_np(h, l, c, period, mult)
        key = f"st{period}_{int(mult * 10)}"
        df[key] = st
        df[f"{key}_dir"] = direction
        prev = pd.Series(direction, index=df.index).shift(1)
        df[f"{key}_flip_up"] = (df[f"{key}_dir"] == 1) & (prev <= 0)
        df[f"{key}_flip_dn"] = (df[f"{key}_dir"] == -1) & (prev >= 0)
        tagged = (df["low"] <= df[key] * 1.0025) | (df["high"] >= df[key] * 0.9975)
        df[f"{key}_tag"] = tagged.shift(1).fillna(False) & (df[f"{key}_dir"].shift(1) == 1)
        df[f"{key}_tag_dn"] = tagged.shift(1).fillna(False) & (df[f"{key}_dir"].shift(1) == -1)
    return df


def attach_daily_st(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    daily = load_price_dataframe(symbol)
    if daily.empty or len(daily) < 30:
        df["dst_up"] = False
        df["dst_dn"] = False
        return df
    st, direction = supertrend_np(
        daily["high"].to_numpy(float),
        daily["low"].to_numpy(float),
        daily["close"].to_numpy(float),
        10,
        3.0,
    )
    lookup = pd.Series(direction, index=pd.to_datetime(daily.index).date).shift(1)
    mapped = df["session"].map(lookup)
    df["dst_up"] = mapped.fillna(0) > 0
    df["dst_dn"] = mapped.fillna(0) < 0
    return df


def prepare(cache: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    stocks = {}
    for i, (sym, df) in enumerate(cache.items(), 1):
        if sym.startswith("_"):
            continue
        en = add_indicators(df)
        en = attach_daily_bias(en, sym)
        en = attach_supertrend(en)
        en = attach_daily_st(en, sym)
        if len(en) >= 50:
            stocks[sym] = en
        if i % 40 == 0:
            print(f"  prepared {i}/{len(cache)}", flush=True)
    idx = cache.get("_INDEX", pd.DataFrame())
    if not idx.empty:
        idx = add_indicators(idx)
        idx["idx_day"] = idx["close"] / idx.groupby("session")["open"].transform("first") - 1.0
        idx = attach_supertrend(idx)
    return stocks, idx


def _liq(df: pd.DataFrame) -> pd.Series:
    return (df["close"] > 60) & (df["vol_sma"] > 25_000) & (df["atr"] / df["close"] >= 0.002)


def sig_st_flip(df, key="st10_30", target_r=1.5, vwap=True, htf=False, daily_st=False, shorts=False, adx_min=0):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = m & df[f"{key}_flip_up"]
    if vwap:
        long &= df["close"] > df["vwap"]
    if htf:
        long &= df["htf_up"]
    if daily_st:
        long &= df["dst_up"]
    if adx_min:
        long &= df["adx"] >= adx_min
    stop = np.minimum(df[key], df["low"])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    if shorts:
        sh = m & df[f"{key}_flip_dn"]
        if vwap:
            sh &= df["close"] < df["vwap"]
        if daily_st:
            sh &= df["dst_dn"]
        sstop = np.maximum(df[key], df["high"])
        srisk = (sstop - df["close"]).clip(lower=df["atr"] * 0.35)
        frames.append(_base_cols(df, sh, "short", sstop, df["close"] - srisk * target_r, score))
    return pd.concat(frames, ignore_index=True)


def sig_st_pullback(df, key="st10_30", target_r=1.5, vwap=True, htf=False, daily_st=False, shorts=False, adx_min=16):
    m = _can_trade_mask(df, 3) & _liq(df)
    long = (
        m
        & (df[f"{key}_dir"] == 1)
        & ~df[f"{key}_flip_up"]
        & df[f"{key}_tag"]
        & (df["close"] > df[key])
        & (df["close_loc"] >= 0.55)
        & (df["rsi"].between(42, 68))
    )
    if vwap:
        long &= df["close"] > df["vwap"]
    if htf:
        long &= df["htf_up"]
    if daily_st:
        long &= df["dst_up"]
    if adx_min:
        long &= df["adx"] >= adx_min
    stop = np.minimum(df[key] * 0.997, df["low"])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.3)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * (df["adx"].fillna(15) / 20) * (df["rsi"] - 40)
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    if shorts:
        sh = (
            m
            & (df[f"{key}_dir"] == -1)
            & ~df[f"{key}_flip_dn"]
            & df[f"{key}_tag_dn"]
            & (df["close"] < df[key])
            & (df["close_loc"] <= 0.45)
        )
        if vwap:
            sh &= df["close"] < df["vwap"]
        if daily_st:
            sh &= df["dst_dn"]
        sstop = np.maximum(df[key] * 1.003, df["high"])
        srisk = (sstop - df["close"]).clip(lower=df["atr"] * 0.3)
        frames.append(_base_cols(df, sh, "short", sstop, df["close"] - srisk * target_r, score))
    return pd.concat(frames, ignore_index=True)


def sig_st_orb(df, key="st10_30", target_r=1.5, vol_min=1.15, daily_st=True):
    m = _can_trade_mask(df, 1) & _liq(df)
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
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_st_pdh(df, key="st10_30", target_r=1.5, vol_min=1.2):
    m = _can_trade_mask(df, 1) & _liq(df)
    long = (
        m
        & (df[f"{key}_dir"] == 1)
        & df["dst_up"]
        & (df["close"] > df["pdh"])
        & (df["prev_close"] <= df["pdh"])
        & (df["rel_vol"] >= vol_min)
        & (df["close"] > df["vwap"])
    )
    stop = np.minimum(df["low"], df[key])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * ((df["close"] / df["pdh"] - 1).clip(lower=0) * 100 + 1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_nr7_break(df, target_r=1.5, htf=True, daily_st=True):
    m = _can_trade_mask(df, 3) & _liq(df)
    rng = (df["high"] - df["low"]).shift(1)
    nr7 = rng <= rng.rolling(7).min()
    long = m & nr7 & (df["close"] > df["prev_high"]) & (df["rel_vol"] >= 1.15) & (df["close"] > df["vwap"])
    if htf:
        long &= df["htf_up"]
    if daily_st:
        long &= df["dst_up"]
    stop = df["prev_low"]
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.3)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_st_ema_mom(df, key="st10_30", target_r=1.5):
    m = _can_trade_mask(df, 2) & _liq(df)
    long = (
        m
        & (df[f"{key}_dir"] == 1)
        & df["dst_up"]
        & (df["ema9"] > df["ema21"])
        & (df["close"] > df["vwap"])
        & (df["close_loc"] >= 0.65)
        & (df["adx"] >= 18)
        & (df["rel_vol"] >= 1.1)
        & df["rsi"].between(50, 70)
    )
    stop = np.minimum(df["ema21"], df[key])
    risk = (df["close"] - stop).clip(lower=df["atr"] * 0.35)
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def family_specs() -> list[tuple[str, object, dict]]:
    specs: list[tuple[str, object, dict]] = []
    for r in (1.2, 1.5, 2.0):
        specs.append((f"ST10/3 flip {r}R VWAP", sig_st_flip, {"key": "st10_30", "target_r": r, "vwap": True, "daily_st": True}))
        specs.append((f"ST10/3 flip both {r}R", sig_st_flip, {"key": "st10_30", "target_r": r, "vwap": True, "shorts": True, "daily_st": True}))
        specs.append((f"ST10/3 pull {r}R VWAP dST", sig_st_pullback, {"key": "st10_30", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 16}))
        specs.append((f"ST7/3 pull {r}R VWAP dST", sig_st_pullback, {"key": "st7_30", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 16}))
        specs.append((f"ST10/2 pull {r}R VWAP dST", sig_st_pullback, {"key": "st10_20", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 18}))
        specs.append((f"ST14/3 pull {r}R VWAP dST", sig_st_pullback, {"key": "st14_30", "target_r": r, "vwap": True, "daily_st": True, "adx_min": 16}))
        specs.append((f"ST+ORB15 {r}R", sig_st_orb, {"key": "st10_30", "target_r": r, "daily_st": True}))
        specs.append((f"ST+PDH {r}R", sig_st_pdh, {"key": "st10_30", "target_r": r}))
        specs.append((f"ST+EMA mom {r}R", sig_st_ema_mom, {"key": "st10_30", "target_r": r}))
        specs.append((f"NR7 {r}R dST", sig_nr7_break, {"target_r": r, "daily_st": True}))
        specs.append((f"ORB15 {r}R HTF", sig_orb15, {"target_r": r, "vol_min": 1.2, "htf": True}))
        specs.append((f"PDH {r}R HTF", sig_pdh_break, {"target_r": r, "vol_min": 1.2, "htf": True}))
        specs.append((f"HOD {r}R HTF", sig_hod_break, {"target_r": r, "htf": True}))
        specs.append((f"GapGo {r}R", sig_gap_and_go, {"target_r": r, "htf": True}))
        specs.append((f"VWAP reclaim {r}R", sig_vwap_reclaim, {"target_r": r, "htf": True}))
        specs.append((f"EMA mom {r}R", sig_ema_momentum, {"target_r": r, "htf": True}))
    specs.append(("VWAP fade 1.8% ATR", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}))
    specs.append(("VWAP fade 1.5% ATR", sig_vwap_fade_v2, {"ext_min": 0.015, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}))
    specs.append(("VWAP fade 1.8% VWAP", sig_vwap_fade_v2, {"ext_min": 0.018, "adx_max": 26, "target_mode": "vwap", "idx_abs_max": 0.008}))
    specs.append(("ORB fade 0.7%", sig_orb_fade_v2, {"ext": 0.007, "adx_max": 26}))
    specs.append(("ST10/3 pull both 1.5R", sig_st_pullback, {"key": "st10_30", "target_r": 1.5, "vwap": True, "daily_st": True, "shorts": True, "adx_min": 16}))
    return specs


def simulate_sized(
    exec_idx,
    signals,
    name,
    *,
    mode: str,
    risk_pct: float,
    alloc_pct: float,
    leverage: float,
    max_pos: int,
    top_k: int,
    max_deploy: float,
    daily_lock: float = 0.0,
    daily_halt: float = 0.0,
    trail: float = 0.0,
):
    from scripts import intraday_15m_n200_search as eng

    orig = eng._qty

    def qty(equity, risk_pct_, entry, stop, side, max_deploy_pct=max_deploy):
        if entry <= 0:
            return 0
        bp = equity * leverage
        cap = int((bp * max_deploy) / entry) if max_deploy > 0 else 10**9
        if mode == "alloc":
            notional = equity * (alloc_pct / 100.0) * leverage
            q = int(notional / entry)
        else:
            risk_ps = (entry - stop) if side == "long" else (stop - entry)
            if risk_ps <= 0:
                return 0
            q = int((equity * risk_pct_ / 100.0) / risk_ps)
        if cap:
            q = min(q, cap)
        return max(q, 0)

    eng._qty = qty
    try:
        res, tdf = simulate_base(
            exec_idx,
            signals,
            name,
            risk_pct=risk_pct,
            max_pos=max_pos,
            top_k=top_k,
            daily_lock=daily_lock,
            daily_halt=daily_halt,
            trail_after_r=trail,
        )
    finally:
        eng._qty = orig
    res.params = {
        "mode": mode,
        "risk_pct": risk_pct,
        "alloc_pct": alloc_pct,
        "leverage": leverage,
        "max_pos": max_pos,
        "top_k": top_k,
        "max_deploy": max_deploy,
        "lock": daily_lock,
        "halt": daily_halt,
        "trail": trail,
    }
    return res, tdf


def months_from_trades(tdf: pd.DataFrame) -> list[dict]:
    if tdf is None or tdf.empty:
        return []
    t = tdf.copy()
    t["month"] = pd.to_datetime(t["exit_ts"]).dt.to_period("M").astype(str)
    rows = []
    for m, g in t.groupby("month"):
        wins = float((g["pnl"] > 0).mean() * 100)
        rows.append({
            "month": m,
            "trades": int(len(g)),
            "pnl": round(float(g["pnl"].sum()), 2),
            "wr": round(wins, 1),
        })
    return rows


def size_combos() -> list[dict]:
    combos = []
    # Stated sizing: invest 5% of equity, 5× MIS.
    for max_pos, top_k, lock, halt in (
        (4, 1, 0.0, 0.0),
        (6, 2, 0.0, 0.0),
        (8, 2, 5000.0, 3000.0),
        (6, 1, 4000.0, 2500.0),
    ):
        combos.append(dict(
            mode="alloc", risk_pct=5.0, alloc_pct=ALLOC_PCT, leverage=LEVERAGE,
            max_pos=max_pos, top_k=top_k, max_deploy=0.35, daily_lock=lock, daily_halt=halt,
        ))
    # Aggressive 5% risk hunt (needed if 100% is the goal).
    for max_pos, top_k, deploy in ((3, 1, 0.4), (4, 2, 0.5), (5, 2, 0.6)):
        combos.append(dict(
            mode="risk", risk_pct=5.0, alloc_pct=0.0, leverage=LEVERAGE,
            max_pos=max_pos, top_k=top_k, max_deploy=deploy, daily_lock=0.0, daily_halt=0.0,
        ))
    return combos


def rank_key(res) -> tuple:
    months_pos = 0
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
    print(f"Nifty 200 15m hunt | ₹{CAPITAL:,.0f} | 5%/trade | {len(symbols)} names")
    if not args.skip_download:
        n = download_15m_universe(symbols)
        print(f"Saved {n} 15m frames")

    cache = load_frames(symbols)
    n_stocks = sum(1 for k in cache if not k.startswith("_"))
    print(f"Loaded {n_stocks} stock frames")
    if n_stocks < 30:
        print("Not enough 15m data.")
        return

    sample = next(df for k, df in cache.items() if not k.startswith("_"))
    print(f"Sample window {sample.index.min()} → {sample.index.max()}  bars={len(sample)}")

    print("Indicators + Supertrend…")
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
    print(f"\nSimulating {total} configs…")
    book = []
    n = 0
    for name, fn, kw in specs:
        sigs = sig_cache[name]
        if sigs is None or sigs.empty:
            continue
        for sz in combos:
            n += 1
            label = (
                f"{name} | {sz['mode']} r{sz['risk_pct']:g} a{sz['alloc_pct']:g} "
                f"p{sz['max_pos']} k{sz['top_k']}"
            )
            res, tdf = simulate_sized(exec_idx, sigs, label, **sz)
            res.params.update(kw)
            res.params["family"] = name
            book.append((res, tdf, months_from_trades(tdf)))
            if n % 20 == 0 or n == total:
                print(f"  {n}/{total}  last {res.total_return_pct:6.1f}%  {res.name[:50]}", flush=True)

    ranked = sorted(book, key=lambda x: rank_key(x[0]), reverse=True)
    print("\n===== TOP 20 (OOS-positive preferred) =====")
    hdr = f"{'name':<54} {'n':>4} {'WR':>5} {'PF':>5} {'ret%':>7} {'PnL':>9} {'OOS':>8} {'DD':>5}"
    print(hdr)
    for res, tdf, months in ranked[:20]:
        print(
            f"{res.name[:54]:<54} {res.trades:4d} {res.win_rate:5.1f} {res.profit_factor:5.2f} "
            f"{res.total_return_pct:7.1f} {res.net_pnl:9.0f} {res.oos_pnl:8.0f} {res.max_dd_pct:5.1f}"
        )

    hits = [x for x in ranked if x[0].total_return_pct >= 100 and x[0].oos_pnl > 0 and x[0].trades >= 20]
    winner_pack = hits[0] if hits else ranked[0]
    wres, wtdf, wmonths = winner_pack

    payload = {
        "capital": CAPITAL,
        "alloc_pct": ALLOC_PCT,
        "leverage": LEVERAGE,
        "universe": "nifty200",
        "timeframe": "15m",
        "window": {
            "start": str(sample.index.min()),
            "end": str(sample.index.max()),
            "stocks": n_stocks,
        },
        "target_note": "100% in 2 months with 5%/trade is the search target, not a guarantee.",
        "hit_100": bool(hits),
        "winner": {**asdict(wres), "months": wmonths},
        "top": [
            {**asdict(r), "months": m}
            for r, _, m in ranked[:15]
        ],
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
