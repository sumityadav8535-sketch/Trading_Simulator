"""
Nifty 200 · 15-minute intraday strategy search.

Capital: ₹1,00,000
Target:  ₹2,000–5,000 average daily P&L

- Signal on 15m close, enter next bar open (no look-ahead)
- Conservative SL-first if stop and target both print in the same bar
- Slippage + commission
- In-sample / out-of-sample split

Usage:
    python scripts/intraday_15m_n200_search.py
    python scripts/intraday_15m_n200_search.py --skip-download
    python scripts/intraday_15m_n200_search.py --phase 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
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

from trading.constants import NIFTY200_PROXY  # noqa: E402
from trading.models import Stock  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402

DATA_5M = ROOT / "data" / "intraday_5m"
DATA_15M = ROOT / "data" / "intraday_15m"
OUT = ROOT / "data" / "intraday_15m_n200_results.json"

CAPITAL = 100_000.0
SLIPPAGE = 0.0005  # 0.05% / side
COMMISSION = 0.0003  # 0.03% / side
COST = SLIPPAGE + COMMISSION  # applied each side
MARKET_OPEN = dtime(9, 15)
NO_ENTRY_AFTER = dtime(14, 30)
FORCE_EXIT = dtime(15, 15)
TARGET_LO, TARGET_HI = 2_000.0, 5_000.0


@dataclass
class SimResult:
    name: str
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_pnl: float = 0.0
    total_return_pct: float = 0.0
    max_dd_pct: float = 0.0
    avg_daily_pnl: float = 0.0
    median_daily_pnl: float = 0.0
    days_ge_2k: int = 0
    days_ge_5k: int = 0
    trading_days: int = 0
    pct_days_ge_2k: float = 0.0
    worst_day: float = 0.0
    best_day: float = 0.0
    oos_avg_daily: float = 0.0
    oos_pnl: float = 0.0
    oos_wr: float = 0.0
    oos_days_ge_2k: int = 0
    oos_days: int = 0
    avg_hold_bars: float = 0.0
    params: dict = field(default_factory=dict)


def _normalize(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        if ticker and ticker in sub.columns.get_level_values(0):
            sub = sub[ticker]
        elif ticker and ticker in sub.columns.get_level_values(1):
            sub = sub.xs(ticker, axis=1, level=1)
        else:
            sub.columns = sub.columns.get_level_values(0)
    rename = {}
    for col in sub.columns:
        key = str(col).lower().replace(" ", "_")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = key
    sub = sub.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    if "volume" not in keep:
        sub["volume"] = 0.0
        keep.append("volume")
    sub = sub[keep].dropna(subset=["open", "high", "low", "close"])
    if sub.index.tz is None:
        sub.index = sub.index.tz_localize("Asia/Kolkata")
    else:
        sub.index = sub.index.tz_convert("Asia/Kolkata")
    return sub.sort_index()


def resample_15m(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty:
        return df5
    out = df5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = out.dropna(subset=["open", "close"])
    t = pd.Series(out.index.time, index=out.index)
    return out.loc[(t >= MARKET_OPEN) & (t <= FORCE_EXIT)]


def nifty200_symbols() -> list[str]:
    return list(
        Stock.objects.filter(is_active=True, is_nifty200=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )


def download_15m(symbol: str, period: str = "60d") -> pd.DataFrame:
    ticker = yfinance_ticker(symbol)
    raw = yf.download(ticker, interval="15m", period=period, progress=False, auto_adjust=False)
    return _normalize(raw, ticker)


def load_universe(symbols: list[str], force: bool = False, skip_download: bool = False) -> dict[str, pd.DataFrame]:
    DATA_15M.mkdir(parents=True, exist_ok=True)
    cache: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        path = DATA_15M / f"{sym}.pkl"
        frames = []

        p5 = DATA_5M / f"{sym}.pkl"
        if p5.exists():
            try:
                df5 = pd.read_pickle(p5)
                r = resample_15m(df5)
                if not r.empty:
                    frames.append(r)
            except Exception:
                pass

        if path.exists() and not force:
            try:
                d15 = pd.read_pickle(path)
                if not d15.empty:
                    frames.append(d15)
            except Exception:
                pass
        elif not skip_download and not frames:
            print(f"  [{i}/{len(symbols)}] download {sym} 15m...", flush=True)
            try:
                d15 = download_15m(sym)
                if not d15.empty:
                    d15.to_pickle(path)
                    frames.append(d15)
            except Exception as exc:
                print(f"    fail {sym}: {exc}")
            time.sleep(0.12)

        if not frames:
            continue
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        t = pd.Series(df.index.time, index=df.index)
        df = df.loc[(t >= MARKET_OPEN) & (t <= FORCE_EXIT)]
        if len(df) >= 40:
            cache[sym] = df

    idx_path = DATA_15M / "_NIFTY200_INDEX.pkl"
    idx = pd.DataFrame()
    if idx_path.exists() and not force:
        idx = pd.read_pickle(idx_path)
    elif not skip_download:
        print("  [index] ^CNX200 15m...", flush=True)
        try:
            raw = yf.download(NIFTY200_PROXY, interval="15m", period="60d", progress=False, auto_adjust=False)
            idx = _normalize(raw, NIFTY200_PROXY)
            if not idx.empty:
                idx.to_pickle(idx_path)
        except Exception as exc:
            print("    index fail", exc)
    if idx.empty:
        p5 = DATA_5M / "_NIFTY100_INDEX.pkl"
        if p5.exists():
            idx = resample_15m(pd.read_pickle(p5))
    if not idx.empty:
        cache["_INDEX"] = idx
    return cache


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"].astype(float)
    out["ema9"] = c.ewm(span=9, adjust=False).mean()
    out["ema21"] = c.ewm(span=21, adjust=False).mean()
    out["ema50"] = c.ewm(span=50, adjust=False).mean()
    out["vol_sma"] = v.rolling(20).mean()
    out["rel_vol"] = v / out["vol_sma"].replace(0, np.nan)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=out.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr"]
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr"]
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["di_plus"] = plus_di
    out["di_minus"] = minus_di

    out["session"] = out.index.tz_convert("Asia/Kolkata").date
    sess = out["session"]
    typical = (h + l + c) / 3
    out["vwap"] = (typical * v).groupby(sess).cumsum() / v.groupby(sess).cumsum().replace(0, np.nan)

    rng = (h - l).replace(0, np.nan)
    out["close_loc"] = (c - l) / rng
    out["range_pct"] = (h - l) / c
    out["ret1"] = c.pct_change()
    out["prev_close"] = c.shift(1)
    out["bar_of_day"] = out.groupby(sess).cumcount()
    out["sess_high"] = h.groupby(sess).cummax()
    out["sess_low"] = l.groupby(sess).cummin()
    out["sess_open"] = out.groupby(sess)["open"].transform("first")
    out["day_chg"] = c / out["sess_open"] - 1.0

    first = out.groupby(sess).head(1)
    out["orb15_high"] = sess.map(first.set_index("session")["high"])
    out["orb15_low"] = sess.map(first.set_index("session")["low"])
    first2 = out.groupby(sess).head(2)
    out["orb30_high"] = sess.map(first2.groupby("session")["high"].max())
    out["orb30_low"] = sess.map(first2.groupby("session")["low"].min())

    out["prev_high"] = h.shift(1)
    out["prev_low"] = l.shift(1)
    out["ema9_prev"] = out["ema9"].shift(1)
    out["rsi_prev"] = out["rsi"].shift(1)
    return out


def attach_daily_bias(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    daily = load_price_dataframe(symbol)
    if daily.empty or len(daily) < 25:
        df["htf_up"] = True
        df["htf_dn"] = True
        df["pdh"] = np.nan
        df["pdl"] = np.nan
        df["pdc"] = np.nan
        return df
    d = daily.copy()
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    # Yesterday's completed daily bar only (no same-day look-ahead).
    d["htf_up"] = ((d["close"] > d["sma20"]) & (d["sma20"] > d["sma50"].fillna(d["sma20"]))).shift(1)
    d["htf_dn"] = (d["close"] < d["sma20"]).shift(1)
    d["pdh"] = d["high"].shift(1)
    d["pdl"] = d["low"].shift(1)
    d["pdc"] = d["close"].shift(1)
    # Use previous completed daily bar for the session date
    lookup = d[["htf_up", "htf_dn", "pdh", "pdl", "pdc"]].copy()
    lookup.index = pd.to_datetime(lookup.index).date
    sess = df["session"]
    for col in ("htf_up", "htf_dn", "pdh", "pdl", "pdc"):
        df[col] = sess.map(lookup[col])
    df["htf_up"] = df["htf_up"].fillna(False).astype(bool)
    df["htf_dn"] = df["htf_dn"].fillna(False).astype(bool)
    return df


def prepare(cache: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
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
        idx["idx_up"] = idx["close"] > idx["vwap"]
        idx["idx_ret"] = idx["close"] / idx.groupby("session")["open"].transform("first") - 1.0
    return stocks, idx


# ── signal extractors: return DataFrame with ts, side, stop, target, score ──


def _series_take(val, mask: pd.Series):
    if np.isscalar(val):
        return np.full(int(mask.sum()), float(val))
    s = val.reindex(mask.index)
    return s.loc[mask].to_numpy(dtype=float)


def _base_cols(df: pd.DataFrame, mask: pd.Series, side: str, stop, target, score) -> pd.DataFrame:
    m = mask.fillna(False)
    if not bool(m.any()):
        return pd.DataFrame(columns=["ts", "side", "stop", "target", "score"])
    out = pd.DataFrame({
        "ts": df.index[m],
        "side": side,
        "stop": _series_take(stop, m),
        "target": _series_take(target, m),
        "score": _series_take(score, m),
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["stop", "target"])
    if side == "long":
        out = out[out["target"] > out["stop"]]
    else:
        out = out[out["target"] < out["stop"]]
    return out


def _can_trade_mask(df: pd.DataFrame, min_bar: int = 1, latest=NO_ENTRY_AFTER) -> pd.Series:
    t = pd.Series(df.index.time, index=df.index)
    return (df["bar_of_day"] >= min_bar) & (t <= latest) & (t >= MARKET_OPEN)


def sig_orb15(df, target_r=1.5, vol_min=1.2, min_rng=0.002, htf=False, shorts=False):
    m = _can_trade_mask(df, 1)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    long = m & (df["close"] > df["orb15_high"]) & (df["rel_vol"] >= vol_min) & (rng >= min_rng)
    long &= df["prev_close"] <= df["orb15_high"]  # cross this bar
    if htf:
        long &= df["htf_up"]
    stop = df["orb15_low"]
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(0) * (1 + df["day_chg"].clip(lower=0) * 20)
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    if shorts:
        sh = m & (df["close"] < df["orb15_low"]) & (df["rel_vol"] >= vol_min) & (rng >= min_rng)
        sh &= df["prev_close"] >= df["orb15_low"]
        if htf:
            sh &= df["htf_dn"]
        sstop = df["orb15_high"]
        srisk = sstop - df["close"]
        stgt = df["close"] - srisk * target_r
        frames.append(_base_cols(df, sh, "short", sstop, stgt, score))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sig_orb30(df, target_r=1.5, vol_min=1.15, min_rng=0.003, htf=False, shorts=False):
    m = _can_trade_mask(df, 2)
    rng = (df["orb30_high"] - df["orb30_low"]) / df["close"]
    long = m & (df["close"] > df["orb30_high"]) & (df["rel_vol"] >= vol_min) & (rng >= min_rng)
    long &= df["prev_close"] <= df["orb30_high"]
    if htf:
        long &= df["htf_up"]
    stop = df["orb30_low"]
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(0) * df["adx"].fillna(15) / 20
    frames = [_base_cols(df, long, "long", stop, tgt, score)]
    if shorts:
        sh = m & (df["close"] < df["orb30_low"]) & (df["rel_vol"] >= vol_min) & (rng >= min_rng)
        sh &= df["prev_close"] >= df["orb30_low"]
        if htf:
            sh &= df["htf_dn"]
        sstop = df["orb30_high"]
        frames.append(_base_cols(df, sh, "short", sstop, df["close"] - (sstop - df["close"]) * target_r, score))
    return pd.concat(frames, ignore_index=True)


def sig_vwap_bounce(df, target_r=1.5, rsi_lo=40, rsi_hi=62, htf=True):
    m = _can_trade_mask(df, 2)
    touch = (df["low"] <= df["vwap"] * 1.0015) & (df["close"] > df["vwap"])
    long = (
        m & touch & (df["ema9"] > df["ema21"]) & df["rsi"].between(rsi_lo, rsi_hi)
        & (df["close_loc"] >= 0.6) & (df["adx"] >= 16)
    )
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"], df["ema21"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * (df["rsi"] - rsi_lo)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_vwap_reclaim(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2)
    long = m & (df["prev_close"] < df["vwap"]) & (df["close"] > df["vwap"])
    long &= (df["rel_vol"] >= 1.2) & (df["close_loc"] >= 0.65) & (df["ema9"] > df["ema21"])
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"], df["vwap"] * 0.997)
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["range_pct"].fillna(0) * 100
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_ema_pullback(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 3)
    stacked = (df["ema9"] > df["ema21"]) & (df["ema21"] > df["ema50"])
    touch = (df["low"] <= df["ema9"] * 1.002) & (df["close"] > df["ema9"])
    long = m & stacked & touch & (df["close"] > df["vwap"]) & (df["rsi"].between(45, 65))
    long &= df["close_loc"] >= 0.6
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"], df["ema21"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["adx"].fillna(15) * df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_ema_momentum(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2)
    long = (
        m & (df["ema9"] > df["ema21"]) & (df["close"] > df["vwap"])
        & (df["close_loc"] >= 0.7) & (df["di_plus"] > df["di_minus"])
        & (df["adx"] >= 18) & (df["rel_vol"] >= 1.15) & (df["rsi"].between(50, 72))
    )
    if htf:
        long &= df["htf_up"]
    stop = df["ema21"]
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_gap_and_go(df, target_r=1.5, gap_min=0.006, htf=True):
    m = _can_trade_mask(df, 1) & (df["bar_of_day"] <= 4)
    gap = df["sess_open"] / df["pdc"] - 1.0
    long = m & (gap >= gap_min) & (df["close"] > df["sess_open"]) & (df["close_loc"] >= 0.65)
    long &= df["rel_vol"].fillna(1) >= 1.1
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["orb15_low"], df["low"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = gap * 100 * df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_pdh_break(df, target_r=1.5, vol_min=1.2, htf=True):
    m = _can_trade_mask(df, 1)
    long = m & (df["close"] > df["pdh"]) & (df["prev_close"] <= df["pdh"]) & (df["rel_vol"] >= vol_min)
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"], df["vwap"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * ((df["close"] / df["pdh"] - 1).clip(lower=0) * 100 + 1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_inside_break(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2)
    inside = (df["prev_high"] < df["high"].shift(2)) & (df["prev_low"] > df["low"].shift(2))
    # break of prior inside bar (the prev bar)
    long = m & inside.shift(1).fillna(False) & (df["close"] > df["prev_high"]) & (df["rel_vol"] >= 1.1)
    if htf:
        long &= df["htf_up"]
    stop = df["prev_low"]
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_rsi_reclaim(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2)
    long = (
        m & (df["rsi_prev"] < 38) & (df["rsi"] > 45) & (df["close"] > df["vwap"])
        & (df["ema21"] > df["ema50"]) & (df["close_loc"] >= 0.6)
    )
    if htf:
        long &= df["htf_up"]
    stop = df["low"].rolling(4).min()
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = (df["rsi"] - df["rsi_prev"]).clip(lower=0) * df["rel_vol"].fillna(1)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_hod_break(df, target_r=1.5, htf=True):
    """Break of running session high after 10:00 with volume."""
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 3) & (t >= dtime(10, 0))
    prior_hod = df["high"].groupby(df["session"]).cummax().shift(1)
    long = m & (df["close"] > prior_hod) & (df["rel_vol"] >= 1.25) & (df["close"] > df["vwap"])
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"], df["ema21"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def sig_combo(df, target_r=1.5, htf=True):
    m = _can_trade_mask(df, 2)
    long = (
        m & (df["close"] > df["vwap"]) & (df["ema9"] > df["ema21"])
        & (df["rsi_prev"] < 48) & df["rsi"].between(45, 62)
        & (df["close_loc"] >= 0.65) & (df["adx"] >= 18)
        & (df["di_plus"] > df["di_minus"]) & (df["rel_vol"] >= 1.1)
    )
    if htf:
        long &= df["htf_up"]
    stop = np.minimum(df["low"].rolling(3).min(), df["ema21"])
    risk = df["close"] - stop
    tgt = df["close"] + risk * target_r
    score = df["rel_vol"].fillna(1) * df["adx"].fillna(15)
    return _base_cols(df, long, "long", stop, tgt, score)


def collect_signals(stocks: dict[str, pd.DataFrame], fn, **kwargs) -> pd.DataFrame:
    rows = []
    for sym, df in stocks.items():
        try:
            sig = fn(df, **kwargs)
        except Exception:
            continue
        if sig is None or sig.empty:
            continue
        sig = sig.copy()
        sig["symbol"] = sym
        rows.append(sig)
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "side", "stop", "target", "score"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["ts", "score"], ascending=[True, False])
    return out


def _qty(equity, risk_pct, entry, stop, side, max_deploy=0.28):
    risk_ps = (entry - stop) if side == "long" else (stop - entry)
    if risk_ps <= 0 or entry <= 0:
        return 0
    qty = int((equity * risk_pct / 100.0) / risk_ps)
    cap = int((equity * max_deploy) / entry)
    return max(min(qty, cap), 0)


def build_exec_index(stocks: dict[str, pd.DataFrame]) -> dict:
    """Precompute next-bar entries and OHLC lookups once."""
    next_ts: dict[tuple, pd.Timestamp] = {}
    ohlc: dict[tuple, tuple] = {}  # (sym, ts) -> (open, high, low, close)
    last_bar: dict[str, tuple] = {}
    all_ts = set()
    for sym, df in stocks.items():
        idx = df.index
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        last_bar[sym] = (idx[-1], float(c[-1]))
        for i, ts in enumerate(idx):
            all_ts.add(ts)
            ohlc[(sym, ts)] = (o[i], h[i], l[i], c[i])
            if i + 1 < len(idx) and idx[i + 1].date() == ts.date():
                next_ts[(sym, ts)] = idx[i + 1]
    return {
        "next_ts": next_ts,
        "ohlc": ohlc,
        "last_bar": last_bar,
        "calendar": sorted(all_ts),
    }


def simulate(
    exec_idx: dict,
    signals: pd.DataFrame,
    name: str,
    risk_pct: float = 2.0,
    max_pos: int = 6,
    top_k: int = 2,
    daily_lock: float = 0.0,
    daily_halt: float = 0.0,
    max_per_sym_day: int = 1,
    trail_after_r: float = 0.0,
) -> tuple[SimResult, pd.DataFrame]:
    if signals.empty:
        return SimResult(name=name), pd.DataFrame()

    next_ts = exec_idx["next_ts"]
    ohlc = exec_idx["ohlc"]
    last_bar = exec_idx["last_bar"]
    calendar = exec_idx["calendar"]

    sigs = signals.copy()
    sigs["entry_ts"] = [next_ts.get((r.symbol, r.ts)) for r in sigs.itertuples(index=False)]
    sigs = sigs.dropna(subset=["entry_ts"])
    if sigs.empty:
        return SimResult(name=name), pd.DataFrame()

    sigs["rank"] = sigs.groupby("ts")["score"].rank(method="first", ascending=False)
    sigs = sigs[sigs["rank"] <= top_k]

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    cash_trades = []
    open_pos: list[dict] = []
    daily_pnl: dict = {}
    per_sym: dict = {}

    sig_by_entry: dict[pd.Timestamp, list] = {}
    for rec in sigs.itertuples(index=False):
        sig_by_entry.setdefault(rec.entry_ts, []).append(rec)

    def close_pos(pos, ts, raw, reason):
        nonlocal equity, peak, max_dd
        exit_p = raw * (1 - COST) if pos["side"] == "long" else raw * (1 + COST)
        if pos["side"] == "long":
            pnl = (exit_p - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - exit_p) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        d = ts.date() if hasattr(ts, "date") else ts
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
        cash_trades.append({
            "symbol": pos["sym"],
            "side": pos["side"],
            "signal_ts": str(pos["sig_ts"]),
            "entry_ts": str(pos["entry_ts"]),
            "exit_ts": str(ts),
            "entry": round(pos["entry"], 2),
            "exit": round(exit_p, 2),
            "qty": pos["qty"],
            "pnl": round(pnl, 2),
            "reason": reason,
        })

    for ts in calendar:
        d = ts.date()
        still = []
        for pos in open_pos:
            bar = ohlc.get((pos["sym"], ts))
            if bar is None:
                still.append(pos)
                continue
            _o, high, low, close = bar
            stop, tgt = pos["stop"], pos["target"]
            if trail_after_r > 0:
                r = abs(pos["entry"] - pos["orig_stop"])
                if pos["side"] == "long" and high >= pos["entry"] + r * trail_after_r:
                    stop = max(stop, pos["entry"])
                    pos["stop"] = stop
                elif pos["side"] == "short" and low <= pos["entry"] - r * trail_after_r:
                    stop = min(stop, pos["entry"])
                    pos["stop"] = stop

            if pos["side"] == "long":
                hit_sl, hit_tp = low <= stop, high >= tgt
            else:
                hit_sl, hit_tp = high >= stop, low <= tgt

            if hit_sl:
                close_pos(pos, ts, stop, "sl")
            elif hit_tp:
                close_pos(pos, ts, tgt, "target")
            elif ts.time() >= FORCE_EXIT:
                close_pos(pos, ts, close, "eod")
            else:
                still.append(pos)
        open_pos = still
        held = {p["sym"] for p in open_pos}

        dpnl = daily_pnl.get(d, 0.0)
        if daily_lock > 0 and dpnl >= daily_lock:
            continue
        if daily_halt > 0 and dpnl <= -daily_halt:
            continue
        if ts.time() > NO_ENTRY_AFTER:
            continue

        recs = sig_by_entry.get(ts)
        if not recs:
            continue
        for rec in recs:
            if len(open_pos) >= max_pos:
                break
            if rec.symbol in held:
                continue
            key = (rec.symbol, d)
            if per_sym.get(key, 0) >= max_per_sym_day:
                continue
            bar = ohlc.get((rec.symbol, ts))
            if bar is None:
                continue
            raw_open = bar[0]
            side = rec.side
            entry = raw_open * (1 + COST) if side == "long" else raw_open * (1 - COST)
            stop = float(rec.stop)
            target = float(rec.target)
            if side == "long" and (entry <= stop or target <= entry):
                continue
            if side == "short" and (entry >= stop or target >= entry):
                continue
            qty = _qty(equity, risk_pct, entry, stop, side)
            if qty <= 0:
                continue
            open_pos.append({
                "sym": rec.symbol,
                "side": side,
                "entry": entry,
                "stop": stop,
                "orig_stop": stop,
                "target": target,
                "qty": qty,
                "entry_ts": ts,
                "sig_ts": rec.ts,
            })
            held.add(rec.symbol)
            per_sym[key] = per_sym.get(key, 0) + 1

    for pos in open_pos:
        ts_last, last_c = last_bar[pos["sym"]]
        close_pos(pos, ts_last, last_c, "final")

    tdf = pd.DataFrame(cash_trades)
    res = _summarize(name, tdf, equity, max_dd)
    return res, tdf


def _summarize(name: str, tdf: pd.DataFrame, equity: float, max_dd: float) -> SimResult:
    res = SimResult(name=name, net_pnl=round(equity - CAPITAL, 2),
                    total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
                    max_dd_pct=round(max_dd, 2))
    if tdf is None or tdf.empty:
        return res
    res.trades = len(tdf)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    gp = float(wins["pnl"].sum()) if len(wins) else 0.0
    gl = float(abs(losses["pnl"].sum())) if len(losses) else 0.0
    res.win_rate = round(len(wins) / len(tdf) * 100, 2)
    res.profit_factor = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)
    tdf = tdf.copy()
    tdf["day"] = pd.to_datetime(tdf["exit_ts"]).dt.date
    daily = tdf.groupby("day")["pnl"].sum()
    res.trading_days = int(daily.shape[0])
    res.avg_daily_pnl = round(float(daily.mean()), 2) if len(daily) else 0.0
    res.median_daily_pnl = round(float(daily.median()), 2) if len(daily) else 0.0
    res.days_ge_2k = int((daily >= TARGET_LO).sum())
    res.days_ge_5k = int((daily >= TARGET_HI).sum())
    res.pct_days_ge_2k = round(res.days_ge_2k / res.trading_days * 100, 1) if res.trading_days else 0.0
    res.worst_day = round(float(daily.min()), 2)
    res.best_day = round(float(daily.max()), 2)

    days = sorted(daily.index)
    if len(days) >= 8:
        cut = days[int(len(days) * 0.7)]
        oos = daily[daily.index >= cut]
        oos_tr = tdf[tdf["day"] >= cut]
        res.oos_days = int(oos.shape[0])
        res.oos_avg_daily = round(float(oos.mean()), 2) if len(oos) else 0.0
        res.oos_pnl = round(float(oos.sum()), 2) if len(oos) else 0.0
        if len(oos_tr):
            res.oos_wr = round(float((oos_tr["pnl"] > 0).mean() * 100), 2)
            res.oos_days_ge_2k = int((oos >= TARGET_LO).sum())
    return res


def family_specs(phase: int) -> list[tuple[str, object, dict]]:
    """phase 1 = broad scan; phase 2 = tighter grids on promising families."""
    specs = []
    if phase == 1:
        for r in (1.25, 1.5, 2.0):
            specs.append((f"ORB15 {r}R", sig_orb15, {"target_r": r, "vol_min": 1.2, "htf": True}))
            specs.append((f"ORB15 {r}R noHTF", sig_orb15, {"target_r": r, "vol_min": 1.15, "htf": False}))
            specs.append((f"ORB15 {r}R both", sig_orb15, {"target_r": r, "vol_min": 1.15, "htf": True, "shorts": True}))
            specs.append((f"ORB30 {r}R", sig_orb30, {"target_r": r, "htf": True}))
            specs.append((f"VWAP bounce {r}R", sig_vwap_bounce, {"target_r": r, "htf": True}))
            specs.append((f"VWAP reclaim {r}R", sig_vwap_reclaim, {"target_r": r, "htf": True}))
            specs.append((f"EMA pull {r}R", sig_ema_pullback, {"target_r": r, "htf": True}))
            specs.append((f"EMA mom {r}R", sig_ema_momentum, {"target_r": r, "htf": True}))
            specs.append((f"GapGo {r}R", sig_gap_and_go, {"target_r": r, "htf": True}))
            specs.append((f"PDH {r}R", sig_pdh_break, {"target_r": r, "htf": True}))
            specs.append((f"HOD {r}R", sig_hod_break, {"target_r": r, "htf": True}))
            specs.append((f"Combo {r}R", sig_combo, {"target_r": r, "htf": True}))
            specs.append((f"RSI reclaim {r}R", sig_rsi_reclaim, {"target_r": r, "htf": True}))
            specs.append((f"Inside {r}R", sig_inside_break, {"target_r": r, "htf": True}))
    else:
        # refined: more volume / HTF / short variants
        for r in (1.2, 1.4, 1.6, 1.8):
            for vol in (1.05, 1.15, 1.3, 1.5):
                specs.append((f"ORB15 r{r} v{vol} HTF", sig_orb15, {"target_r": r, "vol_min": vol, "htf": True}))
                specs.append((f"ORB15 r{r} v{vol} 2way", sig_orb15, {"target_r": r, "vol_min": vol, "htf": True, "shorts": True}))
            specs.append((f"PDH r{r} HTF", sig_pdh_break, {"target_r": r, "vol_min": 1.1, "htf": True}))
            specs.append((f"GapGo r{r}", sig_gap_and_go, {"target_r": r, "gap_min": 0.005, "htf": True}))
            specs.append((f"VWAP rec r{r}", sig_vwap_reclaim, {"target_r": r, "htf": True}))
            specs.append((f"HOD r{r}", sig_hod_break, {"target_r": r, "htf": True}))
            specs.append((f"Combo r{r}", sig_combo, {"target_r": r, "htf": True}))
    return specs


def run_phase(stocks, phase: int) -> list[tuple[SimResult, dict]]:
    book = []
    specs = family_specs(phase)
    print(f"\n=== Phase {phase}: {len(specs)} signal families ===")
    print("Building execution index...")
    exec_idx = build_exec_index(stocks)
    print(f"  calendar bars={len(exec_idx['calendar'])}  ohlc keys={len(exec_idx['ohlc'])}")

    sig_cache: dict[str, pd.DataFrame] = {}
    for name, fn, kw in specs:
        sig_cache[name] = collect_signals(stocks, fn, **kw)
        print(f"  signals {name:28s}  {len(sig_cache[name]):5d}", flush=True)

    if phase == 1:
        combos = [
            (2.0, 6, 2, 3000.0, 2500.0, 0.0),
            (1.5, 4, 1, 0.0, 0.0, 0.0),
            (2.5, 6, 2, 4000.0, 2500.0, 1.0),
        ]
    else:
        combos = []
        for risk in (1.5, 2.0, 2.5, 3.0):
            for pos in (4, 6, 8):
                for topk in (1, 2, 3):
                    for lock in (0.0, 2500.0, 4000.0):
                        for halt in (0.0, 2500.0):
                            combos.append((risk, pos, topk, lock, halt, 0.0))
        combos.append((2.0, 6, 2, 3000.0, 2500.0, 1.0))

    total = len(specs) * len(combos)
    n = 0
    for name, fn, kw in specs:
        sigs = sig_cache[name]
        if sigs.empty:
            continue
        for risk, pos, topk, lock, halt, trail in combos:
            n += 1
            label = f"{name} | r{risk} p{pos} k{topk} lock{int(lock)} halt{int(halt)}"
            if trail:
                label += " trailBE"
            res, tdf = simulate(
                exec_idx, sigs, label,
                risk_pct=risk, max_pos=pos, top_k=topk,
                daily_lock=lock, daily_halt=halt, trail_after_r=trail,
            )
            res.params = {
                "family": name, "risk": risk, "max_pos": pos, "top_k": topk,
                "daily_lock": lock, "daily_halt": halt, "trail": trail, **kw,
            }
            book.append((res, tdf))
            if n % 15 == 0 or n == total:
                print(f"  simulated {n}/{total}  last {res.name[:48]} avg/d={res.avg_daily_pnl}", flush=True)
    return book


def rank_key(r: SimResult):
    # Prefer OOS daily if present, else IS daily; require some trades
    oos = r.oos_avg_daily if r.oos_days >= 5 else r.avg_daily_pnl * 0.5
    return (
        oos,
        r.avg_daily_pnl,
        r.pct_days_ge_2k,
        r.profit_factor,
        -r.max_dd_pct,
    )


def print_board(book: list[tuple[SimResult, dict]], title: str, n=15):
    rows = sorted(book, key=lambda x: rank_key(x[0]), reverse=True)
    print(f"\n===== {title} (top {n}) =====")
    hdr = f"{'name':<52} {'trd':>4} {'WR':>6} {'PF':>5} {'PnL':>9} {'avg/d':>8} {'OOS/d':>8} {'d>=2k':>6} {'DD':>6} {'worst':>8}"
    print(hdr)
    for res, _ in rows[:n]:
        print(
            f"{res.name[:52]:<52} {res.trades:4d} {res.win_rate:6.1f} {res.profit_factor:5.2f} "
            f"{res.net_pnl:9.0f} {res.avg_daily_pnl:8.0f} {res.oos_avg_daily:8.0f} "
            f"{res.days_ge_2k:3d}/{res.trading_days:<2d} {res.max_dd_pct:6.1f} {res.worst_day:8.0f}"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--limit-symbols", type=int, default=0)
    args = ap.parse_args()

    symbols = nifty200_symbols()
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    print(f"Nifty 200 15m search | capital ₹{CAPITAL:,.0f} | target ₹{TARGET_LO:,.0f}–{TARGET_HI:,.0f}/day")
    print(f"Universe {len(symbols)} symbols")

    cache = load_universe(symbols, force=args.force_download, skip_download=args.skip_download)
    stocks_n = sum(1 for k in cache if not k.startswith("_"))
    print(f"Loaded {stocks_n} stock 15m frames")
    if stocks_n < 20:
        print("Not enough data. Run without --skip-download.")
        return

    print("Computing indicators + daily HTF bias...")
    stocks, _idx = prepare(cache)
    print(f"Prepared {len(stocks)} stocks")
    if stocks:
        sample = next(iter(stocks.values()))
        print(f"Sample range {sample.index.min().date()} → {sample.index.max().date()}  bars={len(sample)}")

    book = run_phase(stocks, args.phase)
    rows = print_board(book, f"Phase {args.phase}")

    hits = [r for r, _ in rows if r.avg_daily_pnl >= TARGET_LO and r.oos_avg_daily >= TARGET_LO * 0.6 and r.trades >= 30]
    near = [r for r, _ in rows if r.avg_daily_pnl >= 1000 and r.oos_avg_daily >= 400 and r.trades >= 20]

    payload = {
        "capital": CAPITAL,
        "target": [TARGET_LO, TARGET_HI],
        "stocks": len(stocks),
        "phase": args.phase,
        "hits": [asdict(r) for r in hits[:10]],
        "near": [asdict(r) for r in near[:15]],
        "top": [asdict(r) for r, _ in rows[:25]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {OUT}")
    print(f"Hits (>=₹2k/day and OOS ok): {len(hits)}")
    print(f"Near (>=₹1k/day): {len(near)}")

    if hits:
        best = hits[0]
        print(f"\nWINNER: {best.name}")
        print(f"  avg/day ₹{best.avg_daily_pnl:,.0f}  OOS ₹{best.oos_avg_daily:,.0f}  WR {best.win_rate}%  PF {best.profit_factor}  DD {best.max_dd_pct}%")
    elif rows:
        best = rows[0][0]
        print(f"\nBest so far (below target): {best.name}")
        print(f"  avg/day ₹{best.avg_daily_pnl:,.0f}  OOS ₹{best.oos_avg_daily:,.0f}  WR {best.win_rate}%  PF {best.profit_factor}")


if __name__ == "__main__":
    main()
