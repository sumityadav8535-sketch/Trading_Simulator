"""
Advanced tournament — market filters, exits, ensembles, momentum filters.
Goal: best risk-adjusted consistency (PF, return, max DD proxy).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

import numpy as np
import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size

START = date(2024, 6, 1)
END = date(2025, 6, 1)
CAPITAL = 500_000.0
PROXIES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


@dataclass
class Result:
    name: str
    trades: int
    signals: int
    wr: float
    pf: float
    ret: float
    max_dd: float
    expectancy: float
    exits: dict


def market_ok(cache, ts, min_n=3):
    n = 0
    for s in PROXIES:
        df = cache.get(s)
        if df is None:
            continue
        h = df.loc[:ts]
        if h.empty:
            continue
        r = h.iloc[-1]
        if r.get("ema_50") and float(r["close"]) > float(r["ema_50"]):
            n += 1
    return n >= min_n


def _bullish(r):
    return bool(r.get("bullish_engulfing")) or bool(r.get("hammer")) or bool(r.get("strong_close"))


def _hl(df, n):
    lows = [float(df.iloc[j]["low"]) for j in range(-n, 0)]
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def sig_elite(hist, config, capital, mkt_cache=None, ts=None, min_mkt=0):
    if min_mkt and mkt_cache and ts and not market_ok(mkt_cache, ts, min_mkt):
        return None
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if abs(c - float(e20)) / float(e20) > 0.012 or not _hl(hist, 4):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    if not (bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))):
        return None
    adx = float(row["adx_14"])
    adx_prev = float(hist.iloc[-2]["adx_14"]) if len(hist) >= 2 and pd.notna(hist.iloc[-2].get("adx_14")) else adx
    if adx < adx_prev:  # ADX must be rising
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
    risk = c - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}


def sig_ema20_v2(hist, config, capital, mkt_cache=None, ts=None, min_mkt=0):
    if min_mkt and mkt_cache and ts and not market_ok(mkt_cache, ts, min_mkt):
        return None
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if abs(c - float(e20)) / float(e20) > 0.012 or not _hl(hist, 4):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
    risk = c - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}


def sig_momentum_pullback(hist, config, capital, mkt_cache=None, ts=None, min_mkt=0):
    """Near 52w high + 20 EMA pullback."""
    if min_mkt and mkt_cache and ts and not market_ok(mkt_cache, ts, min_mkt):
        return None
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    h52 = row.get("high_52w")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), h52]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and c >= float(h52) * 0.92):
        return None
    if float(row["adx_14"]) < 24 or abs(c - float(e20)) / float(e20) > 0.015:
        return None
    if not (45 <= float(row["rsi_14"]) <= 58) or not _bullish(row):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row.get("atr_14", 0) or 0))
    risk = c - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}


def sig_champion_layered(hist, config, capital, mkt_cache=None, ts=None, min_mkt=0):
    for fn in (sig_elite, sig_ema20_v2, sig_momentum_pullback):
        s = fn(hist, config, capital, mkt_cache, ts, min_mkt)
        if s:
            return s
    return None


def backtest(name, signal_fn, cache, config, exit_mode="fixed_2r", cooldown=8, min_mkt=0, max_hold=45):
    trades = []
    signals = 0
    equity = CAPITAL
    curve = [CAPITAL]
    peak = CAPITAL
    max_dd = 0.0

    for sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
        pending = None
        in_pos = False
        entry = stop = target = qty = 0.0
        hold = 0
        be_stop = False
        last_exit = None

        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, high, op = float(row["close"]), float(row["low"]), float(row["high"]), float(row["open"])
            ema20 = float(row["ema_20"]) if pd.notna(row.get("ema_20")) else None

            if in_pos:
                hold += 1
                active_stop = stop
                exit_p = reason = None
                if not be_stop and close >= entry + (entry - stop):  # hit 1R
                    be_stop = True
                    active_stop = max(stop, entry)
                if exit_mode == "trail_1r" and be_stop and ema20 and close < ema20:
                    exit_p, reason = close, "trail"
                elif low <= active_stop:
                    exit_p, reason = active_stop, "sl"
                elif close >= target:
                    exit_p, reason = target, "2r"
                elif hold >= max_hold:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    trades.append({"win": pnl > 0, "pnl": pnl, "rr": (exit_p - entry) / risk if risk else 0})
                    equity += pnl
                    curve.append(equity)
                    peak = max(peak, equity)
                    max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
                    in_pos = False
                    hold = 0
                    be_stop = False
                    last_exit = ts
                continue

            if pending is not None:
                sig = pending
                pending = None
                entry = op
                stop = sig["stop"]
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * 2
                qty = sig["qty"]
                in_pos = True
                hold = 0
                be_stop = False
                if low <= stop:
                    pnl = (stop - entry) * qty
                    trades.append({"win": False, "pnl": pnl, "rr": -1})
                    equity += pnl
                    curve.append(equity)
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl, "rr": 2})
                    equity += pnl
                    curve.append(equity)
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < cooldown:
                continue

            sig = signal_fn(hist, config, equity, cache, ts, min_mkt)
            if sig and i + 1 < len(dates):
                signals += 1
                pending = sig

    if not trades:
        return Result(name, 0, signals, 0, 0, 0, 0, 0, {})

    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    wr = len(wins) / len(trades) * 100
    exp = sum(t["rr"] for t in trades) / len(trades)
    return Result(
        name, len(trades), signals, round(wr, 2), round(gp / gl, 2),
        round((equity - CAPITAL) / CAPITAL * 100, 2), round(max_dd, 2),
        round(exp, 2), {},
    )


def build_cache():
    cache = {}
    for sym in get_universe_symbols(nifty200_only=True):
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        df["high_52w"] = df["high"].rolling(252, min_periods=60).max()
        if has_sufficient_history(df):
            cache[sym] = df
    for s in PROXIES:
        if s not in cache:
            df = load_price_dataframe(s)
            if not df.empty:
                cache[s] = compute_indicators(df)
    return cache


if __name__ == "__main__":
    config = StrategyConfig.get_active()
    cache = build_cache()
    print(f"Stocks: {len(cache)}\n")

    configs = [
        ("Champion layered", sig_champion_layered, "fixed_2r", 0, 8),
        ("Champion + market≥3", sig_champion_layered, "fixed_2r", 3, 8),
        ("Elite + ADX rising", sig_elite, "fixed_2r", 0, 8),
        ("Elite + market≥3", sig_elite, "fixed_2r", 3, 8),
        ("Elite + trail@1R", sig_elite, "trail_1r", 0, 8),
        ("Elite + mkt3 + trail", sig_elite, "trail_1r", 3, 8),
        ("EMA20 v2 + market≥3", sig_ema20_v2, "fixed_2r", 3, 8),
        ("Momentum 52w + mkt3", sig_momentum_pullback, "fixed_2r", 3, 8),
        ("Champion + mkt3 + trail", sig_champion_layered, "trail_1r", 3, 10),
        ("EMA20 v2 only", sig_ema20_v2, "fixed_2r", 0, 8),
    ]

    results = []
    for name, fn, exit_m, mkt, cd in configs:
        r = backtest(name, fn, cache, config, exit_mode=exit_m, min_mkt=mkt, cooldown=cd)
        results.append(r)
        print(f"{name:28s} T={r.trades:3d} WR={r.wr:5.1f}% PF={r.pf:4.2f} ret={r.ret:6.2f}% DD={r.max_dd:5.1f}% exp={r.expectancy}R")

    print("\n=== TOP BY CONSISTENCY (PF>1, ret>0, trades>=5) ===")
    good = [r for r in results if r.trades >= 5 and r.pf >= 1.0 and r.ret > 0]
    good.sort(key=lambda x: (-x.pf, -x.ret, -x.wr, x.max_dd))
    for r in good:
        print(f"  {r.name}: WR={r.wr}% PF={r.pf} ret={r.ret}% DD={r.max_dd}% trades={r.trades}")

    print("\n=== TOP BY WIN RATE (trades>=5) ===")
    wrs = sorted([r for r in results if r.trades >= 5], key=lambda x: -x.wr)[:5]
    for r in wrs:
        print(f"  {r.name}: WR={r.wr}% PF={r.pf} ret={r.ret}% trades={r.trades}")