"""Optimize swing strategy parameters for >60% win rate at 2R."""
import os
import sys
from collections import Counter
from datetime import date, timedelta
from dataclasses import dataclass

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


@dataclass
class Params:
    adx_min: float = 25
    rsi_low: float = 45
    rsi_high: float = 58
    vol_mult: float = 1.2
    ema_tol: float = 1.5
    require_ema_stack: bool = True
    require_di_bullish: bool = True
    min_score: int = 8
    cooldown_days: int = 15
    use_trail: bool = False


def market_bullish(hist_map: dict, ts) -> bool:
    proxies = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    above = 0
    for sym in proxies:
        df = hist_map.get(sym)
        if df is None:
            continue
        h = df.loc[:ts]
        if h.empty:
            continue
        row = h.iloc[-1]
        if row.get("ema_50") and row["close"] > row["ema_50"]:
            above += 1
    return above >= 3


def evaluate_apex(symbol, hist, config, capital, params: Params):
    if len(hist) < 220:
        return None
    row = hist.iloc[-1]
    prev = hist.iloc[-2]
    close = float(row["close"])
    low = float(row["low"])
    ema20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None
    ema50 = float(row["ema_50"]) if pd.notna(row["ema_50"]) else None
    ema200 = float(row["ema_200"]) if pd.notna(row["ema_200"]) else None
    adx = float(row["adx_14"]) if pd.notna(row["adx_14"]) else None
    di_p = float(row["di_plus"]) if pd.notna(row.get("di_plus", np.nan)) else None
    di_m = float(row["di_minus"]) if pd.notna(row.get("di_minus", np.nan)) else None
    rsi = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None
    vol = int(row["volume"])
    vol_sma = float(row["vol_sma_20"]) if pd.notna(row["vol_sma_20"]) else None
    atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else None

    score = 0
    if not ema200 or close <= ema200:
        return None
    score += 2
    if params.require_ema_stack:
        if not (ema20 and ema50 and ema20 > ema50 > ema200):
            return None
        score += 2
    if not adx or adx < params.adx_min:
        return None
    score += 2
    if params.require_di_bullish:
        if not di_p or not di_m or di_p <= di_m:
            return None
        score += 1
    tol = params.ema_tol / 100
    at_ema20 = ema20 and abs(close - ema20) / ema20 <= tol
    touched_ema = False
    for i in range(-4, 0):
        if len(hist) + i < 0:
            continue
        r = hist.iloc[i]
        e20 = float(r["ema_20"]) if pd.notna(r["ema_20"]) else None
        if e20 and float(r["low"]) <= e20 * (1 + tol):
            touched_ema = True
    if not (at_ema20 and touched_ema):
        return None
    score += 2
    bullish = bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))
    if not bullish:
        return None
    score += 2
    if not vol_sma or vol < vol_sma * params.vol_mult:
        return None
    score += 1
    if rsi is None or not (params.rsi_low <= rsi <= params.rsi_high):
        return None
    score += 1
    if score < params.min_score:
        return None

    stop = low - (atr or 0) * config.atr_sl_buffer
    if ema20:
        stop = min(stop, ema20 - (atr or 0) * config.atr_sl_buffer)
    entry = close
    risk = entry - stop
    if risk <= 0:
        return None
    target = entry + risk * 2
    rr = 2.0
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {
        "entry": entry, "stop": stop, "target": target, "qty": pos.quantity, "score": score, "rr": rr,
    }


def backtest_params(params: Params, symbols, start, end, capital=500_000):
    config = StrategyConfig.get_active()
    hist_map = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            hist_map[sym] = df

    market_map = {s: hist_map[s] for s in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"] if s in hist_map}

    trades = []
    equity = capital
    last_exit = {}

    for sym, full_df in hist_map.items():
        mask = (full_df.index >= pd.Timestamp(start)) & (full_df.index <= pd.Timestamp(end))
        dates = full_df.index[mask]
        in_pos = False
        entry = stop = target = qty = 0.0
        entry_dt = None

        for ts in dates:
            hist = full_df.loc[:ts]
            row = hist.iloc[-1]
            close = float(row["close"])
            low = float(row["low"])

            if in_pos:
                exit_p = exit_r = None
                if low <= stop:
                    exit_p, exit_r = stop, "stop_loss"
                elif close >= target:
                    exit_p, exit_r = target, "target_2r"
                elif params.use_trail:
                    ema20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None
                    if ema20 and close < ema20 and close >= entry + (entry - stop):
                        exit_p, exit_r = close, "trail_20ema"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    rr = (exit_p - entry) / risk if risk else 0
                    trades.append({"sym": sym, "pnl": pnl, "rr": rr, "reason": exit_r, "win": pnl > 0})
                    equity += pnl
                    in_pos = False
                    last_exit[sym] = ts
                continue

            if sym in last_exit and (ts - last_exit[sym]).days < params.cooldown_days:
                continue
            if not market_bullish(market_map, ts):
                continue

            sig = evaluate_apex(sym, hist, config, equity, params)
            if sig:
                in_pos = True
                entry, stop, target, qty = sig["entry"], sig["stop"], sig["target"], sig["qty"]
                entry_dt = ts

    if not trades:
        return 0, 0, 0, 0
    wins = sum(1 for t in trades if t["win"])
    wr = wins / len(trades) * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 1e-9
    pf = gp / gl
    ret = (equity - capital) / capital * 100
    return len(trades), wr, pf, ret


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    symbols = get_universe_symbols(nifty200_only=True)

    best = None
    for min_score in [7, 8, 9]:
        for adx in [22, 25, 28]:
            for cooldown in [10, 15, 20]:
                p = Params(adx_min=adx, min_score=min_score, cooldown_days=cooldown, use_trail=False)
                n, wr, pf, ret = backtest_params(p, symbols, start, end)
                if wr >= 60 and n >= 10:
                    print(f"GOOD score={min_score} adx={adx} cd={cooldown}: n={n} wr={wr:.1f}% pf={pf:.2f} ret={ret:.1f}%")
                    if best is None or wr > best[1]:
                        best = (p, wr, n, pf, ret)

    if best:
        print("BEST:", best)
    else:
        p = Params()
        n, wr, pf, ret = backtest_params(p, symbols, start, end)
        print(f"Default apex: n={n} wr={wr:.1f}% pf={pf:.2f} ret={ret:.1f}%")