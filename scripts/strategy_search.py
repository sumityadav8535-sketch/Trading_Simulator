"""Search high win-rate swing strategies (sample then full)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from collections import Counter
from datetime import date, timedelta
import numpy as np
import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history, _rsi
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size


def add_rsi2(df):
    df = df.copy()
    df["rsi_2"] = _rsi(df["close"], 2)
    df["high_52w"] = df["high"].rolling(252, min_periods=60).max()
    return df


def simulate(symbols, start, end, signal_fn, use_trail=False, cooldown=10, capital=500_000):
    config = StrategyConfig.get_active()
    trades = []
    equity = capital
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        df = add_rsi2(df)
        if not has_sufficient_history(df):
            continue
        cache[sym] = df

    for sym, full in cache.items():
        mask = (full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))
        dates = full.index[mask]
        in_pos = False
        entry = stop = target = qty = hold = 0
        last_exit = None

        for ts in dates:
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, high = float(row["close"]), float(row["low"]), float(row["high"])

            if in_pos:
                hold += 1
                exit_p = reason = None
                if low <= stop:
                    exit_p, reason = stop, "sl"
                elif close >= target:
                    exit_p, reason = target, "2r"
                elif hold >= 15:
                    exit_p, reason = close, "time"
                elif use_trail:
                    e20 = row.get("ema_20")
                    if pd.notna(e20) and close < float(e20) and close > entry:
                        exit_p, reason = close, "trail"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    trades.append({"sym": sym, "pnl": pnl, "win": pnl > 0, "rr": (exit_p - entry) / risk if risk else 0, "r": reason})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < cooldown:
                continue
            sig = signal_fn(hist, config, equity)
            if sig:
                in_pos = True
                entry, stop, target, qty, hold = sig["entry"], sig["stop"], sig["target"], sig["qty"], 0

    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "ret": 0, "reasons": {}}
    wins = [t for t in trades if t["win"]]
    wr = len(wins) / len(trades) * 100
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    return {
        "n": len(trades), "wr": wr, "pf": gp / gl,
        "ret": (equity - capital) / capital * 100,
        "reasons": Counter(t["r"] for t in trades),
        "avg_rr": sum(t["rr"] for t in trades) / len(trades),
    }


def sig_rsi2_pullback(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    ema200 = row.get("ema_200")
    ema50 = row.get("ema_50")
    rsi2 = row.get("rsi_2")
    adx = row.get("adx_14")
    atr = row.get("atr_14")
    if pd.isna(ema200) or c <= ema200:
        return None
    if pd.isna(ema50) or float(ema50) <= float(ema200):
        return None
    if pd.isna(rsi2) or float(rsi2) > 12:
        return None
    if pd.isna(adx) or float(adx) < 20:
        return None
    if not bool(row.get("strong_close")):
        return None
    low5 = float(hist["low"].iloc[-5:].min())
    stop = min(low5, c - float(atr or 0) * 1.0)
    risk = c - stop
    if risk <= 0:
        return None
    target = c + risk * 2
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": target, "qty": pos.quantity}


def sig_momentum_ema20(hist, config, capital):
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    ema20, ema50, ema200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    adx, atr = row.get("adx_14"), row.get("atr_14")
    h52 = row.get("high_52w")
    rsi = row.get("rsi_14")
    vol, vol_sma = int(row["volume"]), row.get("vol_sma_20")
    if any(pd.isna(x) for x in [ema20, ema50, ema200, adx, h52, rsi]):
        return None
    if c <= float(ema200) or float(ema20) <= float(ema50):
        return None
    if float(adx) < 22:
        return None
    if c < float(h52) * 0.90:
        return None
    tol = 0.015
    if abs(c - float(ema20)) / float(ema20) > tol:
        return None
    touched = any(float(hist.iloc[i]["low"]) <= float(hist.iloc[i]["ema_20"]) * 1.01 for i in range(-3, 0) if len(hist) + i >= 0)
    if not touched:
        return None
    if not (45 <= float(rsi) <= 62):
        return None
    if pd.isna(vol_sma) or vol < float(vol_sma) * 1.1:
        return None
    if not (bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))):
        return None
    stop = min(low - float(atr) * 0.5, float(ema20) - float(atr) * 0.5)
    risk = c - stop
    if risk <= 0:
        return None
    target = c + risk * 2
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": target, "qty": pos.quantity}


def sig_trend_continuation(hist, config, capital):
    """NR7 breakout in trend — high win rate variant."""
    if len(hist) < 10:
        return None
    row = hist.iloc[-1]
    prev = hist.iloc[-2]
    c, low, high = float(row["close"]), float(row["low"]), float(row["high"])
    ema20, ema50, ema200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    atr = row.get("atr_14")
    if any(pd.isna(x) for x in [ema20, ema50, ema200]):
        return None
    if not (c > float(ema200) and float(ema20) > float(ema50) > float(ema200)):
        return None
    ranges = (hist["high"] - hist["low"]).iloc[-8:-1]
    if len(ranges) < 7:
        return None
    nr7 = float(prev["high"] - prev["low"]) <= ranges.min()
    if not nr7:
        return None
    if c <= float(prev["high"]):
        return None
    if not bool(row.get("strong_close")):
        return None
    stop = float(prev["low"]) - float(atr or 0) * 0.3
    risk = c - stop
    if risk <= 0:
        return None
    target = c + risk * 2
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": target, "qty": pos.quantity}


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    all_syms = get_universe_symbols(nifty200_only=True)
    sample = all_syms[:40]

    strategies = [
        ("rsi2_pullback", sig_rsi2_pullback),
        ("momentum_ema20", sig_momentum_ema20),
        ("nr7_breakout", sig_trend_continuation),
    ]

    print("=== SAMPLE (40 stocks) ===")
    for name, fn in strategies:
        r = simulate(sample, start, end, fn)
        print(f"{name}: {r}")

    print("\n=== FULL (204 stocks) — best candidates ===")
    for name, fn in strategies:
        r = simulate(all_syms, start, end, fn)
        print(f"{name}: n={r['n']} wr={r['wr']:.1f}% pf={r['pf']:.2f} ret={r['ret']:.1f}% reasons={dict(r['reasons'])}")