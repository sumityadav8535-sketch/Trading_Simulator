import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from collections import Counter
from datetime import date, timedelta
import numpy as np
import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history, _rsi, _sma
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size


def enrich(df):
    df = df.copy()
    df["rsi_2"] = _rsi(df["close"], 2)
    mid = _sma(df["close"], 20)
    std = df["close"].rolling(20).std()
    df["bb_mid"] = mid
    df["bb_low"] = mid - 2 * std
    df["bb_high"] = mid + 2 * std
    df["ema50_slope"] = df["ema_50"].diff(5)
    return df


def run_bt(symbols, start, end, signal_fn, next_day_entry=True, max_hold=20):
    config = StrategyConfig.get_active()
    capital = 500_000
    equity = capital
    trades = []
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        df = enrich(df)
        if has_sufficient_history(df):
            cache[sym] = df

    for sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))]
        pending = None
        in_pos = False
        entry = stop = target = qty = hold = 0
        last_exit = None

        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]

            if in_pos:
                hold += 1
                low, close = float(row["low"]), float(row["close"])
                exit_p = reason = None
                if low <= stop:
                    exit_p, reason = stop, "sl"
                elif close >= target:
                    exit_p, reason = target, "2r"
                elif hold >= max_hold:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    trades.append({"win": pnl > 0, "pnl": pnl, "rr": (exit_p - entry) / risk if risk else 0, "r": reason})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if pending and not in_pos:
                o = float(row["open"])
                entry = o
                stop, target, qty = pending["stop"], pending["target"], pending["qty"]
                # Recalc target from actual entry
                risk = entry - stop
                if risk <= 0:
                    pending = None
                    continue
                target = entry + risk * 2
                in_pos = True
                hold = 0
                pending = None
                # Check same-day stop/target
                if float(row["low"]) <= stop:
                    pnl = (stop - entry) * qty
                    trades.append({"win": False, "pnl": pnl, "rr": -1, "r": "sl"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                elif float(row["close"]) >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl, "rr": 2, "r": "2r"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < 12:
                continue

            sig = signal_fn(hist, config, equity)
            if not sig:
                continue
            if next_day_entry:
                if i + 1 < len(dates):
                    pending = sig
            else:
                entry, stop, target, qty = sig["entry"], sig["stop"], sig["target"], sig["qty"]
                in_pos = True
                hold = 0

    if not trades:
        return 0, 0, 0, {}
    wr = sum(1 for t in trades if t["win"]) / len(trades) * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 1e-9
    return len(trades), wr, gp / gl, Counter(t["r"] for t in trades)


def sig_bb_bounce(hist, config, capital):
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    for col in ["ema_200", "ema_50", "bb_low", "adx_14", "rsi_14", "atr_14", "ema50_slope"]:
        if pd.isna(row.get(col)):
            return None
    if c <= float(row["ema_200"]):
        return None
    if float(row["ema50_slope"]) <= 0:
        return None
    if float(row["adx_14"]) < 18:
        return None
    if low > float(row["bb_low"]) * 1.005:
        return None
    if not (35 <= float(row["rsi_14"]) <= 52):
        return None
    if not (bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))):
        return None
    stop = min(low - float(row["atr_14"]) * 0.5, float(row["bb_low"]) - float(row["atr_14"]) * 0.3)
    entry = c
    risk = entry - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}


def sig_ema20_bounce_v2(hist, config, capital):
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    if len(hist) < 6:
        return None
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50)):
        return None
    if float(row["adx_14"]) < 22:
        return None
    tol = 0.012
    if abs(c - float(e20)) / float(e20) > tol:
        return None
    # 3 higher lows
    lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
    if not (lows[1] > lows[0] and lows[2] > lows[1] and lows[3] > lows[2]):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
    entry = c
    risk = entry - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}


def sig_combo_elite(hist, config, capital):
    """Strict multi-filter: trend + BB touch + RSI2 oversold bounce."""
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    if any(pd.isna(row.get(x)) for x in ["ema_200", "ema_50", "ema_20", "rsi_2", "rsi_14", "adx_14", "bb_low", "atr_14"]):
        return None
    if not (c > float(row["ema_200"]) and float(row["ema_20"]) > float(row["ema_50"]) > float(row["ema_200"])):
        return None
    if float(row["adx_14"]) < 25:
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m) + 2:
        return None
    # Pullback: touched BB lower or 20 EMA in last 2 bars
    pullback = False
    for j in range(-2, 1):
        r = hist.iloc[j]
        if float(r["low"]) <= float(r["bb_low"]) * 1.01 or float(r["low"]) <= float(r["ema_20"]) * 1.01:
            pullback = True
    if not pullback:
        return None
    if float(row["rsi_2"]) > 20:
        return None
    if float(row["rsi_14"]) < 40 or float(row["rsi_14"]) > 55:
        return None
    if not bool(row.get("strong_close")):
        return None
    vol_sma = row.get("vol_sma_20")
    if pd.isna(vol_sma) or int(row["volume"]) < float(vol_sma):
        return None
    stop = float(hist["low"].iloc[-3:].min()) - float(row["atr_14"]) * 0.5
    entry = c
    risk = entry - stop
    if risk <= 0 or risk / entry > 0.04:
        return None
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    syms = get_universe_symbols(nifty200_only=True)
    for name, fn in [("bb_bounce", sig_bb_bounce), ("ema20_v2", sig_ema20_bounce_v2), ("combo_elite", sig_combo_elite)]:
        for nd in [True, False]:
            n, wr, pf, reasons = run_bt(syms, start, end, fn, next_day_entry=nd)
            print(f"{name} next_day={nd}: n={n} wr={wr:.1f}% pf={pf:.2f} {dict(reasons)}")