import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from collections import Counter
from datetime import date, timedelta
import pandas as pd
import numpy as np

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size

PROXIES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def market_ok(full_cache, ts, min_above=3):
    above = 0
    for sym in PROXIES:
        df = full_cache.get(sym)
        if df is None:
            continue
        h = df.loc[:ts]
        if h.empty:
            continue
        r = h.iloc[-1]
        if r.get("ema_50") and float(r["close"]) > float(r["ema_50"]):
            above += 1
    return above >= min_above


def evaluate_momentum_pivot(hist, config, capital):
    """Momentum Pivot Pro — selective 20 EMA higher-low bounce in strong trend."""
    if len(hist) < 6:
        return None
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    adx, rsi, atr = row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    vol_sma = row.get("vol_sma_20")
    if any(pd.isna(x) for x in [e20, e50, e200, adx, rsi, atr, di_p, di_m]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) > float(e200)):
        return None
    if float(adx) < 23:
        return None
    if float(di_p) <= float(di_m):
        return None
    if abs(c - float(e20)) / float(e20) > 0.012:
        return None
    lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
    if not (lows[1] > lows[0] and lows[2] > lows[1] and lows[3] > lows[2]):
        return None
    if not (48 <= float(rsi) <= 57):
        return None
    if not bool(row.get("strong_close")):
        return None
    if pd.isna(vol_sma) or int(row["volume"]) < float(vol_sma) * 1.05:
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(atr) * 0.75)
    risk = c - stop
    if risk <= 0 or risk / c > 0.035:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {
        "entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity,
        "score": 9, "rr": 2.0,
        "reasons": [
            "EMA 20>50>200 bullish stack",
            f"ADX {float(adx):.1f} with +DI > -DI",
            "4-bar higher-low pivot at 20 EMA",
            f"RSI {float(rsi):.1f} pullback sweet spot",
            "Strong close + volume confirmation",
        ],
    }


def backtest(symbols, start, end, capital=500_000, max_hold=30, no_time_stop=False):
    config = StrategyConfig.get_active()
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            cache[sym] = df

    trades = []
    signals = []
    equity = capital
    pending = {}

    # Build date union for market filter
    all_dates = sorted(set().union(*[
        df.index[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        for df in cache.values()
    ]))

    positions = {}  # sym -> dict
    last_exit = {}

    for ts in all_dates:
        if not market_ok(cache, ts):
            # still manage open positions
            pass

        for sym, full in cache.items():
            if ts not in full.index:
                continue
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, high, op = float(row["close"]), float(row["low"]), float(row["high"]), float(row["open"])

            if sym in positions:
                pos = positions[sym]
                pos["hold"] += 1
                exit_p = reason = None
                if low <= pos["stop"]:
                    exit_p, reason = pos["stop"], "stop_loss"
                elif close >= pos["target"]:
                    exit_p, reason = pos["target"], "target_2r"
                elif not no_time_stop and pos["hold"] >= max_hold:
                    exit_p, reason = close, "time_exit"
                if exit_p is not None:
                    pnl = (exit_p - pos["entry"]) * pos["qty"]
                    risk = pos["entry"] - pos["stop"]
                    trades.append({
                        "symbol": sym, "entry_date": str(pos["entry_date"]), "exit_date": str(ts.date()),
                        "entry_price": round(pos["entry"], 2), "exit_price": round(exit_p, 2),
                        "stop_loss": round(pos["stop"], 2), "target": round(pos["target"], 2),
                        "quantity": pos["qty"], "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl / (pos["entry"] * pos["qty"]) * 100, 2) if pos["qty"] else 0,
                        "rr_achieved": round((exit_p - pos["entry"]) / risk, 2) if risk else 0,
                        "exit_reason": reason,
                    })
                    equity += pnl
                    del positions[sym]
                    last_exit[sym] = ts
                continue

            # Process pending entry (next-day open)
            if sym in pending:
                sig = pending.pop(sym)
                if sym in last_exit and (ts - last_exit[sym]).days < 12:
                    continue
                entry = op
                stop = sig["stop"]
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * 2
                qty = sig["qty"]
                positions[sym] = {
                    "entry": entry, "stop": stop, "target": target, "qty": qty,
                    "entry_date": ts.date(), "hold": 0,
                }
                if low <= stop:
                    pnl = (stop - entry) * qty
                    trades.append({
                        "symbol": sym, "entry_date": str(ts.date()), "exit_date": str(ts.date()),
                        "entry_price": round(entry, 2), "exit_price": round(stop, 2),
                        "stop_loss": round(stop, 2), "target": round(target, 2),
                        "quantity": qty, "pnl": round(pnl, 2), "pnl_pct": round(pnl / (entry * qty) * 100, 2),
                        "rr_achieved": -1.0, "exit_reason": "stop_loss",
                    })
                    equity += pnl
                    del positions[sym]
                    last_exit[sym] = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({
                        "symbol": sym, "entry_date": str(ts.date()), "exit_date": str(ts.date()),
                        "entry_price": round(entry, 2), "exit_price": round(target, 2),
                        "stop_loss": round(stop, 2), "target": round(target, 2),
                        "quantity": qty, "pnl": round(pnl, 2), "pnl_pct": round(pnl / (entry * qty) * 100, 2),
                        "rr_achieved": 2.0, "exit_reason": "target_2r",
                    })
                    equity += pnl
                    del positions[sym]
                    last_exit[sym] = ts
                continue

            if sym in last_exit and (ts - last_exit[sym]).days < 12:
                continue
            if sym in positions or sym in pending:
                continue
            if not market_ok(cache, ts):
                continue

            sig = evaluate_momentum_pivot(hist, config, equity)
            if sig:
                signals.append({
                    "symbol": sym, "date": str(ts.date()),
                    "entry_price": round(sig["entry"], 2), "stop_loss": round(sig["stop"], 2),
                    "target_2r": round(sig["target"], 2), "risk_reward": 2.0,
                    "score": sig["score"], "reasons": sig["reasons"],
                })
                # schedule next bar entry
                idx = full.index.get_loc(ts)
                if idx + 1 < len(full):
                    nxt = full.index[idx + 1]
                    if nxt <= pd.Timestamp(end):
                        pending[sym] = sig

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100 if trades else 0
    return {
        "trades": trades, "signals": signals, "n": len(trades), "signal_count": len(signals),
        "wr": wr, "pf": gp / gl, "ret": (equity - capital) / capital * 100,
        "reasons": Counter(t["exit_reason"] for t in trades),
        "equity": equity,
    }


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    syms = get_universe_symbols(nifty200_only=True)
    for label, kwargs in [
        ("with_time", {}),
        ("no_time", {"no_time_stop": True}),
        ("no_time_hold60", {"no_time_stop": True, "max_hold": 60}),
    ]:
        r = backtest(syms, start, end, **kwargs)
        print(f"{label}: signals={r['signal_count']} trades={r['n']} wr={r['wr']:.1f}% pf={r['pf']:.2f} ret={r['ret']:.1f}% {dict(r['reasons'])}")