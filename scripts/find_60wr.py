"""Find parameter set achieving 60%+ win rate at 2R."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from collections import Counter
from datetime import date, timedelta
import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size

PROXIES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def market_ok(cache, ts, min_above=3):
    above = sum(
        1 for sym in PROXIES
        if (df := cache.get(sym)) is not None
        and not (h := df.loc[:ts]).empty
        and h.iloc[-1].get("ema_50")
        and float(h.iloc[-1]["close"]) > float(h.iloc[-1]["ema_50"])
    )
    return above >= min_above


def make_eval(adx_min=22, rsi_lo=48, rsi_hi=58, ema_stack=False, vol_mult=0.0, market_min=0):
    def evaluate(hist, config, capital):
        if len(hist) < 6:
            return None
        row = hist.iloc[-1]
        c, low = float(row["close"]), float(row["low"])
        e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
        if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
            return None
        if c <= float(e200):
            return None
        if ema_stack:
            if not (float(e20) > float(e50) > float(e200)):
                return None
        elif float(e20) <= float(e50):
            return None
        if float(row["adx_14"]) < adx_min:
            return None
        if abs(c - float(e20)) / float(e20) > 0.012:
            return None
        lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
        if not (lows[1] > lows[0] and lows[2] > lows[1] and lows[3] > lows[2]):
            return None
        if not (rsi_lo <= float(row["rsi_14"]) <= rsi_hi):
            return None
        di_p, di_m = row.get("di_plus"), row.get("di_minus")
        if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
            return None
        if vol_mult and (pd.isna(row.get("vol_sma_20")) or int(row["volume"]) < float(row["vol_sma_20"]) * vol_mult):
            return None
        if not (bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))):
            return None
        stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
        risk = c - stop
        if risk <= 0:
            return None
        pos = calculate_position_size(capital, config.risk_pct, c, stop)
        if pos.quantity <= 0:
            return None
        return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}
    return evaluate


def backtest(evaluate, symbols, start, end, market_min=3, max_hold=25):
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
    equity = 500_000
    for sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))]
        pending = None
        in_pos = False
        entry = stop = target = qty = hold = 0
        last_exit = None

        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, op = float(row["close"]), float(row["low"]), float(row["open"])

            if in_pos:
                hold += 1
                exit_p = reason = None
                if low <= stop:
                    exit_p, reason = stop, "sl"
                elif close >= target:
                    exit_p, reason = target, "2r"
                elif hold >= max_hold:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    trades.append({"win": pnl > 0, "pnl": pnl, "r": reason})
                    equity += pnl
                    in_pos = False
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
                if low <= stop:
                    pnl = (stop - entry) * qty
                    trades.append({"win": False, "pnl": pnl, "r": "sl"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl, "r": "2r"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < 12:
                continue
            if market_min and not market_ok(cache, ts, market_min):
                continue
            sig = evaluate(hist, config, equity)
            if sig and i + 1 < len(dates):
                pending = sig

    if not trades:
        return 0, 0, 0, {}
    wr = sum(1 for t in trades if t["win"]) / len(trades) * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 1e-9
    return len(trades), wr, gp / gl, Counter(t["r"] for t in trades)


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    syms = get_universe_symbols(nifty200_only=True)
    combos = [
        ("v1_base", make_eval(), 0, 25),
        ("v2_mkt3", make_eval(), 3, 25),
        ("v3_adx24", make_eval(adx_min=24), 3, 25),
        ("v4_rsi50", make_eval(adx_min=24, rsi_lo=50, rsi_hi=56), 3, 25),
        ("v5_stack", make_eval(adx_min=24, rsi_lo=50, rsi_hi=56, ema_stack=True), 3, 25),
        ("v6_vol", make_eval(adx_min=24, rsi_lo=50, rsi_hi=56, ema_stack=True, vol_mult=1.05), 3, 40),
        ("v7_nohold", make_eval(adx_min=23, rsi_lo=49, rsi_hi=57), 3, 999),
    ]
    for name, ev, mkt, hold in combos:
        n, wr, pf, r = backtest(ev, syms, start, end, market_min=mkt, max_hold=hold)
        flag = " ***" if wr >= 60 and n >= 8 else ""
        print(f"{name}: n={n} wr={wr:.1f}% pf={pf:.2f} {dict(r)}{flag}")