"""Find EMA20 Elite variants profitable over last 9 months."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS
from scripts.ema20_enhance_backtest import _hl, _base_stop, make_ema20_strong

CAPITAL = 100_000.0
END = date.today()
START_9M = END - timedelta(days=274)
START_2Y = END - timedelta(days=730)


def backtest_variant(name, signal_fn, cache, config, start, end):
    trades = []
    equity = CAPITAL
    for _sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))]
        pending = None
        in_pos = False
        entry = stop = target = qty = 0.0
        hold = 0
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
                elif hold >= MAX_HOLD_DAYS:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    trades.append({"win": pnl > 0, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    hold = 0
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
                    trades.append({"win": False, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue

            sig = signal_fn(hist, config, equity)
            if sig and i + 1 < len(dates):
                pending = sig

    if not trades:
        return None
    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    return {
        "name": name,
        "trades": len(trades),
        "wr": round(len(wins) / len(trades) * 100, 1),
        "ret": round((equity - CAPITAL) / CAPITAL * 100, 2),
        "pf": round(gp / gl, 2),
    }


def build_cache(symbols):
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            cache[sym] = df
    return cache


VARIANTS = [
    ("current", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3)),
    ("ADX24 RSI50-56", make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56, hl_bars=3)),
    ("ADX22 4HL", make_ema20_strong(adx_min=22, hl_bars=4)),
    ("engulf only", make_ema20_strong(adx_min=22, hl_bars=4, candle_mode="engulf_only")),
    ("EMA stack", make_ema20_strong(adx_min=22, hl_bars=4, require_e50_stack=True)),
    ("strict combo", make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56, hl_bars=4, ema_tol=0.012)),
    ("wide stop", make_ema20_strong(adx_min=22, hl_bars=4, stop_mode="wide")),
    ("best v1", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=46, rsi_hi=58, adx_min=21)),
    ("tight RSI 50-56 ADX22", make_ema20_strong(adx_min=22, rsi_lo=50, rsi_hi=56, hl_bars=4, ema_tol=0.012)),
    ("ADX25 4HL", make_ema20_strong(adx_min=25, hl_bars=4, ema_tol=0.012, rsi_lo=50, rsi_hi=58)),
]


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    cache = build_cache(symbols)
    config = StrategyConfig.get_active()
    print(f"Cached {len(cache)} symbols | 9m: {START_9M} to {END}\n")

    rows = []
    for name, fn in VARIANTS:
        r9 = backtest_variant(name, fn, cache, config, START_9M, END)
        r2 = backtest_variant(name, fn, cache, config, START_2Y, END)
        if r9:
            rows.append((r9, r2))

    rows.sort(key=lambda x: (-x[0]["ret"], -x[0]["pf"]))
    print(f"{'Variant':<22} {'9m Tr':>5} {'9m WR':>6} {'9m Ret':>8} {'9m PF':>6} | {'2y Ret':>8}")
    print("-" * 70)
    for r9, r2 in rows:
        r2ret = r2["ret"] if r2 else 0
        print(
            f"{r9['name']:<22} {r9['trades']:>5} {r9['wr']:>5.1f}% {r9['ret']:>7.2f}% {r9['pf']:>6.2f} | {r2ret:>7.2f}%"
        )

    profitable = [x for x in rows if x[0]["ret"] > 0]
    print("\nProfitable on 9m:", len(profitable))
    if profitable:
        best = max(profitable, key=lambda x: (x[0]["ret"], x[0]["pf"]))
        print("BEST:", best[0])


if __name__ == "__main__":
    main()