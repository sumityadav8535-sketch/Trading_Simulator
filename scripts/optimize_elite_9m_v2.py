"""Test regime filter, trail exits, and target R for 9-month profitability."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.market_regime import is_market_bullish
from trading.services.nifty50_index import load_nifty50_frame
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS
from scripts.ema20_enhance_backtest import make_ema20_strong

CAPITAL = 100_000.0
END = date.today()
START = END - timedelta(days=274)


def run_bt(signal_fn, cache, nifty_df, *, regime=False, target_r=2.0, trail_be=False, trail_ema=False):
    trades = []
    equity = CAPITAL
    for _sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
        pending = None
        in_pos = False
        entry = stop = target = qty = 0.0
        hold = 0
        last_exit = None
        trail_on = False

        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, op = float(row["close"]), float(row["low"]), float(row["open"])
            ema20 = float(row["ema_20"]) if pd.notna(row.get("ema_20")) else None

            if in_pos:
                hold += 1
                risk = entry - stop
                if trail_be and close >= entry + risk:
                    stop = max(stop, entry)
                if trail_ema and close >= entry + risk:
                    trail_on = True

                exit_p = reason = None
                if low <= stop:
                    exit_p, reason = stop, "sl"
                elif close >= target:
                    exit_p, reason = target, f"{target_r}r"
                elif trail_ema and trail_on and ema20 and close < ema20:
                    exit_p, reason = close, "trail_ema"
                elif hold >= MAX_HOLD_DAYS:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    trades.append({"win": pnl > 0, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    hold = 0
                    trail_on = False
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
                target = entry + risk * target_r
                qty = sig["qty"]
                in_pos = True
                hold = 0
                trail_on = False
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

            if regime:
                proxies = {NIFTY50_SYMBOL: nifty_df}
                if not is_market_bullish(proxies, ts, min_bullish=1):
                    continue

            sig = signal_fn(hist, StrategyConfig.get_active(), equity)
            if sig and i + 1 < len(dates):
                pending = sig

    if not trades:
        return None
    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    return {
        "trades": len(trades),
        "wr": round(len(wins) / len(trades) * 100, 1),
        "ret": round((equity - CAPITAL) / CAPITAL * 100, 2),
        "pf": round(gp / gl, 2),
    }


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            cache[sym] = df
    nifty = load_nifty50_frame()

    base = make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3)
    strict = make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56, hl_bars=4, ema_tol=0.012)
    stack = make_ema20_strong(adx_min=22, hl_bars=4, require_e50_stack=True, candle_mode="engulf_only")

    tests = [
        ("base", base, {}),
        ("base + regime", base, {"regime": True}),
        ("strict + regime", strict, {"regime": True}),
        ("stack engulf + regime", stack, {"regime": True}),
        ("base + regime + trail BE", base, {"regime": True, "trail_be": True}),
        ("base + regime + trail EMA", base, {"regime": True, "trail_ema": True}),
        ("base + regime 1.5R", base, {"regime": True, "target_r": 1.5}),
        ("strict + regime 1.5R", strict, {"regime": True, "target_r": 1.5}),
        ("strict + regime trail EMA", strict, {"regime": True, "trail_ema": True}),
        ("base wide + regime", make_ema20_strong(adx_min=22, hl_bars=4, stop_mode="wide"), {"regime": True}),
    ]

    print(f"9m window {START} to {END}\n")
    print(f"{'Test':<28} {'Tr':>4} {'WR':>6} {'Ret':>8} {'PF':>5}")
    print("-" * 55)
    best = None
    for name, fn, opts in tests:
        r = run_bt(fn, cache, nifty, **opts)
        if not r:
            print(f"{name:<28}    — no trades")
            continue
        print(f"{name:<28} {r['trades']:>4} {r['wr']:>5.1f}% {r['ret']:>7.2f}% {r['pf']:>5.2f}")
        if r["ret"] > 0 and (best is None or r["ret"] > best["ret"]):
            best = {"name": name, **r, **opts}

    if best:
        print(f"\nBEST profitable: {best}")
    else:
        print("\nNo profitable combo found in this search.")


if __name__ == "__main__":
    main()