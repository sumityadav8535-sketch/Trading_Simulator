"""
Swing strategy tournament — same backtest engine, many entry rules.
Ranks by win rate (min trades), then profit factor, then return.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

import numpy as np
import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history, _rsi
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size

START = date(2024, 6, 1)
END = date(2025, 6, 1)
CAPITAL = 500_000.0
COOLDOWN = 8
MAX_HOLD = 45


@dataclass
class TournamentResult:
    name: str
    signals: int
    trades: int
    win_rate: float
    profit_factor: float
    return_pct: float
    avg_rr: float
    exits: dict
    wins: int
    losses: int


def _bullish(row) -> bool:
    return bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))


def _hl(df: pd.DataFrame, n: int) -> bool:
    if len(df) < n:
        return False
    lows = [float(df.iloc[j]["low"]) for j in range(-n, 0)]
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def _base_stop(hist, row, ema20, ema50, atr, mode: str = "standard"):
    e20 = float(ema20) if ema20 else None
    e50 = float(ema50) if ema50 else None
    a = float(atr) if atr else 0
    if mode == "tight":
        return min(float(hist["low"].iloc[-5:].min()), (e20 or 0) - a)
    if mode == "wide":
        return min(float(hist["low"].iloc[-5:].min()), (e20 or 0) - a * 1.5, (e50 or 0) - a if e50 else 1e9)
    return min(float(hist["low"].iloc[-5:].min()), (e20 or 0) - a * 0.5)


def _pack(hist, config, capital, entry, stop) -> Optional[dict]:
    risk = entry - stop
    if risk <= 0 or risk / entry > 0.05:
        return None
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}


# ── Strategy definitions ──────────────────────────────────────────────

def s_ema20_v2(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50)):
        return None
    if float(row["adx_14"]) < 22 or abs(c - float(e20)) / float(e20) > 0.012:
        return None
    if not _hl(hist, 4):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = _base_stop(hist, row, e20, e50, row["atr_14"], "tight")
    return _pack(hist, config, capital, c, stop)


def s_ema20_strong(hist, config, capital):
    """Enhanced elite path: ADX≥20, 1.4% EMA tol, 3-bar HL + strong/engulf."""
    row = hist.iloc[-1]
    if not bool(row.get("strong_close")) and not bool(row.get("bullish_engulfing")):
        return None
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 20):
        return None
    if abs(c - float(e20)) / float(e20) > 0.014 or not _hl(hist, 3):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = _base_stop(hist, row, e20, e50, row["atr_14"], "tight")
    return _pack(hist, config, capital, c, stop)


def s_ema20_hl3(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if abs(c - float(e20)) / float(e20) > 0.015 or not _hl(hist, 3):
        return None
    if not (45 <= float(row["rsi_14"]) <= 60):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    if not _bullish(row):
        return None
    stop = _base_stop(hist, row, e20, e50, row["atr_14"], "tight")
    return _pack(hist, config, capital, c, stop)


def s_ema50_stack(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) > float(e200)):
        return None
    if float(row["adx_14"]) < 25 or abs(c - float(e50)) / float(e50) > 0.015:
        return None
    if not (45 <= float(row["rsi_14"]) <= 58) or not _bullish(row):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = _base_stop(hist, row, e20, e50, row["atr_14"], "standard")
    return _pack(hist, config, capital, c, stop)


def s_rsi2_bounce(hist, config, capital):
    if "rsi_2" not in hist.columns:
        return None
    row = hist.iloc[-1]
    c = float(row["close"])
    e200, e50 = row.get("ema_200"), row.get("ema_50")
    if any(pd.isna(x) for x in [e200, e50, row.get("adx_14"), row.get("rsi_2"), row.get("atr_14")]):
        return None
    if c <= float(e200) or float(e50) <= float(e200):
        return None
    if float(row["adx_14"]) < 20 or float(row["rsi_2"]) > 15:
        return None
    if not bool(row.get("strong_close")):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), c - float(row["atr_14"]) * 1.2)
    return _pack(hist, config, capital, c, stop)


def s_bb_bounce(hist, config, capital):
    row = hist.iloc[-1]
    c, low = float(row["close"]), float(row["low"])
    for col in ["ema_200", "bb_low", "adx_14", "rsi_14", "atr_14"]:
        if pd.isna(row.get(col)):
            return None
    if c <= float(row["ema_200"]) or float(row["adx_14"]) < 20:
        return None
    if low > float(row["bb_low"]) * 1.008:
        return None
    if not (38 <= float(row["rsi_14"]) <= 52) or not _bullish(row):
        return None
    stop = min(low - float(row["atr_14"]) * 0.5, float(row["bb_low"]) - float(row["atr_14"]) * 0.3)
    return _pack(hist, config, capital, c, stop)


def s_nr7_breakout(hist, config, capital):
    if len(hist) < 10:
        return None
    row, prev = hist.iloc[-1], hist.iloc[-2]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    atr = row.get("atr_14")
    if any(pd.isna(x) for x in [e20, e50, e200, atr]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) > float(e200)):
        return None
    ranges = (hist["high"] - hist["low"]).iloc[-8:-1]
    if float(prev["high"] - prev["low"]) > ranges.min():
        return None
    if c <= float(prev["high"]) or not bool(row.get("strong_close")):
        return None
    stop = float(prev["low"]) - float(atr) * 0.3
    return _pack(hist, config, capital, c, stop)


def s_confluence_7(hist, config, capital):
    """Original confluence strategy — score >= 7."""
    from trading.services.strategy import evaluate_stock
    r = evaluate_stock("", eval_date=hist.index[-1].date(), config=config, capital=capital, indicator_df=hist)
    if not r.is_valid or not r.entry_price or not r.stop_loss:
        return None
    return _pack(hist, config, capital, float(r.entry_price), float(r.stop_loss))


def s_confluence_8(hist, config, capital):
    from trading.services.strategy import evaluate_stock
    r = evaluate_stock("", eval_date=hist.index[-1].date(), config=config, capital=capital, indicator_df=hist)
    if r.confluence_score < 8 or not r.is_valid or not r.entry_price or not r.stop_loss:
        return None
    return _pack(hist, config, capital, float(r.entry_price), float(r.stop_loss))


def s_fib_pullback(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    f382, f618 = row.get("fib_382"), row.get("fib_618")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), f382, f618]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if not (min(float(f382), float(f618)) <= c <= max(float(f382), float(f618))):
        return None
    if not (42 <= float(row["rsi_14"]) <= 58) or not _bullish(row):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = _base_stop(hist, row, e20, e50, row.get("atr_14"), "standard")
    return _pack(hist, config, capital, c, stop)


def s_ema20_adx28(hist, config, capital):
    row = hist.iloc[-1]
    if pd.isna(row.get("adx_14")) or float(row["adx_14"]) < 28:
        return None
    return s_ema20_v2(hist, config, capital)


def s_ema20_rsi_narrow(hist, config, capital):
    row = hist.iloc[-1]
    if pd.isna(row.get("rsi_14")) or not (50 <= float(row["rsi_14"]) <= 56):
        return None
    return s_ema20_v2(hist, config, capital)


def s_dual_pullback(hist, config, capital):
    sig = s_ema20_v2(hist, config, capital)
    if sig:
        return sig
    return s_ema50_stack(hist, config, capital)


def s_ema20_no_di(hist, config, capital):
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
    stop = _base_stop(hist, row, e20, e50, row["atr_14"], "tight")
    return _pack(hist, config, capital, c, stop)


def s_pivot_strict(hist, config, capital):
    row = hist.iloc[-1]
    if not (bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))):
        return None
    return s_ema20_adx28(hist, config, capital)


STRATEGIES: list[tuple[str, Callable]] = [
    ("EMA20 v2 bounce", s_ema20_v2),
    ("EMA20 + strong/engulf", s_ema20_strong),
    ("EMA20 + 3-bar HL", s_ema20_hl3),
    ("EMA20 + ADX28", s_ema20_adx28),
    ("EMA20 + RSI 50-56", s_ema20_rsi_narrow),
    ("EMA20 no DI filter", s_ema20_no_di),
    ("Pivot strict (ADX28+engulf)", s_pivot_strict),
    ("50 EMA stacked", s_ema50_stack),
    ("Fib pullback", s_fib_pullback),
    ("RSI(2) uptrend dip", s_rsi2_bounce),
    ("Bollinger bounce", s_bb_bounce),
    ("NR7 trend breakout", s_nr7_breakout),
    ("Confluence score>=7", s_confluence_7),
    ("Confluence score>=8", s_confluence_8),
    ("Dual 20+50 EMA", s_dual_pullback),
]


def backtest_strategy(name: str, signal_fn: Callable, cache: dict, config) -> TournamentResult:
    trades = []
    signals = 0
    equity = CAPITAL

    for sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
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
                elif hold >= MAX_HOLD:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    trades.append({"win": pnl > 0, "pnl": pnl, "rr": (exit_p - entry) / risk if risk else 0, "r": reason})
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
                    trades.append({"win": False, "pnl": pnl, "rr": -1, "r": "sl"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl, "rr": 2, "r": "2r"})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN:
                continue

            sig = signal_fn(hist, config, equity)
            if sig and i + 1 < len(dates):
                signals += 1
                pending = sig

    if not trades:
        return TournamentResult(name, signals, 0, 0, 0, 0, 0, {}, 0, 0)

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100
    return TournamentResult(
        name=name,
        signals=signals,
        trades=len(trades),
        win_rate=round(wr, 2),
        profit_factor=round(gp / gl, 2),
        return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        avg_rr=round(sum(t["rr"] for t in trades) / len(trades), 2),
        exits=dict(Counter(t["r"] for t in trades)),
        wins=len(wins),
        losses=len(losses),
    )


def build_cache(symbols: list[str]) -> dict:
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        df["rsi_2"] = _rsi(df["close"], 2)
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_low"] = mid - 2 * std
        if has_sufficient_history(df):
            cache[sym] = df
    return cache


if __name__ == "__main__":
    config = StrategyConfig.get_active()
    symbols = get_universe_symbols(nifty200_only=True)
    print(f"Loading {len(symbols)} symbols...")
    cache = build_cache(symbols)
    print(f"Cached {len(cache)} stocks. Running {len(STRATEGIES)} strategies...\n")

    results: list[TournamentResult] = []
    for name, fn in STRATEGIES:
        r = backtest_strategy(name, fn, cache, config)
        results.append(r)
        print(f"{name:30s} sig={r.signals:3d} trades={r.trades:3d} WR={r.win_rate:5.1f}% PF={r.profit_factor:5.2f} ret={r.return_pct:6.2f}% {r.exits}")

    print("\n" + "=" * 80)
    print("RANKING — min 5 trades, sorted by win rate then profit factor")
    print("=" * 80)
    qualified = [r for r in results if r.trades >= 5]
    qualified.sort(key=lambda x: (-x.win_rate, -x.profit_factor, -x.return_pct))

    for i, r in enumerate(qualified[:10], 1):
        print(f"{i:2d}. {r.name:30s} WR={r.win_rate}% PF={r.profit_factor} trades={r.trades} ret={r.return_pct}% exits={r.exits}")

    if qualified:
        best = qualified[0]
        profitable = [r for r in qualified if r.return_pct > 0 and r.win_rate >= 40]
        profitable.sort(key=lambda x: (-x.win_rate, -x.profit_factor))
        print("\nBEST WIN RATE (>=5 trades):", best.name, f"WR={best.win_rate}%")
        if profitable:
            bp = profitable[0]
            print("BEST PROFITABLE (WR>=40%):", bp.name, f"WR={bp.win_rate}% PF={bp.profit_factor} ret={bp.return_pct}%")
        else:
            print("No strategy with WR>=40% and positive return (min 5 trades)")