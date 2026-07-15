"""
EMA20 + strong/engulf enhancement search — 3-year Nifty 200 backtest.
Tests relaxed vs tightened filters to increase trade count while keeping WR > 50%.
"""
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

import pandas as pd

from trading.models import DailyPrice, Stock, StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history, _rsi
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size
from django.db.models import Min, Max

CAPITAL = 500_000.0
COOLDOWN = 8
MAX_HOLD = 45
END = date(2025, 6, 1)
START = END - timedelta(days=3 * 365)


@dataclass
class BtResult:
    name: str
    trades: int
    signals: int
    win_rate: float
    profit_factor: float
    return_pct: float
    avg_rr: float
    exits: dict
    wins: int
    losses: int


def _hl(df: pd.DataFrame, n: int) -> bool:
    if len(df) < n:
        return False
    lows = [float(df.iloc[j]["low"]) for j in range(-n, 0)]
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def _base_stop(hist, e20, e50, atr, mode="tight"):
    e20f = float(e20)
    a = float(atr)
    if mode == "wide":
        return min(float(hist["low"].iloc[-5:].min()), e20f - a * 1.5)
    return min(float(hist["low"].iloc[-5:].min()), e20f - a)


def _pack(hist, config, capital, entry, stop):
    risk = entry - stop
    if risk <= 0 or risk / entry > 0.05:
        return None
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}


def make_ema20_strong(
    *,
    adx_min=22,
    rsi_lo=48,
    rsi_hi=58,
    ema_tol=0.012,
    hl_bars=4,
    require_di=True,
    candle_mode="strong_engulf",  # strong_engulf | any_bullish | engulf_only
    stop_mode="tight",
    require_e50_stack=False,
):
    """Configurable EMA20 + strong/engulf variant."""

    def signal_fn(hist, config, capital):
        row = hist.iloc[-1]
        c = float(row["close"])

        if candle_mode == "strong_engulf":
            if not (bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))):
                return None
        elif candle_mode == "engulf_only":
            if not bool(row.get("bullish_engulfing")):
                return None
        elif candle_mode == "any_bullish":
            if not (
                bool(row.get("strong_close"))
                or bool(row.get("bullish_engulfing"))
                or bool(row.get("hammer"))
            ):
                return None

        e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
        if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
            return None

        if c <= float(e200):
            return None
        if require_e50_stack:
            if not (float(e20) > float(e50) > float(e200)):
                return None
        elif float(e20) <= float(e50):
            return None

        if float(row["adx_14"]) < adx_min:
            return None
        if abs(c - float(e20)) / float(e20) > ema_tol:
            return None
        if not _hl(hist, hl_bars):
            return None
        if not (rsi_lo <= float(row["rsi_14"]) <= rsi_hi):
            return None

        if require_di:
            di_p, di_m = row.get("di_plus"), row.get("di_minus")
            if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
                return None

        stop = _base_stop(hist, e20, e50, row["atr_14"], stop_mode)
        return _pack(hist, config, capital, c, stop)

    return signal_fn


def build_cache(symbols):
    cache = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        df["rsi_2"] = _rsi(df["close"], 2)
        if has_sufficient_history(df):
            cache[sym] = df
    return cache


def backtest(name, signal_fn, cache, config, start=START, end=END):
    trades = []
    signals = 0
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
                elif hold >= MAX_HOLD:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    risk = entry - stop
                    trades.append({
                        "win": pnl > 0,
                        "pnl": pnl,
                        "rr": (exit_p - entry) / risk if risk else 0,
                        "r": reason,
                    })
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
        return BtResult(name, 0, signals, 0, 0, 0, 0, {}, 0, 0)

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100
    return BtResult(
        name=name,
        trades=len(trades),
        signals=signals,
        win_rate=round(wr, 2),
        profit_factor=round(gp / gl, 2),
        return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        avg_rr=round(sum(t["rr"] for t in trades) / len(trades), 2),
        exits=dict(Counter(t["r"] for t in trades)),
        wins=len(wins),
        losses=len(losses),
    )


VARIANTS = [
    ("BASE: EMA20 + strong/engulf (original)", make_ema20_strong()),
    ("A: + hammer allowed", make_ema20_strong(candle_mode="any_bullish")),
    ("B: 3-bar HL (was 4)", make_ema20_strong(hl_bars=3)),
    ("C: wider EMA tol 1.5%", make_ema20_strong(ema_tol=0.015)),
    ("D: RSI 45-60", make_ema20_strong(rsi_lo=45, rsi_hi=60)),
    ("E: ADX >= 20", make_ema20_strong(adx_min=20)),
    ("F: no +DI filter", make_ema20_strong(require_di=False)),
    ("G: 3HL + wider tol", make_ema20_strong(hl_bars=3, ema_tol=0.015)),
    ("H: 3HL + RSI 45-60", make_ema20_strong(hl_bars=3, rsi_lo=45, rsi_hi=60)),
    ("I: 3HL + ADX20", make_ema20_strong(hl_bars=3, adx_min=20)),
    ("J: 3HL + no DI", make_ema20_strong(hl_bars=3, require_di=False)),
    ("K: 3HL + hammer", make_ema20_strong(hl_bars=3, candle_mode="any_bullish")),
    ("L: relaxed combo", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20)),
    ("M: relaxed + hammer", make_ema20_strong(
        hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20, candle_mode="any_bullish"
    )),
    ("N: relaxed + no DI", make_ema20_strong(
        hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20, require_di=False
    )),
    ("O: relaxed + no DI + hammer", make_ema20_strong(
        hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20,
        require_di=False, candle_mode="any_bullish",
    )),
    ("P: 2-bar HL + relaxed", make_ema20_strong(
        hl_bars=2, ema_tol=0.018, rsi_lo=44, rsi_hi=62, adx_min=18, require_di=False,
        candle_mode="any_bullish",
    )),
    ("Q: strict WR boost ADX24 RSI50-56", make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56)),
    ("R: strict + 3HL", make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56, hl_bars=3)),
    ("S: EMA stack + strong/engulf", make_ema20_strong(require_e50_stack=True)),
    ("T: wide stop", make_ema20_strong(hl_bars=3, ema_tol=0.015, stop_mode="wide")),
    ("U: best combo v1", make_ema20_strong(
        hl_bars=3, ema_tol=0.015, rsi_lo=46, rsi_hi=58, adx_min=21, require_di=True,
    )),
    ("V: best combo v2", make_ema20_strong(
        hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=58, adx_min=22, require_di=False,
        candle_mode="any_bullish",
    )),
]


def main():
    dr = DailyPrice.objects.aggregate(mn=Min("date"), mx=Max("date"))
    n200 = Stock.objects.filter(is_nifty200=True, is_active=True).count()
    print(f"Nifty 200 stocks: {n200}")
    print(f"DB date range: {dr['mn']} to {dr['mx']}")
    print(f"Backtest window: {START} to {END} ({(END - START).days} days)\n")

    config = StrategyConfig.get_active()
    symbols = get_universe_symbols(nifty200_only=True)
    print(f"Loading {len(symbols)} symbols...")
    cache = build_cache(symbols)
    print(f"Cached {len(cache)} stocks with sufficient history.\n")

    results = []
    for name, fn in VARIANTS:
        r = backtest(name, fn, cache, config)
        results.append(r)
        print(
            f"{name[:42]:42s}  sig={r.signals:4d}  n={r.trades:4d}  "
            f"WR={r.win_rate:5.1f}%  PF={r.profit_factor:5.2f}  ret={r.return_pct:6.2f}%  {r.exits}"
        )

    print("\n" + "=" * 95)
    print("QUALIFIED: WR >= 50%, sorted by trades (most first), then WR, then PF")
    print("=" * 95)
    qualified = [r for r in results if r.win_rate >= 50 and r.trades >= 5]
    qualified.sort(key=lambda x: (-x.trades, -x.win_rate, -x.profit_factor))

    for i, r in enumerate(qualified, 1):
        print(
            f"{i:2d}. {r.name[:50]:50s}  trades={r.trades:4d}  WR={r.win_rate}%  "
            f"PF={r.profit_factor}  ret={r.return_pct}%  exits={r.exits}"
        )

    base = results[0]
    if qualified:
        best = qualified[0]
        print(f"\nBASELINE: {base.trades} trades, WR={base.win_rate}%, PF={base.profit_factor}")
        print(f"BEST (most trades, WR>=50%): {best.name}")
        print(f"  -> {best.trades} trades (+{best.trades - base.trades}), WR={best.win_rate}%, PF={best.profit_factor}, ret={best.return_pct}%")
    else:
        print("\nNo variant met WR >= 50% with min 5 trades.")


if __name__ == "__main__":
    main()