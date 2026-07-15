"""
Search strategies for >=30% return in 1 year (2% risk, 2R target).
Backtest windows: 9m, 1y, 2y, 3y.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.nifty50_index import load_nifty50_frame
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS
from scripts.ema20_enhance_backtest import build_cache, make_ema20_strong

CAPITAL = 100_000.0
END = date.today()
WINDOWS = {
    "9m": END - timedelta(days=274),
    "1y": END - timedelta(days=365),
    "2y": END - timedelta(days=730),
    "3y": END - timedelta(days=1095),
}


@dataclass
class BtStats:
    trades: int
    win_rate: float
    return_pct: float
    profit_factor: float
    pnl: float


def nifty_regime(nifty, ts, mode: str) -> bool:
    h = nifty.loc[: pd.Timestamp(ts)]
    if h.empty:
        return False
    r = h.iloc[-1]
    c = float(r["close"])
    e50 = float(r["ema_50"]) if pd.notna(r.get("ema_50")) else None
    e200 = float(r["ema_200"]) if pd.notna(r.get("ema_200")) else None
    if mode == "off":
        return True
    if not e50 or c <= e50:
        return False
    if mode == "50+200":
        return bool(e200 and c > e200)
    return True


def backtest(
    cache: dict,
    nifty,
    config,
    signal_fn,
    start: date,
    end: date,
    *,
    regime: str = "off",
    pause_after: int = 0,
    target_r: float = 2.0,
) -> BtStats | None:
    events: list[dict] = []

    for _sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))]
        last_exit = None
        for i, ts in enumerate(dates):
            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue
            if not nifty_regime(nifty, ts, regime):
                continue
            hist = full.loc[:ts]
            sig = signal_fn(hist, config, CAPITAL)
            if not sig or i + 1 >= len(dates):
                continue
            ets = dates[i + 1]
            entry = float(full.loc[ets]["open"])
            stop = sig["stop"]
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + risk * target_r
            qty = sig["qty"]
            exit_p = None
            hold = 0
            xdate = None
            for j in range(i + 1, len(dates)):
                row2 = full.loc[dates[j]]
                c2, low2 = float(row2["close"]), float(row2["low"])
                hold += 1
                if low2 <= stop:
                    exit_p, xdate = stop, dates[j]
                    break
                if c2 >= target:
                    exit_p, xdate = target, dates[j]
                    break
                if hold >= MAX_HOLD_DAYS:
                    exit_p, xdate = c2, dates[j]
                    break
            if exit_p is None:
                continue
            pnl = (exit_p - entry) * qty
            events.append({
                "entry": str(ets.date()),
                "pnl": pnl,
                "win": pnl > 0,
            })
            last_exit = xdate

    if not events:
        return None

    if pause_after:
        ordered = sorted(events, key=lambda x: x["entry"])
        filtered = []
        streak = 0
        for e in ordered:
            if streak >= pause_after:
                continue
            filtered.append(e)
            streak = 0 if e["win"] else streak + 1
        events = filtered

    if not events:
        return None

    wins = [e for e in events if e["win"]]
    gp = sum(e["pnl"] for e in wins)
    gl = abs(sum(e["pnl"] for e in events if not e["win"])) or 1e-9
    total = sum(e["pnl"] for e in events)
    n = len(events)
    return BtStats(
        trades=n,
        win_rate=round(len(wins) / n * 100, 1),
        return_pct=round(total / CAPITAL * 100, 2),
        profit_factor=round(gp / gl, 2),
        pnl=round(total, 2),
    )


def make_dual_path_signal(hist, config, capital):
    """Elite strong/engulf OR v2 bounce — more trade frequency."""
    elite = make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3)(hist, config, capital)
    if elite:
        return elite
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if abs(c - float(e20)) / float(e20) > 0.012:
        return None
    lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
    if not all(lows[i] > lows[i - 1] for i in range(1, 4)):
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


STRATEGIES = [
    ("elite_base", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3), "off", 0),
    ("elite_regime200", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3), "50+200", 0),
    ("elite_regime_pause2", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3), "50+200", 2),
    ("elite_relaxed", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20), "off", 0),
    ("elite_relaxed_regime", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20), "50+200", 0),
    ("elite_adx22_4hl", make_ema20_strong(adx_min=22, ema_tol=0.012, hl_bars=4), "off", 0),
    ("elite_adx22_regime", make_ema20_strong(adx_min=22, ema_tol=0.012, hl_bars=4), "50+200", 0),
    ("elite_engulf", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, candle_mode="engulf_only"), "50+200", 0),
    ("elite_no_di", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, require_di=False), "off", 0),
    ("elite_no_di_regime", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, require_di=False), "50+200", 0),
    ("elite_hammer", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, candle_mode="any_bullish"), "off", 0),
    ("elite_wide_stop", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, stop_mode="wide"), "50+200", 0),
    ("elite_strict", make_ema20_strong(adx_min=24, rsi_lo=50, rsi_hi=56, hl_bars=4, ema_tol=0.012), "50+200", 0),
    ("elite_combo_u", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=46, rsi_hi=58, adx_min=21), "off", 0),
    ("elite_combo_v", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=58, adx_min=22, require_di=False, candle_mode="any_bullish"), "off", 0),
    ("elite_regime50", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3), "50", 0),
    ("dual_path", make_dual_path_signal, "off", 0),
    ("dual_path_regime", make_dual_path_signal, "50+200", 0),
    ("dual_path_pause2", make_dual_path_signal, "50+200", 2),
    ("elite_stack", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, require_e50_stack=True), "50+200", 0),
    ("elite_2hl_relaxed", make_ema20_strong(hl_bars=2, ema_tol=0.018, rsi_lo=44, rsi_hi=62, adx_min=18, require_di=False, candle_mode="any_bullish"), "off", 0),
    ("elite_combo_v_regime", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=58, adx_min=22, require_di=False, candle_mode="any_bullish"), "50+200", 0),
    ("elite_relaxed_pause2", make_ema20_strong(hl_bars=3, ema_tol=0.015, rsi_lo=45, rsi_hi=60, adx_min=20), "50+200", 2),
]


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"Loading {len(symbols)} symbols...", flush=True)
    cache = build_cache(symbols)
    print(f"Cached {len(cache)} | End date {END} | 2% risk, 2R target\n", flush=True)

    config = StrategyConfig.get_active()
    config.risk_pct = 2.0
    nifty = load_nifty50_frame()

    results: list[tuple] = []
    for name, fn, regime, pause in STRATEGIES:
        print(f"  Backtesting {name}...", flush=True)
        row = {"name": name, "regime": regime, "pause": pause}
        for wlabel, wstart in WINDOWS.items():
            stats = backtest(cache, nifty, config, fn, wstart, END, regime=regime, pause_after=pause)
            if stats:
                row[wlabel] = stats.return_pct
                row[f"{wlabel}_n"] = stats.trades
                row[f"{wlabel}_wr"] = stats.win_rate
            else:
                row[wlabel] = None
        results.append(row)

    # Sort by 1y return
    results.sort(key=lambda r: r.get("1y") or -999, reverse=True)

    print("=" * 110)
    print(f"{'Strategy':<22} {'9m ret':>8} {'1y ret':>8} {'2y ret':>8} {'3y ret':>8} | {'1y n':>5} {'1y WR':>6}")
    print("=" * 110)
    for r in results:
        def fmt(k):
            v = r.get(k)
            return f"{v:>7.1f}%" if v is not None else "      —"

        print(
            f"{r['name']:<22} {fmt('9m')} {fmt('1y')} {fmt('2y')} {fmt('3y')} | "
            f"{r.get('1y_n') or 0:>5} {r.get('1y_wr') or 0:>5.1f}%"
        )

    hits_1y = [r for r in results if r.get("1y") is not None and r["1y"] >= 30]
    hits_ann = []
    for r in results:
        for w, days in [("1y", 365), ("2y", 730), ("3y", 1095)]:
            ret = r.get(w)
            if ret is not None and (ret / days * 365) >= 30:
                hits_ann.append((r["name"], w, ret, ret / days * 365))

    print("\n" + "=" * 110)
    print("STRATEGIES WITH >=30% IN 1-YEAR WINDOW:")
    if hits_1y:
        for r in hits_1y:
            print(f"  {r['name']}: 1y={r['1y']}% ({r.get('1y_n')} trades, {r.get('1y_wr')}% WR)")
    else:
        print("  None found.")

    print("\nSTRATEGIES WITH >=30% ANNUALIZED (any window):")
    if hits_ann:
        for name, w, ret, ann in sorted(hits_ann, key=lambda x: -x[3])[:10]:
            print(f"  {name}: {w} raw={ret:.1f}% annualized={ann:.1f}%")
    else:
        print("  None found.")

    if results:
        best = results[0]
        print(f"\nBEST 1-YEAR: {best['name']} -> {best.get('1y')}% ({best.get('1y_n')} trades, {best.get('1y_wr')}% WR)")


if __name__ == "__main__":
    main()