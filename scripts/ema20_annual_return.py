"""
Annual return analysis for EMA20 + strong/engulf (enhanced).
Target: >= 30% return per year on ₹1L capital (₹1L → ₹1.3L).
"""
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

import pandas as pd

from scripts.ema20_enhance_backtest import (
    COOLDOWN, MAX_HOLD, START, END, build_cache, make_ema20_strong,
)
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.position_sizing import calculate_position_size


@dataclass
class AnnualReport:
    name: str
    capital: float
    risk_pct: float
    total_return_pct: float
    annual_returns: dict = field(default_factory=dict)
    min_annual_pct: float = 0.0
    avg_annual_pct: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    final_equity: float = 0.0


def backtest_annual(
    name: str,
    signal_fn,
    cache: dict,
    capital: float,
    risk_pct: float,
    start: date = START,
    end: date = END,
) -> AnnualReport:
    trades = []
    equity = capital
    year_start_equity: dict[int, float] = {}
    year_pnl: dict[int, float] = defaultdict(float)

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
            yr = ts.year

            if yr not in year_start_equity:
                year_start_equity[yr] = equity

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
                    trades.append({"win": pnl > 0, "pnl": pnl, "year": yr, "r": reason})
                    equity += pnl
                    year_pnl[yr] += pnl
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
                    trades.append({"win": False, "pnl": pnl, "year": yr, "r": "sl"})
                    equity += pnl
                    year_pnl[yr] += pnl
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl, "year": yr, "r": "2r"})
                    equity += pnl
                    year_pnl[yr] += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN:
                continue

            # Override risk_pct for sizing
            row = hist.iloc[-1]
            c = float(row["close"])
            e20 = row.get("ema_20")
            atr = row.get("atr_14")
            if any(pd.isna(x) for x in [e20, atr]):
                continue

            # Re-run signal logic via signal_fn but patch sizing
            sig = signal_fn(hist, StrategyConfig.get_active(), equity)
            if not sig:
                continue
            pos = calculate_position_size(equity, risk_pct, sig["entry"], sig["stop"])
            if pos.quantity <= 0:
                continue
            sig["qty"] = pos.quantity
            if i + 1 < len(dates):
                pending = sig

    if not trades:
        return AnnualReport(name, capital, risk_pct, 0, {}, 0, 0, 0, 0, 0, capital)

    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    wr = len(wins) / len(trades) * 100
    total_ret = (equity - capital) / capital * 100

    annual_returns = {}
    sorted_years = sorted(year_start_equity.keys())
    for yr in sorted_years:
        start_eq = year_start_equity[yr]
        ret = year_pnl[yr] / start_eq * 100 if start_eq else 0
        annual_returns[yr] = round(ret, 2)

    annual_vals = list(annual_returns.values())
    return AnnualReport(
        name=name,
        capital=capital,
        risk_pct=risk_pct,
        total_return_pct=round(total_ret, 2),
        annual_returns=annual_returns,
        min_annual_pct=min(annual_vals) if annual_vals else 0,
        avg_annual_pct=round(sum(annual_vals) / len(annual_vals), 2) if annual_vals else 0,
        trades=len(trades),
        win_rate=round(wr, 2),
        profit_factor=round(gp / gl, 2),
        final_equity=round(equity, 2),
    )


def main():
    capital = 100_000.0
    cache = build_cache(get_universe_symbols(nifty200_only=True))

    enhanced = make_ema20_strong(
        adx_min=20, ema_tol=0.014, hl_bars=3, candle_mode="strong_engulf",
    )
    baseline = make_ema20_strong()
    high_wr = make_ema20_strong(adx_min=20, ema_tol=0.015, hl_bars=4)

    print(f"Capital: Rs {capital:,.0f} | Period: {START} to {END}")
    print(f"Stocks: {len(cache)} | Target: >= 30% every calendar year\n")

    scenarios = []
    for risk in [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        scenarios.append((f"Enhanced ADX20+3HL (risk {risk}%)", enhanced, risk))
    scenarios.append(("Baseline original (risk 0.75%)", baseline, 0.75))
    scenarios.append(("High-WR ADX20+tol1.5% (risk 2%)", high_wr, 2.0))

    results = []
    for name, fn, risk in scenarios:
        r = backtest_annual(name, fn, cache, capital, risk)
        results.append(r)
        yrs = " | ".join(f"{y}: {p:+.1f}%" for y, p in sorted(r.annual_returns.items()))
        hit = "PASS" if r.min_annual_pct >= 30 and r.trades >= 5 else "FAIL"
        print(
            f"[{hit}] {name}\n"
            f"     trades={r.trades} WR={r.win_rate}% PF={r.profit_factor} "
            f"total={r.total_return_pct:+.1f}% -> Rs {r.final_equity:,.0f}\n"
            f"     yearly: {yrs}\n"
            f"     min_year={r.min_annual_pct:+.1f}% avg_year={r.avg_annual_pct:+.1f}%\n"
        )

    passing = [r for r in results if r.min_annual_pct >= 30 and r.trades >= 5]
    print("=" * 70)
    if passing:
        best = max(passing, key=lambda x: x.min_annual_pct)
        print(f"MEETS 30%/yr TARGET: {best.name}")
        print(f"  Rs {capital:,.0f} -> Rs {best.final_equity:,.0f} over full period")
        for y, p in sorted(best.annual_returns.items()):
            end_val = capital * (1 + p / 100)  # approx per-year on start
            print(f"  {y}: {p:+.1f}% (~Rs {end_val:,.0f} if 30% hit on Rs 1L)")
    else:
        best = max(results, key=lambda x: x.avg_annual_pct)
        print("NO variant hit 30% in EVERY calendar year.")
        print(f"Best average annual: {best.name} -> {best.avg_annual_pct:+.1f}%/yr")
        # Estimate risk needed for 30% given avg stats
        if best.avg_annual_pct > 0:
            needed_risk = 30 / best.avg_annual_pct * best.risk_pct
            print(f"Rough estimate: need ~{needed_risk:.1f}% risk/trade to target 30%/yr "
                  f"(current {best.risk_pct}% gives {best.avg_annual_pct:+.1f}%/yr)")


if __name__ == "__main__":
    main()