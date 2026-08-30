"""Validate Dual Supertrend + Minervini on the full 5y window, with and without costs."""
from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.st_union_swing import run_st_union_backtest
from trading.services.market_data import get_universe_symbols


def show(label, r):
    print(f"\n=== {label} ===")
    print(f"  return     {r.total_return_pct:+.2f}%")
    print(f"  final      Rs {r.final_cash:,.0f}")
    print(f"  trades     {r.total_trades}")
    print(f"  signals    {r.stage2_entries}")
    print(f"  win rate   {r.win_rate}%")
    print(f"  PF         {r.profit_factor}")
    print(f"  max DD     {r.max_drawdown_pct}%")
    print(f"  avg hold   {r.avg_hold_days}")
    print(f"  peak par   {r.peak_parallel}")
    print(f"  exits      {r.exit_breakdown}")
    wins = sorted([t for t in r.trades if t.pnl > 0], key=lambda t: t.pnl, reverse=True)[:8]
    print("  top wins:")
    for t in wins:
        print(f"    {t.symbol:12} {t.entry_date} → {t.exit_date}  {t.pnl_pct:+7.1f}%  Rs {t.pnl:+,.0f}  {t.exit_reason}  {t.days_held}d")
    by_year = {}
    for t in r.trades:
        y = str(t.exit_date)[:4]
        by_year[y] = by_year.get(y, 0.0) + t.pnl
    print("  PnL vs start capital by exit year:")
    for y in sorted(by_year):
        print(f"    {y}: {by_year[y] / r.capital * 100:+.1f}%")


def main():
    symbols = get_universe_symbols(nifty200_only=True)
    kw = dict(
        symbols=symbols,
        start_date=date(2021, 7, 5),
        end_date=date(2026, 8, 21),
        capital=1_000_000.0,
    )
    r0 = run_st_union_backtest(**kw)
    show("no costs", r0)
    r1 = run_st_union_backtest(**kw, cost_pct=0.1)
    show("0.1% cost each side", r1)


if __name__ == "__main__":
    main()
