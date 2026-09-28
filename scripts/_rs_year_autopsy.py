"""Calendar-year autopsy for RS Pullback Swing (2023–2026)."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.rs_pullback_swing import run_rs_pullback_backtest
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import load_price_dataframe
from trading.services.short_swing import nifty_return, preload_packs


def nifty_months(start: date, end: date) -> None:
    df = load_price_dataframe(NIFTY50_SYMBOL)
    n = df.loc[(df.index >= str(start)) & (df.index <= str(end))]
    if n.empty:
        return
    o = float(n["close"].iloc[0])
    c = float(n["close"].iloc[-1])
    dd = float((n["close"] / n["close"].cummax() - 1).min() * 100)
    print(f"  Nifty {o:.0f} → {c:.0f}  {(c/o-1)*100:+.1f}%  maxDD {dd:.1f}%")
    mclose = n["close"].resample("ME").last()
    mret = mclose.pct_change() * 100
    for ts, r in mret.items():
        if r == r:
            print(f"    {ts.strftime('%Y-%m')}: {r:+5.1f}%")


def autopsy(start: date, end: date) -> None:
    print("\n" + "=" * 72)
    print(f"YEAR {start} → {end}")
    print("=" * 72)
    nifty_months(start, end)
    r = run_rs_pullback_backtest(start_date=start, end_date=end, capital=1_000_000)
    print(
        f"  book  ret={r.total_return_pct}%  WR={r.win_rate}%  PF={r.profit_factor}  "
        f"DD={r.max_drawdown_pct}%  trades={r.total_trades}  signals={r.stage2_entries}  "
        f"skipped_cash={r.signals_skipped_cash}  avg_hold={r.avg_hold_days}  avg_R={r.avg_rr}"
    )
    print(f"  exits {r.exit_breakdown}")
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    if wins:
        print(
            f"  wins n={len(wins)} avg_pct={sum(t.pnl_pct for t in wins)/len(wins):+.1f} "
            f"avg_pnl={sum(t.pnl for t in wins)/len(wins):+,.0f}  "
            f"gross={sum(t.pnl for t in wins):+,.0f}"
        )
    if losses:
        print(
            f"  loss n={len(losses)} avg_pct={sum(t.pnl_pct for t in losses)/len(losses):+.1f} "
            f"avg_pnl={sum(t.pnl for t in losses)/len(losses):+,.0f}  "
            f"gross={sum(t.pnl for t in losses):+,.0f}"
        )
    by_exit = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in r.trades:
        by_exit[t.exit_reason]["n"] += 1
        by_exit[t.exit_reason]["pnl"] += t.pnl
        if t.pnl > 0:
            by_exit[t.exit_reason]["wins"] += 1
    print("  by exit:")
    for k, d in sorted(by_exit.items(), key=lambda x: x[1]["pnl"]):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"    {k:16s} n={d['n']:3d} WR={wr:5.1f}%  PnL={d['pnl']:+,.0f}")

    month = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in r.trades:
        m = t.exit_date[:7]
        month[m]["n"] += 1
        month[m]["pnl"] += t.pnl
        if t.pnl > 0:
            month[m]["wins"] += 1
    print("  P&L by exit month:")
    for m in sorted(month):
        d = month[m]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"    {m}  n={d['n']:3d} WR={wr:4.0f}%  PnL={d['pnl']:+,.0f}")

    print("  worst 8:")
    for t in sorted(losses, key=lambda x: x.pnl)[:8]:
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.pnl_pct:+6.1f}%  "
            f"₹{t.pnl:+,.0f}  {t.exit_reason}  {t.days_held}d"
        )
    print("  best 5:")
    for t in sorted(wins, key=lambda x: -x.pnl)[:5]:
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.pnl_pct:+6.1f}%  "
            f"₹{t.pnl:+,.0f}  {t.exit_reason}  {t.days_held}d"
        )


def main() -> None:
    preload_packs()
    autopsy(date(2023, 1, 1), date(2023, 12, 31))
    autopsy(date(2024, 1, 1), date(2024, 12, 31))
    autopsy(date(2025, 1, 1), date(2025, 12, 31))
    autopsy(date(2026, 1, 1), date.today())


if __name__ == "__main__":
    main()
