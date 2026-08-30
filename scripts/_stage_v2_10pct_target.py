"""
One-off research run: Stage Analysis 2.0, last 1 year, 10% take-profit.

Same entries / stop / 65-day time stop as the live engine.
Changes vs live: +10% take-profit (e.g. 1000 → 1100) instead of 2.5R,
and no Stage 4 (or any stage) exit. Not wired into the app.
"""
from __future__ import annotations

import inspect
import os
import sys
from collections import Counter
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

from stage_analysis_v2.services import backtester as bt
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols


def _patched_runner():
    src = inspect.getsource(bt.run_stage_v2_backtest)
    src = src.replace(
        "target = round(entry_price + risk * target_rr, 2)",
        "target = round(entry_price * 1.10, 2)",
        1,
    )
    src = src.replace(
        'target_exit_label = f"target_{target_rr:g}r"',
        'target_exit_label = "target_10pct"',
        1,
    )
    if "entry_price * 1.10" not in src or "target_10pct" not in src:
        raise RuntimeError("Failed to patch 10% take-profit into backtester source")
    ns = dict(vars(bt))
    exec(compile(src, "<stage_v2_10pct>", "exec"), ns)
    return ns["run_stage_v2_backtest"]


def _print_result(r, title: str) -> None:
    print("")
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"Period:            {r.start_date} → {r.end_date}")
    print(f"Capital:           Rs {r.capital:,.0f}")
    print(f"Stocks scanned:    {r.stocks_scanned}")
    print(f"Stage 2 signals:   {r.stage2_entries}")
    print(f"Trades executed:   {r.total_trades}")
    print(f"Skipped (no cash): {r.signals_skipped_cash}")
    print(f"Peak parallel:     {r.peak_parallel}")
    print(f"Win rate:          {r.win_rate}%")
    print(f"Profit factor:     {r.profit_factor}")
    print(f"Total return:      {r.total_return_pct}%")
    print(f"Max drawdown:      {r.max_drawdown_pct}%")
    print(f"Avg R achieved:    {r.avg_rr}")
    print(f"Avg hold (days):   {r.avg_hold_days}")
    print(f"Final cash:        Rs {r.final_cash:,.0f}")
    print(f"Exit breakdown:    {r.exit_breakdown}")

    if r.trades:
        wins = [t for t in r.trades if t.pnl > 0]
        losses = [t for t in r.trades if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = sum(t.pnl for t in losses)
        print(f"Gross profit:      Rs {gp:,.0f}  ({len(wins)} wins)")
        print(f"Gross loss:        Rs {gl:,.0f}  ({len(losses)} losses)")
        print(f"Avg win:           Rs {(gp / len(wins)) if wins else 0:,.0f}")
        print(f"Avg loss:          Rs {(gl / len(losses)) if losses else 0:,.0f}")
        pcts = [t.pnl_pct for t in r.trades]
        print(f"Avg trade %:       {sum(pcts) / len(pcts):+.2f}%")
        hit_10 = [t for t in r.trades if t.exit_reason == "target_10pct"]
        print(
            f"Hit +10% target:   {len(hit_10)} / {len(r.trades)} "
            f"({len(hit_10) / len(r.trades) * 100:.1f}%)"
        )

    if r.monthly_returns:
        print("")
        print("Monthly P&L:")
        for m in r.monthly_returns:
            sign = "+" if m["pnl"] >= 0 else ""
            print(f"  {m['month']}: {sign}{m['pnl']:,.0f} ({sign}{m['return_pct']}%)")

    if r.trades:
        print("")
        print("Top 15 winners:")
        for t in sorted(r.trades, key=lambda x: x.pnl, reverse=True)[:15]:
            print(
                f"  {t.symbol:12} {t.entry_date}→{t.exit_date}  "
                f"{t.entry_price:.2f}→{t.exit_price:.2f}  "
                f"{t.pnl_pct:+6.1f}%  PnL {t.pnl:+,.0f}  "
                f"{t.days_held:3d}d  {t.exit_reason}"
            )
        print("")
        print("Top 15 losers:")
        for t in sorted(r.trades, key=lambda x: x.pnl)[:15]:
            print(
                f"  {t.symbol:12} {t.entry_date}→{t.exit_date}  "
                f"{t.entry_price:.2f}→{t.exit_price:.2f}  "
                f"{t.pnl_pct:+6.1f}%  PnL {t.pnl:+,.0f}  "
                f"{t.days_held:3d}d  {t.exit_reason}"
            )

        print("")
        print("P&L by exit reason:")
        by_reason: dict[str, list] = {}
        for t in r.trades:
            by_reason.setdefault(t.exit_reason, []).append(t)
        for reason, ts in sorted(by_reason.items(), key=lambda kv: -sum(x.pnl for x in kv[1])):
            pnl = sum(x.pnl for x in ts)
            wr = sum(1 for x in ts if x.pnl > 0) / len(ts) * 100
            avg_hold = sum(x.days_held for x in ts) / len(ts)
            print(
                f"  {reason:16} n={len(ts):3d}  WR={wr:5.1f}%  "
                f"PnL {pnl:+,.0f}  avg hold {avg_hold:.1f}d"
            )


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)
    capital = 1_000_000.0
    symbols = get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]

    print(
        f"Stage Analysis 2.0 | 10% take-profit, no stage exit | {start} → {end} | "
        f"{len(symbols)} symbols | capital Rs {capital:,.0f}"
    )
    print("  Entry: weekly Stage 2 transition")
    print("  Tech filter: daily_mtf (daily Stage 1/2)")
    print("  Stop: 30w MA × 0.95")
    print("  Profit: +10% of entry price (close at 1100 if bought at 1000)")
    print("  Stage exit: OFF (no Stage 4 force-close)")
    print("  Other exits: 65-day time stop, cooldown 40d")
    print("  Sizing: 2% risk of equity, cash-constrained, no leverage")
    print("Loading data and simulating...")

    run = _patched_runner()
    r = run(
        symbols=symbols,
        start_date=start,
        end_date=end,
        capital=capital,
        min_quality_score=0,
        market_filter=False,
        exit_mode=bt.EXIT_NO_STAGE,
        tech_filter="daily_mtf",
        entry_stage=2,
        entry_on=bt.DEFAULT_ENTRY_ON,
    )
    r.strategy_name = "Stage Analysis 2.0 — 10% TP, no stage exit (research)"
    _print_result(r, "RESULTS — 10% take-profit, no Stage 4 exit (last 1 year)")


if __name__ == "__main__":
    main()
