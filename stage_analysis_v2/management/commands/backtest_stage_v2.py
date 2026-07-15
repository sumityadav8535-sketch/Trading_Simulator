"""Run Stage Analysis 2.0 backtest on Nifty 200."""
from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand

from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from trading.services.market_data import get_universe_symbols


class Command(BaseCommand):
    help = "Backtest Stage Analysis 2.0 — buy on weekly Stage 2 entry (Nifty 200)"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=float, default=1.0, help="Lookback years (default 1)")
        parser.add_argument("--capital", type=float, default=1_000_000, help="Starting capital (INR)")
        parser.add_argument("--min-quality", type=int, default=0, help="Min quality score (0-100)")
        parser.add_argument("--market-filter", action="store_true", help="Only trade when Nifty is Stage 1/2")
        parser.add_argument("--strict", action="store_true", help="Strict V2: quality>=75, market filter on")
        parser.add_argument("--top", type=int, default=15, help="Show top N trades by P&L")

    def handle(self, *args, **options):
        end = date.today()
        start = end - timedelta(days=int(options["years"] * 365))
        min_q = options["min_quality"]
        mkt = options["market_filter"]
        if options["strict"]:
            min_q = 75
            mkt = True

        symbols = get_universe_symbols(nifty200_only=True)
        self.stdout.write(
            f"Stage Analysis 2.0 backtest | {start} → {end} | "
            f"{len(symbols)} symbols | capital Rs {options['capital']:,.0f}"
        )
        if min_q:
            self.stdout.write(f"  Min quality score: {min_q}")
        if mkt:
            self.stdout.write("  Market filter: Nifty Stage 1/2 only")

        r = run_stage_v2_backtest(
            symbols=symbols,
            start_date=start,
            end_date=end,
            capital=options["capital"],
            min_quality_score=min_q,
            market_filter=mkt,
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=== RESULTS ==="))
        self.stdout.write(f"Stocks scanned:     {r.stocks_scanned}")
        self.stdout.write(f"Stage 2 entries:    {r.stage2_entries}")
        self.stdout.write(f"Trades executed:    {r.total_trades}")
        self.stdout.write(f"Win rate:           {r.win_rate}%")
        self.stdout.write(f"Profit factor:      {r.profit_factor}")
        self.stdout.write(f"Total return:       {r.total_return_pct}%")
        self.stdout.write(f"Max drawdown:       {r.max_drawdown_pct}%")
        self.stdout.write(f"Avg R achieved:     {r.avg_rr}")
        self.stdout.write(f"Avg hold (days):    {r.avg_hold_days}")
        self.stdout.write(f"Exit breakdown:     {r.exit_breakdown}")

        if r.monthly_returns:
            self.stdout.write("")
            self.stdout.write("Monthly P&L:")
            for m in r.monthly_returns:
                sign = "+" if m["pnl"] >= 0 else ""
                self.stdout.write(f"  {m['month']}: {sign}{m['pnl']:,.0f} ({sign}{m['return_pct']}%)")

        top = options["top"]
        if r.trades and top:
            self.stdout.write("")
            self.stdout.write(f"Top {top} winners:")
            for t in sorted(r.trades, key=lambda x: x.pnl, reverse=True)[:top]:
                self.stdout.write(
                    f"  {t.symbol} {t.entry_date}→{t.exit_date} "
                    f"Q{t.quality_score} RS{t.rs_rating:.0f} "
                    f"PnL {t.pnl:+,.0f} ({t.exit_reason})"
                )
            self.stdout.write("")
            self.stdout.write(f"Top {top} losers:")
            for t in sorted(r.trades, key=lambda x: x.pnl)[:top]:
                self.stdout.write(
                    f"  {t.symbol} {t.entry_date}→{t.exit_date} "
                    f"Q{t.quality_score} RS{t.rs_rating:.0f} "
                    f"PnL {t.pnl:+,.0f} ({t.exit_reason})"
                )