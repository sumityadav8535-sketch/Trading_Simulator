"""Download latest NSE OHLCV from yfinance and upsert into DailyPrice."""
from datetime import date, datetime

from django.core.management.base import BaseCommand

from trading.services.nse_price_sync import active_equity_symbols, sync_universe


class Command(BaseCommand):
    help = "Fetch latest NSE daily prices via yfinance and store in the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated symbols (default: Nifty 200 + Smallcap 250)",
        )
        parser.add_argument(
            "--universe",
            type=str,
            default="tracked",
            choices=["tracked", "nifty200", "nifty_smallcap250", "all"],
            help="Which stocks to update when --symbols is omitted (default: nifty200 + smallcap250)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-fetch even if data already looks current",
        )
        parser.add_argument(
            "--from-date",
            type=str,
            default="",
            help="Backfill history from this date (YYYY-MM-DD). Use 2015-01-01 to extend past the 5y store.",
        )
        parser.add_argument(
            "--no-index",
            action="store_true",
            help="Skip Nifty 50 index sync",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="yfinance download batch size (default: NSE_SYNC_BATCH_SIZE setting)",
        )

    def handle(self, *args, **options):
        backfill_from = None
        raw_from = (options.get("from_date") or "").strip()
        if raw_from:
            backfill_from = datetime.strptime(raw_from, "%Y-%m-%d").date()

        universe = options.get("universe") or "tracked"
        if options["symbols"] or universe != "tracked":
            if options["symbols"]:
                symbols = [s.strip().upper() for s in options["symbols"].split(",") if s.strip()]
            elif universe == "nifty200":
                from trading.services.market_data import get_universe_symbols

                symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
            elif universe == "nifty_smallcap250":
                from trading.services.market_data import get_universe_symbols

                symbols = get_universe_symbols(nifty_smallcap250_only=True)
            else:
                from trading.services.market_data import get_universe_symbols

                symbols = [s for s in get_universe_symbols(nifty200_only=False) if s != "NIFTY50"]

            from trading.services.nse_price_sync import sync_symbols

            stats = sync_symbols(
                symbols,
                batch_size=options["batch_size"],
                force=options["force"],
                backfill_from=backfill_from,
                pause_seconds=1.0 if backfill_from else 0.4,
            )
            if not options["no_index"]:
                from trading.services.nifty50_index import sync_nifty50_from_yfinance

                years = 1
                if backfill_from is not None:
                    years = max(1, (date.today() - backfill_from).days // 365 + 1)
                stats["nifty50_bars"] = sync_nifty50_from_yfinance(years=years)
        else:
            extra = {}
            if backfill_from is not None:
                extra["pause_seconds"] = 1.0
            stats = sync_universe(
                include_index=not options["no_index"],
                force=options["force"],
                batch_size=options["batch_size"],
                backfill_from=backfill_from,
                **extra,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Updated {stats.get('symbols_updated', 0)} symbols, "
                f"{stats.get('bars_upserted', 0)} bars upserted"
                + (
                    f", Nifty50 {stats.get('nifty50_bars', 0)} bars"
                    if stats.get("nifty50_bars") is not None
                    else ""
                )
            )
        )
        if stats.get("errors"):
            self.stdout.write(self.style.WARNING(f"Errors: {stats['errors']}"))

        universe = active_equity_symbols()
        if universe:
            from trading.services.nse_price_sync import get_symbol_last_date

            sample = universe[0]
            self.stdout.write(f"Sample {sample} last bar: {get_symbol_last_date(sample)}")