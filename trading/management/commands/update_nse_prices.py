"""Download latest NSE OHLCV from yfinance and upsert into DailyPrice."""
from django.core.management.base import BaseCommand

from trading.services.nse_price_sync import active_equity_symbols, sync_universe


class Command(BaseCommand):
    help = "Fetch latest NSE daily prices via yfinance and store in the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated symbols (default: all active Nifty 200)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-fetch even if data already looks current",
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
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",") if s.strip()]
            from trading.services.nse_price_sync import sync_symbols

            stats = sync_symbols(
                symbols,
                batch_size=options["batch_size"],
                force=options["force"],
            )
            if not options["no_index"]:
                from trading.services.nifty50_index import sync_nifty50_from_yfinance

                stats["nifty50_bars"] = sync_nifty50_from_yfinance(years=1)
        else:
            stats = sync_universe(
                include_index=not options["no_index"],
                force=options["force"],
                batch_size=options["batch_size"],
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