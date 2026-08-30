"""
Mark Nifty Smallcap 250 constituents and download daily OHLCV.

Usage:
    python manage.py load_nifty_smallcap250
    python manage.py load_nifty_smallcap250 --years 5 --refresh
"""
from datetime import date, timedelta

from django.core.management.base import BaseCommand

from trading.services.nifty_smallcap250 import ensure_nifty_smallcap250_marked
from trading.services.nse_price_sync import get_symbol_first_date, get_symbol_last_date, sync_symbols


class Command(BaseCommand):
    help = "Load Nifty Smallcap 250 universe and last N years of daily prices"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=float, default=5.0, help="Years of daily history (default 5)")
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force refresh the NSE constituent CSV cache",
        )
        parser.add_argument(
            "--skip-prices",
            action="store_true",
            help="Only mark constituents; do not download prices",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-fetch prices even if the store already looks complete",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="yfinance download batch size",
        )

    def handle(self, *args, **options):
        symbols = ensure_nifty_smallcap250_marked(force_refresh=options["refresh"])
        self.stdout.write(
            self.style.SUCCESS(f"Marked {len(symbols)} Nifty Smallcap 250 constituents")
        )
        if options["skip_prices"]:
            return

        years = max(float(options["years"]), 0.25)
        backfill_from = date.today() - timedelta(days=int(years * 365))
        self.stdout.write(
            f"Downloading daily OHLCV from {backfill_from} for {len(symbols)} symbols…"
        )
        stats = sync_symbols(
            symbols,
            batch_size=options["batch_size"],
            force=options["force"],
            backfill_from=backfill_from,
            pause_seconds=1.0,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Updated {stats.get('symbols_updated', 0)} symbols, "
                f"{stats.get('bars_upserted', 0)} bars upserted "
                f"(checked {stats.get('symbols_checked', 0)})"
            )
        )
        if stats.get("errors"):
            self.stdout.write(self.style.WARNING(f"Errors: {stats['errors']}"))

        missing = []
        short = []
        for sym in symbols:
            first = get_symbol_first_date(sym)
            last = get_symbol_last_date(sym)
            if first is None:
                missing.append(sym)
            elif first > backfill_from + timedelta(days=90):
                short.append((sym, first, last))

        if missing:
            preview = ", ".join(missing[:20])
            extra = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
            self.stdout.write(self.style.WARNING(f"No price data: {len(missing)} — {preview}{extra}"))
        if short:
            self.stdout.write(
                self.style.WARNING(
                    f"{len(short)} symbols start after {backfill_from} "
                    f"(listings / Yahoo gaps), e.g. {short[0][0]} from {short[0][1]}"
                )
            )
        sample = next((s for s in symbols if get_symbol_last_date(s)), None)
        if sample:
            self.stdout.write(
                f"Sample {sample}: {get_symbol_first_date(sample)} → {get_symbol_last_date(sample)}"
            )
