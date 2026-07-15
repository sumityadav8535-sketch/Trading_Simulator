"""
Mark Nifty 100 constituents from the official NSE CSV.

Usage:
    python manage.py mark_nifty100
    python manage.py mark_nifty100 --refresh
"""
from django.core.management.base import BaseCommand

from trading.services.nifty100 import ensure_nifty100_marked, fetch_nifty100_constituents


class Command(BaseCommand):
    help = "Set is_nifty100 flag from the official NSE Nifty 100 constituent list"

    def add_arguments(self, parser):
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force refresh the NSE constituent CSV cache",
        )

    def handle(self, *args, **options):
        constituents = fetch_nifty100_constituents(force_refresh=options["refresh"])
        symbols = ensure_nifty100_marked(force_refresh=options["refresh"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Nifty 100 ready: {len(symbols)} symbols marked "
                f"({len(constituents)} from NSE CSV)"
            )
        )