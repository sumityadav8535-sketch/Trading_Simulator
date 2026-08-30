"""
Mark Nifty Smallcap 250 constituents from the official NSE CSV.

Usage:
    python manage.py mark_nifty_smallcap250
    python manage.py mark_nifty_smallcap250 --refresh
"""
from django.core.management.base import BaseCommand

from trading.services.nifty_smallcap250 import (
    ensure_nifty_smallcap250_marked,
    fetch_smallcap250_constituents,
)


class Command(BaseCommand):
    help = "Set is_nifty_smallcap250 flag from the official NSE Smallcap 250 constituent list"

    def add_arguments(self, parser):
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force refresh the NSE constituent CSV cache",
        )

    def handle(self, *args, **options):
        constituents = fetch_smallcap250_constituents(force_refresh=options["refresh"])
        symbols = ensure_nifty_smallcap250_marked(force_refresh=options["refresh"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Nifty Smallcap 250 ready: {len(symbols)} symbols marked "
                f"({len(constituents)} from NSE CSV)"
            )
        )
