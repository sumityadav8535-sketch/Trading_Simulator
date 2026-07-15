"""
Management command: import NSE OHLCV from data/nse_data.sqlite

Usage:
    python manage.py load_nse_data
    python manage.py load_nse_data --no-nifty200
    python manage.py load_nse_data --path C:/path/to/nse_data.sqlite
"""
from django.core.management.base import BaseCommand

from trading.services.data_loader import load_from_legacy_sqlite


class Command(BaseCommand):
    help = "Load stocks and daily prices from legacy NSE SQLite export"

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            type=str,
            default=None,
            help="Override path to nse_data.sqlite (default: data/nse_data.sqlite)",
        )
        parser.add_argument(
            "--no-nifty200",
            action="store_true",
            help="Do not mark all imported stocks as Nifty 200",
        )

    def handle(self, *args, **options):
        path = options["path"]
        stats = load_from_legacy_sqlite(
            db_path=path,
            mark_all_nifty200=not options["no_nifty200"],
        )
        self.stdout.write(self.style.SUCCESS(f"Import complete: {stats}"))