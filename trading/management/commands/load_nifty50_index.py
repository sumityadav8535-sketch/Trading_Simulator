"""Download Nifty 50 index OHLCV from yfinance into DailyPrice."""
from django.core.management.base import BaseCommand

from trading.services.nifty50_index import sync_nifty50_from_yfinance


class Command(BaseCommand):
    help = "Load Nifty 50 index (^NSEI) daily prices for market regime filter"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=5, help="Years of history to fetch")

    def handle(self, *args, **options):
        n = sync_nifty50_from_yfinance(years=options["years"])
        self.stdout.write(self.style.SUCCESS(f"Loaded {n} Nifty 50 daily bars into NIFTY50"))