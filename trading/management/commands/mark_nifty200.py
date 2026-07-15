"""
Mark stocks as Nifty 200 from a comma-separated symbol list or file.

Usage:
    python manage.py mark_nifty200 --all
    python manage.py mark_nifty200 --symbols RELIANCE,TCS,INFY
    python manage.py mark_nifty200 --file nifty200.txt
"""
from pathlib import Path

from django.core.management.base import BaseCommand

from trading.models import Stock


class Command(BaseCommand):
    help = "Set is_nifty200 flag on stocks"

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Mark all active stocks as Nifty 200")
        parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
        parser.add_argument("--file", type=str, default="", help="File with one symbol per line")
        parser.add_argument("--clear", action="store_true", help="Clear nifty200 flag before applying")

    def handle(self, *args, **options):
        if options["clear"]:
            Stock.objects.update(is_nifty200=False)

        symbols = []
        if options["all"]:
            symbols = list(Stock.objects.filter(is_active=True).values_list("symbol", flat=True))
        elif options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",") if s.strip()]
        elif options["file"]:
            text = Path(options["file"]).read_text(encoding="utf-8")
            symbols = [line.strip().upper() for line in text.splitlines() if line.strip()]

        if not symbols:
            self.stderr.write("Provide --all, --symbols, or --file")
            return

        updated = Stock.objects.filter(symbol__in=symbols).update(is_nifty200=True)
        self.stdout.write(self.style.SUCCESS(f"Marked {updated} stocks as Nifty 200"))