"""
Scan Nifty 200 (or watchlist) and persist signals.

Usage:
    python manage.py scan_signals
    python manage.py scan_signals --min-score 7 --watchlist
"""
from django.core.management.base import BaseCommand

from trading.models import Signal, StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.swing_strategy import scan_swing_universe


class Command(BaseCommand):
    help = "Scan universe for EMA20 Elite (strong/engulf) signals"

    def add_arguments(self, parser):
        parser.add_argument("--min-score", type=int, default=7)
        parser.add_argument("--watchlist", action="store_true")
        parser.add_argument("--capital", type=float, default=None)

    def handle(self, *args, **options):
        config = StrategyConfig.get_active()
        capital = options["capital"] or float(config.capital_default)
        symbols = get_universe_symbols(
            nifty200_only=not options["watchlist"],
            watchlist_only=options["watchlist"],
        )
        results = scan_swing_universe(
            symbols,
            config=config,
            capital=capital,
            min_score=options["min_score"],
            elite_only=True,
        )
        saved = 0
        for r in results:
            Signal.objects.update_or_create(
                stock_id=r.symbol,
                date=r.eval_date,
                defaults={
                    "confluence_score": r.confluence_score,
                    "is_valid": r.is_valid,
                    "entry_price": r.entry_price,
                    "stop_loss": r.stop_loss,
                    "target_1r": r.target_1r,
                    "target_2r": r.target_2r,
                    "target_3r": r.target_3r,
                    "risk_reward": r.risk_reward,
                    "position_size": r.position_size,
                    "capital_used": r.capital_used,
                    "reasons": r.reasons,
                    "rejection_reasons": r.rejection_reasons,
                    "indicator_snapshot": r.indicator_snapshot,
                },
            )
            saved += 1
        valid = sum(1 for r in results if r.is_valid)
        self.stdout.write(self.style.SUCCESS(f"Scanned {len(symbols)} symbols, saved {saved} signals ({valid} valid A+)"))