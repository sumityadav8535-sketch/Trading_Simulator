"""Apply enhanced EMA20 elite strategy settings: Rs 1L capital, 2% risk per trade."""
from django.core.management.base import BaseCommand

from trading.models import StrategyConfig


class Command(BaseCommand):
    help = "Set active strategy config to Rs 1L capital and 2% risk per trade."

    def handle(self, *args, **options):
        config = StrategyConfig.get_active()
        config.name = "EMA20 Elite"
        config.risk_pct = 2.0
        config.capital_default = 100_000
        config.adx_min = 20.0
        config.rsi_low = 49.0
        config.rsi_high = 58.0
        config.ema_pullback_tolerance_pct = 1.4
        config.min_risk_reward = 2.0
        config.save()
        self.stdout.write(
            self.style.SUCCESS(
                f"Updated '{config.name}': capital=Rs {config.capital_default:,.0f}, "
                f"risk={config.risk_pct}%/trade, ADX>={config.adx_min}, "
                f"RSI {config.rsi_low}-{config.rsi_high}"
            )
        )