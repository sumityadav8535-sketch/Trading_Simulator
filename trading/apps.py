import os
import sys

from django.apps import AppConfig


class TradingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trading"
    verbose_name = "Confluence Swing Trading"

    def ready(self) -> None:
        if self._should_skip_auto_sync():
            return
        from trading.services.nse_price_sync import schedule_auto_sync
        from trading.services.localhost_auto_refresh import schedule_localhost_auto_refresh

        schedule_auto_sync()
        schedule_localhost_auto_refresh()

    @staticmethod
    def _should_skip_auto_sync() -> bool:
        argv = sys.argv
        if not argv:
            return True
        skip_commands = {
            "migrate",
            "makemigrations",
            "test",
            "shell",
            "update_nse_prices",
            "load_nse_data",
            "collectstatic",
        }
        if any(cmd in argv for cmd in skip_commands):
            return True
        if "runserver" in argv and os.environ.get("RUN_MAIN") != "true":
            return True
        if any("scripts" in arg.replace("\\", "/") for arg in argv):
            return True
        return False