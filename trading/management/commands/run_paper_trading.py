"""
Continuous paper-trading engine for multi-month forward tests.

Usage:
  python manage.py run_paper_trading
  python manage.py run_paper_trading --interval 30 --once

Runs during market hours (and still manages exits / force-exit near close).
Leave this running on a machine that stays on during market sessions for 2–3 months.
"""
from __future__ import annotations

import time
from datetime import time as dt_time

from django.core.management.base import BaseCommand

from trading.models import PaperAccount
from trading.services.fno_engine import FORCE_EXIT
from trading.services.intraday_data import get_market_status
from trading.services.paper_trading import get_dashboard, run_tick


class Command(BaseCommand):
    help = "Run paper trading engine loop (auto place/manage F&O paper orders)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--interval",
            type=int,
            default=45,
            help="Seconds between ticks (default 45)",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Run a single tick and exit",
        )
        parser.add_argument(
            "--auto-on",
            action="store_true",
            help="Enable auto_trade on the active paper account",
        )

    def handle(self, *args, **options):
        interval = max(15, int(options["interval"]))
        from trading.services.paper_trading import ensure_auto_trade

        # Always enable auto-trade so F&O signals open paper orders.
        account = ensure_auto_trade(PaperAccount.get_active())
        self.stdout.write(self.style.SUCCESS("Auto-trade ENABLED"))

        self.stdout.write(
            f"Paper account #{account.id} | cash=₹{float(account.cash):,.0f} | "
            f"auto={account.auto_trade} | interval={interval}s"
        )

        if options["once"]:
            result = run_tick(account, force_refresh=True, ensure_auto=True)
            self.stdout.write(str(result))
            dash = get_dashboard(account)
            self.stdout.write(
                f"Equity ₹{dash['account']['equity']:,.0f} | "
                f"open={len(dash['open_positions'])} | trades={dash['stats']['trades']}"
            )
            return

        self.stdout.write("Looping… Ctrl+C to stop. Keep PC awake during market hours.")
        while True:
            try:
                market = get_market_status()
                # Always tick when market open, or after 15:15 to force-exit, or if positions open
                account.refresh_from_db()
                has_open = account.positions.filter(status="open").exists()
                t = market.now_ist  # display only
                should = market.is_open or has_open

                # Also run briefly post-close for force exits until ~15:25
                from trading.services.paper_trading import _now_ist

                now = _now_ist()
                if FORCE_EXIT <= now.time() <= dt_time(15, 25):
                    should = True

                if should:
                    result = run_tick(account, force_refresh=True, ensure_auto=True)
                    self.stdout.write(
                        f"[{market.now_ist}] opened={result['opened']} closed={result['closed']} "
                        f"cash=₹{result['cash']:,.0f} realized=₹{result['realized_pnl']:,.0f} "
                        f"auto={result['auto_trade']} | {market.message}"
                    )
                else:
                    self.stdout.write(f"[{market.now_ist}] idle — {market.message}")

                time.sleep(interval)
            except KeyboardInterrupt:
                self.stdout.write(self.style.WARNING("\nStopped."))
                break
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"Tick error: {exc}"))
                time.sleep(interval)
