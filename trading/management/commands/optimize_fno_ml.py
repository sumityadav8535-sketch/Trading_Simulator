"""Run F&O ML loss analysis and strategy optimization."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from django.core.management.base import BaseCommand

ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Command(BaseCommand):
    help = "Analyze losses and optimize ML F&O filters; writes intraday_fno_optimized*.json"

    def handle(self, *args, **options):
        for script in ("intraday_fno_optimize.py",):
            path = ROOT / "scripts" / script
            self.stdout.write(f"Running {script}...")
            subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=False)
        out = ROOT / "data" / "intraday_fno_optimized_trades.json"
        if out.exists():
            self.stdout.write(self.style.SUCCESS(f"Optimized trade log: {out}"))
        else:
            self.stderr.write(self.style.ERROR("Optimization did not produce trade log."))