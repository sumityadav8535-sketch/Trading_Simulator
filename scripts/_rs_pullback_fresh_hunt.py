"""Test fresh-only (no consecutive-day repeats) on RS Pullback."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from trading.services.short_swing import (
    DEFAULT_CAPITAL,
    F_FRESH,
    F_QULLA,
    F_RS63,
    SwingParams,
    collect_signals,
    preload_packs,
    simulate,
    slice_cal,
)

END = date.today()
W = {"1y": END - timedelta(days=365), "2y": END - timedelta(days=365 * 2), "5y": END - timedelta(days=365 * 5)}


def main() -> None:
    packs, nifty, calendar = preload_packs(force=True)
    raw = collect_signals(packs, nifty, pd.Timestamp(W["5y"]), pd.Timestamp(END), entry="ema20pb")
    combos = [
        ("qulla", F_QULLA),
        ("qulla+RS", F_QULLA | F_RS63),
        ("qulla+fresh", F_QULLA | F_FRESH),
        ("qulla+RS+fresh", F_QULLA | F_RS63 | F_FRESH),
    ]
    for label, fl in combos:
        bits = []
        for name, start in W.items():
            cal = slice_cal(calendar, start, END)
            n = sum(1 for s in raw if (s["flags"] & fl) == fl and start <= s["sig_ts"].date() <= END)
            p = SwingParams(flags=fl)
            r = simulate(packs, cal, raw, p, capital=DEFAULT_CAPITAL, flatten=True, keep_trades=False)
            bits.append(
                f"{name} sig={n:4d} {r['total_return_pct']:7.1f}% WR{r['win_rate']:5.1f} "
                f"n={r['trades']:3d} PF{r['profit_factor']:4.2f}"
            )
        print(f"{label:18s} " + " | ".join(bits), flush=True)


if __name__ == "__main__":
    main()
