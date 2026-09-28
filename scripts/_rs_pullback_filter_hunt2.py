"""Tighten RS pullback: min momentum, fewer names/day, keep or raise 1y return."""
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
    F_QULLA,
    F_RS63,
    SwingParams,
    collect_signals,
    preload_packs,
    simulate,
    slice_cal,
)

END = date.today()
W = {
    "1y": (END - timedelta(days=365), END),
    "5y": (END - timedelta(days=365 * 5), END),
}
NEED = F_QULLA | F_RS63


def filtered(sigs, min_score=None):
    out = []
    for s in sigs:
        if (s["flags"] & NEED) != NEED:
            continue
        if min_score is not None and s["score"] < min_score:
            continue
        out.append(s)
    return out


def go(packs, calendar, sigs, p, label):
    bits = []
    n1 = 0
    for name, (a, b) in W.items():
        cal = slice_cal(calendar, a, b)
        n = sum(1 for s in sigs if a <= s["sig_ts"].date() <= b)
        if name == "1y":
            n1 = n
        r = simulate(packs, cal, sigs, p, capital=DEFAULT_CAPITAL, flatten=True, keep_trades=False)
        bits.append(
            f"{name}={r['total_return_pct']:.1f}% WR{r['win_rate']:.1f} "
            f"PF{r['profit_factor']:.2f} n={r['trades']} DD{r['max_drawdown_pct']:.1f}"
        )
    print(f"{label:44s} sig1y={n1:4d}  " + "  ".join(bits), flush=True)


def main() -> None:
    packs, nifty, calendar = preload_packs()
    raw = collect_signals(packs, nifty, pd.Timestamp(W["5y"][0]), pd.Timestamp(END), entry="ema20pb")
    base = filtered(raw)
    print(f"qulla+RS 5y sigs={len(base)}", flush=True)

    jobs = []
    for min_s, tag in [(None, "any"), (0.15, "r63>=15"), (0.20, "r63>=20"), (0.25, "r63>=25"), (0.30, "r63>=30"), (0.40, "r63>=40")]:
        sigs = filtered(raw, min_s)
        for new in (1, 2, 3):
            for mx in (2, 3, 4):
                p = SwingParams(flags=NEED, max_new=new, max_open=mx)
                jobs.append((f"{tag} new{new} open{mx}", sigs, p))

    for label, sigs, p in jobs:
        go(packs, calendar, sigs, p, label)


if __name__ == "__main__":
    main()
