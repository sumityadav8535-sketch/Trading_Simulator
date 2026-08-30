"""Detail the promising loss-guards and try a Nifty panic flatten."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from _cup_hunt_100 import collect_signals
from _cup_loss_guard_search import BASE, CAPITAL, END, START, simulate
from stage_analysis_v2.services.backtester import _preload_frames
from stage_analysis_v2.services.cup_breakout import _CupPack
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    packs = {sym: _CupPack(sym, df) for sym, df in frames.items()}
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret60 = nifty["close"].pct_change(60)
    nifty_sma200 = nifty["close"].rolling(200, min_periods=200).mean()
    by_day, n_sig = collect_signals(packs, frames, BASE, nifty_ret60, nifty_sma200, nifty["close"])
    cal = sorted({
        ts for df in frames.values() for ts in df.index.tolist()
        if pd.Timestamp(START) <= ts <= pd.Timestamp(END)
    })
    print(f"signals={n_sig}", flush=True)

    configs = [
        ("baseline", BASE),
        ("nema20", replace(BASE, nifty_ema_period=20)),
        ("nema50", replace(BASE, nifty_ema_period=50)),
        ("strk3/10", replace(BASE, loss_streak=3, loss_streak_cooloff_days=10)),
        ("strk2/10", replace(BASE, loss_streak=2, loss_streak_cooloff_days=10)),
        ("nema20+strk3/10", replace(BASE, nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=10)),
        ("nema20+strk3/15", replace(BASE, nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=15)),
        ("nema20+ngap1.5", replace(BASE, nifty_ema_period=20, nifty_gap_down_pct=1.5)),
        ("nema20+strk+ngap1.5", replace(
            BASE, nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=10, nifty_gap_down_pct=1.5,
        )),
        ("nema20+strk+mcap8", replace(
            BASE, nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=10, max_month_loss_pct=8.0,
        )),
        ("nema20+open3+strk", replace(
            BASE, nifty_ema_period=20, max_open=3, loss_streak=3, loss_streak_cooloff_days=10,
        )),
        ("nema20+strk+dd18", replace(
            BASE, nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=10, halt_dd_pct=18.0,
        )),
    ]

    results = []
    for name, p in configs:
        st = simulate(packs, cal, by_day, nifty, p)
        st["label"] = name
        results.append(st)
        print(
            f"{name:22s} 5y={st['ret']:+7.1f}% 23={st['y2023']:+6.1f}% 24={st['y2024']:+6.1f}% "
            f"25={st['y2025']:+6.1f}% n={st['n']:3d} WR={st['wr']:4.1f}% "
            f"worst {st['worst_m']} {st['worst']:+,.0f}",
            flush=True,
        )

    base = results[0]
    best = max(results, key=lambda r: (r["ret"] - 0.00001 * abs(r["worst"]), r["y2023"]))
    # Prefer pack that cuts |worst| a lot while ret >= baseline
    scored = []
    for r in results:
        if r["ret"] < base["ret"] - 30:
            continue
        scored.append(r)
    scored.sort(key=lambda r: (r["worst"], -r["ret"]))  # still wrong if worst is more negative
    scored.sort(key=lambda r: (-r["worst"], -r["ret"]))  # least-bad worst month, then return

    print("\n===== ranked by least-bad worst-month, ret>=baseline-30pp =====")
    for r in scored:
        print(f"  {r['label']:22s} 5y={r['ret']:+7.1f}% worst={r['worst_m']} {r['worst']:+,.0f}")

    pick = scored[0] if scored else best
    print(f"\n===== {pick['label']} vs baseline (months that matter) =====")
    months = sorted(set(base["monthly"]) | set(pick["monthly"]))
    for m in months:
        b = base["monthly"].get(m, 0)
        n = pick["monthly"].get(m, 0)
        if b <= -40000 or n <= -40000 or abs(n - b) >= 40000:
            print(f"  {m}  base {b:+12,.0f}  new {n:+12,.0f}  Δ {n-b:+12,.0f}")


if __name__ == "__main__":
    main()
