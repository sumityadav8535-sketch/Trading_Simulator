"""Find RS Pullback extra filters that raise last-1y return and win rate."""
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

from trading.services.short_swing import (  # noqa: E402
    DEFAULT_CAPITAL,
    F_ADX20,
    F_FIRST_TOUCH,
    F_MKT20,
    F_NEAR52,
    F_NOT_EXT,
    F_QULLA,
    F_RS63,
    F_RSI_OK,
    F_ST_BULL,
    F_TREND,
    F_VOL,
    LEADERS,
    SwingParams,
    collect_signals,
    nifty_return,
    preload_packs,
    simulate,
    slice_cal,
)

END = date.today()
WINDOWS = {
    "1y": (END - timedelta(days=365), END),
    "2y": (END - timedelta(days=365 * 2), END),
    "5y": (END - timedelta(days=365 * 5), END),
}


def run(packs, nifty, calendar, sigs, flags, label):
    p = SwingParams(flags=flags)
    out = {}
    n_sig = {}
    for name, (a, b) in WINDOWS.items():
        cal = slice_cal(calendar, a, b)
        passed = [s for s in sigs if (s["flags"] & flags) == flags and a <= s["sig_ts"].date() <= b]
        n_sig[name] = len(passed)
        r = simulate(packs, cal, sigs, p, capital=DEFAULT_CAPITAL, flatten=True, keep_trades=False)
        out[name] = r
    print(
        f"{label:42s}  sig1y={n_sig['1y']:4d}  "
        f"1y={out['1y']['total_return_pct']:7.1f}% WR{out['1y']['win_rate']:5.1f} "
        f"PF{out['1y']['profit_factor']:4.2f} n={out['1y']['trades']:3d} "
        f"DD{out['1y']['max_drawdown_pct']:5.1f}  "
        f"2y={out['2y']['total_return_pct']:7.1f}%  "
        f"5y={out['5y']['total_return_pct']:7.1f}% WR{out['5y']['win_rate']:5.1f}",
        flush=True,
    )
    return out, n_sig


def main() -> None:
    packs, nifty, calendar = preload_packs(force=True)
    start = WINDOWS["5y"][0]
    import pandas as pd
    sigs = collect_signals(packs, nifty, pd.Timestamp(start), pd.Timestamp(END), entry="ema20pb")
    print(f"raw ema20pb 5y={len(sigs)}  nifty1y={nifty_return(nifty, *WINDOWS['1y'])}", flush=True)

    combos = [
        ("baseline qulla", F_QULLA),
        ("qulla + first touch", F_QULLA | F_FIRST_TOUCH),
        ("qulla + RS vs Nifty", F_QULLA | F_RS63),
        ("qulla + trend", F_QULLA | F_TREND),
        ("qulla + ST bull", F_QULLA | F_ST_BULL),
        ("qulla + RSI 40-65", F_QULLA | F_RSI_OK),
        ("qulla + ADX20", F_QULLA | F_ADX20),
        ("qulla + volume", F_QULLA | F_VOL),
        ("qulla + not extended", F_QULLA | F_NOT_EXT),
        ("qulla + near 52w", F_QULLA | F_NEAR52),
        ("qulla + nifty>EMA20", F_QULLA | F_MKT20),
        ("qulla + first + RS", F_QULLA | F_FIRST_TOUCH | F_RS63),
        ("qulla + first + trend", F_QULLA | F_FIRST_TOUCH | F_TREND),
        ("qulla + RS + trend", F_QULLA | F_RS63 | F_TREND),
        ("qulla + RS + ST", F_QULLA | F_RS63 | F_ST_BULL),
        ("qulla + first + ST", F_QULLA | F_FIRST_TOUCH | F_ST_BULL),
        ("qulla + first + RSI", F_QULLA | F_FIRST_TOUCH | F_RSI_OK),
        ("qulla + first + not ext", F_QULLA | F_FIRST_TOUCH | F_NOT_EXT),
        ("qulla + RS + RSI", F_QULLA | F_RS63 | F_RSI_OK),
        ("qulla + RS + not ext", F_QULLA | F_RS63 | F_NOT_EXT),
        ("qulla + trend + ST", F_QULLA | F_TREND | F_ST_BULL),
        ("qulla + first + RS + ST", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL),
        ("qulla + first + RS + RSI", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_RSI_OK),
        ("qulla + first + RS + trend", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_TREND),
        ("qulla + RS + ST + RSI", F_QULLA | F_RS63 | F_ST_BULL | F_RSI_OK),
        ("qulla + first + RS + not ext", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_NOT_EXT),
        ("elite first+RS+ST+RSI", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL | F_RSI_OK),
        ("elite + trend", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL | F_TREND),
        ("elite + not ext", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL | F_NOT_EXT),
        ("elite + ADX", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL | F_ADX20),
        ("elite + near52", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_ST_BULL | F_NEAR52),
        ("leaders first+RS+trend+ST", F_QULLA | F_FIRST_TOUCH | F_RS63 | F_TREND | F_ST_BULL),
    ]
    results = []
    for label, fl in combos:
        out, n_sig = run(packs, nifty, calendar, sigs, fl, label)
        results.append((label, fl, out, n_sig))

    base = results[0][2]["1y"]["total_return_pct"]
    base_sig = results[0][3]["1y"]
    print("\nBEATS or matches baseline 1y return:", flush=True)
    for label, _fl, out, n_sig in results:
        if out["1y"]["total_return_pct"] >= base:
            print(
                f"  KEEP {label:42s} 1y {out['1y']['total_return_pct']:.1f}% "
                f"WR{out['1y']['win_rate']:.1f} sig {n_sig['1y']} (was {base_sig})  "
                f"5y {out['5y']['total_return_pct']:.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
