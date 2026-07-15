"""Fine-tune EMA20 strong/engulf filters — target WR>50% with max trades."""
import os
import sys
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

from scripts.ema20_enhance_backtest import (
    START, END, VARIANTS, build_cache, backtest, make_ema20_strong,
)
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols


def main():
    config = StrategyConfig.get_active()
    symbols = get_universe_symbols(nifty200_only=True)
    cache = build_cache(symbols)
    print(f"Cached {len(cache)} stocks. Refining filters {START} to {END}\n")

    grid = []
    for adx, rsi_lo, rsi_hi, tol, hl, di in product(
        [20, 21, 22, 23],
        [47, 48, 49],
        [56, 57, 58],
        [0.013, 0.014, 0.015],
        [3, 4],
        [True],  # +DI required — dropping it hurt WR in coarse search
    ):
        if rsi_lo >= rsi_hi:
            continue
        name = f"adx{adx}_rsi{rsi_lo}-{rsi_hi}_tol{tol}_hl{hl}_di{int(di)}"
        fn = make_ema20_strong(
            adx_min=adx, rsi_lo=rsi_lo, rsi_hi=rsi_hi,
            ema_tol=tol, hl_bars=hl, require_di=di,
            candle_mode="strong_engulf",
        )
        grid.append((name, fn))

    results = []
    for name, fn in grid:
        r = backtest(name, fn, cache, config)
        if r.trades >= 10 and r.win_rate >= 50:
            results.append(r)

    results.sort(key=lambda x: (-x.trades, -x.win_rate, -x.profit_factor))
    print(f"Grid size: {len(grid)} | Qualified (WR>=50%, n>=10): {len(results)}\n")
    for i, r in enumerate(results[:15], 1):
        print(
            f"{i:2d}. {r.name:45s}  n={r.trades:3d}  WR={r.win_rate:5.1f}%  "
            f"PF={r.profit_factor:4.2f}  ret={r.return_pct:6.2f}%  {r.exits}"
        )

    # Also show near-miss (48-50% WR, high trade count)
    print("\n--- Near-miss (48% <= WR < 50%, n >= 20) ---")
    near = []
    for name, fn in grid:
        r = backtest(name, fn, cache, config)
        if r.trades >= 20 and 48 <= r.win_rate < 50:
            near.append(r)
    near.sort(key=lambda x: (-x.trades, -x.win_rate))
    for i, r in enumerate(near[:10], 1):
        print(
            f"{i:2d}. {r.name:45s}  n={r.trades:3d}  WR={r.win_rate:5.1f}%  "
            f"PF={r.profit_factor:4.2f}  ret={r.return_pct:6.2f}%"
        )


if __name__ == "__main__":
    main()