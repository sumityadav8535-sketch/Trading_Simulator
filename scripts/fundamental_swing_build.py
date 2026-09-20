"""Fetch Yahoo fundamentals for Nifty 200 and build Fundamental Compounders results."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from trading.services.fundamental_swing import (  # noqa: E402
    DEFAULT_PARAMS,
    build_results,
    refresh_cache,
    save_results,
)
from trading.services.market_data import get_universe_symbols  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch Yahoo even if cached")
    parser.add_argument("--skip-fetch", action="store_true", help="Rebuild from existing cache only")
    args = parser.parse_args()

    symbols = get_universe_symbols(nifty200_only=True)
    print(f"Universe: {len(symbols)} Nifty 200 symbols", flush=True)
    if not args.skip_fetch:
        print("Fetching Yahoo info + annual statements…", flush=True)
        cache = refresh_cache(symbols, force=args.force, progress=True)
    else:
        from trading.services.fundamental_swing import load_cache

        cache = load_cache()
    cached = len(cache.get("stocks") or {})
    print(f"Cache has {cached} names. Building backtests…", flush=True)
    payload = build_results(cache, symbols=symbols, params=DEFAULT_PARAMS, end=date.today())
    path = save_results(payload)
    print(f"Wrote {path}", flush=True)
    print(f"Picks: {[p['symbol'] for p in payload.get('picks') or []]}", flush=True)
    for w in payload.get("windows") or []:
        print(
            f"  {w.get('title')}: {w.get('total_return_pct')}%  "
            f"CAGR {w.get('cagr_pct')}%  DD {w.get('max_drawdown_pct')}%  "
            f"Nifty {w.get('benchmark_pct')}%  hit100={w.get('hit_100')}",
            flush=True,
        )
    print("Year slices:", flush=True)
    for y in payload.get("year_rows") or []:
        print(
            f"  {y.get('label')}: {y.get('return_pct')}%  "
            f"nifty {y.get('benchmark_pct')}%  hit100={y.get('hit_100')}  "
            f"best={((y.get('best_stock_year') or {}) or {}).get('symbol')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
