"""Targeted last-1y push: 12-1 momentum, min-threshold, sticky leader, multi-year check."""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from trading.constants import NIFTY50_SYMBOL  # noqa: E402
from trading.services.fundamental_swing import (  # noqa: E402
    FundParams, load_cache, load_close_series, rank_at_date, build_close_tech,
    rebalance_dates, pick_at_date, run_window, DEFAULT_CAPITAL,
)
from trading.services.market_data import get_universe_symbols, load_price_dataframe  # noqa: E402

END = date.today()
WINDOWS = [
    ("1y", END - timedelta(days=365), END),
    ("2y", END - timedelta(days=365 * 2), END),
    ("3y", END - timedelta(days=365 * 3), END),
    ("23-24", date(2023, 9, 16), date(2024, 9, 16)),
    ("24-25", date(2024, 9, 16), date(2025, 9, 16)),
]


def main():
    t0 = time.time()
    cache = load_cache()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if (cache.get("stocks") or {}).get(s)]
    closes = load_close_series(symbols + [NIFTY50_SYMBOL])
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    calendar = pd.DatetimeIndex(pd.to_datetime(nifty_df.index))
    print(f"symbols={len(symbols)}", flush=True)

    packs = []
    # Hunt-1 winner
    packs.append(FundParams(
        name="mom21_tight", min_roe=12, min_profit_margin=5, min_revenue_growth=20,
        min_earnings_growth=25, max_pe=60, max_peg=3, max_de=1.5, min_score=40, top_n=1,
        rebalance="21d", rank_by="mom3", tech="sma150", trail_sma=0,
    ))
    # Hunt-1 runner-up diversified
    packs.append(FundParams(
        name="mom21_n2_loose", min_roe=10, min_profit_margin=4, min_revenue_growth=5,
        min_earnings_growth=8, max_pe=80, max_peg=5, max_de=2.0, min_score=30, top_n=2,
        rebalance="monthly", rank_by="mom3", tech="sma150", trail_sma=0,
    ))
    # Hunt-2 best (light + 10d ~ monthly-ish). 10d isn't a mode; use 21d + trail 20
    packs.append(FundParams(
        name="rs_trail20", min_roe=8, min_profit_margin=0, min_revenue_growth=0,
        min_earnings_growth=0, max_pe=120, max_peg=10, max_de=5, min_score=0,
        min_current_ratio=0, top_n=1,
        rebalance="21d", rank_by="mom3", tech="sma150", trail_sma=20,
    ))
    packs.append(FundParams(
        name="rs_n2_sma150", min_roe=8, min_profit_margin=0, min_revenue_growth=0,
        min_earnings_growth=0, max_pe=120, max_peg=10, max_de=5, min_score=0,
        min_current_ratio=0, top_n=2,
        rebalance="monthly", rank_by="mom3", tech="sma150", trail_sma=0,
    ))
    packs.append(FundParams(
        name="tight_n2_21d", min_roe=12, min_profit_margin=5, min_revenue_growth=20,
        min_earnings_growth=25, max_pe=60, max_peg=3, max_de=1.5, min_score=40, top_n=2,
        rebalance="21d", rank_by="mom3", tech="sma150", trail_sma=0,
    ))
    packs.append(FundParams(
        name="tight_trend_n1", min_roe=12, min_profit_margin=5, min_revenue_growth=20,
        min_earnings_growth=25, max_pe=60, max_peg=3, max_de=1.5, min_score=40, top_n=1,
        rebalance="21d", rank_by="mom3", tech="trend", trail_sma=0,
    ))

    for p in packs:
        bits = []
        for label, a, b in WINDOWS:
            r = run_window(cache, symbols, a, b, p, closes, calendar, capital=DEFAULT_CAPITAL)
            bits.append(f"{label}={r.get('total_return_pct')}%/{r.get('max_drawdown_pct')}dd wr={r.get('win_rate')} n={r.get('trades')}")
        print(f"{p.name:16}  " + "  ".join(bits), flush=True)
        # last 1y holdings
        r = run_window(cache, symbols, WINDOWS[0][1], WINDOWS[0][2], p, closes, calendar)
        print("  trades:", ", ".join(
            f"{t['symbol']} {t['return_pct']}%" for t in (r.get("holdings_history") or [])[:14]
        ), flush=True)

    print(f"done {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
