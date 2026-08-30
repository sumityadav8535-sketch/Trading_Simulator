"""
Load OHLCV from Django DailyPrice model into pandas DataFrames.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from trading.models import DailyPrice, Stock


def load_price_dataframe(
    symbol: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Return OHLCV DataFrame indexed by date (ascending).
    Columns: open, high, low, close, volume
    """
    qs = DailyPrice.objects.filter(stock_id=symbol).order_by("date")
    if start:
        qs = qs.filter(date__gte=start)
    if end:
        qs = qs.filter(date__lte=end)

    rows = list(qs.values("date", "open", "high", "low", "close", "volume"))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df


def get_universe_symbols(
    nifty200_only: bool = True,
    nifty100_only: bool = False,
    nifty_smallcap250_only: bool = False,
    watchlist_only: bool = False,
) -> list[str]:
    if watchlist_only:
        from trading.models import WatchlistItem

        return list(
            WatchlistItem.objects.select_related("stock")
            .filter(stock__is_active=True)
            .values_list("stock_id", flat=True)
        )
    qs = Stock.objects.filter(is_active=True)
    if nifty100_only:
        qs = qs.filter(is_nifty100=True)
    elif nifty_smallcap250_only:
        qs = qs.filter(is_nifty_smallcap250=True)
    elif nifty200_only:
        qs = qs.filter(is_nifty200=True)
    return list(qs.values_list("symbol", flat=True))


UNIVERSE_ALIASES = {
    "nifty200": "nifty200",
    "nifty_200": "nifty200",
    "nifty100": "nifty100",
    "nifty_100": "nifty100",
    "nifty_smallcap250": "nifty_smallcap250",
    "niftysmallcap250": "nifty_smallcap250",
    "smallcap250": "nifty_smallcap250",
    "smallcap_250": "nifty_smallcap250",
    "nifty smallcap 250": "nifty_smallcap250",
}


def resolve_universe_symbols(universe: str) -> list[str]:
    """Map a universe name (nifty200, nifty100, nifty_smallcap250) to symbols."""
    key = UNIVERSE_ALIASES.get((universe or "nifty200").strip().lower().replace("-", "_"), "nifty200")
    if key == "nifty100":
        return get_universe_symbols(nifty100_only=True)
    if key == "nifty_smallcap250":
        return get_universe_symbols(nifty_smallcap250_only=True)
    return get_universe_symbols(nifty200_only=True)