"""
Nifty 50 index data — fetch via yfinance (^NSEI) and persist to DailyPrice.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.models import DailyPrice, Stock
from trading.services.indicators import compute_indicators
from trading.services.market_data import load_price_dataframe

logger = logging.getLogger(__name__)

YFINANCE_TICKER = "^NSEI"
MIN_BARS = 220


def ensure_nifty50_stock() -> Stock:
    stock, _ = Stock.objects.get_or_create(
        symbol=NIFTY50_SYMBOL,
        defaults={
            "name": "Nifty 50 Index",
            "sector": "Index",
            "is_active": True,
            "is_nifty200": False,
        },
    )
    return stock


def sync_nifty50_from_yfinance(years: int = 5) -> int:
    """Download Nifty 50 OHLCV and upsert into DailyPrice. Returns rows upserted."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance: pip install yfinance") from exc

    ensure_nifty50_stock()
    end = date.today()
    start = end - timedelta(days=years * 365)

    ticker = yf.Ticker(YFINANCE_TICKER)
    raw = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=True)
    if raw.empty:
        logger.warning("No Nifty 50 data returned from yfinance")
        return 0

    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col]).dt.tz_localize(None)

    upserted = 0
    for _, row in raw.iterrows():
        d = row[date_col].date()
        DailyPrice.objects.update_or_create(
            stock_id=NIFTY50_SYMBOL,
            date=d,
            defaults={
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0) or 0),
            },
        )
        upserted += 1

    last_close = float(raw.iloc[-1]["Close"])
    Stock.objects.filter(pk=NIFTY50_SYMBOL).update(last_price=last_close)
    logger.info("Synced %s Nifty 50 bars (last close %.2f)", upserted, last_close)
    return upserted


def load_nifty50_frame(auto_sync: bool = True) -> pd.DataFrame:
    """Load Nifty 50 with indicators; auto-fetch from yfinance if DB is empty."""
    df = load_price_dataframe(NIFTY50_SYMBOL)
    if (df.empty or len(df) < MIN_BARS) and auto_sync:
        try:
            sync_nifty50_from_yfinance()
            df = load_price_dataframe(NIFTY50_SYMBOL)
        except Exception as exc:
            logger.warning("Nifty 50 auto-sync failed: %s", exc)
    if df.empty:
        return df
    return compute_indicators(df)