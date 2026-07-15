"""
Load OHLCV for V2 analysis from local DB or yfinance.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from stage_analysis.services.stage_detector import daily_to_weekly, normalize_ticker
from trading.constants import NIFTY50_SYMBOL
from trading.models import Stock
from trading.services.market_data import load_price_dataframe
from trading.services.nse_price_sync import nse_symbol_from_ticker, yfinance_ticker


def _yf_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def load_daily(symbol: str) -> pd.DataFrame:
    """Daily OHLCV — DB first (NSE symbols), yfinance fallback."""
    sym = normalize_ticker(symbol)
    base = nse_symbol_from_ticker(sym) if sym.endswith(".NS") else sym

    if sym == NIFTY50_SYMBOL or base == NIFTY50_SYMBOL:
        df = load_price_dataframe(NIFTY50_SYMBOL)
        if not df.empty:
            return df
        return _yf_history("^NSEI")

    db_sym = base if Stock.objects.filter(pk=base).exists() else None
    if db_sym:
        df = load_price_dataframe(db_sym)
        if not df.empty and len(df) >= 200:
            return df

    yf_sym = sym if "." in sym else yfinance_ticker(base)
    return _yf_history(yf_sym)


def load_weekly(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol)
    if daily.empty:
        yf_sym = symbol if "." in symbol else (
            "^NSEI" if symbol == NIFTY50_SYMBOL else yfinance_ticker(nse_symbol_from_ticker(symbol))
        )
        raw = _yf_history(yf_sym, interval="1wk")
        if raw.empty:
            raise ValueError(f"No weekly data for {symbol}")
        return raw
    weekly = daily_to_weekly(daily)
    if weekly.empty:
        raise ValueError(f"No weekly data for {symbol}")
    return weekly


def stock_meta(symbol: str) -> tuple[str, str]:
    sym = normalize_ticker(symbol)
    base = nse_symbol_from_ticker(sym)
    try:
        stock = Stock.objects.get(pk=base)
        return stock.name or base, stock.sector or ""
    except Stock.DoesNotExist:
        try:
            yt = yf.Ticker(sym if "." in sym else yfinance_ticker(base))
            info = yt.info or {}
            return info.get("shortName") or info.get("longName") or sym, info.get("sector") or ""
        except Exception:
            return sym, ""