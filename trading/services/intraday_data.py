"""
Live NSE intraday data for Nifty 100 via yfinance.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from django.core.cache import cache
from django.utils import timezone

from trading.constants import NIFTY100_INDEX_TICKER
from trading.services.nifty100 import ensure_nifty100_marked, get_nifty100_metadata
from trading.services.nse_price_sync import yfinance_ticker

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

VALID_INTERVALS = ("1m", "5m", "15m")
DEFAULT_INTERVAL = "5m"
QUOTES_CACHE_TTL = 30
CHART_CACHE_TTL = 30


@dataclass
class IntradayQuote:
    symbol: str
    name: str
    industry: str
    ltp: float
    prev_close: float
    change: float
    change_pct: float
    open: float
    high: float
    low: float
    volume: int
    last_bar_time: Optional[str] = None


@dataclass
class MarketStatus:
    is_open: bool
    status: str
    message: str
    now_ist: str
    session_date: str


def _now_ist() -> datetime:
    return datetime.now(IST)


def get_market_status(now: Optional[datetime] = None) -> MarketStatus:
    now = now or _now_ist()
    today = now.date()
    is_weekday = today.weekday() < 5
    t = now.time()

    if not is_weekday:
        return MarketStatus(
            is_open=False,
            status="closed",
            message="Market closed (weekend)",
            now_ist=now.strftime("%H:%M:%S IST"),
            session_date=today.isoformat(),
        )

    if t < MARKET_OPEN:
        return MarketStatus(
            is_open=False,
            status="pre_open",
            message="Pre-market — opens 9:15 AM IST",
            now_ist=now.strftime("%H:%M:%S IST"),
            session_date=today.isoformat(),
        )

    if t <= MARKET_CLOSE:
        return MarketStatus(
            is_open=True,
            status="open",
            message="Market open",
            now_ist=now.strftime("%H:%M:%S IST"),
            session_date=today.isoformat(),
        )

    return MarketStatus(
        is_open=False,
        status="closed",
        message="Market closed — post 3:30 PM IST",
        now_ist=now.strftime("%H:%M:%S IST"),
        session_date=today.isoformat(),
    )


def _safe_float(val) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).lower().replace(" ", "_")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = key.title()
    if rename:
        df = df.rename(columns=rename)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[keep] if keep else df


def _extract_ticker_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    sub: pd.DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        if ticker in level0:
            sub = df[ticker].copy()
        elif ticker in level1:
            sub = df.xs(ticker, axis=1, level=1).copy()
        elif "" in level0 or len(level0) == 1:
            sub = df.copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = level0
        else:
            return pd.DataFrame()
    else:
        sub = df.copy()

    sub = sub.dropna(how="all")
    if sub.empty:
        return pd.DataFrame()

    return _normalize_ohlcv_columns(sub)


def _download_intraday_bars(tickers: list[str], interval: str) -> pd.DataFrame:
    period = "1d" if interval in ("1m", "5m", "15m") else "5d"
    return yf.download(
        tickers,
        interval=interval,
        period=period,
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False,
    )


def _download_daily_closes(tickers: list[str]) -> pd.DataFrame:
    return yf.download(
        tickers,
        interval="1d",
        period="5d",
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False,
    )


def _prev_close_from_daily(df_daily: pd.DataFrame, ticker: str, session_date: date) -> float:
    sub = _extract_ticker_frame(df_daily, ticker)
    if sub.empty:
        return 0.0

    closes_by_date: dict[date, float] = {}
    for ts, row in sub.iterrows():
        d = ts.tz_convert(IST).date() if getattr(ts, "tzinfo", None) else ts.date()
        closes_by_date[d] = _safe_float(row.get("Close"))

    prior_dates = sorted(d for d in closes_by_date if d < session_date)
    if prior_dates:
        return closes_by_date[prior_dates[-1]]

    if len(sub) >= 2:
        return _safe_float(sub.iloc[-2]["Close"])
    return _safe_float(sub.iloc[-1]["Open"])


def _quote_from_bars(
    symbol: str,
    ticker: str,
    df_intraday: pd.DataFrame,
    df_daily: pd.DataFrame,
    meta: dict[str, dict[str, str]],
    session_date: date,
) -> Optional[IntradayQuote]:
    sub = _extract_ticker_frame(df_intraday, ticker)
    if sub.empty:
        return None

    sub = sub.dropna(subset=["Close"])
    if sub.empty:
        return None

    last = sub.iloc[-1]
    ltp = _safe_float(last["Close"])
    prev_close = _prev_close_from_daily(df_daily, ticker, session_date)
    if prev_close <= 0:
        prev_close = _safe_float(sub.iloc[0]["Open"]) or ltp

    change = ltp - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    info = meta.get(symbol, {})

    last_ts = sub.index[-1]
    if hasattr(last_ts, "tz_convert"):
        last_ts = last_ts.tz_convert(IST)
    bar_time = last_ts.strftime("%H:%M") if hasattr(last_ts, "strftime") else None

    return IntradayQuote(
        symbol=symbol,
        name=info.get("name", symbol),
        industry=info.get("industry", ""),
        ltp=round(ltp, 2),
        prev_close=round(prev_close, 2),
        change=round(change, 2),
        change_pct=round(change_pct, 2),
        open=round(_safe_float(sub.iloc[0]["Open"]), 2),
        high=round(_safe_float(sub["High"].max()), 2),
        low=round(_safe_float(sub["Low"].min()), 2),
        volume=int(sub["Volume"].fillna(0).sum()),
        last_bar_time=bar_time,
    )


def fetch_nifty100_quotes(interval: str = DEFAULT_INTERVAL, force_refresh: bool = False) -> dict:
    if interval not in VALID_INTERVALS:
        interval = DEFAULT_INTERVAL

    cache_key = f"intraday:quotes:{interval}:{date.today().isoformat()}"
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    symbols = ensure_nifty100_marked()
    meta = get_nifty100_metadata()
    tickers = [yfinance_ticker(s) for s in symbols]
    status = get_market_status()
    session_date = date.fromisoformat(status.session_date)

    try:
        df_intraday = _download_intraday_bars(tickers, interval)
        df_daily = _download_daily_closes(tickers)
    except Exception as exc:
        logger.exception("Intraday batch download failed")
        return {
            "error": str(exc),
            "quotes": [],
            "market": asdict(status),
            "updated_at": timezone.now().isoformat(),
            "interval": interval,
            "symbol_count": 0,
        }

    quotes: list[IntradayQuote] = []
    for symbol in symbols:
        ticker = yfinance_ticker(symbol)
        quote = _quote_from_bars(symbol, ticker, df_intraday, df_daily, meta, session_date)
        if quote:
            quotes.append(quote)

    quotes.sort(key=lambda q: q.change_pct, reverse=True)

    payload = {
        "quotes": [asdict(q) for q in quotes],
        "market": asdict(status),
        "updated_at": timezone.now().isoformat(),
        "interval": interval,
        "symbol_count": len(quotes),
        "advancers": sum(1 for q in quotes if q.change_pct > 0),
        "decliners": sum(1 for q in quotes if q.change_pct < 0),
        "unchanged": sum(1 for q in quotes if q.change_pct == 0),
    }
    cache.set(cache_key, payload, QUOTES_CACHE_TTL)
    return payload


def fetch_index_snapshot() -> dict:
    cache_key = f"intraday:index:{date.today().isoformat()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    status = get_market_status()
    result = {
        "nifty100": {},
        "market": asdict(status),
        "updated_at": timezone.now().isoformat(),
    }

    try:
        ticker = yf.Ticker(NIFTY100_INDEX_TICKER)
        fi = ticker.fast_info
        ltp = _safe_float(getattr(fi, "last_price", 0))
        prev = _safe_float(getattr(fi, "previous_close", 0))
        change = ltp - prev if prev else 0.0
        change_pct = (change / prev * 100) if prev else 0.0
        result["nifty100"] = {
            "name": "Nifty 100",
            "ltp": round(ltp, 2),
            "prev_close": round(prev, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "open": round(_safe_float(getattr(fi, "open", 0)), 2),
            "high": round(_safe_float(getattr(fi, "day_high", 0)), 2),
            "low": round(_safe_float(getattr(fi, "day_low", 0)), 2),
        }
    except Exception as exc:
        logger.warning("Nifty 100 index snapshot failed: %s", exc)
        hist = yf.Ticker(NIFTY100_INDEX_TICKER).history(interval="5m", period="1d")
        if not hist.empty:
            ltp = _safe_float(hist.iloc[-1]["Close"])
            prev = _safe_float(hist.iloc[0]["Open"])
            change = ltp - prev
            change_pct = (change / prev * 100) if prev else 0.0
            result["nifty100"] = {
                "name": "Nifty 100",
                "ltp": round(ltp, 2),
                "prev_close": round(prev, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "open": round(_safe_float(hist.iloc[0]["Open"]), 2),
                "high": round(_safe_float(hist["High"].max()), 2),
                "low": round(_safe_float(hist["Low"].min()), 2),
            }

    cache.set(cache_key, result, QUOTES_CACHE_TTL)
    return result


def fetch_symbol_intraday_bars(symbol: str, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        interval = DEFAULT_INTERVAL

    ticker = yfinance_ticker(symbol.upper())
    df = yf.download(
        ticker,
        interval=interval,
        period="1d",
        progress=False,
        auto_adjust=False,
    )
    sub = _extract_ticker_frame(df, ticker)
    if sub.empty or "Close" not in sub.columns:
        sub = _normalize_ohlcv_columns(df.copy())
    if "Close" not in sub.columns:
        return pd.DataFrame()
    return sub.dropna(subset=["Close"])


def build_intraday_chart_json(symbol: str, interval: str = DEFAULT_INTERVAL) -> str:
    import json

    import plotly.graph_objects as go

    cache_key = f"intraday:chart:{symbol}:{interval}:{date.today().isoformat()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    df = fetch_symbol_intraday_bars(symbol, interval)
    if df.empty:
        payload = json.dumps({"data": [], "layout": {"title": f"No intraday data for {symbol}"}})
        cache.set(cache_key, payload, CHART_CACHE_TTL)
        return payload

    idx = df.index
    if hasattr(idx, "tz_convert"):
        idx = idx.tz_convert(IST)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        )
    )
    fig.add_trace(
        go.Bar(
            x=idx,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(100,116,139,0.45)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"{symbol} — {interval} intraday (NSE)",
        height=420,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(title="Price (₹)", side="left"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12),
    )

    payload = fig.to_json()
    cache.set(cache_key, payload, CHART_CACHE_TTL)
    return payload