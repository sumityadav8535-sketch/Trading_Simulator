"""
Incremental NSE equity OHLCV sync via yfinance (.NS tickers).

Fills gaps from the last stored bar through the latest available trading day.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from django.conf import settings
from django.db.models import Max, Min

from trading.constants import NIFTY50_SYMBOL
from trading.models import DailyPrice, Stock
from trading.services.market_data import get_universe_symbols

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
YFINANCE_SUFFIX = ".NS"
BACKFILL_DAYS = 365 * 5
# NSE symbol → Yahoo ticker when the .NS name was renamed / delisted.
YFINANCE_TICKER_OVERRIDES = {
    "TATAMOTORS": "TMPV.NS",  # continuous Tata Motors series after PV/CV split
    "TMCV": "TMCV.NS",
    "TMPV": "TMPV.NS",
    "ZOMATO": "ETERNAL.NS",  # Zomato rebranded to Eternal
}

_sync_lock = threading.Lock()
_sync_started = False


def yfinance_ticker(symbol: str) -> str:
    return YFINANCE_TICKER_OVERRIDES.get(symbol.upper(), f"{symbol}{YFINANCE_SUFFIX}")


def nse_symbol_from_ticker(ticker: str) -> str:
    return ticker.removesuffix(YFINANCE_SUFFIX)


def expected_latest_bar_date(now: Optional[datetime] = None) -> date:
    """Last NSE trading day we expect EOD data for (IST, after ~4 PM)."""
    now = now or datetime.now(IST)
    d = now.date()
    if d.weekday() < 5 and now.hour >= 16:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def get_symbol_last_date(symbol: str) -> Optional[date]:
    return DailyPrice.objects.filter(stock_id=symbol).aggregate(m=Max("date"))["m"]


def get_symbol_first_date(symbol: str) -> Optional[date]:
    return DailyPrice.objects.filter(stock_id=symbol).aggregate(m=Min("date"))["m"]


def active_equity_symbols(nifty200_only: bool = True) -> list[str]:
    """Equities included in daily price sync.

    When nifty200_only is True, include both Nifty 200 and Nifty Smallcap 250
    so both universes stay current. Pass nifty200_only=False for every active stock.
    """
    symbols = get_universe_symbols(nifty200_only=nifty200_only)
    if nifty200_only:
        extra = get_universe_symbols(nifty_smallcap250_only=True)
        seen = set(symbols)
        for sym in extra:
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)
    return [s for s in symbols if s != NIFTY50_SYMBOL]


def is_universe_stale(symbols: Optional[list[str]] = None) -> bool:
    symbols = symbols or active_equity_symbols()
    if not symbols:
        return False
    expected = expected_latest_bar_date()
    for sym in symbols:
        last = get_symbol_last_date(sym)
        if last is None or last < expected:
            return True
    return False


def _bulk_upsert(buffer: list[DailyPrice]) -> int:
    DailyPrice.objects.bulk_create(
        buffer,
        update_conflicts=True,
        unique_fields=["stock", "date"],
        update_fields=["open", "high", "low", "close", "volume"],
    )
    return len(buffer)


def _upsert_bars(symbol: str, frame: pd.DataFrame) -> int:
    count = 0
    buffer: list[DailyPrice] = []
    for idx, row in frame.iterrows():
        ts = pd.Timestamp(idx)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(IST).tz_localize(None)
        d = ts.date()
        buffer.append(
            DailyPrice(
                stock_id=symbol,
                date=d,
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row.get("Volume", 0) or 0),
            )
        )
        if len(buffer) >= 500:
            count += _bulk_upsert(buffer)
            buffer = []
    if buffer:
        count += _bulk_upsert(buffer)
    return count


def _update_stock_meta(symbol: str, frame: pd.DataFrame) -> None:
    last_close = round(float(frame.iloc[-1]["Close"]), 2)
    vol_tail = frame["Volume"].tail(20).astype(float)
    avg_vol = int(vol_tail.mean()) if len(vol_tail) else 0
    Stock.objects.filter(pk=symbol).update(last_price=last_close, avg_volume_20d=avg_vol)


def _parse_download_frame(
    data: pd.DataFrame,
    tickers: list[str],
    ticker_to_symbol: Optional[dict[str, str]] = None,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if data.empty:
        return result
    ticker_to_symbol = ticker_to_symbol or {}

    def _sym(ticker: str) -> str:
        return ticker_to_symbol.get(ticker, nse_symbol_from_ticker(ticker))

    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in available:
                continue
            sub = data[ticker].dropna(how="all")
            if not sub.empty:
                result[_sym(ticker)] = sub
    else:
        sub = data.dropna(how="all")
        if not sub.empty:
            result[_sym(tickers[0])] = sub
    return result


def sync_symbols(
    symbols: list[str],
    *,
    batch_size: Optional[int] = None,
    pause_seconds: float = 0.4,
    force: bool = False,
    backfill_from: Optional[date] = None,
) -> dict:
    """
    Incrementally download and upsert OHLCV for NSE symbols.
    Returns stats: symbols_checked, symbols_updated, bars_upserted, errors.

    backfill_from: also fetch history from this date (fills years before the
    existing 5y store). Overlap dates are upserted so splits stay consistent.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance: pip install yfinance") from exc

    batch_size = batch_size or getattr(settings, "NSE_SYNC_BATCH_SIZE", 25)
    end = date.today() + timedelta(days=1)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]

    stats = {
        "symbols_checked": 0,
        "symbols_updated": 0,
        "bars_upserted": 0,
        "errors": [],
    }

    pending: list[tuple[str, date]] = []
    expected = expected_latest_bar_date()

    for sym in symbols:
        stats["symbols_checked"] += 1
        last = get_symbol_last_date(sym)
        first = get_symbol_first_date(sym)
        if backfill_from is not None:
            # Need older history, a forward gap, or a forced refresh.
            needs_older = first is None or first > backfill_from
            needs_newer = last is None or last < expected
            if not needs_older and not needs_newer and not force:
                continue
            start = backfill_from
        else:
            if not force and last is not None and last >= expected:
                continue
            start = (last + timedelta(days=1)) if last else (date.today() - timedelta(days=BACKFILL_DAYS))
        pending.append((sym, start))

    if not pending:
        logger.info("NSE equity prices already up to date (%s)", expected)
        return stats

    logger.info("Syncing %s symbols (target through %s)", len(pending), expected)

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        batch_start = min(start for _, start in batch)
        yf_tickers = [yfinance_ticker(sym) for sym, _ in batch]
        ticker_to_symbol = {yfinance_ticker(sym): sym for sym, _ in batch}
        ticker_str = " ".join(yf_tickers)

        try:
            raw = yf.download(
                ticker_str,
                start=batch_start.isoformat(),
                end=end.isoformat(),
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            msg = f"batch {i // batch_size + 1}: {exc}"
            logger.warning("NSE download failed: %s", msg)
            stats["errors"].append(msg)
            continue

        frames = _parse_download_frame(raw, yf_tickers, ticker_to_symbol)

        for sym, start in batch:
            frame = frames.get(sym)
            if frame is None or frame.empty:
                continue
            last = get_symbol_last_date(sym)
            if backfill_from is None and last:
                cutoff = pd.Timestamp(last)
                frame = frame[frame.index > cutoff]
            elif start:
                frame = frame[frame.index >= pd.Timestamp(start)]
            if frame.empty:
                continue
            n = _upsert_bars(sym, frame)
            if n:
                stats["symbols_updated"] += 1
                stats["bars_upserted"] += n
                _update_stock_meta(sym, frame)

        if i + batch_size < len(pending) and pause_seconds:
            time.sleep(pause_seconds)

    return stats


def sync_universe(
    *,
    nifty200_only: bool = True,
    include_index: bool = True,
    force: bool = False,
    backfill_from: Optional[date] = None,
    **kwargs,
) -> dict:
    """Sync all active equities and optionally the Nifty 50 index."""
    stats = sync_symbols(
        active_equity_symbols(nifty200_only=nifty200_only),
        force=force,
        backfill_from=backfill_from,
        **kwargs,
    )

    if include_index:
        try:
            from trading.services.nifty50_index import sync_nifty50_from_yfinance

            if backfill_from is not None:
                years = max(1, (date.today() - backfill_from).days // 365 + 1)
            else:
                years = 1
            stats["nifty50_bars"] = sync_nifty50_from_yfinance(years=years)
        except Exception as exc:
            logger.warning("Nifty 50 sync failed: %s", exc)
            stats["errors"].append(f"NIFTY50: {exc}")

    return stats


def schedule_auto_sync() -> None:
    """Run a background sync once per process if data is stale."""
    global _sync_started

    if not getattr(settings, "NSE_AUTO_SYNC_ENABLED", True):
        return

    with _sync_lock:
        if _sync_started:
            return
        _sync_started = True

    def _worker() -> None:
        try:
            if not is_universe_stale():
                logger.info("NSE data is current; auto-sync skipped")
                return
            logger.info("Auto-syncing NSE prices from yfinance...")
            stats = sync_universe()
            logger.info(
                "NSE auto-sync complete: %s symbols updated, %s bars upserted",
                stats.get("symbols_updated", 0),
                stats.get("bars_upserted", 0),
            )
            if stats.get("errors"):
                logger.warning("NSE auto-sync errors: %s", stats["errors"])
        except Exception:
            logger.exception("Automatic NSE price sync failed")

    threading.Thread(target=_worker, daemon=True, name="nse-auto-sync").start()