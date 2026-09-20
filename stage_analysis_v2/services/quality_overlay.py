"""
Optional quality overlay for Stage 2.0 / RS 70 pack.

Technical legs (Nifty vs SMA150, stock not stretched above SMA150) are
point-in-time from daily bars.

PE and profit-margin use a cached Yahoo snapshot (latest available, not
point-in-time). Missing numbers do not reject a name.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

CACHE_NAME = "stock_quality_fundamentals.json"

# Researched overlay (RS 70 losers: NYKAA / BIOCON / YESBANK / Jan-26 cluster)
DEFAULT_MAX_PCT_ABOVE_MA = 15.0
DEFAULT_NIFTY_SMA_PERIOD = 150
DEFAULT_MIN_PROFIT_MARGIN = 8.0
DEFAULT_MAX_PE = 50.0


def _in_tests() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return any(arg == "test" or arg.endswith("test") for arg in sys.argv)


def _cache_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / CACHE_NAME


def load_quality_fundamentals(
    symbols: list[str],
    *,
    fetch_missing: bool = True,
) -> dict[str, dict]:
    """Return {symbol: {pe, profit_margin}} (margin in percent)."""
    path = _cache_path()
    cache: dict[str, dict] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cache = raw
        except (OSError, json.JSONDecodeError):
            cache = {}

    missing = [s for s in symbols if s not in cache]
    if fetch_missing and missing and not _in_tests():
        try:
            import yfinance as yf
            from trading.services.nse_price_sync import yfinance_ticker
        except Exception:
            yf = None  # type: ignore
            yfinance_ticker = None  # type: ignore
        if yf is not None:
            for sym in missing:
                try:
                    info = yf.Ticker(yfinance_ticker(sym)).info or {}
                except Exception as exc:
                    logger.info("Quality overlay fetch failed for %s: %s", sym, exc)
                    cache[sym] = {"pe": None, "profit_margin": None}
                    continue
                pe = info.get("trailingPE")
                mgn = info.get("profitMargins")
                try:
                    pe_f = float(pe) if pe is not None else None
                except (TypeError, ValueError):
                    pe_f = None
                try:
                    mgn_f = float(mgn) * 100.0 if mgn is not None else None
                except (TypeError, ValueError):
                    mgn_f = None
                cache[sym] = {"pe": pe_f, "profit_margin": mgn_f}
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
            except OSError:
                logger.warning("Could not write quality overlay cache %s", path)

    return {s: dict(cache.get(s) or {}) for s in symbols}


def passes_pe_margin(
    symbol: str,
    fundamentals: dict[str, dict],
    *,
    max_pe: float = 0.0,
    min_profit_margin: float = 0.0,
) -> bool:
    row = fundamentals.get(symbol) or {}
    if max_pe and max_pe > 0:
        pe = row.get("pe")
        try:
            if pe is not None and float(pe) > float(max_pe):
                return False
        except (TypeError, ValueError):
            pass
    if min_profit_margin and min_profit_margin > 0:
        mgn = row.get("profit_margin")
        try:
            if mgn is not None and float(mgn) < float(min_profit_margin):
                return False
        except (TypeError, ValueError):
            pass
    return True


def index_above_sma(daily, week_end, period: int) -> bool:
    """True if index close on/before week_end is above SMA(period)."""
    if daily is None or getattr(daily, "empty", True) or period <= 0:
        return True
    d = daily.loc[daily.index <= week_end]
    if len(d) < period:
        return True
    close = float(d["close"].iloc[-1])
    ma = float(d["close"].astype(float).rolling(int(period)).mean().iloc[-1])
    if ma <= 0 or close != close or ma != ma:
        return True
    return close >= ma
