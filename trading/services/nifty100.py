"""
Nifty 100 universe — official NSE constituent list.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.request
from dataclasses import dataclass
from django.core.cache import cache

from trading.constants import NSE_NIFTY100_CSV_URL
from trading.models import Stock

logger = logging.getLogger(__name__)

CACHE_KEY = "nifty100:constituents"
CACHE_TTL = 86400  # 24 hours


@dataclass(frozen=True)
class Nifty100Constituent:
    symbol: str
    name: str
    industry: str


def _fetch_csv_text() -> str:
    req = urllib.request.Request(
        NSE_NIFTY100_CSV_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TradingSimulator/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def fetch_nifty100_constituents(force_refresh: bool = False) -> list[Nifty100Constituent]:
    """Download Nifty 100 constituents from NSE archives."""
    if not force_refresh:
        cached = cache.get(CACHE_KEY)
        if cached:
            return [Nifty100Constituent(**row) for row in cached]

    text = _fetch_csv_text()
    rows: list[Nifty100Constituent] = []
    for row in csv.DictReader(io.StringIO(text)):
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol:
            continue
        rows.append(
            Nifty100Constituent(
                symbol=symbol,
                name=(row.get("Company Name") or "").strip(),
                industry=(row.get("Industry") or "").strip(),
            )
        )

    cache.set(CACHE_KEY, [row.__dict__ for row in rows], CACHE_TTL)
    return rows


def ensure_nifty100_marked(force_refresh: bool = False) -> list[str]:
    """
    Mark Nifty 100 stocks in the database and return active symbols.
    Creates missing Stock rows so the intraday section always has a full universe.
    """
    constituents = fetch_nifty100_constituents(force_refresh=force_refresh)
    symbols = [c.symbol for c in constituents]
    meta = {c.symbol: c for c in constituents}

    Stock.objects.filter(is_nifty100=True).exclude(symbol__in=symbols).update(is_nifty100=False)

    existing = set(Stock.objects.filter(symbol__in=symbols).values_list("symbol", flat=True))
    missing = [s for s in symbols if s not in existing]
    if missing:
        Stock.objects.bulk_create(
            [
                Stock(
                    symbol=sym,
                    name=meta[sym].name,
                    sector=meta[sym].industry,
                    is_nifty100=True,
                    is_nifty200=True,
                    is_active=True,
                )
                for sym in missing
            ],
            ignore_conflicts=True,
        )

    Stock.objects.filter(symbol__in=symbols).update(is_nifty100=True)
    return symbols


def get_nifty100_symbols() -> list[str]:
    """Return Nifty 100 symbols from DB, refreshing from NSE when empty."""
    symbols = list(
        Stock.objects.filter(is_active=True, is_nifty100=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )
    if len(symbols) < 90:
        symbols = ensure_nifty100_marked()
    return symbols


def get_nifty100_metadata() -> dict[str, dict[str, str]]:
    """Symbol → {name, industry} for display."""
    constituents = fetch_nifty100_constituents()
    return {c.symbol: {"name": c.name, "industry": c.industry} for c in constituents}