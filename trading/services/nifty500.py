"""
Nifty 500 universe — official NSE constituent list.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from trading.constants import NSE_NIFTY500_CSV_URL
from trading.models import Stock

logger = logging.getLogger(__name__)

CACHE_KEY = "nifty500:constituents"
CACHE_TTL = 86400
FALLBACK_CSV = Path(settings.BASE_DIR) / "data" / "nifty500_constituents.csv"


@dataclass(frozen=True)
class Nifty500Constituent:
    symbol: str
    name: str
    industry: str


def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; TradingSimulator/1.0)",
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }


def parse_constituents_csv(text: str) -> list[Nifty500Constituent]:
    rows: list[Nifty500Constituent] = []
    for row in csv.DictReader(io.StringIO(text)):
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol:
            continue
        rows.append(
            Nifty500Constituent(
                symbol=symbol,
                name=(row.get("Company Name") or "").strip(),
                industry=(row.get("Industry") or "").strip(),
            )
        )
    return rows


def _fetch_csv_text() -> str:
    req = urllib.request.Request(NSE_NIFTY500_CSV_URL, headers=_request_headers())
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def fetch_nifty500_constituents(force_refresh: bool = False) -> list[Nifty500Constituent]:
    if not force_refresh:
        cached = cache.get(CACHE_KEY)
        if cached:
            return [Nifty500Constituent(**row) for row in cached]

    text = ""
    try:
        text = _fetch_csv_text()
        FALLBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_CSV.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.warning("NSE Nifty 500 CSV fetch failed: %s", exc)
        if FALLBACK_CSV.exists():
            text = FALLBACK_CSV.read_text(encoding="utf-8")
        elif not force_refresh:
            raise

    rows = parse_constituents_csv(text)
    if not rows:
        raise RuntimeError("Nifty 500 constituent list is empty")

    cache.set(CACHE_KEY, [row.__dict__ for row in rows], CACHE_TTL)
    return rows


def ensure_nifty500_marked(force_refresh: bool = False) -> list[str]:
    constituents = fetch_nifty500_constituents(force_refresh=force_refresh)
    symbols = [c.symbol for c in constituents]
    meta = {c.symbol: c for c in constituents}

    Stock.objects.filter(is_nifty500=True).exclude(symbol__in=symbols).update(is_nifty500=False)

    existing = set(Stock.objects.filter(symbol__in=symbols).values_list("symbol", flat=True))
    missing = [s for s in symbols if s not in existing]
    if missing:
        Stock.objects.bulk_create(
            [
                Stock(
                    symbol=sym,
                    name=meta[sym].name,
                    sector=meta[sym].industry,
                    is_nifty500=True,
                    is_active=True,
                )
                for sym in missing
            ],
            ignore_conflicts=True,
        )

    Stock.objects.filter(symbol__in=symbols).update(is_nifty500=True)
    blank = list(Stock.objects.filter(symbol__in=symbols, name=""))
    for stock in blank:
        c = meta.get(stock.symbol)
        if c:
            stock.name = c.name
            stock.sector = c.industry
    if blank:
        Stock.objects.bulk_update(blank, ["name", "sector"])
    return symbols


def get_nifty500_symbols() -> list[str]:
    symbols = list(
        Stock.objects.filter(is_active=True, is_nifty500=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )
    if len(symbols) < 400:
        symbols = ensure_nifty500_marked()
    return symbols
