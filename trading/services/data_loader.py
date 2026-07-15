"""
Import NSE OHLCV from user's legacy SQLite export into Django models.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from django.conf import settings
from django.db import transaction

from trading.models import DailyPrice, Stock

logger = logging.getLogger(__name__)


def load_from_legacy_sqlite(
    db_path: Path | None = None,
    mark_all_nifty200: bool = True,
    batch_size: int = 5000,
) -> dict:
    """
    Load stocks and prices from data/nse_data.sqlite.
    Returns counts: {stocks_created, stocks_updated, prices_upserted}
    """
    path = db_path or settings.NSE_LEGACY_DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"NSE data not found at {path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    stocks_created = stocks_updated = 0
    prices_upserted = 0

    with transaction.atomic():
        for row in conn.execute("SELECT * FROM stocks ORDER BY symbol"):
            meta = {}
            if row["fundamental_meta"]:
                try:
                    meta = json.loads(row["fundamental_meta"])
                except json.JSONDecodeError:
                    meta = {}

            defaults = {
                "name": row["name"] or row["symbol"],
                "sector": row["sector"] or "",
                "is_active": bool(row["is_active"]),
                "is_nifty200": mark_all_nifty200,
                "last_price": row["last_price"],
                "avg_volume_20d": row["avg_volume_20d"] or 0,
                "sales_growth_yoy": meta.get("revenue_growth_yoy"),
                "profit_growth": meta.get("profit_growth"),
                "roe_5yr_avg": meta.get("roe"),
                "debt_equity": meta.get("debt_equity"),
                "peg": meta.get("peg"),
                "institutional_interest": meta.get("institutional_interest", ""),
            }
            _, created = Stock.objects.update_or_create(symbol=row["symbol"], defaults=defaults)
            if created:
                stocks_created += 1
            else:
                stocks_updated += 1

        price_rows = conn.execute(
            "SELECT symbol, date, open, high, low, close, volume FROM prices ORDER BY symbol, date"
        ).fetchall()

        buffer = []
        for prow in price_rows:
            if not Stock.objects.filter(pk=prow["symbol"]).exists():
                continue
            buffer.append(
                DailyPrice(
                    stock_id=prow["symbol"],
                    date=prow["date"],
                    open=prow["open"],
                    high=prow["high"],
                    low=prow["low"],
                    close=prow["close"],
                    volume=prow["volume"] or 0,
                )
            )
            if len(buffer) >= batch_size:
                prices_upserted += _flush_prices(buffer)
                buffer = []
        if buffer:
            prices_upserted += _flush_prices(buffer)

    conn.close()
    stats = {
        "stocks_created": stocks_created,
        "stocks_updated": stocks_updated,
        "prices_upserted": prices_upserted,
    }
    logger.info("NSE load complete: %s", stats)
    return stats


def _flush_prices(buffer: list[DailyPrice]) -> int:
    DailyPrice.objects.bulk_create(
        buffer,
        update_conflicts=True,
        unique_fields=["stock", "date"],
        update_fields=["open", "high", "low", "close", "volume"],
    )
    return len(buffer)