"""Load Elite ML Short v2 F&O backtest results and trade log."""
from __future__ import annotations

import json
from pathlib import Path

from django.core.cache import cache

ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_PATH = ROOT / "data" / "intraday_fno_optimized.json"
TRADES_PATH = ROOT / "data" / "intraday_fno_optimized_trades.json"
MODEL_META_PATH = ROOT / "data" / "intraday_fno_ml_model.json"

_CACHE_TTL = 300


def _load_json(path: Path, cache_prefix: str) -> dict:
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cache_key = f"{cache_prefix}:{mtime}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    data = json.loads(path.read_text(encoding="utf-8"))
    cache.set(cache_key, data, _CACHE_TTL)
    return data


def load_strategy_results() -> dict:
    return _load_json(RESULTS_PATH, "fno:results")


def load_strategy_trades_data() -> dict:
    return _load_json(TRADES_PATH, "fno:trades")


def load_model_meta() -> dict:
    return _load_json(MODEL_META_PATH, "fno:model_meta")


def get_trades(period: str = "full") -> list[dict]:
    data = load_strategy_trades_data()
    if not data:
        return []
    if period == "oos":
        return data.get("oos", {}).get("trades", [])
    return data.get("full_period", {}).get("trades", [])