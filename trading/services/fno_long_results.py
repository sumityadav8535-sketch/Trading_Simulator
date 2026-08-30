"""Load Elite ML Long v1 backtest results and model meta."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from django.core.cache import cache

ROOT = Path(__file__).resolve().parent.parent.parent
IST = ZoneInfo("Asia/Kolkata")

RESULTS_PATH = ROOT / "data" / "intraday_fno_ml_long_results.json"
TRADES_PATH = ROOT / "data" / "intraday_fno_ml_long_trades.json"
MODEL_META_PATH = ROOT / "data" / "intraday_fno_ml_long_model.json"
FNO_DIR = ROOT / "data" / "intraday_fno"

_CACHE_TTL = 300


def _load_json(path: Path, cache_prefix: str):
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
    return _load_json(RESULTS_PATH, "fno_long:results")


def load_strategy_trades_data() -> dict:
    """Return structured trades payload (full + oos when available)."""
    results = load_strategy_results()
    if results:
        return {
            "full_period": results.get("full_period") or {},
            "oos": results.get("oos_period") or results.get("oos") or {},
            "selected": results.get("selected"),
            "data_range": results.get("data_range"),
            "filters": results.get("filters"),
        }
    # Flat trade list fallback
    trades = _load_json(TRADES_PATH, "fno_long:trades")
    if isinstance(trades, list):
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        return {
            "full_period": {
                "trades": trades,
                "wins": wins,
                "losses": len(trades) - wins,
                "summary": {},
            },
            "oos": {},
        }
    return trades if isinstance(trades, dict) else {}


def load_model_meta() -> dict:
    return _load_json(MODEL_META_PATH, "fno_long:model_meta")


def get_trades(period: str = "full") -> list[dict]:
    data = load_strategy_trades_data()
    if not data:
        # try flat trades file
        raw = _load_json(TRADES_PATH, "fno_long:trades_flat")
        if isinstance(raw, list):
            return raw
        return []
    if period == "oos":
        oos = data.get("oos") or data.get("oos_period") or {}
        return oos.get("trades") or []
    fp = data.get("full_period") or {}
    trades = fp.get("trades")
    if trades:
        return trades
    raw = _load_json(TRADES_PATH, "fno_long:trades_flat2")
    return raw if isinstance(raw, list) else []


def get_recent_day_results(trade_period: str = "full", instrument: str = "NIFTY") -> dict:
    """Yesterday / last trade day snapshot (backtest trades only for long)."""
    trades = get_trades(trade_period)
    today = datetime.now(IST).date()
    yesterday = today - __import__("datetime").timedelta(days=1)

    def _day_bucket(d: date) -> dict:
        day_trades = [
            t for t in trades
            if str(t.get("session_date") or "")[:10] == d.isoformat()
        ]
        wins = sum(1 for t in day_trades if t.get("result") == "WIN")
        losses = len(day_trades) - wins
        pnl = sum(float(t.get("pnl_inr") or 0) for t in day_trades)
        if day_trades:
            status = "traded"
            note = f"{len(day_trades)} trade(s)"
        else:
            status = "flat"
            note = "No backtest trades this day"
        return {
            "label": d.isoformat(),
            "date": d.isoformat(),
            "status": status,
            "note": note,
            "count": len(day_trades),
            "wins": wins,
            "losses": losses,
            "pnl": round(pnl, 2),
            "trades": day_trades,
            "paper_count": 0,
            "paper_wins": 0,
            "paper_losses": 0,
            "paper_pnl": 0,
            "paper_trades": [],
        }

    last_trade_date = None
    if trades:
        dates = sorted({str(t.get("session_date") or "")[:10] for t in trades if t.get("session_date")})
        last_trade_date = dates[-1] if dates else None

    return {
        "yesterday": _day_bucket(yesterday),
        "today": _day_bucket(today),
        "last_trade_date": last_trade_date,
    }
