"""
VWAP Extension Fade — 15-minute Nifty 200 mean-reversion.

The only family that stayed profitable after a broad 15m tournament
(breakouts, ORB, EMA momentum, gap-and-go all lost money in May–Aug 2026 chop).

Signal on the 15m close, enter the next bar open.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings

RESULTS = Path(settings.BASE_DIR) / "data" / "intraday_15m_fade_results.json"
TRADES = Path(settings.BASE_DIR) / "data" / "intraday_15m_fade_trades.json"
HUNT = Path(settings.BASE_DIR) / "data" / "intraday_15m_hunt.json"
HUNT_TRADES = Path(settings.BASE_DIR) / "data" / "intraday_15m_hunt_trades.json"
PACK100 = Path(settings.BASE_DIR) / "data" / "intraday_100pct.json"
PACK100_TRADES = Path(settings.BASE_DIR) / "data" / "intraday_100pct_trades.json"

STRATEGY = {
    "name": "VWAP Extension Fade",
    "slug": "vwap_extension_fade_15m",
    "timeframe": "15m",
    "universe": "Nifty 200",
    "capital": 100_000,
    "description": (
        "Fade stocks that stretch 1.8%+ away from session VWAP on a quiet index day. "
        "Do not fade strong trends (ADX > 26)."
    ),
    "entry": [
        "15-minute bar close",
        "Distance from session VWAP between 1.8% and 4.0%",
        "RSI(14) ≥ 68 to short, or ≤ 32 to buy the dip",
        "ADX(14) ≤ 26 (skip strong trends)",
        "|Nifty 200 day change| ≤ 0.8% (range day)",
        "Price > ₹60 and 20-bar volume SMA > 30,000",
        "Among names that fire on the same bar, take only the most extended",
        "Enter next 15m open",
    ],
    "exit": [
        "Target: 1.1 × ATR(14) back toward the mean",
        "Stop: 0.45 × ATR beyond the signal close",
        "Force flatten at 15:15 IST",
        "Optional day lock +₹4,000 / halt −₹2,500",
    ],
    "risk": {
        "risk_pct": 2.5,
        "max_positions": 5,
        "leverage": 5,
        "max_deploy": 0.30,
        "daily_lock": 4000,
        "daily_halt": 2500,
        "costs": "0.05% slippage + 0.03% commission per side",
    },
}


def load_fade_results() -> dict[str, Any]:
    if not RESULTS.exists():
        return {"strategy": STRATEGY, "result": {}, "months": [], "trades": []}
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    trades: list[dict] = []
    if TRADES.exists():
        raw = json.loads(TRADES.read_text(encoding="utf-8"))
        trades = raw if isinstance(raw, list) else []
        trades = sorted(trades, key=lambda t: str(t.get("exit_ts", "")), reverse=True)
    data.setdefault("strategy", STRATEGY)
    data["trades"] = trades[:250]
    return data


def load_hunt_results() -> dict[str, Any]:
    """2-month Nifty 200 15m tournament (Jun 3–Aug 25 2026)."""
    if not HUNT.exists():
        return {}
    data = json.loads(HUNT.read_text(encoding="utf-8"))
    trades: list[dict] = []
    if HUNT_TRADES.exists():
        raw = json.loads(HUNT_TRADES.read_text(encoding="utf-8"))
        trades = raw if isinstance(raw, list) else []
        trades = sorted(trades, key=lambda t: str(t.get("exit_ts", "")), reverse=True)
    data["trades"] = trades[:250]
    top = data.get("top") or []
    consistent = None
    alloc = None
    for row in top:
        name = row.get("name") or ""
        if consistent is None and "1.8% ATR | risk r5 a0 p3 k1" in name:
            consistent = row
        if alloc is None and row.get("params", {}).get("mode") == "alloc":
            alloc = row
    data["consistent"] = consistent
    data["alloc"] = alloc
    return data


def load_100pct_pack() -> dict[str, Any]:
    if not PACK100.exists():
        return {}
    data = json.loads(PACK100.read_text(encoding="utf-8"))
    trades: list[dict] = []
    if PACK100_TRADES.exists():
        raw = json.loads(PACK100_TRADES.read_text(encoding="utf-8"))
        trades = raw if isinstance(raw, list) else []
        trades = sorted(trades, key=lambda t: str(t.get("exit_ts", "")), reverse=True)
    data["trades"] = trades[:250]
    return data
