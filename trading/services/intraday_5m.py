"""Load 5-minute Nifty 200 hunt results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings

HUNT = Path(settings.BASE_DIR) / "data" / "intraday_5m_hunt.json"
TRADES = Path(settings.BASE_DIR) / "data" / "intraday_5m_hunt_trades.json"


def load_5m_hunt() -> dict[str, Any]:
    if not HUNT.exists():
        return {}
    data = json.loads(HUNT.read_text(encoding="utf-8"))
    trades: list[dict] = []
    if TRADES.exists():
        raw = json.loads(TRADES.read_text(encoding="utf-8"))
        trades = raw if isinstance(raw, list) else []
        trades = sorted(trades, key=lambda t: str(t.get("exit_ts", "")), reverse=True)
    data["trades"] = trades[:250]
    return data
