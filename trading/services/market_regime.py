"""
Market regime filter — Nifty 50 index above 50 EMA.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.services.nifty50_index import load_nifty50_frame

# Kept for API compatibility; index regime is binary (above/below 50 EMA).
DEFAULT_MIN_BULLISH = 1


def load_proxy_frames(symbols=None) -> dict[str, pd.DataFrame]:
    """Load Nifty 50 index frame (legacy name kept for callers)."""
    df = load_nifty50_frame()
    if df.empty:
        return {}
    return {NIFTY50_SYMBOL: df}


def _snapshot_at(df: pd.DataFrame, eval_ts) -> Optional[dict]:
    hist = df.loc[:pd.Timestamp(eval_ts)]
    if hist.empty:
        return None
    row = hist.iloc[-1]
    ema50 = row.get("ema_50")
    ema200 = row.get("ema_200")
    if pd.isna(ema50):
        return None
    close = float(row["close"])
    e50 = float(ema50)
    e200 = float(ema200) if pd.notna(ema200) else None
    above_50 = close > e50
    above_200 = close > e200 if e200 is not None else None
    return {
        "symbol": NIFTY50_SYMBOL,
        "close": round(close, 2),
        "ema_50": round(e50, 2),
        "ema_200": round(e200, 2) if e200 is not None else None,
        "above_50ema": above_50,
        "above_200ema": above_200,
        "status": "bullish" if above_50 else "bearish",
    }


def is_market_bullish(
    proxy_frames: dict[str, pd.DataFrame],
    eval_ts,
    min_bullish: int = DEFAULT_MIN_BULLISH,
) -> bool:
    """True when Nifty 50 close is above its 50 EMA."""
    if not proxy_frames:
        return True
    df = proxy_frames.get(NIFTY50_SYMBOL)
    if df is None and proxy_frames:
        df = next(iter(proxy_frames.values()), None)
    if df is None or df.empty:
        return True
    snap = _snapshot_at(df, eval_ts)
    if snap is None:
        return False
    return snap["above_50ema"]


def is_nifty_strong_regime(
    nifty_df: pd.DataFrame,
    eval_ts,
) -> bool:
    """True when Nifty 50 is above both 50 EMA and 200 EMA (strong bull regime)."""
    if nifty_df is None or nifty_df.empty:
        return True
    snap = _snapshot_at(nifty_df, eval_ts)
    if snap is None:
        return False
    above_200 = snap.get("above_200ema")
    return bool(snap["above_50ema"] and above_200)


def get_market_regime_status(
    eval_date=None,
    proxy_frames: Optional[dict[str, pd.DataFrame]] = None,
    min_bullish: int = DEFAULT_MIN_BULLISH,
) -> dict:
    """Nifty 50 regime snapshot for dashboard display."""
    frames = proxy_frames or load_proxy_frames()
    if not frames:
        return {
            "is_bullish": False,
            "bullish_count": 0,
            "total_checked": 0,
            "min_required": min_bullish,
            "eval_date": None,
            "index": None,
            "proxies": [],
            "message": "Nifty 50 index data unavailable — run: python manage.py load_nifty50_index",
        }

    df = frames.get(NIFTY50_SYMBOL)
    if df is None:
        df = next(iter(frames.values()))
    if eval_date is None:
        eval_ts = df.index.max()
    else:
        eval_ts = pd.Timestamp(eval_date)

    snap = _snapshot_at(df, eval_ts)
    if snap is None:
        return {
            "is_bullish": False,
            "bullish_count": 0,
            "total_checked": 0,
            "min_required": min_bullish,
            "eval_date": str(eval_ts.date())[:10],
            "index": None,
            "proxies": [],
            "message": "Insufficient Nifty 50 history for regime check",
        }

    is_bullish = snap["above_50ema"]
    if is_bullish:
        message = (
            f"Market ON — Nifty 50 ({snap['close']}) above 50 EMA ({snap['ema_50']}). "
            "Strategy signals allowed."
        )
    else:
        message = (
            f"Market OFF — Nifty 50 ({snap['close']}) below 50 EMA ({snap['ema_50']}). "
            "Wait for index to reclaim 50 EMA."
        )

    return {
        "is_bullish": is_bullish,
        "bullish_count": 1 if is_bullish else 0,
        "total_checked": 1,
        "min_required": min_bullish,
        "eval_date": eval_ts.date().isoformat() if hasattr(eval_ts, "date") else str(eval_ts)[:10],
        "index": snap,
        "proxies": [snap],
        "message": message,
    }


def market_breadth(
    proxy_frames: dict[str, pd.DataFrame],
    eval_ts,
) -> tuple[int, int]:
    """Return (bullish_count, total_checked) — 1 or 0 for index regime."""
    if not proxy_frames:
        return 0, 0
    df = proxy_frames.get(NIFTY50_SYMBOL)
    if df is None and proxy_frames:
        df = next(iter(proxy_frames.values()), None)
    if df is None:
        return 0, 0
    snap = _snapshot_at(df, eval_ts)
    if snap is None:
        return 0, 0
    return (1, 1) if snap["above_50ema"] else (0, 1)