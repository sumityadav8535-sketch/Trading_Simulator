"""
Elite ML Long v1 — strategy config, long signals, and filter gate.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from trading.services.fno_engine import FEATURE_COLS, TARGET_R

# OOS-selected champion (can be overridden by model meta filters)
STRATEGY_LONG = {
    "name": "Elite ML Long v1",
    "ml_threshold": 0.72,
    "require_ema_stack": True,
    "adx_min": 0.0,
    "max_risk_pts": 35.0,
    "min_risk_pts": 1.5,
    "hour_start": 9,
    "hour_end": 12,
    "rsi_min": 40.0,
    "rsi_max": 75.0,
    "range_pos_min": 0.0,
    "max_path_pct": 0.8,
    "min_path_pct": -0.55,
    "require_di_bull": True,
    "require_macd_hist_pos": False,
    "soft_checklist": False,
    "soft_checklist_min_score": 6,
    "target_r": 1.8,
    "max_trades_per_day": 4,
}

LONG_TARGET_R = float(STRATEGY_LONG["target_r"])


def sig_ema_long(row, target_r: float | None = None) -> Optional[dict]:
    """Strict bullish EMA stack + above VWAP + RSI ≥ 50."""
    tr = float(target_r if target_r is not None else LONG_TARGET_R)
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("ema_50")):
        return None
    if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
        return None
    if pd.notna(row.get("vwap")) and row["close"] <= row["vwap"]:
        return None
    if row["rsi"] < 50:
        return None
    stop = float(row["ema_21"])
    risk = float(row["close"]) - stop
    if risk <= 0:
        return None
    return {
        "side": 1,
        "stop": stop,
        "target": float(row["close"] + risk * tr),
        "risk_pts": float(risk),
    }


def loose_long_signal(row) -> bool:
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return False
    trend = row["ema_9"] > row["ema_21"]
    above_vwap = pd.isna(row.get("vwap")) or row["close"] > row["vwap"]
    rsi_ok = 40 < row["rsi"] < 75
    return bool(trend and above_vwap and rsi_ok)


def passes_long_filters(row, ts, prob: float, risk_pts: float, cfg: dict | None = None) -> bool:
    f = cfg or STRATEGY_LONG
    if prob is None or prob < f.get("ml_threshold", 0.72):
        return False
    if f.get("require_ema_stack"):
        if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
            return False
        if pd.notna(row.get("vwap")) and row["close"] <= row["vwap"]:
            return False
    adx_min = f.get("adx_min", 0) or 0
    if adx_min > 0:
        adx = row.get("adx")
        if pd.isna(adx) or adx < adx_min:
            return False
    rp_min = f.get("range_pos_min", 0) or 0
    if rp_min > 0:
        rp = row.get("range_pos")
        if pd.isna(rp) or rp < rp_min:
            return False
    max_risk = f.get("max_risk_pts", 999)
    min_risk = f.get("min_risk_pts", 0) or 0
    if risk_pts > max_risk or risk_pts < min_risk:
        return False
    h = ts.hour if hasattr(ts, "hour") else 0
    if h < f.get("hour_start", 9) or h > f.get("hour_end", 12):
        return False
    rsi = row.get("rsi")
    if pd.isna(rsi) or not (f.get("rsi_min", 0) <= rsi <= f.get("rsi_max", 100)):
        return False
    max_path = f.get("max_path_pct")
    if max_path is not None and pd.notna(row.get("pct_from_open")):
        if row["pct_from_open"] > max_path:
            return False
    min_path = f.get("min_path_pct")
    if min_path is not None and pd.notna(row.get("pct_from_open")):
        if row["pct_from_open"] < min_path:
            return False
    if f.get("require_di_bull"):
        if pd.isna(row.get("di_diff")) or row["di_diff"] <= 0:
            return False
    if f.get("require_macd_hist_pos"):
        if pd.isna(row.get("macd_hist")) or row["macd_hist"] <= 0:
            return False
    return True


def merge_strategy_from_meta(meta: dict | None) -> dict:
    """Merge saved model meta filters into STRATEGY_LONG defaults."""
    out = dict(STRATEGY_LONG)
    if not meta:
        return out
    filters = meta.get("strategy_filters") or meta.get("filters") or {}
    for k, v in filters.items():
        if v is not None:
            out[k] = v
    if meta.get("threshold") is not None and "ml_threshold" not in filters:
        out["ml_threshold"] = meta["threshold"]
    if meta.get("strategy_name"):
        out["name"] = meta["strategy_name"]
    return out


__all__ = [
    "FEATURE_COLS",
    "LONG_TARGET_R",
    "STRATEGY_LONG",
    "loose_long_signal",
    "merge_strategy_from_meta",
    "passes_long_filters",
    "sig_ema_long",
]
