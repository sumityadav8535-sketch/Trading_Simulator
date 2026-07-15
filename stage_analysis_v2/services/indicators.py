"""
Technical indicators — pure pandas (SMA, volume, RS helpers).
"""
from __future__ import annotations

import pandas as pd

WEEKLY_MA_PERIOD = 30
DAILY_MA_PERIOD = 150
VOLUME_AVG_PERIOD = 20


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def add_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    out["ma_30w"] = sma(out["close"], WEEKLY_MA_PERIOD)
    out["vol_avg"] = out["volume"].rolling(VOLUME_AVG_PERIOD, min_periods=5).mean()
    out["vol_ratio"] = out["volume"] / out["vol_avg"].replace(0, 1)
    return out


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    out["ma_150d"] = sma(out["close"], DAILY_MA_PERIOD)
    out["ma_50d"] = sma(out["close"], 50)
    out["vol_avg"] = out["volume"].rolling(VOLUME_AVG_PERIOD, min_periods=10).mean()
    out["vol_ratio"] = out["volume"] / out["vol_avg"].replace(0, 1)
    return out


def volume_surge_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    if len(df) < lookback + 1:
        return 0.0
    avg = float(df["volume"].iloc[-lookback - 1:-1].mean()) or 1.0
    return float(df["volume"].iloc[-1]) / avg