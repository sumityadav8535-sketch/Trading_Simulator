"""
Technical indicator engine (pure pandas/numpy — no TA-Lib dependency).
Calculates EMAs, ADX, RSI, volume SMA, ATR, swing levels, Fib retracements.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading.constants import MIN_BARS_FOR_STRATEGY
from trading.services.market_data import load_price_dataframe

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ADX, +DI, -DI."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(high, low, close, length)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich OHLCV DataFrame with all indicators required by the strategy.
    Expects columns: open, high, low, close, volume (lowercase).
    """
    if df.empty or len(df) < 50:
        return df

    out = df.copy()

    out["ema_20"] = _ema(out["close"], 20)
    out["ema_50"] = _ema(out["close"], 50)
    out["ema_200"] = _ema(out["close"], 200)
    out["rsi_14"] = _rsi(out["close"], 14)
    out["vol_sma_20"] = _sma(out["volume"], 20)
    out["atr_14"] = _atr(out["high"], out["low"], out["close"], 14)

    adx, di_plus, di_minus = _adx(out["high"], out["low"], out["close"], 14)
    out["adx_14"] = adx
    out["di_plus"] = di_plus
    out["di_minus"] = di_minus

    out["swing_high"], out["swing_low"] = _detect_swings(out)
    out["fib_382"], out["fib_618"] = _fib_levels(out)

    out["bullish_engulfing"] = _bullish_engulfing(out)
    out["hammer"] = _hammer(out)
    out["strong_close"] = out["close"] > (out["low"] + (out["high"] - out["low"]) * 0.65)

    return out


def _detect_swings(df: pd.DataFrame, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Rolling swing high/low over recent bars (for Fib + SL)."""
    swing_high = df["high"].rolling(lookback * 2 + 1, center=True).max()
    swing_low = df["low"].rolling(lookback * 2 + 1, center=True).min()
    is_swing_high = df["high"] == swing_high
    is_swing_low = df["low"] == swing_low

    sh = pd.Series(np.nan, index=df.index)
    sl = pd.Series(np.nan, index=df.index)
    last_sh = np.nan
    last_sl = np.nan
    for i, idx in enumerate(df.index):
        if is_swing_high.iloc[i]:
            last_sh = df["high"].iloc[i]
        if is_swing_low.iloc[i]:
            last_sl = df["low"].iloc[i]
        sh.iloc[i] = last_sh
        sl.iloc[i] = last_sl
    return sh, sl


def _fib_levels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """38.2% and 61.8% retracement of latest swing leg."""
    fib_382 = pd.Series(np.nan, index=df.index)
    fib_618 = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        sh = df["swing_high"].iloc[i]
        sl = df["swing_low"].iloc[i]
        if pd.isna(sh) or pd.isna(sl) or sh <= sl:
            continue
        leg = sh - sl
        fib_382.iloc[i] = sh - leg * 0.382
        fib_618.iloc[i] = sh - leg * 0.618
    return fib_382, fib_618


def _bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bullish = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    )
    return bullish.fillna(False)


def _hammer(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (
        (lower_wick >= body * 2)
        & (upper_wick <= body * 0.5)
        & (body / rng < 0.35)
    ).fillna(False)


def get_indicator_frame(
    symbol: str,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load prices and compute indicators; optionally slice to as_of date."""
    df = load_price_dataframe(symbol)
    if df.empty:
        return df
    df = compute_indicators(df)
    if as_of is not None:
        df = df.loc[:as_of]
    return df


def has_sufficient_history(df: pd.DataFrame) -> bool:
    return len(df.dropna(subset=["ema_200", "adx_14"])) >= MIN_BARS_FOR_STRATEGY