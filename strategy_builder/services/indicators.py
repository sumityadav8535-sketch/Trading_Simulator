"""
Indicator computation with per-frame caching (compute each unique series once).
Reuses pure-pandas math patterns from trading.services.indicators.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_builder.services.catalog import VALID_INDICATOR_IDS


def _source(df: pd.DataFrame, source: str = "close") -> pd.Series:
    src = (source or "close").lower()
    if src not in df.columns:
        src = "close"
    return df[src].astype(float)


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(max(1, int(n))).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(1, int(n)), adjust=False).mean()


def _wma(s: pd.Series, n: int) -> pd.Series:
    n = max(1, int(n))
    w = np.arange(1, n + 1, dtype=float)

    def f(x):
        return np.dot(x, w) / w.sum()

    return s.rolling(n).apply(f, raw=True)


def _rma(s: pd.Series, n: int) -> pd.Series:
    n = max(1, int(n))
    return s.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def _hma(s: pd.Series, n: int) -> pd.Series:
    n = max(2, int(n))
    half = max(1, n // 2)
    sqrt_n = max(1, int(math.sqrt(n)))
    return _wma(2 * _wma(s, half) - _wma(s, n), sqrt_n)


def _vwma(price: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    n = max(1, int(n))
    return (price * vol).rolling(n).sum() / vol.rolling(n).sum().replace(0, np.nan)


def _dema(s: pd.Series, n: int) -> pd.Series:
    e = _ema(s, n)
    return 2 * e - _ema(e, n)


def _tema(s: pd.Series, n: int) -> pd.Series:
    e1 = _ema(s, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    n = max(1, int(n))
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    n = max(1, int(n))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    n = max(1, int(n))
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    atr = _atr(h, l, c, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, plus_di, minus_di


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _stoch(h, l, c, k=14, d=3, smooth=3):
    k = max(1, int(k))
    d = max(1, int(d))
    smooth = max(1, int(smooth))
    lowest = l.rolling(k).min()
    highest = h.rolling(k).max()
    raw_k = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    k_line = raw_k.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _ensure_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]
    out["_body"] = (c - o).abs()
    out["_range"] = (h - l).replace(0, np.nan)
    out["_body_pct"] = out["_body"] / out["_range"] * 100
    out["_upper_wick"] = h - pd.concat([o, c], axis=1).max(axis=1)
    out["_lower_wick"] = pd.concat([o, c], axis=1).min(axis=1) - l
    out["_is_bullish"] = (c > o).astype(float)
    out["_is_bearish"] = (c < o).astype(float)
    return out


class IndicatorCache:
    """Compute and cache indicator series for one OHLCV frame."""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            self.df = df
        else:
            self.df = _ensure_base(df)
        self._cache: dict[str, pd.Series] = {}

    def _key(self, name: str, params: dict) -> str:
        items = sorted((params or {}).items())
        return name + "|" + ",".join(f"{k}={v}" for k, v in items)

    def series(self, name: str, params: Optional[dict] = None) -> pd.Series:
        params = dict(params or {})
        name = (name or "").lower()
        if name not in VALID_INDICATOR_IDS and name not in {
            "body", "range", "body_pct", "upper_wick", "lower_wick",
            "is_bullish", "is_bearish", "open", "high", "low", "close", "volume",
        }:
            raise ValueError(f"Unknown indicator: {name}")

        key = self._key(name, params)
        if key in self._cache:
            return self._cache[key]

        df = self.df
        if df.empty:
            s = pd.Series(dtype=float)
            self._cache[key] = s
            return s

        h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
        src = _source(df, str(params.get("source", "close")))

        if name in ("open", "high", "low", "close", "volume"):
            s = df[name].astype(float)
        elif name == "body":
            s = df["_body"]
        elif name == "range":
            s = df["_range"]
        elif name == "body_pct":
            s = df["_body_pct"]
        elif name == "upper_wick":
            s = df["_upper_wick"]
        elif name == "lower_wick":
            s = df["_lower_wick"]
        elif name == "is_bullish":
            s = df["_is_bullish"]
        elif name == "is_bearish":
            s = df["_is_bearish"]
        elif name == "sma":
            s = _sma(src, int(params.get("length", 44)))
        elif name == "ema":
            s = _ema(src, int(params.get("length", 20)))
        elif name == "wma":
            s = _wma(src, int(params.get("length", 20)))
        elif name == "hma":
            s = _hma(src, int(params.get("length", 20)))
        elif name == "vwma":
            s = _vwma(src, v, int(params.get("length", 20)))
        elif name == "dema":
            s = _dema(src, int(params.get("length", 20)))
        elif name == "tema":
            s = _tema(src, int(params.get("length", 20)))
        elif name == "rma":
            s = _rma(src, int(params.get("length", 14)))
        elif name == "rsi":
            s = _rsi(src, int(params.get("length", 14)))
        elif name == "macd":
            s, _, _ = _macd(c, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        elif name == "macd_signal":
            _, s, _ = _macd(c, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        elif name == "macd_hist":
            _, _, s = _macd(c, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        elif name == "stoch_k":
            s, _ = _stoch(h, l, c, int(params.get("k", 14)), int(params.get("d", 3)), int(params.get("smooth", 3)))
        elif name == "stoch_d":
            _, s = _stoch(h, l, c, int(params.get("k", 14)), int(params.get("d", 3)), int(params.get("smooth", 3)))
        elif name == "cci":
            n = max(1, int(params.get("length", 20)))
            tp = (h + l + c) / 3
            sma = tp.rolling(n).mean()
            mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            s = (tp - sma) / (0.015 * mad.replace(0, np.nan))
        elif name == "williams_r":
            n = max(1, int(params.get("length", 14)))
            hh = h.rolling(n).max()
            ll = l.rolling(n).min()
            s = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
        elif name == "roc":
            n = max(1, int(params.get("length", 12)))
            s = src.pct_change(n) * 100
        elif name == "momentum":
            n = max(1, int(params.get("length", 10)))
            s = src.diff(n)
        elif name == "mfi":
            n = max(1, int(params.get("length", 14)))
            tp = (h + l + c) / 3
            rmf = tp * v
            pos = rmf.where(tp > tp.shift(1), 0.0)
            neg = rmf.where(tp < tp.shift(1), 0.0)
            mfr = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
            s = 100 - (100 / (1 + mfr))
        elif name == "adx":
            s, _, _ = _adx(h, l, c, int(params.get("length", 14)))
        elif name == "di_plus":
            _, s, _ = _adx(h, l, c, int(params.get("length", 14)))
        elif name == "di_minus":
            _, _, s = _adx(h, l, c, int(params.get("length", 14)))
        elif name == "atr":
            s = _atr(h, l, c, int(params.get("length", 14)))
        elif name == "atr_pct":
            s = _atr(h, l, c, int(params.get("length", 14))) / c * 100
        elif name in ("bb_upper", "bb_mid", "bb_lower"):
            n = max(1, int(params.get("length", 20)))
            mult = float(params.get("mult", 2.0))
            mid = _sma(c, n)
            std = c.rolling(n).std()
            if name == "bb_mid":
                s = mid
            elif name == "bb_upper":
                s = mid + mult * std
            else:
                s = mid - mult * std
        elif name == "donchian_high":
            n = max(1, int(params.get("length", 20)))
            s = h.rolling(n).max().shift(1)  # exclude current bar (no look-ahead for breakout)
        elif name == "donchian_low":
            n = max(1, int(params.get("length", 20)))
            s = l.rolling(n).min().shift(1)
        elif name in ("keltner_upper", "keltner_lower"):
            n = max(1, int(params.get("length", 20)))
            mult = float(params.get("mult", 1.5))
            mid = _ema(c, n)
            atr = _atr(h, l, c, n)
            s = mid + mult * atr if name == "keltner_upper" else mid - mult * atr
        elif name == "vol_sma":
            s = _sma(v, int(params.get("length", 20)))
        elif name == "rel_volume":
            avg = _sma(v, int(params.get("length", 20)))
            s = v / avg.replace(0, np.nan)
        elif name == "obv":
            direction = np.sign(c.diff()).fillna(0)
            s = (direction * v).cumsum()
        elif name == "obv_sma":
            direction = np.sign(c.diff()).fillna(0)
            obv = (direction * v).cumsum()
            s = _sma(obv, int(params.get("length", 20)))
        elif name == "cmf":
            n = max(1, int(params.get("length", 20)))
            mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
            mfv = mfm * v
            s = mfv.rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)
        elif name == "highest_high":
            n = max(1, int(params.get("length", 20)))
            s = h.rolling(n).max().shift(1)
        elif name == "lowest_low":
            n = max(1, int(params.get("length", 20)))
            s = l.rolling(n).min().shift(1)
        elif name == "prev_day_high":
            s = h.shift(1)
        elif name == "prev_day_low":
            s = l.shift(1)
        elif name == "prev_close":
            s = c.shift(1)
        else:
            raise ValueError(f"Indicator not implemented: {name}")

        self._cache[key] = s
        return s

    def value_at(self, name: str, params: Optional[dict], idx: int, offset: int = 0) -> float:
        s = self.series(name, params)
        pos = idx - int(offset or 0)
        if pos < 0 or pos >= len(s):
            return float("nan")
        val = s.iloc[pos]
        if pd.isna(val):
            return float("nan")
        return float(val)
