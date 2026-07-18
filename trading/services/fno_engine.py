"""
F&O intraday engine: indicators, ML features, and signal rules for Nifty / Bank Nifty.
"""
from __future__ import annotations

from datetime import time as dt_time
from typing import Optional

import numpy as np
import pandas as pd

CAPITAL = 500_000.0
RISK_PCT = 1.5
TARGET_R = 1.5
SLIPPAGE_PTS = 0.5
TRADES_PER_DAY = 4

# Elite ML Short v2 — sole F&O strategy
# Soft checklist (score ≥ 6) is applied in fno_live / fno_checklist before ACTIVE.
STRATEGY = {
    "name": "Elite ML Short v2",
    "ml_threshold": 0.58,
    "adx_min": 0.0,
    "range_pos_min": 0.0,
    "max_risk_pts": 22.0,
    "min_risk_pts": 0.0,
    "hour_start": 9,
    "hour_end": 14,
    "require_ema_stack": False,
    "rsi_min": 40.0,
    "rsi_max": 65.0,
    "soft_checklist": True,
    "soft_checklist_min_score": 6,
}

MARKET_OPEN = dt_time(9, 15)
NO_ENTRY_AFTER = dt_time(14, 45)
FORCE_EXIT = dt_time(15, 15)

INSTRUMENTS = {
    "NIFTY": {
        "ticker": "^NSEI",
        "lot_size": 75,
        "mis_margin": 65_000,
        "name": "Nifty 50 Futures",
    },
    "BANKNIFTY": {
        "ticker": "^NSEBANK",
        "lot_size": 30,
        "mis_margin": 85_000,
        "name": "Bank Nifty Futures",
    },
}

FEATURE_COLS = [
    "ema9_ema21", "ema21_ema50", "px_ema9", "px_ema21", "px_ema50", "px_vwap",
    "rsi", "prev_rsi", "rsi_chg", "adx", "di_plus", "di_minus", "di_diff",
    "bb_width", "bb_pct", "macd", "macd_sig", "macd_hist",
    "stoch_k", "stoch_d", "roc_3", "roc_6", "roc_12", "roc_24",
    "vol_ratio", "vol_z", "body_pct", "upper_wick", "lower_wick",
    "range_pos", "pct_from_open", "session_ret", "atr_pct",
    "cci", "williams_r", "ema9_slope", "ema21_slope", "vwap_slope",
    "atr_expansion", "dist_day_high", "bearish_3", "rsi_macd_div", "px_kijun",
    "hour", "mins_from_open", "mins_to_close", "strong_open_reject",
]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(0)
    rename = {c: str(c).lower() for c in sub.columns}
    sub = sub.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    sub = sub[keep].dropna(subset=["close"])
    if sub.index.tz is None:
        sub.index = sub.index.tz_localize("Asia/Kolkata")
    else:
        sub.index = sub.index.tz_convert("Asia/Kolkata")
    return sub.sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema_21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema_50"] = d["close"].ewm(span=50, adjust=False).mean()
    if (d["volume"] > 0).any():
        d["vol_sma"] = d["volume"].rolling(20).mean()
    else:
        d["vol_sma"] = 1.0

    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    prev = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    d["session_date"] = d.index.date
    typical = (d["high"] + d["low"] + d["close"]) / 3
    cum_pv = (typical * d["volume"]).groupby(d["session_date"]).cumsum()
    cum_vol = d["volume"].groupby(d["session_date"]).cumsum().replace(0, np.nan)
    d["vwap"] = cum_pv / cum_vol
    if d["vwap"].isna().all():
        d["vwap"] = typical.groupby(d["session_date"]).transform(lambda s: s.expanding().mean())

    d["day_open"] = d.groupby("session_date")["open"].transform("first")
    d["prev_rsi"] = d["rsi"].shift(1)
    d["prev_close"] = d["close"].shift(1)
    rng = d["high"] - d["low"]
    d["strong_close"] = (d["close"] - d["low"]) / rng.replace(0, np.nan) >= 0.65
    d["strong_open_reject"] = (d["high"] - d["close"]) / rng.replace(0, np.nan) >= 0.65
    return d


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(df)

    d["ema9_ema21"] = (d["ema_9"] - d["ema_21"]) / d["close"] * 100
    d["ema21_ema50"] = (d["ema_21"] - d["ema_50"]) / d["close"] * 100
    d["px_ema9"] = (d["close"] - d["ema_9"]) / d["close"] * 100
    d["px_ema21"] = (d["close"] - d["ema_21"]) / d["close"] * 100
    d["px_ema50"] = (d["close"] - d["ema_50"]) / d["close"] * 100
    d["px_vwap"] = (d["close"] - d["vwap"]) / d["vwap"] * 100

    up = d["high"].diff()
    down = -d["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["di_plus"] = plus_di
    d["di_minus"] = minus_di
    d["di_diff"] = plus_di - minus_di

    d["bb_mid"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"] * 100
    d["bb_pct"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = (d["close"] - low14) / (high14 - low14).replace(0, np.nan) * 100
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    for n in (3, 6, 12, 24):
        d[f"roc_{n}"] = d["close"].pct_change(n) * 100

    if (d["volume"] > 0).any():
        d["vol_ratio"] = d["volume"] / d["vol_sma"].replace(0, np.nan)
        d["vol_z"] = (d["volume"] - d["vol_sma"]) / d["volume"].rolling(20).std().replace(0, np.nan)
    else:
        bar_range = (d["high"] - d["low"]).replace(0, np.nan)
        range_sma = bar_range.rolling(20).mean()
        d["vol_ratio"] = bar_range / range_sma.replace(0, np.nan)
        d["vol_z"] = (bar_range - range_sma) / bar_range.rolling(20).std().replace(0, np.nan)

    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_pct"] = (d["close"] - d["open"]) / rng * 100
    d["upper_wick"] = (d["high"] - d[["open", "close"]].max(axis=1)) / rng * 100
    d["lower_wick"] = (d[["open", "close"]].min(axis=1) - d["low"]) / rng * 100

    d["day_high"] = d.groupby("session_date")["high"].cummax()
    d["day_low"] = d.groupby("session_date")["low"].cummin()
    d["day_range"] = d["day_high"] - d["day_low"]
    d["range_pos"] = (d["close"] - d["day_low"]) / d["day_range"].replace(0, np.nan)
    d["pct_from_open"] = (d["close"] - d["day_open"]) / d["day_open"] * 100
    d["session_ret"] = (
        d.groupby("session_date")["close"].pct_change().groupby(d["session_date"]).cumsum() * 100
    )

    d["hour"] = d.index.hour + d.index.minute / 60
    d["mins_from_open"] = (d.index.hour - 9) * 60 + d.index.minute - 15
    d["mins_to_close"] = (15 * 60 + 30) - (d.index.hour * 60 + d.index.minute)

    d["rsi_chg"] = d["rsi"] - d["prev_rsi"]
    d["atr_pct"] = d["atr"] / d["close"] * 100

    tp = (d["high"] + d["low"] + d["close"]) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda s: np.abs(s - s.mean()).mean(), raw=True)
    d["cci"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    d["williams_r"] = (high14 - d["close"]) / (high14 - low14).replace(0, np.nan) * -100

    d["ema9_slope"] = d["ema_9"].pct_change(3) * 100
    d["ema21_slope"] = d["ema_21"].pct_change(3) * 100
    d["vwap_slope"] = d["vwap"].pct_change(3) * 100
    d["atr_expansion"] = d["atr"] / d["atr"].rolling(20).mean().replace(0, np.nan)
    d["dist_day_high"] = (d["day_high"] - d["close"]) / d["day_range"].replace(0, np.nan)
    d["bearish_3"] = (
        (d["close"] < d["open"])
        & (d["close"].shift(1) < d["open"].shift(1))
        & (d["close"].shift(2) < d["open"].shift(2))
    ).astype(int)
    d["rsi_macd_div"] = d["roc_6"] - d["macd_hist"]

    high26 = d["high"].rolling(26).max()
    low26 = d["low"].rolling(26).min()
    d["px_kijun"] = (d["close"] - (high26 + low26) / 2) / d["close"] * 100

    return d


def sig_ema_short(row, target_r: float = TARGET_R) -> Optional[dict]:
    if not (row["ema_9"] < row["ema_21"] < row["ema_50"]):
        return None
    if row["close"] >= row["vwap"]:
        return None
    if row["rsi"] > 50:
        return None
    stop = row["ema_21"]
    risk = stop - row["close"]
    if risk <= 0:
        return None
    return {
        "side": -1,
        "stop": float(stop),
        "target": float(row["close"] - risk * target_r),
        "risk_pts": float(risk),
    }


def loose_short_signal(row) -> bool:
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return False
    trend = row["ema_9"] < row["ema_21"]
    below_vwap = pd.isna(row.get("vwap")) or row["close"] < row["vwap"]
    rsi_ok = 40 < row["rsi"] < 65
    return trend and below_vwap and rsi_ok


def row_feature_vector(row) -> list[float]:
    return [float(row.get(c, np.nan)) for c in FEATURE_COLS]


def max_lots(capital: float, margin_per_lot: float, deploy_pct: float = 0.85) -> int:
    return max(int(capital * deploy_pct / margin_per_lot), 0)


def passes_strategy_filters(row, ts, prob: float, risk_pts: float, cfg: dict | None = None) -> bool:
    """Elite ML Short v2 entry filters."""
    f = cfg or STRATEGY
    if prob is None or prob < f.get("ml_threshold", 0.58):
        return False
    if f.get("require_ema_stack"):
        if not (row["ema_9"] < row["ema_21"] < row["ema_50"]):
            return False
        if pd.notna(row.get("vwap")) and row["close"] >= row["vwap"]:
            return False
        if row["rsi"] > 50:
            return False
    adx_min = f.get("adx_min", 0)
    if adx_min > 0:
        adx = row.get("adx")
        if pd.isna(adx) or adx < adx_min:
            return False
    rp_min = f.get("range_pos_min", 0)
    if rp_min > 0:
        rp = row.get("range_pos")
        if pd.isna(rp) or rp < rp_min:
            return False
    max_risk = f.get("max_risk_pts", 999)
    if risk_pts > max_risk or risk_pts < f.get("min_risk_pts", 0):
        return False
    h = ts.hour if hasattr(ts, "hour") else 0
    if h < f.get("hour_start", 9) or h > f.get("hour_end", 14):
        return False
    if not (f.get("rsi_min", 0) <= row["rsi"] <= f.get("rsi_max", 100)):
        return False
    return True


def lots_for_risk(capital, risk_pct, stop_pts, lot_size, max_l):
    if stop_pts <= 0:
        return 0
    risk_amt = capital * risk_pct / 100
    lots = int(risk_amt / (stop_pts * lot_size))
    return max(min(lots, max_l), 0)