"""
Technical confirmation filters for Stage Analysis 2.0 signals.

Applied on the daily bar at (or just before) the Stage 2 weekly signal date.
Goal: raise win-rate by requiring trend/breakout confluence without killing edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


# ── Indicator helpers (pure pandas) ─────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


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
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Classic Supertrend.
    Returns (supertrend_line, direction) where direction = 1 bull, -1 bear.
    """
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(close)):
        if basic_ub.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        if basic_lb.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    st.iloc[0] = final_ub.iloc[0]
    direction.iloc[0] = -1.0

    for i in range(1, len(close)):
        if st.iloc[i - 1] == final_ub.iloc[i - 1]:
            if close.iloc[i] > final_ub.iloc[i]:
                st.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1.0
            else:
                st.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1.0
        else:
            if close.iloc[i] < final_lb.iloc[i]:
                st.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1.0
            else:
                st.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1.0

    return st, direction


def enrich_daily_tech(daily: pd.DataFrame, *, include_supertrend: bool = True) -> pd.DataFrame:
    """Add EMA / RSI / ATR / Bollinger / (optional) Supertrend columns to daily OHLCV."""
    if daily.empty or len(daily) < 60:
        return daily
    out = daily.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["ema_20"] = _ema(c, 20)
    out["ema_50"] = _ema(c, 50)
    out["ema_200"] = _ema(c, 200)
    out["rsi_14"] = _rsi(c, 14)
    out["atr_14"] = _atr(h, l, c, 14)
    out["bb_mid"] = _sma(c, 20)
    out["bb_std"] = c.rolling(20, min_periods=20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2 * out["bb_std"]
    out["vol_sma_20"] = _sma(v, 20)
    out["high_20"] = h.rolling(20, min_periods=20).max()
    if include_supertrend:
        st, st_dir = _supertrend(h, l, c, period=10, multiplier=3.0)
        out["supertrend"] = st
        out["st_dir"] = st_dir
    else:
        out["supertrend"] = np.nan
        out["st_dir"] = 0.0
    return out


@dataclass
class TechSnapshot:
    """Technical state at signal time (daily bar ≤ week end)."""
    ok: bool
    close: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    rsi: float = 0.0
    atr: float = 0.0
    bb_mid: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    st_dir: float = 0.0
    supertrend: float = 0.0
    vol_ratio: float = 0.0
    near_20d_high: bool = False
    price_vs_ema20_pct: float = 0.0
    price_vs_bb_mid_pct: float = 0.0
    # derived flags
    ema_stack_bull: bool = False       # close > ema20 > ema50
    ema_trend_bull: bool = False       # close > ema50 > ema200
    supertrend_bull: bool = False
    rsi_healthy: bool = False          # 45–70 momentum
    rsi_not_overbought: bool = False   # rsi < 75
    bb_above_mid: bool = False
    bb_breakout: bool = False
    volume_ok: bool = False
    breakout_20d: bool = False
    not_extended: bool = False         # close ≤ 8% above EMA20
    mild_extended: bool = False        # close ≤ 12% above EMA20


def snapshot_tech(daily_tech: pd.DataFrame, asof: pd.Timestamp) -> TechSnapshot:
    """Take last daily bar on/before asof and extract flags."""
    if daily_tech is None or daily_tech.empty:
        return TechSnapshot(ok=False)
    hist = daily_tech.loc[daily_tech.index <= asof]
    if hist.empty or len(hist) < 60:
        return TechSnapshot(ok=False)
    row = hist.iloc[-1]
    needed = ["ema_20", "ema_50", "ema_200", "rsi_14", "bb_mid"]
    if any(pd.isna(row.get(c)) for c in needed):
        return TechSnapshot(ok=False)

    close = float(row["close"])
    e20 = float(row["ema_20"])
    e50 = float(row["ema_50"])
    e200 = float(row["ema_200"])
    rsi = float(row["rsi_14"])
    atr = float(row["atr_14"]) if pd.notna(row.get("atr_14")) else 0.0
    bb_mid = float(row["bb_mid"])
    bb_up = float(row["bb_upper"]) if pd.notna(row.get("bb_upper")) else bb_mid
    bb_lo = float(row["bb_lower"]) if pd.notna(row.get("bb_lower")) else bb_mid
    st_dir = float(row["st_dir"]) if pd.notna(row.get("st_dir")) else 0.0
    st = float(row["supertrend"]) if pd.notna(row.get("supertrend")) else 0.0
    vol_sma = float(row["vol_sma_20"]) if pd.notna(row.get("vol_sma_20")) and row["vol_sma_20"] else 1.0
    vol_ratio = float(row["volume"]) / vol_sma if vol_sma else 0.0
    high_20 = float(row["high_20"]) if pd.notna(row.get("high_20")) else close
    near_20 = close >= high_20 * 0.98 if high_20 > 0 else False
    ext_pct = ((close - e20) / e20 * 100) if e20 else 0.0

    return TechSnapshot(
        ok=True,
        close=close,
        ema_20=e20,
        ema_50=e50,
        ema_200=e200,
        rsi=rsi,
        atr=atr,
        bb_mid=bb_mid,
        bb_upper=bb_up,
        bb_lower=bb_lo,
        st_dir=st_dir,
        supertrend=st,
        vol_ratio=round(vol_ratio, 2),
        near_20d_high=near_20,
        price_vs_ema20_pct=round(ext_pct, 2),
        price_vs_bb_mid_pct=round((close - bb_mid) / bb_mid * 100, 2) if bb_mid else 0.0,
        ema_stack_bull=close > e20 > e50,
        ema_trend_bull=close > e50 and e50 > e200,
        supertrend_bull=(st_dir > 0 and (st == 0 or close > st)),
        rsi_healthy=45 <= rsi <= 70,
        rsi_not_overbought=rsi < 75,
        bb_above_mid=close > bb_mid,
        bb_breakout=close > bb_up,
        volume_ok=vol_ratio >= 1.2,
        breakout_20d=near_20,
        not_extended=ext_pct <= 8.0,
        mild_extended=ext_pct <= 12.0,
    )


# Production UI packs (validated on last-1y cash Rs10L Stage4 exit)
# daily_mtf raised WR 51%→58.5% and return 32.6%→37.6% in research.
TECH_FILTER_CHOICES: list[tuple[str, str]] = [
    ("none", "No tech filter"),
    ("daily_mtf", "Daily Stage 1/2 (recommended — higher WR + return)"),
    ("not_extended", "Not extended (≤8% above EMA20)"),
    ("daily_mtf_not_ext", "Daily Stage 1/2 + not extended"),
    ("ema_stack", "EMA stack (close > EMA20 > EMA50)"),
    ("bb_mid", "Above Bollinger mid-band"),
    ("ema_bb", "EMA stack + Bollinger mid"),
    ("bb_follow", "Bollinger mid + daily follow-through"),
    ("confluence", "Confluence: EMA stack + BB mid + RSI not overbought"),
]

DEFAULT_TECH_FILTER = "daily_mtf"
TECH_FILTER_LABELS = {k: v for k, v in TECH_FILTER_CHOICES}
VALID_TECH_FILTERS = frozenset(TECH_FILTER_LABELS.keys())

# Research packs (also usable via CLI)
FILTER_PACKS: dict[str, dict[str, Any]] = {
    "none": {"label": "No tech filter", "require": [], "daily_stages": None},
    "daily_mtf": {
        "label": "Daily Stage 1/2",
        "require": [],
        "daily_stages": (1, 2),
    },
    "not_extended": {
        "label": "Not extended ≤8% EMA20",
        "require": ["not_extended"],
        "daily_stages": None,
    },
    "daily_mtf_not_ext": {
        "label": "Daily S1/2 + not extended",
        "require": ["not_extended"],
        "daily_stages": (1, 2),
    },
    "ema_stack": {
        "label": "EMA stack (close > EMA20 > EMA50)",
        "require": ["ema_stack_bull"],
        "daily_stages": None,
    },
    "bb_mid": {
        "label": "Above Bollinger mid",
        "require": ["bb_above_mid"],
        "daily_stages": None,
    },
    "ema_bb": {
        "label": "EMA stack + BB mid",
        "require": ["ema_stack_bull", "bb_above_mid"],
        "daily_stages": None,
    },
    "bb_follow": {
        "label": "BB mid + follow-through",
        "require": ["bb_above_mid"],
        "need_follow_through": True,
        "daily_stages": None,
    },
    "confluence": {
        "label": "Confluence EMA+BB+RSI",
        "require": ["ema_stack_bull", "bb_above_mid", "rsi_not_overbought"],
        "daily_stages": None,
    },
    # extras for research
    "supertrend": {"label": "Supertrend bullish", "require": ["supertrend_bull"]},
    "rsi_healthy": {"label": "RSI 45–70", "require": ["rsi_healthy"]},
    "breakout_20d": {"label": "Near 20-day high", "require": ["breakout_20d"]},
    "volume": {"label": "Volume ≥ 1.2× avg", "require": ["volume_ok"]},
    "st_ema": {"label": "ST + EMA stack", "require": ["supertrend_bull", "ema_stack_bull"]},
    "st_ema_bb": {
        "label": "ST + EMA + BB",
        "require": ["supertrend_bull", "ema_stack_bull", "bb_above_mid"],
    },
    "confluence_vol": {
        "label": "Confluence + volume",
        "require": [
            "supertrend_bull",
            "ema_stack_bull",
            "bb_above_mid",
            "rsi_not_overbought",
            "volume_ok",
        ],
    },
    "strict_momentum": {
        "label": "Strict momentum",
        "require": ["supertrend_bull", "ema_trend_bull", "breakout_20d", "rsi_healthy"],
    },
}


def normalize_tech_filter(tech_filter: str | None) -> str:
    mode = (tech_filter or DEFAULT_TECH_FILTER).strip().lower()
    if mode not in VALID_TECH_FILTERS and mode not in FILTER_PACKS:
        return DEFAULT_TECH_FILTER
    return mode


def pack_needs_tech_snapshot(pack_id: str) -> bool:
    pack = FILTER_PACKS.get(pack_id) or FILTER_PACKS["none"]
    req = pack.get("require") or []
    return bool(req)


def pack_needs_supertrend(pack_id: str) -> bool:
    pack = FILTER_PACKS.get(pack_id) or FILTER_PACKS["none"]
    req = pack.get("require") or []
    return "supertrend_bull" in req


def passes_tech_filter(
    snap: TechSnapshot | None,
    pack_id: str,
    *,
    daily_stage: int = 0,
    follow_through: bool = False,
) -> bool:
    """
    Return True if signal passes the named tech filter pack.
    daily_stage / follow_through come from stage/breakout engines.
    """
    pack = FILTER_PACKS.get(pack_id) or FILTER_PACKS["none"]
    stages = pack.get("daily_stages")
    if stages is not None and daily_stage not in stages:
        return False
    if pack.get("need_follow_through") and not follow_through:
        return False
    req = pack.get("require") or []
    if not req:
        return True
    if snap is None or not snap.ok:
        return False
    for flag in req:
        if not bool(getattr(snap, flag, False)):
            return False
    return True


def passes_custom(
    snap: TechSnapshot,
    *,
    require_supertrend: bool = False,
    require_ema_stack: bool = False,
    require_ema_trend: bool = False,
    require_bb_mid: bool = False,
    require_rsi_healthy: bool = False,
    require_rsi_not_ob: bool = False,
    require_volume: bool = False,
    require_20d_breakout: bool = False,
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
) -> bool:
    if not any([
        require_supertrend, require_ema_stack, require_ema_trend, require_bb_mid,
        require_rsi_healthy, require_rsi_not_ob, require_volume, require_20d_breakout,
        min_rsi is not None, max_rsi is not None,
    ]):
        return True
    if not snap.ok:
        return False
    if require_supertrend and not snap.supertrend_bull:
        return False
    if require_ema_stack and not snap.ema_stack_bull:
        return False
    if require_ema_trend and not snap.ema_trend_bull:
        return False
    if require_bb_mid and not snap.bb_above_mid:
        return False
    if require_rsi_healthy and not snap.rsi_healthy:
        return False
    if require_rsi_not_ob and not snap.rsi_not_overbought:
        return False
    if require_volume and not snap.volume_ok:
        return False
    if require_20d_breakout and not snap.breakout_20d:
        return False
    if min_rsi is not None and snap.rsi < min_rsi:
        return False
    if max_rsi is not None and snap.rsi > max_rsi:
        return False
    return True
