"""
Cup-and-handle breakout swing backtester.

Find a rounded cup (previous high → 15–40% correction → rounded bottom →
recovery to ≥90% of the high → optional handle → close breaks resistance)
and buy the next open. Signal on bar T close only — no look-ahead.

Returns StageV2BacktestResult so the existing backtest UI/charts work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from stage_analysis_v2.services.backtester import (
    EntryFilters,
    StageV2BacktestResult,
    _OpenPos,
    attach_signal_log_from_raw,
    _close_trade,
    _compute_monthly_returns,
    _equity,
    _invested_total,
    _passes_entry_filters,
    _preload_frames,
    fill_performance_metrics,
    maybe_raise_stop,
    stop_exit_reason,
)
from stage_analysis_v2.services.strategy_catalog import (
    CUP_MAX_NEW_PER_DAY,
    DEFAULT_CUP_COOLDOWN_DAYS,
    DEFAULT_CUP_MAX_HOLD_DAYS,
    DEFAULT_CUP_MAX_POS_PCT,
    DEFAULT_CUP_RISK_PCT,
    DEFAULT_CUP_TARGET_RR,
    STRATEGY_CUP,
    STRATEGY_LABELS,
)
from stage_analysis_v2.services.supertrend_swing import _atr, _ema, _rsi
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CUP_EXIT_TARGET_R = "target_r"
CUP_EXIT_EMA20 = "ema20_trail"
CUP_EXIT_MEASURED = "measured_move"
VALID_CUP_EXITS = frozenset({CUP_EXIT_TARGET_R, CUP_EXIT_EMA20, CUP_EXIT_MEASURED})
CUP_EXIT_LABELS = {
    CUP_EXIT_TARGET_R: "Target R (stop distance × R)",
    CUP_EXIT_EMA20: "EMA20 trailing stop",
    CUP_EXIT_MEASURED: "Cup-depth measured move",
}
CUP_EXIT_CHOICES = [
    (CUP_EXIT_TARGET_R, "2R / 3R target (uses Target R:R)"),
    (CUP_EXIT_EMA20, "EMA20 trailing stop"),
    (CUP_EXIT_MEASURED, "Cup-depth measured-move target"),
]

CUP_ENTRY_NEXT_OPEN = "next_open"
CUP_ENTRY_RETEST = "retest"
VALID_CUP_ENTRIES = frozenset({CUP_ENTRY_NEXT_OPEN, CUP_ENTRY_RETEST})
CUP_ENTRY_LABELS = {
    CUP_ENTRY_NEXT_OPEN: "Breakout close → buy next open",
    CUP_ENTRY_RETEST: "Breakout + retest (hold above rim)",
}
CUP_ENTRY_CHOICES = [
    (CUP_ENTRY_NEXT_OPEN, "Breakout close → next open (default)"),
    (CUP_ENTRY_RETEST, "Breakout + retest within ~2%"),
]


def normalize_cup_exit(mode: str | None) -> str:
    m = (mode or CUP_EXIT_TARGET_R).strip().lower()
    return m if m in VALID_CUP_EXITS else CUP_EXIT_TARGET_R


def normalize_cup_entry(mode: str | None) -> str:
    m = (mode or CUP_ENTRY_NEXT_OPEN).strip().lower()
    return m if m in VALID_CUP_ENTRIES else CUP_ENTRY_NEXT_OPEN


def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    if len(s) < n:
        return out
    csum = np.cumsum(s, dtype=float)
    out[n - 1] = csum[n - 1] / n
    out[n:] = (csum[n:] - csum[:-n]) / n
    return out


@dataclass
class CupParams:
    cup_min_days: int = 20
    cup_max_days: int = 180
    min_depth_pct: float = 12.0
    max_depth_pct: float = 45.0
    recovery_pct: float = 90.0
    handle_min_days: int = 5
    handle_max_days: int = 30
    handle_max_depth_pct: float = 15.0
    require_handle: bool = False
    breakout_buffer_pct: float = 0.5
    vol_mult: float = 1.1
    rsi_min: float = 40.0
    rsi_max: float = 85.0
    max_gap_pct: float = 5.0
    min_close_loc: float = 0.70
    require_close_strength: bool = False
    require_sma200_rising: bool = False
    require_rs_vs_nifty: bool = False
    require_nifty_sma200: bool = False
    require_trend_stack: bool = True  # close>EMA20, close>SMA50, SMA50>SMA200
    sma200_slope_bars: int = 20
    min_bottom_days: int = 5
    bottom_zone_pct: float = 8.0
    min_left_days: int = 7
    min_recovery_days: int = 5
    pivot_width: int = 5
    stop_atr_mult: float = 0.5
    entry_mode: str = CUP_ENTRY_NEXT_OPEN
    retest_tol_pct: float = 2.0
    retest_max_days: int = 10
    cup_exit_mode: str = CUP_EXIT_EMA20
    target_rr: float = DEFAULT_CUP_TARGET_RR
    min_stop_pct: float = 0.012
    max_stop_pct: float = 0.18
    max_new_per_day: int = CUP_MAX_NEW_PER_DAY
    # 0 = trail/exit on EMA20 immediately. 1.0 = only after the trade is +1R.
    trail_arm_r: float = 0.0
    # Skip the fill if next-day open gaps down more than this % vs prior close. 0 = off.
    # Default 2% blocks election/crash opens (e.g. 4 Jun 2024) without changing 2023/2025.
    skip_entry_gap_down_pct: float = 2.0
    # Loss guards (0 = off). All use only data known at the fill.
    max_month_loss_pct: float = 0.0  # halt new entries after this % of month-start equity is lost
    halt_dd_pct: float = 0.0  # halt new entries when equity DD from peak exceeds this %
    max_open: int = 0  # cap concurrent names (0 = no cap)
    nifty_ema_period: int = 20  # require Nifty close > EMA(n) on the signal day; 0 = off
    nifty_gap_down_pct: float = 0.0  # skip fill if Nifty opens down more than this %
    loss_streak: int = 3  # halt new entries after this many consecutive losses
    loss_streak_cooloff_days: int = 10

    def __post_init__(self) -> None:
        self.entry_mode = normalize_cup_entry(self.entry_mode)
        self.cup_exit_mode = normalize_cup_exit(self.cup_exit_mode)
        self.cup_min_days = int(min(400, max(10, self.cup_min_days)))
        self.cup_max_days = int(min(500, max(self.cup_min_days, self.cup_max_days)))
        self.handle_min_days = int(min(60, max(0, self.handle_min_days)))
        self.handle_max_days = int(min(80, max(self.handle_min_days, self.handle_max_days)))
        self.retest_max_days = int(min(40, max(2, self.retest_max_days)))
        self.pivot_width = int(min(20, max(3, self.pivot_width)))
        self.target_rr = float(min(10.0, max(0.5, self.target_rr)))


@dataclass
class CupSetup:
    left_idx: int
    low_idx: int
    recovery_idx: int
    cup_high: float
    cup_low: float
    depth_pct: float
    duration: int
    recovery_pct: float
    handle_days: int
    handle_depth_pct: float
    has_handle: bool
    stop_ref: float
    breakout_price: float
    vol_mult: float
    rsi: float
    ema20: float
    sma20: float
    sma50: float
    sma200: float
    quality: int
    close_loc: float
    gap_pct: float
    sma200_rising: bool
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "cup_high": round(self.cup_high, 2),
            "cup_low": round(self.cup_low, 2),
            "cup_depth_pct": round(self.depth_pct, 2),
            "cup_duration": int(self.duration),
            "right_rim_recovery_pct": round(self.recovery_pct, 2),
            "handle_days": int(self.handle_days),
            "handle_depth_pct": round(self.handle_depth_pct, 2),
            "has_handle": bool(self.has_handle),
            "breakout_price": round(self.breakout_price, 2),
            "volume_multiple": round(self.vol_mult, 2),
            "rsi": round(self.rsi, 1) if self.rsi == self.rsi else None,
            "sma20": round(self.sma20, 2) if self.sma20 == self.sma20 else None,
            "sma50": round(self.sma50, 2) if self.sma50 == self.sma50 else None,
            "sma200": round(self.sma200, 2) if self.sma200 == self.sma200 else None,
            "ema20": round(self.ema20, 2) if self.ema20 == self.ema20 else None,
            "stop_ref": round(self.stop_ref, 2),
            "close_loc": round(self.close_loc, 2),
            "gap_pct": round(self.gap_pct, 2),
            "sma200_rising": bool(self.sma200_rising),
            "notes": self.notes,
        }


def _confirmed_peak(h: np.ndarray, L: int, i: int, w: int) -> bool:
    """True if L is a local high fully confirmed by bar i (no look-ahead)."""
    if L < w or L + w >= i or L >= i:
        return False
    return float(h[L]) >= float(np.max(h[L - w : L + w + 1])) - 1e-12


def _right_pullback_low(low: np.ndarray, start: int, end: int) -> float:
    """Most recent 3-bar swing low in [start, end). Fallback: min of the window."""
    if end <= start:
        return float(low[max(0, end - 1)])
    for j in range(end - 2, start, -1):
        if low[j] <= low[j - 1] and low[j] <= low[j + 1]:
            return float(low[j])
    return float(np.min(low[start:end]))


def _u_shape_ok(low: np.ndarray, left: int, recovery: int) -> bool:
    span = recovery - left
    if span < 15:
        return True
    t = span // 3
    left_m = float(np.mean(low[left : left + t]))
    mid_m = float(np.mean(low[left + t : left + 2 * t]))
    right_m = float(np.mean(low[left + 2 * t : recovery]))
    return mid_m <= left_m + 1e-9 and mid_m <= right_m + 1e-9


def detect_cup_shape(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    i: int,
    params: CupParams | None = None,
) -> Optional[CupSetup]:
    """
    Cup geometry at bar i using only bars 0..i (inclusive).

    Left rim is a confirmed pivot high 30–150 days ago. After it, price
    corrects 15–40%, spends time near the low (not a V), recovers to
    ≥ recovery_pct of the rim, optionally forms a handle, and today's
    close breaks the rim by `breakout_buffer_pct`.
    """
    p = params or CupParams()
    n = len(close)
    if i < 0 or i >= n or i < p.cup_min_days + p.pivot_width + 5:
        return None

    lo = i - p.cup_max_days
    hi = i - p.cup_min_days
    if lo < p.pivot_width:
        lo = p.pivot_width
    if hi <= lo:
        return None

    buffer = 1.0 + p.breakout_buffer_pct / 100.0
    recovery_frac = p.recovery_pct / 100.0
    min_depth = p.min_depth_pct / 100.0
    max_depth = p.max_depth_pct / 100.0
    bottom_zone = p.bottom_zone_pct / 100.0
    handle_max_d = p.handle_max_depth_pct / 100.0

    # Highest first — that is the resistance today's close must break.
    order = lo + np.argsort(-high[lo : hi + 1])
    tried = 0
    for L in order:
        L = int(L)
        if tried >= 10:
            break
        if not _confirmed_peak(high, L, i, p.pivot_width):
            continue
        tried += 1
        cup_high = float(high[L])
        if cup_high <= 0:
            continue
        if float(close[i]) < cup_high * buffer:
            continue
        # First close-breakout only (wicks during the handle are allowed).
        if i - L > 1 and float(np.max(close[L + 1 : i])) >= cup_high * buffer:
            continue

        cup_low_rel = int(np.argmin(low[L + 1 : i]))
        low_idx = L + 1 + cup_low_rel
        cup_low = float(low[low_idx])
        if cup_low <= 0 or cup_low >= cup_high:
            continue
        depth = (cup_high - cup_low) / cup_high
        if depth < min_depth or depth > max_depth:
            continue
        if low_idx - L < p.min_left_days:
            continue
        if i - low_idx < p.min_recovery_days:
            continue

        rec_level = cup_high * recovery_frac
        rec_idx = -1
        for j in range(low_idx + 1, i):
            if float(close[j]) >= rec_level:
                rec_idx = j
                break
        if rec_idx < 0:
            continue
        if rec_idx - low_idx < p.min_recovery_days:
            continue

        near = low[L:i] <= (cup_low + bottom_zone * cup_high)
        if int(np.count_nonzero(near)) < p.min_bottom_days:
            continue
        if not _u_shape_ok(low, L, rec_idx):
            # Soft: allow if the stock spent extra time in the bottom zone.
            if int(np.count_nonzero(near)) < p.min_bottom_days + 4:
                continue

        handle_days = i - rec_idx
        handle_low = float(np.min(low[rec_idx:i])) if handle_days > 0 else cup_high
        handle_depth = (cup_high - handle_low) / cup_high if cup_high else 0.0
        in_handle_window = p.handle_min_days <= handle_days <= p.handle_max_days
        has_handle = bool(in_handle_window and handle_depth <= handle_max_d + 1e-12)

        if in_handle_window and handle_depth > handle_max_d:
            continue  # handle exists but is too deep
        if p.require_handle and not has_handle:
            continue

        if has_handle:
            stop_ref = handle_low
        else:
            pull_start = max(rec_idx, low_idx + 1)
            stop_ref = _right_pullback_low(low, pull_start, i)

        right_high = float(np.max(close[low_idx:i]))
        recovery_now = right_high / cup_high * 100.0
        duration = i - L
        return CupSetup(
            left_idx=L,
            low_idx=low_idx,
            recovery_idx=rec_idx,
            cup_high=cup_high,
            cup_low=cup_low,
            depth_pct=depth * 100.0,
            duration=duration,
            recovery_pct=recovery_now,
            handle_days=handle_days if has_handle else 0,
            handle_depth_pct=(handle_depth * 100.0) if has_handle else 0.0,
            has_handle=has_handle,
            stop_ref=stop_ref,
            breakout_price=float(close[i]),
            vol_mult=0.0,
            rsi=float("nan"),
            ema20=float("nan"),
            sma20=float("nan"),
            sma50=float("nan"),
            sma200=float("nan"),
            quality=0,
            close_loc=0.0,
            gap_pct=0.0,
            sma200_rising=False,
        )
    return None


def _cup_quality(setup: CupSetup, sma200_rising: bool) -> int:
    score = 0
    if 45 <= setup.duration <= 100:
        score += 20
    elif 30 <= setup.duration <= 150:
        score += 10
    if 20.0 <= setup.depth_pct <= 30.0:
        score += 20
    elif 15.0 <= setup.depth_pct <= 40.0:
        score += 10
    if setup.has_handle:
        score += 15
    if setup.vol_mult >= 2.0:
        score += 15
    elif setup.vol_mult >= 1.5:
        score += 10
    if setup.close_loc >= 0.80:
        score += 10
    elif setup.close_loc >= 0.70:
        score += 5
    rsi = setup.rsi
    if rsi == rsi and 55.0 <= rsi <= 70.0:
        score += 10
    elif rsi == rsi and 50.0 <= rsi <= 80.0:
        score += 5
    if sma200_rising:
        score += 10
    return int(min(100, score))


def _retest_status(
    low: np.ndarray,
    close: np.ndarray,
    j: int,
    cup_high: float,
    tol: float,
) -> str:
    """'ok' / 'fail' / 'wait' using only bar j vs the cup rim."""
    if cup_high <= 0:
        return "fail"
    if float(close[j]) <= cup_high:
        return "fail"
    if float(low[j]) < cup_high * (1.0 - tol):
        return "fail"
    if float(low[j]) <= cup_high * (1.0 + tol):
        return "ok"
    return "wait"


class _CupPack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c", "v",
        "ema20", "sma20", "sma50", "sma200", "rsi", "atr", "ret60",
        "vol_sma20", "close_loc",
    )

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.index = df.index
        self.loc = {ts: i for i, ts in enumerate(df.index)}
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.ema20 = _ema(c, 20)
        self.sma20 = _sma(c, 20)
        self.sma50 = _sma(c, 50)
        self.sma200 = _sma(c, 200)
        self.rsi = _rsi(c, 14)
        self.atr = _atr(h, l, c, 14)
        self.ret60 = pd.Series(c, index=df.index).pct_change(60).to_numpy()
        self.vol_sma20 = _sma(v, 20)
        rng = np.maximum(h - l, 1e-9)
        self.close_loc = (c - l) / rng


def _passes_breakout_filters(
    s: _CupPack,
    i: int,
    setup: CupSetup,
    params: CupParams,
    nifty_ret60: Optional[pd.Series],
    nifty_sma200: Optional[pd.Series],
    nifty_close: Optional[pd.Series],
) -> bool:
    c = float(s.c[i])
    e20, s50, s200 = float(s.ema20[i]), float(s.sma50[i]), float(s.sma200[i])
    if params.require_trend_stack:
        if any(x != x for x in (e20, s50, s200)):  # NaN
            return False
        if not (c > e20 and c > s50 and s50 > s200):
            return False

    slope_i = i - params.sma200_slope_bars
    rising = False
    if slope_i >= 0 and s.sma200[slope_i] == s.sma200[slope_i]:
        rising = s200 > float(s.sma200[slope_i])
    setup.sma200_rising = rising
    if params.require_sma200_rising and not rising:
        return False

    rsi = float(s.rsi[i])
    setup.rsi = rsi
    if rsi != rsi or rsi < params.rsi_min or rsi > params.rsi_max:
        return False

    avg_vol = float(s.vol_sma20[i]) if s.vol_sma20[i] == s.vol_sma20[i] else 0.0
    # vol_sma includes today; use prior 20 bars when possible (no look-ahead, just cleaner).
    if i >= 20:
        avg_vol = float(np.mean(s.v[i - 20 : i]))
    if avg_vol <= 0:
        return False
    vol_mult = float(s.v[i]) / avg_vol
    if vol_mult < params.vol_mult:
        return False
    setup.vol_mult = vol_mult

    if i == 0:
        return False
    prev = float(s.c[i - 1])
    gap_pct = ((float(s.o[i]) - prev) / prev * 100.0) if prev > 0 else 0.0
    setup.gap_pct = gap_pct
    if gap_pct > params.max_gap_pct:
        return False

    loc = float(s.close_loc[i])
    setup.close_loc = loc
    if params.require_close_strength and loc < params.min_close_loc:
        return False

    setup.ema20 = e20
    setup.sma20 = float(s.sma20[i])
    setup.sma50 = s50
    setup.sma200 = s200

    ts = s.index[i]
    if params.require_rs_vs_nifty and nifty_ret60 is not None and len(nifty_ret60):
        stock_r = float(s.ret60[i]) if s.ret60[i] == s.ret60[i] else None
        n_r = None
        if ts in nifty_ret60.index:
            val = nifty_ret60.loc[ts]
            if pd.notna(val):
                n_r = float(val)
        if stock_r is None or n_r is None or stock_r <= n_r:
            return False

    if params.require_nifty_sma200 and nifty_close is not None and nifty_sma200 is not None:
        if ts not in nifty_close.index or ts not in nifty_sma200.index:
            return False
        nc, ns = nifty_close.loc[ts], nifty_sma200.loc[ts]
        if pd.isna(nc) or pd.isna(ns) or float(nc) <= float(ns):
            return False

    setup.quality = _cup_quality(setup, rising)
    return True


def _target_price(entry: float, stop: float, setup: CupSetup, params: CupParams) -> float:
    mode = params.cup_exit_mode
    if mode == CUP_EXIT_MEASURED:
        depth = setup.cup_high - setup.cup_low
        return round(setup.cup_high + depth, 2) if depth > 0 else 0.0
    if mode == CUP_EXIT_EMA20:
        return 0.0
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return round(entry + params.target_rr * risk, 2)


def run_cup_breakout_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    strategy_id: str = STRATEGY_CUP,
    risk_pct: float | None = None,
    max_hold_days: int | None = None,
    cooldown_days: int | None = None,
    max_pos_pct: float | None = None,
    entry_filters: EntryFilters | None = None,
    params: CupParams | None = None,
) -> StageV2BacktestResult:
    """
    Cup-and-handle breakout swing on the given universe.

    Signal on breakout close (bar T). Default fill is T+1 open.
    Optional retest mode waits for a hold-above retest within ~2%.
    """
    p = params or CupParams()
    if risk_pct is None:
        risk_pct = DEFAULT_CUP_RISK_PCT
    risk_pct = min(50.0, max(0.1, float(risk_pct)))
    if max_hold_days is None:
        max_hold_days = DEFAULT_CUP_MAX_HOLD_DAYS
    max_hold_days = min(500, max(1, int(max_hold_days)))
    if cooldown_days is None:
        cooldown_days = DEFAULT_CUP_COOLDOWN_DAYS
    cooldown_days = min(365, max(0, int(cooldown_days)))
    if max_pos_pct is None:
        max_pos_pct = DEFAULT_CUP_MAX_POS_PCT
    max_pos_pct = min(100.0, max(5.0, float(max_pos_pct)))
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365 * 3))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]
    filters = entry_filters or EntryFilters()

    exit_label = CUP_EXIT_LABELS.get(p.cup_exit_mode, p.cup_exit_mode)
    if p.cup_exit_mode == CUP_EXIT_TARGET_R:
        exit_label = f"{p.target_rr:g}R target"
    entry_label = CUP_ENTRY_LABELS.get(p.entry_mode, p.entry_mode)
    handle_label = "require handle" if p.require_handle else "handle preferred"
    filter_label = (
        f"Cup {p.cup_min_days}–{p.cup_max_days}d, depth {p.min_depth_pct:g}–{p.max_depth_pct:g}%, "
        f"recover ≥{p.recovery_pct:g}%, {handle_label}, vol ≥{p.vol_mult:g}×, "
        f"{entry_label}"
    )

    result = StageV2BacktestResult(
        strategy_name=STRATEGY_LABELS.get(strategy_id, "Cup Breakout Strategy"),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=0,
        market_filter=p.require_nifty_sma200,
        exit_mode=p.cup_exit_mode,
        exit_mode_label=f"{exit_label} · max {max_hold_days}d",
        tech_filter="cup_breakout",
        tech_filter_label=filter_label,
        entry_stage=0,
        entry_on=p.entry_mode,
        entry_filters_label=filters.active_summary(),
        target_rr=p.target_rr if p.cup_exit_mode == CUP_EXIT_TARGET_R else 0.0,
        max_hold_days=max_hold_days,
        stop_ma_mult=1.0,
        trail_ma_mult=1.0,
        risk_pct=risk_pct,
        cooldown_days=cooldown_days,
        strategy_id=strategy_id,
        max_pos_pct=max_pos_pct,
    )

    frames = _preload_frames(symbols)
    result.stocks_scanned = len(frames)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    packs: dict[str, _CupPack] = {}
    for sym, df in frames.items():
        packs[sym] = _CupPack(sym, df)

    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret60 = nifty_df["close"].pct_change(60) if not nifty_df.empty else None
    nifty_sma200 = nifty_df["close"].rolling(200, min_periods=200).mean() if not nifty_df.empty else None
    nifty_close = nifty_df["close"] if not nifty_df.empty else None
    nifty_open = nifty_df["open"] if not nifty_df.empty else None
    nifty_prev_close = nifty_df["close"].shift(1) if not nifty_df.empty else None
    nifty_ema = None
    if p.nifty_ema_period and nifty_close is not None:
        nifty_ema = nifty_close.ewm(span=int(p.nifty_ema_period), adjust=False).mean()
    nifty_available = bool(nifty_ret60 is not None and len(nifty_ret60.dropna()))
    if not nifty_available:
        # User: apply RS / index filters only when Nifty 50 data exists.
        p.require_rs_vs_nifty = False
        p.require_nifty_sma200 = False

    warmup = max(210, p.cup_max_days + p.pivot_width + 5)
    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n_sig = 0
    for sym, s in packs.items():
        daily = frames[sym]
        n = len(s.c)
        for i in range(warmup, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            setup = detect_cup_shape(s.h, s.l, s.c, i, p)
            if setup is None:
                continue
            if not _passes_breakout_filters(
                s, i, setup, p, nifty_ret60, nifty_sma200, nifty_close,
            ):
                continue
            if not _passes_entry_filters(daily, ts, filters):
                continue

            atr = float(s.atr[i]) if s.atr[i] == s.atr[i] else 0.0
            rs = 50.0
            stock_r = float(s.ret60[i]) if s.ret60[i] == s.ret60[i] else 0.0
            if nifty_ret60 is not None and ts in nifty_ret60.index and pd.notna(nifty_ret60.loc[ts]):
                n_r = float(nifty_ret60.loc[ts])
                rs = min(99.0, max(1.0, 50.0 + (stock_r - n_r) * 200.0))

            entry_i = i + 1
            sig_ts = ts
            if p.entry_mode == CUP_ENTRY_RETEST:
                tol = p.retest_tol_pct / 100.0
                found = False
                j_end = min(n - 1, i + 1 + p.retest_max_days)
                for j in range(i + 1, j_end):
                    status = _retest_status(s.l, s.c, j, setup.cup_high, tol)
                    if status == "fail":
                        break
                    if status == "ok":
                        if s.index[j] > end_ts:
                            break
                        entry_i = j + 1
                        sig_ts = s.index[j]
                        found = True
                        break
                if not found or entry_i >= n:
                    continue

            if entry_i >= n:
                continue
            signals_by_day[s.index[entry_i]].append({
                "symbol": sym,
                "sig_ts": sig_ts,
                "breakout_ts": ts,
                "entry_ts": s.index[entry_i],
                "setup": setup,
                "atr": atr,
                "score": setup.quality,
                "quality": setup.quality,
                "rs": round(rs, 1),
            })
            n_sig += 1

    result.stage2_entries = n_sig
    result.total_signals = n_sig

    calendar = sorted({
        ts
        for df in frames.values()
        for ts in df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist()
    })
    if not calendar:
        result.final_cash = capital
        result.final_capital = capital
        result.equity_curve = [{"date": str(start_date), "equity": capital}]
        return result

    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve: list[dict] = [{"date": str(start_date), "equity": capital}]
    peak_parallel = 0
    skipped_cash = 0
    last_curve_date: Optional[date] = None
    peak_eq = float(capital)
    month_key = ""
    month_start_eq = float(capital)
    month_pnl = 0.0
    consec_losses = 0
    cooloff_until: Optional[pd.Timestamp] = None

    def _record_curve(ts: pd.Timestamp, force: bool = False) -> None:
        nonlocal last_curve_date
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if not force and last_curve_date is not None and (d - last_curve_date).days < 1:
            return
        eq = _equity(cash, opens)
        equity_curve.append({"date": str(d), "equity": round(eq, 2)})
        last_curve_date = d

    for ts in calendar:
        closed_today: list[str] = []
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            pos.hold_days += 1
            close = float(s.c[i])
            high = float(s.h[i])
            low = float(s.l[i])
            e20 = float(s.ema20[i]) if s.ema20[i] == s.ema20[i] else 0.0

            exit_price = None
            exit_reason = ""
            # Carry-in stop first (no same-bar trail look-ahead).
            risk_per_share = pos.entry_price - (pos.entry_stop if pos.entry_stop else pos.stop)
            armed = (
                p.trail_arm_r <= 0
                or risk_per_share <= 0
                or (close - pos.entry_price) >= p.trail_arm_r * risk_per_share
            )
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, stop_exit_reason(pos)
            elif pos.target > 0 and high >= pos.target:
                exit_price, exit_reason = pos.target, "target"
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif (
                p.cup_exit_mode == CUP_EXIT_EMA20
                and armed
                and e20 > 0
                and close < e20
            ):
                exit_price, exit_reason = close, "ema20_exit"

            if exit_price is None:
                if p.cup_exit_mode == CUP_EXIT_EMA20 and armed and e20 > 0:
                    maybe_raise_stop(pos, e20, close)
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            trade.setup = dict(pos.setup)
            trade.setup.update({
                "entry": round(pos.entry_price, 2),
                "stop": round(pos.entry_stop if pos.entry_stop else pos.stop, 2),
                "target": round(pos.target, 2),
                "exit": round(exit_price, 2),
                "pnl": trade.pnl,
            })
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)
            month_pnl += trade.pnl
            if trade.pnl <= 0:
                consec_losses += 1
                if p.loss_streak > 0 and consec_losses >= p.loss_streak and p.loss_streak_cooloff_days > 0:
                    cooloff_until = ts + pd.Timedelta(days=int(p.loss_streak_cooloff_days))
            else:
                consec_losses = 0
        for sym in closed_today:
            opens.pop(sym, None)
        if closed_today:
            _record_curve(ts, force=True)

        mk = f"{ts.year:04d}-{ts.month:02d}"
        if mk != month_key:
            month_key = mk
            month_start_eq = _equity(cash, opens)
            month_pnl = 0.0
        peak_eq = max(peak_eq, _equity(cash, opens))

        day_sigs = signals_by_day.get(ts, [])
        if day_sigs:
            day_sigs = sorted(day_sigs, key=lambda x: x["score"], reverse=True)
            taken = 0
            for sig in day_sigs:
                if taken >= p.max_new_per_day:
                    break
                if p.max_open > 0 and len(opens) >= p.max_open:
                    break
                if cooloff_until is not None and ts < cooloff_until:
                    break
                eq_now = _equity(cash, opens)
                if p.halt_dd_pct > 0 and peak_eq > 0:
                    dd = (peak_eq - eq_now) / peak_eq * 100.0
                    if dd >= p.halt_dd_pct:
                        break
                if p.max_month_loss_pct > 0 and month_start_eq > 0:
                    if month_pnl <= -(p.max_month_loss_pct / 100.0) * month_start_eq:
                        break
                if p.nifty_ema_period and nifty_ema is not None and nifty_close is not None:
                    sig_ts = sig.get("sig_ts", ts)
                    if sig_ts in nifty_close.index and sig_ts in nifty_ema.index:
                        nc, ne = nifty_close.loc[sig_ts], nifty_ema.loc[sig_ts]
                        if pd.notna(nc) and pd.notna(ne) and float(nc) <= float(ne):
                            continue
                if p.nifty_gap_down_pct > 0 and nifty_open is not None and nifty_prev_close is not None:
                    if ts in nifty_open.index and ts in nifty_prev_close.index:
                        no, npv = nifty_open.loc[ts], nifty_prev_close.loc[ts]
                        if pd.notna(no) and pd.notna(npv) and float(npv) > 0:
                            gap = (float(npv) - float(no)) / float(npv) * 100.0
                            if gap > p.nifty_gap_down_pct:
                                continue
                sym = sig["symbol"]
                if sym in opens:
                    continue
                s = packs.get(sym)
                if s is None:
                    continue
                prev = last_exit.get(sym)
                if cooldown_days > 0 and prev is not None and (ts - prev).days < cooldown_days:
                    continue
                i = s.loc.get(ts)
                if i is None:
                    continue
                entry = float(s.o[i])
                if entry <= 0:
                    continue
                if p.skip_entry_gap_down_pct > 0 and i > 0:
                    prev_c = float(s.c[i - 1])
                    if prev_c > 0:
                        gap_dn = (prev_c - entry) / prev_c * 100.0
                        if gap_dn > p.skip_entry_gap_down_pct:
                            continue
                setup: CupSetup = sig["setup"]
                atr = float(sig.get("atr") or 0.0)
                stop = float(setup.stop_ref) - p.stop_atr_mult * atr
                if stop <= 0 or stop >= entry:
                    fallback = entry - (1.0 * atr if atr > 0 else entry * 0.04)
                    stop = fallback if 0 < fallback < entry else 0.0
                if stop <= 0 or stop >= entry:
                    continue
                risk = entry - stop
                stop_pct = risk / entry
                if stop_pct > p.max_stop_pct or stop_pct < p.min_stop_pct:
                    continue
                target = _target_price(entry, stop, setup, p)
                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    skipped_cash += 1
                    continue
                ps = calculate_position_size(equity_now, risk_pct, entry, stop)
                max_notional = equity_now * (max_pos_pct / 100.0)
                qty_cash = int(cash // entry) if entry else 0
                qty_cap = int(max_notional // entry) if entry else 0
                qty = min(int(ps.quantity), qty_cash, qty_cap)
                if qty <= 0:
                    skipped_cash += 1
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    skipped_cash += 1
                    continue
                cash -= notional
                invested_after = _invested_total(opens) + notional
                parallel = len(opens) + 1
                peak_parallel = max(peak_parallel, parallel)
                setup_dict = setup.as_dict()
                setup_dict.update({
                    "entry": round(entry, 2),
                    "stop": round(stop, 2),
                    "target": round(target, 2),
                    "atr": round(atr, 2),
                })
                pos = _OpenPos(
                    symbol=sym,
                    entry_date=ts,
                    signal_date=sig["sig_ts"],
                    entry_price=entry,
                    stop=stop,
                    target=target,
                    qty=qty,
                    notional=notional,
                    quality_score=int(sig["quality"]),
                    rs_rating=float(sig["rs"]),
                    weekly_stage=0,
                    hold_days=0,
                    capital_invested=notional,
                    cash_available=cash,
                    total_invested=invested_after,
                    parallel_open=parallel,
                    equity_at_entry=cash + invested_after,
                    entry_stop=stop,
                    setup=setup_dict,
                )
                opens[sym] = pos
                taken += 1

                if s.l[i] <= stop:
                    trade = _close_trade(
                        pos, exit_price=stop, exit_ts=ts,
                        exit_reason="stop_loss", days_held=0,
                    )
                    trade.setup = dict(setup_dict)
                    trade.setup.update({"exit": round(stop, 2), "pnl": trade.pnl})
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                    month_pnl += trade.pnl
                    if trade.pnl <= 0:
                        consec_losses += 1
                        if p.loss_streak > 0 and consec_losses >= p.loss_streak and p.loss_streak_cooloff_days > 0:
                            cooloff_until = ts + pd.Timedelta(days=int(p.loss_streak_cooloff_days))
                    else:
                        consec_losses = 0
            _record_curve(ts, force=True)

        if ts == calendar[-1] or ts.weekday() == 4:
            _record_curve(ts, force=(ts == calendar[-1]))

    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(last_ts, len(s.c) - 1)
            close = float(s.c[i])
            trade = _close_trade(
                pos, exit_price=close, exit_ts=last_ts,
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            trade.setup = dict(pos.setup)
            trade.setup.update({"exit": round(close, 2), "pnl": trade.pnl})
            cash += pos.notional + trade.pnl
            trades.append(trade)
        opens.clear()
        _record_curve(last_ts, force=True)

    trades.sort(key=lambda t: (t.entry_date, t.symbol))
    result.trades = trades
    result.total_trades = len(trades)
    result.equity_curve = equity_curve
    result.peak_parallel = peak_parallel
    result.signals_skipped_cash = skipped_cash
    result.final_cash = round(cash, 2)
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1e-9
        result.win_rate = round(len(wins) / len(trades) * 100, 2)
        result.profit_factor = round(gp / gl, 2)
        result.avg_rr = round(sum(t.rr_achieved for t in trades) / len(trades), 2)
        result.avg_hold_days = round(sum(t.days_held for t in trades) / len(trades), 1)
        result.expectancy_r = result.avg_rr
    result.total_return_pct = round((cash - capital) / capital * 100, 2) if capital else 0.0
    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        if peak:
            max_dd = max(max_dd, (peak - e) / peak * 100)
    result.max_drawdown_pct = round(max_dd, 2)
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns = _compute_monthly_returns(trades, capital)
    raw_signals = []
    for day_ts, day_list in signals_by_day.items():
        for sig in day_list:
            setup = sig.get("setup")
            stop = float(getattr(setup, "stop_ref", 0) or 0) if setup is not None else 0.0
            brk = float(getattr(setup, "breakout_price", 0) or 0) if setup is not None else 0.0
            raw_signals.append({
                "symbol": sig["symbol"],
                "signal_date": str(pd.Timestamp(sig["sig_ts"]).date()),
                "entry_date": str(pd.Timestamp(sig.get("entry_ts") or day_ts).date()),
                "signal_close": round(brk, 2) if brk else None,
                "stop_loss": round(stop, 2) if stop else None,
                "target": 0.0,
                "quality_score": int(sig.get("quality") or 0),
                "rs_rating": float(sig.get("rs") or 0),
            })
    attach_signal_log_from_raw(result, raw_signals)
    fill_performance_metrics(result)
    return result
