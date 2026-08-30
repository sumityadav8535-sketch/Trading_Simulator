"""
Walk-forward backtest for Stage Analysis 2.0 — buy on weekly Stage 2 entry.

Entry: weekly stage transitions into Stage 2 (Weinstein advancing phase).
Exit: stop below 30-week MA, 2.5R target, stage rule (configurable), or max hold.

Exit modes (exit_mode):
  - stage_3_4     — exit when weekly stage becomes 3 or 4 (legacy)
  - stage_4_only  — exit only on Stage 4 (recommended default)
  - trail_ma_s4   — trail stop to 0.98×30w MA weekly + Stage 4 force exit
  - no_stage      — no stage exit (stop / target / time only; research)

Capital model (single account, cash only — no extra leverage):
  - Shared starting capital across all stocks
  - Calendar-order simulation so parallel positions share the same cash pool
  - Position size = risk % of equity, capped so notional ≤ free cash
  - Each trade records: invested ₹, free cash left, parallel open count
"""
from __future__ import annotations

import copy
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.breakout_detector import detect_breakout
from stage_analysis_v2.services.indicators import add_daily_indicators, add_weekly_indicators
from stage_analysis_v2.services.quality_score import compute_quality_score
from stage_analysis_v2.services.relative_strength import compute_relative_strength
from stage_analysis_v2.services.stage_engine import detect_daily_stage, detect_weekly_stage
from stage_analysis_v2.services.tech_filters import (
    DEFAULT_TECH_FILTER,
    TECH_FILTER_LABELS,
    enrich_daily_tech,
    normalize_tech_filter,
    pack_needs_supertrend,
    pack_needs_tech_snapshot,
    passes_tech_filter,
    snapshot_tech,
)
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Stage Analysis 2.0 — Buy at Stage 2"
COOLDOWN_DAYS = 40
MAX_HOLD_DAYS = 65
TARGET_RR = 2.5
MIN_WEEKLY_BARS = 38
TRAIL_MA_MULT = 0.98
STOP_MA_MULT = 0.95  # stop = 30w MA × this (default 5% under MA)
DEFAULT_MAX_HOLD_DAYS = MAX_HOLD_DAYS
DEFAULT_TARGET_RR = TARGET_RR
DEFAULT_STOP_MA_MULT = STOP_MA_MULT
DEFAULT_TRAIL_MA_MULT = TRAIL_MA_MULT
DEFAULT_COOLDOWN_DAYS = COOLDOWN_DAYS
DEFAULT_RISK_PCT = 2.0

# Entry stage selection (weekly Weinstein stage)
ENTRY_STAGE_CHOICES = [
    (1, "Stage 1 (basing)"),
    (2, "Stage 2 (advancing) — default"),
    (3, "Stage 3 (topping)"),
    (4, "Stage 4 (declining)"),
]
DEFAULT_ENTRY_STAGE = 2
VALID_ENTRY_STAGES = frozenset({1, 2, 3, 4})

ENTRY_ON_TRANSITION = "transition"  # only when stage newly becomes target
ENTRY_ON_IN_STAGE = "in_stage"  # any week currently in target stage
DEFAULT_ENTRY_ON = ENTRY_ON_TRANSITION
ENTRY_ON_CHOICES = [
    (ENTRY_ON_TRANSITION, "Transition into stage (recommended)"),
    (ENTRY_ON_IN_STAGE, "Any week while in stage"),
]
VALID_ENTRY_ON = frozenset({ENTRY_ON_TRANSITION, ENTRY_ON_IN_STAGE})

MA_COND_NONE = "none"
MA_COND_ABOVE = "above"
MA_COND_BELOW = "below"
MA_CONDITION_CHOICES = [
    (MA_COND_NONE, "No MA filter"),
    (MA_COND_ABOVE, "Price above MA"),
    (MA_COND_BELOW, "Price below MA"),
]
DEFAULT_MA_PERIOD = 50
DEFAULT_MA_TYPE = "sma"  # sma | ema


def normalize_entry_stage(entry_stage: int | str | None) -> int:
    try:
        stage = int(entry_stage)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_ENTRY_STAGE
    if stage not in VALID_ENTRY_STAGES:
        return DEFAULT_ENTRY_STAGE
    return stage


def normalize_entry_on(entry_on: str | None) -> str:
    mode = (entry_on or DEFAULT_ENTRY_ON).strip().lower()
    if mode not in VALID_ENTRY_ON:
        return DEFAULT_ENTRY_ON
    return mode


@dataclass
class EntryFilters:
    """Optional stock filters applied at signal week-end (using daily data)."""
    min_price: float = 0.0  # 0 = off
    max_price: float = 0.0  # 0 = off
    min_volume: float = 0.0  # absolute volume, 0 = off
    min_volume_ratio: float = 0.0  # vs 20d avg, 0 = off
    ma_period: int = 0  # 0 = MA filter off
    ma_type: str = DEFAULT_MA_TYPE  # sma | ema
    ma_condition: str = MA_COND_NONE  # none | above | below

    def active_summary(self) -> str:
        parts: list[str] = []
        if self.min_price > 0:
            parts.append(f"price≥{self.min_price:g}")
        if self.max_price > 0:
            parts.append(f"price≤{self.max_price:g}")
        if self.min_volume > 0:
            parts.append(f"vol≥{self.min_volume:,.0f}")
        if self.min_volume_ratio > 0:
            parts.append(f"vol≥{self.min_volume_ratio:g}×avg20")
        if self.ma_period > 0 and self.ma_condition != MA_COND_NONE:
            parts.append(
                f"price {self.ma_condition} {self.ma_type.upper()}{self.ma_period}"
            )
        return ", ".join(parts) if parts else "none"

# Exit mode ids (form / CLI / API)
EXIT_STAGE_3_4 = "stage_3_4"
EXIT_STAGE_4_ONLY = "stage_4_only"
EXIT_TRAIL_MA_S4 = "trail_ma_s4"
EXIT_NO_STAGE = "no_stage"
DEFAULT_EXIT_MODE = EXIT_STAGE_4_ONLY

EXIT_MODE_CHOICES = [
    (EXIT_STAGE_4_ONLY, "Stage 4 only (recommended)"),
    (EXIT_TRAIL_MA_S4, "Trail 30w MA + Stage 4"),
    (EXIT_STAGE_3_4, "Stage 3 or 4 (legacy)"),
    (EXIT_NO_STAGE, "No stage exit (research)"),
]

EXIT_MODE_LABELS = {k: v for k, v in EXIT_MODE_CHOICES}
VALID_EXIT_MODES = frozenset(EXIT_MODE_LABELS.keys())


def normalize_exit_mode(exit_mode: str | None) -> str:
    mode = (exit_mode or DEFAULT_EXIT_MODE).strip().lower()
    if mode not in VALID_EXIT_MODES:
        return DEFAULT_EXIT_MODE
    return mode


@dataclass
class StageV2Trade:
    symbol: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    pnl: float
    pnl_pct: float
    rr_achieved: float
    exit_reason: str
    quality_score: int
    rs_rating: float
    weekly_stage: int
    days_held: int
    # Capital account snapshot (₹) — single shared capital pool
    capital_invested: float = 0.0  # notional put into this trade
    cash_available: float = 0.0  # free cash after this entry (for next trade)
    total_invested: float = 0.0  # sum of all open notionals after this entry
    parallel_open: int = 0  # open positions after this entry (incl. this one)
    equity_at_entry: float = 0.0  # cash + invested (cost basis) at entry
    setup: dict = field(default_factory=dict)  # optional strategy-specific debug


@dataclass
class StageV2BacktestResult:
    strategy_name: str
    symbols: list[str]
    start_date: date
    end_date: date
    capital: float
    min_quality_score: int
    market_filter: bool
    min_rs_rating: float = 0.0
    exit_mode: str = DEFAULT_EXIT_MODE
    exit_mode_label: str = ""
    tech_filter: str = DEFAULT_TECH_FILTER
    tech_filter_label: str = ""
    entry_stage: int = DEFAULT_ENTRY_STAGE
    entry_on: str = DEFAULT_ENTRY_ON
    entry_filters_label: str = "none"
    target_rr: float = DEFAULT_TARGET_RR
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS
    stop_ma_mult: float = DEFAULT_STOP_MA_MULT
    trail_ma_mult: float = DEFAULT_TRAIL_MA_MULT
    risk_pct: float = DEFAULT_RISK_PCT
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS
    trades: list[StageV2Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)
    exit_breakdown: dict = field(default_factory=dict)
    total_trades: int = 0
    total_signals: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_rr: float = 0.0
    total_return_pct: float = 0.0
    avg_hold_days: float = 0.0
    expectancy_r: float = 0.0
    stocks_scanned: int = 0
    stage2_entries: int = 0
    peak_parallel: int = 0
    signals_skipped_cash: int = 0
    final_cash: float = 0.0
    # Every Stage 2 signal with portfolio execution status (taken / skipped_*)
    signal_log: list[dict] = field(default_factory=list)
    strategy_id: str = "stage_v2"
    max_pos_pct: float = 100.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    avg_trade: float = 0.0
    final_capital: float = 0.0
    shared_capital: bool = True
    capital_compare: dict = field(default_factory=dict)


@dataclass
class _OpenPos:
    symbol: str
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    qty: int
    notional: float
    quality_score: int
    rs_rating: float
    weekly_stage: int
    hold_days: int = 0
    last_week_check: Optional[pd.Timestamp] = None
    # Snapshots frozen at entry for the trade log
    capital_invested: float = 0.0
    cash_available: float = 0.0
    total_invested: float = 0.0
    parallel_open: int = 0
    equity_at_entry: float = 0.0
    entry_stop: float = 0.0  # original stop at entry (R uses this if stop is trailed)
    setup: dict = field(default_factory=dict)


def _weekly_stage_at(weekly: pd.DataFrame, end_idx: int) -> tuple[int, dict]:
    window = weekly.iloc[: end_idx + 1]
    if len(window) < MIN_WEEKLY_BARS:
        return 0, {}
    try:
        stage, _, metrics = detect_weekly_stage(window)
        return stage, metrics
    except ValueError:
        return 0, {}


def _market_favorable_at(nifty_weekly: pd.DataFrame, week_end: pd.Timestamp) -> bool:
    mask = nifty_weekly.index <= week_end
    window = nifty_weekly.loc[mask]
    if len(window) < MIN_WEEKLY_BARS:
        return True
    try:
        stage, _, _ = detect_weekly_stage(window)
        return stage in (1, 2)
    except ValueError:
        return True


def _passes_entry_filters(
    daily: pd.DataFrame,
    week_end: pd.Timestamp,
    filters: EntryFilters,
) -> bool:
    """Price / volume / MA checks on last daily bar on or before week_end."""
    if daily is None or daily.empty:
        return False
    d = daily.loc[daily.index <= week_end]
    if d.empty:
        return False
    row = d.iloc[-1]
    close = float(row["close"])
    vol = float(row.get("volume") or 0)

    if filters.min_price > 0 and close < filters.min_price:
        return False
    if filters.max_price > 0 and close > filters.max_price:
        return False
    if filters.min_volume > 0 and vol < filters.min_volume:
        return False

    need_vol_ratio = filters.min_volume_ratio > 0
    need_ma = (
        filters.ma_period > 0
        and filters.ma_condition in (MA_COND_ABOVE, MA_COND_BELOW)
    )
    if not need_vol_ratio and not need_ma:
        return True

    # Need enough history for rolling stats
    lookback = max(
        int(filters.ma_period or 0) + 2,
        22 if need_vol_ratio else 0,
        5,
    )
    if len(d) < lookback:
        return False

    if need_vol_ratio:
        avg_vol = float(d["volume"].iloc[-21:-1].mean()) if len(d) >= 21 else 0.0
        if avg_vol <= 0:
            return False
        if (vol / avg_vol) < filters.min_volume_ratio:
            return False

    if need_ma:
        period = int(filters.ma_period)
        closes = d["close"].astype(float)
        if filters.ma_type == "ema":
            ma = float(closes.ewm(span=period, adjust=False).mean().iloc[-1])
        else:
            ma = float(closes.rolling(period).mean().iloc[-1])
        if ma <= 0 or pd.isna(ma):
            return False
        if filters.ma_condition == MA_COND_ABOVE and close < ma:
            return False
        if filters.ma_condition == MA_COND_BELOW and close > ma:
            return False

    return True


def _evaluate_v2_signal(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    week_idx: int,
    *,
    min_quality_score: int,
    market_favorable: bool,
    tech_filter: str = "none",
    daily_tech: Optional[pd.DataFrame] = None,
    entry_stage: int = DEFAULT_ENTRY_STAGE,
    entry_on: str = DEFAULT_ENTRY_ON,
    entry_filters: Optional[EntryFilters] = None,
    target_rr: float = DEFAULT_TARGET_RR,
    stop_ma_mult: float = DEFAULT_STOP_MA_MULT,
    min_rs_rating: float = 0.0,
) -> Optional[dict]:
    prev_stage, _ = _weekly_stage_at(weekly, week_idx - 1)
    curr_stage, metrics = _weekly_stage_at(weekly, week_idx)
    entry_stage = normalize_entry_stage(entry_stage)
    entry_on = normalize_entry_on(entry_on)

    if curr_stage != entry_stage:
        return None
    if entry_on == ENTRY_ON_TRANSITION and prev_stage == entry_stage:
        return None

    week_end = weekly.index[week_idx]
    w_slice = weekly.iloc[: week_idx + 1]
    d_slice = daily.loc[daily.index <= week_end]
    b_slice = bench_weekly.loc[bench_weekly.index <= week_end]

    filters = entry_filters or EntryFilters()
    if not _passes_entry_filters(daily, week_end, filters):
        return None

    d_stage = 0
    if len(d_slice) >= 190:
        try:
            d_stage, _, _ = detect_daily_stage(d_slice)
        except ValueError:
            pass

    rs = compute_relative_strength(w_slice, b_slice, NIFTY50_SYMBOL)
    breakout = detect_breakout(w_slice, d_slice)
    quality = compute_quality_score(
        weekly_stage=curr_stage,
        daily_stage=d_stage,
        weekly_metrics=metrics,
        breakout=breakout,
        rs=rs,
        weekly=w_slice,
        market_favorable=market_favorable,
    )

    if quality.total < min_quality_score:
        return None
    if min_rs_rating > 0 and float(rs.rating or 0) < float(min_rs_rating):
        return None

    # Technical confirmation filters (EMA / BB / daily stage / extension…)
    snap = None
    if pack_needs_tech_snapshot(tech_filter) and daily_tech is not None:
        snap = snapshot_tech(daily_tech, week_end)
    if not passes_tech_filter(
        snap,
        tech_filter,
        daily_stage=d_stage,
        follow_through=bool(breakout.follow_through),
    ):
        return None

    ma = metrics.get("ma", 0.0)
    price = metrics.get("price", float(weekly.iloc[week_idx]["close"]))
    rr = float(target_rr) if target_rr and float(target_rr) > 0 else DEFAULT_TARGET_RR
    stop_mult = float(stop_ma_mult) if stop_ma_mult and float(stop_ma_mult) > 0 else DEFAULT_STOP_MA_MULT
    # Clamp stop mult sensibly (e.g. 0.85–0.99 of MA)
    stop_mult = min(0.999, max(0.5, stop_mult))
    stop = round(ma * stop_mult, 2) if ma else round(price * 0.93, 2)
    target = round(price + (price - stop) * rr, 2) if stop < price else round(price * (1 + 0.1 * rr / 2.5), 2)

    return {
        "signal_date": week_end,
        "quality_score": quality.total,
        "rs_rating": rs.rating,
        "stop": stop,
        "target": target,
        "signal_close": price,
        "weekly_stage": curr_stage,
        "daily_stage": d_stage,
        "breakout_type": breakout.breakout_type,
        "follow_through": bool(breakout.follow_through),
        "target_rr": rr,
    }


def _next_trading_day(daily: pd.DataFrame, after: pd.Timestamp) -> Optional[pd.Timestamp]:
    future = daily.index[daily.index > after]
    return future[0] if len(future) else None


def _collect_stage2_signals(
    symbol: str,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    nifty_weekly: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    min_quality_score: int,
    market_filter: bool,
    tech_filter: str = "none",
    daily_tech: Optional[pd.DataFrame] = None,
    entry_stage: int = DEFAULT_ENTRY_STAGE,
    entry_on: str = DEFAULT_ENTRY_ON,
    entry_filters: Optional[EntryFilters] = None,
    target_rr: float = DEFAULT_TARGET_RR,
    stop_ma_mult: float = DEFAULT_STOP_MA_MULT,
    min_rs_rating: float = 0.0,
) -> list[dict]:
    """Find stage-entry signals in the backtest window (default: Stage 2 transition)."""
    daily_ind = add_daily_indicators(daily)
    signals: list[dict] = []

    for w_idx in range(1, len(weekly)):
        week_end = weekly.index[w_idx]
        if week_end < start_ts or week_end > end_ts:
            continue

        mkt_ok = True
        if market_filter and not nifty_weekly.empty:
            mkt_ok = _market_favorable_at(nifty_weekly, week_end)
        if not mkt_ok:
            continue

        sig = _evaluate_v2_signal(
            weekly,
            daily_ind,
            bench_weekly,
            w_idx,
            min_quality_score=min_quality_score,
            market_favorable=mkt_ok,
            tech_filter=tech_filter,
            daily_tech=daily_tech,
            entry_stage=entry_stage,
            entry_on=entry_on,
            entry_filters=entry_filters,
            target_rr=target_rr,
            stop_ma_mult=stop_ma_mult,
            min_rs_rating=min_rs_rating,
        )
        if sig is None:
            continue

        entry_day = _next_trading_day(daily, week_end)
        if entry_day is None or entry_day > end_ts:
            continue

        sig["symbol"] = symbol
        sig["entry_day"] = entry_day
        signals.append(sig)

    return signals


def _preload_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 250:
            continue
        frames[sym] = df
    return frames


def _compute_monthly_returns(trades: list[StageV2Trade], capital: float) -> list[dict]:
    if not trades:
        return []
    monthly: dict[str, float] = {}
    for t in trades:
        month = t.exit_date[:7]
        monthly[month] = monthly.get(month, 0) + t.pnl
    running = capital
    out = []
    for month in sorted(monthly.keys()):
        pnl = monthly[month]
        running += pnl
        out.append({
            "month": month,
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / capital * 100, 2),
            "equity": round(running, 2),
        })
    return out


def _open_positions(opens) -> list:
    if isinstance(opens, dict):
        return list(opens.values())
    return list(opens)


def _invested_total(opens) -> float:
    return sum(p.notional for p in _open_positions(opens))


def _equity(cash: float, opens) -> float:
    """Account equity at cost (cash + capital locked in open trades)."""
    return cash + _invested_total(opens)


def _book_stats(
    trades: list[StageV2Trade],
    equity_curve: list[dict],
    capital: float,
    skipped_cash: int,
    peak_parallel: int,
    final_cash: float,
) -> dict:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    n = len(trades)
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = float(point.get("equity") or 0)
        peak = max(peak, e)
        if peak:
            max_dd = max(max_dd, (peak - e) / peak * 100)
    return {
        "trades": n,
        "win_rate": round(len(wins) / n * 100, 2) if n else 0.0,
        "profit_factor": round(gp / gl, 2) if n else 0.0,
        "avg_rr": round(sum(t.rr_achieved for t in trades) / n, 2) if n else 0.0,
        "avg_hold_days": round(sum(t.days_held for t in trades) / n, 1) if n else 0.0,
        "avg_trade": round(sum(t.pnl for t in trades) / n, 2) if n else 0.0,
        "total_return_pct": round((final_cash - capital) / capital * 100, 2) if capital else 0.0,
        "max_drawdown_pct": round(max_dd, 2),
        "skipped_cash": int(skipped_cash),
        "final_capital": round(final_cash, 2),
        "peak_parallel": int(peak_parallel),
    }


def stop_exit_reason(pos: _OpenPos) -> str:
    """Initial protective stop vs a stop that has already been trailed higher."""
    if pos.entry_stop > 0 and pos.stop > pos.entry_stop + 1e-9:
        return "trail_stop"
    return "stop_loss"


def maybe_raise_stop(pos: _OpenPos, new_stop: float, close: float) -> None:
    """Ratchet the stop up only after this bar's exit check (no same-bar look-ahead)."""
    if new_stop > pos.stop and 0 < new_stop < close:
        pos.stop = new_stop


def _close_trade(
    pos: _OpenPos,
    *,
    exit_price: float,
    exit_ts: pd.Timestamp,
    exit_reason: str,
    days_held: int,
) -> StageV2Trade:
    stop_for_risk = pos.entry_stop if pos.entry_stop > 0 else pos.stop
    risk = pos.entry_price - stop_for_risk
    rr = (exit_price - pos.entry_price) / risk if risk > 0 else 0.0
    pnl = (exit_price - pos.entry_price) * pos.qty
    return StageV2Trade(
        symbol=pos.symbol,
        signal_date=str(pos.signal_date.date()),
        entry_date=str(pos.entry_date.date()),
        exit_date=str(exit_ts.date()),
        entry_price=round(pos.entry_price, 2),
        exit_price=round(exit_price, 2),
        stop_loss=round(pos.stop, 2),
        target=round(pos.target, 2),
        quantity=int(pos.qty),
        pnl=round(pnl, 2),
        pnl_pct=round(pnl / pos.notional * 100, 2) if pos.notional else 0.0,
        rr_achieved=round(rr, 2),
        exit_reason=exit_reason,
        quality_score=pos.quality_score,
        rs_rating=pos.rs_rating,
        weekly_stage=pos.weekly_stage,
        days_held=days_held,
        capital_invested=round(pos.capital_invested, 2),
        cash_available=round(pos.cash_available, 2),
        total_invested=round(pos.total_invested, 2),
        parallel_open=pos.parallel_open,
        equity_at_entry=round(pos.equity_at_entry, 2),
        setup=dict(pos.setup) if pos.setup else {},
    )


def fill_performance_metrics(result: StageV2BacktestResult) -> StageV2BacktestResult:
    """Populate CAGR, Sharpe, average trade, and final capital on a result.

    Does not change trade lists or engine behaviour — extra stats only.
    """
    capital = float(result.capital or 0.0)
    final = float(result.final_cash if result.final_cash else capital)
    if result.equity_curve:
        final = float(result.equity_curve[-1].get("equity") or final)
    # Prefer cash after all positions are closed (backtesters force-flat at end).
    if result.final_cash:
        final = float(result.final_cash)
    result.final_capital = round(final, 2)

    trades = result.trades or []
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    result.win_count = len(wins)
    result.loss_count = len(losses)
    if trades:
        result.avg_trade = round(sum(t.pnl for t in trades) / len(trades), 2)
    else:
        result.avg_trade = 0.0

    years = max((result.end_date - result.start_date).days, 1) / 365.25
    if capital > 0 and final > 0 and years > 0:
        result.cagr_pct = round(((final / capital) ** (1.0 / years) - 1.0) * 100.0, 2)
    elif capital > 0 and final <= 0:
        result.cagr_pct = -100.0
    else:
        result.cagr_pct = 0.0

    curve = result.equity_curve or []
    rets: list[float] = []
    gaps: list[float] = []
    prev_eq = None
    prev_d = None
    for point in curve:
        eq = float(point.get("equity") or 0.0)
        raw_d = point.get("date")
        try:
            d = date.fromisoformat(str(raw_d)[:10])
        except (TypeError, ValueError):
            d = None
        if prev_eq and prev_eq > 0:
            rets.append((eq - prev_eq) / prev_eq)
            if d is not None and prev_d is not None:
                gaps.append(max((d - prev_d).days, 1))
        prev_eq = eq
        prev_d = d
    if len(rets) >= 2:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(var) if var > 0 else 0.0
        avg_gap = (sum(gaps) / len(gaps)) if gaps else 1.0
        periods = 365.25 / max(avg_gap, 1.0)
        result.sharpe = round((mu / sigma) * math.sqrt(periods), 2) if sigma > 1e-12 else 0.0
    else:
        result.sharpe = 0.0
    return result


def _safe_date_str(value) -> str:
    if value is None:
        return ""
    return str(value)[:10]


def _safe_float(value, ndigits: int = 2):
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    return round(num, ndigits)


def _empty_signal_row(symbol: str, signal_date: str, entry_date: str) -> dict:
    return {
        "symbol": symbol,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "signal_close": None,
        "stop_loss": None,
        "target": None,
        "quality_score": 0,
        "rs_rating": 0.0,
        "weekly_stage": 0,
        "daily_stage": 0,
        "breakout_type": "",
        "follow_through": False,
        "status": "skipped",
        "status_label": "Skipped",
        "entry_price": None,
        "exit_date": None,
        "exit_price": None,
        "pnl": None,
        "pnl_pct": None,
        "exit_reason": None,
        "rr_achieved": None,
        "days_held": None,
    }


def _apply_trade_to_signal_row(row: dict, trade: StageV2Trade) -> dict:
    row["status"] = "taken"
    row["status_label"] = "Taken (backtest trade)"
    row["entry_price"] = _safe_float(trade.entry_price)
    row["exit_date"] = _safe_date_str(trade.exit_date)
    row["exit_price"] = _safe_float(trade.exit_price)
    row["stop_loss"] = _safe_float(trade.stop_loss) if trade.stop_loss else row.get("stop_loss")
    row["target"] = _safe_float(trade.target) if trade.target else row.get("target")
    row["pnl"] = _safe_float(trade.pnl, 2)
    row["pnl_pct"] = _safe_float(trade.pnl_pct, 2)
    row["exit_reason"] = trade.exit_reason or ""
    row["rr_achieved"] = _safe_float(trade.rr_achieved, 2)
    row["days_held"] = int(trade.days_held or 0)
    row["quality_score"] = int(trade.quality_score or row.get("quality_score") or 0)
    row["rs_rating"] = _safe_float(trade.rs_rating, 1) or row.get("rs_rating") or 0
    return row


def attach_signal_log_from_raw(
    result: StageV2BacktestResult,
    raw_signals: list[dict] | None = None,
) -> StageV2BacktestResult:
    """Build result.signal_log from raw setups + taken trades.

    Raw rows should include symbol, signal_date, entry_date, and optionally
    stop_loss / target / quality_score / rs_rating / entry_price / signal_close.
    Taken trades overlay entry, exit, stop, and P&L.
    """
    by_key: dict[tuple[str, str], dict] = {}
    for sig in raw_signals or []:
        symbol = str(sig.get("symbol") or "")
        entry_date = _safe_date_str(sig.get("entry_date"))
        if not symbol or not entry_date:
            continue
        key = (symbol, entry_date)
        row = _empty_signal_row(
            symbol,
            _safe_date_str(sig.get("signal_date")) or entry_date,
            entry_date,
        )
        row["signal_close"] = _safe_float(sig.get("signal_close"))
        row["stop_loss"] = _safe_float(sig.get("stop_loss"))
        row["target"] = _safe_float(sig.get("target"))
        row["quality_score"] = int(sig.get("quality_score") or 0)
        row["rs_rating"] = _safe_float(sig.get("rs_rating"), 1) or 0.0
        row["weekly_stage"] = int(sig.get("weekly_stage") or 0)
        row["entry_price"] = _safe_float(sig.get("entry_price"))
        row["status"] = str(sig.get("status") or "skipped")
        row["status_label"] = str(sig.get("status_label") or "Skipped")
        by_key[key] = row

    for trade in result.trades or []:
        key = (trade.symbol, _safe_date_str(trade.entry_date))
        row = by_key.get(key) or _empty_signal_row(
            trade.symbol,
            _safe_date_str(trade.signal_date) or key[1],
            key[1],
        )
        by_key[key] = _apply_trade_to_signal_row(row, trade)

    result.signal_log = sorted(
        by_key.values(),
        key=lambda r: (r.get("entry_date") or "", r.get("symbol") or ""),
    )
    return result


def _json_ready_row(row: dict) -> dict:
    clean: dict = {}
    for key, value in row.items():
        if hasattr(value, "item") and not isinstance(value, (bytes, str, int, float, bool)):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                value = str(value)
        if isinstance(value, float) and value != value:
            value = None
        clean[key] = value
    return clean


def ensure_signal_log(result: StageV2BacktestResult) -> StageV2BacktestResult:
    """Guarantee signal_log is populated (falls back to taken trades)."""
    if not result.signal_log:
        attach_signal_log_from_raw(result, [])
    result.signal_log = [_json_ready_row(r) for r in result.signal_log]
    return result


def trades_as_json(result: StageV2BacktestResult) -> list[dict]:
    """JSON-safe closed trades for the win/loss review panel."""
    rows: list[dict] = []
    for t in result.trades or []:
        pnl = float(t.pnl or 0)
        rows.append(_json_ready_row({
            "symbol": t.symbol,
            "signal_date": _safe_date_str(t.signal_date),
            "entry_date": _safe_date_str(t.entry_date),
            "exit_date": _safe_date_str(t.exit_date),
            "entry_price": _safe_float(t.entry_price),
            "exit_price": _safe_float(t.exit_price),
            "stop_loss": _safe_float(t.stop_loss),
            "target": _safe_float(t.target),
            "quantity": int(t.quantity or 0),
            "pnl": _safe_float(t.pnl, 2),
            "pnl_pct": _safe_float(t.pnl_pct, 2),
            "rr_achieved": _safe_float(t.rr_achieved, 2),
            "exit_reason": t.exit_reason or "",
            "quality_score": int(t.quality_score or 0),
            "rs_rating": _safe_float(t.rs_rating, 1),
            "days_held": int(t.days_held or 0),
            "outcome": "win" if pnl > 0 else "loss",
        }))
    return rows


def run_stage_v2_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    min_quality_score: int = 0,
    market_filter: bool = False,
    exit_mode: str = DEFAULT_EXIT_MODE,
    tech_filter: str = DEFAULT_TECH_FILTER,
    entry_stage: int = DEFAULT_ENTRY_STAGE,
    entry_on: str = DEFAULT_ENTRY_ON,
    entry_filters: EntryFilters | None = None,
    target_rr: float = DEFAULT_TARGET_RR,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
    stop_ma_mult: float = DEFAULT_STOP_MA_MULT,
    trail_ma_mult: float = DEFAULT_TRAIL_MA_MULT,
    risk_pct: float | None = None,
    cooldown_days: int | None = None,
    min_rs_rating: float = 0.0,
    shared_capital: bool = True,
    config: Optional[StrategyConfig] = None,
) -> StageV2BacktestResult:
    """
    Backtest Stage Analysis 2.0 on Nifty 200 (or custom universe).

    By default one shared cash account: free cash + open notionals only.
    Parallel positions compete for the same capital pool.
    When shared_capital is False, every valid signal is taken (cash / cooldown
    / already-open do not skip). Both books are always computed for comparison.

    entry_stage: weekly stage to enter (1–4, default 2).
    entry_on: transition (new entry into stage) or in_stage.
    entry_filters: optional price / volume / MA filters on signal day.
    target_rr: reward multiple of risk (default 2.5).
    max_hold_days: time stop (default 65).
    stop_ma_mult: stop = 30w MA × mult (default 0.95).
    trail_ma_mult: trail stop = 30w MA × mult when trail exit mode (default 0.98).
    exit_mode: stage_4_only (default), trail_ma_s4, stage_3_4, no_stage.
    tech_filter: daily_mtf (default), not_extended, ema_stack, bb_mid, …
    risk_pct: % of equity risked per trade (default from StrategyConfig, usually 2).
    cooldown_days: min days after exit before re-entry on same symbol (default 40).
    """
    config = config or StrategyConfig.get_active()
    if risk_pct is None:
        risk_pct = float(getattr(config, "risk_pct", DEFAULT_RISK_PCT) or DEFAULT_RISK_PCT)
    else:
        risk_pct = float(risk_pct)
    risk_pct = min(50.0, max(0.1, risk_pct))
    if cooldown_days is None:
        cooldown_days = DEFAULT_COOLDOWN_DAYS
    else:
        try:
            cooldown_days = int(cooldown_days)
        except (TypeError, ValueError):
            cooldown_days = DEFAULT_COOLDOWN_DAYS
    cooldown_days = min(365, max(0, cooldown_days))
    try:
        min_rs_rating = float(min_rs_rating or 0)
    except (TypeError, ValueError):
        min_rs_rating = 0.0
    min_rs_rating = min(100.0, max(0.0, min_rs_rating))
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]
    exit_mode = normalize_exit_mode(exit_mode)
    tech_filter = normalize_tech_filter(tech_filter)
    entry_stage = normalize_entry_stage(entry_stage)
    entry_on = normalize_entry_on(entry_on)
    filters = entry_filters or EntryFilters()
    try:
        target_rr = float(target_rr)
    except (TypeError, ValueError):
        target_rr = DEFAULT_TARGET_RR
    target_rr = min(10.0, max(0.5, target_rr))
    try:
        max_hold_days = int(max_hold_days)
    except (TypeError, ValueError):
        max_hold_days = DEFAULT_MAX_HOLD_DAYS
    max_hold_days = min(500, max(1, max_hold_days))
    try:
        stop_ma_mult = float(stop_ma_mult)
    except (TypeError, ValueError):
        stop_ma_mult = DEFAULT_STOP_MA_MULT
    stop_ma_mult = min(0.999, max(0.5, stop_ma_mult))
    try:
        trail_ma_mult = float(trail_ma_mult)
    except (TypeError, ValueError):
        trail_ma_mult = DEFAULT_TRAIL_MA_MULT
    trail_ma_mult = min(0.999, max(0.5, trail_ma_mult))
    trail_ma = exit_mode == EXIT_TRAIL_MA_S4
    need_tech = pack_needs_tech_snapshot(tech_filter)
    need_st = pack_needs_supertrend(tech_filter)
    target_exit_label = f"target_{target_rr:g}r"

    on_label = dict(ENTRY_ON_CHOICES).get(entry_on, entry_on)
    strategy_name = (
        f"Stage Analysis 2.0 — Buy Stage {entry_stage} ({on_label}) · "
        f"{target_rr:g}R · hold≤{max_hold_days}d"
    )

    result = StageV2BacktestResult(
        strategy_name=strategy_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=min_quality_score,
        min_rs_rating=min_rs_rating,
        market_filter=market_filter,
        strategy_id="stage_v2",
        max_pos_pct=100.0,
        exit_mode=exit_mode,
        exit_mode_label=EXIT_MODE_LABELS.get(exit_mode, exit_mode),
        tech_filter=tech_filter,
        tech_filter_label=TECH_FILTER_LABELS.get(
            tech_filter, tech_filter
        ),
        entry_stage=entry_stage,
        entry_on=entry_on,
        entry_filters_label=filters.active_summary(),
        target_rr=target_rr,
        max_hold_days=max_hold_days,
        stop_ma_mult=stop_ma_mult,
        trail_ma_mult=trail_ma_mult,
        risk_pct=risk_pct,
        cooldown_days=cooldown_days,
    )

    frames = _preload_frames(symbols)
    result.stocks_scanned = len(frames)
    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = (
        add_weekly_indicators(daily_to_weekly(nifty_daily))
        if not nifty_daily.empty
        else pd.DataFrame()
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    weekly_by_sym: dict[str, pd.DataFrame] = {}
    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    all_signals = 0

    for symbol, daily in frames.items():
        weekly = add_weekly_indicators(daily_to_weekly(daily))
        if len(weekly) < MIN_WEEKLY_BARS + 2:
            continue
        weekly_by_sym[symbol] = weekly
        bench_weekly = nifty_weekly if not nifty_weekly.empty else weekly
        daily_tech = None
        if need_tech:
            daily_tech = enrich_daily_tech(daily, include_supertrend=need_st)
        symbol_signals = _collect_stage2_signals(
            symbol,
            weekly,
            daily,
            bench_weekly,
            nifty_weekly,
            start_ts,
            end_ts,
            min_quality_score=min_quality_score,
            market_filter=market_filter,
            tech_filter=tech_filter,
            daily_tech=daily_tech,
            entry_stage=entry_stage,
            entry_on=entry_on,
            entry_filters=filters,
            target_rr=target_rr,
            stop_ma_mult=stop_ma_mult,
            min_rs_rating=min_rs_rating,
        )
        all_signals += len(symbol_signals)
        for sig in symbol_signals:
            signals_by_day[sig["entry_day"]].append(sig)

    result.stage2_entries = all_signals
    result.total_signals = all_signals

    calendar_set: set[pd.Timestamp] = set()
    for df in frames.values():
        calendar_set.update(df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist())
    calendar = sorted(calendar_set)
    if not calendar:
        result.final_cash = capital
        result.equity_curve = [{"date": str(start_date), "equity": capital}]
        return result

    signal_status_template: dict[tuple[str, str], dict] = {}
    for _day_ts, day_list in signals_by_day.items():
        for sig in day_list:
            key = (sig["symbol"], str(pd.Timestamp(sig["entry_day"]).date()))
            signal_status_template[key] = {
                "symbol": sig["symbol"],
                "signal_date": str(pd.Timestamp(sig["signal_date"]).date()),
                "entry_date": str(pd.Timestamp(sig["entry_day"]).date()),
                "signal_close": round(float(sig.get("signal_close") or 0), 2),
                "stop_loss": round(float(sig.get("stop") or 0), 2),
                "target": round(float(sig.get("target") or 0), 2),
                "quality_score": int(sig.get("quality_score") or 0),
                "rs_rating": round(float(sig.get("rs_rating") or 0), 1),
                "weekly_stage": int(sig.get("weekly_stage") or 2),
                "daily_stage": int(sig.get("daily_stage") or 0),
                "breakout_type": sig.get("breakout_type") or "none",
                "follow_through": bool(sig.get("follow_through")),
                "status": "pending",
                "status_label": "Pending",
                "entry_price": None,
                "exit_date": None,
                "exit_price": None,
                "pnl": None,
                "pnl_pct": None,
                "exit_reason": None,
                "rr_achieved": None,
                "days_held": None,
            }

    def _simulate_book(shared_capital: bool) -> dict:
        cash = float(capital)
        opens: list[_OpenPos] = []
        last_exit: dict[str, pd.Timestamp] = {}
        trades: list[StageV2Trade] = []
        equity_curve: list[dict] = [{"date": str(start_date), "equity": capital}]
        peak_parallel = 0
        skipped_cash = 0
        last_curve_date: Optional[date] = None
        signal_status = copy.deepcopy(signal_status_template)

        def _record_curve(ts: pd.Timestamp, force: bool = False) -> None:
            nonlocal last_curve_date
            d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
            if not force and last_curve_date is not None and (d - last_curve_date).days < 5:
                if d.weekday() != 4:
                    return
            eq = _equity(cash, opens)
            equity_curve.append({"date": str(d), "equity": round(eq, 2)})
            last_curve_date = d

        for ts in calendar:
            # ── Exits first (free capital for same-day entries) ──
            still_open: list[_OpenPos] = []
            for pos in opens:
                sym = pos.symbol
                df = frames.get(sym)
                if df is None or ts not in df.index:
                    still_open.append(pos)
                    continue
                row = df.loc[ts]
                close = float(row["close"])
                low = float(row["low"])
                high = float(row["high"])
                pos.hold_days += 1

                exit_price = None
                exit_reason = ""

                stage_now: Optional[int] = None
                ma_now: Optional[float] = None
                if exit_mode != EXIT_NO_STAGE or trail_ma:
                    weekly = weekly_by_sym.get(sym)
                    if weekly is not None and not weekly.empty:
                        week_mask = weekly.index[
                            (weekly.index > (pos.last_week_check or pos.entry_date))
                            & (weekly.index <= ts)
                        ]
                        if len(week_mask):
                            pos.last_week_check = week_mask[-1]
                            w_idx = weekly.index.get_loc(pos.last_week_check)
                            if isinstance(w_idx, slice):
                                w_idx = w_idx.stop - 1
                            stage_now, metrics = _weekly_stage_at(weekly, int(w_idx))
                            if metrics:
                                raw_ma = metrics.get("ma")
                                if raw_ma is not None and float(raw_ma) > 0:
                                    ma_now = float(raw_ma)

                if low <= pos.stop:
                    exit_price = pos.stop
                    exit_reason = stop_exit_reason(pos)
                elif high >= pos.target:
                    exit_price = pos.target
                    exit_reason = target_exit_label
                elif pos.hold_days >= max_hold_days:
                    exit_price = close
                    exit_reason = "time_exit"
                elif stage_now is not None and exit_mode != EXIT_NO_STAGE:
                    force_stage_exit = False
                    if exit_mode == EXIT_STAGE_3_4 and stage_now in (3, 4):
                        force_stage_exit = True
                    elif exit_mode in (EXIT_STAGE_4_ONLY, EXIT_TRAIL_MA_S4) and stage_now == 4:
                        force_stage_exit = True
                    if force_stage_exit:
                        exit_price = close
                        exit_reason = "stage_exit"

                if exit_price is None:
                    if trail_ma and ma_now is not None:
                        maybe_raise_stop(pos, round(ma_now * trail_ma_mult, 2), close)
                    still_open.append(pos)
                    continue

                trade = _close_trade(
                    pos,
                    exit_price=exit_price,
                    exit_ts=ts,
                    exit_reason=exit_reason,
                    days_held=pos.hold_days,
                )
                cash += pos.notional + trade.pnl
                trades.append(trade)
                last_exit[sym] = ts

            closed_any = len(still_open) < len(opens)
            opens = still_open
            if closed_any:
                _record_curve(ts, force=True)

            # ── Entries ──
            day_sigs = signals_by_day.get(ts, [])
            if day_sigs:
                day_sigs = sorted(
                    day_sigs,
                    key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
                    reverse=True,
                )
                open_syms = {p.symbol for p in opens}
                for sig in day_sigs:
                    sym = sig["symbol"]
                    entry_key = (sym, str(pd.Timestamp(ts).date()))
                    log = signal_status.get(entry_key)

                    def _mark(status: str, label: str, _log=log) -> None:
                        if _log is not None:
                            _log["status"] = status
                            _log["status_label"] = label

                    if shared_capital and sym in open_syms:
                        _mark("skipped_open", "Skipped — already in position")
                        continue
                    if sym not in frames or ts not in frames[sym].index:
                        _mark("skipped_no_bar", "Skipped — no price bar")
                        continue
                    prev_x = last_exit.get(sym)
                    if (
                        shared_capital
                        and cooldown_days > 0
                        and prev_x is not None
                        and (ts - prev_x).days < cooldown_days
                    ):
                        _mark("skipped_cooldown", f"Skipped — cooldown ({cooldown_days}d)")
                        continue

                    row = frames[sym].loc[ts]
                    open_price = float(row["open"])
                    high = float(row["high"])
                    low = float(row["low"])
                    stop = float(sig["stop"])
                    entry_price = open_price
                    risk = entry_price - stop
                    if risk <= 0:
                        _mark("skipped_risk", "Skipped — invalid stop vs open")
                        continue
                    target = round(entry_price + risk * target_rr, 2)

                    equity_now = _equity(cash, opens)
                    if shared_capital and (cash <= 0 or equity_now <= 0):
                        skipped_cash += 1
                        _mark("skipped_cash", "Skipped — no free cash")
                        continue

                    size_equity = equity_now if equity_now > 0 else float(capital)
                    pos_size = calculate_position_size(size_equity, risk_pct, entry_price, stop)
                    qty = int(pos_size.quantity)
                    if shared_capital:
                        max_qty_cash = int(cash // entry_price) if entry_price else 0
                        qty = min(qty, max_qty_cash)
                        if qty <= 0:
                            skipped_cash += 1
                            _mark("skipped_cash", "Skipped — no free cash")
                            continue
                    elif qty <= 0:
                        _mark("skipped_risk", "Skipped — size zero")
                        continue

                    notional = qty * entry_price
                    if shared_capital and notional > cash + 1e-6:
                        skipped_cash += 1
                        _mark("skipped_cash", "Skipped — no free cash")
                        continue

                    cash -= notional
                    invested_after = _invested_total(opens) + notional
                    parallel = len(opens) + 1
                    peak_parallel = max(peak_parallel, parallel)
                    equity_at_entry = cash + invested_after

                    if log is not None:
                        log["status"] = "taken"
                        log["status_label"] = "Taken (backtest trade)"
                        log["entry_price"] = round(entry_price, 2)

                    pos = _OpenPos(
                        symbol=sym,
                        entry_date=ts,
                        signal_date=sig["signal_date"],
                        entry_price=entry_price,
                        stop=stop,
                        target=target,
                        qty=qty,
                        notional=notional,
                        quality_score=int(sig.get("quality_score") or 0),
                        rs_rating=float(sig.get("rs_rating") or 0),
                        weekly_stage=int(sig.get("weekly_stage") or 2),
                        hold_days=0,
                        capital_invested=notional,
                        cash_available=cash,
                        total_invested=invested_after,
                        parallel_open=parallel,
                        equity_at_entry=equity_at_entry,
                        entry_stop=stop,
                    )
                    opens.append(pos)
                    open_syms.add(sym)

                    if low <= stop:
                        trade = _close_trade(
                            pos, exit_price=stop, exit_ts=ts, exit_reason="stop_loss", days_held=0
                        )
                        cash += pos.notional + trade.pnl
                        trades.append(trade)
                        last_exit[sym] = ts
                        opens = [p for p in opens if p is not pos]
                        open_syms = {p.symbol for p in opens}
                    elif high >= target:
                        trade = _close_trade(
                            pos, exit_price=target, exit_ts=ts,
                            exit_reason=target_exit_label, days_held=0,
                        )
                        cash += pos.notional + trade.pnl
                        trades.append(trade)
                        last_exit[sym] = ts
                        opens = [p for p in opens if p is not pos]
                        open_syms = {p.symbol for p in opens}

                    _record_curve(ts, force=True)

            if ts == calendar[-1] or ts.weekday() == 4:
                _record_curve(ts, force=(ts == calendar[-1]))

        if opens:
            last_ts = calendar[-1]
            for pos in list(opens):
                df = frames.get(pos.symbol)
                if df is None:
                    continue
                hist = df.loc[df.index <= last_ts]
                if hist.empty:
                    continue
                close = float(hist.iloc[-1]["close"])
                exit_ts = hist.index[-1]
                trade = _close_trade(
                    pos,
                    exit_price=close,
                    exit_ts=exit_ts,
                    exit_reason="eod_force",
                    days_held=pos.hold_days,
                )
                cash += pos.notional + trade.pnl
                trades.append(trade)
            opens.clear()
            _record_curve(last_ts, force=True)

        trades.sort(key=lambda t: (t.entry_date, t.symbol))
        for t in trades:
            key = (t.symbol, t.entry_date)
            row = signal_status.get(key)
            if row is None:
                continue
            row["status"] = "taken"
            row["status_label"] = "Taken (backtest trade)"
            row["entry_price"] = t.entry_price
            row["exit_date"] = t.exit_date
            row["exit_price"] = t.exit_price
            row["pnl"] = t.pnl
            row["pnl_pct"] = t.pnl_pct
            row["exit_reason"] = t.exit_reason
            row["rr_achieved"] = t.rr_achieved
            row["days_held"] = t.days_held
        for row in signal_status.values():
            if row["status"] == "pending":
                row["status"] = "skipped_other"
                row["status_label"] = "Skipped — not simulated"
        signal_log = sorted(
            signal_status.values(),
            key=lambda r: (r["entry_date"], -int(r["quality_score"]), r["symbol"]),
        )
        final_equity = cash
        stats = _book_stats(
            trades, equity_curve, capital, skipped_cash, peak_parallel, final_equity,
        )
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "signal_log": signal_log,
            "stats": stats,
        }

    shared_book = _simulate_book(True)
    take_all_book = _simulate_book(False)
    primary = shared_book if shared_capital else take_all_book
    trades = primary["trades"]
    equity_curve = primary["equity_curve"]
    signal_log = primary["signal_log"]
    stats = primary["stats"]

    result.shared_capital = bool(shared_capital)
    result.capital_compare = {
        "signals": all_signals,
        "shared": shared_book["stats"],
        "take_all": take_all_book["stats"],
    }
    result.trades = trades
    result.total_trades = stats["trades"]
    result.equity_curve = equity_curve
    result.peak_parallel = stats["peak_parallel"]
    result.signals_skipped_cash = stats["skipped_cash"]
    result.final_cash = stats["final_capital"]
    result.signal_log = signal_log
    result.win_rate = stats["win_rate"]
    result.profit_factor = stats["profit_factor"]
    result.avg_rr = stats["avg_rr"]
    result.avg_hold_days = stats["avg_hold_days"]
    result.expectancy_r = stats["avg_rr"]
    result.avg_trade = stats["avg_trade"]
    result.total_return_pct = stats["total_return_pct"]
    result.max_drawdown_pct = stats["max_drawdown_pct"]
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns = _compute_monthly_returns(trades, capital)
    return result

