"""
Walk-forward backtest for Stage Analysis 2.0 — buy on weekly Stage 2 entry.

Entry: weekly stage transitions into Stage 2 (Weinstein advancing phase).
Exit: stop below 30-week MA, 2.5R target, stage 3/4, or max hold.
"""
from __future__ import annotations

import logging
from collections import Counter
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


@dataclass
class StageV2BacktestResult:
    strategy_name: str
    symbols: list[str]
    start_date: date
    end_date: date
    capital: float
    min_quality_score: int
    market_filter: bool
    trades: list[StageV2Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)
    exit_breakdown: dict = field(default_factory=dict)
    total_trades: int = 0
    total_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_rr: float = 0.0
    total_return_pct: float = 0.0
    avg_hold_days: float = 0.0
    expectancy_r: float = 0.0
    stocks_scanned: int = 0
    stage2_entries: int = 0


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


def _evaluate_v2_signal(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    week_idx: int,
    *,
    min_quality_score: int,
    market_favorable: bool,
) -> Optional[dict]:
    prev_stage, _ = _weekly_stage_at(weekly, week_idx - 1)
    curr_stage, metrics = _weekly_stage_at(weekly, week_idx)
    if curr_stage != 2 or prev_stage == 2:
        return None

    week_end = weekly.index[week_idx]
    w_slice = weekly.iloc[: week_idx + 1]
    d_slice = daily.loc[daily.index <= week_end]
    b_slice = bench_weekly.loc[bench_weekly.index <= week_end]

    d_stage = 0
    if len(d_slice) >= 190:
        try:
            d_stage, _, _ = detect_daily_stage(d_slice)
        except ValueError:
            pass

    rs = compute_relative_strength(w_slice, b_slice, NIFTY50_SYMBOL)
    breakout = detect_breakout(w_slice, d_slice)
    quality = compute_quality_score(
        weekly_stage=2,
        daily_stage=d_stage,
        weekly_metrics=metrics,
        breakout=breakout,
        rs=rs,
        weekly=w_slice,
        market_favorable=market_favorable,
    )

    if quality.total < min_quality_score:
        return None

    ma = metrics.get("ma", 0.0)
    price = metrics.get("price", float(weekly.iloc[week_idx]["close"]))
    stop = round(ma * 0.95, 2) if ma else round(price * 0.93, 2)
    target = round(price + (price - stop) * TARGET_RR, 2) if stop < price else round(price * 1.1, 2)

    return {
        "signal_date": week_end,
        "quality_score": quality.total,
        "rs_rating": rs.rating,
        "stop": stop,
        "target": target,
        "signal_close": price,
        "weekly_stage": curr_stage,
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
) -> list[dict]:
    """Find all Stage 2 transition signals in the backtest window."""
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


def run_stage_v2_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    min_quality_score: int = 0,
    market_filter: bool = False,
    config: Optional[StrategyConfig] = None,
) -> StageV2BacktestResult:
    """
    Backtest Stage Analysis 2.0 on Nifty 200 (or custom universe).

    Buys when weekly stage transitions into Stage 2; exits on stop, target,
    stage deterioration, or time limit.
    """
    config = config or StrategyConfig.get_active()
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]

    result = StageV2BacktestResult(
        strategy_name=STRATEGY_NAME,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=min_quality_score,
        market_filter=market_filter,
    )

    frames = _preload_frames(symbols)
    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = add_weekly_indicators(daily_to_weekly(nifty_daily)) if not nifty_daily.empty else pd.DataFrame()

    equity = capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    trades: list[StageV2Trade] = []
    signals = 0

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for symbol, daily in frames.items():
        result.stocks_scanned += 1
        weekly = add_weekly_indicators(daily_to_weekly(daily))
        if len(weekly) < MIN_WEEKLY_BARS + 2:
            continue

        bench_weekly = nifty_weekly if not nifty_weekly.empty else weekly
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
        )
        result.stage2_entries += len(symbol_signals)
        signals += len(symbol_signals)
        signal_by_entry = {s["entry_day"]: s for s in symbol_signals}

        in_position = False
        entry_price = stop = target = qty = 0.0
        entry_dt = signal_dt = None
        entry_quality = 0
        entry_rs = 0.0
        entry_stage = 2
        hold_days = 0
        last_exit: Optional[pd.Timestamp] = None
        last_week_check: Optional[pd.Timestamp] = None

        eval_dates = daily.index[(daily.index >= start_ts) & (daily.index <= end_ts)]

        for ts in eval_dates:
            row = daily.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            open_price = float(row["open"])

            if in_position:
                hold_days += 1
                exit_price = None
                exit_reason = ""

                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif high >= target:
                    exit_price = target
                    exit_reason = "target_2.5r"
                elif hold_days >= MAX_HOLD_DAYS:
                    exit_price = close
                    exit_reason = "time_exit"

                week_mask = weekly.index[(weekly.index > (last_week_check or entry_dt)) & (weekly.index <= ts)]
                if exit_price is None and len(week_mask):
                    last_week_check = week_mask[-1]
                    w_idx = weekly.index.get_loc(last_week_check)
                    stage_now, _ = _weekly_stage_at(weekly, w_idx)
                    if stage_now in (3, 4):
                        exit_price = close
                        exit_reason = "stage_exit"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * qty
                    risk = entry_price - stop
                    rr = (exit_price - entry_price) / risk if risk > 0 else 0
                    trades.append(StageV2Trade(
                        symbol=symbol,
                        signal_date=str(signal_dt.date()),
                        entry_date=str(entry_dt.date()),
                        exit_date=str(ts.date()),
                        entry_price=round(entry_price, 2),
                        exit_price=round(exit_price, 2),
                        stop_loss=round(stop, 2),
                        target=round(target, 2),
                        quantity=int(qty),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty else 0,
                        rr_achieved=round(rr, 2),
                        exit_reason=exit_reason,
                        quality_score=entry_quality,
                        rs_rating=entry_rs,
                        weekly_stage=entry_stage,
                        days_held=hold_days,
                    ))
                    equity += pnl
                    equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
                    in_position = False
                    last_exit = ts
                    hold_days = 0
                    last_week_check = None
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue

            sig = signal_by_entry.get(ts)
            if sig is None:
                continue

            entry_price = open_price
            stop = sig["stop"]
            target = sig["target"]
            risk = entry_price - stop
            if risk <= 0:
                continue
            pos = calculate_position_size(equity, config.risk_pct, entry_price, stop)
            if pos.quantity <= 0:
                continue
            qty = pos.quantity
            entry_dt = ts
            signal_dt = sig["signal_date"]
            entry_quality = sig["quality_score"]
            entry_rs = sig["rs_rating"]
            entry_stage = sig["weekly_stage"]
            in_position = True
            hold_days = 0
            last_week_check = None

            if low <= stop:
                pnl = (stop - entry_price) * qty
                trades.append(StageV2Trade(
                    symbol=symbol,
                    signal_date=str(signal_dt.date()),
                    entry_date=str(entry_dt.date()),
                    exit_date=str(ts.date()),
                    entry_price=round(entry_price, 2),
                    exit_price=round(stop, 2),
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    quantity=int(qty),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty else 0,
                    rr_achieved=-1.0,
                    exit_reason="stop_loss",
                    quality_score=entry_quality,
                    rs_rating=entry_rs,
                    weekly_stage=entry_stage,
                    days_held=0,
                ))
                equity += pnl
                equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
                in_position = False
                last_exit = ts
            elif high >= target:
                pnl = (target - entry_price) * qty
                risk = entry_price - stop
                rr = (target - entry_price) / risk if risk > 0 else 0
                trades.append(StageV2Trade(
                    symbol=symbol,
                    signal_date=str(signal_dt.date()),
                    entry_date=str(entry_dt.date()),
                    exit_date=str(ts.date()),
                    entry_price=round(entry_price, 2),
                    exit_price=round(target, 2),
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    quantity=int(qty),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty else 0,
                    rr_achieved=round(rr, 2),
                    exit_reason="target_2.5r",
                    quality_score=entry_quality,
                    rs_rating=entry_rs,
                    weekly_stage=entry_stage,
                    days_held=0,
                ))
                equity += pnl
                equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
                in_position = False
                last_exit = ts

    trades.sort(key=lambda t: t.entry_date)
    result.trades = trades
    result.total_trades = len(trades)
    result.total_signals = signals
    result.equity_curve = equity_curve

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        result.win_rate = round(len(wins) / len(trades) * 100, 2)
        result.profit_factor = round(gross_profit / gross_loss, 2)
        result.avg_rr = round(sum(t.rr_achieved for t in trades) / len(trades), 2)
        result.total_return_pct = round((equity - capital) / capital * 100, 2)
        result.avg_hold_days = round(sum(t.days_held for t in trades) / len(trades), 1)
        result.expectancy_r = result.avg_rr

    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = round(max_dd, 2)
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns = _compute_monthly_returns(trades, capital)

    return result