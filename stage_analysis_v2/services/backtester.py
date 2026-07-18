"""
Walk-forward backtest for Stage Analysis 2.0 — buy on weekly Stage 2 entry.

Entry: weekly stage transitions into Stage 2 (Weinstein advancing phase).
Exit: stop below 30-week MA, 2.5R target, stage 3/4, or max hold.

Capital model (single account, cash only — no extra leverage):
  - Shared starting capital across all stocks
  - Calendar-order simulation so parallel positions share the same cash pool
  - Position size = risk % of equity, capped so notional ≤ free cash
  - Each trade records: invested ₹, free cash left, parallel open count
"""
from __future__ import annotations

import logging
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
    # Capital account snapshot (₹) — single shared capital pool
    capital_invested: float = 0.0  # notional put into this trade
    cash_available: float = 0.0  # free cash after this entry (for next trade)
    total_invested: float = 0.0  # sum of all open notionals after this entry
    parallel_open: int = 0  # open positions after this entry (incl. this one)
    equity_at_entry: float = 0.0  # cash + invested (cost basis) at entry


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
    peak_parallel: int = 0
    signals_skipped_cash: int = 0
    final_cash: float = 0.0


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


def _invested_total(opens: dict[str, _OpenPos]) -> float:
    return sum(p.notional for p in opens.values())


def _equity(cash: float, opens: dict[str, _OpenPos]) -> float:
    """Account equity at cost (cash + capital locked in open trades)."""
    return cash + _invested_total(opens)


def _close_trade(
    pos: _OpenPos,
    *,
    exit_price: float,
    exit_ts: pd.Timestamp,
    exit_reason: str,
    days_held: int,
) -> StageV2Trade:
    risk = pos.entry_price - pos.stop
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
    )


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

    Single shared capital account: free cash + open notionals only.
    Parallel positions compete for the same capital pool.
    """
    config = config or StrategyConfig.get_active()
    risk_pct = float(getattr(config, "risk_pct", 2.0) or 2.0)
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

    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve: list[dict] = [{"date": str(start_date), "equity": capital}]
    peak_parallel = 0
    skipped_cash = 0
    last_curve_date: Optional[date] = None

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
        closed_today: list[str] = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            pos.hold_days += 1

            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price = pos.stop
                exit_reason = "stop_loss"
            elif high >= pos.target:
                exit_price = pos.target
                exit_reason = "target_2.5r"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                exit_price = close
                exit_reason = "time_exit"
            else:
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
                        stage_now, _ = _weekly_stage_at(weekly, int(w_idx))
                        if stage_now in (3, 4):
                            exit_price = close
                            exit_reason = "stage_exit"

            if exit_price is None:
                continue

            trade = _close_trade(
                pos,
                exit_price=exit_price,
                exit_ts=ts,
                exit_reason=exit_reason,
                days_held=pos.hold_days,
            )
            # Return invested capital + P&L to free cash
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)

        for sym in closed_today:
            opens.pop(sym, None)
        if closed_today:
            _record_curve(ts, force=True)

        # ── Entries (only with free cash from capital pool) ──
        day_sigs = signals_by_day.get(ts, [])
        if day_sigs:
            day_sigs = sorted(
                day_sigs,
                key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
                reverse=True,
            )
            for sig in day_sigs:
                sym = sig["symbol"]
                if sym in opens:
                    continue
                if sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if prev_x is not None and (ts - prev_x).days < COOLDOWN_DAYS:
                    continue

                row = frames[sym].loc[ts]
                open_price = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])
                stop = float(sig["stop"])
                target = float(sig["target"])
                entry_price = open_price
                risk = entry_price - stop
                if risk <= 0:
                    continue

                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    skipped_cash += 1
                    continue

                pos_size = calculate_position_size(equity_now, risk_pct, entry_price, stop)
                qty = int(pos_size.quantity)
                # Cap by free cash only (no leverage beyond account capital)
                max_qty_cash = int(cash // entry_price) if entry_price > 0 else 0
                qty = min(qty, max_qty_cash)
                if qty <= 0:
                    skipped_cash += 1
                    continue

                notional = qty * entry_price
                if notional > cash + 1e-6:
                    skipped_cash += 1
                    continue

                cash -= notional
                invested_after = _invested_total(opens) + notional
                parallel = len(opens) + 1
                peak_parallel = max(peak_parallel, parallel)
                equity_at_entry = cash + invested_after

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
                )
                opens[sym] = pos

                # Same-bar exit
                if low <= stop:
                    trade = _close_trade(
                        pos, exit_price=stop, exit_ts=ts, exit_reason="stop_loss", days_held=0
                    )
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif high >= target:
                    trade = _close_trade(
                        pos, exit_price=target, exit_ts=ts, exit_reason="target_2.5r", days_held=0
                    )
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)

                _record_curve(ts, force=True)

        if ts == calendar[-1] or ts.weekday() == 4:
            _record_curve(ts, force=(ts == calendar[-1]))

    # Force-close leftovers at last close (return capital)
    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
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
    final_equity = cash

    result.trades = trades
    result.total_trades = len(trades)
    result.equity_curve = equity_curve
    result.peak_parallel = peak_parallel
    result.signals_skipped_cash = skipped_cash
    result.final_cash = round(final_equity, 2)

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        result.win_rate = round(len(wins) / len(trades) * 100, 2)
        result.profit_factor = round(gross_profit / gross_loss, 2)
        result.avg_rr = round(sum(t.rr_achieved for t in trades) / len(trades), 2)
        result.avg_hold_days = round(sum(t.days_held for t in trades) / len(trades), 1)
        result.expectancy_r = result.avg_rr

    result.total_return_pct = round((final_equity - capital) / capital * 100, 2)

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
