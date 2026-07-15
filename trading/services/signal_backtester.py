"""
Backtester + signal history for Momentum Pivot Pro strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe
from trading.services.position_sizing import calculate_position_size

from trading.services.swing_strategy import (
    STRATEGY_NAME,
    evaluate_swing_signal,
)

logger = logging.getLogger(__name__)

COOLDOWN_DAYS = 8
MAX_HOLD_DAYS = 45


@dataclass
class HistoricalSignal:
    symbol: str
    signal_date: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_2r: float
    risk_reward: float
    confluence_score: int
    reasons: list[str]
    entry_path: str = ""
    outcome: str = "pending"
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rr_achieved: float = 0.0
    exit_reason: str = ""
    days_held: int = 0
    actual_entry_price: float = 0.0
    outcome_detail: str = ""


@dataclass
class SignalBacktestTrade:
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
    confluence_score: int
    reasons: list[str]
    outcome_detail: str = ""
    signal_close: float = 0.0
    days_held: int = 0


@dataclass
class SignalBacktestResult:
    strategy_name: str
    symbols: list[str]
    start_date: date
    end_date: date
    capital: float
    trades: list[SignalBacktestTrade] = field(default_factory=list)
    signal_history: list[HistoricalSignal] = field(default_factory=list)
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


def build_outcome_detail(
    *,
    outcome: str,
    exit_reason: str,
    signal_close: float,
    actual_entry: float,
    stop: float,
    target: float,
    exit_price: float,
    exit_date: str,
    days_held: int,
    max_high: float,
    min_low: float,
    entry_path: str,
    reasons: list[str],
) -> str:
    """Human-readable explanation of why a trade won or lost."""
    risk = actual_entry - stop
    target_1r = actual_entry + risk if risk > 0 else actual_entry
    gap_pct = (actual_entry - signal_close) / signal_close * 100 if signal_close else 0
    max_fav = max_high - actual_entry
    max_fav_r = max_fav / risk if risk > 0 else 0
    stop_dist_pct = (actual_entry - stop) / actual_entry * 100 if actual_entry else 0

    setup = entry_path or "EMA20 Elite"
    ctx = "; ".join(reasons[1:3]) if len(reasons) > 1 else ""

    if outcome == "win" and exit_reason == "target_2r":
        return (
            f"Hit 2R target at ₹{exit_price:.2f} after {days_held}d. "
            f"Entered ₹{actual_entry:.2f} (signal close ₹{signal_close:.2f}, gap {gap_pct:+.1f}%). "
            f"{setup}: {ctx}"
        )

    if exit_reason == "time_exit":
        return (
            f"Max hold ({MAX_HOLD_DAYS}d) reached on {exit_date}; closed at ₹{exit_price:.2f} "
            f"({'profit' if outcome == 'win' else 'loss'}). "
            f"Best move was +{max_fav_r:.2f}R (high ₹{max_high:.2f}) but 2R target ₹{target:.2f} not reached."
        )

    if exit_reason == "stop_loss":
        parts = []
        if days_held == 0:
            parts.append("Stopped out same day as entry — price rejected the pullback immediately")
        else:
            parts.append(f"Stop loss hit after {days_held}d on {exit_date}")

        parts.append(
            f"entered ₹{actual_entry:.2f} (signal ₹{signal_close:.2f}, next-day gap {gap_pct:+.1f}%)"
        )
        parts.append(f"stop ₹{stop:.2f} ({stop_dist_pct:.1f}% below entry)")

        if max_fav_r > 0.1:
            parts.append(
                f"price reached +{max_fav_r:.2f}R (high ₹{max_high:.2f}) before reversing"
            )
        else:
            parts.append("price never moved meaningfully in favour (failed bounce)")

        if max_high < target_1r:
            parts.append(f"never reached 1R (₹{target_1r:.2f})")
        elif max_high < target:
            parts.append(f"reached 1R but not 2R target (₹{target:.2f})")

        parts.append(f"low breached stop at ₹{min_low:.2f}")
        if ctx:
            parts.append(f"setup was {setup}: {ctx}")

        weak_adx = any("ADX≥20 (20." in r or "ADX≥20 (19." in r for r in reasons)
        if weak_adx:
            parts.append("note: ADX was borderline (≈20) — weak trend strength")

        return ". ".join(parts) + "."

    if outcome == "pending":
        return "Awaiting simulated exit."

    return f"Exit: {exit_reason} at ₹{exit_price:.2f} after {days_held}d."


def _preload_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            frames[sym] = df
    return frames


def run_signal_backtest(
    symbols: list[str],
    start_date: date,
    end_date: date,
    capital: float = 500_000.0,
    config: Optional[StrategyConfig] = None,
) -> SignalBacktestResult:
    """
    Walk-forward backtest with next-day open entry.
    Generates full signal history with simulated outcomes for manual verification.
    """
    config = config or StrategyConfig.get_active()
    result = SignalBacktestResult(
        strategy_name=STRATEGY_NAME,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
    )

    frames = _preload_frames(symbols)
    equity = capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    trades: list[SignalBacktestTrade] = []
    signal_history: list[HistoricalSignal] = []

    for symbol, full_df in frames.items():
        mask = (full_df.index >= pd.Timestamp(start_date)) & (full_df.index <= pd.Timestamp(end_date))
        eval_dates = full_df.index[mask]

        in_position = False
        pending = None
        entry_price = stop = target = qty = 0.0
        entry_dt = signal_dt = None
        entry_reasons: list[str] = []
        entry_score = 0
        entry_path = ""
        signal_close = 0.0
        hold_days = 0
        max_high = 0.0
        min_low = 0.0
        last_exit = None

        def _record_exit(exit_px: float, exit_reason: str, held: int, day_high: float, day_low: float) -> None:
            nonlocal equity, in_position, last_exit, hold_days
            mh = max(max_high, day_high)
            ml = min(min_low, day_low)
            pnl = (exit_px - entry_price) * qty
            risk = entry_price - stop
            rr = (exit_px - entry_price) / risk if risk > 0 else 0
            outcome = "win" if pnl > 0 else "loss"
            detail = build_outcome_detail(
                outcome=outcome,
                exit_reason=exit_reason,
                signal_close=signal_close,
                actual_entry=entry_price,
                stop=stop,
                target=target,
                exit_price=exit_px,
                exit_date=str(ts.date()),
                days_held=held,
                max_high=mh,
                min_low=ml,
                entry_path=entry_path,
                reasons=entry_reasons,
            )
            trade = SignalBacktestTrade(
                symbol=symbol,
                signal_date=str(signal_dt.date()),
                entry_date=str(entry_dt.date()),
                exit_date=str(ts.date()),
                entry_price=round(entry_price, 2),
                exit_price=round(exit_px, 2),
                stop_loss=round(stop, 2),
                target=round(target, 2),
                quantity=int(qty),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty else 0,
                rr_achieved=round(rr, 2),
                exit_reason=exit_reason,
                confluence_score=entry_score,
                reasons=entry_reasons,
                outcome_detail=detail,
                signal_close=round(signal_close, 2),
                days_held=held,
            )
            trades.append(trade)

            for sig in signal_history:
                if sig.symbol == symbol and sig.signal_date == str(signal_dt.date()) and sig.outcome == "pending":
                    sig.outcome = outcome
                    sig.exit_date = str(ts.date())
                    sig.exit_price = round(exit_px, 2)
                    sig.pnl = round(pnl, 2)
                    sig.pnl_pct = trade.pnl_pct
                    sig.rr_achieved = trade.rr_achieved
                    sig.exit_reason = exit_reason
                    sig.days_held = held
                    sig.actual_entry_price = round(entry_price, 2)
                    sig.outcome_detail = detail
                    break

            equity += pnl
            equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
            in_position = False
            last_exit = ts
            hold_days = 0

        for i, ts in enumerate(eval_dates):
            hist = full_df.loc[:ts]
            row = hist.iloc[-1]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            open_price = float(row["open"])

            if in_position:
                hold_days += 1
                max_high = max(max_high, high)
                min_low = min(min_low, low)
                exit_price = None
                exit_reason = ""
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif close >= target:
                    exit_price = target
                    exit_reason = "target_2r"
                elif hold_days >= MAX_HOLD_DAYS:
                    exit_price = close
                    exit_reason = "time_exit"

                if exit_price is not None:
                    _record_exit(exit_price, exit_reason, hold_days, high, low)
                continue

            if pending is not None:
                sig_data = pending
                pending = None
                entry_price = open_price
                stop = sig_data["stop"]
                risk = entry_price - stop
                if risk <= 0:
                    continue
                target = entry_price + risk * 2
                qty = sig_data["qty"]
                entry_dt = ts
                entry_reasons = sig_data["reasons"]
                entry_score = sig_data["score"]
                entry_path = sig_data.get("entry_path", "")
                signal_close = sig_data.get("signal_close", entry_price)
                in_position = True
                hold_days = 0
                max_high = high
                min_low = low

                if low <= stop:
                    _record_exit(stop, "stop_loss", 0, high, low)
                elif close >= target:
                    _record_exit(target, "target_2r", 0, high, low)
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue

            row = hist.iloc[-1]
            close = float(row["close"])
            ema200 = row.get("ema_200")
            if pd.isna(ema200) or close <= float(ema200):
                continue

            eval_result = evaluate_swing_signal(
                symbol,
                eval_date=ts.date(),
                config=config,
                capital=equity,
                indicator_df=hist,
                require_market_filter=False,
            )

            if eval_result.is_valid and eval_result.entry_price and eval_result.stop_loss:
                pos = calculate_position_size(
                    equity, config.risk_pct,
                    eval_result.entry_price, eval_result.stop_loss,
                )
                if pos.quantity <= 0:
                    continue

                signal_dt = ts
                signal_history.append(HistoricalSignal(
                    symbol=symbol,
                    signal_date=str(ts.date()),
                    entry_date="",
                    entry_price=round(eval_result.entry_price, 2),
                    stop_loss=round(eval_result.stop_loss, 2),
                    target_2r=round(eval_result.target_2r or 0, 2),
                    risk_reward=2.0,
                    confluence_score=eval_result.confluence_score,
                    reasons=eval_result.reasons,
                    entry_path=getattr(eval_result, "entry_path", ""),
                    outcome="pending",
                ))

                if i + 1 < len(eval_dates):
                    pending = {
                        "stop": eval_result.stop_loss,
                        "qty": pos.quantity,
                        "reasons": eval_result.reasons,
                        "score": eval_result.confluence_score,
                        "entry_path": eval_result.entry_path,
                        "signal_close": eval_result.entry_price,
                    }
                    signal_history[-1].entry_date = str(eval_dates[i + 1].date())

    result.trades = trades
    result.signal_history = signal_history
    result.total_trades = len(trades)
    result.total_signals = len(signal_history)
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
        hold_days = []
        for sig in signal_history:
            if sig.days_held:
                hold_days.append(sig.days_held)
        result.avg_hold_days = round(sum(hold_days) / len(hold_days), 1) if hold_days else 0
        result.expectancy_r = round(
            sum(t.rr_achieved for t in trades) / len(trades), 2
        )

    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = round(max_dd, 2)

    from collections import Counter
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns = _compute_monthly_returns(trades, capital)

    return result





def _compute_monthly_returns(trades: list[SignalBacktestTrade], capital: float) -> list[dict]:
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