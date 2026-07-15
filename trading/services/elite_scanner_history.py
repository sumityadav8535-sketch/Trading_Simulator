"""
Historical EMA20 Elite signals with simulated trade outcomes for the scanner chart.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS, build_outcome_detail
from trading.services.swing_strategy import (
    PAUSE_AFTER_CONSECUTIVE_LOSSES,
    evaluate_elite_scan_signal,
)

logger = logging.getLogger(__name__)


@dataclass
class EliteHistoryEntry:
    symbol: str
    signal_date: str
    entry_date: str
    exit_date: str
    signal_close: float
    entry_price: float
    stop_loss: float
    target_2r: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    outcome: str
    exit_reason: str
    rr_achieved: float
    days_held: int
    reasons: list[str]
    outcome_detail: str = ""


@dataclass
class EliteHistoryResult:
    start_date: date
    end_date: date
    capital: float
    entries: list[EliteHistoryEntry] = field(default_factory=list)
    daily: list[dict] = field(default_factory=list)
    by_date: dict[str, list[dict]] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


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


def run_elite_scanner_history(
    symbols: list[str],
    start_date: date,
    end_date: date,
    capital: float = 100_000.0,
    config: Optional[StrategyConfig] = None,
) -> EliteHistoryResult:
    """
    Walk-forward elite-only history: signal on close, enter next-day open,
    exit at 2R / stop / max hold. Uses Nifty 50+200 regime and pauses new
    entries after consecutive portfolio losses.
    """
    config = config or StrategyConfig.get_active()
    result = EliteHistoryResult(
        start_date=start_date,
        end_date=end_date,
        capital=capital,
    )

    frames = _preload_frames(symbols)
    equity = capital
    entries: list[EliteHistoryEntry] = []

    for symbol, full_df in frames.items():
        mask = (full_df.index >= pd.Timestamp(start_date)) & (full_df.index <= pd.Timestamp(end_date))
        eval_dates = full_df.index[mask]

        in_position = False
        pending = None
        entry_price = stop = target = qty = 0.0
        entry_dt = signal_dt = None
        entry_reasons: list[str] = []
        signal_close = 0.0
        hold_days = 0
        max_high = 0.0
        min_low = 0.0
        last_exit = None

        def _record(exit_px: float, reason: str, held: int, day_high: float, day_low: float) -> None:
            nonlocal equity, in_position, last_exit, hold_days
            mh = max(max_high, day_high)
            ml = min(min_low, day_low)
            pnl = (exit_px - entry_price) * qty
            risk = entry_price - stop
            rr = (exit_px - entry_price) / risk if risk > 0 else 0
            outcome = "win" if pnl > 0 else "loss"
            detail = build_outcome_detail(
                outcome=outcome,
                exit_reason=reason,
                signal_close=signal_close,
                actual_entry=entry_price,
                stop=stop,
                target=target,
                exit_price=exit_px,
                exit_date=str(ts.date()),
                days_held=held,
                max_high=mh,
                min_low=ml,
                entry_path="Elite 20 EMA",
                reasons=entry_reasons,
            )
            entries.append(
                EliteHistoryEntry(
                    symbol=symbol,
                    signal_date=str(signal_dt.date()),
                    entry_date=str(entry_dt.date()),
                    exit_date=str(ts.date()),
                    signal_close=round(signal_close, 2),
                    entry_price=round(entry_price, 2),
                    stop_loss=round(stop, 2),
                    target_2r=round(target, 2),
                    exit_price=round(exit_px, 2),
                    quantity=int(qty),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty else 0,
                    outcome=outcome,
                    exit_reason=reason,
                    rr_achieved=round(rr, 2),
                    days_held=held,
                    reasons=entry_reasons,
                    outcome_detail=detail,
                )
            )
            nonlocal equity
            equity += pnl
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
                    _record(exit_price, exit_reason, hold_days, high, low)
                continue

            if pending is not None:
                sig = pending
                pending = None
                entry_price = open_price
                stop = sig["stop"]
                risk = entry_price - stop
                if risk <= 0:
                    continue
                target = entry_price + risk * 2
                qty = sig["qty"]
                entry_dt = ts
                entry_reasons = sig["reasons"]
                signal_close = sig["signal_close"]
                in_position = True
                hold_days = 0
                max_high = high
                min_low = low

                if low <= stop:
                    _record(stop, "stop_loss", 0, high, low)
                elif close >= target:
                    _record(target, "target_2r", 0, high, low)
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue

            ev = evaluate_elite_scan_signal(
                symbol,
                eval_date=ts.date(),
                config=config,
                capital=equity,
                indicator_df=hist,
            )
            if not ev.is_valid or not ev.entry_price or not ev.stop_loss:
                continue

            pos = calculate_position_size(
                equity, config.risk_pct, ev.entry_price, ev.stop_loss,
            )
            if pos.quantity <= 0:
                continue

            signal_dt = ts
            if i + 1 < len(eval_dates):
                pending = {
                    "stop": ev.stop_loss,
                    "qty": pos.quantity,
                    "reasons": ev.reasons,
                    "signal_close": ev.entry_price,
                }

    result.entries = _apply_portfolio_pause(entries, PAUSE_AFTER_CONSECUTIVE_LOSSES)
    _aggregate_daily(result)
    return result


def _apply_portfolio_pause(
    entries: list[EliteHistoryEntry],
    pause_after: int,
) -> list[EliteHistoryEntry]:
    """Keep only trades that would fire with a global loss-streak pause (by entry date)."""
    ordered = sorted(entries, key=lambda e: (e.entry_date, e.symbol))
    taken: list[EliteHistoryEntry] = []
    streak = 0
    for entry in ordered:
        if streak >= pause_after:
            continue
        taken.append(entry)
        streak = 0 if entry.outcome == "win" else streak + 1
    return taken


def _aggregate_daily(result: EliteHistoryResult) -> None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for e in result.entries:
        d = e.signal_date
        by_date[d].append({
            "symbol": e.symbol,
            "signal_date": e.signal_date,
            "entry_date": e.entry_date,
            "exit_date": e.exit_date,
            "signal_close": e.signal_close,
            "entry_price": e.entry_price,
            "stop_loss": e.stop_loss,
            "target_2r": e.target_2r,
            "exit_price": e.exit_price,
            "quantity": e.quantity,
            "pnl": e.pnl,
            "pnl_pct": e.pnl_pct,
            "outcome": e.outcome,
            "exit_reason": e.exit_reason,
            "rr_achieved": e.rr_achieved,
            "days_held": e.days_held,
            "reasons": e.reasons,
            "outcome_detail": e.outcome_detail,
        })

    daily = []
    all_dates = sorted(by_date.keys())
    for d in all_dates:
        rows = by_date[d]
        wins = sum(1 for r in rows if r["outcome"] == "win")
        losses = len(rows) - wins
        pnl = sum(r["pnl"] for r in rows)
        daily.append({
            "date": d,
            "count": len(rows),
            "pnl": round(pnl, 2),
            "wins": wins,
            "losses": losses,
        })

    wins_total = sum(1 for e in result.entries if e.outcome == "win")
    total_pnl = sum(e.pnl for e in result.entries)
    result.daily = daily
    result.by_date = dict(by_date)
    result.summary = {
        "signals": len(result.entries),
        "trades": len(result.entries),
        "wins": wins_total,
        "losses": len(result.entries) - wins_total,
        "win_rate": round(wins_total / len(result.entries) * 100, 2) if result.entries else 0,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / result.capital * 100, 2) if result.capital else 0,
    }