"""
Single/multi-stock backtester for Confluence Trend Pullback strategy.
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
from trading.services.strategy import evaluate_stock

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
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


@dataclass
class BacktestResult:
    symbols: list[str]
    start_date: date
    end_date: date
    capital: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_rr: float = 0.0
    total_return_pct: float = 0.0


def run_backtest(
    symbols: list[str],
    start_date: date,
    end_date: date,
    capital: float = 500_000.0,
    config: Optional[StrategyConfig] = None,
) -> BacktestResult:
    """
    Walk-forward backtest: evaluate signal each day, enter on valid setup,
    exit at 2R target, stop loss, or 20 EMA trail after 1R.
    """
    config = config or StrategyConfig.get_active()
    result = BacktestResult(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
    )

    equity = capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    all_trades: list[BacktestTrade] = []

    for symbol in symbols:
        df = load_price_dataframe(symbol, start=start_date, end=end_date)
        if df.empty:
            continue
        full_df = load_price_dataframe(symbol)
        full_df = compute_indicators(full_df)
        if not has_sufficient_history(full_df):
            continue

        mask = (full_df.index >= pd.Timestamp(start_date)) & (full_df.index <= pd.Timestamp(end_date))
        eval_dates = full_df.index[mask]

        in_position = False
        entry_price = stop = target = qty = 0.0
        entry_dt = None
        trail_ema = None

        for ts in eval_dates:
            hist = full_df.loc[:ts]
            if len(hist) < 220:
                continue
            row = hist.iloc[-1]
            close = float(row["close"])
            low = float(row["low"])
            ema20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None

            if in_position:
                exit_price = None
                exit_reason = ""
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif close >= target:
                    exit_price = target
                    exit_reason = "target_2r"
                elif trail_ema and ema20 and close < ema20:
                    exit_price = close
                    exit_reason = "trail_20ema"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * qty
                    risk = entry_price - stop
                    rr = (exit_price - entry_price) / risk if risk > 0 else 0
                    all_trades.append(
                        BacktestTrade(
                            symbol=symbol,
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
                        )
                    )
                    equity += pnl
                    equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
                    in_position = False
                    trail_ema = False
                elif close >= entry_price + (entry_price - stop):
                    trail_ema = True
                continue

            eval_result = evaluate_stock(
                symbol,
                eval_date=ts.date(),
                config=config,
                capital=equity,
                indicator_df=hist,
            )
            if eval_result.is_valid and eval_result.entry_price and eval_result.stop_loss:
                pos = calculate_position_size(
                    equity, config.risk_pct,
                    eval_result.entry_price, eval_result.stop_loss,
                )
                if pos.quantity <= 0:
                    continue
                in_position = True
                entry_price = eval_result.entry_price
                stop = eval_result.stop_loss
                target = eval_result.target_2r or (entry_price + 2 * (entry_price - stop))
                qty = pos.quantity
                entry_dt = ts

    result.trades = all_trades
    result.total_trades = len(all_trades)
    result.equity_curve = equity_curve

    if all_trades:
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        result.win_rate = round(len(wins) / len(all_trades) * 100, 2)
        result.profit_factor = round(gross_profit / gross_loss, 2)
        result.avg_rr = round(sum(t.rr_achieved for t in all_trades) / len(all_trades), 2)
        result.total_return_pct = round((equity - capital) / capital * 100, 2)

    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = round(max_dd, 2)

    return result