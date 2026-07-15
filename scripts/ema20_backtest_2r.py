"""EMA20 Elite (+ 20 EMA v2) — 3-year backtest at 2R, 2% risk."""
from __future__ import annotations

import os
import sys
from collections import Counter
from copy import copy
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import run_signal_backtest
from trading.services.swing_strategy import STRATEGY_NAME

CAPITAL = 100_000.0
RISK_PCT = 2.0
TARGET_R = 2.0
END = date.today()
START = END - timedelta(days=3 * 365)


def main() -> None:
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    cfg = copy(StrategyConfig.get_active())
    cfg.risk_pct = RISK_PCT
    cfg.min_risk_reward = TARGET_R

    print(f"Strategy:  {STRATEGY_NAME}")
    print(f"Period:    {START} to {END}")
    print(f"Capital:   Rs {CAPITAL:,.0f}")
    print(f"Risk:      {RISK_PCT}% per trade")
    print(f"Target:    {TARGET_R}R")
    print(f"Universe:  {len(symbols)} Nifty 200 stocks\n")

    bt = run_signal_backtest(symbols, START, END, capital=CAPITAL, config=cfg)

    wins = sum(1 for t in bt.trades if t.pnl > 0)
    losses = bt.total_trades - wins
    exits = dict(Counter(t.exit_reason for t in bt.trades))
    paths = Counter(getattr(s, "entry_path", "") for s in bt.signal_history)

    print("=== EMA20 Elite (+ 20 EMA v2) @ 2R ===")
    print(f"  Signals:       {bt.total_signals}")
    print(f"  Trades:        {bt.total_trades}")
    print(f"  Win rate:      {bt.win_rate}%  ({wins}W / {losses}L)")
    print(f"  Profit factor: {bt.profit_factor}")
    print(f"  Total return:  {bt.total_return_pct}%")
    print(f"  Final equity:  Rs {CAPITAL * (1 + bt.total_return_pct / 100):,.0f}")
    print(f"  Max drawdown:  {bt.max_drawdown_pct}%")
    print(f"  Avg R/trade:   {bt.avg_rr}")
    print(f"  Avg hold days: {bt.avg_hold_days}")
    print(f"  Exit mix:      {exits}")
    print(f"  Entry paths:   Elite={paths.get('Elite 20 EMA', 0)}, v2={paths.get('20 EMA v2', 0)}")


if __name__ == "__main__":
    main()