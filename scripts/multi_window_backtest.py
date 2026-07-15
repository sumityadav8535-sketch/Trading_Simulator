"""Quick multi-window backtest: production swing + elite scanner history."""
from __future__ import annotations

import os
import sys
from copy import copy
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

from trading.models import StrategyConfig
from trading.services.elite_scanner_history import run_elite_scanner_history
from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import run_signal_backtest

CAPITAL = 100_000.0
RISK = 2.0
TARGET_R = 2.0
END = date.today()
WINDOWS = {
    "9m": END - timedelta(days=274),
    "1y": END - timedelta(days=365),
    "2y": END - timedelta(days=730),
    "3y": END - timedelta(days=1095),
}


def run_swing(symbols, start, end):
    cfg = copy(StrategyConfig.get_active())
    cfg.risk_pct = RISK
    cfg.min_risk_reward = TARGET_R
    return run_signal_backtest(symbols, start, end, capital=CAPITAL, config=cfg)


def run_scanner(symbols, start, end):
    cfg = copy(StrategyConfig.get_active())
    cfg.risk_pct = RISK
    hist = run_elite_scanner_history(symbols, start, end, capital=CAPITAL, config=cfg)
    s = hist.summary
    wins = [e.pnl for e in hist.entries if e.pnl > 0]
    losses = [e.pnl for e in hist.entries if e.pnl <= 0]
    gp = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0
    pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
    return {
        "trades": s.get("trades", 0),
        "win_rate": s.get("win_rate", 0),
        "return_pct": s.get("total_return_pct", 0),
        "profit_factor": pf,
    }


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    print(f"Universe: {len(symbols)} | Rs {CAPITAL:,.0f} | {RISK}% risk | {TARGET_R}R\n")

    print("=== PRODUCTION SWING (Elite + v2 fallback) ===")
    print(f"{'Window':<6} {'Trades':>7} {'WR%':>7} {'Return':>9} {'PF':>6}")
    for label, start in WINDOWS.items():
        bt = run_swing(symbols, start, END)
        print(
            f"{label:<6} {bt.total_trades:>7} {bt.win_rate:>6.1f}% "
            f"{bt.total_return_pct:>8.2f}% {bt.profit_factor:>6.2f}"
        )

    print("\n=== SCANNER ELITE (regime + pause after 2 losses) ===")
    print(f"{'Window':<6} {'Trades':>7} {'WR%':>7} {'Return':>9} {'PF':>6}")
    for label, start in WINDOWS.items():
        r = run_scanner(symbols, start, END)
        print(
            f"{label:<6} {r['trades']:>7} {r['win_rate']:>6.1f}% "
            f"{r['return_pct']:>8.2f}% {r['profit_factor']:>6.2f}"
        )


if __name__ == "__main__":
    main()