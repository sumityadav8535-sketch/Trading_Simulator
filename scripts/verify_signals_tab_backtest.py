"""
Re-verify EMA20 Elite results shown on Signals tab vs fresh backtest.

Signals tab static tables: tournament_results.py (Jun 2022 – Jun 2025)
Live backtest engine: run_signal_backtest → evaluate_swing_signal (elite + v2)
Research elite-only: make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3, rsi 48-58)
"""
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
from scripts.ema20_enhance_backtest import backtest as elite_bt, build_cache, make_ema20_strong

CAPITAL = 100_000.0
RISK = 2.0

# Period shown in Signals tab tournament / annual tables
UI_PERIOD_START = date(2022, 6, 2)
UI_PERIOD_END = date(2025, 6, 1)

# Default form dates in signals_view
DEFAULT_START = UI_PERIOD_END - timedelta(days=3 * 365)
DEFAULT_END = UI_PERIOD_END

TODAY = date.today()


def run_elite_only(cache, config, start, end, rsi_lo=48, rsi_hi=58):
    fn = make_ema20_strong(
        adx_min=20, ema_tol=0.014, hl_bars=3, rsi_lo=rsi_lo, rsi_hi=rsi_hi,
    )
    import scripts.ema20_enhance_backtest as mod

    old_cap = mod.CAPITAL
    mod.CAPITAL = CAPITAL
    r = elite_bt("elite_only", fn, cache, config, start=start, end=end)
    mod.CAPITAL = old_cap
    return r


def run_swing_engine(symbols, start, end):
    cfg = copy(StrategyConfig.get_active())
    cfg.risk_pct = RISK
    cfg.min_risk_reward = 2.0
    return run_signal_backtest(symbols, start, end, capital=CAPITAL, config=cfg)


def fmt_row(label, trades, wr, ret, pf, extra=""):
    return f"{label:<28} {trades:>6} {wr:>6.1f}% {ret:>8.2f}% {pf:>6.2f} {extra}"


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    print(f"Loading {len(symbols)} symbols...", flush=True)
    cache = build_cache(symbols)
    print(f"Cached {len(cache)} stocks with sufficient history\n")

    config = StrategyConfig.get_active()
    config.risk_pct = RISK

    periods = [
        ("UI tournament period", UI_PERIOD_START, UI_PERIOD_END),
        ("Signals default form", DEFAULT_START, DEFAULT_END),
        ("Last 1 year", TODAY - timedelta(days=365), TODAY),
        ("Last 3 years", TODAY - timedelta(days=1095), TODAY),
    ]

    print("=" * 95)
    print(f"{'Period':<28} {'Trades':>6} {'WR':>7} {'Return':>9} {'PF':>6}  Notes")
    print("=" * 95)

    print("\n--- STATIC UI (hardcoded in tournament_results.py) ---")
    print(fmt_row("UI claims (3yr elite)", 56, 50.0, 22.15, 1.89, "Jun22-Jun25"))
    print(fmt_row("UI claims (2024 full yr)", 26, 53.9, 36.4, 2.14, "calendar 2024"))

    for pname, start, end in periods:
        print(f"\n--- {pname}: {start} to {end} ---")

        e48 = run_elite_only(cache, config, start, end, rsi_lo=48, rsi_hi=58)
        e49 = run_elite_only(cache, config, start, end, rsi_lo=49, rsi_hi=58)
        swing = run_swing_engine(symbols, start, end)
        paths = Counter(getattr(s, "entry_path", "") for s in swing.signal_history)

        print(fmt_row("Elite-only RSI 48-58", e48.trades, e48.win_rate, e48.return_pct, e48.profit_factor))
        print(fmt_row("Elite-only RSI 49-58", e49.trades, e49.win_rate, e49.return_pct, e49.profit_factor))
        print(
            fmt_row(
                "Signals engine (elite+v2)",
                swing.total_trades,
                swing.win_rate,
                swing.total_return_pct,
                swing.profit_factor,
                f"elite={paths.get('Elite 20 EMA',0)} v2={paths.get('20 EMA v2',0)}",
            )
        )

    print("\n" + "=" * 95)
    print("KEY DIFFERENCES:")
    print("  1. Signals tab TOP tables = static research numbers (Jun 2022–Jun 2025, ~81 stocks era)")
    print("  2. Live 'Run Backtest' = evaluate_swing_signal (elite + v2 fallback, no Nifty regime)")
    print("  3. Scanner uses elite-only + Nifty 50/200 regime + loss pause")
    print("  4. Fresh elite-only on FULL current DB may differ from old cached research")


if __name__ == "__main__":
    main()