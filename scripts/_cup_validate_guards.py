"""Real-engine check of Nifty EMA20 + 3-loss/10d cooloff."""
from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.cup_breakout import CUP_EXIT_EMA20, CupParams, run_cup_breakout_backtest
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols

WINDOWS = [
    ("y2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("y2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("y2025", date(2025, 1, 1), date(2025, 12, 31)),
    ("full", date(2021, 8, 25), date(2026, 8, 24)),
]


def pack(**kw):
    return CupParams(
        cup_min_days=20,
        cup_max_days=180,
        min_depth_pct=12.0,
        max_depth_pct=45.0,
        min_bottom_days=5,
        min_left_days=7,
        min_recovery_days=5,
        pivot_width=5,
        require_sma200_rising=False,
        require_rs_vs_nifty=False,
        require_close_strength=False,
        require_trend_stack=True,
        vol_mult=1.1,
        rsi_min=40.0,
        rsi_max=85.0,
        max_new_per_day=10,
        cup_exit_mode=CUP_EXIT_EMA20,
        skip_entry_gap_down_pct=2.0,
        **kw,
    )


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    guarded = pack(nifty_ema_period=20, loss_streak=3, loss_streak_cooloff_days=10)
    print("REAL ENGINE  nema20 + 3-loss / 10d cooloff", flush=True)
    for name, start, end in WINDOWS:
        r = run_cup_breakout_backtest(
            symbols=symbols,
            start_date=start,
            end_date=end,
            capital=1_000_000.0,
            risk_pct=10.0,
            max_hold_days=40,
            cooldown_days=0,
            max_pos_pct=100.0,
            params=guarded,
        )
        print(
            f"{name:6s} {start}→{end} ret={r.total_return_pct:+7.1f}% n={r.total_trades:3d} "
            f"WR={r.win_rate:5.1f}% PF={r.profit_factor:5.2f} DD={r.max_drawdown_pct:5.1f}% "
            f"final={r.final_capital:,.0f}",
            flush=True,
        )
        if r.monthly_returns:
            print("  months:")
            for m in r.monthly_returns:
                mark = " <<" if m["pnl"] <= -50000 else ""
                print(f"    {m['month']}  {m['pnl']:+10,.0f}  {m['return_pct']:+7.2f}%  eq={m['equity']:,.0f}{mark}")


if __name__ == "__main__":
    main()
