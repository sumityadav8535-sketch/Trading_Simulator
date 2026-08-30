"""Last-1y Stage 2.0: default vs winner-pattern (RS>=70, price above SMA150)."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
django.setup()

from stage_analysis_v2.services.backtester import (
    EntryFilters,
    MA_COND_ABOVE,
    fill_performance_metrics,
    run_stage_v2_backtest,
)
from trading.services.market_data import get_universe_symbols

OUT = ROOT / "data" / "winner_pattern_backtest.json"
START = date(2025, 8, 27)
END = date(2026, 8, 27)
NAMED = [
    "FEDERALBNK", "AUBANK", "SHRIRAMFIN", "UNIONBANK",
    "BHARATFORG", "JSWSTEEL", "SAIL",
    "BEL", "CUMMINSIND", "GVT&D", "POWERINDIA",
    "GLENMARK", "ADANIENSOL",
]


def _summarize(bt, label: str) -> dict:
    fill_performance_metrics(bt)
    named_log = [r for r in (bt.signal_log or []) if r.get("symbol") in NAMED]
    named_taken = [r for r in named_log if r.get("status") == "taken"]
    named_skip = [r for r in named_log if r.get("status") != "taken"]
    trades = [
        {
            "symbol": t.symbol,
            "signal_date": t.signal_date,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "pnl": round(float(t.pnl), 0),
            "pnl_pct": t.pnl_pct,
            "rr": t.rr_achieved,
            "rs": t.rs_rating,
            "quality": t.quality_score,
            "exit_reason": t.exit_reason,
            "days": t.days_held,
        }
        for t in bt.trades
        if t.symbol in NAMED
    ]
    wins = [t for t in bt.trades if t.pnl > 0]
    return {
        "label": label,
        "signals": bt.total_signals,
        "trades": bt.total_trades,
        "skipped_cash": bt.signals_skipped_cash,
        "win_rate": bt.win_rate,
        "total_return_pct": bt.total_return_pct,
        "cagr_pct": bt.cagr_pct,
        "max_dd_pct": bt.max_drawdown_pct,
        "profit_factor": bt.profit_factor,
        "avg_rr": bt.avg_rr,
        "avg_hold_days": bt.avg_hold_days,
        "final_capital": bt.final_cash,
        "named_signals": len(named_log),
        "named_taken": len(named_taken),
        "named_skipped": len(named_skip),
        "named_pnl": round(sum(float(r.get("pnl") or 0) for r in named_taken), 0),
        "named_trades": trades,
        "named_skips": [
            {
                "symbol": r["symbol"],
                "signal_date": r.get("signal_date"),
                "status": r.get("status"),
                "status_label": r.get("status_label"),
                "rs": r.get("rs_rating"),
                "quality": r.get("quality_score"),
            }
            for r in named_skip
        ],
        "top_trades": sorted(
            [
                {
                    "symbol": t.symbol,
                    "signal_date": t.signal_date,
                    "pnl": round(float(t.pnl), 0),
                    "pnl_pct": t.pnl_pct,
                    "rs": t.rs_rating,
                    "exit_reason": t.exit_reason,
                }
                for t in bt.trades
            ],
            key=lambda x: -x["pnl"],
        )[:12],
        "worst_trades": sorted(
            [
                {
                    "symbol": t.symbol,
                    "signal_date": t.signal_date,
                    "pnl": round(float(t.pnl), 0),
                    "pnl_pct": t.pnl_pct,
                    "rs": t.rs_rating,
                    "exit_reason": t.exit_reason,
                }
                for t in bt.trades
            ],
            key=lambda x: x["pnl"],
        )[:8],
        "wins": len(wins),
        "losses": len(bt.trades) - len(wins),
    }


def _print_run(s: dict) -> None:
    print(f"\n=== {s['label']} ===")
    print(
        f"signals {s['signals']}  trades {s['trades']}  "
        f"WR {s['win_rate']}%  ret {s['total_return_pct']}%  "
        f"CAGR {s['cagr_pct']}%  DD {s['max_dd_pct']}%  PF {s['profit_factor']}  "
        f"skipped_cash {s['skipped_cash']}"
    )
    print(
        f"named sleeve: signals {s['named_signals']} taken {s['named_taken']} "
        f"skipped {s['named_skipped']} pnl Rs {s['named_pnl']:,.0f}"
    )
    print("named trades:")
    for t in s["named_trades"]:
        print(
            f"  {t['symbol']:12} {t['signal_date']}  RS {t['rs']}  "
            f"{t['pnl_pct']}%  Rs {t['pnl']:,.0f}  {t['exit_reason']}"
        )
    if s["named_skips"]:
        print("named skips:")
        for r in s["named_skips"]:
            print(f"  {r['symbol']:12} {r['signal_date']}  {r['status']}  RS {r['rs']}")


def main():
    symbols = get_universe_symbols(nifty200_only=True)
    print(f"universe {len(symbols)}  {START} -> {END}")

    print("running default...")
    base = run_stage_v2_backtest(
        symbols=symbols,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
    )
    s_base = _summarize(base, "Default Stage 2.0 (daily_mtf, Q any, RS any)")

    print("running RS>=70 + price above SMA150...")
    pattern = run_stage_v2_backtest(
        symbols=symbols,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
        min_rs_rating=70.0,
        entry_filters=EntryFilters(
            ma_period=150,
            ma_type="sma",
            ma_condition=MA_COND_ABOVE,
        ),
    )
    s_pat = _summarize(pattern, "Winner pattern: RS>=70 and price > SMA150")

    print("running pattern + cooldown 5d (less cash lock)...")
    pat_cd = run_stage_v2_backtest(
        symbols=symbols,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
        min_rs_rating=70.0,
        cooldown_days=5,
        entry_filters=EntryFilters(
            ma_period=150,
            ma_type="sma",
            ma_condition=MA_COND_ABOVE,
        ),
    )
    s_cd = _summarize(pat_cd, "Winner pattern + 5-day cooldown")

    out = {"window": {"start": str(START), "end": str(END)}, "runs": [s_base, s_pat, s_cd]}
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _print_run(s_base)
    _print_run(s_pat)
    _print_run(s_cd)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
