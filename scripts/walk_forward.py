"""Walk-forward validation: in-sample 2023-2024, out-of-sample 2024-2025."""
from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import run_signal_backtest

CAPITAL = 500_000.0
PERIODS = [
    ("In-Sample (train)", "2023–2024", date(2023, 6, 1), date(2024, 6, 1)),
    ("Out-of-Sample (validate)", "2024–2025", date(2024, 6, 1), date(2025, 6, 1)),
]

RESULTS_PATH = Path(__file__).resolve().parents[1] / "trading" / "services" / "walk_forward_results.py"


def run_period(name, label, start, end):
    r = run_signal_backtest(get_universe_symbols(), start, end, CAPITAL)
    passed = r.total_trades >= 3 and r.profit_factor >= 1.0 and r.total_return_pct > 0
    return {
        "period": name,
        "label": label,
        "signals": r.total_signals,
        "trades": r.total_trades,
        "win_rate": r.win_rate,
        "pf": r.profit_factor,
        "return_pct": r.total_return_pct,
        "max_dd": r.max_drawdown_pct,
        "expectancy_r": r.expectancy_r,
        "passed": passed,
    }


def write_results(rows):
    lines = [
        '"""Walk-forward validation results — updated by scripts/walk_forward.py."""',
        "",
        "WALK_FORWARD_META = {",
        '    "in_sample": "2023-06-01 to 2024-06-01",',
        '    "out_of_sample": "2024-06-01 to 2025-06-01",',
        '    "strategy": "Consistency Edge",',
        '    "note": "Same rules, no re-optimization between periods.",',
        "}",
        "",
        "WALK_FORWARD_RESULTS = [",
    ]
    for row in rows:
        lines.append("    {")
        for k, v in row.items():
            if isinstance(v, str):
                lines.append(f'        "{k}": "{v}",')
            elif isinstance(v, bool):
                lines.append(f'        "{k}": {str(v)},')
            elif isinstance(v, float):
                lines.append(f'        "{k}": {v},')
            else:
                lines.append(f'        "{k}": {v},')
        lines.append("    },")
    lines.append("]")

    ins = rows[0]
    oos = rows[1]
    if ins["passed"] and oos["passed"]:
        verdict = (
            f"Edge holds: in-sample PF {ins['pf']} (+{ins['return_pct']}%), "
            f"out-of-sample PF {oos['pf']} (+{oos['return_pct']}%). Strategy is not overfit to one year."
        )
    elif oos["passed"]:
        verdict = (
            f"Out-of-sample profitable (PF {oos['pf']}, +{oos['return_pct']}%). "
            f"In-sample: {ins['trades']} trades, PF {ins['pf']}."
        )
    else:
        verdict = "Out-of-sample did not pass validation thresholds."

    lines.extend(["", f'WALK_FORWARD_VERDICT = "{verdict}"', ""])
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    symbols = get_universe_symbols()
    print(f"Universe: {len(symbols)} stocks\n")
    results = []
    for name, label, start, end in PERIODS:
        row = run_period(name, label, start, end)
        results.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{name:28s} T={row['trades']:3d} WR={row['win_rate']:5.1f}% "
            f"PF={row['pf']:4.2f} ret={row['return_pct']:6.2f}% DD={row['max_dd']:5.1f}% [{status}]"
        )
    write_results(results)
    print(f"\nUpdated {RESULTS_PATH}")