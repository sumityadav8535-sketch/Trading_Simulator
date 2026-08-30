"""
Focused parameter grid (not full combinatorial) with overfitting warning.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any

from strategy_builder.services.backtester import run_strategy_backtest, result_to_dict


def run_simple_grid(
    definition: dict[str, Any],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """
    Vary stop % and risk % around current strategy — lightweight grid.
    """
    base = deepcopy(definition)
    rows = []
    for stop in (1.5, 2.0, 3.0):
        for risk_pct in (0.5, 1.0, 2.0):
            d = deepcopy(base)
            d.setdefault("risk", {})
            d["risk"]["stop_type"] = "pct"
            d["risk"]["stop_value"] = stop
            d["risk"]["risk_pct"] = risk_pct
            d["name"] = f"{base.get('name', 'Strategy')} stop{stop}_risk{risk_pct}"
            try:
                r = run_strategy_backtest(d, start_date=start_date, end_date=end_date)
                rows.append({
                    "stop_pct": stop,
                    "risk_pct": risk_pct,
                    "trades": r.total_trades,
                    "return_pct": r.total_return_pct,
                    "max_dd": r.max_drawdown_pct,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "sharpe": r.sharpe,
                })
            except Exception as exc:
                rows.append({
                    "stop_pct": stop,
                    "risk_pct": risk_pct,
                    "error": str(exc),
                })
    rows.sort(key=lambda x: (-(x.get("return_pct") or -9999), x.get("max_dd") or 999))
    return {
        "warning": (
            "Optimization results are historical and may be overfit. "
            "Validate parameters using out-of-sample and walk-forward testing."
        ),
        "label": "Best historical result for selected period",
        "rows": rows,
    }
