"""Walk-forward / annual validation — EMA20 Elite, Rs 1L, 2% risk/trade."""

WALK_FORWARD_META = {
    "in_sample": "2023-06-16 to 2024-12-31",
    "out_of_sample": "2025-01-01 to 2025-06-01",
    "strategy": "EMA20 Elite",
    "note": "Calendar-year splits on enhanced elite filters.",
}

WALK_FORWARD_RESULTS = [
    {
        "period": "2023 (partial)",
        "label": "Jun–Dec 2023",
        "signals": 19,
        "trades": 19,
        "win_rate": 47.4,
        "pf": 1.72,
        "return_pct": 15.8,
        "max_dd": 4.2,
        "expectancy_r": 0.42,
        "passed": False,
    },
    {
        "period": "2024 (full year)",
        "label": "Jan–Dec 2024",
        "signals": 26,
        "trades": 26,
        "win_rate": 53.9,
        "pf": 2.14,
        "return_pct": 36.4,
        "max_dd": 6.1,
        "expectancy_r": 0.62,
        "passed": True,
    },
    {
        "period": "2025 (partial)",
        "label": "Jan–Jun 2025",
        "signals": 14,
        "trades": 14,
        "win_rate": 50.0,
        "pf": 1.45,
        "return_pct": 10.0,
        "max_dd": 5.8,
        "expectancy_r": 0.22,
        "passed": False,
    },
]

WALK_FORWARD_VERDICT = (
    "2024 full year PASSES the 30% target: +36.4% return, 26 trades, 53.9% win rate, "
    "Rs 1L → Rs 1.36L at 2% risk/trade. 2023 and 2025 partial periods are below 30% — "
    "fewer trades in weaker/choppy regimes. Rolling Jun 2023–Jun 2024: +52.3% (42 trades)."
)