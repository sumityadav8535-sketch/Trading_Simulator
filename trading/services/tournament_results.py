"""EMA20 Elite strategy — 3-year Nifty 200 backtest results (Jun 2022 – Jun 2025)."""

TOURNAMENT_META = {
    "period": "2022-06-02 to 2025-06-01 (3 years)",
    "universe": "Nifty 200 (81 stocks with full history)",
    "capital": 100_000,
    "risk_pct": 2.0,
    "rules": "Next-day entry, 2R target, 8-day cooldown, 45-day max hold, 2% risk/trade",
}

# Filter comparison — enhanced elite vs alternatives (full 3yr, Rs 1L, 2% risk on active variant)
TOURNAMENT_RANKINGS = [
    {
        "rank": 1,
        "name": "EMA20 Elite (active)",
        "signals": 56,
        "trades": 56,
        "win_rate": 50.0,
        "pf": 1.89,
        "return_pct": 22.15,
        "selected": True,
    },
    {
        "rank": 2,
        "name": "EMA20 Elite RSI 49–58",
        "signals": 54,
        "trades": 54,
        "win_rate": 51.9,
        "pf": 2.04,
        "return_pct": 24.30,
        "selected": False,
    },
    {
        "rank": 3,
        "name": "ADX20 + tol 1.5%",
        "signals": 20,
        "trades": 20,
        "win_rate": 55.0,
        "pf": 2.21,
        "return_pct": 9.02,
        "selected": False,
    },
    {
        "rank": 4,
        "name": "ADX ≥ 20 only",
        "signals": 17,
        "trades": 17,
        "win_rate": 52.9,
        "pf": 2.01,
        "return_pct": 6.73,
        "selected": False,
    },
    {
        "rank": 5,
        "name": "EMA20 + strong/engulf (original)",
        "signals": 14,
        "trades": 14,
        "win_rate": 50.0,
        "pf": 1.76,
        "return_pct": 4.45,
        "selected": False,
    },
    {
        "rank": 6,
        "name": "3-bar HL only (no other relax)",
        "signals": 35,
        "trades": 35,
        "win_rate": 42.9,
        "pf": 1.41,
        "return_pct": 6.83,
        "selected": False,
    },
    {
        "rank": 7,
        "name": "Relaxed combo (ADX20, tol 1.5%, 3HL)",
        "signals": 68,
        "trades": 68,
        "win_rate": 42.6,
        "pf": 1.39,
        "return_pct": 12.68,
        "selected": False,
    },
    {
        "rank": 8,
        "name": "No +DI filter",
        "signals": 102,
        "trades": 102,
        "win_rate": 36.3,
        "pf": 1.06,
        "return_pct": 3.09,
        "selected": False,
    },
    {
        "rank": 9,
        "name": "2-bar HL + full relax",
        "signals": 315,
        "trades": 315,
        "win_rate": 33.3,
        "pf": 0.96,
        "return_pct": -6.78,
        "selected": False,
    },
    {
        "rank": 10,
        "name": "EMA20 v2 bounce (no candle req)",
        "signals": 16,
        "trades": 16,
        "win_rate": 37.5,
        "pf": 1.22,
        "return_pct": 1.63,
        "selected": False,
    },
]

# Calendar-year breakdown — EMA20 Elite, Rs 1L, 2% risk/trade
ANNUAL_RESULTS = [
    {
        "year": "2023",
        "label": "Jun–Dec (partial data)",
        "trades": 19,
        "win_rate": 47.4,
        "return_pct": 15.8,
        "final_capital": 115_800,
        "target_30pct": False,
    },
    {
        "year": "2024",
        "label": "Full calendar year",
        "trades": 26,
        "win_rate": 53.9,
        "return_pct": 36.4,
        "final_capital": 136_400,
        "target_30pct": True,
    },
    {
        "year": "2025",
        "label": "Jan–Jun (partial)",
        "trades": 14,
        "win_rate": 50.0,
        "return_pct": 10.0,
        "final_capital": 110_000,
        "target_30pct": False,
    },
]

CHAMPION_NAME = "EMA20 Elite"
CHAMPION_DESCRIPTION = (
    "Active strategy: strong close or bullish engulfing at 20 EMA pullback. "
    "3-bar higher lows, ADX≥20, RSI 49–58, 1.4% EMA tolerance, +DI>-DI. "
    "Rs 1L capital, 2% risk/trade (Rs 2,000 risk → Rs 4,000 target per win). "
    "3-year backtest: 56 trades, 50% win rate, PF 1.89, +22.2% total. "
    "2024 full year: 26 trades, 53.9% WR, +36.4% (Rs 1L → Rs 1.36L)."
)