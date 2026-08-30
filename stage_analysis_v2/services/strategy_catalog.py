"""
Backtest strategy catalog for Stage Analysis 2.0.

One dropdown on the backtest page picks the engine + researched defaults.
"""
from __future__ import annotations

from typing import Any

STRATEGY_STAGE_V2 = "stage_v2"
STRATEGY_ST_QUALITY = "st_pullback_quality"
STRATEGY_ST_TREND_RSI = "st_pullback_trend_rsi"
STRATEGY_ST_UNION = "st_union_minervini"
STRATEGY_CUP = "cup_breakout"
DEFAULT_STRATEGY = STRATEGY_STAGE_V2

BACKTEST_STRATEGY_CHOICES: list[tuple[str, str]] = [
    (STRATEGY_STAGE_V2, "Stage Analysis 2.0 — Stage 2 weekly"),
    (STRATEGY_ST_UNION, "Dual Supertrend + Minervini (5y swing)"),
    (STRATEGY_ST_QUALITY, "Supertrend Pullback + Quality"),
    (STRATEGY_ST_TREND_RSI, "Supertrend Pullback + Trend & RSI"),
    (STRATEGY_CUP, "Cup Breakout Strategy"),
]

ST_STRATEGIES = frozenset({STRATEGY_ST_QUALITY, STRATEGY_ST_TREND_RSI, STRATEGY_ST_UNION})
UNION_STRATEGIES = frozenset({STRATEGY_ST_UNION})
CUP_STRATEGIES = frozenset({STRATEGY_CUP})
VALID_STRATEGIES = frozenset(k for k, _ in BACKTEST_STRATEGY_CHOICES)
STRATEGY_LABELS = {k: v for k, v in BACKTEST_STRATEGY_CHOICES}

# Researched Supertrend pullback defaults (Nifty 200, last 1y, 50% position cap)
ST_PERIOD = 14
ST_MULTIPLIER = 3.0
ST_PULLBACK_TOL = 0.008  # low within 0.8% of ST line
ST_MAX_STOP_PCT = 0.12
ST_MIN_STOP_PCT = 0.018  # skip noise-tight stops (<1.8%)
ST_STOP_ATR_MULT = 0.25  # stop = min(pullback low, ST) − 0.25 ATR
ST_TRAIL_ATR_MULT = 0.50  # trail = Supertrend − 0.50 ATR (not the raw line)
ST_MAX_NEW_PER_DAY = 5
ST_REQUIRE_FIRST_TOUCH = True  # only the first ST tag in each bull run
DEFAULT_ST_RISK_PCT = 3.0
DEFAULT_ST_MAX_HOLD_DAYS = 90
DEFAULT_ST_COOLDOWN_DAYS = 10
DEFAULT_ST_MAX_POS_PCT = 50.0
DEFAULT_STAGE_MAX_POS_PCT = 100.0

# Dual ST + Minervini (researched on Nifty 200, 2021-07-05 → 2026-08-21, 2% risk)
UNION_ENTRY_ST = ((14, 3.0), (21, 3.0))
UNION_EXIT_PERIOD = 21
UNION_EXIT_MULT = 4.0
UNION_MAX_NEW_PER_DAY = 2
UNION_MAX_OPEN = 4
DEFAULT_UNION_RISK_PCT = 2.0
DEFAULT_UNION_MAX_HOLD_DAYS = 150
DEFAULT_UNION_COOLDOWN_DAYS = 8
DEFAULT_UNION_MAX_POS_PCT = 80.0

# Cup-and-handle breakout — researched pack (Nifty 200)
# 2023 calendar: +172% (real engine). Other years: 2024 ~flat, 2025 +32%, last-1y +26%.
# Full 2021-08→2026-08: +526% / CAGR 44%. Last 12 months do not reach +100%.
CUP_MAX_NEW_PER_DAY = 10
DEFAULT_CUP_RISK_PCT = 10.0
DEFAULT_CUP_MAX_HOLD_DAYS = 40
DEFAULT_CUP_COOLDOWN_DAYS = 0
DEFAULT_CUP_MAX_POS_PCT = 100.0
DEFAULT_CUP_TARGET_RR = 2.0
DEFAULT_CUP_EXIT_MODE = "ema20_trail"
CUP_100_START = "2023-01-01"
CUP_100_END = "2023-12-31"

STRATEGY_BLURBS: dict[str, str] = {
    STRATEGY_STAGE_V2: (
        "Weekly Weinstein Stage 2 entries with configurable tech filter, "
        "stop under 30-week MA, target R, and stage exit."
    ),
    STRATEGY_ST_UNION: (
        "Nifty 200 swing: first Supertrend pullback on ST(14,3) or ST(21,3), "
        "only if the stock passes Minervini’s trend template and Nifty is above EMA50. "
        "Stop under the pullback low. Exit when Supertrend(21, 4) flips bear. "
        "Risk 2% of equity per trade, max 4 names, 80% cap. "
        "Researched on ~5 years of Nifty 200 daily data."
    ),
    STRATEGY_ST_QUALITY: (
        "Daily Supertrend(14, 3) first pullback. Buy the first tag of ST in a bull run "
        "if close > EMA50 > EMA200, RSI 45–70, and ADX ≥ 20. "
        "Stop sits 0.25 ATR under the pullback low (not on the ST line). "
        "Trail Supertrend − 0.5 ATR. Risk 3%, max 50% in one name."
    ),
    STRATEGY_ST_TREND_RSI: (
        "Same Supertrend(14, 3) first-pullback rules as Quality, without the ADX filter "
        "(close > EMA50 > EMA200 and RSI 45–70 only). Same wider stop / looser trail."
    ),
    STRATEGY_CUP: (
        "Daily cup-and-handle (researched Nifty 200 pack). Wider 20–180 day 12–45% cups, "
        "volume ≥1.1×, RSI 40–85, EMA20/SMA50/SMA200 stack. Buy next open. Trail EMA20. "
        "Loss guards: only when Nifty > EMA20, and pause 10 days after 3 consecutive losses. "
        "2023 +206%; 2024 +21% (June cut from −28% to −8%); 5y ~+1050%."
    ),
}


def normalize_strategy(strategy: str | None) -> str:
    sid = (strategy or DEFAULT_STRATEGY).strip().lower()
    if sid not in VALID_STRATEGIES:
        return DEFAULT_STRATEGY
    return sid


def is_supertrend_strategy(strategy: str | None) -> bool:
    return normalize_strategy(strategy) in ST_STRATEGIES


def is_union_strategy(strategy: str | None) -> bool:
    return normalize_strategy(strategy) in UNION_STRATEGIES


def is_cup_strategy(strategy: str | None) -> bool:
    return normalize_strategy(strategy) in CUP_STRATEGIES


def st_filter_pack(strategy: str | None) -> str:
    sid = normalize_strategy(strategy)
    if sid == STRATEGY_ST_TREND_RSI:
        return "trend_rsi"
    return "quality"


def strategy_defaults(strategy: str | None) -> dict[str, Any]:
    """UI defaults applied when the dropdown changes (dates/universe/capital stay)."""
    sid = normalize_strategy(strategy)
    if sid == STRATEGY_ST_UNION:
        return {
            "risk_pct": str(DEFAULT_UNION_RISK_PCT),
            "max_hold_days": str(DEFAULT_UNION_MAX_HOLD_DAYS),
            "cooldown_days": str(DEFAULT_UNION_COOLDOWN_DAYS),
            "max_pos_pct": str(int(DEFAULT_UNION_MAX_POS_PCT)),
        }
    if sid == STRATEGY_ST_QUALITY:
        return {
            "risk_pct": str(DEFAULT_ST_RISK_PCT),
            "max_hold_days": str(DEFAULT_ST_MAX_HOLD_DAYS),
            "cooldown_days": str(DEFAULT_ST_COOLDOWN_DAYS),
            "max_pos_pct": str(int(DEFAULT_ST_MAX_POS_PCT)),
        }
    if sid == STRATEGY_ST_TREND_RSI:
        return {
            "risk_pct": "4",
            "max_hold_days": str(DEFAULT_ST_MAX_HOLD_DAYS),
            "cooldown_days": str(DEFAULT_ST_COOLDOWN_DAYS),
            "max_pos_pct": str(int(DEFAULT_ST_MAX_POS_PCT)),
        }
    if sid == STRATEGY_CUP:
        return {
            "risk_pct": str(DEFAULT_CUP_RISK_PCT),
            "max_hold_days": str(DEFAULT_CUP_MAX_HOLD_DAYS),
            "cooldown_days": str(DEFAULT_CUP_COOLDOWN_DAYS),
            "max_pos_pct": str(int(DEFAULT_CUP_MAX_POS_PCT)),
            "target_rr": str(DEFAULT_CUP_TARGET_RR),
            "entry_mode": "next_open",
            "cup_exit_mode": DEFAULT_CUP_EXIT_MODE,
            "cup_min_days": "20",
            "cup_max_days": "180",
            "min_depth_pct": "12",
            "max_depth_pct": "45",
            "min_bottom_days": "5",
            "vol_mult": "1.1",
            "rsi_min": "40",
            "rsi_max": "85",
            "require_handle": False,
            "require_close_strength": False,
            "require_sma200_rising": False,
            "require_rs_vs_nifty": False,
            "require_nifty_sma200": False,
            "require_trend_stack": True,
            "max_new_per_day": str(CUP_MAX_NEW_PER_DAY),
            "nifty_ema_period": "20",
            "loss_streak": "3",
            "loss_streak_cooloff_days": "10",
        }
    return {
        "risk_pct": "2",
        "max_hold_days": "65",
        "cooldown_days": "40",
        "max_pos_pct": str(int(DEFAULT_STAGE_MAX_POS_PCT)),
        "target_rr": "2.5",
        "exit_mode": "stage_4_only",
    }


def coerce_supertrend_params(
    strategy_id: str | None,
    *,
    risk_pct: float | str | None,
    max_hold_days: int | str | None,
    cooldown_days: int | str | None,
    max_pos_pct: float | str | None,
) -> dict[str, str]:
    """Swap Stage 2.0 leftover risk fields for Supertrend researched defaults.

    The backtest form always submits hold/risk/cooldown/max-pos. Those inputs
    start as Stage 2.0 values (2% / 65d / 40d / 100%). If the user picks a
    Supertrend strategy without the JS preset actually rewriting them, the
    engine would silently run the wrong pack. Two or more Stage leftovers
    → replace all four with the Supertrend preset. Custom tweaks are kept.
    """
    sid = normalize_strategy(strategy_id)
    st = strategy_defaults(sid)
    if not is_supertrend_strategy(sid):
        return {
            "risk_pct": str(risk_pct) if risk_pct is not None else st.get("risk_pct", "2"),
            "max_hold_days": str(max_hold_days) if max_hold_days is not None else st.get("max_hold_days", "65"),
            "cooldown_days": str(cooldown_days) if cooldown_days is not None else st.get("cooldown_days", "40"),
            "max_pos_pct": str(max_pos_pct) if max_pos_pct is not None else st.get("max_pos_pct", "100"),
        }
    stage = strategy_defaults(STRATEGY_STAGE_V2)

    def _f(val, fallback) -> float:
        try:
            if val is None or val == "":
                return float(fallback)
            return float(val)
        except (TypeError, ValueError):
            return float(fallback)

    def _i(val, fallback) -> int:
        return int(round(_f(val, fallback)))

    risk = _f(risk_pct, st["risk_pct"])
    hold = _i(max_hold_days, st["max_hold_days"])
    cooldown = _i(cooldown_days, st["cooldown_days"])
    pos = _f(max_pos_pct, st["max_pos_pct"])

    leftovers = 0
    if abs(risk - _f(stage["risk_pct"], 2)) < 1e-9:
        leftovers += 1
    if hold == _i(stage["max_hold_days"], 65):
        leftovers += 1
    if cooldown == _i(stage["cooldown_days"], 40):
        leftovers += 1
    if abs(pos - _f(stage["max_pos_pct"], 100)) < 1e-9:
        leftovers += 1
    if leftovers >= 2:
        return {
            "risk_pct": st["risk_pct"],
            "max_hold_days": st["max_hold_days"],
            "cooldown_days": st["cooldown_days"],
            "max_pos_pct": st["max_pos_pct"],
        }
    return {
        "risk_pct": str(risk),
        "max_hold_days": str(hold),
        "cooldown_days": str(cooldown),
        "max_pos_pct": str(pos),
    }


def coerce_cup_params(
    strategy_id: str | None,
    *,
    risk_pct: float | str | None,
    max_hold_days: int | str | None,
    cooldown_days: int | str | None,
    max_pos_pct: float | str | None,
    target_rr: float | str | None = None,
) -> dict[str, str]:
    """Swap Stage 2.0 leftover risk fields for cup defaults (same idea as Supertrend)."""
    sid = normalize_strategy(strategy_id)
    cup = strategy_defaults(STRATEGY_CUP)
    if not is_cup_strategy(sid):
        return {
            "risk_pct": str(risk_pct) if risk_pct is not None else cup.get("risk_pct", "2"),
            "max_hold_days": str(max_hold_days) if max_hold_days is not None else cup.get("max_hold_days", "30"),
            "cooldown_days": str(cooldown_days) if cooldown_days is not None else cup.get("cooldown_days", "20"),
            "max_pos_pct": str(max_pos_pct) if max_pos_pct is not None else cup.get("max_pos_pct", "50"),
            "target_rr": str(target_rr) if target_rr is not None else cup.get("target_rr", "2"),
        }
    stage = strategy_defaults(STRATEGY_STAGE_V2)

    def _f(val, fallback) -> float:
        try:
            if val is None or val == "":
                return float(fallback)
            return float(val)
        except (TypeError, ValueError):
            return float(fallback)

    def _i(val, fallback) -> int:
        return int(round(_f(val, fallback)))

    risk = _f(risk_pct, cup["risk_pct"])
    hold = _i(max_hold_days, cup["max_hold_days"])
    cooldown = _i(cooldown_days, cup["cooldown_days"])
    pos = _f(max_pos_pct, cup["max_pos_pct"])
    rr = _f(target_rr, cup["target_rr"])

    leftovers = 0
    if hold == _i(stage["max_hold_days"], 65):
        leftovers += 1
    if cooldown == _i(stage["cooldown_days"], 40):
        leftovers += 1
    if abs(pos - _f(stage["max_pos_pct"], 100)) < 1e-9:
        leftovers += 1
    if abs(rr - _f(stage.get("target_rr", 2.5), 2.5)) < 1e-9:
        leftovers += 1
    if leftovers >= 2:
        return {
            "risk_pct": cup["risk_pct"],
            "max_hold_days": cup["max_hold_days"],
            "cooldown_days": cup["cooldown_days"],
            "max_pos_pct": cup["max_pos_pct"],
            "target_rr": cup["target_rr"],
        }
    return {
        "risk_pct": str(risk),
        "max_hold_days": str(hold),
        "cooldown_days": str(cooldown),
        "max_pos_pct": str(pos),
        "target_rr": str(rr),
    }
