"""
Soft pre-trade checklist for Elite ML Short v2 (score ≥ 6).

Only uses information available at signal time (no full-session close lookahead).
Paper/live ACTIVE entries should pass this gate in addition to base strategy filters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

# Soft mode defaults (from checklist backtest "score6")
SOFT_MIN_SCORE = 6
SOFT_PATH_HARD_GREEN = 0.5  # open→signal %; above this = hard no
SOFT_PATH_STRONG_YES = -0.1
SOFT_PATH_YELLOW = 0.15
SOFT_ML_YELLOW = 0.58
SOFT_ML_YES = 0.62
SOFT_ML_STRONG = 0.66
SOFT_MAX_LOSSES_TODAY = 2  # hard no if already this many losses


@dataclass
class ChecklistResult:
    mode: str = "soft"
    take: bool = False
    score: int = 0
    min_score: int = SOFT_MIN_SCORE
    hard_no: bool = False
    path_ret_pct: Optional[float] = None
    losses_today: int = 0
    answers: dict[str, str] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _path_ret_pct(row) -> Optional[float]:
    try:
        close = float(row["close"])
        day_open = row.get("day_open")
        if day_open is None or (isinstance(day_open, float) and pd.isna(day_open)):
            return None
        day_open = float(day_open)
        if day_open == 0:
            return None
        return (close - day_open) / day_open * 100.0
    except Exception:
        return None


def evaluate_soft_checklist(
    row,
    ts,
    ml_prob: Optional[float],
    *,
    losses_today: int = 0,
) -> ChecklistResult:
    """
    Soft scorecard (≥6) with hard-no blocks.

    Q1 Day path (open→signal)
    Q2 Structure (VWAP + EMA)
    Q3 Time of day
    Q4 ML quality
    Q5 Session risk state (paper losses today)
    """
    result = ChecklistResult(mode="soft", losses_today=int(losses_today or 0))
    score = 0
    hard_no = False
    answers: dict[str, str] = {}
    points: dict[str, int] = {}
    reasons: list[str] = []

    path = _path_ret_pct(row)
    result.path_ret_pct = round(path, 3) if path is not None else None
    hour = int(ts.hour) if hasattr(ts, "hour") else 0

    close = float(row["close"])
    below_vwap = pd.isna(row.get("vwap")) or close < float(row["vwap"])
    ema_ok = float(row["ema_9"]) < float(row["ema_21"])
    full_stack = ema_ok and float(row["ema_21"]) < float(row["ema_50"])

    # ---- Q1 day path ----
    if path is None:
        answers["Q1"] = "YELLOW"
        points["Q1"] = 0
        reasons.append("Q1: day open unavailable")
    elif path > SOFT_PATH_HARD_GREEN:
        hard_no = True
        answers["Q1"] = "NO"
        points["Q1"] = 0
        reasons.append(f"Q1 hard: path {path:+.2f}% strong green")
    elif path <= SOFT_PATH_STRONG_YES:
        score += 2
        answers["Q1"] = "YES"
        points["Q1"] = 2
    elif path <= SOFT_PATH_YELLOW:
        score += 1
        answers["Q1"] = "YELLOW"
        points["Q1"] = 1
        reasons.append(f"Q1 yellow: path {path:+.2f}%")
    else:
        # mild green but under hard threshold — no points, not hard no
        answers["Q1"] = "WEAK"
        points["Q1"] = 0
        reasons.append(f"Q1 weak: path {path:+.2f}%")

    # ---- Q2 structure ----
    if full_stack and below_vwap:
        score += 2
        answers["Q2"] = "YES+"
        points["Q2"] = 2
    elif ema_ok and below_vwap:
        score += 2
        answers["Q2"] = "YES"
        points["Q2"] = 2
    elif ema_ok or below_vwap:
        score += 1
        answers["Q2"] = "YELLOW"
        points["Q2"] = 1
        reasons.append("Q2 yellow: partial structure")
    else:
        hard_no = True
        answers["Q2"] = "NO"
        points["Q2"] = 0
        reasons.append("Q2 hard: no bearish structure")

    # ---- Q3 time ----
    if 11 <= hour <= 12:
        score += 1
        answers["Q3"] = "YES"
        points["Q3"] = 1
    elif hour in (10, 13):
        answers["Q3"] = "YELLOW"
        points["Q3"] = 0
        reasons.append(f"Q3 yellow: {hour:02d}xx hour")
    else:
        answers["Q3"] = "NO"
        points["Q3"] = 0
        reasons.append(f"Q3: hour {hour:02d} outside sweet spot")

    # ---- Q4 ML ----
    if ml_prob is None:
        hard_no = True
        answers["Q4"] = "NO"
        points["Q4"] = 0
        reasons.append("Q4 hard: ML unavailable")
    elif ml_prob >= SOFT_ML_STRONG:
        score += 2
        answers["Q4"] = "YES+"
        points["Q4"] = 2
    elif ml_prob >= SOFT_ML_YES:
        score += 2
        answers["Q4"] = "YES"
        points["Q4"] = 2
    elif ml_prob >= SOFT_ML_YELLOW:
        score += 1
        answers["Q4"] = "YELLOW"
        points["Q4"] = 1
        reasons.append(f"Q4 yellow: ml={ml_prob:.3f}")
    else:
        hard_no = True
        answers["Q4"] = "NO"
        points["Q4"] = 0
        reasons.append(f"Q4 hard: ml={ml_prob:.3f} < {SOFT_ML_YELLOW}")

    # ---- Q5 risk state ----
    losses = int(losses_today or 0)
    if losses >= SOFT_MAX_LOSSES_TODAY:
        hard_no = True
        answers["Q5"] = "NO"
        points["Q5"] = 0
        reasons.append(f"Q5 hard: {losses} losses today")
    elif losses == 0:
        score += 1
        answers["Q5"] = "YES"
        points["Q5"] = 1
    else:
        answers["Q5"] = "YELLOW"
        points["Q5"] = 0
        reasons.append("Q5 yellow: 1 loss already today")

    take = (not hard_no) and score >= SOFT_MIN_SCORE
    if not take and not hard_no:
        reasons.append(f"score {score} < {SOFT_MIN_SCORE}")

    if take:
        summary = f"Soft checklist PASS ({score}/{SOFT_MIN_SCORE})"
    elif hard_no:
        summary = f"Soft checklist HARD NO ({score}/{SOFT_MIN_SCORE})"
    else:
        summary = f"Soft checklist FAIL ({score}/{SOFT_MIN_SCORE})"

    result.take = take
    result.score = score
    result.hard_no = hard_no
    result.answers = answers
    result.points = points
    result.reasons = reasons
    result.summary = summary
    return result


def paper_losses_today(session_date=None) -> int:
    """Count closed paper losses for the IST session (for Q5)."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from trading.models import PaperTrade

        if session_date is None:
            session_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        return PaperTrade.objects.filter(session_date=session_date, pnl__lte=0).count()
    except Exception:
        return 0
