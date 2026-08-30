"""Soft pre-trade checklist for Elite ML Long v1 (score ≥ 6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

SOFT_MIN_SCORE = 6
SOFT_PATH_HARD_RED = -0.65
SOFT_PATH_HARD_GREEN = 0.85
SOFT_PATH_STRONG_DIP = -0.05
SOFT_PATH_OK = 0.40
SOFT_ML_MIN = 0.52
SOFT_ML_YELLOW = 0.58
SOFT_ML_STRONG = 0.68
SOFT_MAX_LOSSES_TODAY = 2


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
            if pd.notna(row.get("pct_from_open")):
                return float(row["pct_from_open"])
            return None
        day_open = float(day_open)
        if day_open == 0:
            return None
        return (close - day_open) / day_open * 100.0
    except Exception:
        return None


def evaluate_soft_checklist_long(
    row,
    ts,
    ml_prob: Optional[float],
    *,
    losses_today: int = 0,
) -> ChecklistResult:
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
    above_vwap = pd.isna(row.get("vwap")) or close > float(row["vwap"])
    ema_ok = float(row["ema_9"]) > float(row["ema_21"])
    full_stack = ema_ok and float(row["ema_21"]) > float(row["ema_50"])

    # Q1 day path — avoid strong red and late-chase green
    if path is None:
        answers["Q1"] = "YELLOW"
        points["Q1"] = 0
        reasons.append("Q1: day open unavailable")
    elif path < SOFT_PATH_HARD_RED:
        hard_no = True
        answers["Q1"] = "NO"
        points["Q1"] = 0
        reasons.append(f"Q1 hard: path {path:+.2f}% strong red")
    elif path > SOFT_PATH_HARD_GREEN:
        hard_no = True
        answers["Q1"] = "NO"
        points["Q1"] = 0
        reasons.append(f"Q1 hard: path {path:+.2f}% already extended")
    elif path <= SOFT_PATH_STRONG_DIP:
        score += 2
        answers["Q1"] = "YES+"
        points["Q1"] = 2
        reasons.append(f"Q1: dip-buy context {path:+.2f}%")
    elif path <= SOFT_PATH_OK:
        score += 1
        answers["Q1"] = "YES"
        points["Q1"] = 1
    else:
        answers["Q1"] = "YELLOW"
        points["Q1"] = 0
        reasons.append(f"Q1: elevated path {path:+.2f}%")

    # Q2 structure
    if full_stack and above_vwap:
        score += 2
        answers["Q2"] = "YES+"
        points["Q2"] = 2
    elif ema_ok and above_vwap:
        score += 1
        answers["Q2"] = "YES"
        points["Q2"] = 1
    elif not above_vwap:
        hard_no = True
        answers["Q2"] = "NO"
        points["Q2"] = 0
        reasons.append("Q2 hard: below VWAP")
    else:
        answers["Q2"] = "YELLOW"
        points["Q2"] = 0

    # Q3 time — morning preferred for long champion
    if 9 <= hour <= 11:
        score += 2
        answers["Q3"] = "YES+"
        points["Q3"] = 2
    elif hour <= 12:
        score += 1
        answers["Q3"] = "YES"
        points["Q3"] = 1
    else:
        answers["Q3"] = "YELLOW"
        points["Q3"] = 0
        reasons.append("Q3: afternoon — lower score")

    # Q4 ML
    p = float(ml_prob) if ml_prob is not None else None
    if p is None:
        answers["Q4"] = "YELLOW"
        points["Q4"] = 0
    elif p >= SOFT_ML_STRONG:
        score += 2
        answers["Q4"] = "YES+"
        points["Q4"] = 2
    elif p >= SOFT_ML_YELLOW:
        score += 1
        answers["Q4"] = "YES"
        points["Q4"] = 1
    elif p < SOFT_ML_MIN:
        hard_no = True
        answers["Q4"] = "NO"
        points["Q4"] = 0
        reasons.append(f"Q4 hard: ML {p:.0%} too low")
    else:
        answers["Q4"] = "YELLOW"
        points["Q4"] = 0

    # Q5 session losses
    if losses_today >= SOFT_MAX_LOSSES_TODAY:
        hard_no = True
        answers["Q5"] = "NO"
        points["Q5"] = 0
        reasons.append(f"Q5 hard: {losses_today} losses today")
    elif losses_today == 0:
        score += 1
        answers["Q5"] = "YES"
        points["Q5"] = 1
    else:
        answers["Q5"] = "YELLOW"
        points["Q5"] = 0

    result.score = score
    result.hard_no = hard_no
    result.answers = answers
    result.points = points
    result.reasons = reasons
    result.take = (not hard_no) and score >= SOFT_MIN_SCORE
    result.summary = (
        f"score {score}/{SOFT_MIN_SCORE}"
        + (" PASS" if result.take else (" HARD NO" if hard_no else " FAIL"))
    )
    return result
