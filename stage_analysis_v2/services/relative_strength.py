"""
Relative Strength vs Nifty 50 (or Bank Nifty).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading.constants import NIFTY50_SYMBOL

RS_LOOKBACK = 13  # weeks for RS trend
RS_LONG_LOOKBACK = 26


@dataclass
class RSResult:
    rating: float
    trend: str
    rs_line: list[dict]
    vs_benchmark_pct: float
    improving: bool
    benchmark: str


def _align_weekly(stock_w: pd.DataFrame, bench_w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    merged = pd.DataFrame({"stock": stock_w["close"], "bench": bench_w["close"]}).dropna()
    return merged["stock"], merged["bench"]


def compute_relative_strength(
    stock_weekly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    benchmark: str = NIFTY50_SYMBOL,
) -> RSResult:
    """
    RS rating 0-100 based on relative performance vs benchmark.
    RS line = normalized (stock/bench) ratio over time.
    """
    stock_c, bench_c = _align_weekly(stock_weekly, bench_weekly)
    if len(stock_c) < RS_LONG_LOOKBACK:
        return RSResult(50.0, "flat", [], 0.0, False, benchmark)

    ratio = stock_c / bench_c
    ratio_norm = (ratio / ratio.iloc[0]) * 100

    recent = float(ratio.iloc[-1])
    short_ago = float(ratio.iloc[-1 - RS_LOOKBACK]) if len(ratio) > RS_LOOKBACK else float(ratio.iloc[0])
    long_ago = float(ratio.iloc[-1 - RS_LONG_LOOKBACK])

    short_chg = ((recent - short_ago) / short_ago) * 100 if short_ago else 0
    long_chg = ((recent - long_ago) / long_ago) * 100 if long_ago else 0

    if short_chg > 3 and long_chg > 5:
        trend = "improving"
    elif short_chg < -3 and long_chg < -5:
        trend = "weakening"
    else:
        trend = "flat"

    improving = trend == "improving" or short_chg > 0

    rating = 50.0
    rating += min(25, max(-25, long_chg * 2))
    rating += min(15, max(-15, short_chg * 3))
    if ratio.iloc[-1] > ratio.iloc[-4:].mean():
        rating += 5
    rating = max(0, min(100, round(rating, 1)))

    rs_line = [
        {"x": idx.strftime("%Y-%m-%d"), "y": round(float(v), 2)}
        for idx, v in ratio_norm.tail(52).items()
    ]

    vs_pct = ((stock_c.iloc[-1] / stock_c.iloc[-1 - RS_LOOKBACK]) /
              (bench_c.iloc[-1] / bench_c.iloc[-1 - RS_LOOKBACK]) - 1) * 100

    return RSResult(
        rating=rating,
        trend=trend,
        rs_line=rs_line,
        vs_benchmark_pct=round(vs_pct, 2),
        improving=improving,
        benchmark=benchmark,
    )