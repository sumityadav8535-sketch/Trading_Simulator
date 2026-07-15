"""
Stan Weinstein Stage Analysis — rule-based detection using 30-week MA.

Stage rules (weekly timeframe):
  Stage 1 (Accumulation): Price near/around MA, MA relatively flat, choppy range.
  Stage 2 (Advancing):   Price above rising MA, higher highs & higher lows.
  Stage 3 (Distribution): Price still near/above MA but MA flattening, topping action.
  Stage 4 (Declining):   Price below falling MA, lower highs & lower lows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import yfinance as yf

MA_PERIOD = 30  # 30-week moving average (Weinstein's primary trend filter)
MA_SLOPE_LOOKBACK = 8  # weeks to measure MA direction
SWING_LOOKBACK = 20  # weeks for higher-high / lower-low structure
FLAT_MA_THRESHOLD = 1.5  # % change over lookback = "flat"
NEAR_MA_THRESHOLD = 3.0  # % distance from MA = "near"


STAGE_ACTIONS = {
    1: "Watch — Prepare to buy on Stage 2 breakout",
    2: "Buy / Hold — Ideal advancing phase",
    3: "Hold / Reduce — Take profits, tighten stops",
    4: "Avoid / Exit — Do not initiate new longs",
}

STAGE_DESCRIPTIONS = {
    1: (
        "The stock is basing after a decline. Price consolidates near the 30-week MA "
        "with the MA relatively flat. Smart money accumulates quietly."
    ),
    2: (
        "The advancing phase. Price trades above a rising 30-week MA with higher highs "
        "and higher lows. This is the primary buying zone in Weinstein's methodology."
    ),
    3: (
        "Distribution / topping. Price may still be above the MA but momentum fades, "
        "the MA slope flattens, and the stock struggles to make new highs."
    ),
    4: (
        "The declining phase. Price is below a falling 30-week MA with lower highs "
        "and lower lows. Avoid new long positions."
    ),
}


@dataclass
class StageResult:
    ticker: str
    company_name: str
    stage: int
    stage_name: str
    stage_description: str
    suggested_action: str
    current_price: float
    ma_30w: float
    price_vs_ma_pct: float
    ma_slope_pct: float
    breakout_level: Optional[float]
    support_level: Optional[float]
    stop_loss: Optional[float]
    reasons: list[str] = field(default_factory=list)
    chart_payload: dict[str, Any] = field(default_factory=dict)
    weekly_df: Optional[pd.DataFrame] = None


def normalize_ticker(ticker: str) -> str:
    """Uppercase and strip whitespace; preserve exchange suffix (.NS, .BO, etc.)."""
    return ticker.strip().upper()


def fetch_weekly_data(ticker: str, period: str = "5y") -> tuple[pd.DataFrame, str]:
    """
    Fetch weekly OHLCV via yfinance.
    Returns (dataframe, company_name).
    """
    symbol = normalize_ticker(ticker)
    yt = yf.Ticker(symbol)
    df = yt.history(period=period, interval="1wk", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for '{symbol}'. Check the ticker symbol.")

    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    info = yt.info or {}
    name = info.get("shortName") or info.get("longName") or symbol
    return df, name


def _compute_ma_slope(ma_series: pd.Series, lookback: int = MA_SLOPE_LOOKBACK) -> float:
    """Percent change in MA over `lookback` weeks."""
    if len(ma_series) < lookback + 1:
        return 0.0
    current = float(ma_series.iloc[-1])
    past = float(ma_series.iloc[-1 - lookback])
    if past == 0:
        return 0.0
    return ((current - past) / past) * 100.0


def _swing_structure(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict[str, bool]:
    """
    Compare recent swing highs/lows to prior swings.
    Returns flags for higher highs, higher lows, lower highs, lower lows.
    """
    window = df.tail(lookback)
    if len(window) < 10:
        return {
            "higher_highs": False,
            "higher_lows": False,
            "lower_highs": False,
            "lower_lows": False,
        }

    mid = len(window) // 2
    first_half = window.iloc[:mid]
    second_half = window.iloc[mid:]

    prior_high = float(first_half["high"].max())
    recent_high = float(second_half["high"].max())
    prior_low = float(first_half["low"].min())
    recent_low = float(second_half["low"].min())

    return {
        "higher_highs": recent_high > prior_high,
        "higher_lows": recent_low > prior_low,
        "lower_highs": recent_high < prior_high,
        "lower_lows": recent_low < prior_low,
    }


def detect_stage(df: pd.DataFrame) -> tuple[int, list[str], dict[str, Any]]:
    """
    Determine Weinstein stage from weekly OHLCV DataFrame.

    Decision tree (evaluated in order):
    1. Price below MA + falling MA + weak structure → Stage 4
    2. Price above MA + rising MA + strong structure → Stage 2
    3. Price near MA + flat MA + mixed structure → Stage 1
    4. Price above/near MA + flat/falling MA + topping → Stage 3
    5. Fallback based on price vs MA and MA slope
    """
    if len(df) < MA_PERIOD + MA_SLOPE_LOOKBACK:
        raise ValueError(
            f"Insufficient data ({len(df)} weeks). Need at least "
            f"{MA_PERIOD + MA_SLOPE_LOOKBACK} weeks."
        )

    df = df.copy()
    df["ma_30w"] = df["close"].rolling(MA_PERIOD).mean()
    df = df.dropna(subset=["ma_30w"])
    if df.empty:
        raise ValueError("Could not compute 30-week moving average.")

    row = df.iloc[-1]
    price = float(row["close"])
    ma = float(row["ma_30w"])
    price_vs_ma_pct = ((price - ma) / ma) * 100.0 if ma else 0.0
    ma_slope_pct = _compute_ma_slope(df["ma_30w"])

    swings = _swing_structure(df)
    ma_rising = ma_slope_pct > FLAT_MA_THRESHOLD
    ma_falling = ma_slope_pct < -FLAT_MA_THRESHOLD
    ma_flat = not ma_rising and not ma_falling
    price_above_ma = price > ma
    price_below_ma = price < ma
    price_near_ma = abs(price_vs_ma_pct) <= NEAR_MA_THRESHOLD

    reasons: list[str] = []
    reasons.append(
        f"Price {'above' if price_above_ma else 'below'} 30-week MA "
        f"({price_vs_ma_pct:+.1f}%)"
    )
    reasons.append(
        f"30-week MA slope: {ma_slope_pct:+.1f}% over {MA_SLOPE_LOOKBACK} weeks "
        f"({'rising' if ma_rising else 'falling' if ma_falling else 'flat'})"
    )

    structure_parts = []
    if swings["higher_highs"]:
        structure_parts.append("higher highs")
    if swings["higher_lows"]:
        structure_parts.append("higher lows")
    if swings["lower_highs"]:
        structure_parts.append("lower highs")
    if swings["lower_lows"]:
        structure_parts.append("lower lows")
    if structure_parts:
        reasons.append(f"Recent structure: {', '.join(structure_parts)}")
    else:
        reasons.append("Recent structure: inconclusive")

    metrics = {
        "price": price,
        "ma_30w": ma,
        "price_vs_ma_pct": price_vs_ma_pct,
        "ma_slope_pct": ma_slope_pct,
        "swings": swings,
        "ma_rising": ma_rising,
        "ma_falling": ma_falling,
        "ma_flat": ma_flat,
    }

    # Stage 4: Declining — below falling MA, weak structure
    if price_below_ma and (ma_falling or swings["lower_lows"]):
        reasons.append("Classified Stage 4: price below MA with bearish trend/structure")
        return 4, reasons, metrics

    # Stage 2: Advancing — above rising MA, bullish structure
    if price_above_ma and ma_rising and (swings["higher_highs"] or swings["higher_lows"]):
        reasons.append("Classified Stage 2: price above rising MA with bullish structure")
        return 2, reasons, metrics

    # Stage 1: Accumulation — near flat MA, basing
    if price_near_ma and ma_flat and not swings["lower_highs"]:
        reasons.append("Classified Stage 1: price basing near flat 30-week MA")
        return 1, reasons, metrics

    # Stage 3: Distribution — above/near MA but MA flattening or topping
    if price_above_ma and (ma_flat or ma_falling or swings["lower_highs"]):
        reasons.append("Classified Stage 3: topping near MA with fading momentum")
        return 3, reasons, metrics

    # Fallbacks
    if price_below_ma:
        reasons.append("Fallback Stage 4: price below 30-week MA")
        return 4, reasons, metrics
    if ma_rising and price_above_ma:
        reasons.append("Fallback Stage 2: price above rising MA")
        return 2, reasons, metrics
    if price_near_ma:
        reasons.append("Fallback Stage 1: price near 30-week MA")
        return 1, reasons, metrics
    reasons.append("Fallback Stage 3: mixed signals near/above MA")
    return 3, reasons, metrics


def _compute_key_levels(df: pd.DataFrame, stage: int, price: float, ma: float) -> tuple[
    Optional[float], Optional[float], Optional[float]
]:
    """Breakout level, support at MA, and stop-loss suggestion."""
    recent = df.tail(SWING_LOOKBACK)
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())

    if stage == 1:
        breakout = range_high
        support = min(ma, range_low)
        stop = support * 0.97
    elif stage == 2:
        breakout = range_high
        support = ma
        stop = ma * 0.95
    elif stage == 3:
        breakout = range_high
        support = ma
        stop = price * 0.93
    else:
        breakout = ma
        support = range_low
        stop = range_high

    return (
        round(breakout, 2),
        round(support, 2),
        round(stop, 2),
    )


def _build_chart_payload(df: pd.DataFrame, stage: int) -> dict[str, Any]:
    """Serialize last 104 weeks (~2 years) for Chart.js."""
    plot_df = df.tail(104).copy()
    plot_df["ma_30w"] = plot_df["close"].rolling(MA_PERIOD).mean()

    labels = [d.strftime("%Y-%m-%d") for d in plot_df.index]
    ohlc = []
    for idx, row in plot_df.iterrows():
        ohlc.append({
            "x": idx.strftime("%Y-%m-%d"),
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "c": round(float(row["close"]), 2),
        })

    ma_values = [
        round(float(v), 2) if pd.notna(v) else None
        for v in plot_df["ma_30w"]
    ]
    volumes = [int(v) for v in plot_df["volume"]]

    stage_colors = {1: "#94a3b8", 2: "#34d399", 3: "#fbbf24", 4: "#f87171"}

    return {
        "labels": labels,
        "ohlc": ohlc,
        "ma_30w": ma_values,
        "volume": volumes,
        "stage": stage,
        "stage_color": stage_colors.get(stage, "#94a3b8"),
    }


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars (Friday close)."""
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return weekly.dropna(subset=["close"])


def result_from_weekly_df(
    ticker: str,
    company_name: str,
    df: pd.DataFrame,
    *,
    include_chart: bool = True,
) -> StageResult:
    """Build a StageResult from an existing weekly OHLCV dataframe."""
    symbol = normalize_ticker(ticker)
    df = df.copy()
    df["ma_30w"] = df["close"].rolling(MA_PERIOD).mean()

    stage, reasons, metrics = detect_stage(df)
    price = metrics["price"]
    ma = metrics["ma_30w"]
    breakout, support, stop = _compute_key_levels(df, stage, price, ma)

    return StageResult(
        ticker=symbol,
        company_name=company_name or symbol,
        stage=stage,
        stage_name=STAGE_DESCRIPTIONS[stage].split(".")[0][:30] + f" (Stage {stage})",
        stage_description=STAGE_DESCRIPTIONS[stage],
        suggested_action=STAGE_ACTIONS[stage],
        current_price=price,
        ma_30w=ma,
        price_vs_ma_pct=metrics["price_vs_ma_pct"],
        ma_slope_pct=metrics["ma_slope_pct"],
        breakout_level=breakout,
        support_level=support,
        stop_loss=stop,
        reasons=reasons,
        chart_payload=_build_chart_payload(df, stage) if include_chart else {},
        weekly_df=df,
    )


def analyze_ticker(ticker: str) -> StageResult:
    """Full pipeline: fetch data, detect stage, compute levels, build chart."""
    symbol = normalize_ticker(ticker)
    df, company_name = fetch_weekly_data(symbol)
    return result_from_weekly_df(symbol, company_name, df)