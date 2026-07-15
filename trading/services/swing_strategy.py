"""
Consistency Edge — market-filtered swing strategy (advanced tournament winner).

Entry paths (priority order):
  1. Elite 20 EMA — strong close/engulfing, 3-bar HL, ADX≥20, 1.4% EMA tol, RSI 49–58
  2. 20 EMA v2 bounce — 4-bar HL, ADX≥22, 1.2% EMA tol (best PF in testing)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from trading.models import StrategyConfig, Stock
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe
from trading.services.market_regime import DEFAULT_MIN_BULLISH, is_market_bullish
from trading.services.position_sizing import calculate_position_size
from trading.services.tournament_results import CHAMPION_NAME

STRATEGY_NAME = CHAMPION_NAME


@dataclass
class SwingSignalResult:
    symbol: str
    eval_date: date
    is_valid: bool = False
    confluence_score: int = 0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1r: Optional[float] = None
    target_2r: Optional[float] = None
    target_3r: Optional[float] = None
    risk_reward: Optional[float] = None
    position_size: int = 0
    capital_used: float = 0.0
    reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    indicator_snapshot: dict = field(default_factory=dict)
    entry_path: str = ""


def _strong_or_engulf(row) -> bool:
    return bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))


def _hl(df: pd.DataFrame, n: int) -> bool:
    if len(df) < n:
        return False
    lows = [float(df.iloc[j]["low"]) for j in range(-n, 0)]
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def _stop(hist, ema20, atr) -> float:
    e20 = float(ema20) if ema20 else 0
    a = float(atr) if atr else 0
    return min(float(hist["low"].iloc[-5:].min()), e20 - a)


def _finalize(
    result: SwingSignalResult,
    path: str,
    score: int,
    reasons: list[str],
    entry: float,
    stop: float,
    capital: float,
    config: StrategyConfig,
) -> SwingSignalResult:
    risk = entry - stop
    if risk <= 0 or risk / entry > 0.05:
        result.rejection_reasons = ["Invalid risk"]
        return result
    target_2r = entry + risk * 2
    if (target_2r - entry) / risk < config.min_risk_reward:
        result.rejection_reasons = ["RR below minimum"]
        return result
    pos = calculate_position_size(capital, config.risk_pct, entry, stop)
    if pos.quantity <= 0:
        result.rejection_reasons = ["Position size zero"]
        return result
    result.entry_path = path
    result.confluence_score = min(score, 10)
    result.reasons = reasons
    result.entry_price = round(entry, 2)
    result.stop_loss = round(stop, 2)
    result.target_1r = round(entry + risk, 2)
    result.target_2r = round(target_2r, 2)
    result.target_3r = round(entry + risk * 3, 2)
    result.risk_reward = 2.0
    result.position_size = pos.quantity
    result.capital_used = round(pos.capital_deployed, 2)
    result.is_valid = True
    return result


# Elite strong/engulf — research backtest (+67% over 2022–2025 Nifty 200, 2% risk, 2R)
ELITE_ADX_MIN = 20
ELITE_EMA_TOL = 0.014
ELITE_HL_BARS = 3
ELITE_RSI_LO = 48
ELITE_RSI_HI = 58
SCANNER_STRATEGY_LABEL = "EMA20 Elite (strong/engulf)"
REQUIRE_NIFTY_200_REGIME = True
PAUSE_AFTER_CONSECUTIVE_LOSSES = 2

_nifty_frame_cache: Optional[pd.DataFrame] = None


def _get_nifty_frame() -> pd.DataFrame:
    global _nifty_frame_cache
    if _nifty_frame_cache is None:
        from trading.services.nifty50_index import load_nifty50_frame
        _nifty_frame_cache = load_nifty50_frame()
    return _nifty_frame_cache

# v2 bounce path — tournament winner, stricter filters
V2_ADX_MIN = 22
V2_EMA_TOL = 0.012
V2_HL_BARS = 4


def _trend_ok(c, e20, e50, e200, adx, di_p, di_m, adx_min: int) -> bool:
    if not all([e20, e50, e200, adx, di_p, di_m]):
        return False
    if c <= float(e200) or float(e20) <= float(e50):
        return False
    if float(adx) < adx_min:
        return False
    if float(di_p) <= float(di_m):
        return False
    return True


def _evaluate_elite_path(
    result: SwingSignalResult,
    c: float,
    e20: Optional[float],
    e50: Optional[float],
    e200: Optional[float],
    adx: Optional[float],
    rsi: Optional[float],
    df: pd.DataFrame,
    di_p: Optional[float],
    di_m: Optional[float],
    stop: float,
    capital: float,
    config: StrategyConfig,
) -> SwingSignalResult:
    if not _trend_ok(c, e20, e50, e200, adx, di_p, di_m, ELITE_ADX_MIN):
        result.rejection_reasons = ["Elite trend/momentum filters not met"]
        return result
    if not _pullback_ok(
        c, e20, rsi, df, ELITE_EMA_TOL, ELITE_HL_BARS, ELITE_RSI_LO, ELITE_RSI_HI,
    ):
        result.rejection_reasons = ["Elite pullback structure not met"]
        return result
    return _finalize(
        result,
        "Elite 20 EMA",
        10,
        [
            "20 EMA elite: strong close/engulfing",
            f"ADX≥{ELITE_ADX_MIN} ({adx:.1f}), {ELITE_HL_BARS}-bar higher lows",
            f"RSI {rsi:.1f} ({ELITE_RSI_LO}–{ELITE_RSI_HI}), EMA tol {ELITE_EMA_TOL * 100:.1f}%",
            "+DI > -DI, price above 200 EMA, 20>50 EMA, Nifty 50>50/200 EMA",
        ],
        c,
        stop,
        capital,
        config,
    )


def evaluate_elite_scan_signal(
    symbol: str,
    eval_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    capital: Optional[float] = None,
    stock: Optional[Stock] = None,
    indicator_df: Optional[pd.DataFrame] = None,
) -> SwingSignalResult:
    """
    Scanner/backtest research rules: EMA20 + strong/engulf only (no v2 fallback).
    Matches scripts/ema20_enhance_backtest.make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3).
    """
    config = config or StrategyConfig.get_active()
    capital = float(capital or config.capital_default)

    if stock is None:
        try:
            stock = Stock.objects.get(pk=symbol)
        except Stock.DoesNotExist:
            return SwingSignalResult(
                symbol=symbol,
                eval_date=eval_date or date.today(),
                rejection_reasons=["Stock not in database"],
            )

    result = SwingSignalResult(symbol=symbol, eval_date=eval_date or date.today())

    if indicator_df is not None:
        df = indicator_df
    else:
        df = load_price_dataframe(symbol)
        if df.empty:
            result.rejection_reasons.append("No price data")
            return result
        df = compute_indicators(df)

    if not has_sufficient_history(df) or len(df) < 5:
        result.rejection_reasons.append("Insufficient history")
        return result

    if eval_date:
        df = df.loc[:pd.Timestamp(eval_date)]
        if df.empty:
            result.rejection_reasons.append("No data for date")
            return result

    row = df.iloc[-1]
    eval_ts = row.name
    result.eval_date = eval_ts.date() if hasattr(eval_ts, "date") else result.eval_date

    if not _strong_or_engulf(row):
        result.rejection_reasons = ["Requires strong close or bullish engulfing at 20 EMA"]
        return result

    if REQUIRE_NIFTY_200_REGIME:
        from trading.services.market_regime import is_nifty_strong_regime

        if not is_nifty_strong_regime(_get_nifty_frame(), eval_ts):
            result.rejection_reasons = [
                "Nifty 50 regime weak (need close above 50 EMA and 200 EMA)",
            ]
            return result

    c = float(row["close"])
    e20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None
    e50 = float(row["ema_50"]) if pd.notna(row["ema_50"]) else None
    e200 = float(row["ema_200"]) if pd.notna(row["ema_200"]) else None
    adx = float(row["adx_14"]) if pd.notna(row["adx_14"]) else None
    rsi = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None
    atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else None
    di_p = float(row["di_plus"]) if pd.notna(row.get("di_plus", np.nan)) else None
    di_m = float(row["di_minus"]) if pd.notna(row.get("di_minus", np.nan)) else None

    result.indicator_snapshot = {
        "close": c,
        "ema_20": e20,
        "ema_50": e50,
        "ema_200": e200,
        "adx_14": adx,
        "rsi_14": rsi,
        "atr_14": atr,
        "entry_path": "Elite 20 EMA",
        "strategy": SCANNER_STRATEGY_LABEL,
    }

    stop = _stop(df, e20, atr)
    return _evaluate_elite_path(
        result, c, e20, e50, e200, adx, rsi, df, di_p, di_m, stop, capital, config,
    )


def scan_swing_universe(
    symbols: list[str],
    eval_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    capital: Optional[float] = None,
    min_score: int = 7,
    elite_only: bool = True,
) -> list[SwingSignalResult]:
    """Scan symbols with EMA20 Elite rules (research script parity for scanner)."""
    import logging

    logger = logging.getLogger(__name__)
    config = config or StrategyConfig.get_active()
    evaluate = evaluate_elite_scan_signal if elite_only else evaluate_swing_signal
    results: list[SwingSignalResult] = []

    for sym in symbols:
        try:
            r = evaluate(sym, eval_date=eval_date, config=config, capital=capital)
            if r.confluence_score >= min_score:
                results.append(r)
        except Exception as exc:
            logger.exception("Swing scan failed for %s: %s", sym, exc)

    results.sort(
        key=lambda x: (-x.confluence_score, -(x.risk_reward or 0), x.symbol),
    )
    return results


def _pullback_ok(
    c, e20, rsi, df, ema_tol: float, hl_bars: int, rsi_lo: int = 48, rsi_hi: int = 58,
) -> bool:
    if rsi is None or e20 is None:
        return False
    if abs(c - e20) / e20 > ema_tol:
        return False
    if not _hl(df, hl_bars):
        return False
    return rsi_lo <= rsi <= rsi_hi


def evaluate_swing_signal(
    symbol: str,
    eval_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    capital: Optional[float] = None,
    stock: Optional[Stock] = None,
    indicator_df: Optional[pd.DataFrame] = None,
    proxy_frames: Optional[dict] = None,
    require_market_filter: bool = False,
) -> SwingSignalResult:
    config = config or StrategyConfig.get_active()
    capital = float(capital or config.capital_default)

    if stock is None:
        try:
            stock = Stock.objects.get(pk=symbol)
        except Stock.DoesNotExist:
            return SwingSignalResult(
                symbol=symbol,
                eval_date=eval_date or date.today(),
                rejection_reasons=["Stock not in database"],
            )

    result = SwingSignalResult(symbol=symbol, eval_date=eval_date or date.today())

    if indicator_df is not None:
        df = indicator_df
    else:
        df = load_price_dataframe(symbol)
        if df.empty:
            result.rejection_reasons.append("No price data")
            return result
        df = compute_indicators(df)

    if not has_sufficient_history(df) or len(df) < 5:
        result.rejection_reasons.append("Insufficient history")
        return result

    if eval_date:
        df = df.loc[:pd.Timestamp(eval_date)]
        if df.empty:
            result.rejection_reasons.append("No data for date")
            return result

    row = df.iloc[-1]
    eval_ts = row.name
    result.eval_date = eval_ts.date() if hasattr(eval_ts, "date") else result.eval_date

    if require_market_filter and proxy_frames is not None:
        if not is_market_bullish(proxy_frames, eval_ts, DEFAULT_MIN_BULLISH):
            result.rejection_reasons = ["Market regime weak (Nifty 50 below 50 EMA)"]
            return result

    c = float(row["close"])
    e20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None
    e50 = float(row["ema_50"]) if pd.notna(row["ema_50"]) else None
    e200 = float(row["ema_200"]) if pd.notna(row["ema_200"]) else None
    adx = float(row["adx_14"]) if pd.notna(row["adx_14"]) else None
    rsi = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None
    atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else None
    di_p = float(row["di_plus"]) if pd.notna(row.get("di_plus", np.nan)) else None
    di_m = float(row["di_minus"]) if pd.notna(row.get("di_minus", np.nan)) else None

    result.indicator_snapshot = {
        "close": c,
        "ema_20": e20,
        "ema_50": e50,
        "ema_200": e200,
        "adx_14": adx,
        "rsi_14": rsi,
        "atr_14": atr,
    }

    stop = _stop(df, e20, atr)

    # Path 1 — elite candle confirmation (research-validated rules)
    if _strong_or_engulf(row):
        return _evaluate_elite_path(
            result, c, e20, e50, e200, adx, rsi, df, di_p, di_m, stop, capital, config,
        )

    # Path 2 — EMA20 v2 bounce (tournament winner)
    if not _trend_ok(c, e20, e50, e200, adx, di_p, di_m, V2_ADX_MIN):
        result.rejection_reasons = ["Trend/momentum filters not met"]
        return result
    if not _pullback_ok(c, e20, rsi, df, V2_EMA_TOL, V2_HL_BARS):
        result.rejection_reasons = ["Pullback structure not met"]
        return result
    return _finalize(
        result,
        "20 EMA v2",
        8,
        [
            "20 EMA v2 pullback",
            f"ADX≥{V2_ADX_MIN} ({adx:.1f}), {V2_HL_BARS}-bar higher lows",
            f"RSI {rsi:.1f}",
        ],
        c,
        stop,
        capital,
        config,
    )