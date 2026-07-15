"""
Confluence Trend Pullback Swing Strategy — signal generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from trading.models import StrategyConfig, Stock
from trading.services.indicators import compute_indicators, get_indicator_frame, has_sufficient_history
from trading.services.market_data import load_price_dataframe
from trading.services.position_sizing import calculate_position_size

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
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


def evaluate_stock(
    symbol: str,
    eval_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    capital: Optional[float] = None,
    stock: Optional[Stock] = None,
    indicator_df: Optional[pd.DataFrame] = None,
) -> StrategyResult:
    """
    Evaluate a single stock on eval_date (or latest bar) and return signal details.
    Confluence score 0-10 based on checklist items.
    Pass indicator_df (pre-computed) in backtests to avoid repeated DB/indicator work.
    """
    config = config or StrategyConfig.get_active()
    capital = float(capital or config.capital_default)

    if stock is None:
        try:
            stock = Stock.objects.get(pk=symbol)
        except Stock.DoesNotExist:
            return StrategyResult(symbol=symbol, eval_date=eval_date or date.today(),
                                  rejection_reasons=["Stock not in database"])

    result = StrategyResult(symbol=symbol, eval_date=eval_date or date.today())

    if not stock.passes_fundamental_filters(config):
        result.rejection_reasons.append("Failed fundamental filters")
        return result

    if indicator_df is not None:
        df = indicator_df
    else:
        df = load_price_dataframe(symbol)
        if df.empty:
            result.rejection_reasons.append("No price data")
            return result
        df = compute_indicators(df)

    if not has_sufficient_history(df):
        result.rejection_reasons.append("Insufficient history for EMA200/ADX")
        return result

    if eval_date:
        ts = pd.Timestamp(eval_date)
        df = df.loc[:ts]
        if df.empty:
            result.rejection_reasons.append(f"No data on or before {eval_date}")
            return result

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    result.eval_date = row.name.date() if hasattr(row.name, "date") else result.eval_date

    score = 0
    reasons: list[str] = []
    rejections: list[str] = []

    close = float(row["close"])
    ema20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None
    ema50 = float(row["ema_50"]) if pd.notna(row["ema_50"]) else None
    ema200 = float(row["ema_200"]) if pd.notna(row["ema_200"]) else None
    adx = float(row["adx_14"]) if pd.notna(row["adx_14"]) else None
    adx_prev = float(prev["adx_14"]) if pd.notna(prev.get("adx_14", np.nan)) else None
    rsi = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None
    rsi_prev = float(prev["rsi_14"]) if pd.notna(prev.get("rsi_14", np.nan)) else None
    vol = int(row["volume"])
    vol_sma = float(row["vol_sma_20"]) if pd.notna(row["vol_sma_20"]) else None
    atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else None

    result.indicator_snapshot = {
        "close": close,
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "adx_14": adx,
        "rsi_14": rsi,
        "volume": vol,
        "vol_sma_20": vol_sma,
        "atr_14": atr,
    }

    # --- Higher timeframe bias ---
    if ema200 is None or close <= ema200:
        rejections.append("Price not above 200 EMA")
    else:
        score += 2
        reasons.append("Price above 200 EMA (uptrend bias)")

    if adx is None or adx < config.adx_min:
        rejections.append(f"ADX below minimum ({config.adx_min})")
    else:
        score += 1 if adx < config.adx_preferred else 2
        reasons.append(f"ADX {adx:.1f} confirms trend strength")
        if adx_prev and adx > adx_prev:
            score += 1
            reasons.append("ADX rising")

    # --- Pullback to support ---
    tol = config.ema_pullback_tolerance_pct / 100.0
    at_ema20 = ema20 and abs(close - ema20) / ema20 <= tol
    at_ema50 = ema50 and abs(close - ema50) / ema50 <= tol
    fib382 = float(row["fib_382"]) if pd.notna(row.get("fib_382", np.nan)) else None
    fib618 = float(row["fib_618"]) if pd.notna(row.get("fib_618", np.nan)) else None
    at_fib = False
    if fib382 and fib618:
        fib_lo, fib_hi = min(fib382, fib618), max(fib382, fib618)
        at_fib = fib_lo <= close <= fib_hi

    if at_ema20 or at_ema50 or at_fib:
        score += 2
        support = "20 EMA" if at_ema20 else ("50 EMA" if at_ema50 else "Fib 38.2-61.8%")
        reasons.append(f"Pullback to {support}")
    else:
        rejections.append("No pullback to 20/50 EMA or Fib zone")

    # --- Confluence entry triggers ---
    bullish_candle = bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))
    if bullish_candle:
        score += 2
        reasons.append("Bullish price action at support")
    else:
        rejections.append("No bullish candle pattern at support")

    if vol_sma and vol >= vol_sma * config.volume_multiplier:
        score += 1
        reasons.append(f"Volume {vol/vol_sma:.1f}x 20d average")
    else:
        rejections.append(f"Volume below {config.volume_multiplier}x 20d avg")

    if rsi is not None:
        rsi_ok = config.rsi_low <= rsi <= config.rsi_high
        rsi_rising = rsi_prev is not None and rsi > rsi_prev
        rsi_cross_50 = rsi_prev is not None and rsi_prev < 50 <= rsi
        if rsi_ok and (rsi_rising or rsi_cross_50):
            score += 2
            reasons.append(f"RSI {rsi:.1f} in zone and momentum improving")
        elif rsi_ok:
            score += 1
            reasons.append(f"RSI {rsi:.1f} in acceptable zone")
        else:
            rejections.append(f"RSI {rsi:.1f} outside {config.rsi_low}-{config.rsi_high}")

    # Resistance check (simple: recent swing high clearance)
    swing_high = float(row["swing_high"]) if pd.notna(row.get("swing_high", np.nan)) else None
    if swing_high and close < swing_high * 0.98:
        score += 1
        reasons.append("Room to swing high target")
    else:
        rejections.append("Major resistance nearby (near swing high)")

    # --- Risk levels ---
    pullback_low = float(row["low"])
    sl_candidates = [pullback_low]
    if ema20:
        sl_candidates.append(ema20 - (atr or 0) * config.atr_sl_buffer)
    if ema50:
        sl_candidates.append(ema50 - (atr or 0) * config.atr_sl_buffer)
    stop_loss = min(sl_candidates)
    if atr:
        stop_loss = min(stop_loss, pullback_low - atr * config.atr_sl_buffer)

    entry = close
    risk = entry - stop_loss
    if risk <= 0:
        rejections.append("Invalid stop loss (above entry)")
        result.rejection_reasons = rejections
        result.reasons = reasons
        result.confluence_score = min(score, 10)
        return result

    target_1r = entry + risk
    target_2r = entry + risk * 2
    target_3r = entry + risk * 3

    if swing_high and swing_high > entry:
        target_2r = min(target_2r, swing_high)

    rr = (target_2r - entry) / risk
    if rr < config.min_risk_reward:
        rejections.append(f"Risk-reward {rr:.2f} below minimum {config.min_risk_reward}")

    pos = calculate_position_size(capital, config.risk_pct, entry, stop_loss)

    result.confluence_score = min(score, 10)
    result.reasons = reasons
    result.rejection_reasons = rejections
    result.entry_price = round(entry, 2)
    result.stop_loss = round(stop_loss, 2)
    result.target_1r = round(target_1r, 2)
    result.target_2r = round(target_2r, 2)
    result.target_3r = round(target_3r, 2)
    result.risk_reward = round(rr, 2)
    result.position_size = pos.quantity
    result.capital_used = round(pos.capital_deployed, 2)

    # Valid A+ setup: score >= 7 and no critical rejections on core rules
    core_ok = (
        ema200 and close > ema200
        and adx and adx >= config.adx_min
        and (at_ema20 or at_ema50 or at_fib)
        and bullish_candle
        and rr >= config.min_risk_reward
        and pos.quantity > 0
    )
    result.is_valid = core_ok and score >= 7

    return result


def scan_universe(
    symbols: list[str],
    eval_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    capital: Optional[float] = None,
    min_score: int = 5,
) -> list[StrategyResult]:
    """Scan multiple symbols and return results sorted by confluence score."""
    config = config or StrategyConfig.get_active()
    results = []
    for sym in symbols:
        try:
            r = evaluate_stock(sym, eval_date=eval_date, config=config, capital=capital)
            if r.confluence_score >= min_score:
                results.append(r)
        except Exception as exc:
            logger.exception("Scan failed for %s: %s", sym, exc)
    results.sort(key=lambda x: (-x.confluence_score, -x.risk_reward if x.risk_reward else 0))
    return results