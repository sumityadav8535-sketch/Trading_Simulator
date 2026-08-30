"""
Evaluate strategy condition AST on indicator cache.
Boolean series / per-bar evaluation — no eval().
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_builder.services.indicators import IndicatorCache


def _resolve_series(
    operand: dict,
    cache: IndicatorCache,
    market_cache: Optional[IndicatorCache] = None,
) -> pd.Series:
    t = operand.get("type")
    offset = int(operand.get("offset") or 0)
    df = cache.df

    if t == "constant":
        val = float(operand.get("value") or 0)
        return pd.Series(val, index=df.index, dtype=float)

    if t == "price":
        field = operand.get("field", "close")
        if field in ("body", "range", "body_pct", "upper_wick", "lower_wick", "is_bullish", "is_bearish"):
            s = cache.series(field, {})
        else:
            s = cache.series(field if field in df.columns else "close", {})
    elif t == "volume":
        s = cache.series("volume", {})
    elif t == "indicator":
        s = cache.series(str(operand.get("name", "")).lower(), operand.get("params") or {})
    elif t == "market":
        if market_cache is None or market_cache.df.empty:
            return pd.Series(np.nan, index=df.index)
        field = operand.get("field", "nifty_close")
        if field == "nifty_sma":
            length = int((operand.get("params") or {}).get("length", 44))
            m = market_cache.series("sma", {"length": length, "source": "close"})
        else:
            m = market_cache.series("close", {})
        # align market to stock calendar (asof)
        s = m.reindex(df.index, method="ffill")
    else:
        s = pd.Series(np.nan, index=df.index)

    if offset:
        s = s.shift(offset)
    return s.astype(float)


def _eval_condition(
    node: dict,
    cache: IndicatorCache,
    market_cache: Optional[IndicatorCache] = None,
) -> pd.Series:
    op = node.get("operator")
    left = _resolve_series(node.get("left") or {}, cache, market_cache)
    idx = cache.df.index
    false = pd.Series(False, index=idx)

    if op in ("rising", "increasing"):
        return (left > left.shift(1)).fillna(False)
    if op in ("falling", "decreasing"):
        return (left < left.shift(1)).fillna(False)

    right = _resolve_series(node.get("right") or {"type": "constant", "value": 0}, cache, market_cache)

    if op == "gt":
        return (left > right).fillna(False)
    if op == "lt":
        return (left < right).fillna(False)
    if op == "gte":
        return (left >= right).fillna(False)
    if op == "lte":
        return (left <= right).fillna(False)
    if op == "eq":
        return ((left - right).abs() < 1e-9).fillna(False)
    if op == "neq":
        return (left != right).fillna(False)
    if op == "cross_above":
        # previous left <= right and now left > right
        prev = (left.shift(1) <= right.shift(1)).fillna(False)
        now = (left > right).fillna(False)
        return prev & now
    if op == "cross_below":
        prev = (left.shift(1) >= right.shift(1)).fillna(False)
        now = (left < right).fillna(False)
        return prev & now
    if op == "between":
        low = _resolve_series(node.get("low") or node.get("right") or {}, cache, market_cache)
        high = _resolve_series(node.get("high") or node.get("right2") or node.get("right") or {}, cache, market_cache)
        return ((left >= low) & (left <= high)).fillna(False)
    if op == "outside":
        low = _resolve_series(node.get("low") or node.get("right") or {}, cache, market_cache)
        high = _resolve_series(node.get("high") or node.get("right2") or node.get("right") or {}, cache, market_cache)
        return ((left < low) | (left > high)).fillna(False)
    return false


def eval_tree(
    node: dict | None,
    cache: IndicatorCache,
    market_cache: Optional[IndicatorCache] = None,
) -> pd.Series:
    """Return boolean Series for condition group/tree."""
    idx = cache.df.index
    if not node or not isinstance(node, dict):
        return pd.Series(False, index=idx)

    if node.get("type") == "condition":
        return _eval_condition(node, cache, market_cache)

    op = (node.get("op") or "AND").upper()
    children = node.get("children") or []
    if not children:
        # empty group: entry=false (no signal), exit=false (no exit rule)
        return pd.Series(False, index=idx)

    if op == "NOT":
        child = eval_tree(children[0], cache, market_cache)
        return (~child).fillna(False)

    series_list = [eval_tree(ch, cache, market_cache) for ch in children]
    if op == "OR":
        out = series_list[0].copy()
        for s in series_list[1:]:
            out = out | s
        return out.fillna(False)

    # AND
    out = series_list[0].copy()
    for s in series_list[1:]:
        out = out & s
    return out.fillna(False)


def signal_dates(
    mask: pd.Series,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> list[pd.Timestamp]:
    m = mask.fillna(False)
    if start is not None:
        m = m.loc[m.index >= start]
    if end is not None:
        m = m.loc[m.index <= end]
    return list(m.index[m])
