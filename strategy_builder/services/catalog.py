"""
Whitelisted indicators, operators, and presets for Strategy Builder.
No arbitrary code — only named indicators from this catalog.
"""
from __future__ import annotations

from typing import Any

# Operators
OPERATORS = [
    {"id": "gt", "label": ">", "arity": 2},
    {"id": "lt", "label": "<", "arity": 2},
    {"id": "gte", "label": ">=", "arity": 2},
    {"id": "lte", "label": "<=", "arity": 2},
    {"id": "eq", "label": "=", "arity": 2},
    {"id": "neq", "label": "!=", "arity": 2},
    {"id": "cross_above", "label": "Cross Above", "arity": 2},
    {"id": "cross_below", "label": "Cross Below", "arity": 2},
    {"id": "rising", "label": "Rising", "arity": 1},
    {"id": "falling", "label": "Falling", "arity": 1},
    {"id": "increasing", "label": "Increasing", "arity": 1},
    {"id": "decreasing", "label": "Decreasing", "arity": 1},
    {"id": "between", "label": "Between", "arity": 3},
    {"id": "outside", "label": "Outside", "arity": 3},
]

# Operand kinds
# type: price | volume | indicator | constant | market
PRICE_FIELDS = [
    {"id": "open", "label": "Open"},
    {"id": "high", "label": "High"},
    {"id": "low", "label": "Low"},
    {"id": "close", "label": "Close"},
]

CANDLE_FIELDS = [
    {"id": "body", "label": "Candle Body"},
    {"id": "range", "label": "Candle Range"},
    {"id": "body_pct", "label": "Body %"},
    {"id": "upper_wick", "label": "Upper Wick"},
    {"id": "lower_wick", "label": "Lower Wick"},
    {"id": "is_bullish", "label": "Bullish Candle"},
    {"id": "is_bearish", "label": "Bearish Candle"},
]

# Indicator registry: id -> metadata
INDICATORS: list[dict[str, Any]] = [
    # Trend
    {"id": "sma", "name": "SMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 44},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "ema", "name": "EMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "wma", "name": "WMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "hma", "name": "HMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "vwma", "name": "VWMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "dema", "name": "DEMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "tema", "name": "TEMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "rma", "name": "RMA / SMMA", "category": "Trend", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    # Momentum
    {"id": "rsi", "name": "RSI", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "macd", "name": "MACD Line", "category": "Momentum", "params": [
        {"key": "fast", "label": "Fast", "type": "int", "default": 12},
        {"key": "slow", "label": "Slow", "type": "int", "default": 26},
        {"key": "signal", "label": "Signal", "type": "int", "default": 9},
    ]},
    {"id": "macd_signal", "name": "MACD Signal", "category": "Momentum", "params": [
        {"key": "fast", "label": "Fast", "type": "int", "default": 12},
        {"key": "slow", "label": "Slow", "type": "int", "default": 26},
        {"key": "signal", "label": "Signal", "type": "int", "default": 9},
    ]},
    {"id": "macd_hist", "name": "MACD Histogram", "category": "Momentum", "params": [
        {"key": "fast", "label": "Fast", "type": "int", "default": 12},
        {"key": "slow", "label": "Slow", "type": "int", "default": 26},
        {"key": "signal", "label": "Signal", "type": "int", "default": 9},
    ]},
    {"id": "stoch_k", "name": "Stochastic %K", "category": "Momentum", "params": [
        {"key": "k", "label": "%K", "type": "int", "default": 14},
        {"key": "d", "label": "%D", "type": "int", "default": 3},
        {"key": "smooth", "label": "Smooth", "type": "int", "default": 3},
    ]},
    {"id": "stoch_d", "name": "Stochastic %D", "category": "Momentum", "params": [
        {"key": "k", "label": "%K", "type": "int", "default": 14},
        {"key": "d", "label": "%D", "type": "int", "default": 3},
        {"key": "smooth", "label": "Smooth", "type": "int", "default": 3},
    ]},
    {"id": "cci", "name": "CCI", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "williams_r", "name": "Williams %R", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    {"id": "roc", "name": "ROC", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 12},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "momentum", "name": "Momentum", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 10},
        {"key": "source", "label": "Source", "type": "source", "default": "close"},
    ]},
    {"id": "mfi", "name": "MFI", "category": "Momentum", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    # Trend strength
    {"id": "adx", "name": "ADX", "category": "Trend Strength", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    {"id": "di_plus", "name": "DI+", "category": "Trend Strength", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    {"id": "di_minus", "name": "DI-", "category": "Trend Strength", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    # Volatility
    {"id": "atr", "name": "ATR", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    {"id": "atr_pct", "name": "ATR %", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 14},
    ]},
    {"id": "bb_upper", "name": "Bollinger Upper", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "mult", "label": "Std Dev", "type": "float", "default": 2.0},
    ]},
    {"id": "bb_mid", "name": "Bollinger Mid", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "mult", "label": "Std Dev", "type": "float", "default": 2.0},
    ]},
    {"id": "bb_lower", "name": "Bollinger Lower", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "mult", "label": "Std Dev", "type": "float", "default": 2.0},
    ]},
    {"id": "donchian_high", "name": "Donchian High", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "donchian_low", "name": "Donchian Low", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "keltner_upper", "name": "Keltner Upper", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "mult", "label": "ATR Mult", "type": "float", "default": 1.5},
    ]},
    {"id": "keltner_lower", "name": "Keltner Lower", "category": "Volatility", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
        {"key": "mult", "label": "ATR Mult", "type": "float", "default": 1.5},
    ]},
    # Volume
    {"id": "volume", "name": "Volume", "category": "Volume", "params": []},
    {"id": "vol_sma", "name": "Volume SMA", "category": "Volume", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "rel_volume", "name": "Relative Volume", "category": "Volume", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "obv", "name": "OBV", "category": "Volume", "params": []},
    {"id": "obv_sma", "name": "OBV SMA", "category": "Volume", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "cmf", "name": "Chaikin Money Flow", "category": "Volume", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    # Structure
    {"id": "highest_high", "name": "Highest High", "category": "Structure", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "lowest_low", "name": "Lowest Low", "category": "Structure", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 20},
    ]},
    {"id": "prev_day_high", "name": "Previous Day High", "category": "Structure", "params": []},
    {"id": "prev_day_low", "name": "Previous Day Low", "category": "Structure", "params": []},
    {"id": "prev_close", "name": "Previous Close", "category": "Structure", "params": []},
]

INDICATOR_BY_ID = {i["id"]: i for i in INDICATORS}
VALID_INDICATOR_IDS = frozenset(INDICATOR_BY_ID.keys())
VALID_OPERATOR_IDS = frozenset(o["id"] for o in OPERATORS)
VALID_PRICE_IDS = frozenset(p["id"] for p in PRICE_FIELDS) | frozenset(c["id"] for c in CANDLE_FIELDS)

# Market operands (Nifty 50)
MARKET_OPERANDS = [
    {"id": "nifty_close", "label": "Nifty 50 Close"},
    {"id": "nifty_sma", "label": "Nifty 50 SMA", "params": [
        {"key": "length", "label": "Length", "type": "int", "default": 44},
    ]},
]


def catalog_payload() -> dict[str, Any]:
    return {
        "operators": OPERATORS,
        "price_fields": PRICE_FIELDS,
        "candle_fields": CANDLE_FIELDS,
        "indicators": INDICATORS,
        "market_operands": MARKET_OPERANDS,
        "categories": sorted({i["category"] for i in INDICATORS}),
    }


def default_strategy() -> dict[str, Any]:
    """Default: Close > SMA(44) AND RSI > 50."""
    return {
        "name": "My Strategy",
        "universe": "nifty200",
        "timeframe": "daily",
        "market": "NSE",
        "long_enabled": True,
        "short_enabled": False,
        "entry_long": {
            "type": "group",
            "op": "AND",
            "children": [
                {
                    "type": "condition",
                    "left": {"type": "price", "field": "close", "offset": 0},
                    "operator": "gt",
                    "right": {
                        "type": "indicator",
                        "name": "sma",
                        "params": {"length": 44, "source": "close"},
                        "offset": 0,
                    },
                },
                {
                    "type": "condition",
                    "left": {
                        "type": "indicator",
                        "name": "rsi",
                        "params": {"length": 14, "source": "close"},
                        "offset": 0,
                    },
                    "operator": "gt",
                    "right": {"type": "constant", "value": 50},
                },
            ],
        },
        "entry_short": {"type": "group", "op": "AND", "children": []},
        "exit_long": {
            "type": "group",
            "op": "OR",
            "children": [
                {
                    "type": "condition",
                    "left": {"type": "price", "field": "close", "offset": 0},
                    "operator": "lt",
                    "right": {
                        "type": "indicator",
                        "name": "ema",
                        "params": {"length": 20, "source": "close"},
                        "offset": 0,
                    },
                },
            ],
        },
        "exit_short": {"type": "group", "op": "OR", "children": []},
        "risk": {
            "capital": 1_000_000,
            "sizing": "risk_pct",  # fixed_qty | fixed_capital | risk_pct
            "risk_pct": 1.0,
            "fixed_qty": 100,
            "fixed_capital": 20000,
            "stop_type": "pct",  # pct | points | atr | none
            "stop_value": 2.0,
            "target_type": "rr",  # pct | points | rr | atr | none
            "target_value": 2.0,
            "trail_type": "none",  # none | pct | atr
            "trail_value": 0.0,
            "max_hold_bars": 20,
            "max_positions": 10,
            "max_per_symbol": 1,
            "cooldown_bars": 5,
            "commission_pct": 0.03,
            "slippage_pct": 0.05,
        },
        "entry_timing": "next_open",  # next_open | same_close (same_close discouraged)
    }


def presets() -> list[dict[str, Any]]:
    """Named presets that populate the builder."""
    def cond(left, op, right):
        return {"type": "condition", "left": left, "operator": op, "right": right}

    def price(f="close", o=0):
        return {"type": "price", "field": f, "offset": o}

    def ind(name, params=None, o=0):
        return {"type": "indicator", "name": name, "params": params or {}, "offset": o}

    def const(v):
        return {"type": "constant", "value": v}

    def group(op, *children):
        return {"type": "group", "op": op, "children": list(children)}

    base_risk = default_strategy()["risk"].copy()

    items = [
        {
            "id": "ema_crossover",
            "name": "EMA Crossover",
            "description": "EMA20 crosses above EMA50",
            "definition": {
                **default_strategy(),
                "name": "EMA Crossover",
                "entry_long": group(
                    "AND",
                    cond(ind("ema", {"length": 20, "source": "close"}), "cross_above",
                         ind("ema", {"length": 50, "source": "close"})),
                ),
                "exit_long": group(
                    "OR",
                    cond(ind("ema", {"length": 20, "source": "close"}), "cross_below",
                         ind("ema", {"length": 50, "source": "close"})),
                ),
                "risk": {**base_risk, "stop_type": "pct", "stop_value": 3.0, "target_type": "rr", "target_value": 2.0},
            },
        },
        {
            "id": "rsi_trend",
            "name": "RSI Trend",
            "description": "RSI > 50 and Close > EMA200",
            "definition": {
                **default_strategy(),
                "name": "RSI Trend",
                "entry_long": group(
                    "AND",
                    cond(ind("rsi", {"length": 14, "source": "close"}), "gt", const(50)),
                    cond(price(), "gt", ind("ema", {"length": 200, "source": "close"})),
                ),
                "exit_long": group(
                    "OR",
                    cond(ind("rsi", {"length": 14, "source": "close"}), "lt", const(45)),
                    cond(price(), "lt", ind("ema", {"length": 20, "source": "close"})),
                ),
            },
        },
        {
            "id": "macd_momentum",
            "name": "MACD Momentum",
            "description": "MACD cross above signal and MACD > 0",
            "definition": {
                **default_strategy(),
                "name": "MACD Momentum",
                "entry_long": group(
                    "AND",
                    cond(ind("macd", {"fast": 12, "slow": 26, "signal": 9}), "cross_above",
                         ind("macd_signal", {"fast": 12, "slow": 26, "signal": 9})),
                    cond(ind("macd", {"fast": 12, "slow": 26, "signal": 9}), "gt", const(0)),
                ),
                "exit_long": group(
                    "OR",
                    cond(ind("macd", {"fast": 12, "slow": 26, "signal": 9}), "cross_below",
                         ind("macd_signal", {"fast": 12, "slow": 26, "signal": 9})),
                ),
            },
        },
        {
            "id": "volume_breakout",
            "name": "Volume Breakout",
            "description": "Close > Highest High(20) and Rel Volume > 1.5",
            "definition": {
                **default_strategy(),
                "name": "Volume Breakout",
                "entry_long": group(
                    "AND",
                    cond(price(), "gt", ind("highest_high", {"length": 20})),
                    cond(ind("rel_volume", {"length": 20}), "gt", const(1.5)),
                ),
                "exit_long": group(
                    "OR",
                    cond(price(), "lt", ind("lowest_low", {"length": 10})),
                ),
                "risk": {**base_risk, "stop_type": "pct", "stop_value": 2.0, "target_type": "rr", "target_value": 2.0},
            },
        },
        {
            "id": "sma44_bounce",
            "name": "44 MA Bounce",
            "description": "Close > SMA44, near MA, RSI > 50",
            "definition": {
                **default_strategy(),
                "name": "44 MA Bounce",
                "entry_long": group(
                    "AND",
                    cond(price(), "gt", ind("sma", {"length": 44, "source": "close"})),
                    cond(ind("rsi", {"length": 14, "source": "close"}), "gt", const(50)),
                    cond(ind("adx", {"length": 14}), "gt", const(20)),
                ),
                "exit_long": group(
                    "OR",
                    cond(price(), "lt", ind("ema", {"length": 20, "source": "close"})),
                ),
            },
        },
        {
            "id": "confluence_trend",
            "name": "Confluence Trend",
            "description": "SMA44 > SMA100, Close > SMA44, RSI > 50, Vol > 1.5x",
            "definition": {
                **default_strategy(),
                "name": "Confluence Trend",
                "entry_long": group(
                    "AND",
                    cond(ind("sma", {"length": 44, "source": "close"}), "gt",
                         ind("sma", {"length": 100, "source": "close"})),
                    cond(price(), "gt", ind("sma", {"length": 44, "source": "close"})),
                    cond(ind("rsi", {"length": 14, "source": "close"}), "gt", const(50)),
                    cond(ind("rel_volume", {"length": 20}), "gt", const(1.5)),
                ),
                "exit_long": group(
                    "OR",
                    cond(price(), "lt", ind("sma", {"length": 44, "source": "close"})),
                ),
            },
        },
    ]
    return items
