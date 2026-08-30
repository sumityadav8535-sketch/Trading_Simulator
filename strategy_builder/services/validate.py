"""Validate strategy JSON AST — whitelist only, no code execution."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from strategy_builder.services.catalog import (
    VALID_INDICATOR_IDS,
    VALID_OPERATOR_IDS,
    VALID_PRICE_IDS,
    default_strategy,
)

ALLOWED_OPERAND_TYPES = frozenset({"price", "indicator", "constant", "market", "volume"})
ALLOWED_GROUP_OPS = frozenset({"AND", "OR", "NOT"})
MAX_DEPTH = 12
MAX_CONDITIONS = 80


class StrategyValidationError(ValueError):
    pass


def _validate_operand(node: dict, path: str) -> None:
    if not isinstance(node, dict):
        raise StrategyValidationError(f"{path}: operand must be object")
    t = node.get("type")
    if t not in ALLOWED_OPERAND_TYPES:
        raise StrategyValidationError(f"{path}: invalid operand type {t}")
    offset = int(node.get("offset") or 0)
    if offset < 0 or offset > 50:
        raise StrategyValidationError(f"{path}: offset out of range")
    if t == "price":
        f = node.get("field", "close")
        if f not in VALID_PRICE_IDS:
            raise StrategyValidationError(f"{path}: invalid price field {f}")
    elif t == "volume":
        pass
    elif t == "constant":
        float(node.get("value", 0))
    elif t == "market":
        field = node.get("field", "nifty_close")
        if field not in ("nifty_close", "nifty_sma"):
            raise StrategyValidationError(f"{path}: invalid market field")
        if field == "nifty_sma":
            length = int((node.get("params") or {}).get("length", 44))
            if length < 1 or length > 500:
                raise StrategyValidationError(f"{path}: bad market SMA length")
    elif t == "indicator":
        name = (node.get("name") or "").lower()
        if name not in VALID_INDICATOR_IDS:
            raise StrategyValidationError(f"{path}: indicator not allowed: {name}")
        params = node.get("params") or {}
        if not isinstance(params, dict):
            raise StrategyValidationError(f"{path}: params must be object")
        for k, v in params.items():
            if not isinstance(k, str) or len(k) > 32:
                raise StrategyValidationError(f"{path}: bad param key")
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                if abs(float(v)) > 1e9:
                    raise StrategyValidationError(f"{path}: param too large")
            elif isinstance(v, str):
                if len(v) > 32:
                    raise StrategyValidationError(f"{path}: param string too long")
            else:
                raise StrategyValidationError(f"{path}: invalid param type")


def _count_conditions(node: dict) -> int:
    if not isinstance(node, dict):
        return 0
    if node.get("type") == "condition":
        return 1
    if node.get("type") == "group":
        return sum(_count_conditions(c) for c in (node.get("children") or []))
    return 0


def _validate_node(node: dict, path: str, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise StrategyValidationError(f"{path}: nesting too deep")
    if not isinstance(node, dict):
        raise StrategyValidationError(f"{path}: node must be object")
    t = node.get("type")
    if t == "group":
        op = (node.get("op") or "AND").upper()
        if op not in ALLOWED_GROUP_OPS:
            raise StrategyValidationError(f"{path}: invalid group op")
        children = node.get("children") or []
        if not isinstance(children, list):
            raise StrategyValidationError(f"{path}: children must be list")
        if op == "NOT" and len(children) != 1:
            raise StrategyValidationError(f"{path}: NOT needs exactly 1 child")
        for i, ch in enumerate(children):
            _validate_node(ch, f"{path}.children[{i}]", depth + 1)
    elif t == "condition":
        op = node.get("operator")
        if op not in VALID_OPERATOR_IDS:
            raise StrategyValidationError(f"{path}: invalid operator {op}")
        _validate_operand(node.get("left") or {}, f"{path}.left")
        unary = op in ("rising", "falling", "increasing", "decreasing")
        if not unary:
            _validate_operand(node.get("right") or {}, f"{path}.right")
        if op in ("between", "outside"):
            _validate_operand(node.get("right2") or node.get("right") or {}, f"{path}.right2")
            if "high" in node:
                _validate_operand(node["high"], f"{path}.high")
            if "low" in node:
                _validate_operand(node["low"], f"{path}.low")
    else:
        raise StrategyValidationError(f"{path}: unknown node type {t}")


def validate_strategy(definition: dict[str, Any]) -> dict[str, Any]:
    """Return cleaned strategy definition or raise StrategyValidationError."""
    if not isinstance(definition, dict):
        raise StrategyValidationError("Strategy must be a JSON object")

    base = default_strategy()
    out = deepcopy(base)
    out.update({k: definition[k] for k in definition if k in base or k in (
        "name", "universe", "timeframe", "market", "long_enabled", "short_enabled",
        "entry_long", "entry_short", "exit_long", "exit_short", "risk", "entry_timing",
    )})

    name = str(out.get("name") or "My Strategy")[:120]
    out["name"] = name
    out["universe"] = str(out.get("universe") or "nifty200")
    if out["universe"] not in ("nifty200", "nifty100", "nifty50"):
        out["universe"] = "nifty200"
    out["timeframe"] = "daily"  # only daily supported currently
    out["long_enabled"] = bool(out.get("long_enabled", True))
    out["short_enabled"] = bool(out.get("short_enabled", False))
    out["entry_timing"] = "next_open"

    for key in ("entry_long", "entry_short", "exit_long", "exit_short"):
        node = out.get(key) or {"type": "group", "op": "AND", "children": []}
        if not isinstance(node, dict):
            raise StrategyValidationError(f"{key} invalid")
        if node.get("type") != "group":
            # wrap single condition
            if node.get("type") == "condition":
                node = {"type": "group", "op": "AND", "children": [node]}
            else:
                raise StrategyValidationError(f"{key} must be group")
        _validate_node(node, key)
        out[key] = node

    total = sum(_count_conditions(out[k]) for k in ("entry_long", "entry_short", "exit_long", "exit_short"))
    if total > MAX_CONDITIONS:
        raise StrategyValidationError("Too many conditions")

    risk = out.get("risk") or {}
    if not isinstance(risk, dict):
        raise StrategyValidationError("risk must be object")
    clean_risk = deepcopy(base["risk"])
    for k, v in risk.items():
        if k in clean_risk:
            clean_risk[k] = v
    # sanitize numbers
    clean_risk["capital"] = max(1000.0, float(clean_risk.get("capital") or 1_000_000))
    clean_risk["risk_pct"] = min(10.0, max(0.1, float(clean_risk.get("risk_pct") or 1)))
    clean_risk["fixed_qty"] = max(1, int(clean_risk.get("fixed_qty") or 100))
    clean_risk["fixed_capital"] = max(1000.0, float(clean_risk.get("fixed_capital") or 20000))
    clean_risk["stop_value"] = max(0.0, float(clean_risk.get("stop_value") or 0))
    clean_risk["target_value"] = max(0.0, float(clean_risk.get("target_value") or 0))
    clean_risk["trail_value"] = max(0.0, float(clean_risk.get("trail_value") or 0))
    clean_risk["max_hold_bars"] = max(1, min(250, int(clean_risk.get("max_hold_bars") or 20)))
    clean_risk["max_positions"] = max(1, min(50, int(clean_risk.get("max_positions") or 10)))
    clean_risk["max_per_symbol"] = max(1, min(5, int(clean_risk.get("max_per_symbol") or 1)))
    clean_risk["cooldown_bars"] = max(0, min(60, int(clean_risk.get("cooldown_bars") or 5)))
    clean_risk["commission_pct"] = max(0.0, min(1.0, float(clean_risk.get("commission_pct") or 0.03)))
    clean_risk["slippage_pct"] = max(0.0, min(1.0, float(clean_risk.get("slippage_pct") or 0.05)))
    if clean_risk.get("sizing") not in ("fixed_qty", "fixed_capital", "risk_pct"):
        clean_risk["sizing"] = "risk_pct"
    if clean_risk.get("stop_type") not in ("pct", "points", "atr", "none"):
        clean_risk["stop_type"] = "pct"
    if clean_risk.get("target_type") not in ("pct", "points", "rr", "atr", "none"):
        clean_risk["target_type"] = "rr"
    if clean_risk.get("trail_type") not in ("none", "pct", "atr"):
        clean_risk["trail_type"] = "none"
    out["risk"] = clean_risk
    return out
