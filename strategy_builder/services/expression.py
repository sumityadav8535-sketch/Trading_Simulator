"""Human-readable strategy expression + explanation from AST."""
from __future__ import annotations

from typing import Any


def _fmt_operand(op: dict | None) -> str:
    if not op:
        return "?"
    t = op.get("type")
    off = int(op.get("offset") or 0)
    suffix = f"[{off}]" if off else ""
    if t == "price":
        return f"{str(op.get('field', 'close')).upper()}{suffix}"
    if t == "volume":
        return f"VOLUME{suffix}"
    if t == "constant":
        return str(op.get("value"))
    if t == "market":
        field = op.get("field", "nifty_close")
        if field == "nifty_sma":
            n = (op.get("params") or {}).get("length", 44)
            return f"NIFTY50.SMA({n}){suffix}"
        return f"NIFTY50.CLOSE{suffix}"
    if t == "indicator":
        name = str(op.get("name", "")).upper()
        params = op.get("params") or {}
        parts = []
        for k in ("length", "fast", "slow", "signal", "k", "d", "smooth", "mult", "source"):
            if k in params:
                parts.append(str(params[k]))
        inner = ",".join(parts)
        return f"{name}({inner}){suffix}" if inner else f"{name}{suffix}"
    return "?"


_OP_LABEL = {
    "gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "=", "neq": "!=",
    "cross_above": "CROSS ABOVE", "cross_below": "CROSS BELOW",
    "rising": "RISING", "falling": "FALLING",
    "increasing": "INCREASING", "decreasing": "DECREASING",
    "between": "BETWEEN", "outside": "OUTSIDE",
}


def _fmt_condition(node: dict) -> str:
    op = node.get("operator", "")
    left = _fmt_operand(node.get("left"))
    if op in ("rising", "falling", "increasing", "decreasing"):
        return f"{left} {_OP_LABEL.get(op, op)}"
    right = _fmt_operand(node.get("right"))
    if op in ("between", "outside"):
        high = _fmt_operand(node.get("high") or node.get("right2") or node.get("right"))
        low = _fmt_operand(node.get("low") or node.get("right"))
        return f"{left} {_OP_LABEL.get(op, op)} {low} AND {high}"
    return f"{left} {_OP_LABEL.get(op, op)} {right}"


def _fmt_group(node: dict, indent: int = 0) -> str:
    if not node:
        return "(none)"
    if node.get("type") == "condition":
        return ("  " * indent) + _fmt_condition(node)
    op = (node.get("op") or "AND").upper()
    children = node.get("children") or []
    if not children:
        return ("  " * indent) + "(no conditions)"
    if op == "NOT" and len(children) == 1:
        return ("  " * indent) + "NOT (\n" + _fmt_group(children[0], indent + 1) + "\n" + ("  " * indent) + ")"
    lines = []
    for i, ch in enumerate(children):
        lines.append(_fmt_group(ch, indent))
        if i < len(children) - 1:
            lines.append(("  " * indent) + op)
    if len(children) > 1:
        return ("  " * indent) + "(\n" + "\n".join(lines) + "\n" + ("  " * indent) + ")"
    return "\n".join(lines)


def strategy_expression(definition: dict[str, Any]) -> str:
    parts = [f"STRATEGY: {definition.get('name', 'Untitled')}"]
    parts.append(f"UNIVERSE: {definition.get('universe', 'nifty200').upper()} · TIMEFRAME: DAILY")
    if definition.get("long_enabled", True):
        parts.append("\nLONG ENTRY:")
        parts.append(_fmt_group(definition.get("entry_long") or {}))
        parts.append("\nLONG EXIT:")
        parts.append(_fmt_group(definition.get("exit_long") or {}))
    if definition.get("short_enabled"):
        parts.append("\nSHORT ENTRY:")
        parts.append(_fmt_group(definition.get("entry_short") or {}))
        parts.append("\nSHORT EXIT:")
        parts.append(_fmt_group(definition.get("exit_short") or {}))
    risk = definition.get("risk") or {}
    parts.append("\nRISK:")
    parts.append(
        f"  Capital ₹{risk.get('capital', 0):,.0f} · Sizing {risk.get('sizing')} · "
        f"Risk {risk.get('risk_pct')}% · Stop {risk.get('stop_type')} {risk.get('stop_value')} · "
        f"Target {risk.get('target_type')} {risk.get('target_value')} · Max hold {risk.get('max_hold_bars')} bars"
    )
    parts.append("\nENTRY TIMING: next bar open (no look-ahead)")
    return "\n".join(parts)


def strategy_explanation(definition: dict[str, Any]) -> str:
    """Short natural-language summary."""
    name = definition.get("name") or "This strategy"
    long_on = definition.get("long_enabled", True)
    bits = [f"{name} trades the {str(definition.get('universe', 'nifty200')).replace('nifty', 'Nifty ')} universe on daily bars."]
    if long_on:
        bits.append(
            "Long entry when all configured long conditions are true on the signal bar; "
            "entry is at the next session open."
        )
        bits.append("Long exit when exit conditions fire, or stop / target / max-hold rules apply.")
    risk = definition.get("risk") or {}
    bits.append(
        f"Position sizing uses {risk.get('sizing', 'risk_pct')} "
        f"(risk {risk.get('risk_pct', 1)}% of capital when risk-based)."
    )
    bits.append("Signals only use data available on or before the signal bar (no look-ahead).")
    bits.append("Backtest uses current index constituents and may include survivorship bias.")
    return " ".join(bits)
