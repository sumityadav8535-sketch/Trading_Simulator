"""
Strategy Builder portfolio backtester.

Architecture mirrors Stage 2.0 shared-capital model:
  - Signal on bar T close (conditions use T and earlier only)
  - Entry at T+1 open (no look-ahead)
  - Shared cash pool, risk-based or fixed sizing
  - Exits: condition / stop / target / trail / max hold / eod_force

Reuses: market_data, position_sizing, chart helpers.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_builder.services.indicators import IndicatorCache
from strategy_builder.services.signal_engine import eval_tree
from strategy_builder.services.validate import validate_strategy
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

logger = logging.getLogger(__name__)


@dataclass
class SBTrade:
    symbol: str
    direction: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    pnl: float
    pnl_pct: float
    days_held: int
    exit_reason: str


@dataclass
class SBBacktestResult:
    strategy_name: str
    start_date: date
    end_date: date
    capital: float
    final_capital: float = 0.0
    net_profit: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_hold_days: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    stocks_scanned: int = 0
    total_signals: int = 0
    trades: list[SBTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    drawdown_curve: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)
    yearly_returns: list[dict] = field(default_factory=list)
    stock_stats: list[dict] = field(default_factory=list)
    exit_breakdown: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _universe_symbols(universe: str) -> list[str]:
    u = (universe or "nifty200").lower()
    if u == "nifty50":
        # approximate: nifty100 is closest available flag if nifty50 not stored
        return [s for s in get_universe_symbols(nifty100_only=True) if s != NIFTY50_SYMBOL][:50]
    from trading.services.market_data import resolve_universe_symbols

    return [s for s in resolve_universe_symbols(u) if s != NIFTY50_SYMBOL]


def _cagr(start: float, end: float, d0: date, d1: date) -> float:
    if start <= 0 or end <= 0:
        return 0.0
    years = max((d1 - d0).days, 1) / 365.25
    if years <= 0:
        return 0.0
    return round(((end / start) ** (1 / years) - 1) * 100, 2)


def _costs(entry: float, exit_p: float, qty: int, commission_pct: float, slippage_pct: float) -> float:
    notional = (entry + exit_p) * qty
    return notional * (commission_pct + slippage_pct) / 100.0


def _stop_target(entry: float, direction: str, risk: dict, atr: float) -> tuple[float, float]:
    stop_type = risk.get("stop_type", "pct")
    stop_val = float(risk.get("stop_value") or 0)
    tgt_type = risk.get("target_type", "rr")
    tgt_val = float(risk.get("target_value") or 0)

    if direction == "long":
        if stop_type == "none" or stop_val <= 0:
            stop = entry * 0.5  # effectively unused floor
        elif stop_type == "pct":
            stop = entry * (1 - stop_val / 100.0)
        elif stop_type == "points":
            stop = entry - stop_val
        elif stop_type == "atr":
            stop = entry - max(atr, 1e-6) * stop_val
        else:
            stop = entry * 0.98

        risk_amt = max(entry - stop, 1e-6)
        if tgt_type == "none" or tgt_val <= 0:
            target = entry * 10
        elif tgt_type == "pct":
            target = entry * (1 + tgt_val / 100.0)
        elif tgt_type == "points":
            target = entry + tgt_val
        elif tgt_type == "atr":
            target = entry + max(atr, 1e-6) * tgt_val
        else:  # rr
            target = entry + risk_amt * tgt_val
        return round(stop, 2), round(target, 2)

    # short
    if stop_type == "none" or stop_val <= 0:
        stop = entry * 1.5
    elif stop_type == "pct":
        stop = entry * (1 + stop_val / 100.0)
    elif stop_type == "points":
        stop = entry + stop_val
    elif stop_type == "atr":
        stop = entry + max(atr, 1e-6) * stop_val
    else:
        stop = entry * 1.02
    risk_amt = max(stop - entry, 1e-6)
    if tgt_type == "none" or tgt_val <= 0:
        target = entry * 0.1
    elif tgt_type == "pct":
        target = entry * (1 - tgt_val / 100.0)
    elif tgt_type == "points":
        target = entry - tgt_val
    elif tgt_type == "atr":
        target = entry - max(atr, 1e-6) * tgt_val
    else:
        target = entry - risk_amt * tgt_val
    return round(stop, 2), round(target, 2)


def _qty(sizing: str, capital_eq: float, cash: float, entry: float, stop: float, risk: dict) -> int:
    if entry <= 0:
        return 0
    if sizing == "fixed_qty":
        q = int(risk.get("fixed_qty") or 100)
    elif sizing == "fixed_capital":
        cap = float(risk.get("fixed_capital") or 20000)
        q = int(cap // entry)
    else:
        ps = calculate_position_size(capital_eq, float(risk.get("risk_pct") or 1), entry, stop)
        q = int(ps.quantity)
    max_q = int(cash // entry)
    return max(0, min(q, max_q))


def _monthly_yearly(trades: list[SBTrade], capital: float) -> tuple[list[dict], list[dict]]:
    monthly: dict[str, float] = defaultdict(float)
    yearly: dict[str, float] = defaultdict(float)
    for t in trades:
        m = t.exit_date[:7]
        y = t.exit_date[:4]
        monthly[m] += t.pnl
        yearly[y] += t.pnl
    m_out = [{"month": k, "pnl": round(v, 2), "return_pct": round(v / capital * 100, 2)} for k, v in sorted(monthly.items())]
    y_out = [{"year": k, "pnl": round(v, 2), "return_pct": round(v / capital * 100, 2)} for k, v in sorted(yearly.items())]
    return m_out, y_out


def _stock_stats(trades: list[SBTrade]) -> list[dict]:
    by: dict[str, list[SBTrade]] = defaultdict(list)
    for t in trades:
        by[t.symbol].append(t)
    rows = []
    for sym, ts in by.items():
        wins = [t for t in ts if t.pnl > 0]
        losses = [t for t in ts if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1e-9
        rows.append({
            "symbol": sym,
            "trades": len(ts),
            "win_rate": round(len(wins) / len(ts) * 100, 1),
            "net_pnl": round(sum(t.pnl for t in ts), 2),
            "avg_pnl": round(sum(t.pnl for t in ts) / len(ts), 2),
            "profit_factor": round(gp / gl, 2),
        })
    rows.sort(key=lambda r: -r["net_pnl"])
    return rows[:40]


def _benchmark(start: date, end: date, capital: float) -> dict:
    df = load_price_dataframe(NIFTY50_SYMBOL, start=start, end=end)
    if df.empty or len(df) < 2:
        return {}
    p0 = float(df.iloc[0]["close"])
    p1 = float(df.iloc[-1]["close"])
    ret = (p1 / p0 - 1) * 100 if p0 else 0
    # simple max dd
    peak = p0
    max_dd = 0.0
    for c in df["close"]:
        peak = max(peak, float(c))
        max_dd = max(max_dd, (peak - float(c)) / peak * 100 if peak else 0)
    return {
        "name": "Nifty 50 Buy & Hold",
        "return_pct": round(ret, 2),
        "cagr_pct": _cagr(p0, p1, start, end),
        "max_drawdown_pct": round(max_dd, 2),
        "final_capital": round(capital * (1 + ret / 100), 2),
    }


def run_strategy_backtest(
    definition: dict[str, Any],
    start_date: date | None = None,
    end_date: date | None = None,
) -> SBBacktestResult:
    definition = validate_strategy(definition)
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))
    risk = definition["risk"]
    capital = float(risk["capital"])

    result = SBBacktestResult(
        strategy_name=definition.get("name") or "Strategy",
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        warnings=[
            "Backtest uses current constituents and may contain survivorship bias.",
            "Signals use next-bar open entry to avoid look-ahead bias.",
        ],
    )

    symbols = _universe_symbols(definition.get("universe", "nifty200"))
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    market_cache = IndicatorCache(nifty_df) if not nifty_df.empty else None

    frames: dict[str, pd.DataFrame] = {}
    caches: dict[str, IndicatorCache] = {}
    entry_masks: dict[str, pd.Series] = {}
    exit_masks: dict[str, pd.Series] = {}
    atr_series: dict[str, pd.Series] = {}

    long_on = bool(definition.get("long_enabled", True))
    short_on = bool(definition.get("short_enabled", False))

    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 80:
            continue
        frames[sym] = df
        cache = IndicatorCache(df)
        caches[sym] = cache
        try:
            if long_on:
                entry_masks[sym] = eval_tree(definition.get("entry_long"), cache, market_cache)
                exit_masks[sym] = eval_tree(definition.get("exit_long"), cache, market_cache)
            else:
                entry_masks[sym] = pd.Series(False, index=df.index)
                exit_masks[sym] = pd.Series(False, index=df.index)
            # short: invert direction handling in sim
            if short_on:
                short_entry = eval_tree(definition.get("entry_short"), cache, market_cache)
                entry_masks[sym] = entry_masks[sym] | short_entry  # marked separately below
            atr_series[sym] = cache.series("atr", {"length": 14})
        except Exception:
            logger.exception("Signal eval failed for %s", sym)
            continue

    result.stocks_scanned = len(frames)

    # Build signal events: (entry_day, symbol, direction, signal_day)
    # Re-eval short separately for direction
    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    total_signals = 0

    for sym, df in frames.items():
        cache = caches[sym]
        long_mask = eval_tree(definition.get("entry_long"), cache, market_cache) if long_on else pd.Series(False, index=df.index)
        short_mask = eval_tree(definition.get("entry_short"), cache, market_cache) if short_on else pd.Series(False, index=df.index)
        exit_long = eval_tree(definition.get("exit_long"), cache, market_cache) if long_on else pd.Series(False, index=df.index)
        exit_short = eval_tree(definition.get("exit_short"), cache, market_cache) if short_on else pd.Series(False, index=df.index)
        exit_masks[sym + "|long"] = exit_long
        exit_masks[sym + "|short"] = exit_short

        for i in range(len(df) - 1):
            ts = df.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            entry_day = df.index[i + 1]
            if entry_day > end_ts:
                continue
            if long_on and bool(long_mask.iloc[i]):
                total_signals += 1
                signals_by_day[entry_day].append({
                    "symbol": sym, "direction": "long", "signal_date": ts, "bar_i": i,
                })
            if short_on and bool(short_mask.iloc[i]):
                total_signals += 1
                signals_by_day[entry_day].append({
                    "symbol": sym, "direction": "short", "signal_date": ts, "bar_i": i,
                })

    result.total_signals = total_signals

    calendar_set: set[pd.Timestamp] = set()
    for df in frames.values():
        calendar_set.update(df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist())
    calendar = sorted(calendar_set)
    if not calendar:
        result.final_capital = capital
        result.equity_curve = [{"date": str(start_date), "equity": capital}]
        return result

    cash = float(capital)
    opens: dict[str, dict] = {}  # key symbol|dir
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[SBTrade] = []
    equity_curve = [{"date": str(start_date), "equity": capital}]
    drawdown_curve: list[dict] = []
    peak = capital
    max_dd = 0.0
    daily_returns: list[float] = []
    prev_eq = capital

    def equity_now() -> float:
        return cash + sum(p["notional"] for p in opens.values())

    def record(ts: pd.Timestamp):
        nonlocal peak, max_dd, prev_eq
        eq = equity_now()
        d = str(ts.date())
        equity_curve.append({"date": d, "equity": round(eq, 2)})
        if prev_eq > 0:
            daily_returns.append((eq - prev_eq) / prev_eq)
        prev_eq = eq
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)
        drawdown_curve.append({"date": d, "dd": round(dd, 2)})

    max_pos = int(risk.get("max_positions") or 10)
    cooldown = int(risk.get("cooldown_bars") or 5)
    max_hold = int(risk.get("max_hold_bars") or 20)
    commission = float(risk.get("commission_pct") or 0.03)
    slippage = float(risk.get("slippage_pct") or 0.05)
    trail_type = risk.get("trail_type") or "none"
    trail_val = float(risk.get("trail_value") or 0)

    for ts in calendar:
        # exits
        closed_keys = []
        for key, pos in list(opens.items()):
            sym = pos["symbol"]
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            pos["hold_days"] += 1
            direction = pos["direction"]

            # trailing
            if trail_type == "pct" and trail_val > 0:
                if direction == "long":
                    pos["stop"] = max(pos["stop"], close * (1 - trail_val / 100))
                else:
                    pos["stop"] = min(pos["stop"], close * (1 + trail_val / 100))
            elif trail_type == "atr" and trail_val > 0:
                atr = float(atr_series.get(sym, pd.Series(dtype=float)).get(ts, 0) or 0)
                if atr > 0:
                    if direction == "long":
                        pos["stop"] = max(pos["stop"], close - atr * trail_val)
                    else:
                        pos["stop"] = min(pos["stop"], close + atr * trail_val)

            exit_price = None
            reason = ""

            if direction == "long":
                if low <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop_loss"
                elif high >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if high >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop_loss"
                elif low <= pos["target"]:
                    exit_price, reason = pos["target"], "target"

            if exit_price is None:
                # condition exit
                mask_key = f"{sym}|{direction}"
                em = exit_masks.get(mask_key)
                if em is not None and ts in em.index and bool(em.loc[ts]):
                    exit_price, reason = close, "condition_exit"

            if exit_price is None and pos["hold_days"] >= max_hold:
                exit_price, reason = close, "time_exit"

            if exit_price is None:
                continue

            # stop preferred already handled before target
            costs = _costs(pos["entry"], exit_price, pos["qty"], commission, slippage)
            if direction == "long":
                gross = (exit_price - pos["entry"]) * pos["qty"]
            else:
                gross = (pos["entry"] - exit_price) * pos["qty"]
            pnl = gross - costs
            cash += pos["notional"] + pnl
            trades.append(SBTrade(
                symbol=sym,
                direction=direction,
                signal_date=pos["signal_date"],
                entry_date=pos["entry_date"],
                exit_date=str(ts.date()),
                entry_price=round(pos["entry"], 2),
                exit_price=round(exit_price, 2),
                stop_loss=round(pos["stop"], 2),
                target=round(pos["target"], 2),
                quantity=pos["qty"],
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / pos["notional"] * 100, 2) if pos["notional"] else 0,
                days_held=pos["hold_days"],
                exit_reason=reason,
            ))
            last_exit[key] = ts
            closed_keys.append(key)

        for k in closed_keys:
            opens.pop(k, None)

        # entries
        day_sigs = sorted(
            signals_by_day.get(ts, []),
            key=lambda s: s["symbol"],
        )
        for sig in day_sigs:
            if len(opens) >= max_pos:
                break
            sym = sig["symbol"]
            direction = sig["direction"]
            key = f"{sym}|{direction}"
            if key in opens:
                continue
            # per symbol max 1 by default
            if any(p["symbol"] == sym for p in opens.values()):
                continue
            prev = last_exit.get(key)
            if prev is not None:
                # approximate cooldown by trading days count later — use calendar days
                if (ts - prev).days < cooldown:
                    continue
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            entry = float(df.loc[ts, "open"])
            if direction == "long":
                entry *= (1 + slippage / 100)
            else:
                entry *= (1 - slippage / 100)
            atr = float(atr_series.get(sym, pd.Series(dtype=float)).get(ts, 0) or 0)
            if atr == 0 or math.isnan(atr):
                atr = entry * 0.02
            stop, target = _stop_target(entry, direction, risk, atr)
            # for risk sizing stop must be correct side
            if direction == "long" and stop >= entry:
                continue
            if direction == "short" and stop <= entry:
                continue
            eq = equity_now()
            if cash <= 0:
                continue
            # calculate_position_size expects long stop < entry
            if direction == "long":
                q = _qty(risk.get("sizing", "risk_pct"), eq, cash, entry, stop, risk)
            else:
                # risk per share = stop - entry
                risk_ps = stop - entry
                if risk_ps <= 0:
                    continue
                if risk.get("sizing") == "fixed_qty":
                    q = int(risk.get("fixed_qty") or 100)
                elif risk.get("sizing") == "fixed_capital":
                    q = int(float(risk.get("fixed_capital") or 20000) // entry)
                else:
                    risk_amt = eq * float(risk.get("risk_pct") or 1) / 100
                    q = int(risk_amt / risk_ps)
                q = min(q, int(cash // entry))
            if q <= 0:
                continue
            notional = q * entry
            if notional > cash:
                continue
            cash -= notional
            opens[key] = {
                "symbol": sym,
                "direction": direction,
                "entry": entry,
                "stop": stop,
                "target": target,
                "qty": q,
                "notional": notional,
                "signal_date": str(pd.Timestamp(sig["signal_date"]).date()),
                "entry_date": str(ts.date()),
                "hold_days": 0,
            }

        if ts == calendar[-1] or ts.weekday() == 4:
            record(ts)

    # force close
    if opens:
        last_ts = calendar[-1]
        for key, pos in list(opens.items()):
            df = frames.get(pos["symbol"])
            if df is None:
                continue
            hist = df.loc[df.index <= last_ts]
            if hist.empty:
                continue
            close = float(hist.iloc[-1]["close"])
            exit_ts = hist.index[-1]
            costs = _costs(pos["entry"], close, pos["qty"], commission, slippage)
            if pos["direction"] == "long":
                gross = (close - pos["entry"]) * pos["qty"]
            else:
                gross = (pos["entry"] - close) * pos["qty"]
            pnl = gross - costs
            cash += pos["notional"] + pnl
            trades.append(SBTrade(
                symbol=pos["symbol"],
                direction=pos["direction"],
                signal_date=pos["signal_date"],
                entry_date=pos["entry_date"],
                exit_date=str(exit_ts.date()),
                entry_price=round(pos["entry"], 2),
                exit_price=round(close, 2),
                stop_loss=round(pos["stop"], 2),
                target=round(pos["target"], 2),
                quantity=pos["qty"],
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / pos["notional"] * 100, 2) if pos["notional"] else 0,
                days_held=pos["hold_days"],
                exit_reason="eod_force",
            ))
        opens.clear()
        record(last_ts)

    trades.sort(key=lambda t: (t.entry_date, t.symbol))
    final = cash
    result.trades = trades
    result.total_trades = len(trades)
    result.final_capital = round(final, 2)
    result.net_profit = round(final - capital, 2)
    result.total_return_pct = round((final - capital) / capital * 100, 2) if capital else 0
    result.cagr_pct = _cagr(capital, final, start_date, end_date)
    result.equity_curve = equity_curve
    result.drawdown_curve = drawdown_curve
    result.max_drawdown_pct = round(max_dd, 2)
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns, result.yearly_returns = _monthly_yearly(trades, capital)
    result.stock_stats = _stock_stats(trades)
    result.benchmark = _benchmark(start_date, end_date, capital)

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = round(len(wins) / len(trades) * 100, 2)
        result.avg_win = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0
        result.avg_loss = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1e-9
        result.profit_factor = round(gp / gl, 2)
        result.expectancy = round(sum(t.pnl for t in trades) / len(trades), 2)
        result.avg_hold_days = round(sum(t.days_held for t in trades) / len(trades), 1)

    if daily_returns:
        arr = np.array(daily_returns, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr)) or 1e-9
        result.sharpe = round(mu / sigma * math.sqrt(252), 2)
        downside = arr[arr < 0]
        dsigma = float(np.std(downside)) if len(downside) else 1e-9
        result.sortino = round(mu / (dsigma or 1e-9) * math.sqrt(252), 2)
        if result.max_drawdown_pct > 0:
            result.calmar = round(result.cagr_pct / result.max_drawdown_pct, 2)

    return result


def result_to_dict(r: SBBacktestResult) -> dict:
    d = asdict(r)
    return d
