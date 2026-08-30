"""
Dual Supertrend first-pullback + Minervini swing (Nifty 200, ~5y researched pack).

Entry: first Supertrend tag on ST(14,3) OR ST(21,3), signal on close T, fill T+1 open.
Filter: Minervini trend template and Nifty close > EMA50.
Stop: min(pullback low, ST) − 0.25 ATR, skip if <1.8% or >12%.
Exit: Supertrend(21, 4) flips bear (close), initial stop, or max hold.
Size: 2% of equity at risk, max 4 names, 80% per-name cap. No leverage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from stage_analysis_v2.services.backtester import (
    EntryFilters,
    StageV2BacktestResult,
    StageV2Trade,
    _OpenPos,
    attach_signal_log_from_raw,
    _close_trade,
    _compute_monthly_returns,
    _equity,
    _invested_total,
    _passes_entry_filters,
    _preload_frames,
    stop_exit_reason,
)
from stage_analysis_v2.services.strategy_catalog import (
    DEFAULT_UNION_COOLDOWN_DAYS,
    DEFAULT_UNION_MAX_HOLD_DAYS,
    DEFAULT_UNION_MAX_POS_PCT,
    DEFAULT_UNION_RISK_PCT,
    ST_MAX_STOP_PCT,
    ST_MIN_STOP_PCT,
    ST_PULLBACK_TOL,
    ST_STOP_ATR_MULT,
    STRATEGY_LABELS,
    STRATEGY_ST_UNION,
    UNION_ENTRY_ST,
    UNION_EXIT_MULT,
    UNION_EXIT_PERIOD,
    UNION_MAX_NEW_PER_DAY,
    UNION_MAX_OPEN,
)
from stage_analysis_v2.services.supertrend_swing import _atr, _ema, supertrend_np
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    if len(s) < n:
        return out
    csum = np.cumsum(s)
    out[n - 1] = csum[n - 1] / n
    out[n:] = (csum[n:] - csum[:-n]) / n
    return out


def _roll_max(s: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(s).rolling(n, min_periods=n).max().to_numpy()


def _roll_min(s: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(s).rolling(n, min_periods=n).min().to_numpy()


def _first_touch(low: np.ndarray, close: np.ndarray, st: np.ndarray, direction: np.ndarray) -> np.ndarray:
    first = np.zeros(len(close), dtype=bool)
    seen = False
    for i in range(len(close)):
        if direction[i] <= 0:
            seen = False
            continue
        line = st[i]
        tagged = (
            not np.isnan(line)
            and low[i] <= line * (1.0 + ST_PULLBACK_TOL)
            and close[i] > line
        )
        if tagged and not seen:
            first[i] = True
            seen = True
    return first


def is_minervini(c: float, sma50: float, sma150: float, sma200: float, sma200_prev: float,
                 low252: float, high252: float) -> bool:
    if any(np.isnan(x) for x in (c, sma50, sma150, sma200, sma200_prev, low252, high252)):
        return False
    if not (c > sma50 > sma150 > sma200):
        return False
    if sma200 <= sma200_prev:
        return False
    if low252 <= 0 or high252 <= 0:
        return False
    if c < 1.25 * low252:
        return False
    if c < 0.75 * high252:
        return False
    return True


class _UnionPack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c",
        "ema50", "sma50", "sma150", "sma200", "atr", "ret63",
        "high252", "low252", "st", "st_dir", "first", "exit_dir",
    )

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.index = df.index
        self.loc = {ts: i for i, ts in enumerate(df.index)}
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        self.o, self.h, self.l, self.c = o, h, l, c
        self.ema50 = _ema(c, 50)
        self.sma50 = _sma(c, 50)
        self.sma150 = _sma(c, 150)
        self.sma200 = _sma(c, 200)
        self.atr = _atr(h, l, c, 14)
        self.ret63 = pd.Series(c, index=df.index).pct_change(63).to_numpy()
        self.high252 = _roll_max(h, 252)
        self.low252 = _roll_min(l, 252)
        self.st = {}
        self.st_dir = {}
        self.first = {}
        for period, mult in UNION_ENTRY_ST:
            st, d = supertrend_np(h, l, c, period, mult)
            self.st[(period, mult)] = st
            self.st_dir[(period, mult)] = d
            self.first[(period, mult)] = _first_touch(l, c, st, d)
        _, exit_d = supertrend_np(h, l, c, UNION_EXIT_PERIOD, UNION_EXIT_MULT)
        self.exit_dir = exit_d


def run_st_union_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    strategy_id: str = STRATEGY_ST_UNION,
    risk_pct: float | None = None,
    max_hold_days: int | None = None,
    cooldown_days: int | None = None,
    max_pos_pct: float | None = None,
    entry_filters: EntryFilters | None = None,
    cost_pct: float = 0.0,
) -> StageV2BacktestResult:
    if risk_pct is None:
        risk_pct = DEFAULT_UNION_RISK_PCT
    risk_pct = min(50.0, max(0.1, float(risk_pct)))
    if max_hold_days is None:
        max_hold_days = DEFAULT_UNION_MAX_HOLD_DAYS
    max_hold_days = min(500, max(1, int(max_hold_days)))
    if cooldown_days is None:
        cooldown_days = DEFAULT_UNION_COOLDOWN_DAYS
    cooldown_days = min(365, max(0, int(cooldown_days)))
    if max_pos_pct is None:
        max_pos_pct = DEFAULT_UNION_MAX_POS_PCT
    max_pos_pct = min(100.0, max(5.0, float(max_pos_pct)))
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365 * 5))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]
    filters = entry_filters or EntryFilters()

    filter_label = (
        "First ST(14,3) or ST(21,3) pullback · Minervini template · Nifty > EMA50 · "
        "exit ST(21,4) flip"
    )
    result = StageV2BacktestResult(
        strategy_name=STRATEGY_LABELS.get(strategy_id, strategy_id),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=0,
        market_filter=True,
        exit_mode="st_flip",
        exit_mode_label="Supertrend(21, 4) flip",
        tech_filter="minervini_mkt",
        tech_filter_label=filter_label,
        entry_stage=0,
        entry_on="pullback",
        entry_filters_label=filters.active_summary(),
        target_rr=0.0,
        max_hold_days=max_hold_days,
        stop_ma_mult=1.0,
        trail_ma_mult=1.0,
        risk_pct=risk_pct,
        cooldown_days=cooldown_days,
        strategy_id=strategy_id,
        max_pos_pct=max_pos_pct,
    )

    frames = _preload_frames(symbols)
    result.stocks_scanned = len(frames)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    packs: dict[str, _UnionPack] = {}
    for sym, df in frames.items():
        packs[sym] = _UnionPack(sym, df)

    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ema50 = (
        _ema(nifty_df["close"].to_numpy(dtype=float), 50)
        if not nifty_df.empty else np.array([])
    )
    nifty_close = nifty_df["close"] if not nifty_df.empty else pd.Series(dtype=float)
    nifty_loc = {ts: i for i, ts in enumerate(nifty_df.index)} if not nifty_df.empty else {}

    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n_sig = 0
    for sym, s in packs.items():
        daily = frames[sym]
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            ni = nifty_loc.get(ts)
            if ni is None or ni >= len(nifty_ema50):
                continue
            if nifty_close.iloc[ni] <= nifty_ema50[ni]:
                continue
            if i < 222:
                continue
            if not is_minervini(
                s.c[i], s.sma50[i], s.sma150[i], s.sma200[i], s.sma200[i - 22],
                s.low252[i], s.high252[i],
            ):
                continue
            if not _passes_entry_filters(daily, ts, filters):
                continue
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            entry_ts = s.index[i + 1]
            for key in UNION_ENTRY_ST:
                d = s.st_dir[key]
                st_line = s.st[key][i]
                if not (d[i] > 0 and d[i - 1] > 0):
                    continue
                if np.isnan(st_line) or st_line <= 0:
                    continue
                if not (s.l[i] <= st_line * (1.0 + ST_PULLBACK_TOL) and s.c[i] > st_line):
                    continue
                if not bool(s.first[key][i]):
                    continue
                stop_ref = min(float(s.l[i]), float(st_line))
                signals_by_day[entry_ts].append({
                    "symbol": sym,
                    "sig_ts": ts,
                    "entry_ts": entry_ts,
                    "stop_st": stop_ref,
                    "sig_low": float(s.l[i]),
                    "atr": atr,
                    "score": r63,
                    "quality": 80,
                    "rs": round(min(99.0, max(1.0, 50.0 + r63 * 100.0)), 1),
                })
                n_sig += 1

    result.stage2_entries = n_sig
    result.total_signals = n_sig

    calendar = sorted({
        ts
        for df in frames.values()
        for ts in df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist()
    })
    if not calendar:
        result.final_cash = capital
        result.equity_curve = [{"date": str(start_date), "equity": capital}]
        return result

    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve: list[dict] = [{"date": str(start_date), "equity": capital}]
    peak_parallel = 0
    skipped_cash = 0
    last_curve_date: Optional[date] = None

    def _record_curve(ts: pd.Timestamp, force: bool = False) -> None:
        nonlocal last_curve_date
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if not force and last_curve_date is not None and (d - last_curve_date).days < 5:
            if d.weekday() != 4:
                return
        eq = _equity(cash, opens)
        equity_curve.append({"date": str(d), "equity": round(eq, 2)})
        last_curve_date = d

    for ts in calendar:
        closed_today: list[str] = []
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            pos.hold_days += 1
            close = float(s.c[i])
            low = float(s.l[i])
            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, stop_exit_reason(pos)
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif s.exit_dir[i] <= 0:
                exit_price, exit_reason = close, "st_flip"
            if exit_price is None:
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            if cost_pct:
                trade.pnl -= (pos.entry_price + exit_price) * pos.qty * cost_pct / 100.0
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)
        for sym in closed_today:
            opens.pop(sym, None)
        if closed_today:
            _record_curve(ts, force=True)

        day_sigs = signals_by_day.get(ts, [])
        if day_sigs:
            day_sigs = sorted(day_sigs, key=lambda x: x["score"], reverse=True)
            taken = 0
            for sig in day_sigs:
                if taken >= UNION_MAX_NEW_PER_DAY or len(opens) >= UNION_MAX_OPEN:
                    break
                sym = sig["symbol"]
                if sym in opens:
                    continue
                s = packs.get(sym)
                if s is None:
                    continue
                prev = last_exit.get(sym)
                if cooldown_days > 0 and prev is not None and (ts - prev).days < cooldown_days:
                    continue
                i = s.loc.get(ts)
                if i is None:
                    continue
                entry = float(s.o[i])
                if entry <= 0:
                    continue
                atr = float(sig.get("atr") or 0.0)
                stop = float(sig.get("stop_st") or 0.0)
                if stop <= 0 or stop >= entry:
                    stop = entry - (1.5 * atr if atr > 0 else entry * 0.04)
                # Keep the structural pullback stop; only tighten if it is inside 0.25 ATR.
                if atr > 0:
                    stop = min(stop, entry - ST_STOP_ATR_MULT * atr)
                if stop <= 0 or stop >= entry:
                    continue
                risk = entry - stop
                stop_pct = risk / entry
                if stop_pct > ST_MAX_STOP_PCT or stop_pct < ST_MIN_STOP_PCT:
                    continue
                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    skipped_cash += 1
                    continue
                ps = calculate_position_size(equity_now, risk_pct, entry, stop)
                max_notional = equity_now * (max_pos_pct / 100.0)
                qty_cash = int(cash // entry) if entry else 0
                qty_cap = int(max_notional // entry) if entry else 0
                qty = min(int(ps.quantity), qty_cash, qty_cap)
                if qty <= 0:
                    skipped_cash += 1
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    skipped_cash += 1
                    continue
                cash -= notional
                invested_after = _invested_total(opens) + notional
                parallel = len(opens) + 1
                peak_parallel = max(peak_parallel, parallel)
                pos = _OpenPos(
                    symbol=sym,
                    entry_date=ts,
                    signal_date=sig["sig_ts"],
                    entry_price=entry,
                    stop=stop,
                    target=0.0,
                    qty=qty,
                    notional=notional,
                    quality_score=int(sig["quality"]),
                    rs_rating=float(sig["rs"]),
                    weekly_stage=0,
                    hold_days=0,
                    capital_invested=notional,
                    cash_available=cash,
                    total_invested=invested_after,
                    parallel_open=parallel,
                    equity_at_entry=cash + invested_after,
                    entry_stop=stop,
                )
                opens[sym] = pos
                taken += 1
                if s.l[i] <= stop:
                    trade = _close_trade(
                        pos, exit_price=stop, exit_ts=ts,
                        exit_reason="stop_loss", days_held=0,
                    )
                    if cost_pct:
                        trade.pnl -= (entry + stop) * qty * cost_pct / 100.0
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
            _record_curve(ts, force=True)

        if ts == calendar[-1] or ts.weekday() == 4:
            _record_curve(ts, force=(ts == calendar[-1]))

    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(last_ts, len(s.c) - 1)
            close = float(s.c[i])
            trade = _close_trade(
                pos, exit_price=close, exit_ts=last_ts,
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            if cost_pct:
                trade.pnl -= (pos.entry_price + close) * pos.qty * cost_pct / 100.0
            cash += pos.notional + trade.pnl
            trades.append(trade)
        opens.clear()
        _record_curve(last_ts, force=True)

    trades.sort(key=lambda t: (t.entry_date, t.symbol))
    result.trades = trades
    result.total_trades = len(trades)
    result.equity_curve = equity_curve
    result.peak_parallel = peak_parallel
    result.signals_skipped_cash = skipped_cash
    result.final_cash = round(cash, 2)
    raw_signals = []
    for day_ts, day_list in signals_by_day.items():
        for sig in day_list:
            raw_signals.append({
                "symbol": sig["symbol"],
                "signal_date": str(pd.Timestamp(sig["sig_ts"]).date()),
                "entry_date": str(pd.Timestamp(sig.get("entry_ts") or day_ts).date()),
                "stop_loss": round(float(sig.get("sig_low") or sig.get("stop_st") or 0), 2),
                "target": 0.0,
                "quality_score": int(sig.get("quality") or 0),
                "rs_rating": float(sig.get("rs") or 0),
            })
    attach_signal_log_from_raw(result, raw_signals)
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1e-9
        result.win_rate = round(len(wins) / len(trades) * 100, 2)
        result.profit_factor = round(gp / gl, 2)
        result.avg_rr = round(sum(t.rr_achieved for t in trades) / len(trades), 2)
        result.avg_hold_days = round(sum(t.days_held for t in trades) / len(trades), 1)
        result.expectancy_r = result.avg_rr
    result.total_return_pct = round((cash - capital) / capital * 100, 2) if capital else 0.0
    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        if peak:
            max_dd = max(max_dd, (peak - e) / peak * 100)
    result.max_drawdown_pct = round(max_dd, 2)
    result.exit_breakdown = dict(Counter(t.exit_reason for t in trades))
    result.monthly_returns = _compute_monthly_returns(trades, capital)
    return result
