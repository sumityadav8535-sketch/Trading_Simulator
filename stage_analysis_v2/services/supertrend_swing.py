"""
Supertrend pullback swing backtester.

Signal on bar T close (no look-ahead), enter T+1 open.
Shared cash, risk-based sizing, optional per-name position cap.
Returns StageV2BacktestResult so the existing backtest UI/charts work.
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
    maybe_raise_stop,
    stop_exit_reason,
)
from stage_analysis_v2.services.strategy_catalog import (
    DEFAULT_ST_COOLDOWN_DAYS,
    DEFAULT_ST_MAX_HOLD_DAYS,
    DEFAULT_ST_MAX_POS_PCT,
    DEFAULT_ST_RISK_PCT,
    ST_MAX_NEW_PER_DAY,
    ST_MAX_STOP_PCT,
    ST_MIN_STOP_PCT,
    ST_MULTIPLIER,
    ST_PERIOD,
    ST_PULLBACK_TOL,
    ST_REQUIRE_FIRST_TOUCH,
    ST_STOP_ATR_MULT,
    ST_TRAIL_ATR_MULT,
    STRATEGY_LABELS,
    STRATEGY_ST_QUALITY,
    st_filter_pack,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def _ema(s: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(s, dtype=float)
    if len(s) == 0:
        return out
    a = 2.0 / (n + 1.0)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = a * s[i] + (1.0 - a) * out[i - 1]
    return out


def _rma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    if len(s) < n:
        return out
    out[n - 1] = s[:n].mean()
    a = 1.0 / n
    for i in range(n, len(s)):
        out[i] = out[i - 1] * (1.0 - a) + s[i] * a
    return out


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_g = _rma(gain, n)
    avg_l = _rma(loss, n)
    rs = avg_g / np.where(avg_l == 0, np.nan, avg_l)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.empty_like(c)
    prev[0] = c[0]
    prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return _rma(tr, n)


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(h, l, c, n)
    plus_di = 100.0 * _rma(plus_dm, n) / atr
    minus_di = 100.0 * _rma(minus_dm, n) / atr
    denom = plus_di + minus_di
    dx = np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom) * 100.0
    return _rma(np.nan_to_num(dx, nan=0.0), n)


def supertrend_np(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic Supertrend. direction = 1 bull, -1 bear."""
    n = len(c)
    st = np.full(n, np.nan)
    direction = np.zeros(n)
    if n < period + 2:
        return st, direction

    atr = _atr(h, l, c, period)
    hl2 = (h + l) * 0.5
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    start = period
    for i in range(start + 1, n):
        if np.isnan(basic_ub[i]) or np.isnan(final_ub[i - 1]):
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
            continue
        if basic_ub[i] < final_ub[i - 1] or c[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]
        if basic_lb[i] > final_lb[i - 1] or c[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

    st[start] = final_ub[start]
    direction[start] = -1.0
    for i in range(start + 1, n):
        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]
            continue
        if direction[i - 1] <= 0:
            if c[i] > final_ub[i]:
                st[i] = final_lb[i]
                direction[i] = 1.0
            else:
                st[i] = final_ub[i]
                direction[i] = -1.0
        else:
            if c[i] < final_lb[i]:
                st[i] = final_ub[i]
                direction[i] = -1.0
            else:
                st[i] = final_lb[i]
                direction[i] = 1.0
    return st, direction


class _STPack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c", "st", "st_dir",
        "ema20", "ema50", "ema200", "rsi", "adx", "ret20", "atr",
        "close_loc", "bull", "st_age", "first_touch",
    )

    def __init__(self, symbol: str, df: pd.DataFrame, period: int, mult: float):
        self.symbol = symbol
        self.index = df.index
        self.loc = {ts: i for i, ts in enumerate(df.index)}
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        self.o, self.h, self.l, self.c = o, h, l, c
        self.ema20 = _ema(c, 20)
        self.ema50 = _ema(c, 50)
        self.ema200 = _ema(c, 200)
        self.rsi = _rsi(c, 14)
        self.adx = _adx(h, l, c, 14)
        self.atr = _atr(h, l, c, 14)
        self.ret20 = pd.Series(c, index=df.index).pct_change(20).to_numpy()
        self.st, self.st_dir = supertrend_np(h, l, c, period, mult)
        rng = np.maximum(h - l, 1e-9)
        self.close_loc = (c - l) / rng
        self.bull = c > o
        age = np.zeros(len(self.st_dir))
        run = 0
        for i, d in enumerate(self.st_dir):
            if d > 0:
                run += 1
            else:
                run = 0
            age[i] = run
        self.st_age = age
        first = np.zeros(len(self.st_dir), dtype=bool)
        seen = False
        for i in range(len(self.st_dir)):
            if self.st_dir[i] <= 0:
                seen = False
                continue
            st_line = self.st[i]
            tagged = (
                not np.isnan(st_line)
                and l[i] <= st_line * (1.0 + ST_PULLBACK_TOL)
                and c[i] > st_line
            )
            if tagged and not seen:
                first[i] = True
                seen = True
        self.first_touch = first


def _passes_pack(pack: str, ema_trend: bool, rsi_healthy: bool, adx20: bool) -> bool:
    if pack == "trend_rsi":
        return ema_trend and rsi_healthy
    # quality (default)
    return ema_trend and rsi_healthy and adx20


def _quality_score(ema_trend: bool, rsi_healthy: bool, adx20: bool) -> int:
    return (40 if ema_trend else 0) + (30 if rsi_healthy else 0) + (30 if adx20 else 0)


def run_supertrend_swing_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    strategy_id: str = STRATEGY_ST_QUALITY,
    risk_pct: float | None = None,
    max_hold_days: int | None = None,
    cooldown_days: int | None = None,
    max_pos_pct: float | None = None,
    entry_filters: EntryFilters | None = None,
    st_period: int = ST_PERIOD,
    st_mult: float = ST_MULTIPLIER,
) -> StageV2BacktestResult:
    """
    Supertrend(14,3) pullback swing on the given universe.

    filter packs:
      quality   — first ST pullback + EMA trend + RSI 45–70 + ADX ≥ 20
      trend_rsi — first ST pullback + EMA trend + RSI 45–70

    Stop is under the pullback low (not the ST line) so wicks don't scratch
    the position. Trail is Supertrend minus 0.5 ATR.
    """
    pack = st_filter_pack(strategy_id)
    if risk_pct is None:
        risk_pct = DEFAULT_ST_RISK_PCT if pack == "quality" else 4.0
    risk_pct = min(50.0, max(0.1, float(risk_pct)))
    if max_hold_days is None:
        max_hold_days = DEFAULT_ST_MAX_HOLD_DAYS
    max_hold_days = min(500, max(1, int(max_hold_days)))
    if cooldown_days is None:
        cooldown_days = DEFAULT_ST_COOLDOWN_DAYS
    cooldown_days = min(365, max(0, int(cooldown_days)))
    if max_pos_pct is None:
        max_pos_pct = DEFAULT_ST_MAX_POS_PCT
    max_pos_pct = min(100.0, max(5.0, float(max_pos_pct)))
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]
    filters = entry_filters or EntryFilters()

    filter_label = {
        "quality": "First ST tag + EMA trend + RSI 45–70 + ADX≥20 · stop under pullback low",
        "trend_rsi": "First ST tag + close>EMA50>EMA200 + RSI 45–70 · stop under pullback low",
    }.get(pack, pack)

    result = StageV2BacktestResult(
        strategy_name=STRATEGY_LABELS.get(strategy_id, strategy_id),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=0,
        market_filter=False,
        exit_mode="st_trail",
        exit_mode_label="Trail Supertrend − 0.5 ATR",
        tech_filter=pack,
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

    packs: dict[str, _STPack] = {}
    for sym, df in frames.items():
        packs[sym] = _STPack(sym, df, st_period, st_mult)

    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret20 = (
        nifty_df["close"].pct_change(20)
        if not nifty_df.empty
        else pd.Series(dtype=float)
    )

    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n_sig = 0
    for sym, s in packs.items():
        daily = frames[sym]
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            if not (s.st_dir[i] > 0 and s.st_dir[i - 1] > 0):
                continue
            st_line = s.st[i]
            if np.isnan(st_line) or st_line <= 0:
                continue
            if not (s.l[i] <= st_line * (1.0 + ST_PULLBACK_TOL) and s.c[i] > st_line):
                continue
            if ST_REQUIRE_FIRST_TOUCH and not bool(s.first_touch[i]):
                continue
            e50, e200 = s.ema50[i], s.ema200[i]
            if np.isnan(e50) or np.isnan(e200):
                continue
            ema_trend = s.c[i] > e50 > e200
            rsi = s.rsi[i]
            rsi_healthy = (not np.isnan(rsi)) and 45.0 <= rsi <= 70.0
            adx = s.adx[i]
            adx20 = (not np.isnan(adx)) and adx >= 20.0
            if not _passes_pack(pack, ema_trend, rsi_healthy, adx20):
                continue
            if not _passes_entry_filters(daily, ts, filters):
                continue
            r20 = float(s.ret20[i]) if not np.isnan(s.ret20[i]) else -9.0
            n_r = float(nifty_ret20.loc[ts]) if ts in nifty_ret20.index and pd.notna(nifty_ret20.loc[ts]) else 0.0
            rs = min(99.0, max(1.0, 50.0 + (r20 - n_r) * 200.0))
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            signals_by_day[s.index[i + 1]].append({
                "symbol": sym,
                "sig_ts": ts,
                "entry_ts": s.index[i + 1],
                "stop_st": float(st_line),
                "sig_low": float(s.l[i]),
                "atr": atr,
                "score": r20,
                "quality": _quality_score(ema_trend, rsi_healthy, adx20),
                "rs": round(rs, 1),
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
            st_line = s.st[i]
            atr_now = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0

            exit_price = None
            exit_reason = ""
            # Hit the stop carried in from the previous bar first. Raising the
            # trail from today's Supertrend and then testing today's low is
            # look-ahead and marks winners as stop_loss.
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, stop_exit_reason(pos)
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif s.st_dir[i] <= 0:
                exit_price, exit_reason = close, "st_flip"

            if exit_price is None:
                if not np.isnan(st_line):
                    trail = float(st_line) - ST_TRAIL_ATR_MULT * atr_now
                    maybe_raise_stop(pos, trail, close)
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
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
                if taken >= ST_MAX_NEW_PER_DAY:
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
                sig_low = float(sig.get("sig_low") or sig["stop_st"])
                stop = min(sig_low, float(sig["stop_st"])) - ST_STOP_ATR_MULT * atr
                if stop <= 0 or stop >= entry:
                    fallback = entry - (1.5 * atr if atr > 0 else entry * 0.04)
                    stop = fallback if 0 < fallback < entry else 0.0
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
