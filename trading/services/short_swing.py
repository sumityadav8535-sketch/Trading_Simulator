"""
RS Pullback Swing — 3 to 15 day Nifty 200 swing.

Buy the bounce off EMA20 in a Qullamaggie-style leader (tight range,
3-month relative strength, ATR alive). Stop under the pullback, trail
a 2.5 ATR chandelier, take 1.5R or time-stop at 15 sessions.

Signal on bar T close. Enter T+1 open. Shared cash, no leverage.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings

from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

logger = logging.getLogger(__name__)

STRATEGY_NAME = "RS Pullback Swing"
RESULTS_NAME = "short_swing_results.json"
DEFAULT_CAPITAL = 1_000_000.0

F_TREND = 1 << 0
F_STACK = 1 << 1
F_RS63 = 1 << 2
F_ADX20 = 1 << 3
F_RSI_OK = 1 << 4
F_VOL = 1 << 5
F_MKT20 = 1 << 6
F_NOT_EXT = 1 << 8
F_NEAR52 = 1 << 9
F_QULLA = 1 << 11
F_ST_BULL = 1 << 12
F_FIRST_TOUCH = 1 << 13
F_FRESH = 1 << 14

WINDOW_YEARS = (1, 2, 3, 4, 5)


@dataclass
class SwingParams:
    name: str = "Leaders Pullback"
    entry: str = "ema20pb"
    filter_name: str = "qulla_rs"
    flags: int = F_QULLA | F_RS63
    exit_mode: str = "chandelier"
    max_hold: int = 15
    risk_pct: float = 8.0
    max_open: int = 4
    max_new: int = 3
    max_pos_pct: float = 0.70
    target_rr: float = 1.5
    cooldown: int = 5
    trail_atr: float = 2.5
    min_stop_pct: float = 0.012
    max_stop_pct: float = 0.12


LEADERS = SwingParams(
    filter_name="qulla_rs",
    flags=F_QULLA | F_RS63,
)
BALANCED = SwingParams(
    name="Balanced 3-name",
    filter_name="qulla_rs",
    flags=F_QULLA | F_RS63,
    max_open=3,
    max_new=3,
    risk_pct=8.0,
    max_pos_pct=0.70,
)
DEFAULT_PACKS = (LEADERS, BALANCED)
DEFAULT_PARAMS = LEADERS


def _ema(s: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(s, dtype=float)
    if len(s) == 0:
        return out
    a = 2.0 / (n + 1.0)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = a * s[i] + (1.0 - a) * out[i - 1]
    return out


def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    if len(s) < n:
        return out
    csum = np.cumsum(s)
    out[n - 1] = csum[n - 1] / n
    out[n:] = (csum[n:] - csum[:-n]) / n
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


def _adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(h, l, c, n)
    plus_di = 100.0 * _rma(plus_dm, n) / atr
    minus_di = 100.0 * _rma(minus_dm, n) / atr
    denom = plus_di + minus_di
    dx = np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom) * 100.0
    return _rma(np.nan_to_num(dx, nan=0.0), n), plus_di, minus_di


def _roll_max(s, n):
    return pd.Series(s).rolling(n, min_periods=n).max().to_numpy()


def _roll_min(s, n):
    return pd.Series(s).rolling(n, min_periods=n).min().to_numpy()


def supertrend_np(h, l, c, period=14, multiplier=3.0):
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


class BarPack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c", "v",
        "ema20", "ema50", "ema200", "rsi", "atr", "adx",
        "high10", "low10", "high252", "ret63", "ret126",
        "st", "st_dir", "adr10", "range10", "vol_sma",
    )

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.index = df.index
        self.loc = {ts: i for i, ts in enumerate(df.index)}
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.ema20 = _ema(c, 20)
        self.ema50 = _ema(c, 50)
        self.ema200 = _ema(c, 200)
        self.rsi = _rsi(c, 14)
        self.atr = _atr(h, l, c, 14)
        self.adx, _, _ = _adx(h, l, c, 14)
        self.vol_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
        self.high10 = _roll_max(h, 10)
        self.low10 = _roll_min(l, 10)
        self.high252 = _roll_max(h, 252)
        cs = pd.Series(c)
        self.ret63 = cs.pct_change(63).to_numpy()
        self.ret126 = cs.pct_change(126).to_numpy()
        self.st, self.st_dir = supertrend_np(h, l, c, 14, 3.0)
        rng_pct = (h - l) / np.maximum(c, 1e-9)
        self.adr10 = pd.Series(rng_pct).rolling(10, min_periods=10).mean().to_numpy()
        self.range10 = (self.high10 - self.low10) / np.maximum(c, 1e-9)


_PACK_CACHE: tuple[dict[str, BarPack], BarPack | None, list] | None = None


def preload_packs(force: bool = False):
    global _PACK_CACHE
    if _PACK_CACHE is not None and not force:
        return _PACK_CACHE
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    packs: dict[str, BarPack] = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 120:
            continue
        packs[sym] = BarPack(sym, df)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty = BarPack("NIFTY50", nifty_df) if not nifty_df.empty else None
    cal_start = pd.Timestamp(date.today() - timedelta(days=365 * 5 + 20))
    cal_end = pd.Timestamp(date.today())
    calendar = sorted({
        ts
        for p in packs.values()
        for ts in p.index[(p.index >= cal_start) & (p.index <= cal_end)].tolist()
    })
    _PACK_CACHE = (packs, nifty, calendar)
    return _PACK_CACHE


def flags_at(s: BarPack, i: int, nifty: BarPack | None) -> int:
    c = s.c[i]
    e20, e50, e200 = s.ema20[i], s.ema50[i], s.ema200[i]
    if np.isnan(e50) or np.isnan(e200):
        return 0
    f = 0
    if c > e50 > e200:
        f |= F_TREND
    if (not np.isnan(e20)) and e20 > e50:
        f |= F_STACK
    rsi = s.rsi[i]
    if not np.isnan(rsi) and 40.0 <= rsi <= 65.0:
        f |= F_RSI_OK
    vs = s.vol_sma[i]
    if vs and not np.isnan(vs) and vs > 0 and s.v[i] >= 1.3 * vs:
        f |= F_VOL
    if not np.isnan(s.adx[i]) and s.adx[i] >= 20:
        f |= F_ADX20
    if (not np.isnan(e20)) and e20 > 0 and (c - e20) / e20 <= 0.08:
        f |= F_NOT_EXT
    if not np.isnan(s.high252[i]) and s.high252[i] > 0 and c >= 0.85 * s.high252[i]:
        f |= F_NEAR52
    if s.st_dir[i] > 0:
        f |= F_ST_BULL
    first = True
    for j in range(1, 6):
        if i - j < 1:
            break
        if not np.isnan(s.ema20[i - j]) and s.l[i - j] <= s.ema20[i - j] * 1.006:
            first = False
            break
    if first:
        f |= F_FIRST_TOUCH
    adr = s.adr10[i]
    r63 = s.ret63[i]
    r126 = s.ret126[i]
    tight = s.range10[i]
    if (
        not np.isnan(adr) and adr >= 0.022
        and ((not np.isnan(r63) and r63 >= 0.15) or (not np.isnan(r126) and r126 >= 0.30))
        and not np.isnan(tight) and tight <= 0.20
        and c > e50
    ):
        f |= F_QULLA
    if nifty is not None:
        ni = nifty.loc.get(s.index[i])
        if ni is not None:
            nr63 = nifty.ret63[ni]
            if not np.isnan(s.ret63[i]) and not np.isnan(nr63) and s.ret63[i] > nr63:
                f |= F_RS63
            if not np.isnan(nifty.ema20[ni]) and nifty.c[ni] > nifty.ema20[ni]:
                f |= F_MKT20
    return f


def is_ema20_pullback(s: BarPack, i: int) -> bool:
    if np.isnan(s.ema20[i]) or np.isnan(s.ema20[i - 1]):
        return False
    return (
        s.l[i] <= s.ema20[i] * 1.006
        and s.c[i] > s.ema20[i]
        and s.c[i] > s.o[i]
        and s.c[i - 1] <= s.ema20[i - 1] * 1.012
    )


def collect_signals(
    packs: dict[str, BarPack],
    nifty: BarPack | None,
    start_ts,
    end_ts,
    *,
    entry: str = "ema20pb",
) -> list[dict]:
    out: list[dict] = []
    for sym, s in packs.items():
        n = len(s.c)
        for i in range(80, n):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            fire = False
            stop_ref = 0.0
            if entry == "ema20pb":
                fire = is_ema20_pullback(s, i)
                stop_ref = float(min(s.l[i], s.low10[i] if not np.isnan(s.low10[i]) else s.l[i]))
            elif entry == "stpb":
                st_line = s.st[i]
                fire = (
                    s.st_dir[i] > 0 and s.st_dir[i - 1] > 0
                    and not np.isnan(st_line)
                    and s.l[i] <= st_line * 1.008
                    and s.c[i] > st_line
                )
                stop_ref = float(min(s.l[i], st_line)) if not np.isnan(st_line) else float(s.l[i])
            elif entry == "donch10":
                h10p = s.high10[i - 1]
                fire = (not np.isnan(h10p)) and s.c[i] >= h10p and s.c[i - 1] < h10p
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            if not fire:
                continue
            fl = flags_at(s, i, nifty)
            prev_same = False
            if entry == "ema20pb" and i >= 1:
                prev_same = is_ema20_pullback(s, i - 1)
            if not prev_same:
                fl |= F_FRESH
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            entry_ts = s.index[i + 1] if i + 1 < n else ts + pd.Timedelta(days=1)
            out.append({
                "symbol": sym,
                "sig_i": i,
                "entry_ts": entry_ts,
                "sig_ts": ts,
                "flags": fl,
                "stop_ref": stop_ref,
                "atr": atr,
                "sig_low": float(s.l[i]),
                "score": r63,
                "close": float(s.c[i]),
                "ema20": float(s.ema20[i]) if not np.isnan(s.ema20[i]) else None,
                "rsi": float(s.rsi[i]) if not np.isnan(s.rsi[i]) else None,
                "adx": float(s.adx[i]) if not np.isnan(s.adx[i]) else None,
                "ret63": r63,
            })
    return out


def _py(v):
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return float(v)
    return v


def plan_trade(
    entry: float,
    stop_ref: float,
    sig_low: float,
    atr: float,
    params: SwingParams,
) -> dict[str, Any] | None:
    """Stop under the pullback, target = entry + R × risk. None if the fill would be skipped."""
    entry = float(entry)
    if entry <= 0:
        return None
    atr = float(atr or 0.0)
    raw = float(stop_ref or 0.0)
    if raw <= 0 or raw >= entry:
        raw = min(float(sig_low or 0.0), entry - (1.5 * atr if atr > 0 else entry * 0.04))
    stop = raw
    if atr > 0:
        stop = min(stop, entry - 0.25 * atr)
    if stop <= 0 or stop >= entry:
        return None
    risk = entry - stop
    spct = risk / entry
    if spct < params.min_stop_pct or spct > params.max_stop_pct:
        return None
    target = entry + risk * params.target_rr
    return {
        "entry": round(entry, 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "risk": round(float(risk), 2),
        "risk_pct": round(spct * 100.0, 1),
        "reward_pct": round((target / entry - 1.0) * 100.0, 1),
        "target_rr": float(params.target_rr),
    }


def simulate(
    packs: dict[str, BarPack],
    calendar: list,
    signals: list[dict],
    params: SwingParams,
    *,
    capital: float = DEFAULT_CAPITAL,
    flatten: bool = True,
    keep_trades: bool = True,
) -> dict[str, Any]:
    by_day: dict = defaultdict(list)
    need = int(params.flags)
    for sig in signals:
        if need and (sig["flags"] & need) != need:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(capital)
    opens: dict[str, dict] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    attempted: list[dict] = []
    skipped_cash = 0
    peak_eq = capital
    max_dd = 0.0
    peak_par = 0
    eq_pts: list[dict] = []

    def equity() -> float:
        marked = cash
        for p in opens.values():
            s = packs.get(p["symbol"])
            px = p["entry"]
            if s is not None:
                i = s.loc.get(eq_pts[-1]["ts"]) if eq_pts else None
                if i is not None:
                    px = float(s.c[i])
            marked += p["qty"] * px
        return cash + sum(p["notional"] for p in opens.values())

    def mark(ts) -> None:
        nonlocal peak_eq, max_dd
        eq = cash + sum(p["notional"] for p in opens.values())
        # mark-to-market
        mtm = cash
        for p in opens.values():
            s = packs.get(p["symbol"])
            px = p["entry"]
            if s is not None:
                i = s.loc.get(ts)
                if i is not None:
                    px = float(s.c[i])
            mtm += p["qty"] * px
        peak_eq = max(peak_eq, mtm)
        if peak_eq:
            max_dd = max(max_dd, (peak_eq - mtm) / peak_eq * 100.0)
        if (not eq_pts) or ts.weekday() == 4 or flatten:
            eq_pts.append({"date": str(ts.date()), "equity": round(mtm, 2), "ts": ts})

    for ts in calendar:
        closed = []
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            pos["hold"] += 1
            high, low, close = s.h[i], s.l[i], s.c[i]
            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= params.max_hold:
                exit_p, reason = close, "time"
            if exit_p is None:
                if params.exit_mode == "chandelier":
                    atr = s.atr[i]
                    if not np.isnan(atr):
                        trail = high - params.trail_atr * atr
                        if trail > pos["stop"] and trail < close:
                            pos["stop"] = float(trail)
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym,
                "entry": str(pos["entry_ts"].date()),
                "exit": str(ts.date()),
                "signal_date": str(pd.Timestamp(pos.get("signal_ts") or pos["entry_ts"]).date()),
                "entry_px": round(pos["entry"], 2),
                "exit_px": round(float(exit_p), 2),
                "stop": round(float(pos["stop"]), 2),
                "target": round(float(pos["target"]), 2),
                "pnl": round(float(pnl), 2),
                "pnl_pct": round((exit_p / pos["entry"] - 1.0) * 100.0, 2),
                "reason": reason,
                "hold": pos["hold"],
                "qty": pos["qty"],
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        taken = 0
        for sig in day_sigs:
            if taken >= params.max_new or len(opens) >= params.max_open:
                break
            sym = sig["symbol"]
            if sym in opens:
                continue
            s = packs.get(sym)
            if s is None:
                continue
            prev = last_exit.get(sym)
            if params.cooldown > 0 and prev is not None and (ts - prev).days < params.cooldown:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            entry = float(s.o[i])
            planned = plan_trade(
                entry,
                float(sig.get("stop_ref") or 0.0),
                float(sig.get("sig_low") or 0.0),
                float(sig.get("atr") or 0.0),
                params,
            )
            if planned is None:
                continue
            stop = float(planned["stop"])
            target = float(planned["target"])
            eq = cash + sum(p["notional"] for p in opens.values())
            if cash <= 0 or eq <= 0:
                continue
            ps = calculate_position_size(eq, params.risk_pct, entry, stop)
            qty = min(int(ps.quantity), int(cash // entry), int((eq * params.max_pos_pct) // entry))
            if qty <= 0 or qty * entry > cash + 1e-6:
                skipped_cash += 1
                attempted.append({
                    "symbol": sym,
                    "sig_ts": sig.get("sig_ts"),
                    "entry_ts": ts,
                    "close": sig.get("close"),
                    "stop_ref": stop,
                    "score": sig.get("score"),
                    "status": "skipped_cash",
                })
                continue
            notional = qty * entry
            cash -= notional
            opens[sym] = {
                "symbol": sym, "entry": entry, "stop": stop, "target": target,
                "qty": qty, "notional": notional, "hold": 0, "entry_ts": ts,
                "signal_ts": sig.get("sig_ts") or ts,
                "score": sig.get("score"),
            }
            peak_par = max(peak_par, len(opens))
            taken += 1
            attempted.append({
                "symbol": sym,
                "sig_ts": sig.get("sig_ts"),
                "entry_ts": ts,
                "close": sig.get("close"),
                "stop_ref": stop,
                "score": sig.get("score"),
                "status": "taken",
            })
            if s.l[i] <= stop:
                pnl = (stop - entry) * qty
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "entry": str(ts.date()), "exit": str(ts.date()),
                    "signal_date": str(pd.Timestamp(sig.get("sig_ts") or ts).date()),
                    "entry_px": round(entry, 2), "exit_px": round(float(stop), 2),
                    "stop": round(float(stop), 2), "target": round(float(target), 2),
                    "pnl": round(float(pnl), 2),
                    "pnl_pct": round((stop / entry - 1.0) * 100.0, 2),
                    "reason": "stop", "hold": 0, "qty": qty,
                })
                last_exit[sym] = ts
                opens.pop(sym, None)
        mark(ts)

    open_book = []
    if flatten and opens and calendar:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(last_ts, len(s.c) - 1)
            close = float(s.c[i])
            pnl = (close - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "entry": str(pos["entry_ts"].date()),
                "exit": str(last_ts.date()),
                "signal_date": str(pd.Timestamp(pos.get("signal_ts") or pos["entry_ts"]).date()),
                "entry_px": round(pos["entry"], 2), "exit_px": round(close, 2),
                "stop": round(float(pos["stop"]), 2),
                "target": round(float(pos["target"]), 2),
                "pnl": round(float(pnl), 2),
                "pnl_pct": round((close / pos["entry"] - 1.0) * 100.0, 2),
                "reason": "eod", "hold": pos["hold"], "qty": pos["qty"],
            })
        opens.clear()
        if eq_pts:
            eq_pts[-1]["equity"] = round(cash, 2)
    else:
        last_ts = calendar[-1] if calendar else None
        for sym, pos in opens.items():
            s = packs.get(sym)
            px = pos["entry"]
            if s is not None and last_ts is not None:
                i = s.loc.get(last_ts, len(s.c) - 1)
                px = float(s.c[i])
            open_book.append({
                "symbol": sym,
                "entry": str(pos["entry_ts"].date()),
                "entry_px": round(pos["entry"], 2),
                "last_px": round(px, 2),
                "pnl_pct": round((px / pos["entry"] - 1.0) * 100.0, 1),
                "hold": pos["hold"],
                "stop": round(pos["stop"], 2),
                "target": round(pos["target"], 2),
                "qty": pos["qty"],
            })

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    final = cash if flatten else (cash + sum(p["qty"] * p.get("entry", 0) for p in []))
    if eq_pts:
        final_eq = eq_pts[-1]["equity"]
    else:
        final_eq = cash
    sampled = eq_pts
    if len(sampled) > 400:
        step = max(1, len(sampled) // 260)
        sampled = sampled[::step]
        if sampled[-1] != eq_pts[-1]:
            sampled.append(eq_pts[-1])
    curve = [{"date": p["date"], "equity": p["equity"]} for p in sampled]

    start = calendar[0].date() if calendar else date.today()
    end = calendar[-1].date() if calendar else date.today()
    days = max((end - start).days, 1)
    years = days / 365.25
    total_ret = (final_eq / capital - 1.0) * 100.0 if capital else 0.0
    cagr = ((final_eq / capital) ** (1.0 / years) - 1.0) * 100.0 if capital > 0 and final_eq > 0 and years > 0 else 0.0

    result = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "capital": capital,
        "final_equity": round(float(_py(final_eq)), 2),
        "total_return_pct": round(float(_py(total_ret)), 2),
        "cagr_pct": round(float(_py(cagr)), 2),
        "win_rate": round(len(wins) / n * 100.0, 1) if n else 0.0,
        "profit_factor": round(float(_py(gp / gl)), 2) if n else 0.0,
        "max_drawdown_pct": round(float(_py(max_dd)), 2),
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "peak_parallel": peak_par,
        "avg_hold": round(sum(t["hold"] for t in trades) / n, 1) if n else 0.0,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0.0,
        "exits": dict(pd.Series([t["reason"] for t in trades]).value_counts().to_dict()) if n else {},
        "params_name": params.name,
        "open_book": open_book,
        "ranked_signals": len(attempted),
        "signals_skipped_cash": skipped_cash,
        "passed_signals": sum(1 for s in signals if (not need) or (s["flags"] & need) == need),
    }
    if keep_trades:
        result["holdings_history"] = trades[-200:]
        result["equity_curve"] = curve
        result["attempted"] = attempted
    return result


def slice_cal(calendar, start: date, end: date):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    return [ts for ts in calendar if a <= ts <= b]


def nifty_return(nifty: BarPack | None, start: date, end: date) -> float | None:
    if nifty is None:
        return None
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    i0 = next((i for i, ts in enumerate(nifty.index) if ts >= a), None)
    i1 = next((i for i in range(len(nifty.index) - 1, -1, -1) if nifty.index[i] <= b), None)
    if i0 is None or i1 is None or nifty.c[i0] <= 0:
        return None
    return round(float((nifty.c[i1] / nifty.c[i0] - 1.0) * 100.0), 2)


def params_by_name(name: str | None) -> SwingParams:
    if not name:
        return DEFAULT_PARAMS
    needle = str(name).strip().lower()
    for p in DEFAULT_PACKS:
        if p.name.lower() == needle:
            return p
    aliases = {"leaders": LEADERS, "leaders pullback": LEADERS, "balanced": BALANCED, "balanced 3-name": BALANCED}
    return aliases.get(needle, DEFAULT_PARAMS)


def parse_iso_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw).strip()[:10])
    except ValueError:
        return None


def _results_path() -> Path:
    root = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return root / "data" / RESULTS_NAME


def run_window(
    start: date,
    end: date,
    params: SwingParams | None = None,
    capital: float = DEFAULT_CAPITAL,
    *,
    flatten: bool = True,
) -> dict[str, Any]:
    params = params or DEFAULT_PARAMS
    packs, nifty, calendar = preload_packs()
    cal = slice_cal(calendar, start, end)
    if not cal:
        return {"error": "No trading days in that window.", "start": start.isoformat(), "end": end.isoformat()}
    sig_start = pd.Timestamp(start - timedelta(days=5))
    sigs = collect_signals(packs, nifty, sig_start, pd.Timestamp(end), entry=params.entry)
    result = simulate(packs, cal, sigs, params, capital=capital, flatten=flatten, keep_trades=True)
    result["benchmark_pct"] = nifty_return(nifty, start, end)
    result["beat_benchmark"] = (
        result.get("total_return_pct") is not None
        and result.get("benchmark_pct") is not None
        and result["total_return_pct"] > result["benchmark_pct"]
    )
    result["title"] = f"{start.isoformat()} → {end.isoformat()}"
    result["years"] = round((end - start).days / 365.25, 2)
    return result


def _signal_row(sig: dict, params: SwingParams | None = None, pack: BarPack | None = None) -> dict:
    params = params or DEFAULT_PARAMS
    close = round(float(sig["close"]), 2) if sig.get("close") is not None else None
    entry_date = str(pd.Timestamp(sig["entry_ts"]).date())
    entry_px = None
    entry_known = False
    if pack is not None:
        i = pack.loc.get(pd.Timestamp(sig["entry_ts"]))
        if i is not None:
            px = float(pack.o[i])
            if px > 0:
                entry_px = px
                entry_known = True
    if entry_px is None and close:
        entry_px = float(close)
    planned = None
    if entry_px:
        planned = plan_trade(
            entry_px,
            float(sig.get("stop_ref") or 0.0),
            float(sig.get("sig_low") or 0.0),
            float(sig.get("atr") or 0.0),
            params,
        )
    stop = planned["stop"] if planned else (round(float(sig["stop_ref"]), 2) if sig.get("stop_ref") else None)
    target = planned["target"] if planned else None
    if target is None and entry_px and stop and entry_px > stop:
        target = round(entry_px + (entry_px - stop) * params.target_rr, 2)
    risk_pct = planned["risk_pct"] if planned else (
        round((entry_px - stop) / entry_px * 100.0, 1) if entry_px and stop and entry_px > stop else None
    )
    reward_pct = planned["reward_pct"] if planned else (
        round((target / entry_px - 1.0) * 100.0, 1) if entry_px and target else None
    )
    return {
        "symbol": sig["symbol"],
        "signal_date": str(pd.Timestamp(sig["sig_ts"]).date()),
        "entry_date": entry_date,
        "entry": round(float(entry_px), 2) if entry_px else None,
        "entry_known": entry_known,
        "entry_label": f"Open {entry_date[8:10]} {pd.Timestamp(entry_date).strftime('%b')}" if entry_known else "Next open (est.)",
        "close": close,
        "ema20": round(float(sig["ema20"]), 2) if sig.get("ema20") else None,
        "rsi": round(float(sig["rsi"]), 1) if sig.get("rsi") else None,
        "adx": round(float(sig["adx"]), 1) if sig.get("adx") else None,
        "ret63_pct": round(float(sig["ret63"]) * 100.0, 1) if sig.get("ret63") is not None else None,
        "stop": stop,
        "target": target,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "target_rr": float(params.target_rr),
        "max_hold": int(params.max_hold),
        "score": round(float(sig["score"]) * 100.0, 1) if sig.get("score") is not None else 0.0,
    }


def _session_pair(calendar: list, asof: date) -> tuple[date | None, date | None]:
    days = [ts.date() for ts in calendar if ts.date() <= asof]
    if not days:
        return None, None
    last = days[-1]
    prev = days[-2] if len(days) >= 2 else None
    return last, prev


def _session_label(d: date | None) -> str:
    if d is None:
        return ""
    return d.strftime("%a %d %b")


def passed_signals(
    params: SwingParams | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[dict]:
    params = params or DEFAULT_PARAMS
    end = end or date.today()
    start = start or (end - timedelta(days=365))
    packs, nifty, _calendar = preload_packs()
    sigs = collect_signals(
        packs, nifty, pd.Timestamp(start), pd.Timestamp(end), entry=params.entry,
    )
    need = int(params.flags)
    rows = []
    for sig in sigs:
        if need and (sig["flags"] & need) != need:
            continue
        rows.append(_signal_row(sig, params=params, pack=packs.get(sig["symbol"])))
    rows.sort(key=lambda r: (r["signal_date"], -float(r["score"] or 0)))
    return rows


def rank_best_per_day(
    rows: list[dict],
    *,
    max_new: int = 3,
    cooldown_days: int = 5,
) -> list[dict]:
    """Keep the same names the backtest would try: top max_new by 3m RS, with cooldown."""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_date[str(row.get("signal_date") or "")].append(row)
    last_kept: dict[str, date] = {}
    out: list[dict] = []
    for day in sorted(k for k in by_date if k):
        try:
            d = date.fromisoformat(day[:10])
        except ValueError:
            continue
        ranked = sorted(by_date[day], key=lambda r: -float(r.get("score") or 0))
        taken = 0
        for row in ranked:
            if taken >= max_new:
                break
            sym = str(row.get("symbol") or "")
            prev = last_kept.get(sym)
            if cooldown_days and prev is not None and (d - prev).days < cooldown_days:
                continue
            out.append(row)
            if sym:
                last_kept[sym] = d
            taken += 1
    return out


def live_signals(params: SwingParams | None = None, asof: date | None = None) -> list[dict]:
    params = params or DEFAULT_PARAMS
    asof = asof or date.today()
    _, _, calendar = preload_packs()
    last, _prev = _session_pair(calendar, asof)
    if last is None:
        return []
    rows = passed_signals(params, start=last - timedelta(days=5), end=last)
    ranked = rank_best_per_day(
        rows, max_new=int(params.max_new), cooldown_days=int(params.cooldown),
    )
    today = [r for r in ranked if r["signal_date"] == last.isoformat()]
    return today


def signal_board(
    params: SwingParams | None = None,
    asof: date | None = None,
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Today / yesterday names plus daily counts for the bar chart."""
    params = params or DEFAULT_PARAMS
    asof = asof or date.today()
    packs, _nifty, calendar = preload_packs()
    last, prev = _session_pair(calendar, asof)
    start = asof - timedelta(days=int(lookback_days) + int(params.cooldown) + 5)
    raw_rows = passed_signals(params, start=start, end=asof)
    rows = rank_best_per_day(
        raw_rows, max_new=int(params.max_new), cooldown_days=int(params.cooldown),
    )
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_date[row["signal_date"]].append(row)

    def _daily(days: int) -> list[dict]:
        cut = (asof - timedelta(days=days)).isoformat()
        out = []
        for day in sorted(by_date):
            if day < cut:
                continue
            items = by_date[day]
            out.append({
                "date": day,
                "count": len(items),
                "symbols": ", ".join(i["symbol"] for i in items[:8]),
            })
        return out

    today_key = last.isoformat() if last else ""
    yest_key = prev.isoformat() if prev else ""
    today_rows = sorted(by_date.get(today_key, []), key=lambda r: -float(r["score"] or 0))
    yest_rows = sorted(by_date.get(yest_key, []), key=lambda r: -float(r["score"] or 0))
    is_today = bool(last and last == asof)
    return {
        "asof": asof.isoformat(),
        "last_session": today_key,
        "prev_session": yest_key,
        "today_title": "Today" if is_today else "Last session",
        "yesterday_title": "Yesterday" if is_today else "Prior session",
        "today_label": _session_label(last),
        "yesterday_label": _session_label(prev),
        "today": today_rows,
        "yesterday": yest_rows,
        "daily_6m": _daily(183),
        "daily_1y": _daily(365),
        "by_date": {k: v for k, v in by_date.items()},
        "universe": len(packs),
    }


def live_book(params: SwingParams | None = None, asof: date | None = None) -> list[dict]:
    params = params or DEFAULT_PARAMS
    asof = asof or date.today()
    start = asof - timedelta(days=120)
    result = run_window(start, asof, params=params, flatten=False)
    return result.get("open_book") or []


def _run_from_signals(
    packs, nifty, calendar, sigs, start: date, end: date, params: SwingParams, capital: float = DEFAULT_CAPITAL,
    *, flatten: bool = True,
) -> dict[str, Any]:
    cal = slice_cal(calendar, start, end)
    if not cal:
        return {"error": "No trading days in that window.", "start": start.isoformat(), "end": end.isoformat()}
    result = simulate(packs, cal, sigs, params, capital=capital, flatten=flatten, keep_trades=True)
    result["benchmark_pct"] = nifty_return(nifty, start, end)
    result["beat_benchmark"] = (
        result.get("total_return_pct") is not None
        and result.get("benchmark_pct") is not None
        and result["total_return_pct"] > result["benchmark_pct"]
    )
    result["title"] = f"{start.isoformat()} → {end.isoformat()}"
    result["years"] = round((end - start).days / 365.25, 2)
    return result


def build_results(params: SwingParams | None = None, end: date | None = None) -> dict[str, Any]:
    params = params or DEFAULT_PARAMS
    end = end or date.today()
    packs, nifty, calendar = preload_packs()
    longest_start = end - timedelta(days=365 * 5 + 5)
    sigs_by_entry: dict[str, list] = {}
    windows = []
    for years in WINDOW_YEARS:
        start = end - timedelta(days=365 * years + 1)
        if params.entry not in sigs_by_entry:
            sigs_by_entry[params.entry] = collect_signals(
                packs, nifty, pd.Timestamp(longest_start), pd.Timestamp(end), entry=params.entry,
            )
        result = _run_from_signals(packs, nifty, calendar, sigs_by_entry[params.entry], start, end, params)
        result["title"] = f"Last {years} year" + ("s" if years != 1 else "")
        result["years"] = years
        row = {
            "title": result["title"],
            "years": years,
            "start": result["start"],
            "end": result["end"],
            "total_return_pct": result.get("total_return_pct"),
            "cagr_pct": result.get("cagr_pct"),
            "max_drawdown_pct": result.get("max_drawdown_pct"),
            "win_rate": result.get("win_rate"),
            "profit_factor": result.get("profit_factor"),
            "trades": result.get("trades"),
            "avg_hold": result.get("avg_hold"),
            "benchmark_pct": result.get("benchmark_pct"),
            "beat_benchmark": result.get("beat_benchmark"),
            "final_equity": result.get("final_equity"),
            "exits": result.get("exits"),
            "equity_curve": result.get("equity_curve") if years in (1, 5) else None,
        }
        if years == 1:
            row["holdings_history"] = result.get("holdings_history")
        windows.append(row)

    pack_runs = {}
    for pack in DEFAULT_PACKS:
        if pack.entry not in sigs_by_entry:
            sigs_by_entry[pack.entry] = collect_signals(
                packs, nifty, pd.Timestamp(longest_start), pd.Timestamp(end), entry=pack.entry,
            )
        pack_windows = []
        for years in WINDOW_YEARS:
            start = end - timedelta(days=365 * years + 1)
            r = _run_from_signals(packs, nifty, calendar, sigs_by_entry[pack.entry], start, end, pack)
            pack_windows.append({
                "title": f"Last {years} year" + ("s" if years != 1 else ""),
                "years": years,
                "start": r.get("start"),
                "end": r.get("end"),
                "total_return_pct": r.get("total_return_pct"),
                "cagr_pct": r.get("cagr_pct"),
                "max_drawdown_pct": r.get("max_drawdown_pct"),
                "win_rate": r.get("win_rate"),
                "profit_factor": r.get("profit_factor"),
                "trades": r.get("trades"),
                "avg_hold": r.get("avg_hold"),
                "benchmark_pct": r.get("benchmark_pct"),
                "beat_benchmark": r.get("beat_benchmark"),
            })
        pack_runs[pack.name] = {"windows": pack_windows}

    picks = live_signals(params, end)
    held = live_book(params, end)
    payload = {
        "strategy": STRATEGY_NAME,
        "blurb": (
            "Nifty 200, 3–15 day swing. Buy the bounce off the 20-day EMA in a "
            "Qullamaggie-style leader: 3-month relative strength, tight 10-day range, "
            "and enough ATR to actually move. Stop under the pullback, trail 2.5 ATR, "
            "take 1.5R or time-stop at 15 sessions. Average hold in sample is about 8 days."
        ),
        "disclaimer": (
            "1.5R (not 2.5R) is the consistency pack: 2024 and 2025 stay green instead of a "
            "large 2024 loss. 8% risk and up to 70% of equity in one name is still aggressive. "
            "Past backtests are not a forecast. No costs, slippage, or gap-through-stop modelling."
        ),
        "params": asdict(params),
        "packs": [asdict(p) for p in DEFAULT_PACKS],
        "windows": windows,
        "pack_runs": pack_runs,
        "picks": picks,
        "open_book": held,
        "coverage": {
            "universe": len(packs),
            "days": len(calendar),
            "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "asof": end.isoformat(),
        "needs_fetch": len(packs) == 0,
    }
    return payload


def save_results(payload: dict[str, Any]) -> Path:
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_results() -> dict[str, Any] | None:
    path = _results_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_page(
    *,
    rebuild: bool = False,
    params: SwingParams | None = None,
    asof: date | None = None,
) -> dict[str, Any]:
    params = params or DEFAULT_PARAMS
    asof = asof or date.today()
    if not rebuild:
        existing = load_results()
        if existing:
            existing.setdefault("needs_fetch", False)
            existing["params"] = asdict(params)
            run = (existing.get("pack_runs") or {}).get(params.name)
            if run and run.get("windows"):
                existing["windows"] = run["windows"]
            existing.setdefault("picks", [])
            existing.setdefault("open_book", [])
            return existing
    packs, _, _ = preload_packs()
    if not packs:
        return {
            "strategy": STRATEGY_NAME,
            "needs_fetch": True,
            "params": asdict(params),
            "packs": [asdict(p) for p in DEFAULT_PACKS],
            "windows": [],
            "pack_runs": {},
            "picks": [],
            "open_book": [],
            "disclaimer": "",
            "blurb": "",
        }
    payload = build_results(params=params, end=asof)
    try:
        save_results(payload)
    except OSError:
        logger.exception("could not save short swing results")
    return payload
