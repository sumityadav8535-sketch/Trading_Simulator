"""Hunt a 3–15 day Nifty 200 swing pack targeting ≥100% last 1 year.

Signal on bar T close (no look-ahead). Enter T+1 open.
Shared cash, risk-based sizing, no leverage.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import numpy as np
import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
TARGET = 100.0
END = date.today()
START_1Y = END - timedelta(days=365)
OUT = os.path.join(ROOT, "data", "short_swing_hunt.json")

F_TREND = 1 << 0
F_STACK = 1 << 1
F_RS63 = 1 << 2
F_ADX20 = 1 << 3
F_RSI_OK = 1 << 4
F_VOL = 1 << 5
F_MKT20 = 1 << 6
F_MKT50 = 1 << 7
F_NOT_EXT = 1 << 8
F_NEAR52 = 1 << 9
F_DI_PLUS = 1 << 10
F_QULLA = 1 << 11


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


class Pack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c", "v",
        "ema8", "ema10", "ema20", "ema50", "ema200",
        "sma20", "sma50", "sma150", "sma200",
        "rsi", "atr", "adx", "plus_di", "minus_di", "vol_sma",
        "high10", "high20", "high252", "low10", "low20", "low14", "low252",
        "ret21", "ret63", "macd_hist", "stoch_k", "stoch_d",
        "bb_lo", "bb_mid", "range7", "nr7", "st", "st_dir", "first_touch",
        "adr10", "range10",
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
        self.ema8 = _ema(c, 8)
        self.ema10 = _ema(c, 10)
        self.ema20 = _ema(c, 20)
        self.ema50 = _ema(c, 50)
        self.ema200 = _ema(c, 200)
        self.sma20 = _sma(c, 20)
        self.sma50 = _sma(c, 50)
        self.sma150 = _sma(c, 150)
        self.sma200 = _sma(c, 200)
        self.rsi = _rsi(c, 14)
        self.atr = _atr(h, l, c, 14)
        self.adx, self.plus_di, self.minus_di = _adx(h, l, c, 14)
        self.vol_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
        self.high10 = _roll_max(h, 10)
        self.high20 = _roll_max(h, 20)
        self.high252 = _roll_max(h, 252)
        self.low10 = _roll_min(l, 10)
        self.low20 = _roll_min(l, 20)
        self.low14 = _roll_min(l, 14)
        self.low252 = _roll_min(l, 252)
        cs = pd.Series(c)
        self.ret21 = cs.pct_change(21).to_numpy()
        self.ret63 = cs.pct_change(63).to_numpy()
        macd = _ema(c, 12) - _ema(c, 26)
        self.macd_hist = macd - _ema(macd, 9)
        hh = _roll_max(h, 14)
        ll = self.low14
        den = np.where((hh - ll) == 0, np.nan, hh - ll)
        raw_k = (c - ll) / den * 100.0
        self.stoch_k = pd.Series(raw_k).rolling(3, min_periods=3).mean().to_numpy()
        self.stoch_d = pd.Series(self.stoch_k).rolling(3, min_periods=3).mean().to_numpy()
        std20 = pd.Series(c).rolling(20, min_periods=20).std().to_numpy()
        self.bb_mid = self.sma20
        self.bb_lo = self.sma20 - 2.0 * std20
        rng = h - l
        self.range7 = pd.Series(rng).rolling(7, min_periods=7).min().to_numpy()
        self.nr7 = rng <= self.range7
        self.st, self.st_dir = supertrend_np(h, l, c, 14, 3.0)
        first = np.zeros(len(c), dtype=bool)
        seen = False
        for i in range(len(c)):
            if self.st_dir[i] <= 0:
                seen = False
                continue
            st_line = self.st[i]
            tagged = (
                not np.isnan(st_line)
                and l[i] <= st_line * 1.008
                and c[i] > st_line
            )
            if tagged and not seen:
                first[i] = True
                seen = True
        self.first_touch = first
        rng_pct = (h - l) / np.maximum(c, 1e-9)
        self.adr10 = pd.Series(rng_pct).rolling(10, min_periods=10).mean().to_numpy()
        self.range10 = (self.high10 - self.low10) / np.maximum(c, 1e-9)


def preload() -> tuple[dict[str, Pack], Pack | None, list]:
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    packs: dict[str, Pack] = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 120:
            continue
        packs[sym] = Pack(sym, df)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty = Pack("NIFTY50", nifty_df) if not nifty_df.empty else None
    cal_start = pd.Timestamp(END - timedelta(days=365 * 5 + 10))
    cal_end = pd.Timestamp(END)
    calendar = sorted({
        ts
        for p in packs.values()
        for ts in p.index[(p.index >= cal_start) & (p.index <= cal_end)].tolist()
    })
    return packs, nifty, calendar


def flags_at(s: Pack, i: int, nifty: Pack | None) -> int:
    c = s.c[i]
    e20, e50, e200 = s.ema20[i], s.ema50[i], s.ema200[i]
    if np.isnan(e50) or np.isnan(e200):
        return 0
    f = 0
    if c > e50 > e200:
        f |= F_TREND
    if (not np.isnan(e20)) and c > e20 > e50:
        f |= F_STACK
    rsi = s.rsi[i]
    if not np.isnan(rsi) and 40 <= rsi <= 70:
        f |= F_RSI_OK
    vs = s.vol_sma[i]
    if vs and not np.isnan(vs) and vs > 0 and s.v[i] >= 1.4 * vs:
        f |= F_VOL
    if not np.isnan(s.adx[i]) and s.adx[i] >= 20:
        f |= F_ADX20
    if (not np.isnan(s.plus_di[i]) and not np.isnan(s.minus_di[i])
            and s.plus_di[i] > s.minus_di[i]):
        f |= F_DI_PLUS
    if (not np.isnan(e20)) and e20 > 0 and (c - e20) / e20 <= 0.10:
        f |= F_NOT_EXT
    if not np.isnan(s.high252[i]) and s.high252[i] > 0 and c >= 0.80 * s.high252[i]:
        f |= F_NEAR52
    adr = s.adr10[i]
    r63 = s.ret63[i]
    tight = s.range10[i]
    if (
        not np.isnan(adr) and adr >= 0.022
        and not np.isnan(r63) and r63 >= 0.15
        and not np.isnan(tight) and tight <= 0.20
        and c > e50
    ):
        f |= F_QULLA
    if nifty is not None:
        ni = nifty.loc.get(s.index[i])
        if ni is not None:
            nc = nifty.c[ni]
            if not np.isnan(nifty.ema20[ni]) and nc > nifty.ema20[ni]:
                f |= F_MKT20
            if not np.isnan(nifty.ema50[ni]) and nc > nifty.ema50[ni]:
                f |= F_MKT50
            nr63 = nifty.ret63[ni]
            if not np.isnan(s.ret63[i]) and not np.isnan(nr63) and s.ret63[i] > nr63:
                f |= F_RS63
    return f


def collect(packs, nifty, entry: str, start_ts, end_ts) -> list[dict]:
    out: list[dict] = []
    for sym, s in packs.items():
        n = len(s.c)
        for i in range(80, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            fire = False
            stop_ref = 0.0
            c0, c1 = s.c[i], s.c[i - 1]
            if entry == "donch10":
                h10p = s.high10[i - 1]
                fire = (not np.isnan(h10p)) and c0 >= h10p and c1 < h10p
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            elif entry == "donch20":
                h20p = s.high20[i - 1]
                fire = (not np.isnan(h20p)) and c0 >= h20p and c1 < h20p
                stop_ref = float(s.low20[i]) if not np.isnan(s.low20[i]) else float(s.l[i])
            elif entry == "ema20pb":
                fire = (
                    not np.isnan(s.ema20[i])
                    and s.l[i] <= s.ema20[i] * 1.006
                    and c0 > s.ema20[i]
                    and c0 > s.o[i]
                    and c1 <= s.ema20[i - 1] * 1.012
                )
                stop_ref = float(min(s.l[i], s.low10[i] if not np.isnan(s.low10[i]) else s.l[i]))
            elif entry == "stpb":
                st_line = s.st[i]
                fire = (
                    s.st_dir[i] > 0 and s.st_dir[i - 1] > 0
                    and not np.isnan(st_line)
                    and s.l[i] <= st_line * 1.008
                    and c0 > st_line
                    and bool(s.first_touch[i])
                )
                stop_ref = float(min(s.l[i], st_line)) if not np.isnan(st_line) else float(s.l[i])
            elif entry == "rsi_reclaim":
                fire = (
                    not np.isnan(s.rsi[i]) and not np.isnan(s.rsi[i - 1])
                    and s.rsi[i - 1] < 45 and s.rsi[i] >= 45
                    and c0 > s.ema50[i]
                    and c0 > s.o[i]
                )
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            elif entry == "macd_turn":
                fire = (
                    not np.isnan(s.macd_hist[i]) and not np.isnan(s.macd_hist[i - 1])
                    and s.macd_hist[i - 1] <= 0 and s.macd_hist[i] > 0
                    and c0 > s.ema50[i]
                )
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            elif entry == "stoch_cross":
                fire = (
                    not np.isnan(s.stoch_k[i]) and not np.isnan(s.stoch_d[i])
                    and not np.isnan(s.stoch_k[i - 1])
                    and s.stoch_k[i - 1] <= s.stoch_d[i - 1]
                    and s.stoch_k[i] > s.stoch_d[i]
                    and s.stoch_k[i] < 35
                    and c0 > s.ema50[i]
                )
                stop_ref = float(s.low14[i]) if not np.isnan(s.low14[i]) else float(s.l[i])
            elif entry == "bb_bounce":
                fire = (
                    not np.isnan(s.bb_lo[i]) and not np.isnan(s.bb_lo[i - 1])
                    and c1 <= s.bb_lo[i - 1]
                    and c0 > s.bb_lo[i]
                    and c0 > s.o[i]
                    and c0 > s.ema50[i]
                )
                stop_ref = float(min(s.l[i], s.bb_lo[i]))
            elif entry == "nr7_bo":
                fire = (
                    bool(s.nr7[i - 1])
                    and c0 > s.h[i - 1]
                    and c0 > s.ema50[i]
                )
                stop_ref = float(s.l[i - 1])
            elif entry == "qulla":
                h10p = s.high10[i - 1]
                fire = (not np.isnan(h10p)) and c0 >= h10p and c1 < h10p
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            if not fire:
                continue
            fl = flags_at(s, i, nifty)
            if entry == "qulla" and not (fl & F_QULLA):
                continue
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            out.append({
                "symbol": sym,
                "sig_i": i,
                "entry_ts": s.index[i + 1],
                "sig_ts": ts,
                "flags": fl,
                "stop_ref": stop_ref,
                "atr": atr,
                "sig_low": float(s.l[i]),
                "score": r63,
            })
    return out


def simulate(
    packs: dict[str, Pack],
    calendar: list,
    signals: list[dict],
    *,
    need_flags: int,
    risk_pct: float,
    exit_mode: str,
    max_hold: int,
    cooldown: int,
    max_open: int,
    max_new: int,
    max_pos_pct: float,
    trail_atr: float,
    min_stop_pct: float,
    max_stop_pct: float,
    target_rr: float,
    keep_trades: bool = False,
) -> dict[str, Any]:
    by_day: dict = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(CAPITAL)
    opens: dict[str, dict] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    peak_eq = CAPITAL
    max_dd = 0.0
    peak_par = 0
    eq_pts: list[tuple] = []

    def equity() -> float:
        return cash + sum(p["notional"] for p in opens.values())

    def mark(ts) -> None:
        nonlocal peak_eq, max_dd
        eq = equity()
        peak_eq = max(peak_eq, eq)
        if peak_eq:
            max_dd = max(max_dd, (peak_eq - eq) / peak_eq * 100.0)
        if (not eq_pts) or ts.weekday() == 4:
            eq_pts.append((ts, eq))

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
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time"
            else:
                if exit_mode == "ema8" and close < s.ema8[i]:
                    exit_p, reason = close, "ema8"
                elif exit_mode == "ema10" and close < s.ema10[i]:
                    exit_p, reason = close, "ema10"
                elif exit_mode == "st_flip" and s.st_dir[i] <= 0:
                    exit_p, reason = close, "st_flip"
            if exit_p is None:
                if exit_mode in ("chandelier", "hybrid"):
                    atr = s.atr[i]
                    trail = pos["stop"]
                    if not np.isnan(atr):
                        trail = max(trail, high - trail_atr * atr)
                    if exit_mode == "hybrid" and not np.isnan(s.st[i]):
                        trail = max(trail, float(s.st[i]))
                    if trail > pos["stop"] and trail < close:
                        pos["stop"] = float(trail)
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym,
                "entry": str(pos["entry_ts"].date()),
                "exit": str(ts.date()),
                "pnl": pnl,
                "pnl_pct": (exit_p / pos["entry"] - 1.0) * 100.0,
                "reason": reason,
                "hold": pos["hold"],
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        taken = 0
        for sig in day_sigs:
            if taken >= max_new or len(opens) >= max_open:
                break
            sym = sig["symbol"]
            if sym in opens:
                continue
            s = packs.get(sym)
            if s is None:
                continue
            prev = last_exit.get(sym)
            if cooldown > 0 and prev is not None and (ts - prev).days < cooldown:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            entry = float(s.o[i])
            if entry <= 0:
                continue
            atr = float(sig["atr"] or 0.0)
            raw = float(sig["stop_ref"] or 0.0)
            if raw <= 0 or raw >= entry:
                raw = min(float(sig["sig_low"]), entry - (1.5 * atr if atr > 0 else entry * 0.04))
            stop = raw
            if atr > 0:
                stop = min(stop, entry - 0.25 * atr)
            if stop <= 0 or stop >= entry:
                continue
            risk = entry - stop
            spct = risk / entry
            if spct < min_stop_pct or spct > max_stop_pct:
                continue
            eq = equity()
            if cash <= 0 or eq <= 0:
                continue
            ps = calculate_position_size(eq, risk_pct, entry, stop)
            qty = min(int(ps.quantity), int(cash // entry), int((eq * max_pos_pct) // entry))
            if qty <= 0:
                continue
            notional = qty * entry
            if notional > cash + 1e-6:
                continue
            cash -= notional
            target = (entry + risk * target_rr) if target_rr > 0 else 0.0
            opens[sym] = {
                "entry": entry, "stop": stop, "target": target, "qty": qty,
                "notional": notional, "hold": 0, "entry_ts": ts,
            }
            peak_par = max(peak_par, len(opens))
            taken += 1
            if s.l[i] <= stop:
                pnl = (stop - entry) * qty
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "entry": str(ts.date()), "exit": str(ts.date()),
                    "pnl": pnl, "pnl_pct": (stop / entry - 1.0) * 100.0,
                    "reason": "stop", "hold": 0,
                })
                last_exit[sym] = ts
                opens.pop(sym, None)
        mark(ts)

    if opens and calendar:
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
                "symbol": sym, "entry": str(pos["entry_ts"].date()), "exit": str(last_ts.date()),
                "pnl": pnl, "pnl_pct": (close / pos["entry"] - 1.0) * 100.0,
                "reason": "eod", "hold": pos["hold"],
            })
        opens.clear()

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    holds = [t["hold"] for t in trades]
    result = {
        "ret": round((cash - CAPITAL) / CAPITAL * 100.0, 2),
        "final": round(cash, 2),
        "wr": round(len(wins) / n * 100.0, 2) if n else 0.0,
        "pf": round(gp / gl, 2) if n else 0.0,
        "dd": round(max_dd, 2),
        "n": n,
        "par": peak_par,
        "avg_hold": round(sum(holds) / n, 1) if n else 0.0,
        "avg_win": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0.0,
        "exits": dict(pd.Series([t["reason"] for t in trades]).value_counts().to_dict()) if n else {},
    }
    if keep_trades:
        result["trades"] = trades
        result["equity"] = [{"date": str(ts.date()), "equity": round(eq, 2)} for ts, eq in eq_pts]
    return result


def slice_cal(calendar, start: date, end: date):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    return [ts for ts in calendar if a <= ts <= b]


def nifty_ret(nifty: Pack | None, start: date, end: date) -> float | None:
    if nifty is None:
        return None
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    i0 = next((i for i, ts in enumerate(nifty.index) if ts >= a), None)
    i1 = next((i for i in range(len(nifty.index) - 1, -1, -1) if nifty.index[i] <= b), None)
    if i0 is None or i1 is None or nifty.c[i0] <= 0:
        return None
    return round((nifty.c[i1] / nifty.c[i0] - 1.0) * 100.0, 2)


def main() -> None:
    t0 = time.time()
    print(f"Loading Nifty 200  last-1y target ≥{TARGET:g}%  hold 3–15d", flush=True)
    packs, nifty, calendar = preload()
    print(f"packs={len(packs)} days={len(calendar)} load {time.time()-t0:.1f}s", flush=True)

    windows = {
        "1y": (START_1Y, END),
        "2y": (END - timedelta(days=365 * 2), END),
        "3y": (END - timedelta(days=365 * 3), END),
        "4y": (END - timedelta(days=365 * 4), END),
        "5y": (END - timedelta(days=365 * 5), END),
    }
    for label, (a, b) in windows.items():
        nr = nifty_ret(nifty, a, b)
        print(f"  Nifty {label} {a}→{b}  {nr:+.2f}%" if nr is not None else f"  Nifty {label} n/a", flush=True)

    start_ts, end_ts = pd.Timestamp(windows["5y"][0]), pd.Timestamp(END)
    entries = [
        "donch10", "donch20", "ema20pb", "stpb", "rsi_reclaim",
        "macd_turn", "stoch_cross", "bb_bounce", "nr7_bo", "qulla",
    ]
    raw: dict[str, list] = {}
    for entry in entries:
        t1 = time.time()
        raw[entry] = collect(packs, nifty, entry, start_ts, end_ts)
        print(f"  signals {entry:12s} {len(raw[entry]):5d}  {time.time()-t1:.1f}s", flush=True)

    filters = [
        ("trend", F_TREND),
        ("trend_rs", F_TREND | F_RS63),
        ("stack_rs", F_STACK | F_RS63),
        ("trend_adx_rs", F_TREND | F_ADX20 | F_RS63),
        ("quality_rs", F_TREND | F_RSI_OK | F_ADX20 | F_RS63),
        ("leaders", F_TREND | F_RS63 | F_NEAR52 | F_DI_PLUS),
        ("qulla", F_QULLA),
        ("rs_mkt20", F_TREND | F_RS63 | F_MKT20),
        ("notext_rs", F_TREND | F_RS63 | F_NOT_EXT),
        ("vol_rs", F_TREND | F_VOL | F_RS63),
    ]
    max_pos = 0.40

    cal1 = slice_cal(calendar, *windows["1y"])
    phase1_entries = ["donch10", "ema20pb", "stpb", "rsi_reclaim", "qulla", "macd_turn", "nr7_bo"]
    jobs = []
    for entry in phase1_entries:
        for fname, fl in filters:
            if entry == "qulla" and fname not in ("qulla", "trend_rs", "leaders"):
                continue
            for ex in ("chandelier", "target"):
                for hold in (12, 15):
                    for risk in (3.0, 5.0):
                        jobs.append({
                            "entry": entry, "filter": fname, "flags": fl,
                            "exit": ex, "hold": hold, "risk": risk,
                            "open": 5, "new": 2, "rr": 2.0, "cd": 5,
                        })
    # de-dup
    seen = set()
    uniq = []
    for j in jobs:
        key = (j["entry"], j["filter"], j["exit"], j["hold"], j["risk"], j["open"], j["rr"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(j)
    jobs = uniq
    print(f"phase1 jobs={len(jobs)} on last 1y ({len(cal1)} days)", flush=True)

    ranked = []
    t2 = time.time()
    for i, j in enumerate(jobs, 1):
        r = simulate(
            packs, cal1, raw[j["entry"]],
            need_flags=j["flags"], risk_pct=j["risk"], exit_mode=j["exit"],
            max_hold=j["hold"], cooldown=j["cd"], max_open=j["open"],
            max_new=j["new"], max_pos_pct=max_pos, trail_atr=2.5,
            min_stop_pct=0.012, max_stop_pct=0.09, target_rr=j["rr"],
        )
        r.update(j)
        ranked.append(r)
        if i % 80 == 0:
            best = max(ranked, key=lambda x: x["ret"])
            print(f"  {i}/{len(jobs)} best1y={best['ret']}% {best['entry']}/{best['filter']} {best['exit']} h{best['hold']} r{best['risk']}", flush=True)
    ranked.sort(key=lambda x: (-x["ret"], -x["pf"], x["dd"]))
    print(f"phase1 done {time.time()-t2:.1f}s  top last-1y:", flush=True)
    for r in ranked[:12]:
        print(
            f"  {r['ret']:7.1f}% WR{r['wr']:5.1f} PF{r['pf']:4.2f} DD{r['dd']:5.1f} "
            f"n={r['n']:3d} hold={r['avg_hold']:4.1f}  "
            f"{r['entry']:12s} {r['filter']:14s} {r['exit']:10s} "
            f"h{r['hold']} risk{r['risk']:g} open{r['open']} {r['rr']}R",
            flush=True,
        )

    # Phase 2: tune top 8 on extra hold/risk/new/cd, still last 1y
    seeds = ranked[:6]
    extra = []
    for s in seeds:
        for hold in (8, 12, 15):
            for risk in (3.0, 5.0, 6.0):
                for mx in (3, 5):
                    for new in (2, 3):
                        for cd in (3, 8):
                            for rr in (2.0, 2.5):
                                extra.append({
                                    "entry": s["entry"], "filter": s["filter"], "flags": s["flags"],
                                    "exit": s["exit"], "hold": hold, "risk": risk,
                                    "open": mx, "new": new, "rr": rr, "cd": cd,
                                })
    seen = set()
    extra_u = []
    for j in extra:
        key = tuple(j[k] for k in ("entry", "filter", "exit", "hold", "risk", "open", "new", "rr", "cd"))
        if key in seen:
            continue
        seen.add(key)
        extra_u.append(j)
    print(f"phase2 tune jobs={len(extra_u)}", flush=True)
    tuned = []
    for i, j in enumerate(extra_u, 1):
        r = simulate(
            packs, cal1, raw[j["entry"]],
            need_flags=j["flags"], risk_pct=j["risk"], exit_mode=j["exit"],
            max_hold=j["hold"], cooldown=j["cd"], max_open=j["open"],
            max_new=j["new"], max_pos_pct=max_pos, trail_atr=2.5,
            min_stop_pct=0.012, max_stop_pct=0.09, target_rr=j["rr"],
        )
        r.update(j)
        tuned.append(r)
    tuned.sort(key=lambda x: (-x["ret"], -x["pf"], x["dd"]))
    print("phase2 top last-1y:", flush=True)
    for r in tuned[:10]:
        print(
            f"  {r['ret']:7.1f}% WR{r['wr']:5.1f} PF{r['pf']:4.2f} DD{r['dd']:5.1f} "
            f"n={r['n']:3d} hold={r['avg_hold']:4.1f}  "
            f"{r['entry']:12s} {r['filter']:14s} {r['exit']:10s} "
            f"h{r['hold']} risk{r['risk']:g} open{r['open']} new{r['new']} cd{r['cd']} {r['rr']}R",
            flush=True,
        )

    # Phase 3: multi-year on packs with last-1y >= 80 or top 12
    candidates = [r for r in tuned if r["ret"] >= 80][:15] or tuned[:12]
    print(f"phase3 multi-year on {len(candidates)} packs", flush=True)
    multi = []
    for j in candidates:
        row = dict(j)
        years = {}
        for label, (a, b) in windows.items():
            cal = slice_cal(calendar, a, b)
            r = simulate(
                packs, cal, raw[j["entry"]],
                need_flags=j["flags"], risk_pct=j["risk"], exit_mode=j["exit"],
                max_hold=j["hold"], cooldown=j["cd"], max_open=j["open"],
                max_new=j["new"], max_pos_pct=max_pos, trail_atr=2.5,
                min_stop_pct=0.012, max_stop_pct=0.09, target_rr=j["rr"],
            )
            years[label] = {
                "ret": r["ret"], "wr": r["wr"], "pf": r["pf"], "dd": r["dd"],
                "n": r["n"], "avg_hold": r["avg_hold"],
                "nifty": nifty_ret(nifty, a, b),
            }
        row["windows"] = years
        # robustness: last 1y, then 5y, then worst of 2-4y
        y1 = years["1y"]["ret"]
        y5 = years["5y"]["ret"]
        worst = min(years[k]["ret"] for k in ("2y", "3y", "4y", "5y"))
        row["score"] = y1 * 2 + y5 * 0.5 + worst * 0.5
        row["worst"] = worst
        multi.append(row)
        print(
            f"  1y={years['1y']['ret']:7.1f} 2y={years['2y']['ret']:7.1f} "
            f"3y={years['3y']['ret']:7.1f} 4y={years['4y']['ret']:7.1f} "
            f"5y={years['5y']['ret']:7.1f} worst={worst:6.1f}  "
            f"{j['entry']}/{j['filter']} {j['exit']} h{j['hold']} r{j['risk']} "
            f"open{j['open']} {j['rr']}R holdavg={years['1y']['avg_hold']}",
            flush=True,
        )

    hit = [m for m in multi if m["windows"]["1y"]["ret"] >= TARGET and m["windows"]["1y"]["avg_hold"] <= 15.5]
    pool = hit or multi
    pool.sort(key=lambda x: (-x["windows"]["1y"]["ret"], -x["score"], x["windows"]["1y"]["dd"]))
    winner = pool[0]
    print("\nWINNER", flush=True)
    print(
        f"  {winner['entry']}/{winner['filter']} exit={winner['exit']} "
        f"hold≤{winner['hold']} risk{winner['risk']}% open{winner['open']} "
        f"new{winner['new']} cd{winner['cd']} {winner['rr']}R",
        flush=True,
    )
    for label, w in winner["windows"].items():
        print(
            f"  {label}: {w['ret']:+.1f}% WR{w['wr']:.1f} PF{w['pf']:.2f} "
            f"DD{w['dd']:.1f} n={w['n']} hold={w['avg_hold']} nifty={w['nifty']}",
            flush=True,
        )

    full = simulate(
        packs, cal1, raw[winner["entry"]],
        need_flags=winner["flags"], risk_pct=winner["risk"], exit_mode=winner["exit"],
        max_hold=winner["hold"], cooldown=winner["cd"], max_open=winner["open"],
        max_new=winner["new"], max_pos_pct=max_pos, trail_atr=2.5,
        min_stop_pct=0.012, max_stop_pct=0.09, target_rr=winner["rr"],
        keep_trades=True,
    )
    payload = {
        "target": TARGET,
        "end": str(END),
        "winner": {k: winner[k] for k in winner if k != "flags"} | {"flags": int(winner["flags"])},
        "winner_1y_trades_head": (full.get("trades") or [])[:25],
        "phase1_top": [
            {k: r[k] for k in r if k != "flags"} | {"flags": int(r["flags"])}
            for r in ranked[:15]
        ],
        "phase2_top": [
            {k: r[k] for k in r if k != "flags"} | {"flags": int(r["flags"])}
            for r in tuned[:15]
        ],
        "multi": [
            {k: m[k] for k in m if k != "flags"} | {"flags": int(m["flags"])}
            for m in pool[:10]
        ],
        "hit_100_count": len(hit),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {OUT}  elapsed {payload['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
