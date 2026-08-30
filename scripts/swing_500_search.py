"""
Hunt a Nifty-200 swing strategy that compounds >= 500% over ~5 years.

Signal on bar T close (no look-ahead). Enter T+1 open.
Shared cash, no leverage. Default size = 2% of current equity as notional
(user constraint). Also searches 2% risk-based sizing (classic 2% rule).

Families (research-backed):
  - Donchian 20/10 breakout (Turtle / 20-day breakout long)
  - Qullamaggie continuation (high RS + tightness + 10d high)
  - Minervini SEPA trend template + 20d high
  - EMA20 pullback in an EMA50/200 uptrend
  - 52-week high breakout with volume
  - Supertrend(14,3) first pullback (existing engine family)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
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
TARGET_RET = 500.0
START = date(2021, 7, 5)
END = date(2026, 8, 21)
POS_PCT = 2.0  # 2% of available equity per trade
OUT = os.path.join(ROOT, "data", "swing_500_search.json")

F_TREND = 1 << 0          # close > ema50 > ema200
F_STACK = 1 << 1          # close > ema20 > ema50
F_ABOVE200 = 1 << 2
F_MINERVINI = 1 << 3
F_RS20 = 1 << 4           # 21d ret > nifty
F_RS63 = 1 << 5           # 63d ret > nifty
F_VOL = 1 << 6            # vol >= 1.5x
F_ADX20 = 1 << 7
F_ADX25 = 1 << 8
F_RSI_OK = 1 << 9         # 40 <= rsi <= 72
F_MKT50 = 1 << 10         # nifty > ema50
F_MKT200 = 1 << 11        # nifty > ema200
F_NEAR52 = 1 << 12        # within 25% of 52w high
F_OFFLOW = 1 << 13        # >= 25% above 52w low
F_QULLA = 1 << 14         # RS run + tightness + ADR
F_NOT_EXT = 1 << 15       # close <= 12% above ema20


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


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _roll_max(s: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(s).rolling(n, min_periods=n).max().to_numpy()


def _roll_min(s: np.ndarray, n: int) -> np.ndarray:
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
        "ema10", "ema20", "ema50", "ema200",
        "sma50", "sma150", "sma200",
        "rsi", "atr", "adx", "vol_sma",
        "high10", "high20", "high55", "high252",
        "low10", "low20", "low55", "low252",
        "ret21", "ret63", "ret126",
        "adr10", "range10", "st", "st_dir", "first_touch",
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
        self.ema10 = _ema(c, 10)
        self.ema20 = _ema(c, 20)
        self.ema50 = _ema(c, 50)
        self.ema200 = _ema(c, 200)
        self.sma50 = _sma(c, 50)
        self.sma150 = _sma(c, 150)
        self.sma200 = _sma(c, 200)
        self.rsi = _rsi(c, 14)
        self.atr = _atr(h, l, c, 14)
        self.adx, _, _ = _adx(h, l, c, 14)
        self.vol_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
        self.high10 = _roll_max(h, 10)
        self.high20 = _roll_max(h, 20)
        self.high55 = _roll_max(h, 55)
        self.high252 = _roll_max(h, 252)
        self.low10 = _roll_min(l, 10)
        self.low20 = _roll_min(l, 20)
        self.low55 = _roll_min(l, 55)
        self.low252 = _roll_min(l, 252)
        cs = pd.Series(c)
        self.ret21 = cs.pct_change(21).to_numpy()
        self.ret63 = cs.pct_change(63).to_numpy()
        self.ret126 = cs.pct_change(126).to_numpy()
        rng = (h - l) / np.maximum(c, 1e-9)
        self.adr10 = pd.Series(rng).rolling(10, min_periods=10).mean().to_numpy()
        self.range10 = (self.high10 - self.low10) / np.maximum(c, 1e-9)
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


def _preload() -> tuple[dict[str, Pack], Pack | None, list]:
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 80:
            continue
        frames[sym] = df
    packs = {sym: Pack(sym, df) for sym, df in frames.items()}
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty = Pack("NIFTY50", nifty_df) if not nifty_df.empty else None
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts
        for df in frames.values()
        for ts in df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist()
    })
    return packs, nifty, calendar


def _flags(s: Pack, i: int, nifty: Pack | None) -> int:
    c = s.c[i]
    flags = 0
    e20, e50, e200 = s.ema20[i], s.ema50[i], s.ema200[i]
    if np.isnan(e50) or np.isnan(e200):
        return 0
    if c > e50 > e200:
        flags |= F_TREND
    if (not np.isnan(e20)) and c > e20 > e50:
        flags |= F_STACK
    if c > e200:
        flags |= F_ABOVE200
    sma50, sma150, sma200 = s.sma50[i], s.sma150[i], s.sma200[i]
    if (
        not np.isnan(sma50) and not np.isnan(sma150) and not np.isnan(sma200)
        and i >= 222
        and c > sma50 > sma150 > sma200
        and sma200 > s.sma200[i - 22]
        and c >= 1.25 * s.low252[i]
        and c >= 0.75 * s.high252[i]
    ):
        flags |= F_MINERVINI
    rsi = s.rsi[i]
    if not np.isnan(rsi) and 40.0 <= rsi <= 72.0:
        flags |= F_RSI_OK
    vs = s.vol_sma[i]
    if vs and not np.isnan(vs) and vs > 0 and s.v[i] >= 1.5 * vs:
        flags |= F_VOL
    adx = s.adx[i]
    if not np.isnan(adx):
        if adx >= 20:
            flags |= F_ADX20
        if adx >= 25:
            flags |= F_ADX25
    if (not np.isnan(e20)) and e20 > 0 and (c - e20) / e20 <= 0.12:
        flags |= F_NOT_EXT
    if not np.isnan(s.high252[i]) and s.high252[i] > 0 and c >= 0.75 * s.high252[i]:
        flags |= F_NEAR52
    if not np.isnan(s.low252[i]) and s.low252[i] > 0 and c >= 1.25 * s.low252[i]:
        flags |= F_OFFLOW
    adr = s.adr10[i]
    r63 = s.ret63[i]
    r126 = s.ret126[i]
    tight = s.range10[i]
    qulla = (
        not np.isnan(adr) and adr >= 0.025
        and ((not np.isnan(r63) and r63 >= 0.20) or (not np.isnan(r126) and r126 >= 0.30))
        and not np.isnan(tight) and tight <= 0.18
        and c > e50
    )
    if qulla:
        flags |= F_QULLA
    if nifty is not None:
        ts = s.index[i]
        ni = nifty.loc.get(ts)
        if ni is not None:
            nc = nifty.c[ni]
            if (not np.isnan(nifty.ema50[ni])) and nc > nifty.ema50[ni]:
                flags |= F_MKT50
            if (not np.isnan(nifty.ema200[ni])) and nc > nifty.ema200[ni]:
                flags |= F_MKT200
            nr21 = nifty.ret21[ni]
            nr63 = nifty.ret63[ni]
            if not np.isnan(s.ret21[i]) and not np.isnan(nr21) and s.ret21[i] > nr21:
                flags |= F_RS20
            if not np.isnan(s.ret63[i]) and not np.isnan(nr63) and s.ret63[i] > nr63:
                flags |= F_RS63
    return flags


def collect(packs: dict[str, Pack], nifty: Pack | None, entry: str, start_ts, end_ts) -> list[dict]:
    out: list[dict] = []
    for sym, s in packs.items():
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            fire = False
            stop_ref = 0.0
            if entry == "donch20":
                h20p = s.high20[i - 1]
                fire = (not np.isnan(h20p)) and s.c[i] >= h20p and s.c[i - 1] < h20p
                stop_ref = float(s.low20[i]) if not np.isnan(s.low20[i]) else 0.0
            elif entry == "donch10":
                h10p = s.high10[i - 1]
                fire = (not np.isnan(h10p)) and s.c[i] >= h10p and s.c[i - 1] < h10p
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else 0.0
            elif entry == "donch55":
                h55p = s.high55[i - 1]
                fire = (not np.isnan(h55p)) and s.c[i] >= h55p and s.c[i - 1] < h55p
                stop_ref = float(s.low20[i]) if not np.isnan(s.low20[i]) else 0.0
            elif entry == "h52":
                h52p = s.high252[i - 1] if i >= 252 else np.nan
                fire = (not np.isnan(h52p)) and s.c[i] >= h52p and s.c[i - 1] < h52p
                stop_ref = float(s.low20[i]) if not np.isnan(s.low20[i]) else 0.0
            elif entry == "ema20pb":
                fire = (
                    (not np.isnan(s.ema20[i]))
                    and s.l[i] <= s.ema20[i] * 1.005
                    and s.c[i] > s.ema20[i]
                    and s.c[i] > s.o[i]
                    and s.c[i - 1] <= s.ema20[i - 1] * 1.01
                )
                stop_ref = float(min(s.l[i], s.low10[i] if not np.isnan(s.low10[i]) else s.l[i]))
            elif entry == "ema10rc":
                fire = (
                    (not np.isnan(s.ema10[i]))
                    and s.c[i] > s.ema10[i]
                    and s.c[i - 1] <= s.ema10[i - 1]
                )
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else float(s.l[i])
            elif entry == "qulla":
                h10p = s.high10[i - 1]
                fire = (not np.isnan(h10p)) and s.c[i] >= h10p and s.c[i - 1] < h10p
                stop_ref = float(s.low10[i]) if not np.isnan(s.low10[i]) else 0.0
            elif entry == "minervini":
                h20p = s.high20[i - 1]
                fire = (not np.isnan(h20p)) and s.c[i] >= h20p and s.c[i - 1] < h20p
                stop_ref = float(s.low20[i]) if not np.isnan(s.low20[i]) else 0.0
            elif entry == "stpb":
                st_line = s.st[i]
                fire = (
                    s.st_dir[i] > 0
                    and s.st_dir[i - 1] > 0
                    and (not np.isnan(st_line))
                    and s.l[i] <= st_line * 1.008
                    and s.c[i] > st_line
                    and bool(s.first_touch[i])
                )
                stop_ref = float(min(s.l[i], st_line)) if not np.isnan(st_line) else float(s.l[i])
            if not fire:
                continue
            flags = _flags(s, i, nifty)
            if entry == "qulla" and not (flags & F_QULLA):
                continue
            if entry == "minervini" and not (flags & F_MINERVINI):
                continue
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            out.append({
                "symbol": sym,
                "sig_i": i,
                "entry_ts": s.index[i + 1],
                "sig_ts": ts,
                "flags": flags,
                "stop_ref": stop_ref,
                "atr": atr,
                "sig_low": float(s.l[i]),
                "score": r63,
            })
    return out


def _year_returns(equity_pts: list[tuple[pd.Timestamp, float]], capital: float) -> dict[str, float]:
    by_year: dict[int, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    for ts, eq in equity_pts:
        by_year[int(ts.year)].append((ts, eq))
    out: dict[str, float] = {}
    prev = capital
    for y in sorted(by_year):
        pts = by_year[y]
        start_eq = pts[0][1]
        end_eq = pts[-1][1]
        # use last equity of previous year if we have it
        ret = (end_eq / prev - 1.0) * 100.0 if prev > 0 else 0.0
        out[str(y)] = round(ret, 1)
        prev = end_eq
    return out


def simulate(
    packs: dict[str, Pack],
    calendar: list,
    signals: list[dict],
    *,
    need_flags: int,
    size_mode: str,
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
    cost_pct: float = 0.0,
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
    eq_pts: list[tuple[pd.Timestamp, float]] = [(calendar[0], CAPITAL)] if calendar else []

    def equity() -> float:
        return cash + sum(p["notional"] for p in opens.values())

    def mark_dd(ts) -> None:
        nonlocal peak_eq, max_dd
        eq = equity()
        peak_eq = max(peak_eq, eq)
        if peak_eq:
            max_dd = max(max_dd, (peak_eq - eq) / peak_eq * 100.0)
        if ts.weekday() == 4:
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
            # ratchet trail from *previous* stop vs today's low (no look-ahead)
            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time"
            else:
                if exit_mode == "ema10" and close < s.ema10[i]:
                    exit_p, reason = close, "ema10"
                elif exit_mode == "ema20" and close < s.ema20[i]:
                    exit_p, reason = close, "ema20"
                elif exit_mode == "donch10" and (not np.isnan(s.low10[i])) and close < s.low10[i]:
                    exit_p, reason = close, "donch10"
                elif exit_mode == "st_flip" and s.st_dir[i] <= 0:
                    exit_p, reason = close, "st_flip"
            if exit_p is None:
                # raise trail after checking today's stop
                if exit_mode in ("chandelier", "st_trail", "hybrid"):
                    atr = s.atr[i]
                    trail = pos["stop"]
                    if exit_mode in ("chandelier", "hybrid") and not np.isnan(atr):
                        cand = high - trail_atr * atr
                        trail = max(trail, cand)
                    if exit_mode in ("st_trail", "hybrid"):
                        st_line = s.st[i]
                        if not np.isnan(st_line):
                            trail = max(trail, float(st_line) - 0.5 * (atr if not np.isnan(atr) else 0.0))
                    if trail > pos["stop"] and trail < close:
                        pos["stop"] = float(trail)
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            if cost_pct:
                pnl -= (pos["entry"] + exit_p) * pos["qty"] * cost_pct / 100.0
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "entry": str(pos["entry_ts"].date()), "exit": str(ts.date()),
                "pnl": pnl, "pnl_pct": (exit_p / pos["entry"] - 1.0) * 100.0,
                "reason": reason, "hold": pos["hold"],
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        taken = 0
        for sig in day_sigs:
            if taken >= max_new:
                break
            if len(opens) >= max_open:
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
            if size_mode == "notional":
                budget = eq * (POS_PCT / 100.0)
                qty = int(min(budget, cash, eq * max_pos_pct) // entry)
            else:
                ps = calculate_position_size(eq, POS_PCT, entry, stop)
                qty_cash = int(cash // entry)
                qty_cap = int((eq * max_pos_pct) // entry)
                qty = min(int(ps.quantity), qty_cash, qty_cap)
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
                if cost_pct:
                    pnl -= (entry + stop) * qty * cost_pct / 100.0
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "entry": str(ts.date()), "exit": str(ts.date()),
                    "pnl": pnl, "pnl_pct": (stop / entry - 1.0) * 100.0,
                    "reason": "stop", "hold": 0,
                })
                last_exit[sym] = ts
                opens.pop(sym, None)
        mark_dd(ts)

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
    years = _year_returns(eq_pts, CAPITAL)
    avg_win = (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0.0
    return {
        "ret": round((cash - CAPITAL) / CAPITAL * 100.0, 2),
        "final": round(cash, 2),
        "wr": round(len(wins) / n * 100.0, 2) if n else 0.0,
        "pf": round(gp / gl, 2) if n else 0.0,
        "dd": round(max_dd, 2),
        "n": n,
        "par": peak_par,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_hold": round(sum(t["hold"] for t in trades) / n, 1) if n else 0.0,
        "years": years,
        "exits": dict(pd.Series([t["reason"] for t in trades]).value_counts().to_dict()) if n else {},
    }


FILTER_PACKS = [
    ("trend", F_TREND),
    ("trend_mkt50", F_TREND | F_MKT50),
    ("trend_mkt200", F_TREND | F_MKT200),
    ("trend_rs", F_TREND | F_RS63),
    ("trend_rs_mkt", F_TREND | F_RS63 | F_MKT50),
    ("minervini", F_MINERVINI),
    ("minervini_mkt", F_MINERVINI | F_MKT50),
    ("qulla", F_QULLA),
    ("qulla_mkt", F_QULLA | F_MKT50),
    ("stack_adx", F_STACK | F_ADX20),
    ("trend_vol", F_TREND | F_VOL),
    ("quality", F_TREND | F_RSI_OK | F_ADX20),
    ("quality_mkt", F_TREND | F_RSI_OK | F_ADX20 | F_MKT50),
    ("near52", F_TREND | F_NEAR52 | F_OFFLOW),
    ("above200_mkt", F_ABOVE200 | F_MKT200),
]


def main() -> None:
    t0 = time.time()
    print(f"Loading Nifty 200  {START} → {END}  target ≥{TARGET_RET:g}%", flush=True)
    packs, nifty, calendar = _preload()
    print(f"packs={len(packs)}  days={len(calendar)}  load {time.time()-t0:.1f}s", flush=True)

    if nifty is not None:
        i0 = nifty.loc.get(pd.Timestamp(START))
        i1 = nifty.loc.get(calendar[-1])
        if i0 is not None and i1 is not None and nifty.c[i0] > 0:
            bh = (nifty.c[i1] / nifty.c[i0] - 1.0) * 100.0
            print(f"Nifty 50 buy&hold over window: {bh:+.1f}%", flush=True)

    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    entries = ["donch20", "donch10", "donch55", "h52", "ema20pb", "qulla", "minervini", "stpb", "ema10rc"]
    raw_cache: dict[str, list] = {}
    for entry in entries:
        t1 = time.time()
        raw_cache[entry] = collect(packs, nifty, entry, start_ts, end_ts)
        print(f"  signals {entry:10s}  n={len(raw_cache[entry]):5d}  {time.time()-t1:.1f}s", flush=True)

    jobs: list[dict[str, Any]] = []

    def add_job(**kw):
        jobs.append(kw)

    # Phase 1 — 2% notional, high deployment (user constraint)
    for entry in entries:
        for fname, flags in FILTER_PACKS:
            if entry == "qulla" and "qulla" not in fname and fname not in ("trend_mkt50", "trend_rs_mkt"):
                continue
            if entry == "minervini" and "minervini" not in fname and fname not in ("trend_mkt50", "near52"):
                continue
            add_job(
                entry=entry, filt=fname, flags=flags, size="notional",
                exit="chandelier", hold=90, cd=5, max_open=50, max_new=8,
                max_pos=0.10, trail=3.0, min_sp=0.015, max_sp=0.14, rr=0.0,
            )

    # Phase 1b — 2% risk (classic 2% rule), concentrated runners
    for entry in ("donch20", "qulla", "minervini", "stpb", "h52", "ema20pb"):
        for fname, flags in FILTER_PACKS:
            if entry == "qulla" and "qulla" not in fname and fname not in ("trend_mkt50", "trend_rs_mkt"):
                continue
            if entry == "minervini" and "minervini" not in fname and fname not in ("trend_mkt50", "near52"):
                continue
            add_job(
                entry=entry, filt=fname, flags=flags, size="risk",
                exit="chandelier", hold=120, cd=8, max_open=12, max_new=4,
                max_pos=0.50, trail=3.0, min_sp=0.018, max_sp=0.12, rr=0.0,
            )

    print(f"\nPhase 1 jobs={len(jobs)}", flush=True)
    rows = []
    for i, job in enumerate(jobs, 1):
        r = simulate(
            packs, calendar, raw_cache[job["entry"]],
            need_flags=job["flags"], size_mode=job["size"], exit_mode=job["exit"],
            max_hold=job["hold"], cooldown=job["cd"], max_open=job["max_open"],
            max_new=job["max_new"], max_pos_pct=job["max_pos"], trail_atr=job["trail"],
            min_stop_pct=job["min_sp"], max_stop_pct=job["max_sp"], target_rr=job["rr"],
        )
        r.update(job)
        r.pop("flags", None)
        rows.append(r)
        flag = " *** HIT 500% ***" if r["ret"] >= TARGET_RET else ""
        if i % 15 == 0 or r["ret"] >= 200 or r["ret"] >= TARGET_RET:
            print(
                f"[{i:3d}/{len(jobs)}] {r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} "
                f"DD={r['dd']:5.1f} n={r['n']:4d} {job['entry']}/{job['filt']} "
                f"{job['size']} {job['exit']}{flag}",
                flush=True,
            )

    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n===== TOP 20 =====", flush=True)
    for r in rows[:20]:
        flag = " *** HIT ***" if r["ret"] >= TARGET_RET else ""
        print(
            f"{r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} "
            f"n={r['n']:4d} par={r['par']:2d} hold={r['avg_hold']:5.1f} "
            f"{r['entry']}/{r['filt']} {r['size']} {r['exit']} {r['years']}{flag}",
            flush=True,
        )

    winners = [r for r in rows if r["ret"] >= TARGET_RET]
    print(f"\nHits ≥{TARGET_RET:g}%: {len(winners)}   elapsed {time.time()-t0:.0f}s", flush=True)

    # Phase 2 — tune top families
    top_families = []
    seen = set()
    for r in rows:
        key = (r["entry"], r["size"])
        if key in seen:
            continue
        seen.add(key)
        top_families.append(r)
        if len(top_families) >= 6:
            break

    print("\n===== PHASE 2 greedy tune of top families =====", flush=True)
    phase2: list[dict] = []
    fmap = dict(FILTER_PACKS)

    def run_job(job: dict) -> dict:
        r = simulate(
            packs, calendar, raw_cache[job["entry"]],
            need_flags=job["flags"], size_mode=job["size"], exit_mode=job["exit"],
            max_hold=job["hold"], cooldown=job["cd"], max_open=job["max_open"],
            max_new=job["max_new"], max_pos_pct=job["max_pos"], trail_atr=job["trail"],
            min_stop_pct=job["min_sp"], max_stop_pct=job["max_sp"], target_rr=job["rr"],
        )
        r.update({k: v for k, v in job.items() if k != "flags"})
        phase2.append(r)
        if r["ret"] >= TARGET_RET:
            print(
                f"  HIT {r['ret']:+.1f}% {job['entry']}/{job['filt']} {job['size']} "
                f"{job['exit']} hold={job['hold']} trail={job['trail']} rr={job['rr']} "
                f"open={job['max_open']} years={r['years']}",
                flush=True,
            )
        return r

    for base in top_families:
        entry = base["entry"]
        size = base["size"]
        cd = 5 if size == "notional" else 8
        min_sp = 0.015 if size == "notional" else 0.018
        max_sp = 0.14 if size == "notional" else 0.12
        seed = dict(
            entry=entry, filt=base["filt"], flags=fmap[base["filt"]], size=size,
            exit=base.get("exit", "chandelier"), hold=base.get("hold", 90),
            cd=cd, max_open=base.get("max_open", 50 if size == "notional" else 12),
            max_new=base.get("max_new", 8 if size == "notional" else 4),
            max_pos=base.get("max_pos", 0.10 if size == "notional" else 0.50),
            trail=base.get("trail", 3.0), min_sp=min_sp, max_sp=max_sp,
            rr=base.get("rr", 0.0),
        )
        best_local = dict(seed)
        best_ret = -1e9

        def consider(updates: dict) -> dict:
            nonlocal best_local, best_ret
            job = dict(best_local)
            job.update(updates)
            if "filt" in updates:
                job["flags"] = fmap[updates["filt"]]
            r = run_job(job)
            if r["ret"] > best_ret:
                best_ret = r["ret"]
                best_local = job
            return r

        for fname in (base["filt"], "trend_mkt50", "trend_rs_mkt", "quality_mkt", "qulla_mkt", "minervini_mkt", "near52"):
            if fname in fmap:
                consider({"filt": fname})
        for exit_m in ("chandelier", "ema10", "ema20", "donch10", "st_flip", "hybrid"):
            consider({"exit": exit_m})
        for hold in (40, 60, 90, 150, 250):
            consider({"hold": hold})
        for trail in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            consider({"trail": trail})
        for rr in (0.0, 2.5, 4.0, 6.0, 10.0):
            consider({"rr": rr})
        if size == "notional":
            for max_open, max_new, max_pos in ((50, 8, 0.08), (40, 6, 0.10), (30, 5, 0.15), (20, 4, 0.20)):
                consider({"max_open": max_open, "max_new": max_new, "max_pos": max_pos})
        else:
            for max_open, max_new, max_pos in ((6, 2, 0.60), (8, 3, 0.45), (12, 4, 0.50), (15, 5, 0.33), (4, 2, 0.80)):
                consider({"max_open": max_open, "max_new": max_new, "max_pos": max_pos})
        print(
            f"  family {entry}/{size} best={best_ret:+.1f}%  {best_local['filt']} "
            f"{best_local['exit']} hold={best_local['hold']} trail={best_local['trail']} "
            f"rr={best_local['rr']} open={best_local['max_open']}",
            flush=True,
        )

    phase2.sort(key=lambda x: x["ret"], reverse=True)
    print("\n===== PHASE 2 TOP 15 =====", flush=True)
    for r in phase2[:15]:
        flag = " *** HIT ***" if r["ret"] >= TARGET_RET else ""
        print(
            f"{r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} "
            f"n={r['n']:4d} {r['entry']}/{r['filt']} {r['size']} {r['exit']} "
            f"hold={r['hold']} trail={r['trail']} rr={r['rr']} open={r['max_open']} "
            f"{r['years']}{flag}",
            flush=True,
        )

    all_rows = rows + phase2
    all_rows.sort(key=lambda x: x["ret"], reverse=True)
    best = all_rows[0] if all_rows else {}
    payload = {
        "window": [str(START), str(END)],
        "capital": CAPITAL,
        "pos_pct": POS_PCT,
        "target": TARGET_RET,
        "best": best,
        "top": all_rows[:40],
        "hits": [r for r in all_rows if r["ret"] >= TARGET_RET][:20],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {OUT}", flush=True)
    print(f"BEST {best.get('ret')}%  {best.get('entry')}/{best.get('filt')} {best.get('size')} {best.get('exit')}", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
