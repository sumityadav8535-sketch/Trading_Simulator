"""
Search Supertrend swing strategies on Nifty 200 for >= 100% in the last 1 year.

Signal on bar T close (no look-ahead). Enter T+1 open.
Shared cash, risk-based sizing (same model as Stage 2.0 / Strategy Builder).
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

from stage_analysis_v2.services.backtester import _preload_frames
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
TARGET_RET = 100.0
END = date.today()
START = END - timedelta(days=365)
COOLDOWN_DAYS = 10
MAX_NEW_PER_DAY = 5
OUT_JSON = os.path.join(ROOT, "data", "supertrend_swing_search.json")

# Filter bit flags stored on each raw signal
F_EMA_TREND = 1 << 0      # close > ema50 > ema200
F_EMA_STACK = 1 << 1      # close > ema20 > ema50
F_RSI_HEALTHY = 1 << 2    # 45 <= rsi <= 70
F_RSI_NOT_OB = 1 << 3     # rsi < 75
F_VOL = 1 << 4            # vol >= 1.2x
F_ADX20 = 1 << 5
F_ADX25 = 1 << 6
F_NOT_EXT = 1 << 7        # close <= 8% above ema20
F_ABOVE_200 = 1 << 8
F_RS = 1 << 9             # 20d ret > nifty 20d
F_MKT_ST = 1 << 10        # nifty ST(10,3) bull
F_MKT_EMA = 1 << 11       # nifty close > ema50
F_BB_MID = 1 << 12
F_NEAR_HIGH = 1 << 13     # close >= 0.98 * 20d high
F_DI_BULL = 1 << 14       # +DI > -DI


FILTER_PACKS: list[tuple[str, int]] = [
    ("none", 0),
    ("ema_trend", F_EMA_TREND),
    ("ema_stack", F_EMA_STACK),
    ("above_200", F_ABOVE_200),
    ("rsi_healthy", F_RSI_HEALTHY),
    ("rsi_not_ob", F_RSI_NOT_OB),
    ("volume", F_VOL),
    ("adx20", F_ADX20),
    ("adx25", F_ADX25),
    ("not_extended", F_NOT_EXT),
    ("rs_outperform", F_RS),
    ("mkt_st", F_MKT_ST),
    ("mkt_ema", F_MKT_EMA),
    ("bb_mid", F_BB_MID),
    ("di_bull", F_DI_BULL),
    ("trend_mkt", F_EMA_TREND | F_MKT_ST),
    ("stack_mkt", F_EMA_STACK | F_MKT_ST),
    ("trend_adx", F_EMA_TREND | F_ADX20),
    ("trend_rsi", F_EMA_TREND | F_RSI_HEALTHY),
    ("quality", F_EMA_TREND | F_RSI_HEALTHY | F_ADX20),
    ("quality_mkt", F_EMA_TREND | F_RSI_HEALTHY | F_ADX20 | F_MKT_ST),
    ("strict", F_EMA_STACK | F_MKT_ST | F_VOL | F_NOT_EXT | F_RSI_NOT_OB),
    ("breakout_st", F_EMA_TREND | F_NEAR_HIGH | F_MKT_ST),
    ("rs_trend", F_EMA_TREND | F_RS | F_MKT_EMA),
]


def _ema(s: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(s, dtype=float)
    if len(s) == 0:
        return out
    a = 2.0 / (n + 1)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = a * s[i] + (1 - a) * out[i - 1]
    return out


def _rma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    if len(s) < n:
        return out
    out[n - 1] = s[:n].mean()
    a = 1.0 / n
    for i in range(n, len(s)):
        out[i] = out[i - 1] * (1 - a) + s[i] * a
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


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14):
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(h, l, c, n)
    plus_di = 100.0 * _rma(plus_dm, n) / atr
    minus_di = 100.0 * _rma(minus_dm, n) / atr
    denom = plus_di + minus_di
    dx = np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom) * 100.0
    adx = _rma(np.nan_to_num(dx, nan=0.0), n)
    return adx, plus_di, minus_di


def supertrend_np(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 10, multiplier: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
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


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return w.dropna(subset=["close"])


class StockPack:
    __slots__ = (
        "symbol", "index", "loc", "o", "h", "l", "c", "v",
        "ema20", "ema50", "ema200", "rsi", "atr", "adx", "di_p", "di_m",
        "vol_sma", "high20", "low20", "bb_mid", "ret20",
        "st", "st_dir",
    )

    def __init__(self, symbol: str, df: pd.DataFrame, st_params: list[tuple[int, float]]):
        self.symbol = symbol
        self.index = df.index
        self.loc = {ts: i for i, ts in enumerate(df.index)}
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.ema20 = _ema(c, 20)
        self.ema50 = _ema(c, 50)
        self.ema200 = _ema(c, 200)
        self.rsi = _rsi(c, 14)
        self.atr = _atr(h, l, c, 14)
        self.adx, self.di_p, self.di_m = _adx(h, l, c, 14)
        self.vol_sma = pd.Series(v, index=df.index).rolling(20, min_periods=20).mean().to_numpy()
        self.high20 = pd.Series(h, index=df.index).rolling(20, min_periods=20).max().to_numpy()
        self.low20 = pd.Series(l, index=df.index).rolling(20, min_periods=20).min().to_numpy()
        self.bb_mid = pd.Series(c, index=df.index).rolling(20, min_periods=20).mean().to_numpy()
        self.ret20 = pd.Series(c, index=df.index).pct_change(20).to_numpy()
        self.st = {}
        self.st_dir = {}
        for p, m in st_params:
            st, d = supertrend_np(h, l, c, p, m)
            self.st[(p, m)] = st
            self.st_dir[(p, m)] = d


def _flags_at(s: StockPack, i: int, nifty_flags: int, nifty_ret20: float) -> int:
    c = s.c[i]
    flags = 0
    e20, e50, e200 = s.ema20[i], s.ema50[i], s.ema200[i]
    if np.isnan(e20) or np.isnan(e50) or np.isnan(e200):
        return 0
    if c > e50 > e200:
        flags |= F_EMA_TREND
    if c > e20 > e50:
        flags |= F_EMA_STACK
    if c > e200:
        flags |= F_ABOVE_200
    rsi = s.rsi[i]
    if not np.isnan(rsi):
        if 45.0 <= rsi <= 70.0:
            flags |= F_RSI_HEALTHY
        if rsi < 75.0:
            flags |= F_RSI_NOT_OB
    vs = s.vol_sma[i]
    if vs and not np.isnan(vs) and vs > 0 and s.v[i] >= 1.2 * vs:
        flags |= F_VOL
    adx = s.adx[i]
    if not np.isnan(adx):
        if adx >= 20:
            flags |= F_ADX20
        if adx >= 25:
            flags |= F_ADX25
    if e20 > 0 and (c - e20) / e20 * 100.0 <= 8.0:
        flags |= F_NOT_EXT
    r20 = s.ret20[i]
    if not np.isnan(r20) and not np.isnan(nifty_ret20) and r20 > nifty_ret20:
        flags |= F_RS
    bb = s.bb_mid[i]
    if not np.isnan(bb) and c > bb:
        flags |= F_BB_MID
    h20 = s.high20[i]
    if not np.isnan(h20) and h20 > 0 and c >= h20 * 0.98:
        flags |= F_NEAR_HIGH
    if not np.isnan(s.di_p[i]) and not np.isnan(s.di_m[i]) and s.di_p[i] > s.di_m[i]:
        flags |= F_DI_BULL
    flags |= nifty_flags
    return flags


def collect_daily_signals(
    packs: dict[str, StockPack],
    nifty: StockPack,
    st_key: tuple[int, float],
    entry: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dual_fast: tuple[int, float] | None = None,
) -> list[dict]:
    """Raw signals with flags + ranking score. Entry is next bar open."""
    out: list[dict] = []
    nifty_st_key = (10, 3.0)
    n_dir = nifty.st_dir.get(nifty_st_key)
    n_ema50 = nifty.ema50
    n_c = nifty.c
    n_ret = nifty.ret20

    for sym, s in packs.items():
        st = s.st.get(st_key)
        d = s.st_dir.get(st_key)
        if st is None or d is None:
            continue
        fast_d = s.st_dir.get(dual_fast) if dual_fast else None
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start or ts > end:
                continue
            ni = nifty.loc.get(ts)
            nifty_flags = 0
            nifty_r20 = np.nan
            if ni is not None:
                nifty_r20 = n_ret[ni]
                if n_dir is not None and n_dir[ni] > 0 and n_c[ni] > nifty.st[nifty_st_key][ni]:
                    nifty_flags |= F_MKT_ST
                if n_c[ni] > n_ema50[ni]:
                    nifty_flags |= F_MKT_EMA

            fire = False
            if entry == "flip":
                fire = d[i] > 0 and d[i - 1] <= 0
            elif entry == "pullback":
                if d[i] > 0 and d[i - 1] > 0 and not np.isnan(st[i]):
                    near = s.l[i] <= st[i] * 1.008
                    closed_ok = s.c[i] > st[i]
                    fire = near and closed_ok
            elif entry == "breakout":
                h20 = s.high20[i]
                h20p = s.high20[i - 1] if i else np.nan
                fire = (
                    d[i] > 0
                    and not np.isnan(h20)
                    and s.c[i] >= h20
                    and (np.isnan(h20p) or s.c[i - 1] < h20p)
                )
            elif entry == "dual_flip":
                if fast_d is None:
                    continue
                fire = fast_d[i] > 0 and fast_d[i - 1] <= 0 and d[i] > 0
            elif entry == "reclaim":
                fire = (
                    d[i] > 0
                    and s.c[i] > s.ema20[i]
                    and s.c[i - 1] <= s.ema20[i - 1]
                    and s.ema20[i] > s.ema50[i]
                )
            if not fire:
                continue

            flags = _flags_at(s, i, nifty_flags, nifty_r20)
            stop_st = float(st[i]) if not np.isnan(st[i]) else 0.0
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            low20 = float(s.low20[i]) if not np.isnan(s.low20[i]) else 0.0
            r20 = float(s.ret20[i]) if not np.isnan(s.ret20[i]) else -9.0
            out.append({
                "symbol": sym,
                "sig_i": i,
                "entry_i": i + 1,
                "sig_ts": ts,
                "entry_ts": s.index[i + 1],
                "flags": flags,
                "stop_st": stop_st,
                "atr": atr,
                "low20": low20,
                "score": r20,
                "adx": float(s.adx[i]) if not np.isnan(s.adx[i]) else 0.0,
            })
    return out


def collect_weekly_signals(
    frames: dict[str, pd.DataFrame],
    packs: dict[str, StockPack],
    nifty_df: pd.DataFrame,
    st_key: tuple[int, float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    """Weekly ST flip → enter next daily bar after that Friday."""
    p, m = st_key
    nifty_w = _to_weekly(nifty_df)
    n_h = nifty_w["high"].to_numpy(float)
    n_l = nifty_w["low"].to_numpy(float)
    n_c = nifty_w["close"].to_numpy(float)
    _, n_dir = supertrend_np(n_h, n_l, n_c, p, m)
    n_ema50 = _ema(n_c, 10)
    nifty_w_loc = {ts: i for i, ts in enumerate(nifty_w.index)}

    out: list[dict] = []
    for sym, df in frames.items():
        s = packs.get(sym)
        if s is None:
            continue
        w = _to_weekly(df)
        if len(w) < 40:
            continue
        h = w["high"].to_numpy(float)
        l = w["low"].to_numpy(float)
        c = w["close"].to_numpy(float)
        st, d = supertrend_np(h, l, c, p, m)
        for i in range(30, len(w) - 1):
            ts = w.index[i]
            if ts < start or ts > end:
                continue
            if not (d[i] > 0 and d[i - 1] <= 0):
                continue
            # next daily bar after this week
            daily_after = s.index[s.index > ts]
            if len(daily_after) == 0:
                continue
            entry_ts = daily_after[0]
            ei = s.loc.get(entry_ts)
            if ei is None or ei <= 0:
                continue
            sig_i = ei - 1  # last daily bar of the week-ish
            # flags from last daily bar <= ts
            daily_upto = s.index[s.index <= ts]
            if len(daily_upto) == 0:
                continue
            di = s.loc[daily_upto[-1]]
            ni = nifty_w_loc.get(ts)
            nifty_flags = 0
            if ni is not None:
                if n_dir[ni] > 0:
                    nifty_flags |= F_MKT_ST
                if n_c[ni] > n_ema50[ni]:
                    nifty_flags |= F_MKT_EMA
            flags = _flags_at(s, di, nifty_flags, float("nan") if True else 0.0)
            # RS vs nifty weekly not available the same way; use daily RS
            stop_st = float(st[i]) if not np.isnan(st[i]) else 0.0
            atr = float(s.atr[di]) if not np.isnan(s.atr[di]) else 0.0
            low20 = float(s.low20[di]) if not np.isnan(s.low20[di]) else 0.0
            r20 = float(s.ret20[di]) if not np.isnan(s.ret20[di]) else -9.0
            out.append({
                "symbol": sym,
                "sig_i": di,
                "entry_i": ei,
                "sig_ts": ts,
                "entry_ts": entry_ts,
                "flags": flags,
                "stop_st": stop_st,
                "atr": atr,
                "low20": low20,
                "score": r20,
                "adx": float(s.adx[di]) if not np.isnan(s.adx[di]) else 0.0,
                "weekly": True,
                "w_st": st,
                "w_dir": d,
                "w_index": w.index,
            })
    return out


def _stop_price(sig: dict, entry: float, stop_mode: str) -> float:
    if stop_mode == "st":
        raw = sig["stop_st"]
    elif stop_mode == "atr15":
        raw = entry - 1.5 * sig["atr"]
    elif stop_mode == "atr20":
        raw = entry - 2.0 * sig["atr"]
    elif stop_mode == "swing":
        raw = sig["low20"]
    else:
        raw = sig["stop_st"]
    if raw <= 0 or raw >= entry:
        # fallback: 1.5 ATR or 4%
        atr = sig["atr"]
        raw = entry - (1.5 * atr if atr > 0 else entry * 0.04)
    if raw <= 0 or raw >= entry:
        return 0.0
    return float(raw)


def simulate(
    packs: dict[str, StockPack],
    calendar: list[pd.Timestamp],
    signals: list[dict],
    *,
    capital: float,
    risk_pct: float,
    need_flags: int,
    stop_mode: str,
    exit_mode: str,
    target_rr: float,
    max_hold: int,
    cooldown: int = COOLDOWN_DAYS,
    max_new: int = MAX_NEW_PER_DAY,
    st_key: tuple[int, float] | None = None,
) -> dict[str, Any]:
    by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(capital)
    opens: dict[str, dict] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    equity_curve = [capital]
    peak_par = 0
    skipped = 0

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
            high = s.h[i]
            low = s.l[i]
            close = s.c[i]
            exit_p = None
            reason = ""

            # trail ST stop
            if exit_mode in ("st_trail", "st_trail_rr") and st_key is not None:
                st_line = s.st[st_key][i]
                if not np.isnan(st_line) and st_line > pos["stop"] and st_line < close:
                    pos["stop"] = float(st_line)

            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop_loss"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time_exit"
            elif exit_mode in ("st_flip", "st_trail", "st_trail_rr", "st_flip_rr") and st_key is not None:
                d = s.st_dir[st_key][i]
                if d <= 0:
                    exit_p, reason = close, "st_flip"

            if exit_p is None:
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym,
                "entry": pos["entry_ts"],
                "exit": ts,
                "pnl": pnl,
                "reason": reason,
                "hold": pos["hold"],
                "entry_px": pos["entry"],
                "exit_px": exit_p,
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = by_day.get(ts, [])
        if day_sigs:
            day_sigs = sorted(day_sigs, key=lambda x: x["score"], reverse=True)
            taken = 0
            for sig in day_sigs:
                if taken >= max_new:
                    break
                sym = sig["symbol"]
                if sym in opens:
                    continue
                s = packs.get(sym)
                if s is None:
                    continue
                prev = last_exit.get(sym)
                if prev is not None and (ts - prev).days < cooldown:
                    continue
                i = sig["entry_i"]
                if i >= len(s.o) or s.index[i] != ts:
                    # entry_ts should match ts
                    i = s.loc.get(ts)
                    if i is None:
                        continue
                entry = float(s.o[i])
                if entry <= 0:
                    continue
                stop = _stop_price(sig, entry, stop_mode)
                if stop <= 0 or stop >= entry:
                    skipped += 1
                    continue
                risk = entry - stop
                if risk / entry > 0.12:
                    # skip absurdly wide stops (>12%)
                    skipped += 1
                    continue
                equity_now = cash + sum(p["notional"] for p in opens.values())
                if cash <= 0 or equity_now <= 0:
                    skipped += 1
                    continue
                ps = calculate_position_size(equity_now, risk_pct, entry, stop)
                qty = min(int(ps.quantity), int(cash // entry) if entry else 0)
                if qty <= 0:
                    skipped += 1
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    skipped += 1
                    continue
                target = 0.0
                if exit_mode in ("rr", "st_flip_rr", "st_trail_rr") and target_rr > 0:
                    target = round(entry + risk * target_rr, 2)
                cash -= notional
                opens[sym] = {
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "qty": qty,
                    "notional": notional,
                    "hold": 0,
                    "entry_ts": ts,
                }
                peak_par = max(peak_par, len(opens))
                taken += 1

                # same-bar stop/target
                if s.l[i] <= stop:
                    pnl = (stop - entry) * qty
                    cash += notional + pnl
                    trades.append({
                        "symbol": sym, "entry": ts, "exit": ts, "pnl": pnl,
                        "reason": "stop_loss", "hold": 0, "entry_px": entry, "exit_px": stop,
                    })
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif target and s.h[i] >= target:
                    pnl = (target - entry) * qty
                    cash += notional + pnl
                    trades.append({
                        "symbol": sym, "entry": ts, "exit": ts, "pnl": pnl,
                        "reason": "target", "hold": 0, "entry_px": entry, "exit_px": target,
                    })
                    last_exit[sym] = ts
                    opens.pop(sym, None)

        equity_curve.append(cash + sum(p["notional"] for p in opens.values()))

    if opens and calendar:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            hist_i = s.loc.get(last_ts)
            if hist_i is None:
                # last available
                hist_i = len(s.c) - 1
            close = float(s.c[hist_i])
            pnl = (close - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "entry": pos["entry_ts"], "exit": last_ts, "pnl": pnl,
                "reason": "eod_force", "hold": pos["hold"], "entry_px": pos["entry"], "exit_px": close,
            })
        opens.clear()

    final = cash
    ret = (final - capital) / capital * 100.0
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100.0 if trades else 0.0
    pf = gp / gl if trades else 0.0
    peak = capital
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)
    avg_hold = sum(t["hold"] for t in trades) / len(trades) if trades else 0.0

    # split halves for robustness
    mid = start_mid = None
    if calendar:
        mid = calendar[len(calendar) // 2]
        h1 = sum(t["pnl"] for t in trades if t["exit"] <= mid)
        h2 = sum(t["pnl"] for t in trades if t["exit"] > mid)
    else:
        h1 = h2 = 0.0

    return {
        "ret": round(ret, 2),
        "wr": round(wr, 2),
        "pf": round(pf, 2),
        "dd": round(max_dd, 2),
        "n": len(trades),
        "wins": len(wins),
        "final": round(final, 2),
        "peak_par": peak_par,
        "skipped": skipped,
        "avg_hold": round(avg_hold, 1),
        "h1_pnl": round(h1, 0),
        "h2_pnl": round(h2, 0),
        "trades": trades,
        "exits": {k: sum(1 for t in trades if t["reason"] == k)
                  for k in {t["reason"] for t in trades}},
    }


def _score(r: dict) -> tuple:
    """Prefer >=100% with controlled DD, then PF, then return."""
    hit = 1 if r["ret"] >= TARGET_RET and r["n"] >= 15 and r["dd"] <= 45 else 0
    almost = 1 if r["ret"] >= 70 and r["n"] >= 15 else 0
    dd_pen = -r["dd"]
    return (hit, almost, r["pf"], dd_pen, r["ret"], r["wr"])


def main() -> None:
    t0 = time.time()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"Supertrend swing search | {START} → {END} | capital ₹{CAPITAL:,.0f}", flush=True)
    print(f"Universe Nifty200: {len(symbols)} symbols", flush=True)

    frames = _preload_frames(symbols)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    print(f"Loaded {len(frames)} stocks + NIFTY50 bars={len(nifty_df)}", flush=True)

    st_params: list[tuple[int, float]] = [
        (7, 3.0),
        (10, 2.0),
        (10, 3.0),
        (10, 4.0),
        (14, 2.0),
        (14, 3.0),
        (21, 3.0),
    ]
    print("Computing indicators + Supertrend…", flush=True)
    packs: dict[str, StockPack] = {}
    for i, (sym, df) in enumerate(frames.items()):
        packs[sym] = StockPack(sym, df, st_params)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)
    nifty = StockPack("NIFTY50", nifty_df, st_params + [(10, 3.0)])

    st_ts, et_ts = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts
        for df in frames.values()
        for ts in df.index[(df.index >= st_ts) & (df.index <= et_ts)].tolist()
    })
    print(f"Calendar days={len(calendar)}", flush=True)

    # Buy & hold nifty
    n0 = nifty.loc.get(calendar[0]) if calendar else None
    n1 = nifty.loc.get(calendar[-1]) if calendar else None
    if n0 is None:
        n0 = next((nifty.loc[ts] for ts in calendar if ts in nifty.loc), None)
    if n1 is None:
        n1 = nifty.loc[calendar[-1]] if calendar[-1] in nifty.loc else len(nifty.c) - 1
    bh = 0.0
    if n0 is not None and n1 is not None and nifty.c[n0] > 0:
        bh = (nifty.c[n1] / nifty.c[n0] - 1) * 100.0
    print(f"Nifty50 buy&hold: {bh:+.1f}%", flush=True)

    entries = ["flip", "pullback", "breakout", "dual_flip", "reclaim"]
    results: list[dict] = []

    print("\n=== PHASE 1: probe families (risk 3%, ST trail, hold 60) ===", flush=True)
    raw_cache: dict[tuple, list] = {}

    for st_key in st_params:
        for entry in entries:
            cache_key = (st_key, entry)
            dual = (7, 3.0) if entry == "dual_flip" else None
            if entry == "dual_flip" and st_key == (7, 3.0):
                continue  # dual needs slower ST as the trend filter
            raw = collect_daily_signals(
                packs, nifty, st_key, entry, st_ts, et_ts, dual_fast=dual,
            )
            raw_cache[cache_key] = raw
            print(f"  ST{st_key[0]},{st_key[1]:g} {entry:<10} raw_sig={len(raw)}", flush=True)

            for fname, need in FILTER_PACKS:
                if len(raw) < 8:
                    continue
                r = simulate(
                    packs, calendar, raw,
                    capital=CAPITAL, risk_pct=3.0, need_flags=need,
                    stop_mode="st", exit_mode="st_trail", target_rr=0,
                    max_hold=60, st_key=st_key,
                )
                r.update({
                    "st": f"{st_key[0]},{st_key[1]:g}",
                    "entry": entry,
                    "filter": fname,
                    "tf": "daily",
                    "risk": 3.0,
                    "stop": "st",
                    "exit": "st_trail",
                    "hold": 60,
                    "rr": 0,
                    "n_sig": sum(1 for s in raw if (s["flags"] & need) == need) if need else len(raw),
                })
                results.append(r)

    # Weekly ST flips
    print("\n=== Weekly Supertrend flips ===", flush=True)
    for st_key in [(7, 3.0), (10, 3.0), (14, 3.0), (10, 2.0)]:
        raw = collect_weekly_signals(frames, packs, nifty_df, st_key, st_ts, et_ts)
        raw_cache[("weekly", st_key)] = raw
        print(f"  weekly ST{st_key[0]},{st_key[1]:g} raw_sig={len(raw)}", flush=True)
        for fname, need in FILTER_PACKS:
            if len(raw) < 8:
                continue
            r = simulate(
                packs, calendar, raw,
                capital=CAPITAL, risk_pct=3.0, need_flags=need,
                stop_mode="st", exit_mode="st_flip", target_rr=0,
                max_hold=120, st_key=None,  # weekly exit handled as time/stop; flip checked on daily ST of same params
            )
            # weekly: also try daily ST trail of same params
            r.update({
                "st": f"{st_key[0]},{st_key[1]:g}",
                "entry": "weekly_flip",
                "filter": fname,
                "tf": "weekly",
                "risk": 3.0,
                "stop": "st",
                "exit": "st_flip",
                "hold": 120,
                "rr": 0,
                "n_sig": sum(1 for s in raw if (s["flags"] & need) == need) if need else len(raw),
            })
            results.append(r)
            r2 = simulate(
                packs, calendar, raw,
                capital=CAPITAL, risk_pct=3.0, need_flags=need,
                stop_mode="st", exit_mode="st_trail", target_rr=0,
                max_hold=120, st_key=st_key,
            )
            r2.update({
                "st": f"{st_key[0]},{st_key[1]:g}",
                "entry": "weekly_flip",
                "filter": fname,
                "tf": "weekly",
                "risk": 3.0,
                "stop": "st",
                "exit": "st_trail",
                "hold": 120,
                "rr": 0,
                "n_sig": r["n_sig"],
            })
            results.append(r2)

    def show_top(title: str, rows: list[dict], k: int = 12):
        print(f"\n{title}", flush=True)
        rows = [x for x in rows if x["n"] >= 10]
        rows = sorted(rows, key=lambda x: x["ret"], reverse=True)
        for r in rows[:k]:
            print(
                f"  {r['tf']:6} ST{r['st']:<7} {r['entry']:<12} {r['filter']:<16} "
                f"n={r['n']:3d} WR={r['wr']:5.1f}% ret={r['ret']:+7.1f}% "
                f"PF={r['pf']:.2f} DD={r['dd']:5.1f}% hold={r['avg_hold']:.0f} "
                f"h1={r['h1_pnl']:+.0f} h2={r['h2_pnl']:+.0f}",
                flush=True,
            )
        return rows

    phase1 = [r for r in results if r["n"] >= 10]
    show_top("=== PHASE 1 TOP BY RETURN (n>=10) ===", phase1, 15)
    hits = [r for r in phase1 if r["ret"] >= TARGET_RET]
    print(f"\nHit >=100%: {len(hits)}  |  >=70%: {sum(1 for r in phase1 if r['ret']>=70)}", flush=True)

    # PHASE 2: refine top families
    print("\n=== PHASE 2: refine top 12 families across risk/exit/stop/hold ===", flush=True)
    families = []
    seen_f = set()
    for r in sorted(phase1, key=lambda x: x["ret"], reverse=True):
        key = (r["tf"], r["st"], r["entry"], r["filter"])
        if key in seen_f:
            continue
        seen_f.add(key)
        families.append(r)
        if len(families) >= 12:
            break
    # always include a few well-known combos even if not top
    extra_keys = [
        ("daily", "10,3", "flip", "trend_mkt"),
        ("daily", "10,3", "flip", "quality"),
        ("daily", "10,2", "flip", "mkt_st"),
        ("daily", "7,3", "flip", "adx25"),
        ("daily", "10,3", "pullback", "ema_trend"),
        ("daily", "10,3", "breakout", "mkt_st"),
        ("weekly", "10,3", "weekly_flip", "mkt_st"),
    ]
    have = {(f["tf"], f["st"], f["entry"], f["filter"]) for f in families}
    lookup = {(r["tf"], r["st"], r["entry"], r["filter"]): r for r in phase1}
    for k in extra_keys:
        if k not in have and k in lookup:
            families.append(lookup[k])

    refined: list[dict] = []
    risk_grid = [2.0, 3.0, 4.0, 5.0, 6.0]
    stop_grid = ["st", "atr15", "atr20", "swing"]
    exit_grid = [
        ("st_trail", 0, 40),
        ("st_trail", 0, 60),
        ("st_trail", 0, 90),
        ("st_flip", 0, 60),
        ("st_trail_rr", 2.0, 60),
        ("st_trail_rr", 3.0, 90),
        ("rr", 2.5, 40),
        ("rr", 3.0, 60),
        ("st_flip_rr", 2.5, 60),
    ]

    def parse_st(s: str) -> tuple[int, float]:
        a, b = s.split(",")
        return int(a), float(b)

    for fam in families:
        st_key = parse_st(fam["st"])
        if fam["tf"] == "weekly":
            raw = raw_cache.get(("weekly", st_key), [])
        else:
            raw = raw_cache.get((st_key, fam["entry"]), [])
        if not raw:
            continue
        need = dict(FILTER_PACKS).get(fam["filter"], 0)
        print(f"  refine {fam['tf']} ST{fam['st']} {fam['entry']} {fam['filter']} raw={len(raw)}", flush=True)
        for risk in risk_grid:
            for stop_m in stop_grid:
                for exit_m, rr, hold in exit_grid:
                    r = simulate(
                        packs, calendar, raw,
                        capital=CAPITAL, risk_pct=risk, need_flags=need,
                        stop_mode=stop_m, exit_mode=exit_m, target_rr=rr,
                        max_hold=hold, st_key=st_key if fam["tf"] == "daily" or "trail" in exit_m else (
                            st_key if exit_m.startswith("st") else None
                        ),
                    )
                    r.update({
                        "st": fam["st"],
                        "entry": fam["entry"],
                        "filter": fam["filter"],
                        "tf": fam["tf"],
                        "risk": risk,
                        "stop": stop_m,
                        "exit": exit_m,
                        "hold": hold,
                        "rr": rr,
                        "n_sig": fam.get("n_sig", 0),
                    })
                    refined.append(r)

    all_rows = results + refined
    usable = [r for r in all_rows if r["n"] >= 12]
    show_top("=== ALL-TIME TOP RETURN ===", usable, 20)
    hits = [r for r in usable if r["ret"] >= TARGET_RET]
    print(f"\nConfigs >=100% with n>=12: {len(hits)}", flush=True)
    if hits:
        hits_sorted = sorted(hits, key=_score, reverse=True)
        print("\n=== >=100% ranked (prefer lower DD, higher PF) ===", flush=True)
        for r in hits_sorted[:15]:
            print(
                f"  ST{r['st']:<7} {r['entry']:<12} {r['filter']:<16} "
                f"risk={r['risk']:g}% stop={r['stop']:<5} exit={r['exit']:<12} hold={r['hold']:<3} "
                f"n={r['n']:3d} WR={r['wr']:5.1f}% ret={r['ret']:+7.1f}% "
                f"PF={r['pf']:.2f} DD={r['dd']:5.1f}%",
                flush=True,
            )
        best = hits_sorted[0]
    else:
        print("No config hit 100%. Showing best overall.", flush=True)
        best = max(usable, key=_score) if usable else None

    # Best at risk<=3% (more realistic)
    realistic = [r for r in usable if r["risk"] <= 3.0]
    best_real = None
    if realistic:
        print("\n=== BEST WITH RISK ≤ 3% ===", flush=True)
        for r in sorted(realistic, key=lambda x: x["ret"], reverse=True)[:8]:
            print(
                f"  ST{r['st']:<7} {r['entry']:<12} {r['filter']:<16} "
                f"risk={r['risk']:g}% stop={r['stop']:<5} exit={r['exit']:<12} "
                f"n={r['n']:3d} WR={r['wr']:5.1f}% ret={r['ret']:+7.1f}% PF={r['pf']:.2f} DD={r['dd']:5.1f}%",
                flush=True,
            )
        real_hits = [r for r in realistic if r["ret"] >= TARGET_RET]
        best_real = max(real_hits, key=_score) if real_hits else max(realistic, key=_score)

    # Re-run best with trades for monthly + samples
    def describe(r: dict) -> str:
        return (
            f"ST({r['st']}) {r['tf']} {r['entry']} + {r['filter']} | "
            f"risk {r['risk']:g}% stop={r['stop']} exit={r['exit']} "
            f"hold={r['hold']} RR={r['rr']}"
        )

    def rerun(r: dict) -> dict:
        st_key = parse_st(r["st"])
        if r["tf"] == "weekly":
            raw = raw_cache.get(("weekly", st_key), [])
        else:
            raw = raw_cache.get((st_key, r["entry"]), [])
        need = dict(FILTER_PACKS).get(r["filter"], 0)
        return simulate(
            packs, calendar, raw,
            capital=CAPITAL, risk_pct=r["risk"], need_flags=need,
            stop_mode=r["stop"], exit_mode=r["exit"], target_rr=r["rr"],
            max_hold=r["hold"], st_key=st_key if str(r["exit"]).startswith("st") else st_key,
        )

    winners = []
    if best:
        winners.append(("BEST_OVERALL", best))
    if best_real and (not best or best_real is not best):
        winners.append(("BEST_RISK_LE_3", best_real))

    summary = {
        "window": [str(START), str(END)],
        "capital": CAPITAL,
        "nifty_bh": round(bh, 2),
        "n_configs": len(all_rows),
        "n_hit_100": len(hits),
        "elapsed_sec": round(time.time() - t0, 1),
        "winners": [],
    }

    for label, cfg in winners:
        full = rerun(cfg)
        monthly: dict[str, float] = defaultdict(float)
        for t in full["trades"]:
            monthly[str(t["exit"])[:7]] += t["pnl"]
        print(f"\n*** {label} ***", flush=True)
        print(f"  {describe(cfg)}", flush=True)
        print(
            f"  n={full['n']} WR={full['wr']}% ret={full['ret']}% PF={full['pf']} "
            f"DD={full['dd']}% peak_par={full['peak_par']} avg_hold={full['avg_hold']}",
            flush=True,
        )
        print(f"  exits={full['exits']}", flush=True)
        print(f"  half-year PnL: H1={full['h1_pnl']:+.0f}  H2={full['h2_pnl']:+.0f}", flush=True)
        print("  monthly:", flush=True)
        for m in sorted(monthly):
            print(f"    {m}: {monthly[m]:+,.0f}", flush=True)
        top_w = sorted(full["trades"], key=lambda t: t["pnl"], reverse=True)[:5]
        top_l = sorted(full["trades"], key=lambda t: t["pnl"])[:5]
        print("  top winners:", flush=True)
        for t in top_w:
            print(
                f"    {t['symbol']:<12} {str(t['entry'])[:10]} → {str(t['exit'])[:10]} "
                f"pnl={t['pnl']:+,.0f} {t['reason']}",
                flush=True,
            )
        print("  worst losers:", flush=True)
        for t in top_l:
            print(
                f"    {t['symbol']:<12} {str(t['entry'])[:10]} → {str(t['exit'])[:10]} "
                f"pnl={t['pnl']:+,.0f} {t['reason']}",
                flush=True,
            )
        # strip trades for json size
        slim = {k: v for k, v in {**cfg, **{kk: full[kk] for kk in
                ("ret", "wr", "pf", "dd", "n", "wins", "final", "peak_par", "avg_hold",
                 "h1_pnl", "h2_pnl", "exits")}}.items() if k != "trades"}
        slim["label"] = label
        slim["rules"] = describe(cfg)
        slim["monthly"] = {m: round(v, 0) for m, v in monthly.items()}
        slim["sample_wins"] = [
            {"symbol": t["symbol"], "pnl": round(t["pnl"], 0), "entry": str(t["entry"])[:10],
             "exit": str(t["exit"])[:10]}
            for t in top_w
        ]
        summary["winners"].append(slim)

    # persist top 40 without trades
    top40 = sorted(usable, key=lambda x: x["ret"], reverse=True)[:40]
    summary["top40"] = [
        {k: v for k, v in r.items() if k != "trades"} for r in top40
    ]
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved {OUT_JSON} | elapsed {time.time()-t0:.0f}s | configs={len(all_rows)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
