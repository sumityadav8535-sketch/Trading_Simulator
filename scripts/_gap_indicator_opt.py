"""
Intraday filters for the Nifty 200 gap-open book.

Decisions use only what is known at the 9:30 open: the completed 9:15–9:25
opening range, VWAP, relative volume, yesterday's RSI / trend, and the Nifty gap.
Costs and stop-first fills match the live gap simulator.

    python scripts/_gap_indicator_opt.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, time as dtime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    COST,
    add_indicators,
    nifty200_symbols,
)
from scripts.intraday_5m_hunt import DATA_5M, load_frames  # noqa: E402
from trading.services.indicators import _rsi  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_indicator_opt.json"
LEVERAGE = 5.0
DAYVOL = 8.66  # sqrt(75) five-minute bars, gap measured in day-volatility
EOD_MIN = 15 * 60 + 10


def session_prior_close_at(df: pd.DataFrame, hour: int = 15, minute: int = 10) -> pd.Series:
    """Previous session's hh:mm close, skipping a closing-auction spike."""
    if df is None or df.empty or "session" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"].replace(0, np.nan)
    rng = (df["high"] - df["low"]) / close
    auction = (rng >= 0.012) & (df["close"] >= df["high"] * 0.997)
    work = df.loc[~auction.fillna(False)]
    if work.empty:
        return pd.Series(dtype=float)
    times = work.index
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert("Asia/Kolkata")
    hm = np.asarray(times.hour) * 60 + np.asarray(times.minute)
    cutoff = hour * 60 + minute
    exact = work.loc[hm == cutoff].groupby("session", sort=True)["close"].last()
    before = work.loc[hm <= cutoff].groupby("session", sort=True)["close"].last()
    return exact.combine_first(before).sort_index().shift(1)


def nifty_context() -> dict[date, dict]:
    path = DATA_5M / "_NIFTY100_INDEX.pkl"
    if not path.exists():
        return {}
    df = pd.read_pickle(path)
    if df is None or df.empty:
        return {}
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df["session"] = df.index.date
    prior = session_prior_close_at(df, 15, 10)
    out: dict[date, dict] = {}
    for sess, g in df.groupby("session", sort=True):
        mins = g.index.hour * 60 + g.index.minute
        i915 = np.where(mins == 555)[0]
        i925 = np.where(mins == 565)[0]
        if len(i915) == 0:
            continue
        pdc = prior.get(sess) if len(prior) else None
        try:
            pdc_f = float(pdc)
        except (TypeError, ValueError):
            pdc_f = 0.0
        o = float(g["open"].iloc[int(i915[0])])
        gap = (o / pdc_f - 1.0) if pdc_f > 0 else np.nan
        or_ret = np.nan
        if len(i925):
            c = float(g["close"].iloc[int(i925[0])])
            or_ret = c / o - 1.0 if o else np.nan
        out[sess] = {"gap": gap, "or": or_ret}
    return out


def daily_asof(symbol: str) -> dict:
    d = load_price_dataframe(symbol)
    if d.empty or len(d) < 20:
        return {}
    close = d["close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = _rsi(close, 14)
    htf_up = (close > sma20) & (sma20 > sma50.fillna(sma20))
    htf_dn = close < sma20
    ords = np.array([t.toordinal() for t in d.index])
    return {
        "ords": ords,
        "rsi": rsi.to_numpy(dtype=float),
        "htf_up": htf_up.fillna(False).to_numpy(dtype=bool),
        "htf_dn": htf_dn.fillna(False).to_numpy(dtype=bool),
        "pdl": d["low"].astype(float).to_numpy(),
        "pdh": d["high"].astype(float).to_numpy(),
    }


def _asof_i(table: dict, sess: date) -> int:
    if not table:
        return -1
    return int(np.searchsorted(table["ords"], sess.toordinal(), side="left") - 1)


def build_book(stocks, nifty):
    """One pass per symbol. ATR is yesterday's last 5-minute ATR."""
    rows = []
    paths = []
    calendar = set()
    n_sym = 0
    for sym, raw in stocks.items():
        if sym.startswith("_") or raw is None or raw.empty:
            continue
        df = add_indicators(raw)
        if len(df) < 80 or "session" not in df.columns:
            continue
        n_sym += 1
        calendar.update(pd.unique(df["session"]))
        prior = session_prior_close_at(df, 15, 10)
        daily = daily_asof(sym)
        prev_atr = 0.0
        prev_adx = np.nan
        or_hist: list[float] = []
        for sess, g in df.groupby("session", sort=True):
            if len(g) < 8:
                continue
            mins = (g.index.hour * 60 + g.index.minute).to_numpy()
            end_ix = np.where(mins <= EOD_MIN)[0]
            if len(end_ix) == 0:
                continue
            end = int(end_ix[-1])
            atr_today_close = float(g["atr"].iloc[end]) if np.isfinite(g["atr"].iloc[end]) else prev_atr
            adx_today_close = float(g["adx"].iloc[end]) if np.isfinite(g["adx"].iloc[end]) else prev_adx

            i915a = np.where(mins == 555)[0]
            i925a = np.where(mins == 565)[0]
            i930a = np.where(mins == 570)[0]
            or_slice = (mins >= 555) & (mins <= 565)
            or_vol = float(g["volume"].to_numpy()[or_slice].sum()) if or_slice.any() else 0.0
            med = float(np.median(or_hist[-10:])) if or_hist else 0.0
            or_hist.append(or_vol)

            if len(i915a) == 0 or len(i925a) == 0 or len(i930a) == 0:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            i915, i925, i930 = int(i915a[0]), int(i925a[0]), int(i930a[0])
            if i930 > end:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            o915 = float(g["open"].iloc[i915])
            pdc = prior.get(sess) if len(prior) else None
            try:
                pdc_f = float(pdc)
            except (TypeError, ValueError):
                pdc_f = 0.0
            if pdc_f <= 0 or o915 < 60:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            gap = o915 / pdc_f - 1.0
            if abs(gap) < 0.004 or abs(gap) > 0.10:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            o930 = float(g["open"].iloc[i930])
            if o930 < 60:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            highs = g["high"].to_numpy(dtype=float)
            lows = g["low"].to_numpy(dtype=float)
            or_high = float(highs[or_slice].max())
            or_low = float(lows[or_slice].min())
            or_close = float(g["close"].iloc[i925])
            span = or_high - or_low
            di = _asof_i(daily, sess)
            vwap925 = g["vwap"].iloc[i925]
            vwap930 = g["vwap"].iloc[i930]
            rsi5 = g["rsi"].iloc[i925]
            c930 = float(g["close"].iloc[i930])
            i935a = np.where(mins == 575)[0]
            o935 = float(g["open"].iloc[int(i935a[0])]) if len(i935a) else np.nan
            off935 = int(i935a[0] - i930) if len(i935a) and int(i935a[0]) >= i930 else -1
            if prev_atr <= 0:
                prev_atr, prev_adx = atr_today_close, adx_today_close
                continue
            atr = prev_atr
            sl = slice(i930, end + 1)
            nx = nifty.get(sess) or {}
            rows.append({
                "symbol": sym,
                "session": sess,
                "gap": gap,
                "pdc": pdc_f,
                "atr": atr,
                "adx": prev_adx,
                "open915": o915,
                "close915": float(g["close"].iloc[i915]),
                "open930": o930,
                "open935": o935,
                "or_high": or_high,
                "or_low": or_low,
                "or_close": or_close,
                "or_loc": ((or_close - or_low) / span) if span > 0 else 0.5,
                "or_bull": or_close >= o915,
                "or_above_vwap": bool(pd.notna(vwap925) and or_close >= float(vwap925)),
                "or_rvol": (or_vol / med) if med > 0 else np.nan,
                "rsi": float(daily["rsi"][di]) if di >= 0 else np.nan,
                "rsi5": float(rsi5) if pd.notna(rsi5) else np.nan,
                "htf_up": bool(daily["htf_up"][di]) if di >= 0 else False,
                "htf_dn": bool(daily["htf_dn"][di]) if di >= 0 else False,
                "pdl": float(daily["pdl"][di]) if di >= 0 else np.nan,
                "above_pdl": bool(di >= 0 and o915 >= float(daily["pdl"][di])),
                "nifty_gap": float(nx.get("gap", np.nan)),
                "nifty_or": float(nx.get("or", np.nan)),
                "c930_bull": c930 > o930,
                "c930_above_vwap": bool(pd.notna(vwap930) and c930 >= float(vwap930)),
                "bounce": o930 >= float(g["close"].iloc[i915]),
                "off935": off935,
                "weekday": int(sess.weekday()),
                "gap_dayvol": abs(gap) / ((atr / o930) * DAYVOL) if atr > 0 else np.nan,
            })
            paths.append({
                "open": g["open"].to_numpy(dtype=float)[sl].copy(),
                "high": highs[sl].copy(),
                "low": lows[sl].copy(),
                "close": g["close"].to_numpy(dtype=float)[sl].copy(),
                "minute": mins[sl].copy(),
            })
            prev_atr, prev_adx = atr_today_close, adx_today_close
        if n_sym % 40 == 0:
            print(f"  features {n_sym}", flush=True)
    feat = pd.DataFrame(rows)
    print(f"  events {len(feat)} from {n_sym} names", flush=True)
    return feat, paths, sorted(d for d in calendar if d is not None)


# Exit specs. side is chosen by the book, not the spec.
EXITS = {
    "p05_a15": dict(tp="pct", pct=0.005, stop="atr", sl=1.5),
    "p06_a10": dict(tp="pct", pct=0.006, stop="atr", sl=1.0),
    "p08_a12": dict(tp="pct", pct=0.008, stop="atr", sl=1.2),
    "p10_a15": dict(tp="pct", pct=0.010, stop="atr", sl=1.5),
    "p05_t1000": dict(tp="pct", pct=0.005, stop="atr", sl=1.2, time=10 * 60),
    "p06_t1030": dict(tp="pct", pct=0.006, stop="atr", sl=1.5, time=10 * 60 + 30),
    "half_a12": dict(tp="fill", frac=0.5, stop="atr", sl=1.2),
    "qtr_a10": dict(tp="fill", frac=0.25, stop="atr", sl=1.0),
    "p08_or": dict(tp="pct", pct=0.008, stop="or", sl=0.0),
    "p06_or_t1030": dict(tp="pct", pct=0.006, stop="or", sl=0.0, time=10 * 60 + 30),
    "p08_be": dict(tp="pct", pct=0.008, stop="atr", sl=1.2, be=0.0035),
    "r15_or": dict(tp="r", r=1.5, stop="or", sl=0.0),
    "p07_hold": dict(tp="pct", pct=0.007, stop="atr", sl=0.8, time=10 * 60 + 30, need=0.003),
}


def _levels(row, spec, side: str, entry: float):
    atr = float(row["atr"])
    if spec["stop"] == "atr":
        dist = spec["sl"] * atr
    else:
        dist = (entry - float(row["or_low"])) if side == "long" else (float(row["or_high"]) - entry)
    if dist <= 0:
        return None
    stop = entry - dist if side == "long" else entry + dist
    if spec["tp"] == "pct":
        target = entry * (1 + spec["pct"]) if side == "long" else entry * (1 - spec["pct"])
    elif spec["tp"] == "fill":
        target = entry + spec["frac"] * (float(row["pdc"]) - entry)
    else:
        target = entry + spec["r"] * dist if side == "long" else entry - spec["r"] * dist
    return stop, target


def _walk(path, start, entry, stop, target, side, spec):
    op, hi, lo, cl, minute = path["open"], path["high"], path["low"], path["close"], path["minute"]
    n = len(op)
    if start < 0 or start >= n:
        return None
    cost = COST
    time_cut = spec.get("time")
    be_pct = spec.get("be")
    need = spec.get("need")
    if side == "long":
        entry_f = entry * (1 + cost)
        if not (target > entry_f > stop):
            return None
        risk = entry_f - stop
        be_px = entry * (1 + be_pct) if be_pct else 0.0
        stop_now = stop
        armed = False
        peak = entry
        held = False
        for i in range(start, n):
            m = int(minute[i])
            if time_cut and not held and m >= time_cut:
                if need and peak >= entry * (1 + need):
                    held = True
                else:
                    pnl = op[i] * (1 - cost) - entry_f
                    return pnl, risk, entry_f, "time"
            h = hi[i]
            l = lo[i]
            if l <= stop_now:
                pnl = stop_now * (1 - cost) - entry_f
                return pnl, risk, entry_f, "be" if armed else "sl"
            if h >= target:
                pnl = target * (1 - cost) - entry_f
                return pnl, risk, entry_f, "tp"
            if h > peak:
                peak = h
            if be_px and not armed and h >= be_px:
                armed = True
                stop_now = entry
            if m >= EOD_MIN:
                pnl = cl[i] * (1 - cost) - entry_f
                return pnl, risk, entry_f, "eod"
        pnl = cl[-1] * (1 - cost) - entry_f
        return pnl, risk, entry_f, "eod"
    entry_f = entry * (1 - cost)
    if not (target < entry_f < stop):
        return None
    risk = stop - entry_f
    be_px = entry * (1 - be_pct) if be_pct else 0.0
    stop_now = stop
    armed = False
    trough = entry
    held = False
    for i in range(start, n):
        m = int(minute[i])
        if time_cut and not held and m >= time_cut:
            if need and trough <= entry * (1 - need):
                held = True
            else:
                pnl = entry_f - op[i] * (1 + cost)
                return pnl, risk, entry_f, "time"
        h = hi[i]
        l = lo[i]
        if h >= stop_now:
            pnl = entry_f - stop_now * (1 + cost)
            return pnl, risk, entry_f, "be" if armed else "sl"
        if l <= target:
            pnl = entry_f - target * (1 + cost)
            return pnl, risk, entry_f, "tp"
        if l < trough:
            trough = l
        if be_px and not armed and l <= be_px:
            armed = True
            stop_now = entry
        if m >= EOD_MIN:
            pnl = entry_f - cl[i] * (1 + cost)
            return pnl, risk, entry_f, "eod"
    pnl = entry_f - cl[-1] * (1 + cost)
    return pnl, risk, entry_f, "eod"


def eval_book(feat, paths, mask, exit_name, calendar, oos_cut, *, side="long", entry_at="930",
              risk_pct=8.0, max_deploy=0.5, top_k=0, cap_total=False, name=""):
    spec = EXITS[exit_name]
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    by_day: dict = {}
    entry_col = "open930" if entry_at == "930" else "open935"
    off_col = "off935"
    for i in idx:
        row = feat.iloc[int(i)]
        entry = float(row[entry_col]) if pd.notna(row[entry_col]) else 0.0
        if entry <= 0:
            continue
        start = 0 if entry_at == "930" else int(row[off_col])
        if start < 0:
            continue
        levels = _levels(row, spec, side, entry)
        if levels is None:
            continue
        walked = _walk(paths[int(i)], start, entry, levels[0], levels[1], side, spec)
        if walked is None:
            continue
        pnl_ps, risk, entry_f, reason = walked
        by_day.setdefault(row["session"], []).append({
            "i": int(i),
            "score": abs(float(row["gap"])),
            "pnl_ps": pnl_ps,
            "risk": risk,
            "entry_f": entry_f,
            "reason": reason,
            "symbol": row["symbol"],
            "weekday": int(row["weekday"]),
        })

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_lev = 0.0
    day_pnl = {}
    trades = []
    reasons: dict[str, int] = {}
    for sess in calendar:
        cands = by_day.get(sess) or []
        cands.sort(key=lambda r: -r["score"])
        if top_k:
            cands = cands[: int(top_k)]
        used = 0.0
        bp = equity * LEVERAGE
        day_p = 0.0
        open_eq = equity
        fills = 0
        for c in cands:
            qty = int((equity * risk_pct / 100.0) / c["risk"]) if c["risk"] > 0 else 0
            cap = int((bp * max_deploy) / c["entry_f"]) if c["entry_f"] > 0 else 0
            qty = max(min(qty, cap), 0)
            if cap_total:
                room = bp - used
                qty = min(qty, int(room / c["entry_f"]) if c["entry_f"] > 0 else 0)
            if qty <= 0:
                continue
            used += qty * c["entry_f"]
            pnl = c["pnl_ps"] * qty
            day_p += pnl
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
            fills += 1
            trades.append({
                "session": str(sess), "symbol": c["symbol"], "pnl": round(pnl, 2),
                "reason": c["reason"], "weekday": c["weekday"], "qty": qty,
            })
        if open_eq > 0 and fills:
            max_lev = max(max_lev, used / open_eq)
        equity += day_p
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
        if fills:
            day_pnl[sess] = day_p

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl"] > 0] if n else tdf
    losses = tdf[tdf["pnl"] <= 0] if n else tdf
    gp = float(wins["pnl"].sum()) if n else 0.0
    gl = float(abs(losses["pnl"].sum())) if n else 0.0
    net = equity - CAPITAL
    n_days = len(calendar) or 1
    active_pnls = list(day_pnl.values())
    months = {}
    if n:
        tdf = tdf.copy()
        tdf["month"] = tdf["session"].str.slice(0, 7)
        for m, g in tdf.groupby("month"):
            months[m] = {
                "trades": int(len(g)),
                "pnl": round(float(g["pnl"].sum()), 2),
                "wr": round(float((g["pnl"] > 0).mean() * 100), 1),
            }
    oos_pnl = 0.0
    oos_days = 0
    for d, v in day_pnl.items():
        if d >= oos_cut:
            oos_pnl += v
            oos_days += 1
    ge5 = sum(1 for v in day_pnl.values() if v >= 5000)
    return {
        "name": name,
        "exit": exit_name,
        "side": side,
        "entry": entry_at,
        "trades": n,
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0),
        "net_pnl": round(net, 2),
        "avg_day": round(net / n_days, 2),
        "avg_trade_day": round(float(np.mean(active_pnls)), 2) if active_pnls else 0.0,
        "trade_days": len(day_pnl),
        "days_ge_5k": int(ge5),
        "worst_day": round(min(day_pnl.values()), 2) if day_pnl else 0.0,
        "best_day": round(max(day_pnl.values()), 2) if day_pnl else 0.0,
        "max_dd_pct": round(max_dd, 2),
        "max_leverage": round(max_lev, 2),
        "oos_pnl": round(oos_pnl, 2),
        "oos_days": oos_days,
        "months": months,
        "reasons": reasons,
        "worst_month": round(min((m["pnl"] for m in months.values()), default=0.0), 2),
        "green_months": int(sum(1 for m in months.values() if m["pnl"] > 0)),
        "red_months": int(sum(1 for m in months.values() if m["pnl"] < 0)),
    }


def _line(r: dict) -> str:
    mo = " ".join(f"{k}:{v['pnl']:.0f}" for k, v in sorted(r["months"].items()))
    return (
        f"{r['name'][:48]:<48} n={r['trades']:3d} WR {r['win_rate']:5.1f} PF {r['profit_factor']:5.2f} "
        f"net {r['net_pnl']:8.0f} day {r['avg_day']:7.0f} tday {r['avg_trade_day']:7.0f} "
        f"5k {r['days_ge_5k']:2d} DD {r['max_dd_pct']:5.1f} oos {r['oos_pnl']:8.0f} lev {r['max_leverage']:4.1f} {mo}"
    )


def rank_key(r: dict):
    return (
        1 if r["oos_pnl"] > 0 else 0,
        1 if r["red_months"] == 0 and r["trades"] >= 20 else 0,
        r["avg_day"],
        r["profit_factor"],
        -r["max_dd_pct"],
    )


def main():
    t0 = time.time()
    symbols = nifty200_symbols()
    print(f"Gap indicator search | {len(symbols)} names | ₹{CAPITAL:,.0f}", flush=True)
    cache = load_frames(symbols)
    stocks = {k: v for k, v in cache.items() if not k.startswith("_")}
    print(f"Loaded {len(stocks)} frames in {time.time() - t0:.0f}s", flush=True)
    nifty = nifty_context()
    print(f"Nifty sessions {len(nifty)}", flush=True)
    feat, paths, calendar = build_book(stocks, nifty)
    if feat.empty:
        print("No events")
        return
    oos_cut = calendar[int(len(calendar) * 0.70)]
    print(
        f"Window {calendar[0]} → {calendar[-1]}  sessions {len(calendar)}  oos from {oos_cut}",
        flush=True,
    )
    print(
        f"Gap-down 2–6% RSI 45–70 bounce: "
        f"{int(((feat.gap <= -0.02) & (feat.gap >= -0.06) & feat.rsi.between(45, 70) & feat.bounce).sum())}",
        flush=True,
    )

    gap_bands = [
        (0.02, 0.06, "g2-6"),
        (0.01, 0.03, "g1-3"),
        (0.01, 0.045, "g1-4.5"),
        (0.015, 0.05, "g1.5-5"),
        (0.008, 0.025, "g0.8-2.5"),
        (0.008, 0.04, "g0.8-4"),
    ]
    rsi_bands = [(45, 70, "rsi45-70"), (40, 65, "rsi40-65"), (30, 55, "rsi30-55"), (None, None, "rsiAny")]
    exits = list(EXITS)
    size = dict(risk_pct=8.0, max_deploy=0.5, top_k=0, cap_total=False)

    def long_mask(glo, ghi, rlo, rhi, bounce=True):
        m = (feat["gap"] <= -glo) & (feat["gap"] >= -ghi)
        if rlo is not None:
            m = m & feat["rsi"].between(rlo, rhi)
        if bounce:
            m = m & feat["bounce"]
        return m.fillna(False)

    book = []
    n = 0
    total = len(gap_bands) * len(rsi_bands) * 2 * len(exits)
    for glo, ghi, gname in gap_bands:
        for rlo, rhi, rname in rsi_bands:
            for bounce, bname in ((True, "bounce"), (False, "nobounce")):
                mask = long_mask(glo, ghi, rlo, rhi, bounce)
                if int(mask.sum()) < 8:
                    n += len(exits)
                    continue
                for ex in exits:
                    n += 1
                    label = f"L {gname} {rname} {bname} {ex}"
                    res = eval_book(feat, paths, mask, ex, calendar, oos_cut, name=label, **size)
                    res["mask_kind"] = "long"
                    res["glo"] = glo
                    res["ghi"] = ghi
                    res["rlo"] = rlo
                    res["rhi"] = rhi
                    res["bounce"] = bounce
                    book.append(res)
                    if n % 80 == 0:
                        print(f"  {n}/{total} {_line(max(book, key=rank_key))}", flush=True)

    book.sort(key=rank_key, reverse=True)
    print("\n===== STAGE 1 TOP 12 (long fade, live size) =====", flush=True)
    for r in book[:12]:
        print(_line(r), flush=True)

    live_mask = long_mask(0.02, 0.06, 45, 70, True)
    live = eval_book(
        feat, paths, live_mask, "p05_a15", calendar, oos_cut,
        name="LIVE bounce rsi45-70 g2-6 p0.5 sl1.5", **size,
    )
    print("\nLIVE", _line(live), flush=True)

    filters = {
        "or_bull": feat["or_bull"].astype(bool),
        "or_loc60": feat["or_loc"] >= 0.60,
        "or_loc70": feat["or_loc"] >= 0.70,
        "above_vwap": feat["or_above_vwap"].astype(bool),
        "rvol12": feat["or_rvol"] >= 1.2,
        "rvol_not_climax": feat["or_rvol"] <= 2.2,
        "adx_low": feat["adx"] <= 25,
        "htf_up": feat["htf_up"].astype(bool),
        "not_downtrend": ~feat["htf_dn"].astype(bool),
        "above_pdl": feat["above_pdl"].astype(bool),
        "nifty_flat": feat["nifty_gap"] > -0.004,
        "nifty_ok": feat["nifty_gap"] > -0.008,
        "nifty_or_green": feat["nifty_or"] >= 0,
        "rsi5_20_45": feat["rsi5"].between(20, 45),
        "gapvol_0.5_1.8": feat["gap_dayvol"].between(0.5, 1.8),
        "not_chasing": feat["open930"] < feat["or_high"],
    }

    print("\n===== STAGE 2 filters on the live book =====", flush=True)
    filter_rows = []
    for fname, fmask in filters.items():
        res = eval_book(
            feat, paths, live_mask & fmask.fillna(False), "p05_a15", calendar, oos_cut,
            name=f"LIVE+{fname}", **size,
        )
        res["filter"] = fname
        filter_rows.append(res)
        print(_line(res), flush=True)

    helpful = [
        r for r in filter_rows
        if r["oos_pnl"] > 0 and r["trades"] >= 15 and r["avg_day"] > live["avg_day"] and r["red_months"] <= live["red_months"]
    ]
    helpful.sort(key=rank_key, reverse=True)
    print("\nFilters that beat live on rupees/day and stay positive out of sample:", flush=True)
    if not helpful:
        print("  none", flush=True)
    for r in helpful:
        print(" ", _line(r), flush=True)

    # Also stack filters on the stage-1 leader if it is a different exit/gap.
    leader = book[0]
    print("\n===== STAGE 3 stacks =====", flush=True)
    stacks = []
    names = [r["filter"] for r in helpful[:6]]
    for k in (1, 2, 3):
        for combo in combinations(names, k):
            m = live_mask.copy()
            for fname in combo:
                m = m & filters[fname].fillna(False)
            if int(m.sum()) < 10:
                continue
            label = "LIVE+" + "+".join(combo)
            res = eval_book(feat, paths, m, "p05_a15", calendar, oos_cut, name=label, **size)
            res["filters"] = list(combo)
            res["base"] = "live"
            stacks.append(res)
    # Best stage-1 geometry plus each helpful filter, and the bare leader.
    stacks.append(leader)
    for r in helpful[:6]:
        m = long_mask(leader["glo"], leader["ghi"], leader["rlo"], leader["rhi"], leader["bounce"])
        m = m & filters[r["filter"]].fillna(False)
        res = eval_book(
            feat, paths, m, leader["exit"], calendar, oos_cut,
            name=f"LEAD+{r['filter']} {leader['exit']}", **size,
        )
        res["filters"] = [r["filter"]]
        res["base"] = "lead"
        res["glo"] = leader["glo"]
        res["ghi"] = leader["ghi"]
        res["rlo"] = leader["rlo"]
        res["rhi"] = leader["rhi"]
        res["bounce"] = leader["bounce"]
        stacks.append(res)
    stacks.sort(key=rank_key, reverse=True)
    for r in stacks[:12]:
        print(_line(r), flush=True)

    # Short fade of gap-ups: 15-minute candle rejected (close in the bottom of the range).
    print("\n===== SHORT fade =====", flush=True)
    shorts = []
    reject = (~feat["or_bull"]) & (feat["or_loc"] <= 0.40) & (feat["open930"] <= feat["or_close"])
    for glo, ghi, gname in ((0.01, 0.04, "g1-4"), (0.015, 0.05, "g1.5-5"), (0.02, 0.06, "g2-6")):
        base = (feat["gap"] >= glo) & (feat["gap"] <= ghi) & reject
        for rlo, rhi, rname in ((55, 80, "rsi55-80"), (None, None, "rsiAny")):
            m = base if rlo is None else base & feat["rsi"].between(rlo, rhi)
            m = m.fillna(False)
            if int(m.sum()) < 8:
                continue
            for ex in ("p05_a15", "p06_a10", "half_a12", "p08_or", "p06_t1030"):
                res = eval_book(
                    feat, paths, m, ex, calendar, oos_cut, side="short",
                    name=f"S {gname} {rname} {ex}", **size,
                )
                shorts.append(res)
    shorts.sort(key=rank_key, reverse=True)
    for r in shorts[:8]:
        print(_line(r), flush=True)

    # Opening-range break the other way: gap down, then the 9:30 bar closes back over
    # the opening-range high. Enter the 9:35 open. Known only after 9:35.
    print("\n===== 9:35 ORB after a gap =====", flush=True)
    orbs = []
    reclaim = feat["c930_bull"] & feat["c930_above_vwap"] & (feat["close" if False else "open935"] > 0)
    # break of the opening range: 9:30 close is not stored; c930_bull + close above vwap
    # plus the 9:30 high isn't stored. Use open935 >= or_high, which is known at the 9:35 open
    # only if we require the 9:30 bar's high — open935 >= or_high means price is already
    # through the range at the next open.
    brk = feat["open935"].notna() & (feat["open935"] >= feat["or_high"]) & feat["c930_bull"] & feat["c930_above_vwap"]
    for glo, ghi, gname, direction in (
        (0.008, 0.04, "down0.8-4", "down"),
        (0.004, 0.02, "up0.4-2", "up"),
    ):
        if direction == "down":
            m = brk & (feat["gap"] <= -glo) & (feat["gap"] >= -ghi) & feat["htf_up"]
        else:
            m = brk & (feat["gap"] >= glo) & (feat["gap"] <= ghi) & feat["htf_up"] & (feat["or_rvol"] >= 1.2)
        m = m.fillna(False)
        for ex in ("r15_or", "p08_a12", "p08_or", "p10_a15"):
            res = eval_book(
                feat, paths, m, ex, calendar, oos_cut, entry_at="935",
                name=f"ORB935 {gname} {ex}", **size,
            )
            orbs.append(res)
            print(_line(res), flush=True)

    # Size sweep on live and the best robust long.
    robust = [
        r for r in [live, *book[:8], *stacks[:8], *shorts[:3]]
        if r["oos_pnl"] > 0 and r["trades"] >= 20 and r["red_months"] == 0
    ]
    robust.sort(key=rank_key, reverse=True)
    pick = robust[0] if robust else live
    print("\n===== SIZE SWEEP =====", flush=True)
    print("Anchor", _line(pick), flush=True)

    def mask_of(r):
        if r.get("base") == "live" or (r.get("filter") and str(r["name"]).startswith("LIVE")):
            m = live_mask.copy()
            for fname in r.get("filters") or ([r["filter"]] if r.get("filter") else []):
                m = m & filters[fname].fillna(False)
            return m, r.get("exit") or "p05_a15", "long", "930"
        if r.get("glo") is not None:
            m = long_mask(r["glo"], r["ghi"], r["rlo"], r["rhi"], r["bounce"])
            for fname in r.get("filters") or []:
                m = m & filters[fname].fillna(False)
            return m, r["exit"], "long", "930"
        return live_mask, "p05_a15", "long", "930"

    sizes = [
        ("live size, all names", dict(risk_pct=8, max_deploy=0.5, top_k=0, cap_total=False)),
        ("top 2, 5x cap", dict(risk_pct=8, max_deploy=0.5, top_k=2, cap_total=True)),
        ("top 1, full 5x", dict(risk_pct=8, max_deploy=1.0, top_k=1, cap_total=True)),
        ("top 3, 5x cap", dict(risk_pct=10, max_deploy=0.34, top_k=3, cap_total=True)),
        ("all names, each 25% BP", dict(risk_pct=8, max_deploy=0.25, top_k=0, cap_total=True)),
    ]
    sweep = []
    anchor_mask, anchor_exit, anchor_side, anchor_entry = mask_of(pick)
    for label, sz in sizes:
        res = eval_book(
            feat, paths, anchor_mask, anchor_exit, calendar, oos_cut,
            side=anchor_side, entry_at=anchor_entry, name=f"SIZE {label}", **sz,
        )
        sweep.append(res)
        need = (CAPITAL * 5000 / res["avg_day"]) if res["avg_day"] > 0 else None
        extra = f"  capital for ₹5k/session ≈ ₹{need:,.0f}" if need else "  does not pay every session"
        print(_line(res) + extra, flush=True)

    # Same sweep on the untouched live book, so size is not confused with a new filter.
    if pick["name"] != live["name"]:
        print("\nLive book, size only", flush=True)
        for label, sz in sizes:
            res = eval_book(
                feat, paths, live_mask, "p05_a15", calendar, oos_cut,
                name=f"LIVE SIZE {label}", **sz,
            )
            sweep.append(res)
            need = (CAPITAL * 5000 / res["avg_day"]) if res["avg_day"] > 0 else None
            extra = f"  capital for ₹5k/session ≈ ₹{need:,.0f}" if need else ""
            print(_line(res) + extra, flush=True)

    payload = {
        "window": {"start": str(calendar[0]), "end": str(calendar[-1]), "sessions": len(calendar), "oos_from": str(oos_cut)},
        "capital": CAPITAL,
        "leverage": LEVERAGE,
        "live": live,
        "stage1_top": book[:15],
        "filters": filter_rows,
        "stacks": stacks[:15],
        "shorts_top": shorts[:8],
        "orb": orbs,
        "size_sweep": sweep,
        "anchor": pick["name"],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
