"""
Gap-open filter hunt.

Enrich every Nifty 200 9:15 gap with features known at the open
(yesterday's daily RSI/ADX/SMAs, prior 5m RSI, index gap, etc.),
measure same-day fill / R-multiple, then re-simulate the live packs
with the filters that look real.

    python scripts/_gap_filter_search.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_n200_search import (  # noqa: E402
    add_indicators,
    nifty200_symbols,
)
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts.intraday_gap_hunt import (  # noqa: E402
    collect_events,
    make_signals,
    months_from_trades,
    prepare,
    simulate_open,
)
from trading.services.indicators import _adx, _atr, _rsi  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402

SL_ATR = 0.6
GAP_MAX = 0.15
MIN_N = 12


def _rsi2(close: pd.Series) -> pd.Series:
    return _rsi(close, 2)


def daily_features(symbol: str) -> pd.DataFrame:
    """One row per calendar date, values as-of the prior completed daily bar."""
    d = load_price_dataframe(symbol)
    if d.empty or len(d) < 30:
        return pd.DataFrame()
    d = d.copy()
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["sma200"] = d["close"].rolling(200).mean()
    d["rsi14"] = _rsi(d["close"], 14)
    d["rsi2"] = _rsi2(d["close"])
    d["atr14"] = _atr(d["high"], d["low"], d["close"], 14)
    adx, di_p, di_m = _adx(d["high"], d["low"], d["close"], 14)
    d["adx"] = adx
    d["di_plus"] = di_p
    d["di_minus"] = di_m
    d["vol_sma20"] = d["volume"].rolling(20).mean()
    d["rel_vol"] = d["volume"] / d["vol_sma20"].replace(0, np.nan)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_loc"] = (d["close"] - d["low"]) / rng
    d["day_ret"] = d["close"].pct_change()
    d["down_streak"] = (
        d["day_ret"].lt(0).astype(int).groupby((d["day_ret"] >= 0).cumsum()).cumsum()
    )
    d["up_streak"] = (
        d["day_ret"].gt(0).astype(int).groupby((d["day_ret"] <= 0).cumsum()).cumsum()
    )
    d["rng_pct"] = rng / d["close"]
    d["nr7"] = rng <= rng.rolling(7).min()
    d["dist_sma20"] = d["close"] / d["sma20"] - 1.0
    d["dist_sma50"] = d["close"] / d["sma50"] - 1.0
    d["above_sma20"] = d["close"] > d["sma20"]
    d["above_sma50"] = d["close"] > d["sma50"]
    d["above_sma200"] = d["close"] > d["sma200"].fillna(d["sma50"])
    d["htf_up"] = (d["close"] > d["sma20"]) & (d["sma20"] > d["sma50"].fillna(d["sma20"]))
    keep = [
        "rsi14", "rsi2", "adx", "di_plus", "di_minus", "rel_vol", "close_loc",
        "day_ret", "down_streak", "up_streak", "rng_pct", "nr7", "atr14",
        "dist_sma20", "dist_sma50", "above_sma20", "above_sma50", "above_sma200",
        "htf_up", "sma20", "close",
    ]
    shifted = d[keep].shift(1)
    shifted.index = pd.to_datetime(shifted.index).date
    shifted.columns = [f"d_{c}" for c in shifted.columns]
    return shifted


def index_gaps(idx: pd.DataFrame) -> dict:
    """Session date -> Nifty 200 overnight gap at 9:15."""
    if idx is None or idx.empty:
        return {}
    first = idx.groupby("session", sort=True).head(1)
    last = idx.groupby("session", sort=True).tail(1)
    last_by = last.set_index("session")["close"]
    prev_close = None
    out = {}
    for _, row in first.iterrows():
        sess = row["session"]
        o = float(row["open"])
        if prev_close and prev_close > 0 and o > 0:
            out[sess] = o / prev_close - 1.0
        prev_close = float(last_by.get(sess, o))
    return out


def last_rsi_5m(df: pd.DataFrame) -> dict:
    last = df.groupby("session", sort=True).tail(1)
    prev = {}
    prev_rsi = np.nan
    for ts, row in last.iterrows():
        sess = row["session"]
        prev[sess] = prev_rsi
        prev_rsi = float(row.get("rsi") or np.nan)
    return prev


def enrich_events(events: pd.DataFrame, stocks: dict[str, pd.DataFrame], idx_gap: dict) -> pd.DataFrame:
    daily_cache: dict[str, pd.DataFrame] = {}
    rsi5_cache: dict[str, dict] = {}
    rows = []
    for r in events.itertuples(index=False):
        rec = {
            "symbol": r.symbol, "ts": r.ts, "session": r.session, "open": r.open,
            "pdc": r.pdc, "gap": r.gap, "atr": r.atr, "high": r.high,
            "low": r.low, "close1": r.close1,
        }
        if r.symbol not in daily_cache:
            daily_cache[r.symbol] = daily_features(r.symbol)
        dfeat = daily_cache[r.symbol]
        if not dfeat.empty and r.session in dfeat.index:
            rowd = dfeat.loc[r.session]
            if isinstance(rowd, pd.DataFrame):
                rowd = rowd.iloc[-1]
            for col, val in rowd.items():
                rec[col] = val
        if r.symbol not in rsi5_cache:
            rsi5_cache[r.symbol] = last_rsi_5m(stocks[r.symbol])
        rec["rsi5_prev"] = rsi5_cache[r.symbol].get(r.session, np.nan)
        rec["idx_gap"] = idx_gap.get(r.session, np.nan)
        rec["dow"] = pd.Timestamp(r.session).day_name() if r.session else ""
        rec["abs_gap"] = abs(r.gap)
        atr_d = rec.get("d_atr14")
        if atr_d is not None and pd.notna(atr_d) and r.pdc > 0 and atr_d > 0:
            rec["gap_vs_atr"] = abs(r.gap) * r.pdc / float(atr_d)
        else:
            rec["gap_vs_atr"] = np.nan
        sma20 = rec.get("d_sma20")
        if sma20 is not None and pd.notna(sma20) and sma20 > 0:
            rec["open_vs_sma20"] = r.open / float(sma20) - 1.0
        else:
            rec["open_vs_sma20"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def outcome_scan(ev: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fills, bounces, mfes, maes, rs, reasons = [], [], [], [], [], []
    for r in ev.itertuples(index=False):
        df = stocks.get(r.symbol)
        if df is None:
            fills.append(False); bounces.append(False)
            mfes.append(np.nan); maes.append(np.nan); rs.append(np.nan); reasons.append("na")
            continue
        day = df[df["session"] == r.session]
        if day.empty:
            fills.append(False); bounces.append(False)
            mfes.append(np.nan); maes.append(np.nan); rs.append(np.nan); reasons.append("na")
            continue
        o, pdc, atr = float(r.open), float(r.pdc), float(r.atr)
        gap = float(r.gap)
        hi = day["high"].to_numpy(float)
        lo = day["low"].to_numpy(float)
        cl = day["close"].to_numpy(float)
        if gap < 0:
            mfe = (hi.max() / o - 1.0) * 100
            mae = (lo.min() / o - 1.0) * 100
            filled = bool(hi.max() >= pdc)
            bounced = bool(cl[-1] > o)
            stop = o - SL_ATR * atr
            tgt = pdc
            side = "long"
        else:
            mfe = (1.0 - lo.min() / o) * 100
            mae = (hi.max() / o - 1.0) * 100
            filled = bool(lo.min() <= pdc)
            bounced = bool(cl[-1] < o)
            stop = o + SL_ATR * atr
            tgt = pdc
            side = "short"
        fills.append(filled)
        bounces.append(bounced)
        mfes.append(mfe)
        maes.append(mae)
        # path: SL first
        rmult = np.nan
        reason = "eod"
        risk = (o - stop) if side == "long" else (stop - o)
        if risk <= 0:
            rs.append(np.nan); reasons.append("bad")
            continue
        for i, (h, l, c) in enumerate(zip(hi, lo, cl)):
            if side == "long":
                hit_sl, hit_tp = l <= stop, h >= tgt
                px_sl, px_tp, px_eod = stop, tgt, c
            else:
                hit_sl, hit_tp = h >= stop, l <= tgt
                px_sl, px_tp, px_eod = stop, tgt, c
            if hit_sl:
                rmult = (px_sl - o) / risk if side == "long" else (o - px_sl) / risk
                reason = "sl"
                break
            if hit_tp:
                rmult = (px_tp - o) / risk if side == "long" else (o - px_tp) / risk
                reason = "target"
                break
        else:
            rmult = (px_eod - o) / risk if side == "long" else (o - px_eod) / risk
            reason = "eod"
        rs.append(rmult)
        reasons.append(reason)
    out = ev.copy()
    out["filled"] = fills
    out["bounced"] = bounces
    out["mfe"] = mfes
    out["mae"] = maes
    out["R"] = rs
    out["reason"] = reasons
    out["win"] = out["R"] > 0
    return out


def _fmt_bucket(name, g):
    n = len(g)
    if n == 0:
        return None
    wr = 100.0 * g["win"].mean()
    fill = 100.0 * g["filled"].mean()
    bounce = 100.0 * g["bounced"].mean()
    avg_r = float(g["R"].mean())
    med_r = float(g["R"].median())
    avg_mfe = float(g["mfe"].mean())
    return (
        f"  {name:<28} n={n:4d}  fill {fill:5.1f}%  bounce {bounce:5.1f}%  "
        f"WR {wr:5.1f}%  avgR {avg_r:6.2f}  medR {med_r:6.2f}  MFE {avg_mfe:5.2f}%"
    )


def print_buckets(label: str, df: pd.DataFrame, col: str, bins, names=None):
    print(f"\n--- {label} ---")
    if col not in df.columns:
        print("  (missing)")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    for i, (lo, hi) in enumerate(bins):
        mask = s.ge(lo) & s.lt(hi)
        g = df[mask]
        nm = names[i] if names else f"{lo}–{hi}"
        line = _fmt_bucket(nm, g)
        if line:
            print(line)


def print_flags(label: str, df: pd.DataFrame, specs: list[tuple[str, pd.Series]]):
    print(f"\n--- {label} ---")
    line = _fmt_bucket("ALL", df)
    if line:
        print(line)
    for name, mask in specs:
        g = df[mask.fillna(False)]
        line = _fmt_bucket(name, g)
        if line:
            print(line)


def chronic_gappers(events: pd.DataFrame, thresh: float = 0.03, frac: float = 0.35) -> set[str]:
    bad = set()
    for sym, g in events.groupby("symbol"):
        rate = (g["gap"].abs() >= thresh).mean()
        if rate >= frac and len(g) >= 20:
            bad.add(sym)
    return bad


def apply_filter(events: pd.DataFrame, spec: dict) -> pd.DataFrame:
    m = pd.Series(True, index=events.index)
    side = spec.get("side")
    if side == "down":
        m &= events["gap"] < 0
    elif side == "up":
        m &= events["gap"] > 0
    gmin = spec.get("gap_min")
    gmax = spec.get("gap_max", GAP_MAX)
    if gmin is not None:
        m &= events["gap"].abs() >= gmin
        m &= events["gap"].abs() <= gmax
    for col, op, val in spec.get("rules", []):
        s = pd.to_numeric(events[col], errors="coerce") if col not in ("dow",) else events[col]
        if op == "<":
            m &= s < val
        elif op == "<=":
            m &= s <= val
        elif op == ">":
            m &= s > val
        elif op == ">=":
            m &= s >= val
        elif op == "==":
            m &= s == val
        elif op == "in":
            m &= s.isin(val)
        elif op == "notna":
            m &= s.notna()
        elif op == "true":
            m &= events[col].fillna(False).astype(bool)
        elif op == "false":
            m &= ~events[col].fillna(False).astype(bool)
        elif op == "notin":
            m &= ~events["symbol"].isin(val)
    return events[m]


def run_pack(stocks, events, name, mode, gap_min, size, extra_mask=None, gap_max=GAP_MAX):
    sigs = make_signals(events, mode, gap_min, gap_max, SL_ATR, "fill")
    if extra_mask is not None and not sigs.empty:
        key = extra_mask
        sigs = sigs.merge(key, on=["symbol", "ts"], how="inner")
    res, tdf = simulate_open(stocks, sigs, name, **size)
    months = months_from_trades(tdf)
    return res, tdf, months


def mask_keys(filtered: pd.DataFrame) -> pd.DataFrame:
    return filtered[["symbol", "ts"]].drop_duplicates()


def main():
    symbols = nifty200_symbols()
    print(f"Loading {len(symbols)} Nifty 200 5m frames…", flush=True)
    cache = load_frames(symbols)
    stocks = prepare(cache)
    idx = cache.get("_INDEX")
    if idx is not None and not idx.empty:
        idx = add_indicators(idx)
    idx_gap = index_gaps(idx) if idx is not None else {}
    events = collect_events(stocks)
    print(f"Events {len(events)}  stocks {len(stocks)}  index-gap days {len(idx_gap)}", flush=True)

    print("Enriching with daily RSI / ADX / SMA / streaks…", flush=True)
    ev = enrich_events(events, stocks, idx_gap)
    print("Scanning same-day outcomes…", flush=True)
    ev = outcome_scan(ev, stocks)

    chronic = chronic_gappers(events)
    print(f"\nChronic ≥3% gappers (≥35% of days): {sorted(chronic) or 'none'}")
    ev["chronic"] = ev["symbol"].isin(chronic)

    down = ev[(ev["gap"] <= -0.02) & (ev["gap"] >= -0.15)].copy()
    up = ev[(ev["gap"] >= 0.02) & (ev["gap"] <= 0.15)].copy()
    down3 = ev[(ev["gap"] <= -0.03) & (ev["gap"] >= -0.15)].copy()
    up3 = ev[(ev["gap"] >= 0.03) & (ev["gap"] <= 0.15)].copy()

    print("\n========== EVENT-LEVEL EDGE (0.6 ATR stop, target = prior close) ==========")
    print(_fmt_bucket("gap-down ≥2%", down))
    print(_fmt_bucket("gap-down ≥3%", down3))
    print(_fmt_bucket("gap-up ≥2%", up))
    print(_fmt_bucket("gap-up ≥3%", up3))

    print_buckets("DOWN ≥2%  daily RSI(14)", down, "d_rsi14",
                  [(0, 30), (30, 40), (40, 45), (45, 50), (50, 55), (55, 65), (65, 100)],
                  ["RSI<30", "30-40", "40-45", "45-50", "50-55", "55-65", "RSI≥65"])
    print_buckets("DOWN ≥2%  daily RSI(2)", down, "d_rsi2",
                  [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70), (70, 101)],
                  ["RSI2<5", "5-10", "10-20", "20-40", "40-70", "RSI2≥70"])
    print_buckets("DOWN ≥2%  5m RSI (prev session)", down, "rsi5_prev",
                  [(0, 30), (30, 40), (40, 50), (50, 60), (60, 101)],
                  ["RSI5<30", "30-40", "40-50", "50-60", "RSI5≥60"])
    print_buckets("DOWN ≥2%  daily ADX", down, "d_adx",
                  [(0, 15), (15, 20), (20, 25), (25, 35), (35, 80)],
                  ["ADX<15", "15-20", "20-25", "25-35", "ADX≥35"])
    print_buckets("DOWN ≥2%  prev-day return %", down, "d_day_ret",
                  [(-1, -0.03), (-0.03, -0.01), (-0.01, 0), (0, 0.01), (0.01, 0.03), (0.03, 1)],
                  ["prev≤-3%", "-3 to -1%", "-1 to 0%", "0 to +1%", "+1 to +3%", "prev≥+3%"])
    print_buckets("DOWN ≥2%  close location yday", down, "d_close_loc",
                  [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)],
                  ["weak close", "lower half", "upper half", "strong close"])
    print_buckets("DOWN ≥2%  dist vs SMA20", down, "d_dist_sma20",
                  [(-1, -0.08), (-0.08, -0.03), (-0.03, 0), (0, 0.03), (0.03, 0.08), (0.08, 1)],
                  ["<-8%", "-8 to -3%", "-3 to 0%", "0 to +3%", "+3 to +8%", ">+8%"])
    print_buckets("DOWN ≥2%  gap / daily ATR", down, "gap_vs_atr",
                  [(0, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 20)],
                  ["<0.8 ATR", "0.8-1.2", "1.2-1.8", "1.8-2.5", ">2.5 ATR"])
    print_buckets("DOWN ≥2%  yday relative volume", down, "d_rel_vol",
                  [(0, 0.7), (0.7, 1.0), (1.0, 1.4), (1.4, 2.0), (2.0, 20)],
                  ["quiet <0.7", "0.7-1.0", "1.0-1.4", "1.4-2.0", "spike >2x"])
    print_buckets("DOWN ≥2%  index gap", down, "idx_gap",
                  [(-1, -0.01), (-0.01, -0.003), (-0.003, 0.003), (0.003, 0.01), (0.01, 1)],
                  ["idx ≤-1%", "-1 to -0.3%", "flat", "+0.3 to +1%", "idx ≥+1%"])
    print_buckets("DOWN ≥2%  down-day streak", down, "d_down_streak",
                  [(0, 1), (1, 2), (2, 3), (3, 4), (4, 20)],
                  ["0", "1", "2", "3", "4+"])

    print_flags("DOWN ≥2%  boolean filters", down, [
        ("htf_up (SMA20>50)", down["d_htf_up"].fillna(False).astype(bool)),
        ("NOT htf_up", ~down["d_htf_up"].fillna(False).astype(bool)),
        ("above SMA200", down["d_above_sma200"].fillna(False).astype(bool)),
        ("below SMA200", ~down["d_above_sma200"].fillna(False).astype(bool)),
        ("above SMA20", down["d_above_sma20"].fillna(False).astype(bool)),
        ("below SMA20", ~down["d_above_sma20"].fillna(False).astype(bool)),
        ("+DI > -DI", pd.to_numeric(down["d_di_plus"], errors="coerce") > pd.to_numeric(down["d_di_minus"], errors="coerce")),
        ("-DI > +DI", pd.to_numeric(down["d_di_minus"], errors="coerce") > pd.to_numeric(down["d_di_plus"], errors="coerce")),
        ("NR7 yesterday", down["d_nr7"].fillna(False).astype(bool)),
        ("skip chronic", ~down["chronic"]),
        ("chronic only", down["chronic"]),
        ("Mon/Tue", down["dow"].isin(["Monday", "Tuesday"])),
        ("Wed", down["dow"].eq("Wednesday")),
        ("Thu/Fri", down["dow"].isin(["Thursday", "Friday"])),
        ("RSI14<40", pd.to_numeric(down["d_rsi14"], errors="coerce") < 40),
        ("RSI14<45", pd.to_numeric(down["d_rsi14"], errors="coerce") < 45),
        ("RSI14 30-50", pd.to_numeric(down["d_rsi14"], errors="coerce").between(30, 50)),
        ("RSI2<10", pd.to_numeric(down["d_rsi2"], errors="coerce") < 10),
        ("RSI2<20", pd.to_numeric(down["d_rsi2"], errors="coerce") < 20),
        ("RSI5<40", pd.to_numeric(down["rsi5_prev"], errors="coerce") < 40),
        ("Connors: SMA200 + RSI2<10", down["d_above_sma200"].fillna(False).astype(bool) & (pd.to_numeric(down["d_rsi2"], errors="coerce") < 10)),
        ("htf_up + RSI14<45", down["d_htf_up"].fillna(False).astype(bool) & (pd.to_numeric(down["d_rsi14"], errors="coerce") < 45)),
        ("htf_up + RSI2<20", down["d_htf_up"].fillna(False).astype(bool) & (pd.to_numeric(down["d_rsi2"], errors="coerce") < 20)),
        ("idiosyncratic (idx>-0.4%)", pd.to_numeric(down["idx_gap"], errors="coerce") > -0.004),
        ("sympathy (idx<-0.4%)", pd.to_numeric(down["idx_gap"], errors="coerce") <= -0.004),
        ("gap 2-4% only", down["gap"].between(-0.04, -0.02)),
        ("gap 4-8%", down["gap"].between(-0.08, -0.04)),
        ("ADX<22", pd.to_numeric(down["d_adx"], errors="coerce") < 22),
        ("ADX≥25", pd.to_numeric(down["d_adx"], errors="coerce") >= 25),
        ("prev green", pd.to_numeric(down["d_day_ret"], errors="coerce") > 0),
        ("prev red", pd.to_numeric(down["d_day_ret"], errors="coerce") < 0),
        ("strong yday close", pd.to_numeric(down["d_close_loc"], errors="coerce") >= 0.7),
        ("weak yday close", pd.to_numeric(down["d_close_loc"], errors="coerce") <= 0.3),
    ])

    print_buckets("UP ≥2%  daily RSI(14)", up, "d_rsi14",
                  [(0, 40), (40, 50), (50, 55), (55, 60), (60, 70), (70, 101)],
                  ["RSI<40", "40-50", "50-55", "55-60", "60-70", "RSI≥70"])
    print_buckets("UP ≥2%  daily RSI(2)", up, "d_rsi2",
                  [(0, 30), (30, 60), (60, 80), (80, 90), (90, 101)],
                  ["RSI2<30", "30-60", "60-80", "80-90", "RSI2≥90"])
    print_buckets("UP ≥2%  5m RSI prev", up, "rsi5_prev",
                  [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)],
                  ["RSI5<40", "40-50", "50-60", "60-70", "RSI5≥70"])
    print_buckets("UP ≥2%  daily ADX", up, "d_adx",
                  [(0, 15), (15, 20), (20, 25), (25, 35), (35, 80)],
                  ["ADX<15", "15-20", "20-25", "25-35", "ADX≥35"])
    print_buckets("UP ≥2%  index gap", up, "idx_gap",
                  [(-1, -0.003), (-0.003, 0.003), (0.003, 0.01), (0.01, 1)],
                  ["idx down", "flat", "+0.3 to +1%", "idx ≥+1%"])
    print_flags("UP ≥2%  boolean filters", up, [
        ("htf_up", up["d_htf_up"].fillna(False).astype(bool)),
        ("NOT htf_up", ~up["d_htf_up"].fillna(False).astype(bool)),
        ("above SMA200", up["d_above_sma200"].fillna(False).astype(bool)),
        ("below SMA200", ~up["d_above_sma200"].fillna(False).astype(bool)),
        ("skip chronic", ~up["chronic"]),
        ("chronic only", up["chronic"]),
        ("RSI14>60", pd.to_numeric(up["d_rsi14"], errors="coerce") > 60),
        ("RSI14>70", pd.to_numeric(up["d_rsi14"], errors="coerce") > 70),
        ("RSI2>80", pd.to_numeric(up["d_rsi2"], errors="coerce") > 80),
        ("RSI2>90", pd.to_numeric(up["d_rsi2"], errors="coerce") > 90),
        ("RSI5>60", pd.to_numeric(up["rsi5_prev"], errors="coerce") > 60),
        ("NOT htf + RSI14>60", (~up["d_htf_up"].fillna(False).astype(bool)) & (pd.to_numeric(up["d_rsi14"], errors="coerce") > 60)),
        ("ADX<22", pd.to_numeric(up["d_adx"], errors="coerce") < 22),
        ("gap 2-4%", up["gap"].between(0.02, 0.04)),
        ("gap 3-6%", up["gap"].between(0.03, 0.06)),
        ("idiosyncratic idx<+0.4%", pd.to_numeric(up["idx_gap"], errors="coerce") < 0.004),
        ("sympathy idx>+0.4%", pd.to_numeric(up["idx_gap"], errors="coerce") >= 0.004),
        ("prev green", pd.to_numeric(up["d_day_ret"], errors="coerce") > 0),
        ("prev red", pd.to_numeric(up["d_day_ret"], errors="coerce") < 0),
        ("strong yday close", pd.to_numeric(up["d_close_loc"], errors="coerce") >= 0.7),
    ])

    # By symbol concentration
    print("\n--- Names with ≥6 gap-down (≥2%) events ---")
    for sym, g in down.groupby("symbol"):
        if len(g) < 6:
            continue
        print(_fmt_bucket(sym, g))

    print("\n========== SIZED BACKTESTS (same engine as the live page) ==========")
    story_sz = dict(risk_pct=5.0, max_pos=4, top_k=2, max_deploy=0.35)
    win_sz = dict(risk_pct=10.0, max_pos=5, top_k=3, max_deploy=0.40)

    # Build key frames for filters (events already have ts)
    def keys_from(mask_df):
        return mask_df[["symbol", "ts"]].drop_duplicates()

    packs = []

    def add(label, mode, gmin, size, filt_df):
        key = keys_from(filt_df) if filt_df is not None else None
        res, tdf, months = run_pack(stocks, events, label, mode, gmin, size, extra_mask=key)
        packs.append((res, months, tdf))
        print(
            f"{res.total_return_pct:7.1f}%  WR {res.win_rate:5.1f}  PF {res.profit_factor:5.2f}  "
            f"n={res.trades:3d}  DD {res.max_dd_pct:5.1f}  OOS {res.oos_pnl:8.0f}  {label}",
            flush=True,
        )

    print("\n-- story pack: down_bounce ≥2%  5% risk --")
    add("BASE down≥2% r5", "down_bounce", 0.02, story_sz, None)

    d2 = down.copy()
    add("skip chronic", "down_bounce", 0.02, story_sz, d2[~d2["chronic"]])
    add("RSI14<45", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_rsi14"], errors="coerce") < 45])
    add("RSI14<40", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_rsi14"], errors="coerce") < 40])
    add("RSI14 30-50", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_rsi14"], errors="coerce").between(30, 50)])
    add("RSI2<20", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_rsi2"], errors="coerce") < 20])
    add("RSI2<10", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_rsi2"], errors="coerce") < 10])
    add("RSI5<40", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["rsi5_prev"], errors="coerce") < 40])
    add("htf_up", "down_bounce", 0.02, story_sz, d2[d2["d_htf_up"].fillna(False).astype(bool)])
    add("above SMA200", "down_bounce", 0.02, story_sz, d2[d2["d_above_sma200"].fillna(False).astype(bool)])
    add("Connors SMA200+RSI2<10", "down_bounce", 0.02, story_sz,
        d2[d2["d_above_sma200"].fillna(False).astype(bool) & (pd.to_numeric(d2["d_rsi2"], errors="coerce") < 10)])
    add("htf_up + RSI14<45", "down_bounce", 0.02, story_sz,
        d2[d2["d_htf_up"].fillna(False).astype(bool) & (pd.to_numeric(d2["d_rsi14"], errors="coerce") < 45)])
    add("htf_up + RSI2<20", "down_bounce", 0.02, story_sz,
        d2[d2["d_htf_up"].fillna(False).astype(bool) & (pd.to_numeric(d2["d_rsi2"], errors="coerce") < 20)])
    add("ADX<22", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_adx"], errors="coerce") < 22])
    add("ADX≥25", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_adx"], errors="coerce") >= 25])
    add("idiosyncratic idx>-0.4%", "down_bounce", 0.02, story_sz,
        d2[pd.to_numeric(d2["idx_gap"], errors="coerce") > -0.004])
    add("sympathy idx≤-0.4%", "down_bounce", 0.02, story_sz,
        d2[pd.to_numeric(d2["idx_gap"], errors="coerce") <= -0.004])
    add("gap 2-4% cap", "down_bounce", 0.02, story_sz, d2[d2["gap"] >= -0.04])
    add("prev green", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_day_ret"], errors="coerce") > 0])
    add("prev red", "down_bounce", 0.02, story_sz, d2[pd.to_numeric(d2["d_day_ret"], errors="coerce") < 0])
    add("strong yday close", "down_bounce", 0.02, story_sz,
        d2[pd.to_numeric(d2["d_close_loc"], errors="coerce") >= 0.7])
    add("weak yday close", "down_bounce", 0.02, story_sz,
        d2[pd.to_numeric(d2["d_close_loc"], errors="coerce") <= 0.3])
    add("gap/ATR 1.2-2.5", "down_bounce", 0.02, story_sz,
        d2[pd.to_numeric(d2["gap_vs_atr"], errors="coerce").between(1.2, 2.5)])
    add("below SMA20", "down_bounce", 0.02, story_sz, d2[~d2["d_above_sma20"].fillna(False).astype(bool)])
    add("RSI14<45 + skip chronic", "down_bounce", 0.02, story_sz,
        d2[(~d2["chronic"]) & (pd.to_numeric(d2["d_rsi14"], errors="coerce") < 45)])
    add("RSI14<45 + htf_up + skip chron", "down_bounce", 0.02, story_sz,
        d2[(~d2["chronic"]) & d2["d_htf_up"].fillna(False).astype(bool)
           & (pd.to_numeric(d2["d_rsi14"], errors="coerce") < 45)])
    add("RSI5<40 + skip chronic", "down_bounce", 0.02, story_sz,
        d2[(~d2["chronic"]) & (pd.to_numeric(d2["rsi5_prev"], errors="coerce") < 40)])
    add("SMA200 + RSI14<50", "down_bounce", 0.02, story_sz,
        d2[d2["d_above_sma200"].fillna(False).astype(bool) & (pd.to_numeric(d2["d_rsi14"], errors="coerce") < 50)])
    add("SMA200 + RSI2<20 + skip chron", "down_bounce", 0.02, story_sz,
        d2[(~d2["chronic"]) & d2["d_above_sma200"].fillna(False).astype(bool)
           & (pd.to_numeric(d2["d_rsi2"], errors="coerce") < 20)])

    print("\n-- winner pack: both_fill ≥3%  10% risk --")
    add("BASE both≥3% r10", "both_fill", 0.03, win_sz, None)
    both3 = ev[(ev["gap"].abs() >= 0.03) & (ev["gap"].abs() <= 0.15)].copy()
    # directional filters
    long_ok = (both3["gap"] < 0) & (pd.to_numeric(both3["d_rsi14"], errors="coerce") < 45)
    short_ok = (both3["gap"] > 0) & (pd.to_numeric(both3["d_rsi14"], errors="coerce") > 55)
    add("RSI14 dir (L<45 S>55)", "both_fill", 0.03, win_sz, both3[long_ok | short_ok])
    long_ok = (both3["gap"] < 0) & (pd.to_numeric(both3["d_rsi2"], errors="coerce") < 20)
    short_ok = (both3["gap"] > 0) & (pd.to_numeric(both3["d_rsi2"], errors="coerce") > 80)
    add("RSI2 dir (L<20 S>80)", "both_fill", 0.03, win_sz, both3[long_ok | short_ok])
    add("skip chronic both", "both_fill", 0.03, win_sz, both3[~both3["chronic"]])
    long_ok = (both3["gap"] < 0) & both3["d_htf_up"].fillna(False).astype(bool)
    short_ok = (both3["gap"] > 0) & (~both3["d_htf_up"].fillna(False).astype(bool))
    add("fade only with-trend-against", "both_fill", 0.03, win_sz, both3[long_ok | short_ok])
    add("ADX<22 both", "both_fill", 0.03, win_sz, both3[pd.to_numeric(both3["d_adx"], errors="coerce") < 22])
    # longs oversold in uptrend, shorts overbought not in uptrend
    long_ok = (
        (both3["gap"] < 0)
        & both3["d_above_sma200"].fillna(False).astype(bool)
        & (pd.to_numeric(both3["d_rsi2"], errors="coerce") < 20)
    )
    short_ok = (
        (both3["gap"] > 0)
        & (~both3["d_htf_up"].fillna(False).astype(bool))
        & (pd.to_numeric(both3["d_rsi2"], errors="coerce") > 80)
    )
    add("Connors both ways", "both_fill", 0.03, win_sz, both3[long_ok | short_ok])
    add("idiosyncratic both |idx|<0.4%", "both_fill", 0.03, win_sz,
        both3[pd.to_numeric(both3["idx_gap"], errors="coerce").abs() < 0.004])

    print("\n-- extra: down ≥1% with quality filters (more samples) --")
    d1 = ev[(ev["gap"] <= -0.01) & (ev["gap"] >= -0.15)].copy()
    add("BASE down≥1% r5", "down_bounce", 0.01, story_sz, None)
    add("≥1% RSI14<45 skip chron", "down_bounce", 0.01, story_sz,
        d1[(~d1["chronic"]) & (pd.to_numeric(d1["d_rsi14"], errors="coerce") < 45)])
    add("≥1% SMA200 + RSI2<20", "down_bounce", 0.01, story_sz,
        d1[d1["d_above_sma200"].fillna(False).astype(bool) & (pd.to_numeric(d1["d_rsi2"], errors="coerce") < 20)])
    add("≥1% htf_up + RSI14<50", "down_bounce", 0.01, story_sz,
        d1[d1["d_htf_up"].fillna(False).astype(bool) & (pd.to_numeric(d1["d_rsi14"], errors="coerce") < 50)])
    add("≥1% RSI5<40 + SMA200", "down_bounce", 0.01, story_sz,
        d1[d1["d_above_sma200"].fillna(False).astype(bool) & (pd.to_numeric(d1["rsi5_prev"], errors="coerce") < 40)])

    ranked = sorted(packs, key=lambda x: (x[0].oos_pnl > 0, x[0].total_return_pct, x[0].win_rate), reverse=True)
    print("\n===== RANKED PACKS (OOS>0 first, then return, then WR) =====")
    print(f"{'ret%':>7} {'WR':>6} {'PF':>5} {'n':>4} {'DD':>6} {'OOS':>8}  name")
    for res, months, _ in ranked:
        if res.trades < 8:
            continue
        print(
            f"{res.total_return_pct:7.1f} {res.win_rate:6.1f} {res.profit_factor:5.2f} "
            f"{res.trades:4d} {res.max_dd_pct:6.1f} {res.oos_pnl:8.0f}  {res.name}"
        )
        if months:
            ms = " | ".join(f"{m['month'][-2:]} ₹{m['pnl']:.0f} {m['wr']:.0f}%" for m in months)
            print(f"         months: {ms}")


if __name__ == "__main__":
    main()
