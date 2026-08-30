"""Phase 3 — focused push from 296% toward 500% over 5 years.

Baseline winner: Supertrend first-pullback + Minervini + Nifty>EMA50,
2% equity at risk, chandelier trail, max 4 names.

This round adds: true chandelier from running high, ST param search,
EMA20 union entries, weekly 20-week breakout, delayed trail, SMA50 ride,
RS-top gate, Nifty ADX chop filter. No leverage. 2% risk per trade.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import numpy as np
import pandas as pd

from scripts.swing_500_search import (
    CAPITAL,
    END,
    F_MKT50,
    F_MINERVINI,
    F_QULLA,
    F_RS63,
    F_TREND,
    START,
    TARGET_RET,
    _flags,
    _preload,
    supertrend_np,
)
from trading.services.position_sizing import calculate_position_size

OUT = os.path.join(ROOT, "data", "swing_500_phase3.json")


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return w.dropna(subset=["close"])


def collect_st(packs, nifty, start_ts, end_ts, period, mult, extra_ema20=False):
    out = []
    for sym, s in packs.items():
        if (period, mult) == (14, 3.0):
            st, d, first = s.st, s.st_dir, s.first_touch
        else:
            st, d = supertrend_np(s.h, s.l, s.c, period, mult)
            first = np.zeros(len(s.c), dtype=bool)
            seen = False
            for i in range(len(s.c)):
                if d[i] <= 0:
                    seen = False
                    continue
                line = st[i]
                tagged = (not np.isnan(line)) and s.l[i] <= line * 1.008 and s.c[i] > line
                if tagged and not seen:
                    first[i] = True
                    seen = True
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            kind = None
            st_line = st[i]
            if (
                d[i] > 0 and d[i - 1] > 0
                and (not np.isnan(st_line))
                and s.l[i] <= st_line * 1.008
                and s.c[i] > st_line
                and bool(first[i])
            ):
                kind = "stpb"
            elif extra_ema20 and (not np.isnan(s.ema20[i])) and (
                s.l[i] <= s.ema20[i] * 1.005
                and s.c[i] > s.ema20[i]
                and s.c[i] > s.o[i]
                and s.c[i - 1] <= s.ema20[i - 1] * 1.01
            ):
                kind = "ema20pb"
            if kind is None:
                continue
            flags = _flags(s, i, nifty)
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            stop_ref = float(s.l[i])
            if kind == "stpb":
                stop_ref = min(stop_ref, float(st_line))
            out.append({
                "symbol": sym,
                "entry_ts": s.index[i + 1],
                "sig_ts": ts,
                "flags": flags,
                "stop_ref": stop_ref,
                "atr": atr,
                "sig_low": float(s.l[i]),
                "score": r63,
                "kind": kind,
            })
    return out


def collect_weekly_donch(packs, frames, nifty, start_ts, end_ts):
    """20-week high breakout → enter next daily open. Stop = 10-week low."""
    out = []
    n_w = _to_weekly(pd.DataFrame({
        "open": nifty.o, "high": nifty.h, "low": nifty.l, "close": nifty.c, "volume": nifty.v,
    }, index=nifty.index)) if nifty is not None else None
    for sym, s in packs.items():
        df = pd.DataFrame({
            "open": s.o, "high": s.h, "low": s.l, "close": s.c, "volume": s.v,
        }, index=s.index)
        w = _to_weekly(df)
        if len(w) < 30:
            continue
        wh = w["high"].to_numpy(float)
        wl = w["low"].to_numpy(float)
        wc = w["close"].to_numpy(float)
        h20 = pd.Series(wh).rolling(20, min_periods=20).max().to_numpy()
        l10 = pd.Series(wl).rolling(10, min_periods=10).min().to_numpy()
        for i in range(20, len(w) - 1):
            ts = w.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            h20p = h20[i - 1]
            if np.isnan(h20p) or wc[i] < h20p or wc[i - 1] >= h20p:
                continue
            after = s.index[s.index > ts]
            if len(after) == 0:
                continue
            entry_ts = after[0]
            di = s.loc.get(entry_ts)
            if di is None or di <= 0:
                continue
            sig_i = di - 1
            flags = _flags(s, sig_i, nifty)
            atr = float(s.atr[sig_i]) if not np.isnan(s.atr[sig_i]) else 0.0
            r63 = float(s.ret63[sig_i]) if not np.isnan(s.ret63[sig_i]) else -9.0
            stop_ref = float(l10[i]) if not np.isnan(l10[i]) else float(s.l[sig_i])
            out.append({
                "symbol": sym,
                "entry_ts": entry_ts,
                "sig_ts": ts,
                "flags": flags,
                "stop_ref": stop_ref,
                "atr": atr,
                "sig_low": float(s.l[sig_i]),
                "score": r63,
                "kind": "w20",
            })
    return out


def nifty_adx_ok(nifty, ts, min_adx=0.0):
    if nifty is None or min_adx <= 0:
        return True
    i = nifty.loc.get(ts)
    if i is None:
        return True
    a = nifty.adx[i]
    return (not np.isnan(a)) and a >= min_adx


def simulate(
    packs, nifty, calendar, signals, *,
    need_flags: int,
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
    rs_top: int,
    delay_r: float,
    be_r: float,
    min_nifty_adx: float = 0.0,
    pyramid: bool = False,
    cost_pct: float = 0.0,
) -> dict[str, Any]:
    by_day = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(CAPITAL)
    opens: dict[str, dict] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades = []
    peak_eq = CAPITAL
    max_dd = 0.0
    peak_par = 0
    eq_year_end = {}

    def equity():
        return cash + sum(p["notional"] for p in opens.values())

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
            high, low, close = float(s.h[i]), float(s.l[i]), float(s.c[i])
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else pos["atr"]
            pos["hh"] = max(pos["hh"], high)
            r_now = (close - pos["entry"]) / pos["risk"] if pos["risk"] > 0 else 0.0

            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time"
            else:
                if exit_mode == "ema20" and close < s.ema20[i]:
                    exit_p, reason = close, "ema20"
                elif exit_mode == "sma50" and (not np.isnan(s.sma50[i])) and close < s.sma50[i]:
                    exit_p, reason = close, "sma50"
                elif exit_mode == "ema50" and close < s.ema50[i]:
                    exit_p, reason = close, "ema50"
                elif exit_mode == "st_flip" and s.st_dir[i] <= 0:
                    exit_p, reason = close, "st_flip"
            if exit_p is None:
                allow_trail = delay_r <= 0 or r_now >= delay_r
                if be_r > 0 and r_now >= be_r:
                    be = pos["entry"] + 0.05 * pos["risk"]
                    if be > pos["stop"] and be < close:
                        pos["stop"] = float(be)
                if allow_trail and exit_mode in ("chandelier", "hybrid"):
                    if atr and atr > 0:
                        cand = pos["hh"] - trail_atr * atr
                        if cand > pos["stop"] and cand < close:
                            pos["stop"] = float(cand)
                if allow_trail and exit_mode == "hybrid":
                    st_line = s.st[i]
                    if not np.isnan(st_line):
                        cand = float(st_line) - 0.5 * (atr or 0.0)
                        if cand > pos["stop"] and cand < close:
                            pos["stop"] = cand
                # optional pyramid at +1R (another 2% risk, same stop trail)
                if pyramid and not pos.get("py") and r_now >= 1.0 and len(opens) < max_open + 2:
                    eq = equity()
                    add_stop = pos["stop"]
                    if add_stop < close:
                        risk = close - add_stop
                        if risk / close >= min_stop_pct:
                            ps = calculate_position_size(eq, 2.0, close, add_stop)
                            qty = min(int(ps.quantity), int(cash // close) if close else 0)
                            if qty > 0:
                                notional = qty * close
                                cash -= notional
                                pos["qty"] += qty
                                pos["notional"] += notional
                                # weighted entry
                                pos["entry"] = (
                                    (pos["entry"] * (pos["qty"] - qty) + close * qty) / pos["qty"]
                                )
                                pos["py"] = True
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            if cost_pct:
                pnl -= (pos["entry"] + exit_p) * pos["qty"] * cost_pct / 100.0
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "pnl": pnl,
                "pnl_pct": (exit_p / pos["entry"] - 1) * 100,
                "reason": reason, "hold": pos["hold"],
                "entry": str(pos["entry_ts"].date()), "exit": str(ts.date()),
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        eq = equity()
        peak_eq = max(peak_eq, eq)
        if peak_eq:
            max_dd = max(max_dd, (peak_eq - eq) / peak_eq * 100)
        if ts.month == 12 and ts.day >= 28:
            eq_year_end[str(ts.year)] = round((eq / CAPITAL - 1) * 100, 1)

        if min_nifty_adx > 0 and not nifty_adx_ok(nifty, ts, min_nifty_adx):
            continue

        day_sigs = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        if rs_top > 0:
            day_sigs = day_sigs[:rs_top]
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
            stop = float(sig["stop_ref"] or 0.0)
            if stop <= 0 or stop >= entry:
                stop = entry - (1.5 * atr if atr > 0 else entry * 0.04)
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
            ps = calculate_position_size(eq, 2.0, entry, stop)
            qty_cash = int(cash // entry)
            qty_cap = int((eq * max_pos_pct) // entry)
            qty = min(int(ps.quantity), qty_cash, qty_cap)
            if qty <= 0:
                continue
            notional = qty * entry
            cash -= notional
            target = (entry + risk * target_rr) if target_rr > 0 else 0.0
            opens[sym] = {
                "entry": entry, "stop": stop, "target": target, "qty": qty,
                "notional": notional, "hold": 0, "entry_ts": ts,
                "risk": risk, "atr": atr, "hh": entry, "py": False,
            }
            peak_par = max(peak_par, len(opens))
            taken += 1
            if s.l[i] <= stop:
                pnl = (stop - entry) * qty
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "pnl": pnl, "pnl_pct": (stop / entry - 1) * 100,
                    "reason": "stop", "hold": 0,
                    "entry": str(ts.date()), "exit": str(ts.date()),
                })
                last_exit[sym] = ts
                opens.pop(sym, None)

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
                "symbol": sym, "pnl": pnl, "pnl_pct": (close / pos["entry"] - 1) * 100,
                "reason": "eod", "hold": pos["hold"],
                "entry": str(pos["entry_ts"].date()), "exit": str(last_ts.date()),
            })

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    by_year = defaultdict(float)
    for t in trades:
        by_year[t["exit"][:4]] += t["pnl"]
    years = {y: round(by_year[y] / CAPITAL * 100, 1) for y in sorted(by_year)}
    return {
        "ret": round((cash - CAPITAL) / CAPITAL * 100.0, 2),
        "final": round(cash, 2),
        "wr": round(len(wins) / n * 100, 2) if n else 0.0,
        "pf": round(gp / gl, 2) if n else 0.0,
        "dd": round(max_dd, 2),
        "n": n,
        "par": peak_par,
        "avg_hold": round(sum(t["hold"] for t in trades) / n, 1) if n else 0.0,
        "avg_win": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0.0,
        "years": years,
        "exits": dict(pd.Series([t["reason"] for t in trades]).value_counts().to_dict()) if n else {},
        "trades": trades if n <= 200 else trades[:80] + trades[-40:],
    }


def fmt(r, tag=""):
    flag = " *** HIT 500% ***" if r["ret"] >= TARGET_RET else ""
    print(
        f"{tag}{r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} "
        f"n={r['n']:4d} par={r['par']} hold={r['avg_hold']} win={r['avg_win']} "
        f"loss={r['avg_loss']} {r.get('name','')} years={r['years']}{flag}",
        flush=True,
    )


def main():
    t0 = time.time()
    print(f"Phase 3 focused  {START}→{END}  target ≥{TARGET_RET:g}%  2% risk", flush=True)
    packs, nifty, calendar = _preload()
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    print(f"packs={len(packs)} days={len(calendar)} {time.time()-t0:.1f}s", flush=True)

    rows = []

    def go(name, sigs, **kw):
        r = simulate(packs, nifty, calendar, sigs, **kw)
        r["name"] = name
        rows.append(r)
        fmt(r)
        return r

    base_kw = dict(
        need_flags=F_MINERVINI | F_MKT50,
        exit_mode="chandelier",
        max_hold=60, cooldown=8, max_open=4, max_new=2, max_pos_pct=0.80,
        trail_atr=3.0, min_stop_pct=0.018, max_stop_pct=0.12, target_rr=0.0,
        rs_top=0, delay_r=0.0, be_r=0.0, min_nifty_adx=0.0, pyramid=False,
    )

    print("\n### ST param + EMA20 union ###", flush=True)
    st_raw = {}
    for p, m in ((14, 3.0), (10, 3.0), (10, 2.0), (7, 3.0), (14, 2.0), (21, 3.0), (12, 2.5)):
        for extra in (False, True):
            key = (p, m, extra)
            st_raw[key] = collect_st(packs, nifty, start_ts, end_ts, p, m, extra)
            tag = f"ST{p},{m:g}{'+ema20' if extra else ''}"
            print(f"  {tag} sig={len(st_raw[key])}", flush=True)
            go(tag, st_raw[key], **base_kw)

    print("\n### weekly 20-week breakout ###", flush=True)
    w20 = collect_weekly_donch(packs, None, nifty, start_ts, end_ts)
    print(f"  weekly20 sig={len(w20)}", flush=True)
    for flags, fname in (
        (F_MINERVINI | F_MKT50, "minervini_mkt"),
        (F_TREND | F_MKT50, "trend_mkt"),
        (F_TREND | F_RS63 | F_MKT50, "trend_rs_mkt"),
        (0, "none"),
    ):
        kw = dict(base_kw)
        kw["need_flags"] = flags
        kw["max_hold"] = 180
        go(f"w20/{fname}", w20, **kw)

    # Tune the best ST so far
    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n### TOP after first pass ###", flush=True)
    for r in rows[:8]:
        fmt(r, tag="  ")

    # Identify best ST key by name prefix
    print("\n### trail / hold / concentration / delayed trail ###", flush=True)
    best_name = rows[0]["name"]
    # map name back to raw
    use_sigs = None
    for (p, m, extra), sigs in st_raw.items():
        tag = f"ST{p},{m:g}{'+ema20' if extra else ''}"
        if tag == best_name:
            use_sigs = sigs
            break
    if use_sigs is None:
        use_sigs = st_raw[(14, 3.0, False)]

    for trail in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        kw = dict(base_kw); kw["trail_atr"] = trail
        go(f"{best_name} trail={trail:g}", use_sigs, **kw)
    for hold in (40, 60, 90, 150, 250):
        kw = dict(base_kw); kw["max_hold"] = hold
        go(f"{best_name} hold={hold}", use_sigs, **kw)
    for exit_m in ("chandelier", "hybrid", "sma50", "ema20", "ema50", "st_flip"):
        kw = dict(base_kw); kw["exit_mode"] = exit_m; kw["max_hold"] = 150
        go(f"{best_name} exit={exit_m}", use_sigs, **kw)
    for max_open, max_new, max_pos in ((2, 1, 0.90), (3, 2, 0.70), (4, 2, 0.80), (5, 2, 0.50), (6, 3, 0.45)):
        kw = dict(base_kw); kw["max_open"] = max_open; kw["max_new"] = max_new; kw["max_pos_pct"] = max_pos
        go(f"{best_name} open={max_open} pos={max_pos}", use_sigs, **kw)
    for delay_r, be_r in ((0, 0), (0, 1), (1, 1), (1.5, 1), (2, 1.5)):
        kw = dict(base_kw); kw["delay_r"] = delay_r; kw["be_r"] = be_r
        go(f"{best_name} delayR={delay_r} beR={be_r}", use_sigs, **kw)
    for rr in (0.0, 3.0, 5.0, 8.0, 12.0):
        kw = dict(base_kw); kw["target_rr"] = rr
        go(f"{best_name} rr={rr:g}", use_sigs, **kw)
    for rs_top in (0, 1, 2, 3, 5):
        kw = dict(base_kw); kw["rs_top"] = rs_top; kw["max_new"] = min(2, rs_top or 2)
        go(f"{best_name} rs_top={rs_top}", use_sigs, **kw)
    for adx in (0, 15, 18, 22):
        kw = dict(base_kw); kw["min_nifty_adx"] = adx
        go(f"{best_name} niftyADX>={adx}", use_sigs, **kw)
    for min_sp, max_sp in ((0.012, 0.10), (0.018, 0.12), (0.025, 0.10), (0.02, 0.08), (0.015, 0.16)):
        kw = dict(base_kw); kw["min_stop_pct"] = min_sp; kw["max_stop_pct"] = max_sp
        go(f"{best_name} stop={min_sp}-{max_sp}", use_sigs, **kw)

    kw = dict(base_kw); kw["pyramid"] = True
    go(f"{best_name} pyramid+1R", use_sigs, **kw)

    # Combine weekly + daily ST
    combo = list(use_sigs) + list(w20)
    kw = dict(base_kw); kw["max_hold"] = 120
    go(f"{best_name}+w20", combo, **kw)

    # Other filters on best ST
    print("\n### filter sweep on best ST ###", flush=True)
    best_st_sigs = use_sigs
    for flags, fname in (
        (F_MINERVINI | F_MKT50, "minervini_mkt"),
        (F_MINERVINI, "minervini"),
        (F_TREND | F_MKT50, "trend_mkt"),
        (F_TREND | F_RS63 | F_MKT50, "trend_rs_mkt"),
        (F_QULLA | F_MKT50, "qulla_mkt"),
        (F_TREND | F_MKT50 | F_QULLA, "trend_qulla_mkt"),
    ):
        kw = dict(base_kw); kw["need_flags"] = flags
        go(f"{best_name}/{fname}", best_st_sigs, **kw)

    # Take current best and do a small combo grid
    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n### combo grid around leader ###", flush=True)
    leader = rows[0]
    print(f"leader {leader['name']} {leader['ret']}%", flush=True)

    # re-parse is hard; run a compact grid on ST14,3 and ST10,3 ± ema20
    for key in ((14, 3.0, False), (14, 3.0, True), (10, 3.0, False), (10, 2.0, False), (7, 3.0, False)):
        sigs = st_raw[key]
        tag = f"ST{key[0]},{key[1]:g}{'+ema20' if key[2] else ''}"
        for trail in (2.5, 3.0, 3.5, 4.5):
            for hold in (60, 120, 250):
                for max_open, max_pos in ((3, 0.80), (4, 0.80), (2, 0.90)):
                    for exit_m in ("chandelier", "sma50"):
                        kw = dict(base_kw)
                        kw.update(trail_atr=trail, max_hold=hold, max_open=max_open,
                                  max_new=2, max_pos_pct=max_pos, exit_mode=exit_m,
                                  be_r=1.0)
                        go(f"{tag} t={trail} h={hold} o={max_open} {exit_m}", sigs, **kw)

    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n===== PHASE 3 TOP 20 =====", flush=True)
    for r in rows[:20]:
        fmt(r)

    hits = [r for r in rows if r["ret"] >= TARGET_RET]
    best = rows[0]
    slim = [{k: v for k, v in r.items() if k != "trades"} for r in rows[:40]]
    payload = {
        "best": {k: v for k, v in best.items() if k != "trades"},
        "best_trades": best.get("trades", []),
        "top": slim,
        "hits": [{k: v for k, v in r.items() if k != "trades"} for r in hits],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nHits={len(hits)}  BEST {best['ret']}%  {best['name']}", flush=True)
    print(f"Wrote {OUT}  elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
