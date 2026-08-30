"""Phase 4 — close the gap from 372% to 500%.

Leader: ST(14,3) first pullback, Minervini + Nifty>EMA50, 2% risk,
exit on Supertrend flip (avg win 25%, PF 3.75, 2025 gave back 29pts).

This round: Nifty ST regime (cash in corrections), breakeven after 1R,
dual Supertrend (fast entry / slow exit), ST-flip entries, any-touch
pullbacks, slower exit ST. Still 2% equity at risk, no leverage.
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
    F_RS63,
    F_TREND,
    START,
    TARGET_RET,
    _flags,
    _preload,
    supertrend_np,
)
from trading.services.position_sizing import calculate_position_size

OUT = os.path.join(ROOT, "data", "swing_500_phase4.json")


def st_pack(s, period, mult):
    if (period, mult) == (14, 3.0):
        st, d = s.st, s.st_dir
        first = s.first_touch
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
    return st, d, first


def collect(packs, nifty, start_ts, end_ts, period, mult, mode):
    """mode: first | any | flip"""
    out = []
    for sym, s in packs.items():
        st, d, first = st_pack(s, period, mult)
        n = len(s.c)
        for i in range(60, n - 1):
            ts = s.index[i]
            if ts < start_ts or ts > end_ts:
                continue
            st_line = st[i]
            fire = False
            if mode == "flip":
                fire = d[i] > 0 and d[i - 1] <= 0
            elif mode == "first":
                fire = (
                    d[i] > 0 and d[i - 1] > 0
                    and (not np.isnan(st_line))
                    and s.l[i] <= st_line * 1.008
                    and s.c[i] > st_line
                    and bool(first[i])
                )
            elif mode == "any":
                fire = (
                    d[i] > 0
                    and (not np.isnan(st_line))
                    and s.l[i] <= st_line * 1.008
                    and s.c[i] > st_line
                )
            if not fire:
                continue
            flags = _flags(s, i, nifty)
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r63 = float(s.ret63[i]) if not np.isnan(s.ret63[i]) else -9.0
            stop_ref = float(s.l[i])
            if not np.isnan(st_line):
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
            })
    return out


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
    be_r: float,
    mkt_st: bool,
    mkt_exit: bool,
    exit_st: tuple[int, float] | None,
    rs_top: int = 0,
) -> dict[str, Any]:
    by_day = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    exit_st_dir = {}
    if exit_st is not None:
        for sym, s in packs.items():
            _, d, _ = st_pack(s, exit_st[0], exit_st[1])
            exit_st_dir[sym] = d

    cash = float(CAPITAL)
    opens: dict[str, dict] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades = []
    peak_eq = CAPITAL
    max_dd = 0.0
    peak_par = 0

    def equity():
        return cash + sum(p["notional"] for p in opens.values())

    def nifty_st_bull(ts) -> bool:
        if nifty is None:
            return True
        i = nifty.loc.get(ts)
        if i is None:
            return True
        return nifty.st_dir[i] > 0

    for ts in calendar:
        # portfolio crash protection
        if mkt_exit and nifty is not None:
            ni = nifty.loc.get(ts)
            if ni is not None and ni > 0 and nifty.st_dir[ni] <= 0 and nifty.st_dir[ni - 1] > 0:
                for sym, pos in list(opens.items()):
                    s = packs.get(sym)
                    if s is None:
                        continue
                    i = s.loc.get(ts)
                    if i is None:
                        continue
                    close = float(s.c[i])
                    pnl = (close - pos["entry"]) * pos["qty"]
                    cash += pos["notional"] + pnl
                    trades.append({
                        "symbol": sym, "pnl": pnl,
                        "pnl_pct": (close / pos["entry"] - 1) * 100,
                        "reason": "mkt_exit", "hold": pos["hold"],
                        "entry": str(pos["entry_ts"].date()), "exit": str(ts.date()),
                    })
                    last_exit[sym] = ts
                opens.clear()

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
            st_d = exit_st_dir[sym][i] if sym in exit_st_dir else s.st_dir[i]

            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time"
            elif exit_mode in ("st_flip", "st_flip_ch") and st_d <= 0:
                exit_p, reason = close, "st_flip"
            elif exit_mode == "ema50" and close < s.ema50[i]:
                exit_p, reason = close, "ema50"
            if exit_p is None:
                if be_r > 0 and r_now >= be_r:
                    be = pos["entry"] + 0.05 * pos["risk"]
                    if be > pos["stop"] and be < close:
                        pos["stop"] = float(be)
                if exit_mode in ("chandelier", "st_flip_ch") and atr and atr > 0:
                    cand = pos["hh"] - trail_atr * atr
                    if cand > pos["stop"] and cand < close:
                        pos["stop"] = float(cand)
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
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

        if mkt_st and not nifty_st_bull(ts):
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
            qty = min(int(ps.quantity), int(cash // entry), int((eq * max_pos_pct) // entry))
            if qty <= 0:
                continue
            notional = qty * entry
            cash -= notional
            target = (entry + risk * target_rr) if target_rr > 0 else 0.0
            opens[sym] = {
                "entry": entry, "stop": stop, "target": target, "qty": qty,
                "notional": notional, "hold": 0, "entry_ts": ts,
                "risk": risk, "atr": atr, "hh": entry,
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
        "trades": trades,
    }


def fmt(r, tag=""):
    flag = " *** HIT 500% ***" if r["ret"] >= TARGET_RET else ""
    print(
        f"{tag}{r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} "
        f"n={r['n']:3d} hold={r['avg_hold']} W={r['avg_win']} L={r['avg_loss']} "
        f"{r.get('name','')} {r['years']}{flag}",
        flush=True,
    )


def main():
    t0 = time.time()
    packs, nifty, calendar = _preload()
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    print(f"Phase 4  packs={len(packs)} days={len(calendar)} {time.time()-t0:.1f}s", flush=True)

    rows = []
    cache = {}
    for period, mult, mode in (
        (14, 3.0, "first"), (14, 3.0, "any"), (14, 3.0, "flip"),
        (21, 3.0, "first"), (21, 3.0, "flip"),
        (10, 3.0, "first"), (10, 3.0, "flip"),
        (14, 4.0, "first"), (21, 4.0, "first"),
        (14, 2.5, "first"),
    ):
        cache[(period, mult, mode)] = collect(packs, nifty, start_ts, end_ts, period, mult, mode)
        print(f"  ST{period},{mult:g} {mode} sig={len(cache[(period, mult, mode)])}", flush=True)

    def go(name, sigs, **kw):
        r = simulate(packs, nifty, calendar, sigs, **kw)
        r["name"] = name
        rows.append(r)
        fmt(r)
        return r

    base = dict(
        need_flags=F_MINERVINI | F_MKT50,
        exit_mode="st_flip",
        max_hold=250, cooldown=8, max_open=4, max_new=2, max_pos_pct=0.80,
        trail_atr=4.5, min_stop_pct=0.018, max_stop_pct=0.12, target_rr=0.0,
        be_r=0.0, mkt_st=False, mkt_exit=False, exit_st=None, rs_top=0,
    )

    print("\n### recreate leader + overlays ###", flush=True)
    sig = cache[(14, 3.0, "first")]
    go("leader st_flip", sig, **base)

    for be in (0.0, 1.0, 1.5):
        kw = dict(base); kw["be_r"] = be
        go(f"st_flip beR={be:g}", sig, **kw)

    kw = dict(base); kw["mkt_st"] = True
    go("st_flip + niftyST entries", sig, **kw)
    kw = dict(base); kw["mkt_exit"] = True
    go("st_flip + niftyST flatten", sig, **kw)
    kw = dict(base); kw["mkt_st"] = True; kw["mkt_exit"] = True
    go("st_flip + niftyST both", sig, **kw)

    kw = dict(base); kw["mkt_st"] = True; kw["mkt_exit"] = True; kw["be_r"] = 1.0
    go("st_flip niftyST both be1", sig, **kw)

    print("\n### dual ST / other entries ###", flush=True)
    for key, label in (
        ((14, 3.0, "first"), "first14,3"),
        ((14, 3.0, "any"), "any14,3"),
        ((14, 3.0, "flip"), "flip14,3"),
        ((21, 3.0, "first"), "first21,3"),
        ((21, 3.0, "flip"), "flip21,3"),
        ((10, 3.0, "first"), "first10,3"),
        ((10, 3.0, "flip"), "flip10,3"),
        ((14, 4.0, "first"), "first14,4"),
        ((21, 4.0, "first"), "first21,4"),
        ((14, 2.5, "first"), "first14,2.5"),
    ):
        for exit_st in (None, (21, 3.0), (14, 4.0), (21, 4.0), (14, 3.0)):
            kw = dict(base); kw["exit_st"] = exit_st
            go(f"{label} exitST={exit_st}", cache[key], **kw)

    print("\n### st_flip_ch + regime ###", flush=True)
    for trail in (3.0, 4.5, 6.0):
        for mkt in (False, True):
            kw = dict(base)
            kw["exit_mode"] = "st_flip_ch"
            kw["trail_atr"] = trail
            kw["mkt_st"] = mkt
            kw["mkt_exit"] = mkt
            kw["be_r"] = 1.0
            go(f"flip_ch t={trail:g} mkt={mkt}", sig, **kw)

    print("\n### filters / size / rank on best so far ###", flush=True)
    rows.sort(key=lambda x: x["ret"], reverse=True)
    for r in rows[:8]:
        fmt(r, tag="  top ")

    # compact grid on first14,3 with nifty overlays and size knobs
    for flags, fname in (
        (F_MINERVINI | F_MKT50, "min_mkt"),
        (F_MINERVINI, "min"),
        (F_TREND | F_MKT50, "trend_mkt"),
        (F_TREND | F_RS63 | F_MKT50, "trend_rs_mkt"),
        (F_MINERVINI | F_MKT50 | F_RS63, "min_rs_mkt"),
    ):
        for mkt_st, mkt_exit in ((False, False), (True, False), (True, True), (False, True)):
            for max_open, max_pos in ((3, 0.85), (4, 0.80), (2, 0.95)):
                kw = dict(base)
                kw.update(need_flags=flags, mkt_st=mkt_st, mkt_exit=mkt_exit,
                          max_open=max_open, max_pos_pct=max_pos, be_r=1.0,
                          exit_st=(21, 3.0))
                go(f"first14 {fname} mkt{int(mkt_st)}{int(mkt_exit)} o={max_open}", sig, **kw)

    # also try flip entry with same overlays
    fsig = cache[(14, 3.0, "flip")]
    for mkt_st, mkt_exit in ((False, False), (True, True)):
        for exit_st in (None, (21, 3.0), (14, 4.0)):
            kw = dict(base)
            kw.update(mkt_st=mkt_st, mkt_exit=mkt_exit, exit_st=exit_st, be_r=1.0)
            go(f"flip14 mkt{int(mkt_st)}{int(mkt_exit)} exitST={exit_st}", fsig, **kw)

    # stop band on the leader family
    for min_sp, max_sp in ((0.012, 0.10), (0.018, 0.12), (0.025, 0.09), (0.015, 0.08), (0.02, 0.07)):
        kw = dict(base)
        kw.update(min_stop_pct=min_sp, max_stop_pct=max_sp, be_r=1.0,
                  mkt_st=True, mkt_exit=True, exit_st=(21, 3.0))
        go(f"first14 stop {min_sp}-{max_sp} regime", sig, **kw)

    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n===== PHASE 4 TOP 15 =====", flush=True)
    for r in rows[:15]:
        fmt(r)

    hits = [r for r in rows if r["ret"] >= TARGET_RET]
    best = rows[0]
    print(f"\nHits={len(hits)} BEST {best['ret']}% {best['name']}", flush=True)
    slim = [{k: v for k, v in r.items() if k != "trades"} for r in rows[:25]]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "best": {k: v for k, v in best.items() if k != "trades"},
            "best_trades": best.get("trades", []),
            "top": slim,
            "hits": [{k: v for k, v in r.items() if k != "trades"} for r in hits],
        }, f, indent=2, default=str)
    print(f"Wrote {OUT} elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
