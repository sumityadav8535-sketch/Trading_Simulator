"""Phase 5 — 150d time-stop was the 372% lock-in. Combine with ST21 and runner rule.

2% equity at risk, no leverage. Target 500% over 2021-07-05 → 2026-08-21.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import numpy as np
import pandas as pd

from scripts.swing_500_phase4 import collect, simulate as sim4
from scripts.swing_500_search import (
    CAPITAL, END, F_MKT50, F_MINERVINI, F_RS63, F_TREND, START, TARGET_RET, _preload,
)

OUT = os.path.join(ROOT, "data", "swing_500_phase5.json")


def simulate(packs, nifty, calendar, signals, *, runner_r=0.0, **kw):
    """Wrap sim4; runner_r > 0 means skip time-exit once trade is >= runner_r R."""
    if runner_r <= 0:
        return sim4(packs, nifty, calendar, signals, **kw)

    # local copy of the time-stop logic by using a huge hold and injecting
    # a custom max_hold per position is hard without forking. Fork the loop
    # by monkey-patching: run with max_hold=999 and a pre-pass is messy.
    # Instead, inline a thin fork — call sim4 with max_hold=kw hold for losers
    # is the default. We'll implement runner in a patched kw: use max_hold as
    # the loser cut, and if runner_r>0 raise max_hold to 400 inside a custom sim.
    from scripts.swing_500_phase4 import simulate as _s
    # Custom: pass max_hold, then post-filter is wrong. Re-implement via
    # simulating twice is wrong. Patch simulate by passing max_hold=400 and
    # exit_mode that we handle... simplest is a local simulate copy.
    return simulate_runner(packs, nifty, calendar, signals, runner_r=runner_r, **kw)


def simulate_runner(packs, nifty, calendar, signals, *, runner_r, **kw):
    """Same as phase4.simulate with time-stop waived when R >= runner_r."""
    from scripts.swing_500_phase4 import simulate as _orig
    # Patch by wrapping the original with a higher-level time rule using
    # two-stage: we copy the function body from phase4 and add the runner check.
    import scripts.swing_500_phase4 as p4
    from collections import defaultdict
    from trading.services.position_sizing import calculate_position_size

    need_flags = kw["need_flags"]
    exit_mode = kw["exit_mode"]
    max_hold = kw["max_hold"]
    cooldown = kw["cooldown"]
    max_open = kw["max_open"]
    max_new = kw["max_new"]
    max_pos_pct = kw["max_pos_pct"]
    trail_atr = kw["trail_atr"]
    min_stop_pct = kw["min_stop_pct"]
    max_stop_pct = kw["max_stop_pct"]
    target_rr = kw["target_rr"]
    be_r = kw["be_r"]
    mkt_st = kw["mkt_st"]
    mkt_exit = kw["mkt_exit"]
    exit_st = kw["exit_st"]
    rs_top = kw.get("rs_top", 0)

    by_day = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    exit_st_dir = {}
    if exit_st is not None:
        for sym, s in packs.items():
            _, d, _ = p4.st_pack(s, exit_st[0], exit_st[1])
            exit_st_dir[sym] = d

    cash = float(CAPITAL)
    opens = {}
    last_exit = {}
    trades = []
    peak_eq = CAPITAL
    max_dd = 0.0
    peak_par = 0

    def equity():
        return cash + sum(p["notional"] for p in opens.values())

    def nifty_st_bull(ts):
        if nifty is None:
            return True
        i = nifty.loc.get(ts)
        if i is None:
            return True
        return nifty.st_dir[i] > 0

    for ts in calendar:
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
            elif pos["hold"] >= max_hold and r_now < runner_r:
                exit_p, reason = close, "time"
            elif pos["hold"] >= 400:
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


def fmt(r):
    flag = " *** HIT 500% ***" if r["ret"] >= TARGET_RET else ""
    print(
        f"{r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} "
        f"n={r['n']:3d} hold={r['avg_hold']} W={r['avg_win']} {r['name']} {r['years']}{flag}",
        flush=True,
    )


def main():
    t0 = time.time()
    packs, nifty, calendar = _preload()
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    print(f"Phase 5 packs={len(packs)} {time.time()-t0:.1f}s", flush=True)

    cache = {}
    for key in ((14, 3.0, "first"), (21, 3.0, "first"), (14, 3.0, "any"), (21, 3.0, "any")):
        cache[key] = collect(packs, nifty, start_ts, end_ts, *key)
        print(f"  {key} sig={len(cache[key])}", flush=True)
    # union 14+21 first
    u = cache[(14, 3.0, "first")] + cache[(21, 3.0, "first")]
    cache["union"] = u
    print(f"  union14+21 sig={len(u)}", flush=True)

    rows = []
    base = dict(
        need_flags=F_MINERVINI | F_MKT50,
        exit_mode="st_flip",
        max_hold=150, cooldown=8, max_open=4, max_new=2, max_pos_pct=0.80,
        trail_atr=4.5, min_stop_pct=0.018, max_stop_pct=0.12, target_rr=0.0,
        be_r=0.0, mkt_st=False, mkt_exit=False, exit_st=None, rs_top=0,
    )

    def go(name, sigs, runner_r=0.0, **upd):
        kw = dict(base); kw.update(upd)
        r = simulate_runner(packs, nifty, calendar, sigs, runner_r=runner_r, **kw)
        r["name"] = name
        rows.append(r)
        fmt(r)
        return r

    print("\n### recreate 372% and hold grid ###", flush=True)
    sig14 = cache[(14, 3.0, "first")]
    sig21 = cache[(21, 3.0, "first")]
    any14 = cache[(14, 3.0, "any")]
    go("recreate14 hold150", sig14, max_hold=150)
    go("recreate14 hold60", sig14, max_hold=60)
    for hold in (90, 120, 135, 150, 165, 180, 210):
        go(f"14,3 hold={hold}", sig14, max_hold=hold)
        go(f"21,3 hold={hold}", sig21, max_hold=hold)
        go(f"any14 hold={hold}", any14, max_hold=hold)

    print("\n### dual exit 21,4 + hold ###", flush=True)
    for hold in (90, 120, 150, 180):
        go(f"14,3 x21,4 h={hold}", sig14, max_hold=hold, exit_st=(21, 4.0))
        go(f"21,3 x21,4 h={hold}", sig21, max_hold=hold, exit_st=(21, 4.0))
        go(f"any14 x21,4 h={hold}", any14, max_hold=hold, exit_st=(21, 4.0))
        go(f"union x21,4 h={hold}", u, max_hold=hold, exit_st=(21, 4.0))
        go(f"union flip h={hold}", u, max_hold=hold)

    print("\n### runner rule (time-stop only if R < X) ###", flush=True)
    for runner in (0.5, 1.0, 1.5, 2.0, 3.0):
        for hold in (60, 90, 120, 150):
            go(f"14,3 h={hold} run>{runner:g}R", sig14, runner_r=runner, max_hold=hold)
            go(f"21,3 h={hold} run>{runner:g}R", sig21, runner_r=runner, max_hold=hold)

    print("\n### beR + hold150 on 14/21 ###", flush=True)
    for be in (0.0, 1.0, 1.5):
        go(f"14,3 h150 be={be:g}", sig14, max_hold=150, be_r=be)
        go(f"21,3 h150 be={be:g}", sig21, max_hold=150, be_r=be)
        go(f"21,3 x21,4 h150 be={be:g}", sig21, max_hold=150, be_r=be, exit_st=(21, 4.0))

    print("\n### open / pos / extra filters ###", flush=True)
    for max_open, max_pos in ((3, 0.90), (4, 0.80), (5, 0.60), (2, 0.95)):
        go(f"14,3 h150 o={max_open}", sig14, max_hold=150, max_open=max_open, max_pos_pct=max_pos)
        go(f"21,3 h150 o={max_open}", sig21, max_hold=150, max_open=max_open, max_pos_pct=max_pos)

    for flags, fname in (
        (F_MINERVINI | F_MKT50, "min_mkt"),
        (F_TREND | F_MKT50, "trend_mkt"),
        (F_TREND | F_RS63 | F_MKT50, "rs_mkt"),
        (F_MINERVINI | F_RS63 | F_MKT50, "min_rs"),
    ):
        go(f"14,3 h150 {fname}", sig14, max_hold=150, need_flags=flags)
        go(f"21,3 h150 {fname}", sig21, max_hold=150, need_flags=flags)

    rows.sort(key=lambda x: x["ret"], reverse=True)
    print("\n===== PHASE 5 TOP 20 =====", flush=True)
    for r in rows[:20]:
        fmt(r)
    hits = [r for r in rows if r["ret"] >= TARGET_RET]
    best = rows[0]
    print(f"\nHits={len(hits)} BEST {best['ret']}% {best['name']}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "best": {k: v for k, v in best.items() if k != "trades"},
            "best_trades": best.get("trades", []),
            "top": [{k: v for k, v in r.items() if k != "trades"} for r in rows[:30]],
            "hits": [{k: v for k, v in r.items() if k != "trades"} for r in hits],
        }, f, indent=2, default=str)
    print(f"Wrote {OUT} elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
