"""Diagnose Supertrend Quality stop-outs and search WR+return upgrades."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import numpy as np
import pandas as pd

from stage_analysis_v2.services.backtester import _preload_frames
from stage_analysis_v2.services.strategy_catalog import ST_MAX_STOP_PCT, ST_PULLBACK_TOL
from stage_analysis_v2.services.supertrend_swing import _STPack, _passes_pack
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
START = date.today() - timedelta(days=365)
END = date.today()
RISK = 3.0
HOLD = 90
COOLDOWN = 10
MAX_POS = 0.50
MAX_NEW = 5


def collect(packs, nifty_ret20, start_ts, end_ts):
    out = []
    for sym, s in packs.items():
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
            e50, e200 = s.ema50[i], s.ema200[i]
            if np.isnan(e50) or np.isnan(e200):
                continue
            ema_trend = s.c[i] > e50 > e200
            rsi = s.rsi[i]
            rsi_healthy = (not np.isnan(rsi)) and 45.0 <= rsi <= 70.0
            adx = s.adx[i]
            adx20 = (not np.isnan(adx)) and adx >= 20.0
            if not _passes_pack("quality", ema_trend, rsi_healthy, adx20):
                continue
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            r20 = float(s.ret20[i]) if not np.isnan(s.ret20[i]) else -9.0
            n_r = 0.0
            if ts in nifty_ret20.index and pd.notna(nifty_ret20.loc[ts]):
                n_r = float(nifty_ret20.loc[ts])
            rsi_up = i > 0 and not np.isnan(s.rsi[i - 1]) and s.rsi[i] > s.rsi[i - 1]
            above_e20 = s.c[i] > s.ema20[i]
            out.append({
                "symbol": sym,
                "i": i,
                "sig_ts": ts,
                "entry_ts": s.index[i + 1],
                "st": float(st_line),
                "low": float(s.l[i]),
                "atr": atr,
                "score": r20,
                "rs": min(99.0, max(1.0, 50.0 + (r20 - n_r) * 200.0)),
                "close_loc": float(s.close_loc[i]),
                "bull": bool(s.bull[i]),
                "st_age": int(s.st_age[i]),
                "first_touch": bool(s.first_touch[i]),
                "rsi_up": bool(rsi_up),
                "above_e20": bool(above_e20),
                "adx": float(adx) if not np.isnan(adx) else 0.0,
            })
    return out


def stop_price(sig, entry, mode, atr_k):
    atr = sig["atr"]
    if mode == "st":
        raw = sig["st"]
    elif mode == "low":
        raw = min(sig["low"], sig["st"]) - atr_k * atr
    elif mode == "st_atr":
        raw = sig["st"] - atr_k * atr
    else:
        raw = sig["st"]
    if raw <= 0 or raw >= entry:
        raw = entry - (1.5 * atr if atr > 0 else entry * 0.04)
    return float(raw) if 0 < raw < entry else 0.0


def simulate(packs, calendar, sigs, *, stop_mode, atr_k, trail_mode, trail_k,
             min_stop_pct, filters, delay_trail_r=0.0):
    by_day = defaultdict(list)
    for sig in sigs:
        if filters.get("bull") and not sig["bull"]:
            continue
        if filters.get("upper") is not None and sig["close_loc"] < filters["upper"]:
            continue
        if filters.get("min_age") and sig["st_age"] < filters["min_age"]:
            continue
        if filters.get("first") and not sig["first_touch"]:
            continue
        if filters.get("rsi_up") and not sig["rsi_up"]:
            continue
        if filters.get("ema20") and not sig["above_e20"]:
            continue
        if filters.get("adx25") and sig["adx"] < 25:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(CAPITAL)
    opens = {}
    last_exit = {}
    trades = []
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
            close, low = float(s.c[i]), float(s.l[i])
            st_line = s.st[i]
            atr = float(s.atr[i]) if not np.isnan(s.atr[i]) else 0.0
            # trail
            can_trail = True
            if delay_trail_r > 0:
                r_now = (close - pos["entry"]) / pos["risk"] if pos["risk"] > 0 else 0
                can_trail = r_now >= delay_trail_r
            if can_trail and trail_mode != "none" and not np.isnan(st_line):
                if trail_mode == "st":
                    new_stop = float(st_line)
                else:
                    new_stop = float(st_line) - trail_k * atr
                if new_stop > pos["stop"] and new_stop < close:
                    pos["stop"] = new_stop
            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop_loss"
            elif pos["hold"] >= HOLD:
                exit_p, reason = close, "time_exit"
            elif s.st_dir[i] <= 0:
                exit_p, reason = close, "st_flip"
            if exit_p is None:
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "hold": pos["hold"], "pnl": pnl, "reason": reason,
                "stop_pct": pos["stop_pct"],
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        taken = 0
        for sig in day:
            if taken >= MAX_NEW:
                break
            sym = sig["symbol"]
            if sym in opens:
                continue
            s = packs.get(sym)
            if s is None:
                continue
            prev = last_exit.get(sym)
            if prev is not None and (ts - prev).days < COOLDOWN:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            entry = float(s.o[i])
            stop = stop_price(sig, entry, stop_mode, atr_k)
            if stop <= 0 or stop >= entry:
                continue
            risk = entry - stop
            sp = risk / entry
            if sp > ST_MAX_STOP_PCT or sp < min_stop_pct:
                continue
            equity = cash + sum(p["notional"] for p in opens.values())
            if cash <= 0 or equity <= 0:
                skipped += 1
                continue
            ps = calculate_position_size(equity, RISK, entry, stop)
            max_n = equity * MAX_POS
            qty = min(int(ps.quantity), int(cash // entry), int(max_n // entry))
            if qty <= 0:
                skipped += 1
                continue
            notional = qty * entry
            cash -= notional
            opens[sym] = {
                "entry": entry, "stop": stop, "qty": qty, "notional": notional,
                "hold": 0, "risk": risk, "stop_pct": sp * 100,
            }
            taken += 1
            if s.l[i] <= stop:
                pnl = (stop - entry) * qty
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "hold": 0, "pnl": pnl, "reason": "stop_loss",
                    "stop_pct": sp * 100,
                })
                last_exit[sym] = ts
                opens.pop(sym, None)

    if opens:
        last = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs[sym]
            i = s.loc.get(last, len(s.c) - 1)
            pnl = (float(s.c[i]) - pos["entry"]) * pos["qty"]
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "hold": pos["hold"], "pnl": pnl, "reason": "eod_force",
                "stop_pct": pos["stop_pct"],
            })

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n * 100 if n else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    ret = (cash - CAPITAL) / CAPITAL * 100
    sl = [t for t in trades if t["reason"] == "stop_loss"]
    sl0 = sum(1 for t in sl if t["hold"] == 0)
    sl3 = sum(1 for t in sl if t["hold"] <= 3)
    return {
        "n": n, "wr": round(wr, 2), "ret": round(ret, 2),
        "pf": round(gp / gl, 2), "wins": len(wins),
        "sl": len(sl), "sl0": sl0, "sl3": sl3,
        "avg_hold_loss": round(sum(t["hold"] for t in losses) / len(losses), 1) if losses else 0,
        "exits": dict(Counter(t["reason"] for t in trades)),
        "trades": trades,
    }


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"Loading {len(symbols)} symbols {START}→{END}", flush=True)
    frames = _preload_frames(symbols)
    packs = {}
    for i, (sym, df) in enumerate(frames.items()):
        packs[sym] = _STPack(sym, df, 14, 3.0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret20 = nifty["close"].pct_change(20) if not nifty.empty else pd.Series(dtype=float)
    st, et = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })
    raw = collect(packs, nifty_ret20, st, et)
    print(f"Quality raw signals: {len(raw)}  calendar={len(calendar)}", flush=True)

    base = simulate(
        packs, calendar, raw, stop_mode="st", atr_k=0, trail_mode="st", trail_k=0,
        min_stop_pct=0.0, filters={},
    )
    print("\n=== BASELINE Quality (stop=ST, trail=ST) ===")
    print(
        f"n={base['n']} WR={base['wr']}% ret={base['ret']}% PF={base['pf']} "
        f"SL={base['sl']} same-bar SL={base['sl0']} SL<=3d={base['sl3']} "
        f"avg_hold_loss={base['avg_hold_loss']} exits={base['exits']}"
    )
    if base["trades"]:
        med_stop = np.median([t["stop_pct"] for t in base["trades"]])
        print(f"median stop width {med_stop:.2f}%")

    filter_grid = [
        ("none", {}),
        ("bull", {"bull": True}),
        ("upper50", {"upper": 0.50}),
        ("upper60", {"upper": 0.60}),
        ("bull+u50", {"bull": True, "upper": 0.50}),
        ("bull+u60", {"bull": True, "upper": 0.60}),
        ("age5", {"min_age": 5}),
        ("age8", {"min_age": 8}),
        ("first", {"first": True}),
        ("rsi_up", {"rsi_up": True}),
        ("ema20", {"ema20": True}),
        ("adx25", {"adx25": True}),
        ("bull+u50+age5", {"bull": True, "upper": 0.50, "min_age": 5}),
        ("bull+u50+rsi", {"bull": True, "upper": 0.50, "rsi_up": True}),
        ("bull+u50+e20", {"bull": True, "upper": 0.50, "ema20": True}),
        ("strict", {"bull": True, "upper": 0.55, "min_age": 5, "rsi_up": True}),
    ]
    stop_grid = [
        ("st", 0.0),
        ("st_atr", 0.25),
        ("st_atr", 0.50),
        ("st_atr", 0.75),
        ("low", 0.25),
        ("low", 0.50),
        ("low", 0.75),
    ]
    trail_grid = [
        ("st", 0.0, 0.0),
        ("st_atr", 0.25, 0.0),
        ("st_atr", 0.50, 0.0),
        ("st", 0.0, 1.0),  # delay trail until 1R
        ("st_atr", 0.35, 1.0),
    ]
    min_stops = [0.0, 0.012, 0.018]

    rows = []
    print("\nSearching...", flush=True)
    for fname, filt in filter_grid:
        for sm, ak in stop_grid:
            for tm, tk, delay in trail_grid:
                for ms in min_stops:
                    r = simulate(
                        packs, calendar, raw, stop_mode=sm, atr_k=ak,
                        trail_mode=tm, trail_k=tk, min_stop_pct=ms,
                        filters=filt, delay_trail_r=delay,
                    )
                    r.update({
                        "filter": fname, "stop": f"{sm}@{ak:g}",
                        "trail": f"{tm}@{tk:g}+{delay:g}R", "minsp": ms,
                    })
                    rows.append(r)

    usable = [r for r in rows if r["n"] >= 20]
    print(f"Sims={len(rows)} usable={len(usable)}")
    print(f"Baseline WR={base['wr']} ret={base['ret']}")

    better = [
        r for r in usable
        if r["wr"] >= max(base["wr"] + 8, 35.0) and r["ret"] >= base["ret"] - 0.5
    ]
    better.sort(key=lambda x: (x["ret"], x["wr"], x["pf"]), reverse=True)
    print(f"\n=== WR>=max(base+8,35) AND ret>=baseline-0.5  ({len(better)}) ===")
    for r in better[:20]:
        print(
            f"  {r['filter']:<18} stop={r['stop']:<10} trail={r['trail']:<16} "
            f"minsp={r['minsp']:.3f} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+7.1f}% PF={r['pf']:.2f} sl0={r['sl0']}"
        )

    wr40 = [r for r in usable if r["wr"] >= 40 and r["ret"] >= base["ret"]]
    wr40.sort(key=lambda x: (x["ret"], x["wr"]), reverse=True)
    print(f"\n=== WR>=40 AND ret>=baseline ({len(wr40)}) ===")
    for r in wr40[:15]:
        print(
            f"  {r['filter']:<18} stop={r['stop']:<10} trail={r['trail']:<16} "
            f"minsp={r['minsp']:.3f} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+7.1f}% PF={r['pf']:.2f} sl0={r['sl0']}"
        )

    top_wr = sorted([r for r in usable if r["ret"] >= base["ret"]], key=lambda x: x["wr"], reverse=True)
    print("\n=== HIGHEST WR with ret>=baseline ===")
    for r in top_wr[:10]:
        print(
            f"  {r['filter']:<18} stop={r['stop']:<10} trail={r['trail']:<16} "
            f"minsp={r['minsp']:.3f} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+7.1f}% PF={r['pf']:.2f}"
        )

    top_ret = sorted(usable, key=lambda x: x["ret"], reverse=True)
    print("\n=== TOP RETURN overall ===")
    for r in top_ret[:10]:
        print(
            f"  {r['filter']:<18} stop={r['stop']:<10} trail={r['trail']:<16} "
            f"minsp={r['minsp']:.3f} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+7.1f}% PF={r['pf']:.2f}"
        )


if __name__ == "__main__":
    main()
