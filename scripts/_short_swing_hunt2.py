"""Phase 2 hunt: size/concentration + rotation to clear 100% last 1y."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts._short_swing_hunt import (  # noqa: E402
    CAPITAL,
    F_QULLA,
    F_RS63,
    F_TREND,
    collect,
    nifty_ret,
    preload,
    simulate,
    slice_cal,
)

END = date.today()
WINDOWS = {
    "1y": (END - timedelta(days=365), END),
    "2y": (END - timedelta(days=365 * 2), END),
    "3y": (END - timedelta(days=365 * 3), END),
    "4y": (END - timedelta(days=365 * 4), END),
    "5y": (END - timedelta(days=365 * 5), END),
}
OUT = os.path.join(ROOT, "data", "short_swing_hunt2.json")


def run_one(packs, cal, sigs, flags, **kw):
    return simulate(
        packs, cal, sigs,
        need_flags=flags,
        risk_pct=kw.get("risk", 5.0),
        exit_mode=kw.get("exit", "chandelier"),
        max_hold=kw.get("hold", 15),
        cooldown=kw.get("cd", 8),
        max_open=kw.get("open", 3),
        max_new=kw.get("new", 3),
        max_pos_pct=kw.get("pos", 0.8),
        trail_atr=kw.get("trail", 2.5),
        min_stop_pct=kw.get("min_sp", 0.012),
        max_stop_pct=kw.get("max_sp", 0.12),
        target_rr=kw.get("rr", 2.5),
        keep_trades=kw.get("keep", False),
    )


def main() -> None:
    t0 = time.time()
    packs, nifty, calendar = preload()
    print(f"packs={len(packs)} days={len(calendar)}", flush=True)
    start_ts = __import__("pandas").Timestamp(WINDOWS["5y"][0])
    end_ts = __import__("pandas").Timestamp(END)
    cal1 = slice_cal(calendar, *WINDOWS["1y"])

    entries = ["ema20pb", "stpb", "donch10", "qulla", "macd_turn", "rsi_reclaim"]
    raw = {}
    for e in entries:
        raw[e] = collect(packs, nifty, e, start_ts, end_ts)
        print(f"  {e:12s} {len(raw[e])}", flush=True)

    union_eq = raw["ema20pb"] + raw["qulla"] + raw["stpb"]
    union_break = raw["donch10"] + raw["qulla"] + raw["ema20pb"]
    raw["union_pb"] = union_eq
    raw["union_bo"] = union_break

    jobs = []
    base_filters = [
        ("qulla", F_QULLA),
        ("trend_rs", F_TREND | F_RS63),
    ]
    for entry in ["ema20pb", "qulla", "stpb", "union_pb", "union_bo", "donch10"]:
        for fname, fl in base_filters:
            for pos in (0.5, 0.7, 1.0):
                for mx in (2, 3, 4):
                    for risk in (5.0, 8.0, 10.0):
                        for hold in (12, 15):
                            for rr in (2.0, 2.5):
                                jobs.append({
                                    "entry": entry, "filter": fname, "flags": fl,
                                    "pos": pos, "open": mx, "risk": risk,
                                    "hold": hold, "rr": rr, "new": min(3, mx),
                                    "cd": 5, "exit": "chandelier",
                                })
    # de-dup
    seen, uniq = set(), []
    for j in jobs:
        key = (j["entry"], j["filter"], j["pos"], j["open"], j["risk"], j["hold"], j["rr"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(j)
    jobs = uniq
    print(f"size-grid jobs={len(jobs)}", flush=True)

    ranked = []
    for i, j in enumerate(jobs, 1):
        r = run_one(packs, cal1, raw[j["entry"]], j["flags"], **{k: v for k, v in j.items() if k != "flags"})
        r.update(j)
        ranked.append(r)
        if i % 60 == 0:
            best = max(ranked, key=lambda x: x["ret"])
            print(
                f"  {i}/{len(jobs)} best={best['ret']}% {best['entry']}/{best['filter']} "
                f"pos{best['pos']} open{best['open']} r{best['risk']} h{best['hold']}",
                flush=True,
            )
    ranked.sort(key=lambda x: (-x["ret"], -x["pf"], x["dd"]))
    print("TOP last-1y", flush=True)
    for r in ranked[:15]:
        print(
            f"  {r['ret']:7.1f}% WR{r['wr']:5.1f} PF{r['pf']:4.2f} DD{r['dd']:5.1f} "
            f"n={r['n']:3d} hold={r['avg_hold']:4.1f} par={r['par']}  "
            f"{r['entry']:10s} {r['filter']:10s} pos{r['pos']} open{r['open']} "
            f"r{r['risk']:g} h{r['hold']} {r['rr']}R",
            flush=True,
        )

    cands = [r for r in ranked if r["ret"] >= 90][:8] or ranked[:8]
    print(f"multi-year {len(cands)} packs", flush=True)
    multi = []
    for j in cands:
        years = {}
        for label, (a, b) in WINDOWS.items():
            cal = slice_cal(calendar, a, b)
            r = run_one(packs, cal, raw[j["entry"]], j["flags"], **{k: v for k, v in j.items() if k != "flags"})
            years[label] = {
                "ret": r["ret"], "wr": r["wr"], "pf": r["pf"], "dd": r["dd"],
                "n": r["n"], "avg_hold": r["avg_hold"], "nifty": nifty_ret(nifty, a, b),
            }
        row = dict(j)
        row["windows"] = years
        row["worst"] = min(years[k]["ret"] for k in years)
        multi.append(row)
        print(
            f"  1y={years['1y']['ret']:7.1f} 2y={years['2y']['ret']:7.1f} "
            f"3y={years['3y']['ret']:7.1f} 4y={years['4y']['ret']:7.1f} "
            f"5y={years['5y']['ret']:7.1f}  {j['entry']}/{j['filter']} "
            f"pos{j['pos']} open{j['open']} r{j['risk']} h{j['hold']}",
            flush=True,
        )

    hit = [m for m in multi if m["windows"]["1y"]["ret"] >= 100]
    pool = hit or multi
    pool.sort(key=lambda x: (-x["windows"]["1y"]["ret"], -x.get("worst", 0)))
    winner = pool[0]
    print("\nWINNER", winner["entry"], winner["filter"], winner, flush=True)

    payload = {
        "end": str(END),
        "hit_100": len(hit),
        "phase_top": [{k: r[k] for k in r if k != "flags"} for r in ranked[:20]],
        "multi": [{k: m[k] for k in m if k != "flags"} for m in pool[:10]],
        "winner": {k: winner[k] for k in winner if k != "flags"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {OUT} {payload['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
