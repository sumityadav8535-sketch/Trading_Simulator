"""Hunt RS Pullback filters for higher win rate and bad-market survival.

Baseline is the live 1.5R pack. Tests market regime (Nifty weekly stage),
hold length, tighter leader filters, and a few combinations.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.stage_engine import detect_weekly_stage
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import load_price_dataframe
from trading.services.short_swing import (
    DEFAULT_CAPITAL,
    F_ADX20,
    F_MKT20,
    F_NEAR52,
    F_QULLA,
    F_RS63,
    F_RSI_OK,
    F_ST_BULL,
    F_TREND,
    SwingParams,
    collect_signals,
    preload_packs,
    simulate,
    slice_cal,
)

NEED = F_QULLA | F_RS63
YEARS = [
    ("2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("2025", date(2025, 1, 1), date(2025, 12, 31)),
    ("2026YTD", date(2026, 1, 1), date.today()),
    ("last1y", date.today() - timedelta(days=365), date.today()),
]


def nifty_weekly_stage_series(nifty_pack) -> pd.Series:
    df = pd.DataFrame(
        {"open": nifty_pack.o, "high": nifty_pack.h, "low": nifty_pack.l, "close": nifty_pack.c, "volume": nifty_pack.v},
        index=nifty_pack.index,
    )
    weekly = daily_to_weekly(df)
    stages = []
    for i in range(len(weekly)):
        if i < 38:
            stages.append(None)
            continue
        try:
            st, _, _ = detect_weekly_stage(weekly.iloc[: i + 1])
        except Exception:
            st = None
        stages.append(st)
    weekly = weekly.copy()
    weekly["stage"] = stages
    daily = weekly["stage"].reindex(df.index, method="ffill")
    return daily


def attach_regime(sigs, nifty, stage_s: pd.Series):
    for s in sigs:
        ts = pd.Timestamp(s["sig_ts"])
        st = stage_s.loc[ts] if ts in stage_s.index else None
        if st is None or (isinstance(st, float) and pd.isna(st)):
            idx = stage_s.index[stage_s.index <= ts]
            st = stage_s.loc[idx[-1]] if len(idx) else None
        s["nifty_stage"] = int(st) if st == st and st is not None else 0
        ni = nifty.loc.get(ts) if nifty is not None else None
        s["nifty_gt20"] = bool(ni is not None and nifty.c[ni] > nifty.ema20[ni])
        s["nifty_gt50"] = bool(ni is not None and nifty.c[ni] > nifty.ema50[ni])
        s["nifty_gt200"] = bool(
            ni is not None and not pd.isna(nifty.ema200[ni]) and nifty.c[ni] > nifty.ema200[ni]
        )
        s["nifty_ret10"] = 0.0
        if ni is not None and ni >= 10:
            prev = nifty.c[ni - 10]
            if prev:
                s["nifty_ret10"] = float(nifty.c[ni] / prev - 1.0)


def filtered(sigs, flags=NEED, pred=None):
    out = []
    for s in sigs:
        if (s["flags"] & flags) != flags:
            continue
        if pred is not None and not pred(s):
            continue
        out.append(s)
    return out


def run_one(packs, calendar, sigs, params, a, b):
    cal = slice_cal(calendar, a, b)
    r = simulate(packs, cal, sigs, params, capital=DEFAULT_CAPITAL, flatten=True, keep_trades=False)
    return {
        "ret": r["total_return_pct"],
        "wr": r["win_rate"],
        "pf": r["profit_factor"],
        "n": r["trades"],
        "dd": r["max_drawdown_pct"],
        "hold": r.get("avg_hold"),
    }


def fmt(r):
    return f"{r['ret']:+6.1f}% WR{r['wr']:5.1f} n={r['n']:3d} DD{r['dd']:5.1f}"


def main() -> None:
    print("loading…", flush=True)
    packs, nifty, calendar = preload_packs()
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    stage_s = nifty_weekly_stage_series(nifty)
    print("weekly stages ready", flush=True)

    raw = collect_signals(
        packs, nifty, pd.Timestamp("2022-12-15"), pd.Timestamp(date.today()), entry="ema20pb"
    )
    attach_regime(raw, nifty, stage_s)
    base = filtered(raw)
    print(f"qulla+RS signals={len(base)}", flush=True)

    # How often was Nifty in each stage on signal days, by year
    print("\n=== Nifty weekly stage on baseline signal days ===")
    for name, a, b in YEARS:
        rows = [s for s in base if a <= s["sig_ts"].date() <= b]
        from collections import Counter
        c = Counter(s["nifty_stage"] for s in rows)
        print(f"  {name:8s} n={len(rows):4d}  " + "  ".join(f"S{k}={c[k]}" for k in sorted(c)))

    jobs = []

    def add(label, sigs, p=None):
        jobs.append((label, sigs, p or SwingParams()))

    add("LIVE 1.5R", base)
    add("skip Stage 4", filtered(raw, pred=lambda s: s["nifty_stage"] != 4))
    add("skip Stage 3+4", filtered(raw, pred=lambda s: s["nifty_stage"] in (1, 2)))
    add("only Stage 2", filtered(raw, pred=lambda s: s["nifty_stage"] == 2))
    add("Nifty > EMA20", filtered(raw, flags=NEED | F_MKT20))
    add("Nifty > EMA50", filtered(raw, pred=lambda s: s["nifty_gt50"]))
    add("Nifty > EMA200", filtered(raw, pred=lambda s: s["nifty_gt200"]))
    add("skip Nifty 10d<-4%", filtered(raw, pred=lambda s: s["nifty_ret10"] > -0.04))
    add("skip Nifty 10d<-6%", filtered(raw, pred=lambda s: s["nifty_ret10"] > -0.06))
    add("ST bull", filtered(raw, flags=NEED | F_ST_BULL))
    add("trend 50>200", filtered(raw, flags=NEED | F_TREND))
    add("ADX>=20", filtered(raw, flags=NEED | F_ADX20))
    add("RSI 40-65", filtered(raw, flags=NEED | F_RSI_OK))
    add("near 52w", filtered(raw, flags=NEED | F_NEAR52))
    add("r63>=20%", filtered(raw, pred=lambda s: s["score"] >= 0.20))
    add("r63>=25%", filtered(raw, pred=lambda s: s["score"] >= 0.25))
    add("r63>=30%", filtered(raw, pred=lambda s: s["score"] >= 0.30))
    add("r63>=40%", filtered(raw, pred=lambda s: s["score"] >= 0.40))
    for h in (8, 10, 12, 18, 21):
        add(f"hold {h}d", base, SwingParams(max_hold=h))
    for rr in (1.0, 1.2, 2.0):
        add(f"target {rr:g}R", base, SwingParams(target_rr=rr))
    add("max_new=1 open=2", base, SwingParams(max_new=1, max_open=2))
    add("max_new=2 open=3", base, SwingParams(max_new=2, max_open=3))

    print("\n=== PHASE 1 single changes ===")
    header = f"{'pack':22s} " + "  ".join(f"{n:>22s}" for n, _, _ in YEARS) + "  minWR  red"
    print(header, flush=True)
    phase1 = []
    for label, sigs, p in jobs:
        bits = []
        rec = {"label": label, "years": {}}
        wrs, rets = [], []
        red = 0
        ok_n = True
        for name, a, b in YEARS:
            r = run_one(packs, calendar, sigs, p, a, b)
            rec["years"][name] = r
            bits.append(fmt(r))
            wrs.append(r["wr"])
            rets.append(r["ret"])
            if r["ret"] < 0:
                red += 1
            if r["n"] < 12:
                ok_n = False
        rec["min_wr"] = min(wrs) if wrs else 0
        rec["avg_wr"] = sum(wrs) / len(wrs) if wrs else 0
        rec["min_ret"] = min(rets) if rets else 0
        rec["red"] = red
        rec["ok_n"] = ok_n
        phase1.append(rec)
        print(f"{label:22s} " + "  ".join(bits) + f"  {rec['min_wr']:5.1f}  {red}", flush=True)

    print("\n=== PHASE 1 ranked by min WR (n>=12 each year, then fewer red years) ===")
    ranked = sorted(phase1, key=lambda x: (-int(x["ok_n"]), x["red"], -x["min_wr"], -x["min_ret"]))
    for rec in ranked[:12]:
        print(
            f"  {rec['label']:22s} minWR={rec['min_wr']:.1f} avgWR={rec['avg_wr']:.1f} "
            f"minRet={rec['min_ret']:+.1f} red={rec['red']}"
        )

    # Combinations aimed at WR + skip bear tape
    print("\n=== PHASE 2 combinations ===")
    combos = [
        ("s4 + ST", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_stage"] != 4), SwingParams()),
        ("s4 + trend", filtered(raw, flags=NEED | F_TREND, pred=lambda s: s["nifty_stage"] != 4), SwingParams()),
        ("s4 + ADX", filtered(raw, flags=NEED | F_ADX20, pred=lambda s: s["nifty_stage"] != 4), SwingParams()),
        ("s4 + r63>=25", filtered(raw, pred=lambda s: s["nifty_stage"] != 4 and s["score"] >= 0.25), SwingParams()),
        ("s4 + hold10", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(max_hold=10)),
        ("s4 + hold12", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(max_hold=12)),
        ("s4 + 1.2R", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(target_rr=1.2)),
        ("s4 + 1.0R", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(target_rr=1.0)),
        ("s4 + new1", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(max_new=1, max_open=2)),
        ("s34 + 1.2R", filtered(raw, pred=lambda s: s["nifty_stage"] in (1, 2)), SwingParams(target_rr=1.2)),
        ("s34 + ST", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_stage"] in (1, 2)), SwingParams()),
        ("ST + 1.2R", filtered(raw, flags=NEED | F_ST_BULL), SwingParams(target_rr=1.2)),
        ("ST + hold10", filtered(raw, flags=NEED | F_ST_BULL), SwingParams(max_hold=10)),
        ("ADX + 1.2R", filtered(raw, flags=NEED | F_ADX20), SwingParams(target_rr=1.2)),
        ("r63>=25 + 1.2R", filtered(raw, pred=lambda s: s["score"] >= 0.25), SwingParams(target_rr=1.2)),
        ("r63>=25 + hold10", filtered(raw, pred=lambda s: s["score"] >= 0.25), SwingParams(max_hold=10)),
        ("new1 + 1.2R", base, SwingParams(max_new=1, max_open=2, target_rr=1.2)),
        ("s4 + ST + 1.2R", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_stage"] != 4), SwingParams(target_rr=1.2)),
        ("s4 + ST + hold10", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_stage"] != 4), SwingParams(max_hold=10)),
        ("s4 + r25 + 1.2R", filtered(raw, pred=lambda s: s["nifty_stage"] != 4 and s["score"] >= 0.25), SwingParams(target_rr=1.2)),
        ("s4 + new1 + 1.2R", filtered(raw, pred=lambda s: s["nifty_stage"] != 4), SwingParams(max_new=1, max_open=2, target_rr=1.2)),
        ("gt50 + ST", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_gt50"]), SwingParams()),
        ("gt50 + 1.2R", filtered(raw, pred=lambda s: s["nifty_gt50"]), SwingParams(target_rr=1.2)),
        ("no crash + ST", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_ret10"] > -0.04), SwingParams()),
        ("s4 + no crash", filtered(raw, pred=lambda s: s["nifty_stage"] != 4 and s["nifty_ret10"] > -0.04), SwingParams()),
        ("s2 + ST + 1.2R", filtered(raw, flags=NEED | F_ST_BULL, pred=lambda s: s["nifty_stage"] == 2), SwingParams(target_rr=1.2)),
    ]
    print(header, flush=True)
    phase2 = []
    for label, sigs, p in combos:
        bits = []
        rec = {"label": label, "years": {}}
        wrs, rets = [], []
        red = 0
        ok_n = True
        for name, a, b in YEARS:
            r = run_one(packs, calendar, sigs, p, a, b)
            rec["years"][name] = r
            bits.append(fmt(r))
            wrs.append(r["wr"])
            rets.append(r["ret"])
            if r["ret"] < 0:
                red += 1
            if r["n"] < 12:
                ok_n = False
        rec["min_wr"] = min(wrs) if wrs else 0
        rec["avg_wr"] = sum(wrs) / len(wrs) if wrs else 0
        rec["min_ret"] = min(rets) if rets else 0
        rec["red"] = red
        rec["ok_n"] = ok_n
        phase2.append(rec)
        print(f"{label:22s} " + "  ".join(bits) + f"  {rec['min_wr']:5.1f}  {red}", flush=True)

    print("\n=== BEST (min WR, no/few red years, enough trades) ===")
    allr = phase1 + phase2
    best = sorted(allr, key=lambda x: (x["red"], -int(x["ok_n"]), -x["min_wr"], -x["avg_wr"], -x["min_ret"]))
    for rec in best[:15]:
        print(
            f"  {rec['label']:22s} red={rec['red']} minWR={rec['min_wr']:.1f} "
            f"avgWR={rec['avg_wr']:.1f} minRet={rec['min_ret']:+.1f} ok_n={rec['ok_n']}"
        )
        for name, _, _ in YEARS:
            r = rec["years"][name]
            print(f"      {name:8s} {fmt(r)}")


if __name__ == "__main__":
    main()
