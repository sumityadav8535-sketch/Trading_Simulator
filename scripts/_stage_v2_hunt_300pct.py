"""
Hunt for Stage Analysis 2.0 configs that deliver >= 300% over ~3 years.

Uses real shared-capital model (no unlimited leverage fantasy).
Preloads price frames once; reuses signal sets where possible.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.backtester import (  # noqa: E402
    DEFAULT_EXIT_MODE,
    DEFAULT_TECH_FILTER,
    run_stage_v2_backtest,
)
from trading.services.market_data import get_universe_symbols  # noqa: E402

START = date(2023, 1, 1)
END = date(2026, 2, 28)
CAPITAL = 1_000_000.0
TARGET_RET = 300.0


@dataclass
class Hit:
    name: str
    ret: float
    wr: float
    pf: float
    dd: float
    trades: int
    signals: int
    peak_par: int
    params: dict[str, Any]


def run(params: dict[str, Any], symbols: list[str]) -> Hit:
    risk = float(params.get("risk_pct", 2.0))
    r = run_stage_v2_backtest(
        symbols=symbols,
        start_date=START,
        end_date=END,
        capital=CAPITAL,
        min_quality_score=int(params.get("min_quality_score", 0)),
        market_filter=bool(params.get("market_filter", False)),
        exit_mode=params.get("exit_mode", DEFAULT_EXIT_MODE),
        tech_filter=params.get("tech_filter", DEFAULT_TECH_FILTER),
        entry_stage=int(params.get("entry_stage", 2)),
        entry_on=params.get("entry_on", "transition"),
        target_rr=float(params.get("target_rr", 2.5)),
        max_hold_days=int(params.get("max_hold_days", 65)),
        stop_ma_mult=float(params.get("stop_ma_mult", 0.95)),
        trail_ma_mult=float(params.get("trail_ma_mult", 0.98)),
        config=SimpleNamespace(risk_pct=risk),
    )
    name = (
        f"risk={risk:g}% stage={params.get('entry_stage', 2)}/{params.get('entry_on', 'transition')} "
        f"tech={params.get('tech_filter', DEFAULT_TECH_FILTER)} exit={params.get('exit_mode', DEFAULT_EXIT_MODE)} "
        f"RR={params.get('target_rr', 2.5)} hold={params.get('max_hold_days', 65)} "
        f"stop={params.get('stop_ma_mult', 0.95)} Q>={params.get('min_quality_score', 0)} "
        f"mkt={'Y' if params.get('market_filter') else 'N'}"
    )
    return Hit(
        name=name,
        ret=float(r.total_return_pct),
        wr=float(r.win_rate),
        pf=float(r.profit_factor),
        dd=float(r.max_drawdown_pct),
        trades=int(r.total_trades),
        signals=int(r.stage2_entries),
        peak_par=int(r.peak_parallel),
        params=dict(params),
    )


def print_hit(h: Hit, tag: str = "") -> None:
    flag = " *** HIT 300% ***" if h.ret >= TARGET_RET else ""
    print(
        f"{tag}ret={h.ret:+7.1f}% WR={h.wr:5.1f}% PF={h.pf:5.2f} DD={h.dd:5.1f}% "
        f"n={h.trades:3d} sig={h.signals:3d} par={h.peak_par:2d} | {h.name}{flag}",
        flush=True,
    )


def main() -> None:
    symbols = get_universe_symbols(nifty200_only=True)
    print(f"Hunting ≥{TARGET_RET:g}% | {START} → {END} | {len(symbols)} symbols | capital ₹{CAPITAL:,.0f}")
    print("=" * 100, flush=True)

    all_hits: list[Hit] = []
    winners: list[Hit] = []
    t0 = time.time()

    def try_params(params: dict[str, Any], tag: str = "") -> Hit:
        h = run(params, symbols)
        all_hits.append(h)
        print_hit(h, tag=tag)
        if h.ret >= TARGET_RET:
            winners.append(h)
        return h

    # ── Phase 1: risk scaling on defaults ──────────────────────────────────
    print("\n### PHASE 1 — risk_pct scaling (default entry/exit/tech) ###", flush=True)
    base = {
        "entry_stage": 2,
        "entry_on": "transition",
        "tech_filter": "daily_mtf",
        "exit_mode": "stage_4_only",
        "target_rr": 2.5,
        "max_hold_days": 65,
        "stop_ma_mult": 0.95,
        "trail_ma_mult": 0.98,
        "min_quality_score": 0,
        "market_filter": False,
    }
    for risk in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
        try_params({**base, "risk_pct": risk}, tag="[P1] ")

    # ── Phase 2: exit / RR / hold around best risk so far ─────────────────
    print("\n### PHASE 2 — exit mode × RR × hold (top risk candidates) ###", flush=True)
    # pick risks that were strongest so far
    top_risks = sorted(
        {h.params.get("risk_pct", 2.0) for h in all_hits},
        key=lambda r: max((h.ret for h in all_hits if h.params.get("risk_pct") == r), default=-999),
        reverse=True,
    )[:4]
    if not top_risks:
        top_risks = [4.0, 5.0, 6.0]

    exits = ["stage_4_only", "trail_ma_s4", "no_stage", "stage_3_4"]
    rrs = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    holds = [40, 65, 90, 130, 200]
    stops = [0.93, 0.95, 0.97]

    phase2_combos = []
    for risk, exit_m, rr, hold, stop in itertools.product(top_risks[:3], exits, rrs, holds, stops):
        # prune absurd: very tight stop + short hold already covered
        phase2_combos.append({
            **base,
            "risk_pct": risk,
            "exit_mode": exit_m,
            "target_rr": rr,
            "max_hold_days": hold,
            "stop_ma_mult": stop,
            "trail_ma_mult": 0.98 if exit_m == "trail_ma_s4" else 0.98,
        })

    # Too many — sample smartly: prioritize high risk + trail/no_stage + long hold + high RR
    def score_combo(p: dict) -> float:
        s = float(p["risk_pct"]) * 10
        if p["exit_mode"] in ("trail_ma_s4", "no_stage"):
            s += 20
        if p["max_hold_days"] >= 90:
            s += 10
        if p["target_rr"] >= 3:
            s += 5
        return s

    phase2_combos.sort(key=score_combo, reverse=True)
    # run top 60 first, then expand if no winner
    seen = set()
    phase2_run = []
    for p in phase2_combos:
        key = (
            p["risk_pct"], p["exit_mode"], p["target_rr"],
            p["max_hold_days"], p["stop_ma_mult"],
        )
        if key in seen:
            continue
        seen.add(key)
        phase2_run.append(p)
        if len(phase2_run) >= 80:
            break

    for p in phase2_run:
        try_params(p, tag="[P2] ")
        if winners:
            print(f"\n>>> Found {len(winners)} winner(s) in phase 2 — continuing for better DD/WR", flush=True)

    # ── Phase 3: tech filters + entry variants on best skeletons ──────────
    print("\n### PHASE 3 — tech / entry / quality on best skeletons ###", flush=True)
    skeletons = sorted(all_hits, key=lambda h: h.ret, reverse=True)[:8]
    tech_opts = ["none", "daily_mtf", "not_extended", "ema_stack", "bb_mid", "confluence"]
    entry_opts = [
        (2, "transition"),
        (2, "in_stage"),
        (1, "transition"),
        (1, "in_stage"),
    ]
    q_opts = [0, 50, 75]
    trail_opts = [0.96, 0.98, 0.99]

    phase3_count = 0
    for sk in skeletons:
        for tech, (estage, eon), q in itertools.product(tech_opts, entry_opts, q_opts):
            p = {**sk.params, "tech_filter": tech, "entry_stage": estage, "entry_on": eon, "min_quality_score": q}
            # skip exact duplicate of skeleton
            key = tuple(sorted(p.items()))
            if any(tuple(sorted(h.params.items())) == key for h in all_hits):
                continue
            if p.get("exit_mode") == "trail_ma_s4":
                for tr in trail_opts:
                    p2 = {**p, "trail_ma_mult": tr}
                    try_params(p2, tag="[P3] ")
                    phase3_count += 1
            else:
                try_params(p, tag="[P3] ")
                phase3_count += 1
            if phase3_count >= 120 and winners:
                break
            if phase3_count >= 200:
                break
        if phase3_count >= 200:
            break

    # ── Phase 4: aggressive push if still short ───────────────────────────
    if not winners:
        print("\n### PHASE 4 — aggressive push (higher risk + ride winners) ###", flush=True)
        aggressive = []
        for risk in [6.0, 8.0, 10.0, 12.0, 15.0]:
            for exit_m in ["trail_ma_s4", "no_stage", "stage_4_only"]:
                for rr in [2.5, 3.0, 4.0, 5.0, 6.0]:
                    for hold in [90, 130, 200, 300]:
                        for tech in ["none", "daily_mtf", "not_extended"]:
                            for estage, eon in [(2, "transition"), (1, "transition"), (2, "in_stage")]:
                                aggressive.append({
                                    "risk_pct": risk,
                                    "exit_mode": exit_m,
                                    "target_rr": rr,
                                    "max_hold_days": hold,
                                    "stop_ma_mult": 0.95,
                                    "trail_ma_mult": 0.98,
                                    "tech_filter": tech,
                                    "entry_stage": estage,
                                    "entry_on": eon,
                                    "min_quality_score": 0,
                                    "market_filter": False,
                                })
        # Prefer high-risk trail / no_stage
        aggressive.sort(
            key=lambda p: (
                p["risk_pct"],
                1 if p["exit_mode"] in ("trail_ma_s4", "no_stage") else 0,
                p["max_hold_days"],
                p["target_rr"],
            ),
            reverse=True,
        )
        seen4 = set()
        for p in aggressive:
            key = (
                p["risk_pct"], p["exit_mode"], p["target_rr"], p["max_hold_days"],
                p["tech_filter"], p["entry_stage"], p["entry_on"],
            )
            if key in seen4:
                continue
            seen4.add(key)
            try_params(p, tag="[P4] ")
            if winners and len(winners) >= 5:
                break
            if len(all_hits) > 350:
                break

    # ── Phase 5: refine winners / near-winners for max return ─────────────
    print("\n### PHASE 5 — refine near-misses / winners ###", flush=True)
    near = sorted(all_hits, key=lambda h: h.ret, reverse=True)[:12]
    for h in near:
        p0 = h.params
        # micro-grid around params
        risks = sorted(set([
            max(1.0, float(p0.get("risk_pct", 4)) - 1),
            float(p0.get("risk_pct", 4)),
            float(p0.get("risk_pct", 4)) + 1,
            float(p0.get("risk_pct", 4)) + 2,
            float(p0.get("risk_pct", 4)) + 3,
        ]))
        holds = sorted(set([
            max(30, int(p0.get("max_hold_days", 65)) - 30),
            int(p0.get("max_hold_days", 65)),
            int(p0.get("max_hold_days", 65)) + 40,
            int(p0.get("max_hold_days", 65)) + 80,
        ]))
        rrs = sorted(set([
            max(1.0, float(p0.get("target_rr", 2.5)) - 0.5),
            float(p0.get("target_rr", 2.5)),
            float(p0.get("target_rr", 2.5)) + 0.5,
            float(p0.get("target_rr", 2.5)) + 1.0,
        ]))
        for risk, hold, rr in itertools.product(risks, holds, rrs):
            p = {**p0, "risk_pct": risk, "max_hold_days": hold, "target_rr": rr}
            key = tuple(sorted(p.items()))
            if any(tuple(sorted(x.params.items())) == key for x in all_hits):
                continue
            try_params(p, tag="[P5] ")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print(f"DONE in {elapsed/60:.1f} min | tested {len(all_hits)} configs | winners ≥{TARGET_RET:g}%: {len(winners)}")
    print("=" * 100)

    ranked = sorted(all_hits, key=lambda h: h.ret, reverse=True)
    print("\n### TOP 25 BY RETURN ###")
    for i, h in enumerate(ranked[:25], 1):
        print_hit(h, tag=f"#{i:02d} ")

    if winners:
        best = sorted(winners, key=lambda h: (h.ret, -h.dd), reverse=True)
        print("\n### WINNERS (≥300%) — sorted by return then lower DD ###")
        for i, h in enumerate(best, 1):
            print_hit(h, tag=f"W{i} ")
            print(f"     params: {h.params}")
        champ = best[0]
        print("\n### CHAMPION STRATEGY ###")
        print(champ.params)
        print(
            f"Return {champ.ret}% | WR {champ.wr}% | PF {champ.pf} | "
            f"MaxDD {champ.dd}% | Trades {champ.trades}"
        )
    else:
        print("\n### NO 300% YET — best so far ###")
        for h in ranked[:5]:
            print_hit(h)
            print(f"     params: {h.params}")
        print("Need another pass with more aggressive / multi-entry variants.")

    # write results
    out = os.path.join(ROOT, "data", "_stage_v2_hunt_300_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"tested={len(all_hits)} winners={len(winners)} target={TARGET_RET}\n")
        for h in ranked:
            f.write(
                f"{h.ret:+.2f}% WR={h.wr:.1f} PF={h.pf:.2f} DD={h.dd:.1f} "
                f"n={h.trades} | {h.params}\n"
            )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
