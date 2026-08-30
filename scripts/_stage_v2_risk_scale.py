"""Quick baseline + risk scale + aggressive combos for 300% hunt."""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from trading.services.market_data import get_universe_symbols

START, END = date(2023, 1, 1), date(2026, 2, 28)
SYMS = get_universe_symbols(nifty200_only=True)


def go(label: str, **kw):
    risk = float(kw.pop("risk_pct", 2.0))
    t0 = time.time()
    r = run_stage_v2_backtest(
        symbols=SYMS,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
        config=SimpleNamespace(risk_pct=risk),
        **kw,
    )
    secs = time.time() - t0
    flag = " ***300***" if r.total_return_pct >= 300 else ""
    print(
        f"{label:40s} ret={r.total_return_pct:+7.1f}% WR={r.win_rate:5.1f}% "
        f"PF={r.profit_factor:5.2f} DD={r.max_drawdown_pct:5.1f}% "
        f"n={r.total_trades:3d} skip={r.signals_skipped_cash:3d} peak={r.peak_parallel:2d} "
        f"risk={risk:g} ({secs:.0f}s){flag}",
        flush=True,
    )
    return r


def main():
    print(f"Universe {len(SYMS)} | {START} -> {END}", flush=True)

    print("\n=== BASELINE ===", flush=True)
    go("default daily_mtf", risk_pct=2.0, tech_filter="daily_mtf",
       exit_mode="stage_4_only", target_rr=2.5, max_hold_days=65)

    print("\n=== RISK SCALE daily_mtf ===", flush=True)
    for risk in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
        go(f"risk {risk}", risk_pct=risk, tech_filter="daily_mtf",
           exit_mode="stage_4_only", target_rr=2.5, max_hold_days=65)

    print("\n=== RISK SCALE tech=none ===", flush=True)
    for risk in [2.0, 4.0, 6.0, 8.0]:
        go(f"none risk {risk}", risk_pct=risk, tech_filter="none",
           exit_mode="stage_4_only", target_rr=2.5, max_hold_days=65)

    print("\n=== EXIT / RR / HOLD @ risk 4 & 6 ===", flush=True)
    combos = []
    for risk in [4.0, 6.0, 8.0]:
        for exit_m in ["stage_4_only", "trail_ma_s4", "no_stage"]:
            for rr in [2.0, 2.5, 3.0, 4.0, 5.0]:
                for hold in [65, 90, 130, 200]:
                    combos.append((risk, exit_m, rr, hold))
    # prioritize trail / no_stage / high RR / long hold
    combos.sort(key=lambda x: (
        -x[0],
        0 if x[1] in ("trail_ma_s4", "no_stage") else 1,
        -x[2],
        -x[3],
    ))
    best = None
    for risk, exit_m, rr, hold in combos[:60]:
        r = go(
            f"r{risk:g} {exit_m} {rr}R h{hold}",
            risk_pct=risk,
            tech_filter="daily_mtf",
            exit_mode=exit_m,
            target_rr=rr,
            max_hold_days=hold,
        )
        if best is None or r.total_return_pct > best.total_return_pct:
            best = r
            print(f"  >> new best {best.total_return_pct}%", flush=True)
        if r.total_return_pct >= 300:
            print("FOUND 300+", flush=True)

    print("\n=== ENTRY VARIANTS @ best-ish risk ===", flush=True)
    for risk in [4.0, 6.0, 8.0]:
        for estage, eon in [(2, "transition"), (2, "in_stage"), (1, "transition")]:
            for tech in ["daily_mtf", "none", "not_extended", "daily_mtf_not_ext"]:
                for exit_m in ["trail_ma_s4", "stage_4_only", "no_stage"]:
                    go(
                        f"r{risk:g} s{estage}/{eon} {tech} {exit_m}",
                        risk_pct=risk,
                        entry_stage=estage,
                        entry_on=eon,
                        tech_filter=tech,
                        exit_mode=exit_m,
                        target_rr=3.0,
                        max_hold_days=130,
                    )


if __name__ == "__main__":
    main()
