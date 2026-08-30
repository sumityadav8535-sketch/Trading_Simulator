"""Phase 2: score year windows, push risk, find any 1y ≥100% pack."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from _cup_hunt_100 import (  # type: ignore
    WINDOWS,
    collect_signals,
    filter_packs,
    simulate,
)
from stage_analysis_v2.services.backtester import _preload_frames
from stage_analysis_v2.services.cup_breakout import (
    CUP_EXIT_EMA20,
    CUP_EXIT_MEASURED,
    CUP_EXIT_TARGET_R,
    CupParams,
    _CupPack,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def grid():
    exits = [
        (CUP_EXIT_MEASURED, 0.0),
        (CUP_EXIT_EMA20, 0.0),
        (CUP_EXIT_TARGET_R, 2.0),
        (CUP_EXIT_TARGET_R, 3.0),
        (CUP_EXIT_TARGET_R, 1.5),
    ]
    for risk in (5.0, 8.0, 10.0, 12.0, 15.0):
        for hold in (40, 60, 90, 120):
            for cooldown in (0, 3):
                for pos in (100.0,):
                    for exit_mode, rr in exits:
                        for max_new in (10, 20):
                            yield {
                                "risk_pct": risk,
                                "max_hold_days": hold,
                                "cooldown_days": cooldown,
                                "max_pos_pct": pos,
                                "cup_exit_mode": exit_mode,
                                "target_rr": rr,
                                "max_new_per_day": max_new,
                            }


def main():
    t0 = time.time()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    packs = {sym: _CupPack(sym, df) for sym, df in frames.items()}
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret60 = nifty_df["close"].pct_change(60) if not nifty_df.empty else None
    nifty_sma200 = nifty_df["close"].rolling(200, min_periods=200).mean() if not nifty_df.empty else None
    nifty_close = nifty_df["close"] if not nifty_df.empty else None
    full_cal = sorted({ts for df in frames.values() for ts in df.index.tolist()})
    window_cals = {
        name: [ts for ts in full_cal if pd.Timestamp(w0) <= ts <= pd.Timestamp(w1)]
        for name, w0, w1 in WINDOWS
    }

    want = {"wide_cup", "no_stack", "loose_confirm"}
    hits = []
    best_by_window: dict[str, dict] = {}

    for pack_name, p0 in filter_packs():
        if pack_name not in want:
            continue
        by_day, n_sig = collect_signals(packs, frames, p0, nifty_ret60, nifty_sma200, nifty_close)
        print(f"\n=== {pack_name} signals={n_sig} ===", flush=True)
        for wname, w0, w1 in WINDOWS:
            n_w = sum(1 for ts, sigs in by_day.items() if pd.Timestamp(w0) <= ts <= pd.Timestamp(w1) for _ in sigs)
            print(f"  {wname} entry-signals={n_w}", flush=True)

        for g in grid():
            p = replace(
                p0,
                cup_exit_mode=g["cup_exit_mode"],
                target_rr=max(0.5, float(g["target_rr"] or 2.0)),
                max_new_per_day=g["max_new_per_day"],
            )
            year_stats = {}
            for wname, w0, w1 in WINDOWS:
                year_stats[wname] = simulate(
                    packs, window_cals[wname], by_day,
                    risk_pct=g["risk_pct"],
                    max_hold_days=g["max_hold_days"],
                    cooldown_days=g["cooldown_days"],
                    max_pos_pct=g["max_pos_pct"],
                    params=p,
                    start_ts=pd.Timestamp(w0),
                    end_ts=pd.Timestamp(w1),
                )
            for wname, st in year_stats.items():
                cur = best_by_window.get(wname)
                if cur is None or st["ret"] > cur["ret"]:
                    best_by_window[wname] = {"pack": pack_name, **g, **st}
                if st["ret"] >= 100.0 and st["n"] >= 12:
                    hits.append({
                        "window": wname,
                        "pack": pack_name,
                        **g,
                        "stat": st,
                        "years": year_stats,
                    })

    print("\n===== BEST PER WINDOW =====", flush=True)
    for wname, _, _ in WINDOWS:
        b = best_by_window.get(wname)
        if not b:
            continue
        print(
            f"{wname:10s} {b['ret']:+7.1f}% n={b['n']:3d} WR={b['wr']:5.1f}% DD={b['dd']:5.1f}% "
            f"PF={b['pf']:.2f}  {b['pack']} risk={b['risk_pct']:g} hold={b['max_hold_days']} "
            f"cd={b['cooldown_days']} {b['cup_exit_mode']}:{b['target_rr']} new={b['max_new_per_day']}",
            flush=True,
        )

    print(f"\n===== HITS ≥100% ({len(hits)}) =====", flush=True)
    hits.sort(key=lambda r: (-r["stat"]["ret"], r["stat"]["dd"]))
    shown = 0
    for row in hits:
        if shown >= 25:
            break
        st = row["stat"]
        y = row["years"]
        print(
            f"{row['window']:10s} {st['ret']:+7.1f}% n={st['n']:3d} WR={st['wr']:5.1f}% "
            f"DD={st['dd']:5.1f}%  {row['pack']} risk={row['risk_pct']:g} hold={row['max_hold_days']} "
            f"cd={row['cooldown_days']} {row['cup_exit_mode']}:{row['target_rr']} "
            f"| last={y['last_1y']['ret']:+.0f} 23={y['y2023']['ret']:+.0f} "
            f"24={y['y2024']['ret']:+.0f} 25={y['y2025']['ret']:+.0f}",
            flush=True,
        )
        shown += 1
    print(f"Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
