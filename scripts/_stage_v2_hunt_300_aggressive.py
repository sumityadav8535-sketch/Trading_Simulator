"""
Aggressive targeted hunt for ≥300% Stage Analysis 2.0 (2023→early 2026).

Faster: fewer packs, smart risk ladder, dual-stage merge, cooldown/maxpos knobs.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from collections import defaultdict
from datetime import date
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import (
    COOLDOWN_DAYS,
    EXIT_NO_STAGE,
    EXIT_STAGE_3_4,
    EXIT_STAGE_4_ONLY,
    EXIT_TRAIL_MA_S4,
    StageV2Trade,
    _OpenPos,
    _close_trade,
    _collect_stage2_signals,
    _equity,
    _invested_total,
    _preload_frames,
    _weekly_stage_at,
    normalize_exit_mode,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.tech_filters import enrich_daily_tech, pack_needs_tech_snapshot
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

START = date(2023, 1, 1)
END = date(2026, 2, 28)
CAPITAL = 1_000_000.0
TARGET = 300.0


def simulate(
    frames, weekly_by_sym, signals_by_day, calendar, *,
    capital, risk_pct, exit_mode, target_rr, max_hold_days,
    trail_ma_mult=0.98, cooldown_days=COOLDOWN_DAYS, max_positions=999,
    # optional: risk boost when quality high
    quality_risk_boost=False,
) -> dict[str, Any]:
    exit_mode = normalize_exit_mode(exit_mode)
    trail_ma = exit_mode == EXIT_TRAIL_MA_S4
    target_label = f"target_{target_rr:g}r"
    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve = [capital]
    peak_parallel = 0

    for ts in calendar:
        closed_today = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close, low, high = float(row["close"]), float(row["low"]), float(row["high"])
            pos.hold_days += 1
            exit_price = exit_reason = None
            stage_now = ma_now = None

            if exit_mode != EXIT_NO_STAGE or trail_ma:
                weekly = weekly_by_sym.get(sym)
                if weekly is not None and not weekly.empty:
                    week_mask = weekly.index[
                        (weekly.index > (pos.last_week_check or pos.entry_date)) & (weekly.index <= ts)
                    ]
                    if len(week_mask):
                        pos.last_week_check = week_mask[-1]
                        w_idx = weekly.index.get_loc(pos.last_week_check)
                        if isinstance(w_idx, slice):
                            w_idx = w_idx.stop - 1
                        stage_now, metrics = _weekly_stage_at(weekly, int(w_idx))
                        if metrics and metrics.get("ma"):
                            ma_now = float(metrics["ma"])
                        if trail_ma and ma_now and ma_now > 0:
                            new_stop = round(ma_now * trail_ma_mult, 2)
                            if new_stop > pos.stop and new_stop < close:
                                pos.stop = new_stop

            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, "stop_loss"
            elif high >= pos.target:
                exit_price, exit_reason = pos.target, target_label
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif stage_now is not None and exit_mode != EXIT_NO_STAGE:
                if exit_mode == EXIT_STAGE_3_4 and stage_now in (3, 4):
                    exit_price, exit_reason = close, "stage_exit"
                elif exit_mode in (EXIT_STAGE_4_ONLY, EXIT_TRAIL_MA_S4) and stage_now == 4:
                    exit_price, exit_reason = close, "stage_exit"

            if exit_price is None:
                continue
            trade = _close_trade(pos, exit_price=exit_price, exit_ts=ts,
                                 exit_reason=exit_reason, days_held=pos.hold_days)
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)
        for sym in closed_today:
            opens.pop(sym, None)

        day_sigs = signals_by_day.get(ts, [])
        if day_sigs:
            day_sigs = sorted(
                day_sigs,
                key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
                reverse=True,
            )
            for sig in day_sigs:
                if len(opens) >= max_positions:
                    break
                sym = sig["symbol"]
                if sym in opens or sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if cooldown_days > 0 and prev_x is not None and (ts - prev_x).days < cooldown_days:
                    continue
                row = frames[sym].loc[ts]
                entry_price = float(row["open"])
                high, low = float(row["high"]), float(row["low"])
                stop = float(sig["stop"])
                risk_amt = entry_price - stop
                if risk_amt <= 0:
                    continue
                target = round(entry_price + risk_amt * target_rr, 2)
                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    continue
                use_risk = risk_pct
                if quality_risk_boost:
                    q = int(sig.get("quality_score") or 0)
                    if q >= 85:
                        use_risk = risk_pct * 1.5
                    elif q >= 75:
                        use_risk = risk_pct * 1.25
                    elif q < 50:
                        use_risk = risk_pct * 0.75
                pos_size = calculate_position_size(equity_now, use_risk, entry_price, stop)
                qty = min(int(pos_size.quantity), int(cash // entry_price) if entry_price > 0 else 0)
                if qty <= 0:
                    continue
                notional = qty * entry_price
                if notional > cash + 1e-6:
                    continue
                cash -= notional
                invested_after = _invested_total(opens) + notional
                parallel = len(opens) + 1
                peak_parallel = max(peak_parallel, parallel)
                pos = _OpenPos(
                    symbol=sym, entry_date=ts, signal_date=sig["signal_date"],
                    entry_price=entry_price, stop=stop, target=target, qty=qty,
                    notional=notional, quality_score=int(sig.get("quality_score") or 0),
                    rs_rating=float(sig.get("rs_rating") or 0),
                    weekly_stage=int(sig.get("weekly_stage") or 2),
                    capital_invested=notional, cash_available=cash,
                    total_invested=invested_after, parallel_open=parallel,
                    equity_at_entry=cash + invested_after,
                )
                opens[sym] = pos
                if low <= stop:
                    trade = _close_trade(pos, exit_price=stop, exit_ts=ts,
                                         exit_reason="stop_loss", days_held=0)
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif high >= target:
                    trade = _close_trade(pos, exit_price=target, exit_ts=ts,
                                         exit_reason=target_label, days_held=0)
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)

        equity_curve.append(_equity(cash, opens))

    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None:
                continue
            hist = df.loc[df.index <= last_ts]
            if hist.empty:
                continue
            trade = _close_trade(
                pos, exit_price=float(hist.iloc[-1]["close"]), exit_ts=hist.index[-1],
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)

    final_eq = cash
    ret = (final_eq - capital) / capital * 100
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    wr = 100 * len(wins) / len(trades) if trades else 0.0
    pf = gp / gl if trades else 0.0
    peak, max_dd = capital, 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)
    return {
        "ret": round(ret, 2), "wr": round(wr, 2), "pf": round(pf, 2),
        "dd": round(max_dd, 2), "n": len(trades), "final": round(final_eq, 2),
        "peak_par": peak_parallel,
    }


def collect(frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et,
            entry_stage, entry_on, tech, q, stop_m):
    need = pack_needs_tech_snapshot(tech)
    sbd = defaultdict(list)
    n = 0
    for sym, weekly in weekly_by_sym.items():
        sigs = _collect_stage2_signals(
            sym, weekly, frames[sym],
            nifty_weekly if not nifty_weekly.empty else weekly,
            nifty_weekly, st, et,
            min_quality_score=q, market_filter=False, tech_filter=tech,
            daily_tech=tech_by_sym.get(sym) if need else None,
            entry_stage=entry_stage, entry_on=entry_on,
            target_rr=2.5, stop_ma_mult=stop_m,
        )
        for s in sigs:
            sbd[s["entry_day"]].append(s)
            n += 1
    return sbd, n


def merge_signals(*packs):
    """Merge multiple signal dicts; de-dupe by (symbol, entry_day) keep highest Q."""
    best = {}
    for sbd, _n in packs:
        for day, sigs in sbd.items():
            for s in sigs:
                key = (s["symbol"], day)
                prev = best.get(key)
                if prev is None or int(s.get("quality_score") or 0) > int(prev.get("quality_score") or 0):
                    best[key] = s
    out = defaultdict(list)
    for (_sym, day), s in best.items():
        out[day].append(s)
    return out, len(best)


def run_grid(name, sbd, nsig, frames, weekly_by_sym, calendar, results, winners):
    print(f"\n### {name} | signals={nsig} ###", flush=True)
    if nsig < 5:
        return

    # Compact high-value grid
    configs = []
    for risk in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
        for exit_m in ["stage_4_only", "trail_ma_s4", "no_stage"]:
            for rr in [2.0, 2.5, 3.0, 4.0]:
                for hold in [65, 90, 130, 200]:
                    configs.append((risk, exit_m, rr, hold, COOLDOWN_DAYS, 999, False, 0.98))

    # Cooldown / concentration / quality-boost variants around mid-high risk
    for risk in [4, 5, 6, 8, 10, 12, 15]:
        for exit_m in ["trail_ma_s4", "stage_4_only", "no_stage"]:
            for rr in [2.5, 3.0, 4.0]:
                for hold in [90, 130]:
                    for cd in [0, 5, 20]:
                        configs.append((risk, exit_m, rr, hold, cd, 999, False, 0.98))
                    configs.append((risk, exit_m, rr, hold, 0, 8, True, 0.98))
                    configs.append((risk, exit_m, rr, hold, 5, 5, True, 0.97))

    # Dedup
    seen = set()
    uniq = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    print(f"  testing {len(uniq)} configs…", flush=True)
    local_best = -999.0
    for i, (risk, exit_m, rr, hold, cd, mp, qboost, trail) in enumerate(uniq, 1):
        r = simulate(
            frames, weekly_by_sym, sbd, calendar,
            capital=CAPITAL, risk_pct=float(risk), exit_mode=exit_m,
            target_rr=float(rr), max_hold_days=int(hold),
            trail_ma_mult=trail, cooldown_days=int(cd), max_positions=int(mp),
            quality_risk_boost=qboost,
        )
        row = {
            **r, "name": name, "risk_pct": risk, "exit_mode": exit_m,
            "target_rr": rr, "max_hold_days": hold, "cooldown": cd,
            "max_pos": mp, "qboost": qboost, "trail": trail, "signals": nsig,
        }
        results.append(row)
        if r["ret"] > local_best:
            local_best = r["ret"]
        if r["ret"] >= TARGET:
            winners.append(row)
            print(
                f"  *** HIT {r['ret']}% *** risk={risk} {exit_m} RR={rr} hold={hold} "
                f"cd={cd} mp={mp} qboost={qboost} WR={r['wr']} DD={r['dd']} n={r['n']}",
                flush=True,
            )
        if i % 80 == 0:
            gbest = max(results, key=lambda x: x["ret"])
            print(
                f"  … {i}/{len(uniq)} local_best={local_best:.1f}% "
                f"global={gbest['ret']}% ({gbest['name']} risk={gbest['risk_pct']} "
                f"{gbest['exit_mode']} RR={gbest['target_rr']} hold={gbest['max_hold_days']} "
                f"cd={gbest['cooldown']})",
                flush=True,
            )

    print(f"  pack best={local_best:.1f}%", flush=True)


def main():
    t0 = time.time()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print("Preload…", flush=True)
    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_w = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    weekly_by_sym, tech_by_sym = {}, {}
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < 40:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(d, include_supertrend=True)
    st, et = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })
    print(f"ready frames={len(frames)} days={len(calendar)}", flush=True)

    results, winners = [], []

    # Base packs
    pack_specs = [
        ("S2_trans_mtf", 2, "transition", "daily_mtf", 0, 0.95),
        ("S2_trans_none", 2, "transition", "none", 0, 0.95),
        ("S2_trans_notext", 2, "transition", "not_extended", 0, 0.95),
        ("S2_trans_mtf_ne", 2, "transition", "daily_mtf_not_ext", 0, 0.95),
        ("S2_in_mtf", 2, "in_stage", "daily_mtf", 0, 0.95),
        ("S2_in_none", 2, "in_stage", "none", 0, 0.95),
        ("S1_trans_mtf", 1, "transition", "daily_mtf", 0, 0.95),
        ("S1_trans_none", 1, "transition", "none", 0, 0.95),
        ("S2_trans_mtf_stop93", 2, "transition", "daily_mtf", 0, 0.93),
        ("S2_trans_mtf_Q50", 2, "transition", "daily_mtf", 50, 0.95),
        ("S2_in_mtf_Q50", 2, "in_stage", "daily_mtf", 50, 0.95),
    ]

    collected = {}
    for name, es, eon, tech, q, stop in pack_specs:
        t1 = time.time()
        sbd, n = collect(frames, weekly_by_sym, tech_by_sym, nifty_w, st, et, es, eon, tech, q, stop)
        print(f"collected {name}: {n} signals ({time.time()-t1:.1f}s)", flush=True)
        collected[name] = (sbd, n)
        run_grid(name, sbd, n, frames, weekly_by_sym, calendar, results, winners)

    # Dual stage merges
    print("\n### DUAL-STAGE MERGES ###", flush=True)
    merges = [
        ("S1+S2_trans_mtf", ["S1_trans_mtf", "S2_trans_mtf"]),
        ("S1+S2_trans_none", ["S1_trans_none", "S2_trans_none"]),
        ("S2_trans+in_mtf", ["S2_trans_mtf", "S2_in_mtf"]),
        ("S1+S2_trans+in_mtf", ["S1_trans_mtf", "S2_trans_mtf", "S2_in_mtf"]),
    ]
    for mname, keys in merges:
        packs = [collected[k] for k in keys if k in collected]
        if not packs:
            continue
        sbd, n = merge_signals(*packs)
        run_grid(mname, sbd, n, frames, weekly_by_sym, calendar, results, winners)

    # If still short, extreme risk on best packs
    if not winners:
        print("\n### EXTREME RISK PUSH on best packs ###", flush=True)
        best_names = []
        by_name = defaultdict(list)
        for r in results:
            by_name[r["name"]].append(r["ret"])
        ranked_names = sorted(by_name.keys(), key=lambda n: max(by_name[n]), reverse=True)
        for name in ranked_names[:4]:
            if name not in collected and not name.startswith("S1+"):
                # rebuild merge
                continue
            if name in collected:
                sbd, nsig = collected[name]
            else:
                continue
            print(f"extreme on {name} (best was {max(by_name[name]):.1f}%)", flush=True)
            for risk in [10, 12, 15, 18, 20, 25, 30]:
                for exit_m in ["trail_ma_s4", "no_stage", "stage_4_only"]:
                    for rr in [2.5, 3, 4, 5, 6]:
                        for hold in [100, 130, 200, 300]:
                            for cd in [0, 5, COOLDOWN_DAYS]:
                                r = simulate(
                                    frames, weekly_by_sym, sbd, calendar,
                                    capital=CAPITAL, risk_pct=float(risk),
                                    exit_mode=exit_m, target_rr=float(rr),
                                    max_hold_days=hold, cooldown_days=cd,
                                    max_positions=999, quality_risk_boost=True,
                                )
                                row = {
                                    **r, "name": name, "risk_pct": risk,
                                    "exit_mode": exit_m, "target_rr": rr,
                                    "max_hold_days": hold, "cooldown": cd,
                                    "max_pos": 999, "qboost": True, "trail": 0.98,
                                    "signals": nsig,
                                }
                                results.append(row)
                                if r["ret"] >= TARGET:
                                    winners.append(row)
                                    print(f"  *** HIT {r['ret']}% extreme risk={risk} {exit_m} RR={rr} hold={hold} cd={cd}", flush=True)

    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)
    print("\n" + "=" * 100)
    print(f"DONE {(time.time()-t0)/60:.1f}m | sims={len(results)} winners={len(winners)}")
    print("TOP 50:")
    for i, r in enumerate(ranked[:50], 1):
        mark = " ***" if r["ret"] >= TARGET else ""
        print(
            f"#{i:02d} {r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} n={r['n']:3d} "
            f"| {r['name']} risk={r['risk_pct']} {r['exit_mode']} RR={r['target_rr']} "
            f"hold={r['max_hold_days']} cd={r['cooldown']} mp={r['max_pos']} qb={r['qboost']}{mark}"
        )

    out = os.path.join(ROOT, "data", "_stage_v2_hunt_300_aggressive.txt")
    with open(out, "w", encoding="utf-8") as f:
        for r in ranked:
            f.write(f"{r}\n")
    print(f"Saved {out}")

    if winners:
        # Best return
        c1 = max(winners, key=lambda x: x["ret"])
        # Best return with DD < 45
        mild = [w for w in winners if w["dd"] < 45]
        # Best ret/dd
        c2 = max(winners, key=lambda x: x["ret"] / max(x["dd"], 1))
        print("\nCHAMPION max return:", c1)
        if mild:
            print("CHAMPION DD<45:", max(mild, key=lambda x: x["ret"]))
        print("CHAMPION ret/DD:", c2)
    else:
        print("\nNo 300% yet. Best:", ranked[0] if ranked else None)
        print("Will need structural changes (pyramiding / multi-strategy).")


if __name__ == "__main__":
    main()
