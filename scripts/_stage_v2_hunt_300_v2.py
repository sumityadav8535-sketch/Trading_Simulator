"""
Focused hunt for ≥300% (2023-01-01 → 2026-02-28).
Preload once, few signal packs, dense risk/exit/RR/hold grid, optional no-cooldown.
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
    frames,
    weekly_by_sym,
    signals_by_day,
    calendar,
    *,
    capital: float,
    risk_pct: float,
    exit_mode: str,
    target_rr: float,
    max_hold_days: int,
    trail_ma_mult: float = 0.98,
    cooldown_days: int = COOLDOWN_DAYS,
    max_positions: int = 999,
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
        closed_today: list[str] = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            pos.hold_days += 1

            exit_price = None
            exit_reason = ""
            stage_now: Optional[int] = None
            ma_now: Optional[float] = None

            if exit_mode != EXIT_NO_STAGE or trail_ma:
                weekly = weekly_by_sym.get(sym)
                if weekly is not None and not weekly.empty:
                    week_mask = weekly.index[
                        (weekly.index > (pos.last_week_check or pos.entry_date))
                        & (weekly.index <= ts)
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
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
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
                if sym in opens:
                    continue
                if sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if cooldown_days > 0 and prev_x is not None and (ts - prev_x).days < cooldown_days:
                    continue

                row = frames[sym].loc[ts]
                entry_price = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])
                stop = float(sig["stop"])
                risk = entry_price - stop
                if risk <= 0:
                    continue
                target = round(entry_price + risk * target_rr, 2)

                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    continue
                pos_size = calculate_position_size(equity_now, risk_pct, entry_price, stop)
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
                    symbol=sym,
                    entry_date=ts,
                    signal_date=sig["signal_date"],
                    entry_price=entry_price,
                    stop=stop,
                    target=target,
                    qty=qty,
                    notional=notional,
                    quality_score=int(sig.get("quality_score") or 0),
                    rs_rating=float(sig.get("rs_rating") or 0),
                    weekly_stage=int(sig.get("weekly_stage") or 2),
                    capital_invested=notional,
                    cash_available=cash,
                    total_invested=invested_after,
                    parallel_open=parallel,
                    equity_at_entry=cash + invested_after,
                )
                opens[sym] = pos

                if low <= stop:
                    trade = _close_trade(
                        pos, exit_price=stop, exit_ts=ts, exit_reason="stop_loss", days_held=0
                    )
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif high >= target:
                    trade = _close_trade(
                        pos, exit_price=target, exit_ts=ts, exit_reason=target_label, days_held=0
                    )
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
                pos,
                exit_price=float(hist.iloc[-1]["close"]),
                exit_ts=hist.index[-1],
                exit_reason="eod_force",
                days_held=pos.hold_days,
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
        "ret": round(ret, 2),
        "wr": round(wr, 2),
        "pf": round(pf, 2),
        "dd": round(max_dd, 2),
        "n": len(trades),
        "final": round(final_eq, 2),
        "peak_par": peak_parallel,
    }


def collect_pack(frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et, spec):
    entry_stage, entry_on, tech, q, stop_m = spec
    need_tech = pack_needs_tech_snapshot(tech)
    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n = 0
    for sym, weekly in weekly_by_sym.items():
        daily = frames[sym]
        sigs = _collect_stage2_signals(
            sym, weekly, daily,
            nifty_weekly if not nifty_weekly.empty else weekly,
            nifty_weekly, st, et,
            min_quality_score=q,
            market_filter=False,
            tech_filter=tech,
            daily_tech=tech_by_sym.get(sym) if need_tech else None,
            entry_stage=entry_stage,
            entry_on=entry_on,
            target_rr=2.5,
            stop_ma_mult=stop_m,
        )
        for s in sigs:
            signals_by_day[s["entry_day"]].append(s)
            n += 1
    return signals_by_day, n


def main():
    t0 = time.time()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print("Preload…", flush=True)
    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
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
    print(f"frames={len(frames)} days={len(calendar)}", flush=True)

    packs = [
        (2, "transition", "daily_mtf", 0, 0.95),
        (2, "transition", "none", 0, 0.95),
        (2, "transition", "not_extended", 0, 0.95),
        (2, "transition", "daily_mtf_not_ext", 0, 0.95),
        (2, "transition", "daily_mtf", 0, 0.93),
        (2, "transition", "daily_mtf", 0, 0.97),
        (2, "transition", "daily_mtf", 50, 0.95),
        (2, "transition", "daily_mtf", 75, 0.95),
        (2, "in_stage", "daily_mtf", 0, 0.95),
        (2, "in_stage", "none", 0, 0.95),
        (1, "transition", "daily_mtf", 0, 0.95),
        (1, "transition", "none", 0, 0.95),
        (1, "in_stage", "daily_mtf", 0, 0.95),
        (2, "transition", "ema_stack", 0, 0.95),
        (2, "transition", "bb_mid", 0, 0.95),
        (2, "transition", "confluence", 0, 0.95),
        (2, "transition", "daily_mtf", 0, 0.92),
        (2, "in_stage", "daily_mtf", 50, 0.95),
        (2, "in_stage", "not_extended", 0, 0.95),
        (1, "transition", "not_extended", 0, 0.95),
    ]

    results = []
    winners = []

    # Dense but finite grid
    risks = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    exits = ["stage_4_only", "trail_ma_s4", "no_stage", "stage_3_4"]
    rrs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    holds = [40, 50, 65, 80, 100, 130, 180, 250]
    cooldowns = [COOLDOWN_DAYS, 0, 5]
    max_pos_opts = [999, 5, 8, 12]

    for pi, spec in enumerate(packs):
        print(f"\n=== PACK {pi+1}/{len(packs)} {spec} ===", flush=True)
        t1 = time.time()
        sbd, nsig = collect_pack(frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et, spec)
        print(f"signals={nsig} in {time.time()-t1:.1f}s", flush=True)
        if nsig < 10:
            continue

        # Phase A: risk scale on default exit
        for risk in risks:
            r = simulate(
                frames, weekly_by_sym, sbd, calendar,
                capital=CAPITAL, risk_pct=risk, exit_mode="stage_4_only",
                target_rr=2.5, max_hold_days=65,
            )
            row = {**r, "risk_pct": risk, "exit_mode": "stage_4_only", "target_rr": 2.5,
                   "max_hold_days": 65, "cooldown": COOLDOWN_DAYS, "max_pos": 999,
                   "pack": spec, "signals": nsig}
            results.append(row)
            if r["ret"] >= TARGET:
                winners.append(row)
                print(f"  HIT risk-scale {r['ret']}% risk={risk}", flush=True)

        probe = max(
            (x for x in results if x["pack"] == spec and x["exit_mode"] == "stage_4_only"),
            key=lambda x: x["ret"],
        )
        print(f"  best risk-scale on pack: {probe['ret']}% @ risk={probe['risk_pct']}", flush=True)

        # Phase B: full grid around promising risks
        top_risks = sorted(
            {x["risk_pct"] for x in results if x["pack"] == spec},
            key=lambda rk: max(x["ret"] for x in results if x["pack"] == spec and x["risk_pct"] == rk),
            reverse=True,
        )[:5]
        # always include high risks
        for rk in [4.0, 6.0, 8.0, 10.0, 12.0]:
            if rk not in top_risks:
                top_risks.append(rk)

        count_b = 0
        for risk, exit_m, rr, hold in itertools.product(top_risks, exits, rrs, holds):
            r = simulate(
                frames, weekly_by_sym, sbd, calendar,
                capital=CAPITAL, risk_pct=risk, exit_mode=exit_m,
                target_rr=rr, max_hold_days=hold,
            )
            row = {**r, "risk_pct": risk, "exit_mode": exit_m, "target_rr": rr,
                   "max_hold_days": hold, "cooldown": COOLDOWN_DAYS, "max_pos": 999,
                   "pack": spec, "signals": nsig}
            results.append(row)
            count_b += 1
            if r["ret"] >= TARGET:
                winners.append(row)
                print(
                    f"  *** HIT {r['ret']}% risk={risk} {exit_m} RR={rr} hold={hold} "
                    f"WR={r['wr']} DD={r['dd']} n={r['n']}",
                    flush=True,
                )
            if count_b % 100 == 0:
                best = max(results, key=lambda x: x["ret"])
                print(f"  … {count_b} sims, global best {best['ret']}%", flush=True)

        # Phase C: cooldown / max positions on best 3 of this pack
        pack_best = sorted(
            [x for x in results if x["pack"] == spec],
            key=lambda x: x["ret"], reverse=True,
        )[:5]
        for b in pack_best:
            for cd, mp in itertools.product(cooldowns, max_pos_opts):
                if cd == COOLDOWN_DAYS and mp == 999:
                    continue
                r = simulate(
                    frames, weekly_by_sym, sbd, calendar,
                    capital=CAPITAL,
                    risk_pct=b["risk_pct"],
                    exit_mode=b["exit_mode"],
                    target_rr=b["target_rr"],
                    max_hold_days=b["max_hold_days"],
                    cooldown_days=cd,
                    max_positions=mp,
                )
                row = {**r, "risk_pct": b["risk_pct"], "exit_mode": b["exit_mode"],
                       "target_rr": b["target_rr"], "max_hold_days": b["max_hold_days"],
                       "cooldown": cd, "max_pos": mp, "pack": spec, "signals": nsig}
                results.append(row)
                if r["ret"] >= TARGET:
                    winners.append(row)
                    print(f"  *** HIT {r['ret']}% cooldown={cd} maxpos={mp}", flush=True)

        pack_top = max(x["ret"] for x in results if x["pack"] == spec)
        print(f"  pack top so far: {pack_top}% | winners total={len(winners)}", flush=True)

        # If we have winners with DD < 60, keep going for better but can stop early after 8 packs
        solid = [w for w in winners if w["dd"] < 60]
        if solid and pi >= 7:
            print("Solid winners found — finishing remaining high-priority packs only…", flush=True)

    # Final refine on global top 10
    print("\n=== REFINE TOP ===", flush=True)
    top10 = sorted(results, key=lambda x: x["ret"], reverse=True)[:10]
    for b in top10:
        spec = b["pack"]
        sbd, nsig = collect_pack(frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et, spec)
        for risk in [b["risk_pct"] - 0.5, b["risk_pct"], b["risk_pct"] + 0.5, b["risk_pct"] + 1, b["risk_pct"] + 2]:
            if risk < 1:
                continue
            for rr in [b["target_rr"] - 0.5, b["target_rr"], b["target_rr"] + 0.5, b["target_rr"] + 1]:
                if rr < 1:
                    continue
                for hold in [b["max_hold_days"] - 20, b["max_hold_days"], b["max_hold_days"] + 30, b["max_hold_days"] + 60]:
                    if hold < 20:
                        continue
                    for trail in [0.95, 0.96, 0.97, 0.98, 0.99]:
                        r = simulate(
                            frames, weekly_by_sym, sbd, calendar,
                            capital=CAPITAL, risk_pct=risk, exit_mode=b["exit_mode"],
                            target_rr=rr, max_hold_days=int(hold), trail_ma_mult=trail,
                            cooldown_days=b.get("cooldown", COOLDOWN_DAYS),
                            max_positions=b.get("max_pos", 999),
                        )
                        row = {**r, "risk_pct": risk, "exit_mode": b["exit_mode"],
                               "target_rr": rr, "max_hold_days": int(hold),
                               "trail": trail, "cooldown": b.get("cooldown", COOLDOWN_DAYS),
                               "max_pos": b.get("max_pos", 999), "pack": spec, "signals": nsig}
                        results.append(row)
                        if r["ret"] >= TARGET:
                            winners.append(row)
                            print(f"  refine HIT {r['ret']}%", flush=True)

    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)
    print("\n" + "=" * 100)
    print(f"DONE {(time.time()-t0)/60:.1f} min | sims={len(results)} | winners={len(winners)}")
    print("=" * 100)
    print("\nTOP 40:")
    for i, r in enumerate(ranked[:40], 1):
        print(
            f"#{i:02d} {r['ret']:+7.1f}% WR={r['wr']:5.1f} PF={r['pf']:5.2f} DD={r['dd']:5.1f} n={r['n']:3d} "
            f"| risk={r['risk_pct']} {r['exit_mode']} RR={r['target_rr']} hold={r['max_hold_days']} "
            f"cd={r.get('cooldown')} mp={r.get('max_pos')} pack={r['pack']}"
        )

    out = os.path.join(ROOT, "data", "_stage_v2_hunt_300_v2.txt")
    with open(out, "w", encoding="utf-8") as f:
        for r in ranked:
            f.write(f"{r}\n")
    print(f"Saved {out}")

    if winners:
        champ = max(winners, key=lambda x: (x["ret"], -x["dd"]))
        print("\nCHAMPION:")
        print(champ)
    else:
        print("\nNo 300% — best:")
        print(ranked[0])


if __name__ == "__main__":
    main()
