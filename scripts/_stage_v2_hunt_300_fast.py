"""
Fast hunt for ≥300% over ~3y on Stage Analysis 2.0 (shared cash, real sizing).

Preloads once → collect signal packs → simulate many risk/exit/RR/hold combos.
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
from stage_analysis_v2.services.tech_filters import (
    enrich_daily_tech,
    pack_needs_supertrend,
    pack_needs_tech_snapshot,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

START = date(2023, 1, 1)
END = date(2026, 2, 28)
CAPITAL = 1_000_000.0
TARGET = 300.0


def build_context(symbols: list[str]):
    print("Preloading frames…", flush=True)
    frames = _preload_frames(symbols)
    print(f"  {len(frames)} stocks with data", flush=True)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = (
        add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    )
    weekly_by_sym: dict[str, pd.DataFrame] = {}
    tech_by_sym: dict[str, pd.DataFrame] = {}
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < 40:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(d, include_supertrend=True)

    st, et = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts
        for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })
    print(f"  calendar days={len(calendar)} weekly maps={len(weekly_by_sym)}", flush=True)
    return frames, weekly_by_sym, tech_by_sym, nifty_weekly, calendar, st, et


def collect_signals(
    frames,
    weekly_by_sym,
    tech_by_sym,
    nifty_weekly,
    st,
    et,
    *,
    entry_stage: int,
    entry_on: str,
    tech_filter: str,
    min_quality: int,
    market_filter: bool,
    stop_ma_mult: float,
) -> dict[pd.Timestamp, list[dict]]:
    need_tech = pack_needs_tech_snapshot(tech_filter)
    signals_by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n = 0
    for sym, weekly in weekly_by_sym.items():
        daily = frames[sym]
        daily_tech = tech_by_sym.get(sym) if need_tech else None
        bench = nifty_weekly if not nifty_weekly.empty else weekly
        sigs = _collect_stage2_signals(
            sym,
            weekly,
            daily,
            bench,
            nifty_weekly,
            st,
            et,
            min_quality_score=min_quality,
            market_filter=market_filter,
            tech_filter=tech_filter,
            daily_tech=daily_tech,
            entry_stage=entry_stage,
            entry_on=entry_on,
            target_rr=2.5,  # recomputed in sim
            stop_ma_mult=stop_ma_mult,
        )
        for s in sigs:
            signals_by_day[s["entry_day"]].append(s)
            n += 1
    return signals_by_day, n


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
    skipped_cash = 0

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
                exit_price = pos.stop
                exit_reason = "stop_loss"
            elif high >= pos.target:
                exit_price = pos.target
                exit_reason = target_label
            elif pos.hold_days >= max_hold_days:
                exit_price = close
                exit_reason = "time_exit"
            elif stage_now is not None and exit_mode != EXIT_NO_STAGE:
                if exit_mode == EXIT_STAGE_3_4 and stage_now in (3, 4):
                    exit_price = close
                    exit_reason = "stage_exit"
                elif exit_mode in (EXIT_STAGE_4_ONLY, EXIT_TRAIL_MA_S4) and stage_now == 4:
                    exit_price = close
                    exit_reason = "stage_exit"

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
        if not day_sigs:
            equity_curve.append(_equity(cash, opens))
            continue

        day_sigs = sorted(
            day_sigs,
            key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
            reverse=True,
        )
        for sig in day_sigs:
            sym = sig["symbol"]
            if sym in opens:
                continue
            if sym not in frames or ts not in frames[sym].index:
                continue
            prev_x = last_exit.get(sym)
            if prev_x is not None and (ts - prev_x).days < COOLDOWN_DAYS:
                continue

            row = frames[sym].loc[ts]
            entry_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            stop = float(sig["stop"])
            risk = entry_price - stop
            if risk <= 0:
                continue
            # recompute target from this RR
            target = round(entry_price + risk * target_rr, 2)

            equity_now = _equity(cash, opens)
            if cash <= 0 or equity_now <= 0:
                skipped_cash += 1
                continue
            pos_size = calculate_position_size(equity_now, risk_pct, entry_price, stop)
            qty = int(pos_size.quantity)
            max_qty_cash = int(cash // entry_price) if entry_price > 0 else 0
            qty = min(qty, max_qty_cash)
            if qty <= 0:
                skipped_cash += 1
                continue
            notional = qty * entry_price
            if notional > cash + 1e-6:
                skipped_cash += 1
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
            close = float(hist.iloc[-1]["close"])
            trade = _close_trade(
                pos, exit_price=close, exit_ts=hist.index[-1],
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)
        opens.clear()

    final_eq = cash
    ret = (final_eq - capital) / capital * 100
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    pf = gp / gl if trades else 0.0
    peak = capital
    max_dd = 0.0
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
        "skipped_cash": skipped_cash,
        "trades": trades,
    }


def main() -> None:
    t0 = time.time()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames, weekly_by_sym, tech_by_sym, nifty_weekly, calendar, st, et = build_context(symbols)

    # Signal packs: entry/tech/quality/stop/market (expensive)
    packs_spec = []
    for entry_stage, entry_on in [(2, "transition"), (2, "in_stage"), (1, "transition"), (1, "in_stage")]:
        for tech in ["none", "daily_mtf", "not_extended", "daily_mtf_not_ext", "ema_stack", "bb_mid"]:
            for q in [0, 50, 75]:
                for stop_m in [0.93, 0.95, 0.97]:
                    for mkt in [False]:  # market filter kills returns
                        packs_spec.append((entry_stage, entry_on, tech, q, stop_m, mkt))

    # Prioritize Stage2 transition + daily_mtf first
    def pack_prio(p):
        es, eon, tech, q, stop, mkt = p
        s = 0
        if es == 2 and eon == "transition":
            s += 50
        if tech in ("daily_mtf", "none", "not_extended"):
            s += 20
        if q == 0:
            s += 10
        if stop == 0.95:
            s += 5
        return -s

    packs_spec.sort(key=pack_prio)

    results: list[dict] = []
    winners: list[dict] = []
    packs_done = 0
    max_packs = 40  # expand if needed
    sims_done = 0

    risk_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    exit_grid = ["stage_4_only", "trail_ma_s4", "no_stage", "stage_3_4"]
    rr_grid = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    hold_grid = [40, 65, 90, 130, 200, 300]

    print(f"\nHunting ≥{TARGET}% | {START}→{END} | shared capital ₹{CAPITAL:,.0f}", flush=True)
    print("=" * 110, flush=True)

    for (entry_stage, entry_on, tech, q, stop_m, mkt) in packs_spec:
        if packs_done >= max_packs and winners:
            break
        if packs_done >= max_packs * 2:
            break

        print(
            f"\n--- Signal pack {packs_done+1}: stage={entry_stage}/{entry_on} tech={tech} "
            f"Q>={q} stop={stop_m} mkt={mkt} ---",
            flush=True,
        )
        t1 = time.time()
        signals_by_day, nsig = collect_signals(
            frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et,
            entry_stage=entry_stage,
            entry_on=entry_on,
            tech_filter=tech,
            min_quality=q,
            market_filter=mkt,
            stop_ma_mult=stop_m,
        )
        print(f"  signals={nsig} collect {time.time()-t1:.1f}s", flush=True)
        packs_done += 1
        if nsig < 5:
            continue

        # Adaptive sim grid: denser when pack looks promising
        # Quick probe at risk=5 stage_4 2.5R hold65
        probe = simulate(
            frames, weekly_by_sym, signals_by_day, calendar,
            capital=CAPITAL, risk_pct=5.0, exit_mode="stage_4_only",
            target_rr=2.5, max_hold_days=65,
        )
        print(f"  probe risk5: ret={probe['ret']}% n={probe['n']} WR={probe['wr']}%", flush=True)

        # Choose grids based on probe
        if probe["ret"] >= 100 or nsig >= 150:
            risks = risk_grid
            exits = exit_grid
            rrs = rr_grid
            holds = hold_grid
        elif probe["ret"] >= 40:
            risks = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
            exits = ["stage_4_only", "trail_ma_s4", "no_stage"]
            rrs = [2.0, 2.5, 3.0, 4.0, 5.0]
            holds = [65, 90, 130, 200]
        else:
            risks = [5.0, 8.0, 10.0, 12.0]
            exits = ["trail_ma_s4", "no_stage", "stage_4_only"]
            rrs = [2.5, 3.0, 4.0]
            holds = [90, 130, 200]

        for risk, exit_m, rr, hold in itertools.product(risks, exits, rrs, holds):
            r = simulate(
                frames, weekly_by_sym, signals_by_day, calendar,
                capital=CAPITAL,
                risk_pct=risk,
                exit_mode=exit_m,
                target_rr=rr,
                max_hold_days=hold,
                trail_ma_mult=0.98,
            )
            sims_done += 1
            row = {
                **r,
                "risk_pct": risk,
                "exit_mode": exit_m,
                "target_rr": rr,
                "max_hold_days": hold,
                "entry_stage": entry_stage,
                "entry_on": entry_on,
                "tech_filter": tech,
                "min_quality_score": q,
                "stop_ma_mult": stop_m,
                "market_filter": mkt,
                "signals": nsig,
            }
            results.append(row)
            if r["ret"] >= TARGET:
                winners.append(row)
                print(
                    f"  *** HIT {r['ret']}% *** risk={risk} exit={exit_m} RR={rr} hold={hold} "
                    f"WR={r['wr']}% DD={r['dd']}% n={r['n']}",
                    flush=True,
                )
            elif sims_done % 40 == 0:
                best_so_far = max(results, key=lambda x: x["ret"])
                print(
                    f"  … sims={sims_done} best_so_far={best_so_far['ret']}% "
                    f"(risk={best_so_far['risk_pct']} {best_so_far['exit_mode']} "
                    f"RR={best_so_far['target_rr']} hold={best_so_far['max_hold_days']})",
                    flush=True,
                )

        # If this pack produced winners, refine around them
        pack_winners = [w for w in winners if w.get("tech_filter") == tech and w.get("entry_stage") == entry_stage and w.get("entry_on") == entry_on]
        if pack_winners:
            best_w = max(pack_winners, key=lambda x: x["ret"])
            print(f"  Refining around best pack winner {best_w['ret']}%…", flush=True)
            for risk in [best_w["risk_pct"] - 1, best_w["risk_pct"], best_w["risk_pct"] + 1, best_w["risk_pct"] + 2, best_w["risk_pct"] + 3]:
                if risk < 1:
                    continue
                for rr in [best_w["target_rr"] - 0.5, best_w["target_rr"], best_w["target_rr"] + 0.5, best_w["target_rr"] + 1]:
                    if rr < 1:
                        continue
                    for hold in [best_w["max_hold_days"] - 20, best_w["max_hold_days"], best_w["max_hold_days"] + 40, best_w["max_hold_days"] + 80]:
                        if hold < 20:
                            continue
                        for trail in [0.96, 0.97, 0.98, 0.99]:
                            r = simulate(
                                frames, weekly_by_sym, signals_by_day, calendar,
                                capital=CAPITAL,
                                risk_pct=risk,
                                exit_mode=best_w["exit_mode"],
                                target_rr=rr,
                                max_hold_days=int(hold),
                                trail_ma_mult=trail,
                            )
                            sims_done += 1
                            row = {
                                **r,
                                "risk_pct": risk,
                                "exit_mode": best_w["exit_mode"],
                                "target_rr": rr,
                                "max_hold_days": int(hold),
                                "trail_ma_mult": trail,
                                "entry_stage": entry_stage,
                                "entry_on": entry_on,
                                "tech_filter": tech,
                                "min_quality_score": q,
                                "stop_ma_mult": stop_m,
                                "market_filter": mkt,
                                "signals": nsig,
                            }
                            results.append(row)
                            if r["ret"] >= TARGET:
                                winners.append(row)
                                if r["ret"] > best_w["ret"]:
                                    print(
                                        f"  *** BETTER HIT {r['ret']}% *** risk={risk} RR={rr} hold={hold} trail={trail} DD={r['dd']}%",
                                        flush=True,
                                    )

        # Early stop if we have a solid winner with DD < 50%
        good = [w for w in winners if w["dd"] < 55 and w["ret"] >= TARGET]
        if good and packs_done >= 6:
            print("\nEnough solid winners — stopping pack expansion early.", flush=True)
            break

    # If no winners, expand with more aggressive packs we may have skipped
    if not winners:
        print("\n### No 300% yet — expanding remaining high-priority packs ###", flush=True)
        extra = [
            (2, "transition", "none", 0, 0.95, False),
            (2, "transition", "daily_mtf", 0, 0.93, False),
            (2, "in_stage", "daily_mtf", 0, 0.95, False),
            (1, "transition", "daily_mtf", 0, 0.95, False),
            (2, "transition", "not_extended", 0, 0.95, False),
            (2, "transition", "daily_mtf_not_ext", 0, 0.95, False),
        ]
        for spec in extra:
            if packs_done > 60:
                break
            entry_stage, entry_on, tech, q, stop_m, mkt = spec
            signals_by_day, nsig = collect_signals(
                frames, weekly_by_sym, tech_by_sym, nifty_weekly, st, et,
                entry_stage=entry_stage, entry_on=entry_on, tech_filter=tech,
                min_quality=q, market_filter=mkt, stop_ma_mult=stop_m,
            )
            packs_done += 1
            print(f"extra pack signals={nsig} {spec}", flush=True)
            for risk, exit_m, rr, hold in itertools.product(
                [6, 8, 10, 12, 15, 18, 20],
                ["trail_ma_s4", "no_stage", "stage_4_only"],
                [2.5, 3, 4, 5, 6],
                [90, 130, 200, 300],
            ):
                r = simulate(
                    frames, weekly_by_sym, signals_by_day, calendar,
                    capital=CAPITAL, risk_pct=float(risk), exit_mode=exit_m,
                    target_rr=float(rr), max_hold_days=int(hold),
                )
                sims_done += 1
                row = {
                    **r, "risk_pct": float(risk), "exit_mode": exit_m,
                    "target_rr": float(rr), "max_hold_days": int(hold),
                    "entry_stage": entry_stage, "entry_on": entry_on,
                    "tech_filter": tech, "min_quality_score": q,
                    "stop_ma_mult": stop_m, "market_filter": mkt, "signals": nsig,
                }
                results.append(row)
                if r["ret"] >= TARGET:
                    winners.append(row)
                    print(f"  *** HIT {r['ret']}% *** {row}", flush=True)

    elapsed = time.time() - t0
    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)

    print("\n" + "=" * 110)
    print(f"DONE {elapsed/60:.1f} min | packs={packs_done} sims={sims_done} winners={len(winners)}")
    print("=" * 110)
    print("\n### TOP 30 ###")
    for i, r in enumerate(ranked[:30], 1):
        print(
            f"#{i:02d} ret={r['ret']:+7.1f}% WR={r['wr']:5.1f}% PF={r['pf']:5.2f} DD={r['dd']:5.1f}% "
            f"n={r['n']:3d} | risk={r['risk_pct']}% {r['exit_mode']} RR={r['target_rr']} hold={r['max_hold_days']} "
            f"stage={r['entry_stage']}/{r['entry_on']} tech={r['tech_filter']} Q>={r['min_quality_score']} stop={r['stop_ma_mult']}"
        )

    out = os.path.join(ROOT, "data", "_stage_v2_hunt_300_fast.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"sims={sims_done} winners={len(winners)}\n")
        for r in ranked:
            f.write(f"{r}\n")
    print(f"\nSaved {out}")

    if winners:
        # Prefer high return with controlled DD
        champ = sorted(winners, key=lambda x: (x["ret"], -x["dd"]), reverse=True)[0]
        # Also best return/DD ratio among winners
        robust = sorted(
            [w for w in winners if w["n"] >= 30],
            key=lambda x: (x["ret"] / max(x["dd"], 1), x["ret"]),
            reverse=True,
        )
        print("\n### CHAMPION (max return among ≥300%) ###")
        print(champ)
        if robust:
            print("\n### MOST ROBUST winner (ret/DD, n≥30) ###")
            print(robust[0])
    else:
        print("\nNo 300% config found. Best:")
        print(ranked[0] if ranked else "none")


if __name__ == "__main__":
    main()
