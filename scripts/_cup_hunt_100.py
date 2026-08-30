"""Hunt a Cup Breakout pack that can do ≥100% in 1 year, then score other years."""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import numpy as np
import pandas as pd

from stage_analysis_v2.services.backtester import (
    EntryFilters,
    _OpenPos,
    _close_trade,
    _equity,
    _invested_total,
    _passes_entry_filters,
    _preload_frames,
    maybe_raise_stop,
    stop_exit_reason,
)
from stage_analysis_v2.services.cup_breakout import (
    CUP_ENTRY_NEXT_OPEN,
    CUP_EXIT_EMA20,
    CUP_EXIT_MEASURED,
    CUP_EXIT_TARGET_R,
    CupParams,
    CupSetup,
    _CupPack,
    _passes_breakout_filters,
    _retest_status,
    _target_price,
    detect_cup_shape,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
TODAY = date(2026, 8, 24)

WINDOWS = [
    ("last_1y", date(2025, 8, 24), date(2026, 8, 24)),
    ("y2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("y2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("y2025", date(2025, 1, 1), date(2025, 12, 31)),
    ("y2023_24", date(2023, 8, 24), date(2024, 8, 23)),
    ("y2024_25", date(2024, 8, 24), date(2025, 8, 23)),
]


def collect_signals(packs, frames, p: CupParams, nifty_ret60, nifty_sma200, nifty_close):
    filters = EntryFilters()
    warmup = max(210, p.cup_max_days + p.pivot_width + 5)
    by_day: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    n_sig = 0
    for sym, s in packs.items():
        daily = frames[sym]
        n = len(s.c)
        for i in range(warmup, n - 1):
            setup = detect_cup_shape(s.h, s.l, s.c, i, p)
            if setup is None:
                continue
            if not _passes_breakout_filters(
                s, i, setup, p, nifty_ret60, nifty_sma200, nifty_close,
            ):
                continue
            ts = s.index[i]
            if not _passes_entry_filters(daily, ts, filters):
                continue
            atr = float(s.atr[i]) if s.atr[i] == s.atr[i] else 0.0
            rs = 50.0
            stock_r = float(s.ret60[i]) if s.ret60[i] == s.ret60[i] else 0.0
            if nifty_ret60 is not None and ts in nifty_ret60.index and pd.notna(nifty_ret60.loc[ts]):
                n_r = float(nifty_ret60.loc[ts])
                rs = min(99.0, max(1.0, 50.0 + (stock_r - n_r) * 200.0))
            entry_i = i + 1
            sig_ts = ts
            if p.entry_mode != CUP_ENTRY_NEXT_OPEN:
                tol = p.retest_tol_pct / 100.0
                found = False
                j_end = min(n - 1, i + 1 + p.retest_max_days)
                for j in range(i + 1, j_end):
                    status = _retest_status(s.l, s.c, j, setup.cup_high, tol)
                    if status == "fail":
                        break
                    if status == "ok":
                        entry_i = j + 1
                        sig_ts = s.index[j]
                        found = True
                        break
                if not found or entry_i >= n:
                    continue
            if entry_i >= n:
                continue
            by_day[s.index[entry_i]].append({
                "symbol": sym,
                "sig_ts": sig_ts,
                "setup": setup,
                "atr": atr,
                "score": setup.quality,
                "quality": setup.quality,
                "rs": round(rs, 1),
            })
            n_sig += 1
    return by_day, n_sig


def simulate(
    packs,
    calendar,
    signals_by_day,
    *,
    risk_pct: float,
    max_hold_days: int,
    cooldown_days: int,
    max_pos_pct: float,
    params: CupParams,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
):
    cash = float(CAPITAL)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    n_trades = 0
    n_wins = 0
    gp = 0.0
    gl = 0.0
    hold_sum = 0
    peak_eq = CAPITAL
    max_dd = 0.0
    skipped = 0

    def mark(eq: float) -> None:
        nonlocal peak_eq, max_dd
        peak_eq = max(peak_eq, eq)
        if peak_eq:
            max_dd = max(max_dd, (peak_eq - eq) / peak_eq * 100.0)

    for ts in calendar:
        if ts < start_ts or ts > end_ts:
            continue
        closed_today: list[str] = []
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            pos.hold_days += 1
            close = float(s.c[i])
            high = float(s.h[i])
            low = float(s.l[i])
            e20 = float(s.ema20[i]) if s.ema20[i] == s.ema20[i] else 0.0
            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, stop_exit_reason(pos)
            elif pos.target > 0 and high >= pos.target:
                exit_price, exit_reason = pos.target, "target"
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif params.cup_exit_mode == CUP_EXIT_EMA20 and e20 > 0 and close < e20:
                exit_price, exit_reason = close, "ema20_exit"
            if exit_price is None:
                if params.cup_exit_mode == CUP_EXIT_EMA20 and e20 > 0:
                    maybe_raise_stop(pos, e20, close)
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            n_trades += 1
            hold_sum += pos.hold_days
            if trade.pnl > 0:
                n_wins += 1
                gp += trade.pnl
            else:
                gl += abs(trade.pnl)
            last_exit[sym] = ts
            closed_today.append(sym)
        for sym in closed_today:
            opens.pop(sym, None)

        day_sigs = signals_by_day.get(ts)
        if day_sigs:
            taken = 0
            for sig in sorted(day_sigs, key=lambda x: x["score"], reverse=True):
                if taken >= params.max_new_per_day:
                    break
                sym = sig["symbol"]
                if sym in opens:
                    continue
                s = packs.get(sym)
                if s is None:
                    continue
                prev = last_exit.get(sym)
                if cooldown_days > 0 and prev is not None and (ts - prev).days < cooldown_days:
                    continue
                i = s.loc.get(ts)
                if i is None:
                    continue
                entry = float(s.o[i])
                if entry <= 0:
                    continue
                setup: CupSetup = sig["setup"]
                atr = float(sig.get("atr") or 0.0)
                stop = float(setup.stop_ref) - params.stop_atr_mult * atr
                if stop <= 0 or stop >= entry:
                    fallback = entry - (1.0 * atr if atr > 0 else entry * 0.04)
                    stop = fallback if 0 < fallback < entry else 0.0
                if stop <= 0 or stop >= entry:
                    continue
                stop_pct = (entry - stop) / entry
                if stop_pct > params.max_stop_pct or stop_pct < params.min_stop_pct:
                    continue
                target = _target_price(entry, stop, setup, params)
                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    skipped += 1
                    continue
                ps = calculate_position_size(equity_now, risk_pct, entry, stop)
                max_notional = equity_now * (max_pos_pct / 100.0)
                qty_cash = int(cash // entry) if entry else 0
                qty_cap = int(max_notional // entry) if entry else 0
                qty = min(int(ps.quantity), qty_cash, qty_cap)
                if qty <= 0:
                    skipped += 1
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    skipped += 1
                    continue
                cash -= notional
                pos = _OpenPos(
                    symbol=sym,
                    entry_date=ts,
                    signal_date=sig["sig_ts"],
                    entry_price=entry,
                    stop=stop,
                    target=target,
                    qty=qty,
                    notional=notional,
                    quality_score=int(sig["quality"]),
                    rs_rating=float(sig["rs"]),
                    weekly_stage=0,
                    hold_days=0,
                    entry_stop=stop,
                )
                opens[sym] = pos
                taken += 1
                if s.l[i] <= stop:
                    trade = _close_trade(
                        pos, exit_price=stop, exit_ts=ts,
                        exit_reason="stop_loss", days_held=0,
                    )
                    cash += pos.notional + trade.pnl
                    n_trades += 1
                    if trade.pnl > 0:
                        n_wins += 1
                        gp += trade.pnl
                    else:
                        gl += abs(trade.pnl)
                    last_exit[sym] = ts
                    opens.pop(sym, None)

        if ts.weekday() == 4 or ts == calendar[-1]:
            mark(_equity(cash, opens))

    if opens:
        last_ts = min(end_ts, calendar[-1])
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(last_ts, len(s.c) - 1)
            close = float(s.c[i])
            trade = _close_trade(
                pos, exit_price=close, exit_ts=last_ts,
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            n_trades += 1
            hold_sum += pos.hold_days
            if trade.pnl > 0:
                n_wins += 1
                gp += trade.pnl
            else:
                gl += abs(trade.pnl)
        opens.clear()
        mark(cash)

    ret = (cash - CAPITAL) / CAPITAL * 100.0 if CAPITAL else 0.0
    wr = (n_wins / n_trades * 100.0) if n_trades else 0.0
    pf = (gp / gl) if gl > 1e-9 else (99.0 if gp > 0 else 0.0)
    avg_hold = (hold_sum / n_trades) if n_trades else 0.0
    return {
        "ret": round(ret, 2),
        "wr": round(wr, 2),
        "pf": round(pf, 2),
        "dd": round(max_dd, 2),
        "n": n_trades,
        "final": round(cash, 0),
        "avg_hold": round(avg_hold, 1),
        "skip": skipped,
    }


def filter_packs() -> list[tuple[str, CupParams]]:
    base = CupParams()
    return [
        ("default", base),
        ("loose_confirm", replace(
            base,
            require_sma200_rising=False,
            require_rs_vs_nifty=False,
            require_close_strength=False,
            vol_mult=1.2,
            rsi_min=40.0,
            rsi_max=85.0,
            max_gap_pct=8.0,
            min_bottom_days=6,
        )),
        ("wide_cup", replace(
            base,
            cup_min_days=20,
            cup_max_days=180,
            min_depth_pct=12.0,
            max_depth_pct=45.0,
            min_bottom_days=5,
            min_left_days=7,
            min_recovery_days=5,
            pivot_width=5,
            require_sma200_rising=False,
            require_rs_vs_nifty=False,
            require_close_strength=False,
            vol_mult=1.1,
            rsi_min=40.0,
            rsi_max=85.0,
        )),
        ("no_stack", replace(
            base,
            cup_min_days=20,
            cup_max_days=180,
            min_depth_pct=12.0,
            max_depth_pct=50.0,
            min_bottom_days=4,
            min_left_days=6,
            min_recovery_days=5,
            pivot_width=5,
            require_sma200_rising=False,
            require_rs_vs_nifty=False,
            require_close_strength=False,
            require_trend_stack=False,
            vol_mult=1.0,
            rsi_min=35.0,
            rsi_max=90.0,
            max_gap_pct=8.0,
            min_stop_pct=0.008,
            max_stop_pct=0.22,
        )),
        ("mid_stack_off_rs", replace(
            base,
            cup_min_days=25,
            cup_max_days=160,
            min_depth_pct=12.0,
            max_depth_pct=42.0,
            min_bottom_days=5,
            require_sma200_rising=False,
            require_rs_vs_nifty=False,
            require_close_strength=True,
            min_close_loc=0.55,
            vol_mult=1.2,
            rsi_min=45.0,
            rsi_max=85.0,
            require_trend_stack=True,
        )),
    ]


def sim_grid():
    exits = [
        (CUP_EXIT_TARGET_R, 1.5),
        (CUP_EXIT_TARGET_R, 2.0),
        (CUP_EXIT_TARGET_R, 3.0),
        (CUP_EXIT_TARGET_R, 4.0),
        (CUP_EXIT_MEASURED, 0.0),
        (CUP_EXIT_EMA20, 0.0),
    ]
    for risk in (4.0, 6.0, 8.0, 10.0):
        for hold in (45, 65, 90):
            for cooldown in (0, 5):
                for pos in (80.0, 100.0):
                    for exit_mode, rr in exits:
                        for max_new in (8, 12):
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
    print(f"Preloading {len(symbols)} symbols…", flush=True)
    frames = _preload_frames(symbols)
    packs = {sym: _CupPack(sym, df) for sym, df in frames.items()}
    print(f"  packs={len(packs)} ({time.time()-t0:.0f}s)", flush=True)

    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret60 = nifty_df["close"].pct_change(60) if not nifty_df.empty else None
    nifty_sma200 = nifty_df["close"].rolling(200, min_periods=200).mean() if not nifty_df.empty else None
    nifty_close = nifty_df["close"] if not nifty_df.empty else None

    full_cal = sorted({
        ts for df in frames.values() for ts in df.index.tolist()
    })
    window_cals = {
        name: [ts for ts in full_cal if pd.Timestamp(w0) <= ts <= pd.Timestamp(w1)]
        for name, w0, w1 in WINDOWS
    }

    hits: list[dict] = []
    for pack_name, p0 in filter_packs():
        t1 = time.time()
        by_day, n_sig = collect_signals(packs, frames, p0, nifty_ret60, nifty_sma200, nifty_close)
        print(f"\n=== pack {pack_name}  signals={n_sig}  collect={time.time()-t1:.0f}s ===", flush=True)
        if n_sig < 20:
            print("  too few signals, skip", flush=True)
            continue

        n_grid = 0
        pack_hits = 0
        best_last = None
        for g in sim_grid():
            n_grid += 1
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
            last = year_stats["last_1y"]
            if best_last is None or last["ret"] > best_last["ret"]:
                best_last = {**g, "pack": pack_name, **last}
            if last["ret"] >= 100.0 and last["n"] >= 15:
                pack_hits += 1
                other = [year_stats[k]["ret"] for k in year_stats if k != "last_1y"]
                row = {
                    "pack": pack_name,
                    **g,
                    "last": last,
                    "years": year_stats,
                    "min_other": min(other) if other else 0.0,
                    "avg_other": sum(other) / len(other) if other else 0.0,
                }
                hits.append(row)
                if pack_hits <= 8:
                    print(
                        f"  HIT last={last['ret']:+.0f}% n={last['n']} WR={last['wr']:.0f}% "
                        f"DD={last['dd']:.0f}% PF={last['pf']:.2f}  "
                        f"risk={g['risk_pct']:g} hold={g['max_hold_days']} cd={g['cooldown_days']} "
                        f"pos={g['max_pos_pct']:g} exit={g['cup_exit_mode']}:{g['target_rr']} "
                        f"new={g['max_new_per_day']}  "
                        f"min_other={row['min_other']:+.0f}% avg_other={row['avg_other']:+.0f}%",
                        flush=True,
                    )
        print(
            f"  grid={n_grid} hits={pack_hits} best_last={best_last['ret']:+.1f}% "
            f"n={best_last['n']} risk={best_last['risk_pct']} hold={best_last['max_hold_days']} "
            f"exit={best_last['cup_exit_mode']}:{best_last['target_rr']}",
            flush=True,
        )

    print("\n========== BEST ≥100% last-1y (sorted) ==========", flush=True)
    hits.sort(key=lambda r: (
        -r["last"]["ret"],
        -r["min_other"],
        r["last"]["dd"],
    ))
    for row in hits[:15]:
        y = row["years"]
        print(
            f"{row['pack']:16s} last={row['last']['ret']:+7.1f}% n={row['last']['n']:3d} "
            f"WR={row['last']['wr']:5.1f}% DD={row['last']['dd']:5.1f}% PF={row['last']['pf']:.2f} "
            f"| 2023={y['y2023']['ret']:+6.1f} 2024={y['y2024']['ret']:+6.1f} "
            f"2025={y['y2025']['ret']:+6.1f} 23-24={y['y2023_24']['ret']:+6.1f} "
            f"24-25={y['y2024_25']['ret']:+6.1f} "
            f"| risk={row['risk_pct']:g} hold={row['max_hold_days']} cd={row['cooldown_days']} "
            f"pos={row['max_pos_pct']:g} {row['cup_exit_mode']}:{row['target_rr']} "
            f"new={row['max_new_per_day']}",
            flush=True,
        )
    if not hits:
        print("No pack reached +100% in last 1y. See best_last lines above.", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
