"""
Hunt ≥500% over last 5 years WITHOUT a double-sized stop.

Lane B problem: +8% target vs −15% stop → losers ≈ 2× winners, and 2% risk
on a 15% stop only deploys ~13% equity, so return caps around +80%.

This run:
  - stop ≤ target (never 2×)
  - close-only stop and/or breakeven after a cushion
  - size by % of equity (not risk-to-wide-stop)
  - let winners run (15–25%) and optional scale-out
Research only — not wired into the app.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from copy import copy
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from _stage_v2_high_wr_search import apply_pred, collect
from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import (
    MIN_WEEKLY_BARS,
    _OpenPos,
    _close_trade,
    _invested_total,
    _preload_frames,
    _weekly_stage_at,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.tech_filters import enrich_daily_tech
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0


def simulate(
    frames,
    weekly_by_sym,
    signals_by_day,
    calendar,
    *,
    capital,
    start_date,
    profit_pct=0.08,
    stop_pct=0.08,
    stop_on="close",  # close | low | none
    equity_pct=25.0,  # % of equity notional; 0 = use risk_pct
    risk_pct=2.0,
    be_trigger=0.0,  # move stop to entry next bar after this gain
    max_hold=65,
    cooldown=40,
    stage_exit=False,
    max_pos=99,
    scale_at=0.0,  # close this fraction at first target
    scale_frac=0.5,
    scale_final=0.20,  # second target after scale
):
    assert stop_on == "none" or stop_pct <= profit_pct + 1e-9
    cash = float(capital)
    opens = {}
    last_exit = {}
    trades = []
    peak = capital
    max_dd = 0.0
    skipped_cash = 0

    for ts in calendar:
        closed = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            pos.hold_days += 1
            stage_now = None
            weekly = weekly_by_sym.get(sym)
            if weekly is not None and not weekly.empty and stage_exit:
                week_mask = weekly.index[
                    (weekly.index > (pos.last_week_check or pos.entry_date))
                    & (weekly.index <= ts)
                ]
                if len(week_mask):
                    pos.last_week_check = week_mask[-1]
                    w_idx = weekly.index.get_loc(pos.last_week_check)
                    if isinstance(w_idx, slice):
                        w_idx = w_idx.stop - 1
                    stage_now, _ = _weekly_stage_at(weekly, int(w_idx))

            exit_price = None
            exit_reason = ""

            # Apply pending BE / trail from prior bar (no same-bar look-ahead).
            if getattr(pos, "_pending_stop", None) is not None:
                if pos._pending_stop > pos.stop:
                    pos.stop = pos._pending_stop
                pos._pending_stop = None

            hit_stop = False
            if stop_on == "low" and low <= pos.stop:
                hit_stop = True
                exit_price, exit_reason = pos.stop, "stop_loss"
            elif stop_on == "close" and close <= pos.stop:
                hit_stop = True
                exit_price, exit_reason = close, "stop_close"

            if not hit_stop and high >= pos.target:
                # Optional scale-out at first target
                if scale_at > 0 and not getattr(pos, "scaled", False) and pos.qty >= 2:
                    take = max(1, int(pos.qty * scale_frac))
                    take = min(take, pos.qty - 1)
                    part = copy(pos)
                    part.qty = take
                    part.notional = take * pos.entry_price
                    t = _close_trade(
                        part, exit_price=pos.target, exit_ts=ts,
                        exit_reason="scale_out", days_held=pos.hold_days,
                    )
                    cash += part.notional + t.pnl
                    trades.append(t)
                    pos.qty -= take
                    pos.notional = pos.qty * pos.entry_price
                    pos.scaled = True
                    pos.target = round(pos.entry_price * (1.0 + scale_final), 2)
                    pos._pending_stop = pos.entry_price  # BE next bar
                    continue
                exit_price, exit_reason = pos.target, "target"
            elif not hit_stop and pos.hold_days >= max_hold:
                exit_price, exit_reason = close, "time_exit"
            elif not hit_stop and stage_exit and stage_now == 4:
                exit_price, exit_reason = close, "stage_exit"

            if exit_price is None:
                if be_trigger > 0 and high >= pos.entry_price * (1.0 + be_trigger):
                    be = pos.entry_price
                    if be > pos.stop:
                        pos._pending_stop = be
                continue

            t = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            cash += pos.notional + t.pnl
            trades.append(t)
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = list(signals_by_day.get(ts, []))
        if day_sigs and len(opens) < max_pos:
            day_sigs.sort(
                key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
                reverse=True,
            )
            for sig in day_sigs:
                if len(opens) >= max_pos:
                    break
                sym = sig["symbol"]
                if sym in opens or sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if cooldown > 0 and prev_x is not None and (ts - prev_x).days < cooldown:
                    continue
                row = frames[sym].loc[ts]
                entry = float(row["open"])
                high, low = float(row["high"]), float(row["low"])
                if entry <= 0:
                    continue
                stop = round(entry * (1.0 - stop_pct), 2) if stop_on != "none" else entry * 0.50
                target = round(entry * (1.0 + profit_pct), 2)
                if stop_on != "none" and entry - stop <= 0:
                    continue
                equity_now = cash + _invested_total(opens)
                if cash <= 0 or equity_now <= 0:
                    skipped_cash += 1
                    continue
                if equity_pct > 0:
                    alloc = equity_now * (equity_pct / 100.0)
                    qty = int(min(alloc, cash) // entry)
                else:
                    size_stop = stop if stop_on != "none" else entry * 0.92
                    qty = int(calculate_position_size(equity_now, risk_pct, entry, size_stop).quantity)
                    qty = min(qty, int(cash // entry))
                if qty <= 0:
                    skipped_cash += 1
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    skipped_cash += 1
                    continue
                cash -= notional
                inv = _invested_total(opens) + notional
                pos = _OpenPos(
                    symbol=sym, entry_date=ts, signal_date=sig["signal_date"],
                    entry_price=entry, stop=stop if stop_on != "none" else entry * 0.01,
                    target=target, qty=qty, notional=notional,
                    quality_score=int(sig.get("quality_score") or 0),
                    rs_rating=float(sig.get("rs_rating") or 0), weekly_stage=2,
                    capital_invested=notional, cash_available=cash, total_invested=inv,
                    parallel_open=len(opens) + 1, equity_at_entry=cash + inv,
                )
                pos._pending_stop = None
                pos.scaled = False
                opens[sym] = pos

                if high >= target:
                    t = _close_trade(pos, exit_price=target, exit_ts=ts, exit_reason="target", days_held=0)
                    cash += pos.notional + t.pnl
                    trades.append(t)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif stop_on == "low" and low <= pos.stop:
                    t = _close_trade(pos, exit_price=pos.stop, exit_ts=ts, exit_reason="stop_loss", days_held=0)
                    cash += pos.notional + t.pnl
                    trades.append(t)
                    last_exit[sym] = ts
                    opens.pop(sym, None)

        eq = cash + _invested_total(opens)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100 if peak else 0)

    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None:
                continue
            hist = df.loc[df.index <= last_ts]
            if hist.empty:
                continue
            t = _close_trade(
                pos, exit_price=float(hist.iloc[-1]["close"]), exit_ts=hist.index[-1],
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            cash += pos.notional + t.pnl
            trades.append(t)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    stops = sum(1 for t in trades if str(t.exit_reason).startswith("stop"))
    avg_w = (gp / len(wins)) if wins else 0
    avg_l = (sum(t.pnl for t in losses) / len(losses)) if losses else 0
    return {
        "n": len(trades),
        "wr": round(wr, 1),
        "pf": round(gp / gl, 2),
        "ret": round(pnl / capital * 100, 2),
        "dd": round(max_dd, 2),
        "stops": stops,
        "avg_w": round(avg_w, 0),
        "avg_l": round(avg_l, 0),
        "wl": round(abs(avg_w / avg_l), 2) if avg_l else 0,
        "skip": skipped_cash,
        "final": round(capital + pnl, 2),
        "trades": trades,
    }


def main():
    end = date.today()
    start = end - timedelta(days=int(5 * 365))
    years = (end - start).days / 365.25
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"500% hunt | {start} → {end} | {years:.2f}y | stop ≤ target")
    print("Loading...")
    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nw = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    st, et = pd.Timestamp(start), pd.Timestamp(end)
    weekly_by_sym, tech_by_sym = {}, {}
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < MIN_WEEKLY_BARS + 2:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(d)
    print("Collecting Stage 2 signals...")
    sigs = collect(frames, weekly_by_sym, tech_by_sym, nw, st, et)
    print(f"Raw signals: {len(sigs)}")
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })

    packs = {
        "LaneB": lambda s: s["daily_stage"] in (1, 2) and s["rs_rating"] >= 70 and s["not_extended"],
        "RS70": lambda s: s["daily_stage"] in (1, 2) and s["rs_rating"] >= 70,
        "dmtf": lambda s: s["daily_stage"] in (1, 2),
        "Q60": lambda s: s["daily_stage"] in (1, 2) and s["quality_score"] >= 60,
        "LaneB+mkt": lambda s: (
            s["daily_stage"] in (1, 2) and s["rs_rating"] >= 70
            and s["not_extended"] and s["mkt_ok"]
        ),
        "dmtf+mkt": lambda s: s["daily_stage"] in (1, 2) and s["mkt_ok"],
    }

    variants = [
        # Equal 8/8 — kill the double stop, size like a real swing
        ("LaneB 8/8 close eq25%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=25)),
        ("LaneB 8/8 close eq40%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("LaneB 8/8 close eq50%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=50)),
        ("LaneB 8/8 wick eq40%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="low", equity_pct=40)),
        ("LaneB 8/8 close eq40% BE@4%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=40, be_trigger=0.04)),
        ("LaneB 8/8 close eq40% cd0", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=40, cooldown=0)),
        # Winners > losers (8% stop, 15–25% target)
        ("LaneB 15/8 close eq25%", "LaneB", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=25)),
        ("LaneB 15/8 close eq40%", "LaneB", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("LaneB 15/8 close eq40% BE@4%", "LaneB", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=40, be_trigger=0.04)),
        ("LaneB 20/8 close eq40%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("LaneB 20/8 close eq50%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=50)),
        ("LaneB 25/8 close eq40%", "LaneB", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("LaneB 20/8 close eq40% hold90", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40, max_hold=90)),
        ("LaneB 20/8 close eq40% cd0 hold90", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40, cooldown=0, max_hold=90)),
        # Scale-out: bank 50% at +8%, rest to +20%, stop 8% then BE
        ("LaneB scale 8→20 8sl eq40%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=40, scale_at=0.08, scale_frac=0.5, scale_final=0.20)),
        ("LaneB scale 8→25 8sl eq40%", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=40, scale_at=0.08, scale_frac=0.5, scale_final=0.25)),
        # More signals
        ("RS70 15/8 close eq40%", "RS70", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("RS70 20/8 close eq40%", "RS70", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("dmtf 15/8 close eq40%", "dmtf", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("dmtf 20/8 close eq40%", "dmtf", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("dmtf 20/8 close eq50% cd0", "dmtf", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=50, cooldown=0)),
        ("dmtf 20/8 close eq40% hold90", "dmtf", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40, max_hold=90)),
        ("Q60 20/8 close eq40%", "Q60", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40)),
        # Market filter (skip Stage 3/4 Nifty — Jan 2025 defence)
        ("LaneB mkt 20/8 eq40%", "LaneB+mkt", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=40)),
        ("dmtf mkt 20/8 eq50%", "dmtf+mkt", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=50)),
        ("dmtf mkt 20/8 eq50% cd0 hold90", "dmtf+mkt", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=50, cooldown=0, max_hold=90)),
        # Risk-% sizing with EQUAL 8% stop (risk 6% → ~75% equity — concentrated)
        ("LaneB 20/8 risk4%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=4)),
        ("LaneB 20/8 risk6%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=6)),
        ("dmtf 20/8 risk6% cd0", "dmtf", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=6, cooldown=0)),
        ("dmtf 25/8 risk8% cd0 hold90", "dmtf", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, cooldown=0, max_hold=90)),
        ("RS70 25/8 risk8% eq0 hold90", "RS70", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, cooldown=0, max_hold=90)),
        ("dmtf mkt 25/8 risk8% cd0", "dmtf+mkt", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, cooldown=0, max_hold=90)),
        # Fewer concurrent names, fatter each
        ("LaneB 20/8 eq70% max3", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=70, max_pos=3)),
        ("dmtf 20/8 eq60% max4 cd0", "dmtf", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=60, max_pos=4, cooldown=0)),
        ("dmtf 15/8 eq50% BE@4% cd0", "dmtf", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=50, be_trigger=0.04, cooldown=0)),
        # Wave 2: bigger winners + longer hold + more size (still stop ≤ target)
        ("LaneB 30/8 close eq50% hold130", "LaneB", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=50, max_hold=130)),
        ("LaneB 40/8 close eq50% hold130", "LaneB", dict(profit_pct=0.40, stop_pct=0.08, stop_on="close", equity_pct=50, max_hold=130)),
        ("LaneB 30/8 close eq70% max3 hold130", "LaneB", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=70, max_pos=3, max_hold=130)),
        ("LaneB 40/8 risk8% hold130", "LaneB", dict(profit_pct=0.40, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, max_hold=130)),
        ("LaneB 30/8 risk10% hold130", "LaneB", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=10, max_hold=130)),
        ("LaneB 20/8 risk10%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=10)),
        ("LaneB 20/8 risk12%", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=12)),
        ("Q60 30/8 eq50% hold130", "Q60", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=50, max_hold=130)),
        ("Q60 40/8 eq50% hold200", "Q60", dict(profit_pct=0.40, stop_pct=0.08, stop_on="close", equity_pct=50, max_hold=200)),
        ("RS70 30/8 eq50% hold130", "RS70", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=50, max_hold=130)),
        ("RS70 40/8 risk8% hold130", "RS70", dict(profit_pct=0.40, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, max_hold=130)),
        ("LaneB scale 8→40 eq50% hold130", "LaneB", dict(profit_pct=0.08, stop_pct=0.08, stop_on="close", equity_pct=50, scale_at=0.08, scale_frac=0.4, scale_final=0.40, max_hold=130)),
        ("LaneB 25/8 eq50% BE@5% hold90", "LaneB", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=50, be_trigger=0.05, max_hold=90)),
        ("dmtf 30/8 risk10% hold130", "dmtf", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=10, max_hold=130)),
        ("dmtf 40/8 risk10% cd0 hold200", "dmtf", dict(profit_pct=0.40, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=10, cooldown=0, max_hold=200)),
        ("LaneB+mkt 30/8 eq70% hold130", "LaneB+mkt", dict(profit_pct=0.30, stop_pct=0.08, stop_on="close", equity_pct=70, max_hold=130, max_pos=3)),
        ("Q60 25/8 risk8% hold90", "Q60", dict(profit_pct=0.25, stop_pct=0.08, stop_on="close", equity_pct=0, risk_pct=8, max_hold=90)),
        ("LaneB 15/8 eq80% max2", "LaneB", dict(profit_pct=0.15, stop_pct=0.08, stop_on="close", equity_pct=80, max_pos=2)),
        ("LaneB 20/8 eq80% max2 hold90", "LaneB", dict(profit_pct=0.20, stop_pct=0.08, stop_on="close", equity_pct=80, max_pos=2, max_hold=90)),
    ]

    rows = []
    print("")
    hdr = f"{'variant':<42} {'n':>3} {'WR':>6} {'ret':>8} {'PF':>5} {'DD':>6} {'SL':>3} {'W/L':>5}"
    print(hdr)
    print("-" * 90)
    for name, pack, kw in variants:
        bd = apply_pred(sigs, packs[pack])
        r = simulate(
            frames, weekly_by_sym, bd, calendar,
            capital=CAPITAL, start_date=start, **kw,
        )
        rows.append((name, r))
        flag = ""
        if r["ret"] >= 500 and r["n"] >= 20:
            flag = "  ***500***"
        elif r["ret"] >= 300:
            flag = "  *300*"
        elif r["ret"] >= 200:
            flag = "  +200"
        print(
            f"{name:<42} {r['n']:3d} {r['wr']:5.1f}% {r['ret']:+7.1f}% "
            f"{r['pf']:5.2f} {r['dd']:5.1f}% {r['stops']:3d} {r['wl']:5.2f}{flag}"
        )

    print("")
    print("=== Ranked by return ===")
    for name, r in sorted(rows, key=lambda x: x[1]["ret"], reverse=True)[:12]:
        cagr = (r["final"] / CAPITAL) ** (1.0 / years) * 100 - 100 if r["final"] > 0 else 0
        print(
            f"  {name:<42} ret={r['ret']:+7.1f}% CAGR={cagr:5.1f}% "
            f"WR={r['wr']:5.1f}% DD={r['dd']:5.1f}% n={r['n']} SL={r['stops']} "
            f"avgW {r['avg_w']:,.0f} avgL {r['avg_l']:,.0f}"
        )


if __name__ == "__main__":
    main()
