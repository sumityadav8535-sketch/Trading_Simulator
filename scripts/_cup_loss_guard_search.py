"""Search loss-guards that cut clustered down-months without giving up 2023."""
from __future__ import annotations

import os
import sys
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

from _cup_hunt_100 import collect_signals
from stage_analysis_v2.services.backtester import (
    _OpenPos,
    _close_trade,
    _equity,
    _invested_total,
    _preload_frames,
    maybe_raise_stop,
    stop_exit_reason,
)
from stage_analysis_v2.services.cup_breakout import (
    CUP_EXIT_EMA20,
    CupParams,
    CupSetup,
    _CupPack,
    _target_price,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
START = date(2021, 8, 25)
END = date(2026, 8, 24)

BASE = CupParams(
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
    require_trend_stack=True,
    vol_mult=1.1,
    rsi_min=40.0,
    rsi_max=85.0,
    max_new_per_day=10,
    cup_exit_mode=CUP_EXIT_EMA20,
    skip_entry_gap_down_pct=2.0,
)


def simulate(packs, calendar, signals_by_day, nifty, params: CupParams, *,
             risk_pct=10.0, max_hold_days=40, cooldown_days=0, max_pos_pct=100.0,
             start_ts=None, end_ts=None):
    cash = float(CAPITAL)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades = []
    peak_eq = CAPITAL
    month_key = ""
    month_start_eq = CAPITAL
    month_pnl = 0.0
    consec_losses = 0
    cooloff_until = None
    monthly = {}

    def on_close(trade, ts):
        nonlocal cash, month_pnl, consec_losses, cooloff_until
        cash += trade.pos_notional + trade.pnl if False else 0
        # handled by caller

    nifty_close = nifty["close"] if nifty is not None else None
    nifty_open = nifty["open"] if nifty is not None else None
    nifty_prev = nifty_close.shift(1) if nifty_close is not None else None
    nifty_ema = None
    if params.nifty_ema_period and nifty_close is not None:
        nifty_ema = nifty_close.ewm(span=int(params.nifty_ema_period), adjust=False).mean()

    for ts in calendar:
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        mk = f"{ts.year:04d}-{ts.month:02d}"
        if mk != month_key:
            if month_key:
                monthly[month_key] = month_pnl
            month_key = mk
            month_start_eq = _equity(cash, opens)
            month_pnl = 0.0

        closed_today = []
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
            risk_ps = pos.entry_price - (pos.entry_stop if pos.entry_stop else pos.stop)
            armed = (
                params.trail_arm_r <= 0 or risk_ps <= 0
                or (close - pos.entry_price) >= params.trail_arm_r * risk_ps
            )
            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, stop_exit_reason(pos)
            elif pos.target > 0 and high >= pos.target:
                exit_price, exit_reason = pos.target, "target"
            elif pos.hold_days >= max_hold_days:
                exit_price, exit_reason = close, "time_exit"
            elif params.cup_exit_mode == CUP_EXIT_EMA20 and armed and e20 > 0 and close < e20:
                exit_price, exit_reason = close, "ema20_exit"
            if exit_price is None:
                if params.cup_exit_mode == CUP_EXIT_EMA20 and armed and e20 > 0:
                    maybe_raise_stop(pos, e20, close)
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)
            month_pnl += trade.pnl
            if trade.pnl <= 0:
                consec_losses += 1
                if params.loss_streak > 0 and consec_losses >= params.loss_streak and params.loss_streak_cooloff_days > 0:
                    cooloff_until = ts + pd.Timedelta(days=int(params.loss_streak_cooloff_days))
            else:
                consec_losses = 0
        for sym in closed_today:
            opens.pop(sym, None)

        peak_eq = max(peak_eq, _equity(cash, opens))

        day_sigs = signals_by_day.get(ts)
        if not day_sigs:
            continue
        taken = 0
        for sig in sorted(day_sigs, key=lambda x: x["score"], reverse=True):
            if taken >= params.max_new_per_day:
                break
            if params.max_open > 0 and len(opens) >= params.max_open:
                break
            if cooloff_until is not None and ts < cooloff_until:
                break
            eq_now = _equity(cash, opens)
            if params.halt_dd_pct > 0 and peak_eq > 0:
                if (peak_eq - eq_now) / peak_eq * 100.0 >= params.halt_dd_pct:
                    break
            if params.max_month_loss_pct > 0 and month_start_eq > 0:
                if month_pnl <= -(params.max_month_loss_pct / 100.0) * month_start_eq:
                    break
            if params.nifty_ema_period and nifty_ema is not None and nifty_close is not None:
                sig_ts = sig.get("sig_ts", ts)
                if sig_ts in nifty_close.index and sig_ts in nifty_ema.index:
                    nc, ne = nifty_close.loc[sig_ts], nifty_ema.loc[sig_ts]
                    if pd.notna(nc) and pd.notna(ne) and float(nc) <= float(ne):
                        continue
            if params.nifty_gap_down_pct > 0 and nifty_open is not None and nifty_prev is not None:
                if ts in nifty_open.index and ts in nifty_prev.index:
                    no, npv = nifty_open.loc[ts], nifty_prev.loc[ts]
                    if pd.notna(no) and pd.notna(npv) and float(npv) > 0:
                        if (float(npv) - float(no)) / float(npv) * 100.0 > params.nifty_gap_down_pct:
                            continue
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
            if params.skip_entry_gap_down_pct > 0 and i > 0:
                prev_c = float(s.c[i - 1])
                if prev_c > 0 and (prev_c - entry) / prev_c * 100.0 > params.skip_entry_gap_down_pct:
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
                continue
            ps = calculate_position_size(equity_now, risk_pct, entry, stop)
            max_notional = equity_now * (max_pos_pct / 100.0)
            qty_cash = int(cash // entry) if entry else 0
            qty_cap = int(max_notional // entry) if entry else 0
            qty = min(int(ps.quantity), qty_cash, qty_cap)
            if qty <= 0:
                continue
            notional = qty * entry
            if notional > cash + 1e-6:
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
                trades.append(trade)
                last_exit[sym] = ts
                opens.pop(sym, None)
                month_pnl += trade.pnl
                if trade.pnl <= 0:
                    consec_losses += 1
                    if params.loss_streak > 0 and consec_losses >= params.loss_streak and params.loss_streak_cooloff_days > 0:
                        cooloff_until = ts + pd.Timedelta(days=int(params.loss_streak_cooloff_days))
                else:
                    consec_losses = 0

        peak_eq = max(peak_eq, _equity(cash, opens))

    if month_key:
        monthly[month_key] = month_pnl
    if opens:
        last_ts = calendar[-1] if end_ts is None else min(end_ts, calendar[-1])
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(last_ts, len(s.c) - 1)
            trade = _close_trade(
                pos, exit_price=float(s.c[i]), exit_ts=last_ts,
                exit_reason="eod_force", days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)
            mk = str(last_ts)[:7]
            monthly[mk] = monthly.get(mk, 0.0) + trade.pnl

    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 1e-9
    ret = (cash - CAPITAL) / CAPITAL * 100.0
    wr = len(wins) / n * 100.0 if n else 0.0
    worst = min(monthly.values()) if monthly else 0.0
    worst_m = min(monthly, key=monthly.get) if monthly else ""
    bad = {m: v for m, v in monthly.items() if v <= -50000}
    y2023 = sum(v for m, v in monthly.items() if m.startswith("2023"))
    y2024 = sum(v for m, v in monthly.items() if m.startswith("2024"))
    y2025 = sum(v for m, v in monthly.items() if m.startswith("2025"))
    return {
        "ret": round(ret, 1),
        "n": n,
        "wr": round(wr, 1),
        "pf": round(gp / gl, 2),
        "final": round(cash, 0),
        "worst": round(worst, 0),
        "worst_m": worst_m,
        "bad": {m: round(v, 0) for m, v in sorted(bad.items())},
        "y2023": round(y2023 / CAPITAL * 100, 1),
        "y2024": round(y2024 / CAPITAL * 100, 1),
        "y2025": round(y2025 / CAPITAL * 100, 1),
        "monthly": {m: round(v, 0) for m, v in sorted(monthly.items())},
        "dd_proxy": round(min(0.0, worst / CAPITAL * 100), 1),
    }


def combos():
    out = []
    for month_cap in (0, 6, 8, 10):
        for nifty_gap in (0, 1.0, 1.5, 2.0):
            for ema in (0, 20, 50):
                for mx_open in (0, 3):
                    for streak, cool in ((0, 0), (3, 10), (3, 15)):
                        for halt in (0, 20, 25):
                            out.append(replace(
                                BASE,
                                max_month_loss_pct=float(month_cap),
                                nifty_gap_down_pct=float(nifty_gap),
                                nifty_ema_period=int(ema),
                                max_open=int(mx_open),
                                loss_streak=int(streak),
                                loss_streak_cooloff_days=int(cool),
                                halt_dd_pct=float(halt),
                            ))
    return out


def label(p: CupParams) -> str:
    parts = []
    if p.max_month_loss_pct:
        parts.append(f"mcap{p.max_month_loss_pct:g}")
    if p.nifty_gap_down_pct:
        parts.append(f"ngap{p.nifty_gap_down_pct:g}")
    if p.nifty_ema_period:
        parts.append(f"nema{p.nifty_ema_period}")
    if p.max_open:
        parts.append(f"open{p.max_open}")
    if p.loss_streak:
        parts.append(f"strk{p.loss_streak}/{p.loss_streak_cooloff_days}d")
    if p.halt_dd_pct:
        parts.append(f"dd{p.halt_dd_pct:g}")
    return "+".join(parts) or "baseline"


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print("preload…", flush=True)
    frames = _preload_frames(symbols)
    packs = {sym: _CupPack(sym, df) for sym, df in frames.items()}
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_ret60 = nifty["close"].pct_change(60) if not nifty.empty else None
    nifty_sma200 = nifty["close"].rolling(200, min_periods=200).mean() if not nifty.empty else None
    nifty_close = nifty["close"] if not nifty.empty else None
    print("collect signals…", flush=True)
    by_day, n_sig = collect_signals(packs, frames, BASE, nifty_ret60, nifty_sma200, nifty_close)
    print(f"  signals={n_sig}", flush=True)
    cal = sorted({ts for df in frames.values() for ts in df.index.tolist()
                  if pd.Timestamp(START) <= ts <= pd.Timestamp(END)})

    base = simulate(packs, cal, by_day, nifty, BASE)
    print("\n===== BASELINE 5y =====")
    print(
        f"ret={base['ret']:+.1f}% n={base['n']} WR={base['wr']}% PF={base['pf']} "
        f"final={base['final']:,.0f}  2023={base['y2023']:+.1f}% 2024={base['y2024']:+.1f}% 2025={base['y2025']:+.1f}%"
    )
    print("all months (₹ vs start cap %):")
    for m, v in base["monthly"].items():
        pct = v / CAPITAL * 100
        mark = " **LOSS**" if v <= -50000 else ""
        print(f"  {m}  {v:+10,.0f}  {pct:+6.2f}%{mark}")
    print("bad months ≤ -50k:", base["bad"])

    rows = []
    seen = set()
    for p in combos():
        key = label(p)
        if key in seen:
            continue
        seen.add(key)
        st = simulate(packs, cal, by_day, nifty, p)
        st["label"] = key
        st["params"] = p
        rows.append(st)

    # Keep 2023 strong and 5y not crushed; rank by lifting the worst month then 5y ret.
    keep = [
        r for r in rows
        if r["y2023"] >= 140.0 and r["ret"] >= base["ret"] - 80
    ]
    keep.sort(key=lambda r: (r["worst"], -r["ret"], -r["y2024"]))
    print("\n===== BEST GUARDS (2023≥140%, 5y within 80pp of baseline) =====")
    print(f"{'pack':40s} {'5y':>8} {'2023':>7} {'2024':>7} {'2025':>7} {'worst':>12} {'n':>4} WR")
    for r in keep[:25]:
        print(
            f"{r['label'][:40]:40s} {r['ret']:+7.1f}% {r['y2023']:+6.1f}% {r['y2024']:+6.1f}% "
            f"{r['y2025']:+6.1f}% {r['worst_m']} {r['worst']:+9,.0f} {r['n']:4d} {r['wr']:4.1f}%"
        )

    if keep:
        w = keep[0]
        print("\n===== WINNER months vs baseline =====")
        print(f"winner: {w['label']}")
        months = sorted(set(base["monthly"]) | set(w["monthly"]))
        for m in months:
            b = base["monthly"].get(m, 0)
            n = w["monthly"].get(m, 0)
            if b <= -30000 or n <= -30000 or abs(n - b) >= 30000:
                print(f"  {m}  base {b:+10,.0f}  new {n:+10,.0f}  Δ {n-b:+10,.0f}")


if __name__ == "__main__":
    main()
