"""Validate Supertrend pullback winners: position size, costs, and 25% cap."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from scripts.supertrend_swing_search import (
    CAPITAL,
    FILTER_PACKS,
    START,
    END,
    StockPack,
    _preload_frames,
    collect_daily_signals,
    simulate,
    supertrend_np,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def simulate_capped(
    packs, calendar, signals, *, capital, risk_pct, need_flags, stop_mode,
    exit_mode, target_rr, max_hold, st_key, max_pos_pct=0.25, cost_pct=0.0,
):
    """Same as simulate but cap notional at max_pos_pct of equity and optional costs."""
    from collections import defaultdict
    import numpy as np

    by_day = defaultdict(list)
    for sig in signals:
        if need_flags and (sig["flags"] & need_flags) != need_flags:
            continue
        by_day[sig["entry_ts"]].append(sig)

    cash = float(capital)
    opens = {}
    last_exit = {}
    trades = []
    equity_curve = [capital]
    pos_fracs = []
    stop_pcts = []

    def _stop_price(sig, entry):
        raw = sig["stop_st"] if stop_mode == "st" else entry - 1.5 * sig["atr"]
        if raw <= 0 or raw >= entry:
            atr = sig["atr"]
            raw = entry - (1.5 * atr if atr > 0 else entry * 0.04)
        return float(raw) if 0 < raw < entry else 0.0

    for ts in calendar:
        closed = []
        for sym, pos in list(opens.items()):
            s = packs.get(sym)
            if s is None:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            pos["hold"] += 1
            high, low, close = s.h[i], s.l[i], s.c[i]
            if exit_mode in ("st_trail", "st_trail_rr") and st_key is not None:
                st_line = s.st[st_key][i]
                if not np.isnan(st_line) and st_line > pos["stop"] and st_line < close:
                    pos["stop"] = float(st_line)
            exit_p = reason = None
            if low <= pos["stop"]:
                exit_p, reason = pos["stop"], "stop_loss"
            elif pos["target"] and high >= pos["target"]:
                exit_p, reason = pos["target"], "target"
            elif pos["hold"] >= max_hold:
                exit_p, reason = close, "time_exit"
            elif exit_mode in ("st_flip", "st_trail") and st_key is not None:
                if s.st_dir[st_key][i] <= 0:
                    exit_p, reason = close, "st_flip"
            if exit_p is None:
                continue
            pnl = (exit_p - pos["entry"]) * pos["qty"]
            if cost_pct:
                pnl -= (pos["entry"] + exit_p) * pos["qty"] * cost_pct / 100.0
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "entry": pos["entry_ts"], "exit": ts, "pnl": pnl,
                "reason": reason, "hold": pos["hold"], "entry_px": pos["entry"],
                "exit_px": exit_p, "qty": pos["qty"], "notional": pos["notional"],
                "stop_pct": pos["stop_pct"], "pos_frac": pos["pos_frac"],
            })
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = sorted(by_day.get(ts, []), key=lambda x: x["score"], reverse=True)
        taken = 0
        for sig in day_sigs:
            if taken >= 5:
                break
            sym = sig["symbol"]
            if sym in opens:
                continue
            s = packs.get(sym)
            if s is None:
                continue
            prev = last_exit.get(sym)
            if prev is not None and (ts - prev).days < 10:
                continue
            i = s.loc.get(ts)
            if i is None:
                continue
            entry = float(s.o[i])
            if entry <= 0:
                continue
            stop = _stop_price(sig, entry)
            if stop <= 0 or stop >= entry:
                continue
            risk = entry - stop
            if risk / entry > 0.12:
                continue
            equity_now = cash + sum(p["notional"] for p in opens.values())
            if cash <= 0 or equity_now <= 0:
                continue
            ps = calculate_position_size(equity_now, risk_pct, entry, stop)
            max_notional = equity_now * max_pos_pct
            qty_cap = int(min(cash, max_notional) // entry) if entry else 0
            qty = min(int(ps.quantity), qty_cap)
            if qty <= 0:
                continue
            notional = qty * entry
            cash -= notional
            frac = notional / equity_now
            spct = risk / entry * 100
            pos_fracs.append(frac)
            stop_pcts.append(spct)
            target = 0.0
            if exit_mode in ("rr", "st_flip_rr", "st_trail_rr") and target_rr > 0:
                target = round(entry + risk * target_rr, 2)
            opens[sym] = {
                "entry": entry, "stop": stop, "target": target, "qty": qty,
                "notional": notional, "hold": 0, "entry_ts": ts,
                "stop_pct": spct, "pos_frac": frac,
            }
            taken += 1
            if s.l[i] <= stop:
                pnl = (stop - entry) * qty
                if cost_pct:
                    pnl -= (entry + stop) * qty * cost_pct / 100.0
                cash += notional + pnl
                trades.append({
                    "symbol": sym, "entry": ts, "exit": ts, "pnl": pnl,
                    "reason": "stop_loss", "hold": 0, "entry_px": entry,
                    "exit_px": stop, "qty": qty, "notional": notional,
                    "stop_pct": spct, "pos_frac": frac,
                })
                last_exit[sym] = ts
                opens.pop(sym, None)

        equity_curve.append(cash + sum(p["notional"] for p in opens.values()))

    if opens:
        last_ts = calendar[-1]
        for sym, pos in list(opens.items()):
            s = packs[sym]
            i = s.loc.get(last_ts, len(s.c) - 1)
            close = float(s.c[i])
            pnl = (close - pos["entry"]) * pos["qty"]
            if cost_pct:
                pnl -= (pos["entry"] + close) * pos["qty"] * cost_pct / 100.0
            cash += pos["notional"] + pnl
            trades.append({
                "symbol": sym, "entry": pos["entry_ts"], "exit": last_ts, "pnl": pnl,
                "reason": "eod_force", "hold": pos["hold"], "entry_px": pos["entry"],
                "exit_px": close, "qty": pos["qty"], "notional": pos["notional"],
                "stop_pct": pos["stop_pct"], "pos_frac": pos["pos_frac"],
            })

    final = cash
    ret = (final - capital) / capital * 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    wr = len(wins) / len(trades) * 100 if trades else 0
    pf = gp / gl if trades else 0
    peak, max_dd = capital, 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)
    import numpy as np
    return {
        "ret": round(ret, 2), "wr": round(wr, 2), "pf": round(pf, 2),
        "dd": round(max_dd, 2), "n": len(trades), "wins": len(wins),
        "final": round(final, 2),
        "avg_pos_frac": round(float(np.mean(pos_fracs)) * 100, 1) if pos_fracs else 0,
        "med_pos_frac": round(float(np.median(pos_fracs)) * 100, 1) if pos_fracs else 0,
        "max_pos_frac": round(float(np.max(pos_fracs)) * 100, 1) if pos_fracs else 0,
        "avg_stop_pct": round(float(np.mean(stop_pcts)), 2) if stop_pcts else 0,
        "med_stop_pct": round(float(np.median(stop_pcts)), 2) if stop_pcts else 0,
        "trades": trades,
    }


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    st_params = [(7, 3.0), (10, 2.0), (10, 3.0), (14, 2.0), (14, 3.0), (21, 3.0)]
    packs = {sym: StockPack(sym, df, st_params) for sym, df in frames.items()}
    nifty = StockPack("NIFTY50", nifty_df, st_params)
    st_ts, et_ts = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st_ts) & (df.index <= et_ts)].tolist()
    })

    # sanity: GVT&D around Mar-Jun 2026
    g = frames.get("GVT&D")
    if g is not None:
        print("=== GVT&D sample ===")
        sl = g.loc["2026-03-20":"2026-06-25"]
        if not sl.empty:
            print(f"  {sl.index[0].date()} O={sl.iloc[0]['open']:.1f} C={sl.iloc[0]['close']:.1f}")
            print(f"  {sl.index[-1].date()} O={sl.iloc[-1]['open']:.1f} C={sl.iloc[-1]['close']:.1f}")
            print(f"  move {sl.iloc[-1]['close']/sl.iloc[0]['open']-1:+.1%}")

    filters = dict(FILTER_PACKS)
    configs = [
        ("ST10,3 pullback ADX20  risk3 trail", (10, 3.0), "pullback", "adx20", 3.0, "st", "st_trail", 60),
        ("ST10,3 pullback ADX20  risk5 flip", (10, 3.0), "pullback", "adx20", 5.0, "st", "st_flip", 60),
        ("ST14,3 pullback quality risk3 trail", (14, 3.0), "pullback", "quality", 3.0, "st", "st_trail", 60),
        ("ST14,3 pullback quality risk2 trail", (14, 3.0), "pullback", "quality", 2.0, "st", "st_trail", 60),
        ("ST14,2 pullback RSI45-70 risk5 flip", (14, 2.0), "pullback", "rsi_healthy", 5.0, "st", "st_flip", 60),
        ("ST21,3 pullback >EMA200 risk2 trail", (21, 3.0), "pullback", "above_200", 2.0, "st", "st_trail", 60),
        ("ST14,3 pullback trend_rsi risk3 atr15", (14, 3.0), "pullback", "trend_rsi", 3.0, "atr15", "st_trail", 60),
        ("ST10,3 flip trend+mkt risk3 trail", (10, 3.0), "flip", "trend_mkt", 3.0, "st", "st_trail", 60),
        ("ST10,3 flip none risk3 trail", (10, 3.0), "flip", "none", 3.0, "st", "st_trail", 60),
    ]

    raw_cache = {}
    print("\n=== UNCAPPED vs 25% CAP vs 25%+0.1% COST ===")
    header = f"{'config':<44} {'mode':<14} {'n':>4} {'WR':>6} {'ret':>8} {'PF':>5} {'DD':>6} {'avgPos':>7} {'medStop':>8}"
    print(header)
    for name, st_key, entry, filt, risk, stop, exit_m, hold in configs:
        ck = (st_key, entry)
        if ck not in raw_cache:
            raw_cache[ck] = collect_daily_signals(packs, nifty, st_key, entry, st_ts, et_ts)
        raw = raw_cache[ck]
        need = filters[filt]
        variants = [
            ("uncapped", 1.00, 0.0),
            ("cap25", 0.25, 0.0),
            ("cap25+cost", 0.25, 0.10),
            ("cap33", 0.33, 0.0),
            ("cap50", 0.50, 0.0),
        ]
        for vname, cap, cost in variants:
            r = simulate_capped(
                packs, calendar, raw, capital=CAPITAL, risk_pct=risk,
                need_flags=need, stop_mode=stop, exit_mode=exit_m,
                target_rr=0, max_hold=hold, st_key=st_key,
                max_pos_pct=cap, cost_pct=cost,
            )
            print(
                f"{name:<44} {vname:<14} {r['n']:4d} {r['wr']:5.1f}% "
                f"{r['ret']:+7.1f}% {r['pf']:5.2f} {r['dd']:5.1f}% "
                f"{r['avg_pos_frac']:6.1f}% {r['med_stop_pct']:6.2f}%"
            )
        print()

    # Concentration of uncapped quality
    print("=== Uncapped quality trade book (top 8 by |pnl|) ===")
    raw = raw_cache[((14, 3.0), "pullback")]
    r = simulate_capped(
        packs, calendar, raw, capital=CAPITAL, risk_pct=3.0,
        need_flags=filters["quality"], stop_mode="st", exit_mode="st_trail",
        target_rr=0, max_hold=60, st_key=(14, 3.0), max_pos_pct=1.0,
    )
    for t in sorted(r["trades"], key=lambda x: abs(x["pnl"]), reverse=True)[:8]:
        print(
            f"  {t['symbol']:<12} {str(t['entry'])[:10]}→{str(t['exit'])[:10]} "
            f"px {t['entry_px']:.1f}→{t['exit_px']:.1f} qty={t['qty']} "
            f"notional={t['notional']:,.0f} pos={t['pos_frac']*100:.0f}% "
            f"stop={t['stop_pct']:.2f}% pnl={t['pnl']:+,.0f} {t['reason']}"
        )


if __name__ == "__main__":
    main()
