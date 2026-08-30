"""
Hunt for 80-90% win-rate on Stage 2.0 (last 1 year), 10% take-profit research.

Ideas sourced from Weinstein, Minervini/VCP, Bulkowski, pullback-to-MA literature.
Not wired into the app.
"""
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

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import (
    COOLDOWN_DAYS,
    MIN_WEEKLY_BARS,
    _OpenPos,
    _close_trade,
    _invested_total,
    _market_favorable_at,
    _preload_frames,
    _weekly_stage_at,
)
from stage_analysis_v2.services.breakout_detector import detect_breakout
from stage_analysis_v2.services.indicators import add_daily_indicators, add_weekly_indicators
from stage_analysis_v2.services.quality_score import compute_quality_score
from stage_analysis_v2.services.relative_strength import compute_relative_strength
from stage_analysis_v2.services.stage_engine import detect_daily_stage
from stage_analysis_v2.services.tech_filters import enrich_daily_tech, snapshot_tech
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size

CAPITAL = 1_000_000.0
HOLD_DEFAULT = 65


def collect(frames, weekly_by_sym, tech_by_sym, nifty_weekly, start_ts, end_ts):
    sigs = []
    for sym, weekly in weekly_by_sym.items():
        daily = frames[sym]
        daily_ind = add_daily_indicators(daily)
        tech = tech_by_sym[sym]
        bench = nifty_weekly if not nifty_weekly.empty else weekly
        for w_idx in range(1, len(weekly)):
            week_end = weekly.index[w_idx]
            if week_end < start_ts or week_end > end_ts:
                continue
            prev_stage, _ = _weekly_stage_at(weekly, w_idx - 1)
            curr_stage, metrics = _weekly_stage_at(weekly, w_idx)
            if curr_stage != 2 or prev_stage == 2:
                continue
            w_slice = weekly.iloc[: w_idx + 1]
            d_slice = daily_ind.loc[daily_ind.index <= week_end]
            b_slice = bench.loc[bench.index <= week_end]
            d_stage = 0
            if len(d_slice) >= 190:
                try:
                    d_stage, _, _ = detect_daily_stage(d_slice)
                except ValueError:
                    pass
            rs = compute_relative_strength(w_slice, b_slice, NIFTY50_SYMBOL)
            breakout = detect_breakout(w_slice, d_slice)
            mkt_ok = _market_favorable_at(nifty_weekly, week_end) if not nifty_weekly.empty else True
            quality = compute_quality_score(
                weekly_stage=2,
                daily_stage=d_stage,
                weekly_metrics=metrics,
                breakout=breakout,
                rs=rs,
                weekly=w_slice,
                market_favorable=mkt_ok,
            )
            ma = float(metrics.get("ma") or 0.0)
            price = float(metrics.get("price") or weekly.iloc[w_idx]["close"])
            nxt = daily.index[daily.index > week_end]
            if len(nxt) == 0 or nxt[0] > end_ts:
                continue
            entry_day = nxt[0]
            snap = snapshot_tech(tech, week_end)
            atr_pct = (snap.atr / snap.close * 100) if snap.ok and snap.close else 0.0
            dist_ma = ((price - ma) / ma * 100) if ma else 99.0
            hist = daily.loc[daily.index <= week_end]
            tight = 99.0
            if len(hist) >= 20:
                hh = float(hist["high"].iloc[-20:].max())
                ll = float(hist["low"].iloc[-20:].min())
                if price:
                    tight = (hh - ll) / price * 100
            pb_entry = _pullback_entry(daily, tech, entry_day, end_ts, max_wait=15)
            sigs.append({
                "symbol": sym,
                "signal_date": week_end,
                "entry_day": entry_day,
                "pb_entry_day": pb_entry,
                "quality_score": quality.total,
                "rs_rating": rs.rating,
                "ma": ma,
                "signal_close": price,
                "daily_stage": d_stage,
                "breakout_type": breakout.breakout_type,
                "follow_through": bool(breakout.follow_through),
                "volume_ratio_bo": float(breakout.volume_ratio or 0),
                "mkt_ok": mkt_ok,
                "tech_ok": snap.ok,
                "ema_stack_bull": snap.ema_stack_bull if snap.ok else False,
                "ema_trend_bull": snap.ema_trend_bull if snap.ok else False,
                "supertrend_bull": snap.supertrend_bull if snap.ok else False,
                "bb_above_mid": snap.bb_above_mid if snap.ok else False,
                "rsi_healthy": snap.rsi_healthy if snap.ok else False,
                "rsi_not_overbought": snap.rsi_not_overbought if snap.ok else False,
                "volume_ok": snap.volume_ok if snap.ok else False,
                "not_extended": snap.not_extended if snap.ok else False,
                "mild_extended": snap.mild_extended if snap.ok else False,
                "ext_pct": snap.price_vs_ema20_pct if snap.ok else 99.0,
                "atr_pct": atr_pct,
                "dist_ma": dist_ma,
                "tight_20d": tight,
                "rsi": snap.rsi if snap.ok else 0.0,
                "vol_ratio": snap.vol_ratio if snap.ok else 0.0,
            })
    return sigs


def _pullback_entry(daily, tech, first_entry, end_ts, max_wait=15):
    """Weinstein/pullback: wait for close within 2.5% of EMA20, still above EMA50."""
    future = daily.index[(daily.index >= first_entry) & (daily.index <= end_ts)][: max_wait]
    for ts in future:
        hist = tech.loc[tech.index <= ts] if tech is not None and not tech.empty else None
        if hist is None or hist.empty:
            continue
        row = hist.iloc[-1]
        e20 = row.get("ema_20")
        e50 = row.get("ema_50")
        if pd.isna(e20) or pd.isna(e50) or float(e20) <= 0:
            continue
        close = float(daily.loc[ts, "close"])
        if close > float(e50) and abs(close - float(e20)) / float(e20) <= 0.025:
            nxt = daily.index[daily.index > ts]
            if len(nxt) and nxt[0] <= end_ts:
                return nxt[0]
    return None


def apply_pred(sigs, pred, entry_key="entry_day"):
    by_day = defaultdict(list)
    for s in sigs:
        if not pred(s):
            continue
        day = s.get(entry_key) if entry_key != "entry_day" else s["entry_day"]
        if day is None:
            continue
        row = dict(s)
        row["entry_day"] = day
        by_day[day].append(row)
    return by_day


def simulate(
    frames,
    weekly_by_sym,
    signals_by_day,
    calendar,
    *,
    capital,
    risk_pct,
    start_date,
    profit_pct=0.10,
    stop_mode="wick_ma",  # wick_ma | close_ma | pct | none
    stop_ma_mult=0.95,
    stop_pct=0.12,
    stage_exit=True,
    max_hold=HOLD_DEFAULT,
    stop_grace_days=0,
):
    cash = float(capital)
    opens = {}
    last_exit = {}
    trades = []
    peak = capital
    max_dd = 0.0

    def _stop_for(entry, sig):
        ma = float(sig.get("ma") or 0)
        if stop_mode == "none":
            return entry * 0.50  # unreachable dummy; sizing uses 8% below
        if stop_mode == "pct":
            return round(entry * (1.0 - stop_pct), 2)
        if ma > 0:
            return round(ma * stop_ma_mult, 2)
        return round(entry * 0.93, 2)

    def _size_stop(entry, raw_stop):
        if stop_mode == "none":
            return entry * 0.92
        return raw_stop

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
                    if stop_mode in ("wick_ma", "close_ma") and metrics:
                        ma_now = metrics.get("ma")
                        if ma_now and float(ma_now) > 0:
                            # keep original stop (no trail) — only used for weekly close-below-MA
                            pos._ma_now = float(ma_now)

            exit_price = None
            exit_reason = ""
            armed = pos.hold_days >= stop_grace_days
            if high >= pos.target:
                exit_price, exit_reason = pos.target, "target"
            elif armed and stop_mode == "wick_ma" and low <= pos.stop:
                exit_price, exit_reason = pos.stop, "stop_loss"
            elif armed and stop_mode == "close_ma" and close <= pos.stop:
                exit_price, exit_reason = close, "stop_close"
            elif armed and stop_mode == "pct" and low <= pos.stop:
                exit_price, exit_reason = pos.stop, "stop_loss"
            elif pos.hold_days >= max_hold:
                exit_price, exit_reason = close, "time_exit"
            elif stage_exit and stage_now == 4:
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
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = list(signals_by_day.get(ts, []))
        if day_sigs:
            day_sigs.sort(
                key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
                reverse=True,
            )
            for sig in day_sigs:
                sym = sig["symbol"]
                if sym in opens or sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if prev_x is not None and (ts - prev_x).days < COOLDOWN_DAYS:
                    continue
                row = frames[sym].loc[ts]
                entry = float(row["open"])
                high, low = float(row["high"]), float(row["low"])
                raw_stop = _stop_for(entry, sig)
                size_stop = _size_stop(entry, raw_stop)
                if entry - size_stop <= 0:
                    continue
                target = round(entry * (1.0 + profit_pct), 2)
                equity_now = cash + _invested_total(opens)
                if cash <= 0 or equity_now <= 0:
                    continue
                qty = int(calculate_position_size(equity_now, risk_pct, entry, size_stop).quantity)
                qty = min(qty, int(cash // entry) if entry > 0 else 0)
                if qty <= 0:
                    continue
                notional = qty * entry
                if notional > cash + 1e-6:
                    continue
                cash -= notional
                inv = _invested_total(opens) + notional
                stop = raw_stop if stop_mode != "none" else 0.0
                pos = _OpenPos(
                    symbol=sym, entry_date=ts, signal_date=sig["signal_date"],
                    entry_price=entry, stop=stop if stop else entry * 0.01,
                    target=target, qty=qty, notional=notional,
                    quality_score=int(sig.get("quality_score") or 0),
                    rs_rating=float(sig.get("rs_rating") or 0), weekly_stage=2,
                    capital_invested=notional, cash_available=cash, total_invested=inv,
                    parallel_open=len(opens) + 1, equity_at_entry=cash + inv,
                )
                opens[sym] = pos
                if high >= target:
                    t = _close_trade(pos, exit_price=target, exit_ts=ts, exit_reason="target", days_held=0)
                    cash += pos.notional + t.pnl
                    trades.append(t)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif stop_mode == "wick_ma" and stop_grace_days == 0 and low <= pos.stop:
                    t = _close_trade(pos, exit_price=pos.stop, exit_ts=ts, exit_reason="stop_loss", days_held=0)
                    cash += pos.notional + t.pnl
                    trades.append(t)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif stop_mode == "pct" and stop_grace_days == 0 and low <= pos.stop:
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
    hits = sum(1 for t in trades if t.exit_reason == "target")
    return {
        "n": len(trades),
        "wr": round(wr, 1),
        "pf": round(gp / gl, 2),
        "ret": round(pnl / capital * 100, 2),
        "dd": round(max_dd, 2),
        "stops": stops,
        "hits": hits,
        "avg_hold": round(sum(t.days_held for t in trades) / len(trades), 1) if trades else 0,
        "pnl": round(pnl, 2),
        "gp": round(gp, 2),
        "gl": round(-abs(sum(t.pnl for t in losses)), 2),
        "final": round(capital + pnl, 2),
        "trades": trades,
    }


def main():
    end = date.today()
    start = end - timedelta(days=365)
    risk_pct = float(getattr(StrategyConfig.get_active(), "risk_pct", 2.0) or 2.0)
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"High-WR search | {start} → {end} | {len(symbols)} symbols")
    print("Loading frames...")
    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nw = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    st, et = pd.Timestamp(start), pd.Timestamp(end)
    weekly_by_sym, tech_by_sym = {}, {}
    print("Weekly + tech...")
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < MIN_WEEKLY_BARS + 2:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(d)
    print("Collecting Stage 2 signals...")
    sigs = collect(frames, weekly_by_sym, tech_by_sym, nw, st, et)
    print(f"Raw Stage 2 transitions: {len(sigs)}")
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })

    def dmtf(s):
        return s["daily_stage"] in (1, 2)

    variants = []

    def add(name, pred, **kw):
        variants.append((name, pred, kw))

    # ── A. Stop / exit mechanics (daily_mtf, 10% TP) ──
    add("BASE 10%TP + S4 + wick 0.95MA", dmtf)
    add("Close-only stop (ignore wicks)", dmtf, stop_mode="close_ma")
    add("Wider stop 0.90×MA", dmtf, stop_ma_mult=0.90)
    add("Wider stop 0.85×MA", dmtf, stop_ma_mult=0.85)
    add("Hard 12% stop", dmtf, stop_mode="pct", stop_pct=0.12)
    add("Hard 15% stop (wide)", dmtf, stop_mode="pct", stop_pct=0.15)
    add("Hard 20% stop (Bulkowski max)", dmtf, stop_mode="pct", stop_pct=0.20)
    add("No stop — 10% or 65d", dmtf, stop_mode="none", stage_exit=False)
    add("No stop — 10% or 90d", dmtf, stop_mode="none", stage_exit=False, max_hold=90)
    add("No stop — 10% or 130d", dmtf, stop_mode="none", stage_exit=False, max_hold=130)
    add("No Stage4 + close stop", dmtf, stop_mode="close_ma", stage_exit=False)
    add("3-day stop grace (no first-week wick)", dmtf, stop_grace_days=3)
    add("5-day stop grace", dmtf, stop_grace_days=5)

    # ── B. Quality / RS / structure filters ──
    add("RS≥70 leaders", lambda s: dmtf(s) and s["rs_rating"] >= 70)
    add("RS≥80 leaders", lambda s: dmtf(s) and s["rs_rating"] >= 80)
    add("Quality≥60", lambda s: dmtf(s) and s["quality_score"] >= 60)
    add("Quality≥70", lambda s: dmtf(s) and s["quality_score"] >= 70)
    add("Not extended ≤8% EMA20", lambda s: dmtf(s) and s["not_extended"])
    add("EMA stack 20>50", lambda s: dmtf(s) and s["ema_stack_bull"])
    add("Minervini trend EMA50>200", lambda s: dmtf(s) and s["ema_trend_bull"])
    add("Follow-through", lambda s: dmtf(s) and s["follow_through"])
    add("Breakout volume ≥1.2×", lambda s: dmtf(s) and s["volume_ratio_bo"] >= 1.2)
    add("RSI 45-70 healthy", lambda s: dmtf(s) and s["rsi_healthy"])
    add("Supertrend bull", lambda s: dmtf(s) and s["supertrend_bull"])
    add("Nifty Stage 1/2 only", lambda s: dmtf(s) and s["mkt_ok"])
    add("Near 30w MA (≤8%)", lambda s: dmtf(s) and s["dist_ma"] <= 8)
    add("Near 30w MA (≤12%)", lambda s: dmtf(s) and s["dist_ma"] <= 12)
    add("Low ATR ≤2.5%", lambda s: dmtf(s) and 0 < s["atr_pct"] <= 2.5)
    add("Low ATR ≤3%", lambda s: dmtf(s) and 0 < s["atr_pct"] <= 3.0)
    add("Tight 20d range ≤15%", lambda s: dmtf(s) and s["tight_20d"] <= 15)
    add("Skip tiny stops (dist to MA ≥4%)", lambda s: dmtf(s) and s["dist_ma"] >= 4)

    # ── C. Confluence (literature: 3+ factors) ──
    add("RS70 + not ext", lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["not_extended"])
    add("RS70 + Q60", lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["quality_score"] >= 60)
    add("RS80 + not ext", lambda s: dmtf(s) and s["rs_rating"] >= 80 and s["not_extended"])
    add("RS70 + ATR≤3%", lambda s: dmtf(s) and s["rs_rating"] >= 70 and 0 < s["atr_pct"] <= 3)
    add("EMA stack + not ext + RSI healthy",
        lambda s: dmtf(s) and s["ema_stack_bull"] and s["not_extended"] and s["rsi_healthy"])
    add("RS70 + EMA stack + not ext",
        lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["ema_stack_bull"] and s["not_extended"])
    add("Leaders + trend + FT",
        lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["ema_trend_bull"] and s["follow_through"])
    add("Q60 + RS70 + close-stop",
        lambda s: dmtf(s) and s["quality_score"] >= 60 and s["rs_rating"] >= 70,
        stop_mode="close_ma")
    add("Not ext + close-stop + no S4",
        lambda s: dmtf(s) and s["not_extended"],
        stop_mode="close_ma", stage_exit=False)

    # ── D. Easier target / asymmetric R (how 80% WR systems are built) ──
    add("6% TP + 15% stop", dmtf, profit_pct=0.06, stop_mode="pct", stop_pct=0.15, stage_exit=False)
    add("8% TP + 15% stop", dmtf, profit_pct=0.08, stop_mode="pct", stop_pct=0.15, stage_exit=False)
    add("5% TP + 15% stop", dmtf, profit_pct=0.05, stop_mode="pct", stop_pct=0.15, stage_exit=False)
    add("8% TP + close-stop + no S4", dmtf, profit_pct=0.08, stop_mode="close_ma", stage_exit=False)
    add("6% TP + close-stop + RS70",
        lambda s: dmtf(s) and s["rs_rating"] >= 70,
        profit_pct=0.06, stop_mode="close_ma", stage_exit=False)
    add("8% TP + 15% stop + RS70 + not ext",
        lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["not_extended"],
        profit_pct=0.08, stop_mode="pct", stop_pct=0.15, stage_exit=False)
    add("5% TP + 20% stop + RS70",
        lambda s: dmtf(s) and s["rs_rating"] >= 70,
        profit_pct=0.05, stop_mode="pct", stop_pct=0.20, stage_exit=False)
    add("6% TP no-stop 90d + RS70",
        lambda s: dmtf(s) and s["rs_rating"] >= 70,
        profit_pct=0.06, stop_mode="none", stage_exit=False, max_hold=90)
    add("10% TP + 15% stop + RS70 + not ext",
        lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["not_extended"],
        stop_mode="pct", stop_pct=0.15, stage_exit=False)
    add("8% TP + close-stop + RS70 + Q60",
        lambda s: dmtf(s) and s["rs_rating"] >= 70 and s["quality_score"] >= 60,
        profit_pct=0.08, stop_mode="close_ma", stage_exit=False)

    # ── E. Pullback entry (Weinstein: don't chase, wait for MA) ──
    add("PB to EMA20 (wait ≤15d)", dmtf, _entry="pb_entry_day")
    add("PB EMA20 + RS70", lambda s: dmtf(s) and s["rs_rating"] >= 70, _entry="pb_entry_day")
    add("PB EMA20 + close-stop", dmtf, stop_mode="close_ma", _entry="pb_entry_day")
    add("PB + 8% TP + 15% stop + RS70",
        lambda s: dmtf(s) and s["rs_rating"] >= 70,
        profit_pct=0.08, stop_mode="pct", stop_pct=0.15, stage_exit=False, _entry="pb_entry_day")

    rows = []
    print("")
    print(f"{'variant':<48} {'n':>3} {'WR':>6} {'ret':>7} {'PF':>5} {'DD':>5} {'SL':>3} {'TP':>3} hold")
    print("-" * 100)
    for name, pred, kw in variants:
        entry_key = kw.pop("_entry", "entry_day")
        bd = apply_pred(sigs, pred, entry_key=entry_key)
        r = simulate(
            frames, weekly_by_sym, bd, calendar,
            capital=CAPITAL, risk_pct=risk_pct, start_date=start, **kw,
        )
        rows.append((name, r))
        flag = " <<" if r["wr"] >= 80 and r["n"] >= 8 else (" <" if r["wr"] >= 70 and r["n"] >= 8 else "")
        print(
            f"{name:<48} {r['n']:3d} {r['wr']:5.1f}% {r['ret']:+6.1f}% "
            f"{r['pf']:5.2f} {r['dd']:4.1f}% {r['stops']:3d} {r['hits']:3d} {r['avg_hold']:4.1f}{flag}"
        )
        kw["_entry"] = entry_key  # restore not needed

    print("")
    print("=== Best win-rate (n≥8) ===")
    ranked = [x for x in rows if x[1]["n"] >= 8]
    ranked.sort(key=lambda x: (x[1]["wr"], x[1]["ret"]), reverse=True)
    for name, r in ranked[:12]:
        print(
            f"  {name:<48} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%"
        )
    print("")
    print("=== Best return among WR≥70% (n≥8) ===")
    hi = [x for x in ranked if x[1]["wr"] >= 70]
    hi.sort(key=lambda x: x[1]["ret"], reverse=True)
    if not hi:
        print("  (none reached 70% WR with n≥8)")
    for name, r in hi[:8]:
        print(
            f"  {name:<48} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%"
        )


if __name__ == "__main__":
    main()
