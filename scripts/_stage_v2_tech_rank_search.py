"""
Find ways to raise Stage 2.0 win-rate without cutting return.

Tests:
  1) Hard tech filters
  2) Soft ranking (tech confluence score for capital priority)
  3) Soft filter: reject only clearly bad tech (extended / no EMA / weak)
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
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
    MAX_HOLD_DAYS,
    _OpenPos,
    _close_trade,
    _invested_total,
    _preload_frames,
    _weekly_stage_at,
)
from stage_analysis_v2.services.indicators import add_daily_indicators as add_d_ind
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.breakout_detector import detect_breakout
from stage_analysis_v2.services.quality_score import compute_quality_score
from stage_analysis_v2.services.relative_strength import compute_relative_strength
from stage_analysis_v2.services.stage_engine import detect_daily_stage
from stage_analysis_v2.services.tech_filters import enrich_daily_tech, snapshot_tech
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def collect_rich_signals(frames, weekly_by_sym, tech_by_sym, nifty_weekly, start_ts, end_ts):
    """Collect Stage2 signals with tech + breakout + daily stage attached."""
    from stage_analysis_v2.services.backtester import TARGET_RR, _weekly_stage_at

    all_sigs = []
    for sym, weekly in weekly_by_sym.items():
        daily = frames[sym]
        daily_ind = add_d_ind(daily)
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
            quality = compute_quality_score(
                weekly_stage=2,
                daily_stage=d_stage,
                weekly_metrics=metrics,
                breakout=breakout,
                rs=rs,
                weekly=w_slice,
                market_favorable=True,
            )

            ma = metrics.get("ma", 0.0)
            price = metrics.get("price", float(weekly.iloc[w_idx]["close"]))
            stop = round(ma * 0.95, 2) if ma else round(price * 0.93, 2)
            target = round(price + (price - stop) * TARGET_RR, 2) if stop < price else round(price * 1.1, 2)

            entry_day = daily.index[daily.index > week_end]
            if len(entry_day) == 0 or entry_day[0] > end_ts:
                continue
            entry_day = entry_day[0]

            snap = snapshot_tech(tech, week_end)
            # extension: how far above ema20
            ext = snap.price_vs_ema20_pct if snap.ok else 0.0
            not_extended = snap.ok and ext <= 8.0  # within 8% of ema20
            mild_ext = snap.ok and ext <= 12.0

            tech_score = 0
            if snap.ok:
                tech_score += 25 if snap.supertrend_bull else 0
                tech_score += 20 if snap.ema_stack_bull else 0
                tech_score += 15 if snap.ema_trend_bull else 0
                tech_score += 15 if snap.bb_above_mid else 0
                tech_score += 10 if snap.rsi_healthy else 0
                tech_score += 5 if snap.rsi_not_overbought else 0
                tech_score += 10 if snap.volume_ok else 0
                tech_score += 10 if snap.breakout_20d else 0
                tech_score += 10 if not_extended else 0
                tech_score += 15 if d_stage == 2 else (5 if d_stage == 1 else 0)
                tech_score += 20 if breakout.breakout_type == "clean" else (
                    8 if breakout.breakout_type == "weak" else 0
                )
                tech_score += 10 if breakout.follow_through else 0

            rank_score = quality.total + tech_score * 0.5 + rs.rating * 0.15

            all_sigs.append({
                "symbol": sym,
                "signal_date": week_end,
                "entry_day": entry_day,
                "quality_score": quality.total,
                "rs_rating": rs.rating,
                "stop": stop,
                "target": target,
                "signal_close": price,
                "weekly_stage": 2,
                "daily_stage": d_stage,
                "breakout_type": breakout.breakout_type,
                "follow_through": breakout.follow_through,
                "volume_ratio_bo": breakout.volume_ratio,
                "tech_ok": snap.ok,
                "ema_stack_bull": snap.ema_stack_bull if snap.ok else False,
                "ema_trend_bull": snap.ema_trend_bull if snap.ok else False,
                "supertrend_bull": snap.supertrend_bull if snap.ok else False,
                "bb_above_mid": snap.bb_above_mid if snap.ok else False,
                "rsi_healthy": snap.rsi_healthy if snap.ok else False,
                "rsi_not_overbought": snap.rsi_not_overbought if snap.ok else False,
                "volume_ok": snap.volume_ok if snap.ok else False,
                "breakout_20d": snap.breakout_20d if snap.ok else False,
                "not_extended": not_extended,
                "mild_ext": mild_ext,
                "rsi": snap.rsi if snap.ok else 0,
                "ext_pct": ext,
                "tech_score": tech_score,
                "rank_score": rank_score,
            })
    return all_sigs


def simulate(frames, weekly_by_sym, signals_by_day, calendar, capital, risk_pct, start_date, rank_key="quality"):
    cash = float(capital)
    opens = {}
    last_exit = {}
    trades = []
    peak = capital
    max_dd = 0.0
    peak_par = 0

    for ts in calendar:
        closed = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close, low, high = float(row["close"]), float(row["low"]), float(row["high"])
            pos.hold_days += 1
            stage_now = None
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
                    stage_now, _ = _weekly_stage_at(weekly, int(w_idx))

            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price, exit_reason = pos.stop, "stop_loss"
            elif high >= pos.target:
                exit_price, exit_reason = pos.target, "target_2.5r"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                exit_price, exit_reason = close, "time_exit"
            elif stage_now == 4:
                exit_price, exit_reason = close, "stage_exit"
            if exit_price is None:
                continue
            trade = _close_trade(pos, exit_price=exit_price, exit_ts=ts, exit_reason=exit_reason, days_held=pos.hold_days)
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed.append(sym)
        for sym in closed:
            opens.pop(sym, None)

        day_sigs = list(signals_by_day.get(ts, []))
        if day_sigs:
            if rank_key == "quality":
                day_sigs.sort(key=lambda s: (s.get("quality_score", 0), s.get("rs_rating", 0)), reverse=True)
            elif rank_key == "tech":
                day_sigs.sort(key=lambda s: (s.get("tech_score", 0), s.get("quality_score", 0), s.get("rs_rating", 0)), reverse=True)
            elif rank_key == "rank":
                day_sigs.sort(key=lambda s: (s.get("rank_score", 0), s.get("quality_score", 0)), reverse=True)
            elif rank_key == "rs":
                day_sigs.sort(key=lambda s: (s.get("rs_rating", 0), s.get("quality_score", 0)), reverse=True)

            for sig in day_sigs:
                sym = sig["symbol"]
                if sym in opens or sym not in frames or ts not in frames[sym].index:
                    continue
                prev_x = last_exit.get(sym)
                if prev_x is not None and (ts - prev_x).days < COOLDOWN_DAYS:
                    continue
                row = frames[sym].loc[ts]
                entry_price = float(row["open"])
                high, low = float(row["high"]), float(row["low"])
                stop, target = float(sig["stop"]), float(sig["target"])
                if entry_price - stop <= 0:
                    continue
                equity_now = cash + _invested_total(opens)
                if cash <= 0 or equity_now <= 0:
                    continue
                qty = int(calculate_position_size(equity_now, risk_pct, entry_price, stop).quantity)
                qty = min(qty, int(cash // entry_price) if entry_price > 0 else 0)
                if qty <= 0:
                    continue
                notional = qty * entry_price
                if notional > cash + 1e-6:
                    continue
                cash -= notional
                inv = _invested_total(opens) + notional
                peak_par = max(peak_par, len(opens) + 1)
                pos = _OpenPos(
                    symbol=sym, entry_date=ts, signal_date=sig["signal_date"],
                    entry_price=entry_price, stop=stop, target=target, qty=qty, notional=notional,
                    quality_score=int(sig.get("quality_score") or 0),
                    rs_rating=float(sig.get("rs_rating") or 0), weekly_stage=2,
                    capital_invested=notional, cash_available=cash, total_invested=inv,
                    parallel_open=len(opens) + 1, equity_at_entry=cash + inv,
                )
                opens[sym] = pos
                if low <= stop:
                    t = _close_trade(pos, exit_price=stop, exit_ts=ts, exit_reason="stop_loss", days_held=0)
                    cash += pos.notional + t.pnl
                    trades.append(t)
                    last_exit[sym] = ts
                    opens.pop(sym, None)
                elif high >= target:
                    t = _close_trade(pos, exit_price=target, exit_ts=ts, exit_reason="target_2.5r", days_held=0)
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
    return {
        "n": len(trades),
        "wr": round(wr, 2),
        "pf": round(gp / gl, 2),
        "ret": round(pnl / capital * 100, 2),
        "pnl": round(pnl, 2),
        "dd": round(max_dd, 2),
        "exits": dict(Counter(t.exit_reason for t in trades)),
        "avg_q": round(sum(t.quality_score for t in trades) / len(trades), 1) if trades else 0,
    }


def apply_pred(sigs, pred):
    by_day = defaultdict(list)
    for s in sigs:
        if pred(s):
            by_day[s["entry_day"]].append(s)
    return by_day


def main():
    end = date.today()
    start = end - timedelta(days=365)
    capital = 1_000_000.0
    risk_pct = float(getattr(StrategyConfig.get_active(), "risk_pct", 2.0) or 2.0)

    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print("Loading...")
    frames = _preload_frames(symbols)
    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = add_weekly_indicators(daily_to_weekly(nifty_daily)) if not nifty_daily.empty else pd.DataFrame()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    weekly_by_sym, tech_by_sym = {}, {}
    print("Tech enrich...")
    for sym, daily in frames.items():
        w = add_weekly_indicators(daily_to_weekly(daily))
        if len(w) < 40:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(daily)

    print("Rich signals...")
    sigs = collect_rich_signals(frames, weekly_by_sym, tech_by_sym, nifty_weekly, start_ts, end_ts)
    print(f"Signals: {len(sigs)}")

    cal = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist()
    })

    experiments = []

    def run(name, by_day, rank="quality"):
        r = simulate(frames, weekly_by_sym, by_day, cal, capital, risk_pct, start, rank_key=rank)
        r["name"] = name
        r["sig"] = sum(len(v) for v in by_day.values())
        experiments.append(r)
        print(f"  {name[:48]:<48} sig={r['sig']:3d} n={r['n']:3d} WR={r['wr']:5.1f}% ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%")

    print("\n--- Experiments ---")
    # baseline
    run("BASE quality-rank", apply_pred(sigs, lambda s: True), "quality")
    run("BASE tech-rank (no filter)", apply_pred(sigs, lambda s: True), "tech")
    run("BASE composite-rank", apply_pred(sigs, lambda s: True), "rank")
    run("BASE rs-rank", apply_pred(sigs, lambda s: True), "rs")

    # soft reject: only drop clearly bad
    run("soft: EMA stack OR BB mid", apply_pred(sigs, lambda s: s["ema_stack_bull"] or s["bb_above_mid"]))
    run("soft: not extended ≤8% EMA20", apply_pred(sigs, lambda s: s["not_extended"]))
    run("soft: mild ext ≤12%", apply_pred(sigs, lambda s: s["mild_ext"]))
    run("soft: daily stage 1or2", apply_pred(sigs, lambda s: s["daily_stage"] in (1, 2)))
    run("soft: daily stage 2", apply_pred(sigs, lambda s: s["daily_stage"] == 2))
    run("soft: clean|weak breakout", apply_pred(sigs, lambda s: s["breakout_type"] in ("clean", "weak")))
    run("soft: clean breakout only", apply_pred(sigs, lambda s: s["breakout_type"] == "clean"))
    run("soft: follow-through", apply_pred(sigs, lambda s: s["follow_through"]))
    run("soft: RSI not OB (<75)", apply_pred(sigs, lambda s: s["rsi_not_overbought"]))
    run("soft: RSI 50-68", apply_pred(sigs, lambda s: 50 <= s["rsi"] <= 68))
    run("soft: tech_score≥40", apply_pred(sigs, lambda s: s["tech_score"] >= 40))
    run("soft: tech_score≥50", apply_pred(sigs, lambda s: s["tech_score"] >= 50))
    run("soft: tech_score≥60", apply_pred(sigs, lambda s: s["tech_score"] >= 60))
    run("soft: rank_score≥80", apply_pred(sigs, lambda s: s["rank_score"] >= 80))
    run("soft: rank_score≥90", apply_pred(sigs, lambda s: s["rank_score"] >= 90))

    # combos aiming WR up, ret hold
    run("EMA stack + not extended", apply_pred(sigs, lambda s: s["ema_stack_bull"] and s["not_extended"]))
    run("EMA stack + BB + not ext", apply_pred(sigs, lambda s: s["ema_stack_bull"] and s["bb_above_mid"] and s["mild_ext"]))
    run("BB mid + not extended", apply_pred(sigs, lambda s: s["bb_above_mid"] and s["not_extended"]))
    run("BB mid + daily S2", apply_pred(sigs, lambda s: s["bb_above_mid"] and s["daily_stage"] == 2))
    run("BB mid + follow-through", apply_pred(sigs, lambda s: s["bb_above_mid"] and s["follow_through"]))
    run("tech≥50 + rank sort", apply_pred(sigs, lambda s: s["tech_score"] >= 50), "rank")
    run("tech≥40 + rank sort", apply_pred(sigs, lambda s: s["tech_score"] >= 40), "rank")
    run("not ext + tech rank", apply_pred(sigs, lambda s: s["not_extended"]), "tech")
    run("mild ext + composite rank", apply_pred(sigs, lambda s: s["mild_ext"]), "rank")
    run("BB mid + composite rank", apply_pred(sigs, lambda s: s["bb_above_mid"]), "rank")
    run("EMA stack + composite rank", apply_pred(sigs, lambda s: s["ema_stack_bull"]), "rank")
    run("daily S2 + composite", apply_pred(sigs, lambda s: s["daily_stage"] == 2), "rank")
    run("clean BO + composite", apply_pred(sigs, lambda s: s["breakout_type"] == "clean"), "rank")
    run("weak|clean BO + tech≥40", apply_pred(sigs, lambda s: s["breakout_type"] in ("clean", "weak") and s["tech_score"] >= 40), "rank")
    run("FT + BB + mild ext", apply_pred(sigs, lambda s: s["follow_through"] and s["bb_above_mid"] and s["mild_ext"]), "rank")
    run("FT + EMA stack", apply_pred(sigs, lambda s: s["follow_through"] and s["ema_stack_bull"]), "rank")
    run("Q≥55 + tech≥40 + rank", apply_pred(sigs, lambda s: s["quality_score"] >= 55 and s["tech_score"] >= 40), "rank")
    run("Q≥50 + not ext + rank", apply_pred(sigs, lambda s: s["quality_score"] >= 50 and s["not_extended"]), "rank")
    run("RS≥55 + BB + rank", apply_pred(sigs, lambda s: s["rs_rating"] >= 55 and s["bb_above_mid"]), "rank")
    run("RS≥50 + EMA stack + rank", apply_pred(sigs, lambda s: s["rs_rating"] >= 50 and s["ema_stack_bull"]), "rank")

    base = experiments[0]
    print("\n" + "=" * 90)
    print(f"BASELINE: WR={base['wr']}% ret={base['ret']}% PF={base['pf']} n={base['n']}")
    print("=" * 90)

    # WR up, ret not down more than 1pp
    good = [e for e in experiments if e["wr"] > base["wr"] + 0.5 and e["ret"] >= base["ret"] - 1.0 and e["n"] >= 15]
    good.sort(key=lambda x: (x["wr"], x["ret"]), reverse=True)
    print("\n=== WR↑ AND ret ≥ baseline-1pp ===")
    for e in good[:12]:
        print(f"  {e['name'][:50]:<50} n={e['n']:3d} WR={e['wr']:5.1f}% ret={e['ret']:+6.1f}% PF={e['pf']:.2f} ΔWR={e['wr']-base['wr']:+.1f} Δret={e['ret']-base['ret']:+.1f}")

    good2 = [e for e in experiments if e["wr"] > base["wr"] and e["ret"] >= base["ret"] - 3 and e["n"] >= 15]
    good2.sort(key=lambda x: (x["ret"] + (x["wr"] - base["wr"]) * 1.5), reverse=True)
    print("\n=== BEST COMPROMISE (WR↑, ret ≥ base-3) scored ===")
    for e in good2[:15]:
        score = e["ret"] + (e["wr"] - base["wr"]) * 1.5
        print(f"  {e['name'][:50]:<50} n={e['n']:3d} WR={e['wr']:5.1f}% ret={e['ret']:+6.1f}% score={score:.1f}")

    # any with higher ret AND higher WR
    both = [e for e in experiments if e["wr"] > base["wr"] and e["ret"] > base["ret"] and e["n"] >= 15]
    print("\n=== STRICT: WR↑ AND ret↑ ===")
    if both:
        for e in sorted(both, key=lambda x: (x["wr"], x["ret"]), reverse=True):
            print(f"  {e['name'][:50]:<50} n={e['n']:3d} WR={e['wr']:5.1f}% ret={e['ret']:+6.1f}%")
    else:
        print("  none")

    # best ranking-only (no hard filter)
    print("\n=== RANKING ONLY ===")
    for e in experiments[:4]:
        print(f"  {e['name']}: WR={e['wr']}% ret={e['ret']}%")

    print("Done.")


if __name__ == "__main__":
    main()
