"""
Search technical filter packs for Stage 2.0:
  raise win-rate without reducing return vs baseline (stage_4_only, Rs10L).
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Optional

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
    EXIT_STAGE_4_ONLY,
    MAX_HOLD_DAYS,
    StageV2Trade,
    _OpenPos,
    _close_trade,
    _collect_stage2_signals,
    _equity,
    _invested_total,
    _preload_frames,
    _weekly_stage_at,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.tech_filters import (
    FILTER_PACKS,
    enrich_daily_tech,
    passes_tech_filter,
    snapshot_tech,
)
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def simulate_cash(
    frames,
    weekly_by_sym,
    signals_by_day,
    calendar,
    *,
    capital: float,
    risk_pct: float,
    start_date: date,
    exit_mode: str = EXIT_STAGE_4_ONLY,
) -> dict:
    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve = [{"date": str(start_date), "equity": capital}]
    peak = capital
    max_dd = 0.0
    skipped_cash = 0
    peak_parallel = 0

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
                    stage_now, _ = _weekly_stage_at(weekly, int(w_idx))

            exit_price = None
            exit_reason = ""
            if low <= pos.stop:
                exit_price = pos.stop
                exit_reason = "stop_loss"
            elif high >= pos.target:
                exit_price = pos.target
                exit_reason = "target_2.5r"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                exit_price = close
                exit_reason = "time_exit"
            elif stage_now is not None and exit_mode == EXIT_STAGE_4_ONLY and stage_now == 4:
                exit_price = close
                exit_reason = "stage_exit"
            elif stage_now is not None and exit_mode == "stage_3_4" and stage_now in (3, 4):
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
            closed.append(sym)

        for sym in closed:
            opens.pop(sym, None)

        day_sigs = signals_by_day.get(ts, [])
        if day_sigs:
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
                target = float(sig["target"])
                risk = entry_price - stop
                if risk <= 0:
                    continue

                equity_now = cash + _invested_total(opens)
                if cash <= 0 or equity_now <= 0:
                    skipped_cash += 1
                    continue
                pos_size = calculate_position_size(equity_now, risk_pct, entry_price, stop)
                qty = int(pos_size.quantity)
                max_qty = int(cash // entry_price) if entry_price > 0 else 0
                qty = min(qty, max_qty)
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
                    weekly_stage=2,
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
                        pos, exit_price=target, exit_ts=ts, exit_reason="target_2.5r", days_held=0
                    )
                    cash += pos.notional + trade.pnl
                    trades.append(trade)
                    last_exit[sym] = ts
                    opens.pop(sym, None)

        eq = cash + _invested_total(opens)
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)

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

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    pf = gp / gl if trades else 0.0
    ret = total_pnl / capital * 100

    return {
        "n": len(trades),
        "signals": sum(len(v) for v in signals_by_day.values()),
        "wr": round(wr, 2),
        "pf": round(pf, 2),
        "ret": round(ret, 2),
        "pnl": round(total_pnl, 2),
        "dd": round(max_dd, 2),
        "avg_r": round(sum(t.rr_achieved for t in trades) / len(trades), 2) if trades else 0,
        "exits": dict(Counter(t.exit_reason for t in trades)),
        "skipped_cash": skipped_cash,
        "peak_parallel": peak_parallel,
        "trades": trades,
    }


def main():
    end = date.today()
    start = end - timedelta(days=365)
    capital = 1_000_000.0
    config = StrategyConfig.get_active()
    risk_pct = float(getattr(config, "risk_pct", 2.0) or 2.0)

    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"Loading {len(symbols)} symbols | {start} → {end}")
    frames = _preload_frames(symbols)
    print(f"Loaded {len(frames)}")

    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = (
        add_weekly_indicators(daily_to_weekly(nifty_daily))
        if not nifty_daily.empty
        else pd.DataFrame()
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Enrich tech once
    print("Enriching daily tech (EMA/RSI/BB/Supertrend)...")
    tech_by_sym = {}
    weekly_by_sym = {}
    for i, (sym, daily) in enumerate(frames.items()):
        weekly = add_weekly_indicators(daily_to_weekly(daily))
        if len(weekly) < 40:
            continue
        weekly_by_sym[sym] = weekly
        tech_by_sym[sym] = enrich_daily_tech(daily)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(frames)}...")

    print("Collecting Stage 2 signals + tech snapshots...")
    raw_signals = []  # list of sig dicts with entry_day, tech snap flags
    for sym, weekly in weekly_by_sym.items():
        daily = frames[sym]
        tech = tech_by_sym.get(sym)
        bench = nifty_weekly if not nifty_weekly.empty else weekly
        sigs = _collect_stage2_signals(
            sym, weekly, daily, bench, nifty_weekly,
            start_ts, end_ts, min_quality_score=0, market_filter=False,
        )
        for sig in sigs:
            snap = snapshot_tech(tech, sig["signal_date"]) if tech is not None else None
            # attach flags for filtering
            if snap and snap.ok:
                sig["tech_ok"] = True
                sig["ema_stack_bull"] = snap.ema_stack_bull
                sig["ema_trend_bull"] = snap.ema_trend_bull
                sig["supertrend_bull"] = snap.supertrend_bull
                sig["rsi_healthy"] = snap.rsi_healthy
                sig["rsi_not_overbought"] = snap.rsi_not_overbought
                sig["bb_above_mid"] = snap.bb_above_mid
                sig["volume_ok"] = snap.volume_ok
                sig["breakout_20d"] = snap.breakout_20d
                sig["rsi"] = snap.rsi
            else:
                sig["tech_ok"] = False
            raw_signals.append(sig)

    print(f"Total Stage 2 signals: {len(raw_signals)}")
    tech_ok_n = sum(1 for s in raw_signals if s.get("tech_ok"))
    print(f"With valid tech snapshot: {tech_ok_n}")

    calendar_set = set()
    for df in frames.values():
        calendar_set.update(df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist())
    calendar = sorted(calendar_set)

    def filter_signals(pack_id: str, extra_min_q: int = 0, min_rs: float = 0):
        by_day = defaultdict(list)
        pack = FILTER_PACKS[pack_id]
        req = pack.get("require") or []
        for sig in raw_signals:
            if int(sig.get("quality_score") or 0) < extra_min_q:
                continue
            if float(sig.get("rs_rating") or 0) < min_rs:
                continue
            if req:
                if not sig.get("tech_ok"):
                    continue
                # build mini snap-like check
                ok = all(bool(sig.get(f)) for f in req)
                if not ok:
                    continue
            by_day[sig["entry_day"]].append(sig)
        return by_day

    # Also test quality-only and hybrid packs not in FILTER_PACKS
    experiments = []
    for pack_id in FILTER_PACKS:
        experiments.append((pack_id, 0, 0, FILTER_PACKS[pack_id]["label"]))

    # hybrids with quality / RS
    hybrids = [
        ("st_ema", 60, 0, "ST+EMA stack + Q≥60"),
        ("st_ema", 70, 0, "ST+EMA stack + Q≥70"),
        ("st_ema_rsi", 0, 0, "ST+EMA+RSI"),
        ("st_ema_rsi", 60, 0, "ST+EMA+RSI + Q≥60"),
        ("confluence", 0, 0, "Confluence ST+EMA+BB+RSI"),
        ("confluence", 60, 0, "Confluence + Q≥60"),
        ("st_ema_breakout", 0, 0, "ST+EMA+20d high"),
        ("st_ema_breakout", 60, 0, "ST+EMA+20d + Q≥60"),
        ("ema_trend_st", 0, 55, "EMA trend+ST + RS≥55"),
        ("ema_trend_st_rsi", 0, 0, "EMA trend+ST+RSI"),
        ("strict_momentum", 0, 0, "Strict momentum"),
        ("none", 75, 0, "Q≥75 only"),
        ("none", 0, 70, "RS≥70 only"),
        ("none", 70, 60, "Q≥70 + RS≥60"),
        ("st_ema", 0, 60, "ST+EMA + RS≥60"),
        ("confluence_vol", 0, 0, "Confluence+vol"),
        ("supertrend", 65, 0, "Supertrend + Q≥65"),
    ]
    for pack_id, q, rs, label in hybrids:
        experiments.append((pack_id, q, rs, label))

    # de-dupe by label
    seen = set()
    uniq = []
    for e in experiments:
        if e[3] in seen:
            continue
        seen.add(e[3])
        uniq.append(e)

    print(f"\nRunning {len(uniq)} filter experiments (cash Rs10L, exit Stage4 only)...")
    results = []
    for pack_id, min_q, min_rs, label in uniq:
        by_day = filter_signals(pack_id, min_q, min_rs)
        n_sig = sum(len(v) for v in by_day.values())
        r = simulate_cash(
            frames, weekly_by_sym, by_day, calendar,
            capital=capital, risk_pct=risk_pct, start_date=start,
        )
        r["label"] = label
        r["pack"] = pack_id
        r["min_q"] = min_q
        r["min_rs"] = min_rs
        r["n_sig"] = n_sig
        results.append(r)
        print(
            f"  {label[:42]:<42} sig={n_sig:3d} n={r['n']:3d} "
            f"WR={r['wr']:5.1f}% ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%"
        )

    baseline = next(r for r in results if r["label"] == "No tech filter")
    base_ret = baseline["ret"]
    base_wr = baseline["wr"]

    print("\n" + "=" * 100)
    print(f"BASELINE: WR={base_wr}% ret={base_ret}% PF={baseline['pf']} n={baseline['n']}")
    print("=" * 100)

    # Goal: WR up, ret not down (within 2pp tolerance) , prefer n>=15
    improved = [
        r for r in results
        if r["wr"] > base_wr + 0.5
        and r["ret"] >= base_ret - 2.0
        and r["n"] >= 12
    ]
    improved.sort(key=lambda x: (x["wr"], x["ret"], x["pf"]), reverse=True)

    print("\n=== IMPROVED WR & RETURN ≥ baseline-2pp (n≥12) ===")
    if not improved:
        print("  (none with strict rule — showing best WR with ret ≥ baseline-5pp)")
        improved = [
            r for r in results
            if r["wr"] > base_wr and r["ret"] >= base_ret - 5 and r["n"] >= 12
        ]
        improved.sort(key=lambda x: (x["wr"], x["ret"]), reverse=True)

    for r in improved[:15]:
        print(
            f"  {r['label'][:45]:<45} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}% "
            f"ΔWR={r['wr']-base_wr:+.1f} Δret={r['ret']-base_ret:+.1f}"
        )

    # Best overall score: maximize WR subject to ret >= baseline
    keep_ret = [r for r in results if r["ret"] >= base_ret - 0.5 and r["n"] >= 15]
    keep_ret.sort(key=lambda x: (x["wr"], x["ret"], x["pf"]), reverse=True)
    print("\n=== BEST WR WITH RETURN ≈ BASELINE (ret ≥ baseline-0.5, n≥15) ===")
    for r in keep_ret[:10]:
        print(
            f"  {r['label'][:45]:<45} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%"
        )

    # Pareto: high return and high WR
    print("\n=== TOP BY RETURN ===")
    for r in sorted(results, key=lambda x: x["ret"], reverse=True)[:8]:
        print(
            f"  {r['label'][:45]:<45} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f}"
        )

    print("\n=== TOP BY WIN RATE (n≥15) ===")
    for r in sorted([x for x in results if x["n"] >= 15], key=lambda x: x["wr"], reverse=True)[:10]:
        print(
            f"  {r['label'][:45]:<45} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f}"
        )

    # Recommend
    candidates = [r for r in results if r["ret"] >= base_ret - 1.0 and r["wr"] > base_wr and r["n"] >= 15]
    if not candidates:
        candidates = [r for r in results if r["ret"] >= base_ret - 3 and r["wr"] > base_wr and r["n"] >= 12]
    if candidates:
        # score = wr improvement + small ret weight
        best = max(candidates, key=lambda r: (r["wr"] - base_wr) * 2 + (r["ret"] - base_ret) + r["pf"])
        print("\n*** RECOMMENDED ***")
        print(f"  {best['label']}")
        print(
            f"  n={best['n']} WR={best['wr']}% (Δ{best['wr']-base_wr:+.1f}) "
            f"ret={best['ret']}% (Δ{best['ret']-base_ret:+.1f}) PF={best['pf']} DD={best['dd']}%"
        )
        print(f"  pack={best['pack']} min_q={best['min_q']} min_rs={best['min_rs']}")
        print(f"  exits={best['exits']}")
    else:
        print("\nNo clear WR upgrade without return hit — check top lists.")

    print("Done.")


if __name__ == "__main__":
    main()
