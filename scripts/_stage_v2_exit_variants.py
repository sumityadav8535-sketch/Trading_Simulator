"""
Compare Stage 2.0 exit / filter variants on the last 1 year.

Goal: reduce damage from stage_exit without killing the +30% edge.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
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
    MAX_HOLD_DAYS,
    TARGET_RR,
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
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


@dataclass
class VariantResult:
    name: str
    trades: list[StageV2Trade] = field(default_factory=list)
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    total_trades: int = 0
    exit_breakdown: dict = field(default_factory=dict)
    monthly: dict = field(default_factory=dict)
    worst_month: str = ""
    worst_month_pnl: float = 0.0
    stage_exit_pnl: float = 0.0
    final_equity: float = 0.0


def simulate(
    name: str,
    frames: dict[str, pd.DataFrame],
    weekly_by_sym: dict[str, pd.DataFrame],
    signals_by_day: dict,
    calendar: list,
    *,
    capital: float,
    risk_pct: float,
    start_date: date,
    # Exit modes
    stage_exit_mode: str = "stage_3_4",  # off | stage_3_4 | stage_4_only | confirm_2w | stage_below_ma
    trail_ma: bool = False,  # trail stop up to 30w MA * 0.98 each week
    tighten_stop_on_stage3: bool = False,  # move stop to MA when stage 3, don't force exit
) -> VariantResult:
    """
    stage_exit_mode:
      off            — never exit on stage
      stage_3_4      — current behavior
      stage_4_only   — only stage 4
      confirm_2w     — need 2 consecutive weeks in 3/4
      stage_below_ma — stage 3/4 AND close < weekly MA
    """
    cash = float(capital)
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    equity_curve = [{"date": str(start_date), "equity": capital}]
    last_curve_date = None
    # confirm_2w state
    bad_stage_weeks: dict[str, int] = defaultdict(int)

    def record_curve(ts, force=False):
        nonlocal last_curve_date
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if not force and last_curve_date is not None and (d - last_curve_date).days < 5:
            if d.weekday() != 4:
                return
        eq = _equity(cash, opens)
        equity_curve.append({"date": str(d), "equity": round(eq, 2)})
        last_curve_date = d

    for ts in calendar:
        closed_today = []
        for sym, pos in list(opens.items()):
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            pos.hold_days += 1

            # Optional: trail stop to weekly MA
            weekly = weekly_by_sym.get(sym)
            stage_now = None
            ma_now = None
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
                    w_idx = int(w_idx)
                    stage_now, metrics = _weekly_stage_at(weekly, w_idx)
                    ma_now = metrics.get("ma") if metrics else None
                    if ma_now is None and w_idx < len(weekly):
                        # fall back from weekly close series if available
                        if "ma_30" in weekly.columns:
                            ma_now = float(weekly.iloc[w_idx].get("ma_30") or 0) or None
                        elif "ma" in weekly.columns:
                            ma_now = float(weekly.iloc[w_idx].get("ma") or 0) or None

                    if trail_ma and ma_now and ma_now > 0:
                        new_stop = round(ma_now * 0.98, 2)
                        # only trail up (never loosen)
                        if new_stop > pos.stop and new_stop < pos.entry_price * 1.5:
                            # allow stop above entry once in profit (trail)
                            if new_stop < close:  # don't stop out same bar artificially
                                pos.stop = new_stop

                    if tighten_stop_on_stage3 and stage_now == 3 and ma_now and ma_now > pos.stop:
                        pos.stop = round(min(ma_now, close * 0.99), 2)

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
            elif stage_now is not None and stage_exit_mode != "off":
                is_bad = stage_now in (3, 4)
                is_s4 = stage_now == 4

                if stage_exit_mode == "stage_3_4" and is_bad:
                    exit_price = close
                    exit_reason = "stage_exit"
                elif stage_exit_mode == "stage_4_only" and is_s4:
                    exit_price = close
                    exit_reason = "stage_exit"
                elif stage_exit_mode == "confirm_2w":
                    if is_bad:
                        bad_stage_weeks[sym] += 1
                    else:
                        bad_stage_weeks[sym] = 0
                    if bad_stage_weeks[sym] >= 2:
                        exit_price = close
                        exit_reason = "stage_exit"
                elif stage_exit_mode == "stage_below_ma" and is_bad:
                    # only exit if also price below MA (confirmed breakdown)
                    if ma_now and close < ma_now:
                        exit_price = close
                        exit_reason = "stage_exit"
                    elif ma_now is None and is_s4:
                        exit_price = close
                        exit_reason = "stage_exit"

            if exit_price is None:
                continue

            trade = _close_trade(
                pos,
                exit_price=exit_price,
                exit_ts=ts,
                exit_reason=exit_reason,
                days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)
            last_exit[sym] = ts
            closed_today.append(sym)
            bad_stage_weeks.pop(sym, None)

        for sym in closed_today:
            opens.pop(sym, None)
        if closed_today:
            record_curve(ts, force=True)

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
                open_price = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])
                stop = float(sig["stop"])
                target = float(sig["target"])
                entry_price = open_price
                risk = entry_price - stop
                if risk <= 0:
                    continue

                equity_now = _equity(cash, opens)
                if cash <= 0 or equity_now <= 0:
                    continue

                pos_size = calculate_position_size(equity_now, risk_pct, entry_price, stop)
                qty = int(pos_size.quantity)
                max_qty_cash = int(cash // entry_price) if entry_price > 0 else 0
                qty = min(qty, max_qty_cash)
                if qty <= 0:
                    continue

                notional = qty * entry_price
                if notional > cash + 1e-6:
                    continue

                cash -= notional
                invested_after = _invested_total(opens) + notional
                parallel = len(opens) + 1
                equity_at_entry = cash + invested_after

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
                    hold_days=0,
                    capital_invested=notional,
                    cash_available=cash,
                    total_invested=invested_after,
                    parallel_open=parallel,
                    equity_at_entry=equity_at_entry,
                )
                opens[sym] = pos
                bad_stage_weeks[sym] = 0

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

                record_curve(ts, force=True)

        if ts == calendar[-1] or ts.weekday() == 4:
            record_curve(ts, force=(ts == calendar[-1]))

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
            exit_ts = hist.index[-1]
            trade = _close_trade(
                pos,
                exit_price=close,
                exit_ts=exit_ts,
                exit_reason="eod_force",
                days_held=pos.hold_days,
            )
            cash += pos.notional + trade.pnl
            trades.append(trade)
        opens.clear()

    trades.sort(key=lambda t: (t.entry_date, t.symbol))
    final_equity = cash

    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        monthly[t.exit_date[:7]] += t.pnl

    worst_month = ""
    worst_pnl = 0.0
    if monthly:
        worst_month, worst_pnl = min(monthly.items(), key=lambda x: x[1])

    stage_pnl = sum(t.pnl for t in trades if t.exit_reason == "stage_exit")

    wr = 0.0
    pf = 0.0
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        wr = round(len(wins) / len(trades) * 100, 2)
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1e-9
        pf = round(gp / gl, 2)

    peak = capital
    max_dd = 0.0
    for point in equity_curve:
        e = point["equity"]
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)

    return VariantResult(
        name=name,
        trades=trades,
        total_return_pct=round((final_equity - capital) / capital * 100, 2),
        win_rate=wr,
        profit_factor=pf,
        max_dd_pct=round(max_dd, 2),
        total_trades=len(trades),
        exit_breakdown=dict(Counter(t.exit_reason for t in trades)),
        monthly=dict(monthly),
        worst_month=worst_month,
        worst_month_pnl=round(worst_pnl, 2),
        stage_exit_pnl=round(stage_pnl, 2),
        final_equity=round(final_equity, 2),
    )


def collect_signals(
    symbols,
    frames,
    nifty_weekly,
    start_ts,
    end_ts,
    min_quality: int,
    market_filter: bool,
):
    weekly_by_sym = {}
    signals_by_day = defaultdict(list)
    all_n = 0
    for symbol, daily in frames.items():
        weekly = add_weekly_indicators(daily_to_weekly(daily))
        if len(weekly) < 40:
            continue
        weekly_by_sym[symbol] = weekly
        bench = nifty_weekly if not nifty_weekly.empty else weekly
        sigs = _collect_stage2_signals(
            symbol,
            weekly,
            daily,
            bench,
            nifty_weekly,
            start_ts,
            end_ts,
            min_quality_score=min_quality,
            market_filter=market_filter,
        )
        all_n += len(sigs)
        for sig in sigs:
            signals_by_day[sig["entry_day"]].append(sig)
    return weekly_by_sym, signals_by_day, all_n


def main():
    end = date.today()
    start = end - timedelta(days=365)
    capital = 1_000_000.0
    config = StrategyConfig.get_active()
    risk_pct = float(getattr(config, "risk_pct", 2.0) or 2.0)

    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    print(f"Loading frames for {len(symbols)} symbols...")
    frames = _preload_frames(symbols)
    print(f"Loaded {len(frames)}")

    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = (
        add_weekly_indicators(daily_to_weekly(nifty_daily)) if not nifty_daily.empty else pd.DataFrame()
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    calendar_set = set()
    for df in frames.values():
        calendar_set.update(df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist())
    calendar = sorted(calendar_set)

    # Precompute signal sets for different entry filters
    print("Collecting signals (Q0 / Q75 / market filter)...")
    w0, s0, n0 = collect_signals(symbols, frames, nifty_weekly, start_ts, end_ts, 0, False)
    w75, s75, n75 = collect_signals(symbols, frames, nifty_weekly, start_ts, end_ts, 75, False)
    wm, sm, nm = collect_signals(symbols, frames, nifty_weekly, start_ts, end_ts, 0, True)
    w75m, s75m, n75m = collect_signals(symbols, frames, nifty_weekly, start_ts, end_ts, 75, True)
    print(f"Signals: Q0={n0} Q75={n75} mkt={nm} Q75+mkt={n75m}")

    variants = []

    def run(name, weekly, sigs, **kwargs):
        print(f"  sim: {name}...")
        res = simulate(
            name,
            frames,
            weekly,
            sigs,
            calendar,
            capital=capital,
            risk_pct=risk_pct,
            start_date=start,
            **kwargs,
        )
        variants.append(res)

    # Exit-focused (same entries Q0)
    run("A_baseline_stage3_4", w0, s0, stage_exit_mode="stage_3_4")
    run("B_no_stage_exit", w0, s0, stage_exit_mode="off")
    run("C_stage4_only", w0, s0, stage_exit_mode="stage_4_only")
    run("D_confirm_2_weeks", w0, s0, stage_exit_mode="confirm_2w")
    run("E_stage_below_MA", w0, s0, stage_exit_mode="stage_below_ma")
    run("F_trail_MA_no_stage", w0, s0, stage_exit_mode="off", trail_ma=True)
    run("G_tighten_stop_S3", w0, s0, stage_exit_mode="off", tighten_stop_on_stage3=True)
    run("H_trail_MA_plus_S4", w0, s0, stage_exit_mode="stage_4_only", trail_ma=True)

    # Entry filters with baseline exit
    run("I_Q75_baseline_exit", w75, s75, stage_exit_mode="stage_3_4")
    run("J_mkt_filter_baseline", wm, sm, stage_exit_mode="stage_3_4")
    run("K_Q75_mkt_baseline", w75m, s75m, stage_exit_mode="stage_3_4")

    # Best combos: good entry + soft exit
    run("L_Q75_mkt_no_stage", w75m, s75m, stage_exit_mode="off")
    run("M_Q75_mkt_S4_only", w75m, s75m, stage_exit_mode="stage_4_only")
    run("N_Q75_mkt_trail_no_stage", w75m, s75m, stage_exit_mode="off", trail_ma=True)
    run("O_Q75_confirm2w", w75, s75, stage_exit_mode="confirm_2w")
    run("P_Q75_mkt_confirm2w", w75m, s75m, stage_exit_mode="confirm_2w")

    print("\n" + "=" * 100)
    print(
        f"{'Variant':<32} {'Ret%':>7} {'WR%':>6} {'PF':>5} {'DD%':>6} "
        f"{'N':>4} {'WorstMonth':>10} {'WorstPnL':>10} {'StageExitPnL':>12}  exits"
    )
    print("=" * 100)
    for v in sorted(variants, key=lambda x: x.total_return_pct, reverse=True):
        print(
            f"{v.name:<32} {v.total_return_pct:>7.2f} {v.win_rate:>6.1f} {v.profit_factor:>5.2f} "
            f"{v.max_dd_pct:>6.2f} {v.total_trades:>4} {v.worst_month:>10} "
            f"{v.worst_month_pnl:>10,.0f} {v.stage_exit_pnl:>12,.0f}  {v.exit_breakdown}"
        )

    print("\n=== TOP 5 BY RETURN ===")
    for v in sorted(variants, key=lambda x: x.total_return_pct, reverse=True)[:5]:
        print(f"\n{v.name}: +{v.total_return_pct}% | WR {v.win_rate}% | PF {v.profit_factor} | DD {v.max_dd_pct}%")
        print(f"  exits: {v.exit_breakdown}")
        months = sorted(v.monthly.items())
        print("  monthly:", ", ".join(f"{m}:{p:+,.0f}" for m, p in months))

    print("\n=== TOP 5 BY PROFIT FACTOR (min 20 trades) ===")
    for v in sorted(
        [x for x in variants if x.total_trades >= 20],
        key=lambda x: x.profit_factor,
        reverse=True,
    )[:5]:
        print(
            f"{v.name}: PF {v.profit_factor} | ret {v.total_return_pct}% | "
            f"WR {v.win_rate}% | DD {v.max_dd_pct}% | n={v.total_trades}"
        )

    print("\n=== BEST RISK-ADJUSTED (return / max(DD,1)) ===")
    for v in sorted(variants, key=lambda x: x.total_return_pct / max(x.max_dd_pct, 1), reverse=True)[:5]:
        ratio = v.total_return_pct / max(v.max_dd_pct, 1)
        print(
            f"{v.name}: ratio {ratio:.2f} | ret {v.total_return_pct}% | DD {v.max_dd_pct}% | PF {v.profit_factor}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
