"""
Stage 2.0 last-1-year: take every trade with NO free-cash skips.

Compares:
  A) Cash-constrained (₹10L shared pool — current model)
  B) Unlimited cash (same rules; never skip for cash)
  C) Independent trades: every signal sized at 2% risk of ₹10L,
     P&L summed as if each trade is standalone (max theoretical trade set)
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
    EXIT_STAGE_4_ONLY,
    MAX_HOLD_DAYS,
    StageV2Trade,
    _OpenPos,
    _close_trade,
    _collect_stage2_signals,
    _preload_frames,
    _weekly_stage_at,
    run_stage_v2_backtest,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size


def summarize(label: str, r, ref_capital: float | None = None) -> None:
    cap = ref_capital if ref_capital is not None else r.capital
    print(f"\n=== {label} ===")
    print(f"  Capital model base: Rs {r.capital:,.0f}")
    if ref_capital is not None:
        print(f"  Return shown vs ref capital: Rs {ref_capital:,.0f}")
    print(f"  Exit mode:          {r.exit_mode} ({r.exit_mode_label})")
    print(f"  Stocks scanned:     {r.stocks_scanned}")
    print(f"  Stage 2 signals:    {r.stage2_entries}")
    print(f"  Trades executed:    {r.total_trades}")
    print(f"  Skipped (no cash):  {r.signals_skipped_cash}")
    print(f"  Peak parallel:      {r.peak_parallel}")
    print(f"  Win rate:           {r.win_rate}%")
    print(f"  Profit factor:      {r.profit_factor}")
    print(f"  Max drawdown:       {r.max_drawdown_pct}%")
    print(f"  Avg R:              {r.avg_rr}")
    print(f"  Avg hold days:      {r.avg_hold_days}")
    print(f"  Exit breakdown:     {r.exit_breakdown}")
    if ref_capital is not None:
        ret = (r.final_cash - r.capital)  # absolute PnL same
        # Unlimited uses huge capital; report absolute PnL + return on ₹10L basis
        abs_pnl = r.final_cash - r.capital
        # When capital is huge, % on huge capital is tiny — recompute % on ref
        # Better: sum trade pnls
        trade_pnl = sum(t.pnl for t in r.trades)
        print(f"  Total trade P&L:    Rs {trade_pnl:+,.0f}")
        print(f"  Return on Rs10L:    {trade_pnl / ref_capital * 100:+.2f}%")
        print(f"  Final cash (raw):   Rs {r.final_cash:,.0f}")
    else:
        print(f"  Total return:       {r.total_return_pct}%")
        print(f"  Final cash:         Rs {r.final_cash:,.0f}")
        print(f"  Total trade P&L:    Rs {sum(t.pnl for t in r.trades):+,.0f}")

    monthly: dict[str, float] = defaultdict(float)
    for t in r.trades:
        monthly[t.exit_date[:7]] += t.pnl
    print("  Monthly P&L:")
    for m in sorted(monthly.keys()):
        p = monthly[m]
        sign = "+" if p >= 0 else ""
        print(f"    {m}: {sign}{p:,.0f}")


def independent_all_trades(
    frames: dict[str, pd.DataFrame],
    weekly_by_sym: dict[str, pd.DataFrame],
    signals_by_day: dict,
    calendar: list,
    *,
    risk_capital: float,
    risk_pct: float,
    start_date: date,
    exit_mode: str = EXIT_STAGE_4_ONLY,
    one_per_symbol: bool = True,
    use_cooldown: bool = True,
) -> dict:
    """
    Take signals with no cash constraint.
      - each trade sized at risk_pct of risk_capital (fixed ₹10L risk base)
      - one_per_symbol: at most one open position per name
      - use_cooldown: COOLDOWN_DAYS after exit before re-entry
      - if both False: literally every Stage 2 signal is a separate paper trade
    """
    # opens: symbol -> pos  OR trade_id -> pos when stacking allowed
    opens: dict[str, _OpenPos] = {}
    last_exit: dict[str, pd.Timestamp] = {}
    trades: list[StageV2Trade] = []
    skipped_open = 0
    skipped_cooldown = 0
    taken = 0
    peak_parallel = 0
    trade_seq = 0

    trail_ma = exit_mode == "trail_ma_s4"

    def _pos_key(sym: str) -> str:
        nonlocal trade_seq
        if one_per_symbol:
            return sym
        trade_seq += 1
        return f"{sym}#{trade_seq}"

    for ts in calendar:
        closed = []
        for key, pos in list(opens.items()):
            sym = pos.symbol
            df = frames.get(sym)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            pos.hold_days += 1

            stage_now = None
            ma_now = None
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
                        new_stop = round(ma_now * 0.98, 2)
                        if new_stop > pos.stop and new_stop < close:
                            pos.stop = new_stop

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
            elif stage_now is not None:
                if exit_mode == "stage_3_4" and stage_now in (3, 4):
                    exit_price = close
                    exit_reason = "stage_exit"
                elif exit_mode in ("stage_4_only", "trail_ma_s4") and stage_now == 4:
                    exit_price = close
                    exit_reason = "stage_exit"

            if exit_price is None:
                continue
            trade = _close_trade(
                pos, exit_price=exit_price, exit_ts=ts,
                exit_reason=exit_reason, days_held=pos.hold_days,
            )
            trades.append(trade)
            last_exit[sym] = ts
            closed.append(key)

        for key in closed:
            opens.pop(key, None)

        day_sigs = signals_by_day.get(ts, [])
        if not day_sigs:
            continue
        day_sigs = sorted(
            day_sigs,
            key=lambda s: (int(s.get("quality_score") or 0), float(s.get("rs_rating") or 0)),
            reverse=True,
        )
        for sig in day_sigs:
            sym = sig["symbol"]
            if one_per_symbol and any(p.symbol == sym for p in opens.values()):
                skipped_open += 1
                continue
            if sym not in frames or ts not in frames[sym].index:
                continue
            if use_cooldown:
                prev_x = last_exit.get(sym)
                if prev_x is not None and (ts - prev_x).days < COOLDOWN_DAYS:
                    skipped_cooldown += 1
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

            pos_size = calculate_position_size(risk_capital, risk_pct, entry_price, stop)
            qty = int(pos_size.quantity)
            if qty <= 0:
                continue
            notional = qty * entry_price
            taken += 1
            key = _pos_key(sym)
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
                cash_available=0.0,
                total_invested=notional,
                parallel_open=len(opens) + 1,
                equity_at_entry=risk_capital,
            )
            opens[key] = pos
            peak_parallel = max(peak_parallel, len(opens))

            if low <= stop:
                trade = _close_trade(
                    pos, exit_price=stop, exit_ts=ts, exit_reason="stop_loss", days_held=0
                )
                trades.append(trade)
                last_exit[sym] = ts
                opens.pop(key, None)
            elif high >= target:
                trade = _close_trade(
                    pos, exit_price=target, exit_ts=ts, exit_reason="target_2.5r", days_held=0
                )
                trades.append(trade)
                last_exit[sym] = ts
                opens.pop(key, None)

    if opens:
        last_ts = calendar[-1]
        for key, pos in list(opens.items()):
            df = frames.get(pos.symbol)
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
            trades.append(trade)

    trades.sort(key=lambda t: (t.entry_date, t.symbol))
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1e-9
    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    pf = gp / gl if trades else 0

    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        monthly[t.exit_date[:7]] += t.pnl

    by_exit: dict[str, float] = defaultdict(float)
    for t in trades:
        by_exit[t.exit_date] += t.pnl
    running = risk_capital
    peak = risk_capital
    max_dd = 0.0
    for d in sorted(by_exit.keys()):
        running += by_exit[d]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)

    # Capital required if all peak parallel notionals held at once (avg)
    avg_notional = (
        sum(t.capital_invested for t in trades) / len(trades) if trades else 0
    )

    return {
        "trades": trades,
        "total_trades": len(trades),
        "taken_signals": taken,
        "skipped_open": skipped_open,
        "skipped_cooldown": skipped_cooldown,
        "peak_parallel": peak_parallel,
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / risk_capital * 100, 2),
        "max_dd_approx": round(max_dd, 2),
        "avg_rr": round(sum(t.rr_achieved for t in trades) / len(trades), 2) if trades else 0,
        "avg_hold": round(sum(t.days_held for t in trades) / len(trades), 1) if trades else 0,
        "exit_breakdown": dict(Counter(t.exit_reason for t in trades)),
        "monthly": dict(monthly),
        "gross_profit": round(gp, 2),
        "gross_loss": round(sum(t.pnl for t in losses), 2) if losses else 0.0,
        "wins": len(wins),
        "losses": len(losses),
        "avg_notional": round(avg_notional, 2),
        "est_capital_at_peak": round(avg_notional * peak_parallel, 2),
    }


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)
    ref_cap = 1_000_000.0
    exit_mode = EXIT_STAGE_4_ONLY  # current recommended default

    print(f"Period: {start} → {end}")
    print(f"Exit mode: {exit_mode}")
    print(f"Universe: Nifty 200")
    print("=" * 70)

    # A) Cash constrained ₹10L
    print("\nRunning A: cash-constrained Rs 10L...")
    a = run_stage_v2_backtest(
        start_date=start,
        end_date=end,
        capital=ref_cap,
        min_quality_score=0,
        market_filter=False,
        exit_mode=exit_mode,
    )
    summarize("A) CASH-CONSTRAINED (Rs 10L shared pool)", a)

    # Shared signal collection for unconstrained runs
    print("\nCollecting all Stage 2 signals...")
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    nifty_daily = load_price_dataframe(NIFTY50_SYMBOL)
    nifty_weekly = (
        add_weekly_indicators(daily_to_weekly(nifty_daily))
        if not nifty_daily.empty
        else pd.DataFrame()
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    weekly_by_sym = {}
    signals_by_day = defaultdict(list)
    n_sig = 0
    for symbol, daily in frames.items():
        weekly = add_weekly_indicators(daily_to_weekly(daily))
        if len(weekly) < 40:
            continue
        weekly_by_sym[symbol] = weekly
        bench = nifty_weekly if not nifty_weekly.empty else weekly
        sigs = _collect_stage2_signals(
            symbol, weekly, daily, bench, nifty_weekly,
            start_ts, end_ts, min_quality_score=0, market_filter=False,
        )
        n_sig += len(sigs)
        for sig in sigs:
            signals_by_day[sig["entry_day"]].append(sig)

    calendar_set = set()
    for df in frames.values():
        calendar_set.update(df.index[(df.index >= start_ts) & (df.index <= end_ts)].tolist())
    calendar = sorted(calendar_set)

    config = StrategyConfig.get_active()
    risk_pct = float(getattr(config, "risk_pct", 2.0) or 2.0)
    print(f"Signals={n_sig} | risk={risk_pct}% of Rs {ref_cap:,.0f} per trade")

    # B) No cash limit, still 1 position/name + cooldown (portfolio without cash cap)
    print("\nRunning B: no cash limit (1 pos/name + cooldown, fixed 2% risk of Rs10L)...")
    b = independent_all_trades(
        frames, weekly_by_sym, signals_by_day, calendar,
        risk_capital=ref_cap, risk_pct=risk_pct, start_date=start, exit_mode=exit_mode,
        one_per_symbol=True, use_cooldown=True,
    )

    # C) Literally every Stage 2 signal as its own trade (no cash / open / cooldown skips)
    print("\nRunning C: EVERY signal as independent trade (no skips at all)...")
    c = independent_all_trades(
        frames, weekly_by_sym, signals_by_day, calendar,
        risk_capital=ref_cap, risk_pct=risk_pct, start_date=start, exit_mode=exit_mode,
        one_per_symbol=False, use_cooldown=False,
    )

    def print_unconstrained(label: str, d: dict, n_signals: int) -> None:
        print(f"\n=== {label} ===")
        print(f"  Stage 2 signals found:  {n_signals}")
        print(f"  Trades executed:        {d['total_trades']}")
        print(f"  Skipped (already open): {d['skipped_open']}")
        print(f"  Skipped (cooldown):     {d['skipped_cooldown']}")
        print(f"  Peak parallel trades:   {d['peak_parallel']}")
        print(f"  Avg notional / trade:   Rs {d['avg_notional']:,.0f}")
        print(f"  Est. capital @ peak*:   Rs {d['est_capital_at_peak']:,.0f}")
        print(f"  Win rate:               {d['win_rate']}%  ({d['wins']}W / {d['losses']}L)")
        print(f"  Profit factor:          {d['profit_factor']}")
        print(f"  Avg R:                  {d['avg_rr']}")
        print(f"  Avg hold days:          {d['avg_hold']}")
        print(f"  Gross profit:           Rs {d['gross_profit']:+,.0f}")
        print(f"  Gross loss:             Rs {d['gross_loss']:+,.0f}")
        print(f"  TOTAL P&L:              Rs {d['total_pnl']:+,.0f}")
        print(f"  Return vs Rs10L:        {d['return_pct']:+.2f}%")
        print(f"  Max DD (cum. exits):    {d['max_dd_approx']}%")
        print(f"  Exit breakdown:         {d['exit_breakdown']}")
        print("  Monthly P&L:")
        for m in sorted(d["monthly"].keys()):
            p = d["monthly"][m]
            sign = "+" if p >= 0 else ""
            print(f"    {m}: {sign}{p:,.0f}")
        print("  (* est. capital = avg notional × peak parallel — rough funding need)")

    print_unconstrained(
        "B) NO CASH LIMIT — 1 position/name + cooldown (fixed 2% of Rs10L risk)",
        b, n_sig,
    )
    print_unconstrained(
        "C) EVERY SINGLE SIGNAL — no cash / open / cooldown skips (fixed 2% of Rs10L risk)",
        c, n_sig,
    )

    print("\n  Top 10 winners (C — every signal):")
    for t in sorted(c["trades"], key=lambda x: x.pnl, reverse=True)[:10]:
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date} "
            f"Q{t.quality_score} PnL {t.pnl:+,.0f} R={t.rr_achieved} {t.exit_reason}"
        )
    print("  Top 10 losers (C — every signal):")
    for t in sorted(c["trades"], key=lambda x: x.pnl)[:10]:
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date} "
            f"Q{t.quality_score} PnL {t.pnl:+,.0f} R={t.rr_achieved} {t.exit_reason}"
        )

    # Side-by-side
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE (Stage 4 only, last 1 year, risk sized on Rs 10L @ 2%)")
    print("=" * 78)
    a_pnl = sum(t.pnl for t in a.trades)
    print(
        f"{'Metric':<30} {'A Cash Rs10L':>14} {'B No cash cap':>14} {'C All signals':>14}"
    )
    print(f"{'Stage 2 signals':<30} {a.stage2_entries:>14} {n_sig:>14} {n_sig:>14}")
    print(
        f"{'Trades taken':<30} {a.total_trades:>14} {b['total_trades']:>14} {c['total_trades']:>14}"
    )
    print(
        f"{'Skipped no cash':<30} {a.signals_skipped_cash:>14} {'0':>14} {'0':>14}"
    )
    print(
        f"{'Peak parallel':<30} {a.peak_parallel:>14} {b['peak_parallel']:>14} {c['peak_parallel']:>14}"
    )
    print(
        f"{'Win rate %':<30} {a.win_rate:>14.2f} {b['win_rate']:>14.2f} {c['win_rate']:>14.2f}"
    )
    print(
        f"{'Profit factor':<30} {a.profit_factor:>14.2f} {b['profit_factor']:>14.2f} {c['profit_factor']:>14.2f}"
    )
    print(
        f"{'Total P&L Rs':<30} {a_pnl:>14,.0f} {b['total_pnl']:>14,.0f} {c['total_pnl']:>14,.0f}"
    )
    print(
        f"{'Return on Rs10L %':<30} {a.total_return_pct:>14.2f} {b['return_pct']:>14.2f} {c['return_pct']:>14.2f}"
    )
    print(
        f"{'Max DD %':<30} {a.max_drawdown_pct:>14.2f} {b['max_dd_approx']:>14.2f} {c['max_dd_approx']:>14.2f}"
    )
    print(
        f"{'Est. capital @ peak Rs':<30} {'~10L pool':>14} "
        f"{b['est_capital_at_peak']:>14,.0f} {c['est_capital_at_peak']:>14,.0f}"
    )
    print(
        "\nA = real portfolio with Rs10L cash pool (current app model)."
        "\nB = take every free name-slot (no cash skip); still 1 open/name + 40d cooldown."
        "\nC = paper-trade EVERY Stage 2 signal independently (true max trade set)."
        "\nP&L for B/C uses same Rs20k risk budget per trade (2% of 10L) — comparable."
        "\nReturn % on 10L for B/C is NOT achievable with only 10L cash when peak parallel is high."
    )
    print("Done.")


if __name__ == "__main__":
    main()
