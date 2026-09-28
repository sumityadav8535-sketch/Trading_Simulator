"""Adapter: RS Pullback Swing → StageV2BacktestResult for the 2.0 backtest page."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stage_analysis_v2.services.backtester import (
    EntryFilters,
    StageV2BacktestResult,
    StageV2Trade,
    _compute_monthly_returns,
    attach_signal_log_from_raw,
    fill_performance_metrics,
)
from stage_analysis_v2.services.strategy_catalog import (
    DEFAULT_RS_COOLDOWN_DAYS,
    DEFAULT_RS_MAX_HOLD_DAYS,
    DEFAULT_RS_MAX_NEW,
    DEFAULT_RS_MAX_OPEN,
    DEFAULT_RS_MAX_POS_PCT,
    DEFAULT_RS_RISK_PCT,
    DEFAULT_RS_TARGET_RR,
    STRATEGY_LABELS,
    STRATEGY_RS_PULLBACK,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.short_swing import (
    BarPack,
    F_QULLA,
    F_RS63,
    SwingParams,
    collect_signals,
    preload_packs,
    simulate,
    slice_cal,
)

RS_TRADE_FLAGS = F_QULLA | F_RS63


def _pos_frac(max_pos_pct: float) -> float:
    val = float(max_pos_pct)
    if val > 1.5:
        val = val / 100.0
    return min(1.0, max(0.05, val))


def run_rs_pullback_backtest(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    capital: float = 1_000_000.0,
    strategy_id: str = STRATEGY_RS_PULLBACK,
    risk_pct: float | None = None,
    max_hold_days: int | None = None,
    cooldown_days: int | None = None,
    max_pos_pct: float | None = None,
    target_rr: float | None = None,
    entry_filters: EntryFilters | None = None,
) -> StageV2BacktestResult:
    """EMA20 pullback in RS leaders. Signal T close, enter T+1 open."""
    if risk_pct is None:
        risk_pct = DEFAULT_RS_RISK_PCT
    risk_pct = min(50.0, max(0.1, float(risk_pct)))
    if max_hold_days is None:
        max_hold_days = DEFAULT_RS_MAX_HOLD_DAYS
    max_hold_days = min(60, max(3, int(max_hold_days)))
    if cooldown_days is None:
        cooldown_days = DEFAULT_RS_COOLDOWN_DAYS
    cooldown_days = min(365, max(0, int(cooldown_days)))
    if max_pos_pct is None:
        max_pos_pct = DEFAULT_RS_MAX_POS_PCT
    max_pos_pct = min(100.0, max(5.0, float(max_pos_pct)))
    if target_rr is None:
        target_rr = DEFAULT_RS_TARGET_RR
    target_rr = min(8.0, max(0.5, float(target_rr)))
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != NIFTY50_SYMBOL]
    filters = entry_filters or EntryFilters()

    params = SwingParams(
        name=STRATEGY_LABELS.get(strategy_id, "RS Pullback Swing"),
        entry="ema20pb",
        filter_name="qulla_rs",
        flags=RS_TRADE_FLAGS,
        exit_mode="chandelier",
        max_hold=max_hold_days,
        risk_pct=risk_pct,
        max_open=DEFAULT_RS_MAX_OPEN,
        max_new=DEFAULT_RS_MAX_NEW,
        max_pos_pct=_pos_frac(max_pos_pct),
        target_rr=target_rr,
        cooldown=cooldown_days,
    )

    packs, nifty, calendar = preload_packs()
    missing = [s for s in symbols if s not in packs]
    for sym in missing:
        df = load_price_dataframe(sym)
        if df.empty or len(df) < 120:
            continue
        packs[sym] = BarPack(sym, df)
    wanted = set(symbols)
    packs = {s: p for s, p in packs.items() if s in wanted}

    cal = slice_cal(calendar, start_date, end_date)
    sig_start = pd.Timestamp(start_date - timedelta(days=5))
    sigs = collect_signals(packs, nifty, sig_start, pd.Timestamp(end_date), entry=params.entry)
    raw = simulate(packs, cal, sigs, params, capital=capital, flatten=True, keep_trades=True)

    trades: list[StageV2Trade] = []
    for t in raw.get("holdings_history") or []:
        entry_px = float(t.get("entry_px") or 0)
        exit_px = float(t.get("exit_px") or 0)
        stop = float(t.get("stop") or 0)
        target = float(t.get("target") or 0)
        risk = entry_px - stop if entry_px > stop > 0 else 0.0
        rr = (exit_px - entry_px) / risk if risk > 0 else 0.0
        reason = str(t.get("reason") or "")
        if reason == "stop":
            reason = "stop_loss"
        elif reason == "target":
            reason = f"target_{target_rr:g}r"
        elif reason == "time":
            reason = "time_exit"
        elif reason == "eod":
            reason = "eod_force"
        trades.append(StageV2Trade(
            symbol=str(t.get("symbol") or ""),
            signal_date=str(t.get("signal_date") or t.get("entry") or ""),
            entry_date=str(t.get("entry") or ""),
            exit_date=str(t.get("exit") or ""),
            entry_price=entry_px,
            exit_price=exit_px,
            stop_loss=stop,
            target=target,
            quantity=int(t.get("qty") or 0),
            pnl=float(t.get("pnl") or 0),
            pnl_pct=float(t.get("pnl_pct") or 0),
            rr_achieved=round(rr, 2),
            exit_reason=reason,
            quality_score=0,
            rs_rating=round(float(t.get("score") or 0) * 100.0, 1) if t.get("score") else 0.0,
            weekly_stage=2,
            days_held=int(t.get("hold") or 0),
        ))

    result = StageV2BacktestResult(
        strategy_name=STRATEGY_LABELS.get(strategy_id, "RS Pullback Swing"),
        symbols=list(packs.keys()),
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=0,
        market_filter=False,
        exit_mode="chandelier",
        exit_mode_label=f"2.5 ATR trail · {target_rr:g}R · hold≤{max_hold_days}d",
        tech_filter="rs_pullback",
        tech_filter_label="EMA20 pullback · leader (beats Nifty 3m, tight range, ATR)",
        entry_stage=0,
        entry_on="next_open",
        entry_filters_label=filters.active_summary(),
        target_rr=target_rr,
        max_hold_days=max_hold_days,
        stop_ma_mult=1.0,
        trail_ma_mult=1.0,
        risk_pct=risk_pct,
        cooldown_days=cooldown_days,
        strategy_id=strategy_id,
        max_pos_pct=max_pos_pct,
        trades=trades,
        equity_curve=raw.get("equity_curve") or [],
        monthly_returns=_compute_monthly_returns(trades, capital),
        exit_breakdown=raw.get("exits") or {},
        total_trades=int(raw.get("trades") or len(trades)),
        total_signals=int(raw.get("ranked_signals") or 0),
        win_rate=float(raw.get("win_rate") or 0),
        profit_factor=float(raw.get("profit_factor") or 0),
        max_drawdown_pct=float(raw.get("max_drawdown_pct") or 0),
        avg_hold_days=float(raw.get("avg_hold") or 0),
        total_return_pct=float(raw.get("total_return_pct") or 0),
        peak_parallel=int(raw.get("peak_parallel") or 0),
        stocks_scanned=len(packs),
        stage2_entries=int(raw.get("ranked_signals") or 0),
        signals_skipped_cash=int(raw.get("signals_skipped_cash") or 0),
        final_cash=float(raw.get("final_equity") or capital),
        shared_capital=True,
    )
    if trades:
        result.avg_rr = round(sum(t.rr_achieved for t in trades) / len(trades), 2)
        result.expectancy_r = result.avg_rr
    raw_signals = []
    for s in raw.get("attempted") or []:
        sig_ts = s.get("sig_ts")
        entry_ts = s.get("entry_ts")
        raw_signals.append({
            "symbol": s.get("symbol"),
            "signal_date": str(pd.Timestamp(sig_ts).date()) if sig_ts is not None else "",
            "entry_date": str(pd.Timestamp(entry_ts).date()) if entry_ts is not None else "",
            "signal_close": round(float(s.get("close") or 0), 2),
            "stop_loss": round(float(s.get("stop_ref") or 0), 2),
            "target": 0.0,
            "quality_score": 0,
            "rs_rating": round(float(s.get("score") or 0) * 100.0, 1),
            "status": s.get("status") or "",
        })
    attach_signal_log_from_raw(result, raw_signals)
    fill_performance_metrics(result)
    return result
