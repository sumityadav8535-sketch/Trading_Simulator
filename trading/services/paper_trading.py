"""
Paper trading engine — fake-money forward test of live F&O strategy.

Mirrors real MIS futures behaviour:
  - Block margin on entry, release + P&L on exit
  - Adverse slippage on entry/exit
  - SL / target checked against bar high/low
  - Force exit near session end (15:15 IST)
  - Risk-based lot sizing from account equity
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, datetime, time as dt_time
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

from django.db import transaction
from django.db.models import Sum
from django.utils import timezone

from trading.models import PaperAccount, PaperEvent, PaperPosition, PaperTrade
from trading.services.fno_engine import (
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    SLIPPAGE_PTS,
    STRATEGY,
    lots_for_risk,
    max_lots,
)
from trading.services.fno_live import evaluate_signal, fetch_instrument_bars
from trading.services.intraday_data import get_market_status

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Brokerage + charges approx on index futures (round-trip), INR
BROKERAGE_RT_INR = 40.0


def _now_ist() -> datetime:
    return datetime.now(IST)


def _d(value) -> Decimal:
    return Decimal(str(round(float(value), 2)))


def _f(value) -> float:
    return float(value)


def log_event(
    account: PaperAccount,
    message: str,
    level: str = PaperEvent.LEVEL_INFO,
    payload: Optional[dict] = None,
) -> None:
    PaperEvent.objects.create(
        account=account,
        level=level,
        message=message,
        payload=payload or {},
    )


def get_or_create_account(
    starting_capital: float = 500_000,
    name: str = "Paper Account",
) -> PaperAccount:
    return PaperAccount.get_active()


def ensure_auto_trade(account: Optional[PaperAccount] = None) -> PaperAccount:
    """Turn auto-trade on so active F&O signals open paper positions."""
    account = account or PaperAccount.get_active()
    if not account.auto_trade:
        account.auto_trade = True
        account.save(update_fields=["auto_trade", "updated_at"])
        log_event(account, "Auto-trade enabled (F&O signals → paper orders)")
    return account


def reset_account(account: PaperAccount, capital: Optional[float] = None) -> PaperAccount:
    """Close open positions without P&L credit, wipe trades, reset cash."""
    with transaction.atomic():
        account.positions.filter(status=PaperPosition.STATUS_OPEN).update(
            status=PaperPosition.STATUS_CLOSED,
            notes="reset",
        )
        account.trades.all().delete()
        account.events.all().delete()
        cap = _d(capital if capital is not None else account.starting_capital)
        account.starting_capital = cap
        account.cash = cap
        account.margin_blocked = _d(0)
        account.realized_pnl = _d(0)
        account.peak_equity = cap
        # Keep auto-trade on so new signals still paper-fill after a reset
        account.auto_trade = True
        account.save()
        log_event(account, f"Account reset to ₹{float(cap):,.0f}", PaperEvent.LEVEL_INFO)
    return account


def set_capital(account: PaperAccount, capital: float) -> PaperAccount:
    """Set starting capital when no open positions (for new test run)."""
    if account.positions.filter(status=PaperPosition.STATUS_OPEN).exists():
        raise ValueError("Close open positions before changing capital")
    cap = _d(capital)
    if cap < 10_000:
        raise ValueError("Minimum capital is ₹10,000")
    # Adjust free cash by delta on equity base
    account.starting_capital = cap
    account.cash = cap - account.margin_blocked
    if account.cash < 0:
        account.cash = cap
        account.margin_blocked = _d(0)
    account.peak_equity = max(account.peak_equity, cap)
    account.save()
    log_event(account, f"Capital set to ₹{float(cap):,.0f}")
    return account


def deposit(account: PaperAccount, amount: float) -> PaperAccount:
    if amount <= 0:
        raise ValueError("Deposit must be positive")
    account.cash = _d(_f(account.cash) + amount)
    account.starting_capital = _d(_f(account.starting_capital) + amount)
    account.peak_equity = max(account.peak_equity, _d(_f(account.equity)))
    account.save()
    log_event(account, f"Deposited ₹{amount:,.0f}")
    return account


def account_equity(account: PaperAccount, open_mtm: float = 0.0) -> float:
    return _f(account.cash) + _f(account.margin_blocked) + open_mtm


def unrealized_mtm(pos: PaperPosition, ltp: float) -> float:
    entry = _f(pos.entry_price)
    qty = pos.quantity
    if pos.side == PaperPosition.SIDE_SHORT:
        return (entry - ltp) * qty
    return (ltp - entry) * qty


def _latest_bar(instrument: str) -> Optional[dict]:
    df = fetch_instrument_bars(instrument, force=False)
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    ts = df.index[-1]
    return {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "time": ts,
    }


def _apply_slippage(price: float, side: str, is_entry: bool) -> float:
    """Adverse fill: long pays more / short sells lower on entry; reverse on exit."""
    slip = SLIPPAGE_PTS
    if side == PaperPosition.SIDE_SHORT:
        if is_entry:
            return price - slip  # short entry: worse fill is lower sell
        return price + slip  # short exit: cover higher
    if is_entry:
        return price + slip
    return price - slip


def _trades_today(account: PaperAccount, session: date) -> int:
    return account.trades.filter(session_date=session).count()


def _open_for(account: PaperAccount, instrument: str) -> Optional[PaperPosition]:
    return (
        account.positions.filter(
            status=PaperPosition.STATUS_OPEN,
            instrument=instrument.upper(),
        )
        .order_by("-entry_time")
        .first()
    )


@transaction.atomic
def open_position_from_signal(
    account: PaperAccount,
    signal: dict,
    force: bool = False,
) -> Optional[PaperPosition]:
    """Place a paper order from an active F&O signal dict."""
    if signal.get("status") != "active" and not force:
        return None

    instrument = str(signal.get("instrument", "")).upper()
    if instrument not in INSTRUMENTS:
        return None
    if instrument not in (account.instruments or list(INSTRUMENTS.keys())):
        return None

    if _open_for(account, instrument):
        return None

    now = _now_ist()
    session = now.date()
    if _trades_today(account, session) >= account.max_trades_per_day:
        log_event(
            account,
            f"Skip {instrument}: max trades/day ({account.max_trades_per_day}) reached",
            PaperEvent.LEVEL_WARN,
        )
        return None

    inst = INSTRUMENTS[instrument]
    # Prefer backtest-aligned entry: next-bar open (entry_ref) over live mid-bar LTP
    ltp = float(
        signal.get("entry_ref")
        or signal.get("signal_close")
        or signal.get("ltp")
        or 0
    )
    stop = float(signal.get("stop") or 0)
    target = float(signal.get("target") or 0)
    risk_pts = float(signal.get("risk_pts") or 0)
    side = str(signal.get("side") or "SHORT").upper()
    if side not in (PaperPosition.SIDE_SHORT, PaperPosition.SIDE_LONG):
        side = PaperPosition.SIDE_SHORT

    if ltp <= 0 or stop <= 0 or target <= 0 or risk_pts <= 0:
        log_event(account, f"Skip {instrument}: incomplete signal levels", PaperEvent.LEVEL_WARN)
        return None

    equity = account_equity(account)
    risk_pct = float(account.risk_pct or RISK_PCT)
    max_l = max_lots(equity, inst["mis_margin"])
    lots = lots_for_risk(equity, risk_pct, risk_pts, inst["lot_size"], max_l)
    if lots <= 0:
        log_event(account, f"Skip {instrument}: lots=0 (margin/risk)", PaperEvent.LEVEL_WARN)
        return None

    margin_needed = lots * inst["mis_margin"]
    if _f(account.cash) < margin_needed:
        # Scale down lots to fit cash
        lots = int(_f(account.cash) // inst["mis_margin"])
        if lots <= 0:
            log_event(account, f"Skip {instrument}: insufficient cash for margin", PaperEvent.LEVEL_WARN)
            return None
        margin_needed = lots * inst["mis_margin"]

    fill = _apply_slippage(ltp, side, is_entry=True)
    # Recheck risk after slippage for short (entry lower → slightly more risk to stop)
    if side == PaperPosition.SIDE_SHORT:
        risk_pts = max(stop - fill, 0.01)
    else:
        risk_pts = max(fill - stop, 0.01)

    account.cash = _d(_f(account.cash) - margin_needed)
    account.margin_blocked = _d(_f(account.margin_blocked) + margin_needed)
    account.save(update_fields=["cash", "margin_blocked", "updated_at"])

    pos = PaperPosition.objects.create(
        account=account,
        instrument=instrument,
        side=side,
        lots=lots,
        lot_size=inst["lot_size"],
        entry_price=_d(fill),
        entry_time=timezone.now(),
        signal_bar_time=str(signal.get("bar_time") or ""),
        stop_loss=_d(stop),
        target=_d(target),
        margin_blocked=_d(margin_needed),
        risk_pts=risk_pts,
        ml_prob=signal.get("ml_prob"),
        strategy_name=str(
            (signal.get("strategy") or {}).get("name")
            or signal.get("strategy_name")
            or STRATEGY.get("name", "")
        ),
        status=PaperPosition.STATUS_OPEN,
        notes=str(signal.get("message") or "")[:255],
    )
    log_event(
        account,
        f"OPEN {side} {instrument} {lots} lots @ {fill:.2f} SL={stop:.2f} T={target:.2f}",
        PaperEvent.LEVEL_TRADE,
        {
            "position_id": pos.id,
            "instrument": instrument,
            "lots": lots,
            "entry": fill,
            "stop": stop,
            "target": target,
            "margin": margin_needed,
            "ml_prob": signal.get("ml_prob"),
        },
    )
    return pos


@transaction.atomic
def close_position(
    pos: PaperPosition,
    exit_price: float,
    reason: str,
    exit_time: Optional[datetime] = None,
) -> PaperTrade:
    """Close open position, release margin, book P&L."""
    if pos.status != PaperPosition.STATUS_OPEN:
        raise ValueError("Position already closed")

    account = PaperAccount.objects.select_for_update().get(pk=pos.account_id)
    fill = _apply_slippage(float(exit_price), pos.side, is_entry=False)
    entry = _f(pos.entry_price)
    qty = pos.quantity

    if pos.side == PaperPosition.SIDE_SHORT:
        pnl_pts = entry - fill
    else:
        pnl_pts = fill - entry
    raw_pnl = pnl_pts * qty
    pnl = raw_pnl - BROKERAGE_RT_INR

    margin = _f(pos.margin_blocked)
    account.margin_blocked = _d(max(_f(account.margin_blocked) - margin, 0))
    account.cash = _d(_f(account.cash) + margin + pnl)
    account.realized_pnl = _d(_f(account.realized_pnl) + pnl)
    eq = account_equity(account)
    if eq > _f(account.peak_equity):
        account.peak_equity = _d(eq)
    account.save()

    pos.status = PaperPosition.STATUS_CLOSED
    pos.notes = (pos.notes + f" | exit:{reason}").strip(" |")[:255]
    pos.save(update_fields=["status", "notes"])

    risk = float(pos.risk_pts) or 1.0
    r_mult = pnl_pts / risk if risk else 0.0
    when = exit_time or timezone.now()
    if timezone.is_naive(when):
        when = timezone.make_aware(when, IST)

    trade = PaperTrade.objects.create(
        account=account,
        position=pos,
        instrument=pos.instrument,
        side=pos.side,
        lots=pos.lots,
        lot_size=pos.lot_size,
        entry_price=pos.entry_price,
        exit_price=_d(fill),
        entry_time=pos.entry_time,
        exit_time=when,
        stop_loss=pos.stop_loss,
        target=pos.target,
        exit_reason=reason,
        pnl=_d(pnl),
        pnl_pts=round(pnl_pts, 2),
        r_multiple=round(r_mult, 3),
        risk_pts=pos.risk_pts,
        ml_prob=pos.ml_prob,
        strategy_name=pos.strategy_name,
        session_date=_now_ist().date(),
    )
    log_event(
        account,
        f"CLOSE {pos.side} {pos.instrument} @ {fill:.2f} ({reason}) PnL ₹{pnl:,.0f}",
        PaperEvent.LEVEL_TRADE,
        {
            "trade_id": trade.id,
            "position_id": pos.id,
            "exit": fill,
            "reason": reason,
            "pnl": pnl,
            "r": r_mult,
        },
    )
    return trade


def _check_exit(pos: PaperPosition, bar: dict, now: datetime) -> Optional[tuple[float, str]]:
    """Return (exit_price, reason) if position should close on this bar."""
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    stop = _f(pos.stop_loss)
    target = _f(pos.target)
    t = now.time()

    # Force exit after 15:15 IST
    if t >= FORCE_EXIT:
        return close, PaperTrade.EXIT_FORCE

    if pos.side == PaperPosition.SIDE_SHORT:
        # Stop hit first if both in same bar (conservative)
        if high >= stop:
            return stop, PaperTrade.EXIT_SL
        if low <= target:
            return target, PaperTrade.EXIT_TARGET
    else:
        if low <= stop:
            return stop, PaperTrade.EXIT_SL
        if high >= target:
            return target, PaperTrade.EXIT_TARGET
    return None


def manage_open_positions(account: PaperAccount) -> list[PaperTrade]:
    closed: list[PaperTrade] = []
    now = _now_ist()
    opens = list(account.positions.filter(status=PaperPosition.STATUS_OPEN))
    for pos in opens:
        bar = _latest_bar(pos.instrument)
        if not bar:
            continue
        decision = _check_exit(pos, bar, now)
        if decision:
            price, reason = decision
            try:
                trade = close_position(pos, price, reason, exit_time=now)
                closed.append(trade)
            except Exception as exc:
                logger.exception("Failed to close paper position %s", pos.id)
                log_event(account, f"Close error {pos.instrument}: {exc}", PaperEvent.LEVEL_ERROR)
    return closed


def try_entries(account: PaperAccount, force_refresh: bool = False) -> list[PaperPosition]:
    """Evaluate strategy on each instrument and open paper positions if active."""
    if not account.auto_trade:
        return []

    now = _now_ist()
    market = get_market_status(now)
    if not market.is_open:
        return []

    t = now.time()
    if t < MARKET_OPEN or t > NO_ENTRY_AFTER:
        return []

    opened: list[PaperPosition] = []
    instruments = account.instruments or list(INSTRUMENTS.keys())
    for key in instruments:
        key = str(key).upper()
        if key not in INSTRUMENTS:
            continue
        if _open_for(account, key):
            continue
        try:
            signal = evaluate_signal(key, force=force_refresh)
        except Exception as exc:
            logger.warning("Signal eval failed %s: %s", key, exc)
            log_event(account, f"Signal error {key}: {exc}", PaperEvent.LEVEL_ERROR)
            continue

        # Avoid re-entering on same signal bar
        last_open = (
            account.positions.filter(instrument=key)
            .order_by("-entry_time")
            .first()
        )
        bar_time = str(signal.get("bar_time") or "")
        if (
            last_open
            and last_open.signal_bar_time
            and last_open.signal_bar_time == bar_time
            and last_open.entry_time.date() == now.date()
        ):
            continue

        pos = open_position_from_signal(account, signal)
        if pos:
            opened.append(pos)
    return opened


def run_tick(
    account: Optional[PaperAccount] = None,
    force_refresh: bool = False,
    *,
    ensure_auto: bool = False,
) -> dict[str, Any]:
    """
    One engine cycle: manage exits, then try new entries.
    Call every 30–60s during market hours (browser poll or management command).

    ensure_auto=True forces auto_trade on (used from F&O Live so signals always
    land in the paper section without a manual toggle).
    """
    account = account or PaperAccount.get_active()
    if ensure_auto:
        account = ensure_auto_trade(account)
    market = get_market_status()

    closed = manage_open_positions(account)
    opened: list[PaperPosition] = []
    if account.auto_trade and market.is_open:
        opened = try_entries(account, force_refresh=force_refresh)

    account.last_tick_at = timezone.now()
    account.save(update_fields=["last_tick_at", "updated_at"])

    return {
        "account_id": account.id,
        "closed": len(closed),
        "opened": len(opened),
        "opened_ids": [p.id for p in opened],
        "auto_trade": account.auto_trade,
        "market": asdict(market),
        "cash": _f(account.cash),
        "margin_blocked": _f(account.margin_blocked),
        "realized_pnl": _f(account.realized_pnl),
        "tick_at": account.last_tick_at.isoformat() if account.last_tick_at else None,
    }


def position_snapshot(pos: PaperPosition) -> dict:
    bar = _latest_bar(pos.instrument)
    ltp = float(bar["close"]) if bar else _f(pos.entry_price)
    mtm = unrealized_mtm(pos, ltp)
    return {
        "id": pos.id,
        "instrument": pos.instrument,
        "side": pos.side,
        "lots": pos.lots,
        "lot_size": pos.lot_size,
        "quantity": pos.quantity,
        "entry_price": _f(pos.entry_price),
        "entry_time": pos.entry_time.astimezone(IST).strftime("%Y-%m-%d %H:%M") if pos.entry_time else "",
        "stop_loss": _f(pos.stop_loss),
        "target": _f(pos.target),
        "risk_pts": pos.risk_pts,
        "ml_prob": pos.ml_prob,
        "strategy_name": pos.strategy_name,
        "margin_blocked": _f(pos.margin_blocked),
        "ltp": round(ltp, 2),
        "mtm": round(mtm, 2),
        "notes": pos.notes,
        "signal_bar_time": pos.signal_bar_time,
    }


def trade_snapshot(t: PaperTrade) -> dict:
    return {
        "id": t.id,
        "instrument": t.instrument,
        "side": t.side,
        "lots": t.lots,
        "entry_price": _f(t.entry_price),
        "exit_price": _f(t.exit_price),
        "entry_time": t.entry_time.astimezone(IST).strftime("%Y-%m-%d %H:%M") if t.entry_time else "",
        "exit_time": t.exit_time.astimezone(IST).strftime("%Y-%m-%d %H:%M") if t.exit_time else "",
        "exit_reason": t.exit_reason,
        "pnl": _f(t.pnl),
        "pnl_pts": t.pnl_pts,
        "r_multiple": t.r_multiple,
        "ml_prob": t.ml_prob,
        "session_date": t.session_date.isoformat() if t.session_date else "",
        "strategy_name": t.strategy_name,
    }


def get_dashboard(account: Optional[PaperAccount] = None) -> dict:
    account = account or PaperAccount.get_active()
    opens = list(account.positions.filter(status=PaperPosition.STATUS_OPEN))
    open_snaps = [position_snapshot(p) for p in opens]
    open_mtm = sum(s["mtm"] for s in open_snaps)
    equity = account_equity(account, open_mtm)
    peak = max(_f(account.peak_equity), equity)
    dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0

    trades_qs = account.trades.all()
    trades = [trade_snapshot(t) for t in trades_qs[:100]]
    n = trades_qs.count()
    wins = trades_qs.filter(pnl__gt=0).count()
    losses = trades_qs.filter(pnl__lte=0).count()
    total_pnl = _f(trades_qs.aggregate(s=Sum("pnl"))["s"] or 0)
    win_rate = (wins / n * 100) if n else 0.0
    gp = _f(trades_qs.filter(pnl__gt=0).aggregate(s=Sum("pnl"))["s"] or 0)
    gl = abs(_f(trades_qs.filter(pnl__lte=0).aggregate(s=Sum("pnl"))["s"] or 0))
    pf = (gp / gl) if gl > 0 else (gp if gp > 0 else 0.0)

    # Equity curve from closed trades (running realized + starting)
    curve = []
    running = _f(account.starting_capital)
    curve.append({"t": account.created_at.astimezone(IST).strftime("%Y-%m-%d"), "equity": round(running, 2)})
    for t in trades_qs.order_by("exit_time"):
        running += _f(t.pnl)
        curve.append({
            "t": t.exit_time.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
            "equity": round(running, 2),
        })
    if open_mtm:
        curve.append({"t": "now", "equity": round(_f(account.starting_capital) + total_pnl + open_mtm, 2)})

    events = [
        {
            "level": e.level,
            "message": e.message,
            "created_at": e.created_at.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for e in account.events.all()[:40]
    ]

    today = _now_ist().date()
    today_trades = _trades_today(account, today)
    market = get_market_status()

    ret_pct = ((equity - _f(account.starting_capital)) / _f(account.starting_capital) * 100) if account.starting_capital else 0

    return {
        "account": {
            "id": account.id,
            "name": account.name,
            "auto_trade": account.auto_trade,
            "starting_capital": _f(account.starting_capital),
            "cash": _f(account.cash),
            "margin_blocked": _f(account.margin_blocked),
            "equity": round(equity, 2),
            "open_mtm": round(open_mtm, 2),
            "realized_pnl": _f(account.realized_pnl),
            "peak_equity": _f(account.peak_equity),
            "drawdown_pct": round(dd_pct, 2),
            "return_pct": round(ret_pct, 2),
            "risk_pct": account.risk_pct,
            "max_trades_per_day": account.max_trades_per_day,
            "instruments": account.instruments or ["NIFTY", "BANKNIFTY"],
            "last_tick_at": account.last_tick_at.astimezone(IST).strftime("%H:%M:%S IST")
            if account.last_tick_at
            else None,
            "created_at": account.created_at.astimezone(IST).strftime("%Y-%m-%d"),
        },
        "stats": {
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "total_pnl": round(total_pnl, 2),
            "today_trades": today_trades,
        },
        "open_positions": open_snaps,
        "trades": trades,
        "equity_curve": curve,
        "events": events,
        "market": asdict(market),
        "strategy_name": STRATEGY.get("name", "Elite ML Short v2"),
    }
