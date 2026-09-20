"""
On localhost (DEBUG), fetch remaining 5m history through today and scan
F&O Live, F&O Long, and Gap Open for today / previous session / last signal.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from django.conf import settings
from django.utils import timezone

from trading.services.fno_engine import INSTRUMENTS
from trading.services.intraday_data import get_market_status
from trading.services.intraday_history_sync import (
    get_history_coverage,
    get_sync_status,
    history_needs_update,
    run_intraday_history_sync,
)

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
LOOKBACK_SESSIONS = 10

_lock = threading.Lock()
_status_lock = threading.Lock()
_started = False
_status: dict[str, Any] = {
    "enabled": False,
    "running": False,
    "phase": "idle",
    "message": "Idle",
    "started_at": None,
    "finished_at": None,
    "target_date": None,
    "fetch": {},
    "signals": {},
    "error": None,
    "skipped_fetch": False,
}


def _now_ist() -> datetime:
    return datetime.now(IST)


def is_auto_refresh_enabled() -> bool:
    if getattr(settings, "INTRADAY_AUTO_REFRESH_FORCE", False):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if any(arg == "test" or arg.endswith("test") for arg in sys.argv):
        return False
    if not getattr(settings, "DEBUG", False):
        return False
    return bool(getattr(settings, "INTRADAY_AUTO_REFRESH_ON_LOCALHOST", True))


def get_auto_refresh_status() -> dict[str, Any]:
    with _status_lock:
        snap = dict(_status)
        fetch = dict(snap.get("fetch") or {})
        signals = dict(snap.get("signals") or {})
    snap["enabled"] = is_auto_refresh_enabled()
    snap["fetch"] = fetch
    snap["signals"] = signals
    if snap.get("phase") == "fetching":
        snap["fetch"] = get_sync_status()
        snap["message"] = get_sync_status().get("message") or snap.get("message")
    snap["coverage"] = get_history_coverage()
    return snap


def reset_auto_refresh_for_tests() -> None:
    global _started
    with _lock:
        _started = False
        with _status_lock:
            _status.update({
                "enabled": False,
                "running": False,
                "phase": "idle",
                "message": "Idle",
                "started_at": None,
                "finished_at": None,
                "target_date": None,
                "fetch": {},
                "signals": {},
                "error": None,
                "skipped_fetch": False,
            })


def _set_status(**kwargs) -> None:
    with _status_lock:
        _status.update(kwargs)


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def previous_session_date(as_of: date, known: Optional[set[date]] = None) -> date:
    if known:
        prior = [d for d in known if d < as_of]
        if prior:
            return max(prior)
    d = as_of - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _session_dates_from_frame(df) -> list[date]:
    if df is None or getattr(df, "empty", True):
        return []
    try:
        return sorted({d for d in df.index.date})
    except Exception:
        return []


def _compact_trades(trades: list[dict], instrument: Optional[str] = None) -> list[dict]:
    out = []
    for t in trades or []:
        out.append({
            "instrument": t.get("instrument") or instrument,
            "symbol": t.get("symbol") or t.get("instrument") or instrument,
            "side": t.get("side"),
            "result": t.get("result"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "pnl_inr": t.get("pnl_inr") if t.get("pnl_inr") is not None else t.get("pnl"),
            "ml_prob": t.get("ml_prob"),
            "gap_pct": t.get("gap_pct"),
        })
    return out


def combine_fno_days(parts: list[dict], session: date, label: str) -> dict[str, Any]:
    trades: list[dict] = []
    bars = 0
    messages = []
    instruments = []
    for raw in parts:
        inst = raw.get("instrument")
        if inst:
            instruments.append(inst)
        day_trades = _compact_trades(raw.get("trades") or [], inst)
        for t in day_trades:
            t["instrument"] = inst
        trades.extend(day_trades)
        bars += int(raw.get("bars_today") or 0)
        if raw.get("message"):
            messages.append(f"{inst or '?'}: {raw['message']}")
    wins = sum(1 for t in trades if str(t.get("result") or "").upper() in ("WIN",))
    losses = sum(1 for t in trades if str(t.get("result") or "").upper() in ("LOSS",))
    net = round(sum(float(t.get("pnl_inr") or 0) for t in trades), 2)
    count = len(trades)
    if count:
        message = f"{count} trade(s) · {wins}W / {losses}L · ₹{net:,.0f}"
        status = "traded"
    else:
        message = "No strategy trades"
        status = "flat" if bars else "no_data"
    return {
        "date": session.isoformat(),
        "label": label,
        "status": status,
        "trade_count": count,
        "wins": wins,
        "losses": losses,
        "net_pnl": net,
        "bars": bars,
        "instruments": instruments,
        "trades": trades,
        "message": message,
        "detail": messages,
    }


def pick_last_signal(days: list[dict]) -> Optional[dict[str, Any]]:
    """Newest day (in the given order, newest-first) that actually traded."""
    for day in days:
        if int(day.get("trade_count") or 0) > 0:
            return day
    return None


def _gap_to_day(summary: dict, session: date, label: str) -> dict[str, Any]:
    trades = []
    for t in summary.get("trades") or []:
        trades.append({
            "symbol": t.get("symbol"),
            "side": t.get("side") or "long",
            "result": str(t.get("result") or "").upper(),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_ts"),
            "pnl_inr": t.get("pnl"),
            "gap_pct": t.get("gap_pct"),
        })
    count = int(summary.get("count") or len(trades))
    status = summary.get("status") or ("traded" if count else "flat")
    return {
        "date": session.isoformat(),
        "label": label,
        "status": status,
        "trade_count": count,
        "wins": int(summary.get("wins") or 0),
        "losses": int(summary.get("losses") or 0),
        "net_pnl": float(summary.get("pnl") or 0),
        "open_count": int(summary.get("open_count") or 0),
        "trades": trades,
        "message": summary.get("note") or summary.get("headline") or "",
        "headline": summary.get("headline"),
    }


def _scan_fno_side(
    side: str,
    today: date,
    yesterday: date,
    *,
    lookback: int = LOOKBACK_SESSIONS,
) -> dict[str, Any]:
    from trading.services.fno_day_check import check_today_trades
    from trading.services.fno_engine import enrich_features
    from trading.services.fno_live import load_stored_instrument_bars
    from trading.services.fno_long_day_check import check_today_long_trades

    checker = check_today_trades if side == "short" else check_today_long_trades
    known: set[date] = set()
    stored: dict[str, Any] = {}
    for key in INSTRUMENTS:
        df = load_stored_instrument_bars(key)
        known.update(_session_dates_from_frame(df))
        if df is not None and not getattr(df, "empty", True):
            stored[key] = enrich_features(df)
        else:
            stored[key] = df

    extra = sorted(d for d in known if d < yesterday)
    extra = extra[-max(lookback - 2, 0):]
    extra.reverse()  # newest first after today/yesterday

    def _run(session: date) -> dict:
        parts = []
        for key in INSTRUMENTS:
            bars = stored.get(key)
            if bars is None or getattr(bars, "empty", True):
                parts.append(checker(key, session=session, force_fetch=False))
            else:
                parts.append(checker(key, session=session, force_fetch=False, bars=bars))
        label = (
            "Today" if session == today
            else "Yesterday" if session == yesterday
            else session.strftime("%a %d %b")
        )
        return combine_fno_days(parts, session, label)

    today_day = _run(today)
    yest_day = _run(yesterday)
    other_days = []
    last = pick_last_signal([today_day, yest_day])
    if last is None:
        for sess in extra:
            day = _run(sess)
            other_days.append(day)
            if int(day.get("trade_count") or 0) > 0:
                last = day
                break
    else:
        # still expose the most recent extra day with a trade for "any other day"
        for sess in extra:
            day = _run(sess)
            if int(day.get("trade_count") or 0) > 0:
                other_days.append(day)
                break

    return {
        "today": today_day,
        "yesterday": yest_day,
        "other": other_days[0] if other_days else None,
        "last_signal": last,
    }


def _scan_gap(today: date, yesterday: date, *, lookback: int = LOOKBACK_SESSIONS) -> dict[str, Any]:
    from trading.services.intraday_gap import (
        known_gap_sessions,
        nifty200_symbols,
        summarize_gap_session,
        _daily_frames,
        _load_symbol_frames,
    )

    symbols = nifty200_symbols()
    daily = _daily_frames(symbols)
    frames, _ = _load_symbol_frames(symbols)
    known = set(known_gap_sessions(symbols))

    today_sum = summarize_gap_session(
        today, symbols=symbols, daily=daily, frames=frames, title="Today",
    )
    yest_sum = summarize_gap_session(
        yesterday, symbols=symbols, daily=daily, frames=frames, title="Yesterday",
    )
    today_day = _gap_to_day(today_sum, today, "Today")
    yest_day = _gap_to_day(yest_sum, yesterday, "Yesterday")

    extra = sorted(d for d in known if d < yesterday)
    extra = list(reversed(extra[-max(lookback - 2, 0):]))
    other = None
    last = pick_last_signal([today_day, yest_day])
    for sess in extra:
        summary = summarize_gap_session(
            sess, symbols=symbols, daily=daily, frames=frames,
            title=sess.strftime("%a %d %b"),
        )
        day = _gap_to_day(summary, sess, sess.strftime("%a %d %b"))
        if int(day.get("trade_count") or 0) > 0:
            other = day
            if last is None:
                last = day
            break

    return {
        "today": today_day,
        "yesterday": yest_day,
        "other": other,
        "last_signal": last,
    }


def scan_recent_signals(lookback: int = LOOKBACK_SESSIONS) -> dict[str, Any]:
    today = _now_ist().date()
    from trading.services.fno_live import load_stored_instrument_bars
    from trading.services.intraday_gap import known_gap_sessions

    known: set[date] = set(known_gap_sessions())
    for key in INSTRUMENTS:
        known.update(_session_dates_from_frame(load_stored_instrument_bars(key)))
    yesterday = previous_session_date(today, known)

    fno_short = _scan_fno_side("short", today, yesterday, lookback=lookback)
    fno_long = _scan_fno_side("long", today, yesterday, lookback=lookback)
    gap = _scan_gap(today, yesterday, lookback=lookback)

    return {
        "as_of": today.isoformat(),
        "today": today.isoformat(),
        "yesterday": yesterday.isoformat(),
        "checked_at": timezone.now().isoformat(),
        "fno_short": fno_short,
        "fno_long": fno_long,
        "gap": gap,
    }


def _should_fetch(today: Optional[date] = None) -> bool:
    today = today or _now_ist().date()
    if history_needs_update(today):
        return True
    market = get_market_status()
    # Same calendar day can still be missing later 5m bars while the session is live.
    return market.status in ("open", "pre_open")


def _wait_if_history_running() -> None:
    while get_sync_status().get("running"):
        _set_status(
            phase="fetching",
            message=get_sync_status().get("message") or "Fetching remaining 5m data…",
            fetch=get_sync_status(),
        )
        time.sleep(1)


def _execute() -> None:
    today = _now_ist().date()
    _set_status(
        running=True,
        phase="fetching",
        message=f"Fetching remaining 5m data through {today.isoformat()}…",
        started_at=timezone.now().isoformat(),
        finished_at=None,
        target_date=today.isoformat(),
        error=None,
        skipped_fetch=False,
    )
    try:
        skipped = False
        if _should_fetch(today):
            _wait_if_history_running()
            if _should_fetch(today):
                result = run_intraday_history_sync(today)
                if not result.get("ok") and "already running" in str(result.get("error") or "").lower():
                    _wait_if_history_running()
                elif not result.get("ok"):
                    raise RuntimeError(result.get("error") or "History sync failed")
            _set_status(fetch=get_sync_status())
        else:
            skipped = True
            _set_status(
                skipped_fetch=True,
                fetch=get_sync_status(),
                message="5m history already through today — checking signals…",
            )

        _set_status(
            phase="scanning",
            skipped_fetch=skipped,
            message="Checking F&O Live, F&O Long, and Gap Open signals…",
        )
        signals = scan_recent_signals()
        _set_status(
            running=False,
            phase="done",
            signals=signals,
            fetch=get_sync_status(),
            coverage_after=get_history_coverage(),
            finished_at=timezone.now().isoformat(),
            message=(
                "History stored. Signal check complete."
                if not skipped
                else "History already current. Signal check complete."
            ),
            error=None,
        )
    except Exception as exc:
        logger.exception("Localhost auto-refresh failed")
        _set_status(
            running=False,
            phase="error",
            error=str(exc),
            finished_at=timezone.now().isoformat(),
            message=f"Failed: {exc}",
            fetch=get_sync_status(),
        )


def start_localhost_auto_refresh(*, force: bool = False) -> dict[str, Any]:
    """Start the boot scan. Without force, runs at most once per process."""
    global _started
    if not is_auto_refresh_enabled() and not force:
        return {
            "ok": False,
            "enabled": False,
            "started": False,
            "message": "Auto-refresh runs on localhost (DEBUG) only",
            "status": get_auto_refresh_status(),
        }
    with _lock:
        if _status.get("running"):
            return {
                "ok": True,
                "enabled": True,
                "started": False,
                "message": _status.get("message") or "Already running",
                "status": get_auto_refresh_status(),
            }
        if _started and not force:
            return {
                "ok": True,
                "enabled": True,
                "started": False,
                "message": "Already ran at application start",
                "status": get_auto_refresh_status(),
            }
        _started = True
        _set_status(running=True, phase="queued", message="Queued…")

    threading.Thread(
        target=_execute,
        name="localhost-auto-refresh",
        daemon=True,
    ).start()
    return {
        "ok": True,
        "enabled": True,
        "started": True,
        "message": "Fetching remaining 5m data through today…",
        "status": get_auto_refresh_status(),
    }


def schedule_localhost_auto_refresh() -> None:
    """Kick once per process when the local Django server boots."""
    if not is_auto_refresh_enabled():
        return
    start_localhost_auto_refresh()
