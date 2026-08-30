"""Gap-down bounce with daily RSI(14) 45–70: live scan + backtest loader."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache

from trading.models import DailyPrice, Stock
from trading.services.indicators import _atr, _rsi
from trading.services.intraday_data import get_market_status

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

HUNT = Path(settings.BASE_DIR) / "data" / "intraday_gap_hunt.json"
TRADES = Path(settings.BASE_DIR) / "data" / "intraday_gap_trades.json"
DATA_5M = Path(settings.BASE_DIR) / "data" / "intraday_5m"

CAPITAL = 100_000.0
LEVERAGE = 5.0
GAP_MIN = 0.02
GAP_MAX = 0.06
RSI_LO = 45.0
RSI_HI = 70.0
SL_ATR = 0.6
RISK_PCT = 8.0
MAX_POS = 4
TOP_K = 4
MAX_DEPLOY = 0.50
MIN_PRICE = 60.0
FORCE_EXIT = time(15, 15)
# Bounce target, not yesterday's close (full fills rarely print).
TP_KIND = "pct1.0"
ENTRY_HHMM = "09:30"
REQUIRE_BOUNCE = True  # 9:30 open must hold ≥ 9:15 close
REASON_LABELS = {
    "sl": "Stop",
    "target": "Target",
    "eod": "Flatten 15:15",
    "open": "In play",
    "no_data": "No 5m bars",
    "flat": "Flat",
}

STRATEGY = {
    "name": "Gap-down bounce · RSI 45–70",
    "mode": "down_bounce",
    "side": "long",
    "gap_min": GAP_MIN,
    "gap_max": GAP_MAX,
    "rsi_lo": RSI_LO,
    "rsi_hi": RSI_HI,
    "sl_atr": SL_ATR,
    "risk_pct": RISK_PCT,
    "max_pos": MAX_POS,
    "top_k": TOP_K,
    "max_deploy": MAX_DEPLOY,
    "leverage": LEVERAGE,
    "capital": CAPITAL,
    "tp_kind": TP_KIND,
    "entry_hhmm": ENTRY_HHMM,
    "require_bounce": REQUIRE_BOUNCE,
    "entry": [
        "Yesterday’s official close vs today’s 9:15 open",
        "Gap down ≥ 2% and ≤ 6% (skip crash gaps >6%)",
        "Yesterday’s daily RSI(14) between 45 and 70",
        "Price > ₹60",
        "Wait for the 9:30 open",
        "9:30 open must be ≥ 9:15 close (bounce confirmation — skip if still falling)",
        "If several names qualify, take the 4 largest gaps",
    ],
    "exit": [
        "Target: 1.0% above the 9:30 entry",
        "Stop: 0.6 × 5-minute ATR below the 9:30 entry",
        "Flatten 15:15 IST if not filled",
        "8% equity risk, 5× MIS cap, 50% of buying power per name",
    ],
}


def passes_rsi_band(rsi: float | None, lo: float = RSI_LO, hi: float = RSI_HI) -> bool:
    if rsi is None:
        return False
    try:
        val = float(rsi)
    except (TypeError, ValueError):
        return False
    if np.isnan(val):
        return False
    return lo <= val <= hi


def passes_gap_down(gap: float | None, gap_min: float = GAP_MIN, gap_max: float = GAP_MAX) -> bool:
    if gap is None:
        return False
    try:
        val = float(gap)
    except (TypeError, ValueError):
        return False
    if np.isnan(val):
        return False
    return (-gap_max) <= val <= (-gap_min)


def compute_long_target(
    entry: float,
    pdc: float,
    atr: float,
    stop: float,
    kind: str | None = None,
) -> float:
    """Long take-profit. `kind`: fill / half / qtr / fill75 / eod / r1.5 / atr1 / pct1.0."""
    k = (kind or TP_KIND or "fill").lower().strip()
    risk = float(entry) - float(stop)
    gap_room = float(pdc) - float(entry)
    atr = float(atr) if atr and not (isinstance(atr, float) and np.isnan(atr)) else 0.0
    if k == "fill":
        tgt = float(pdc)
    elif k in ("half", "fill50"):
        tgt = float(entry) + 0.5 * gap_room
    elif k in ("qtr", "fill25"):
        tgt = float(entry) + 0.25 * gap_room
    elif k == "fill75":
        tgt = float(entry) + 0.75 * gap_room
    elif k == "eod":
        tgt = float(entry) + max(abs(gap_room), max(atr, 0.0) * 4.0, float(entry) * 0.04)
    elif k.startswith("r"):
        tgt = float(entry) + float(k[1:]) * max(risk, 0.0)
    elif k.startswith("atr"):
        tgt = float(entry) + float(k[3:]) * max(atr, 0.0)
    elif k.startswith("pct"):
        tgt = float(entry) * (1.0 + float(k[3:]) / 100.0)
    else:
        tgt = float(pdc)
    return round(float(tgt), 4)


def tp_kind_label(kind: str | None = None) -> str:
    k = (kind or TP_KIND or "fill").lower().strip()
    labels = {
        "fill": "yesterday’s close (full gap fill)",
        "half": "50% gap fill",
        "fill50": "50% gap fill",
        "qtr": "25% gap fill",
        "fill25": "25% gap fill",
        "fill75": "75% gap fill",
        "eod": "no hard target — flatten 15:15",
    }
    if k in labels:
        return labels[k]
    if k.startswith("r"):
        return f"{k[1:]}R (stop distance)"
    if k.startswith("atr"):
        return f"{k[3:]} × 5-minute ATR"
    if k.startswith("pct"):
        return f"{k[3:]}% from entry"
    return k


def size_long(
    entry: float,
    stop: float,
    target: float,
    equity: float = CAPITAL,
    risk_pct: float = RISK_PCT,
    leverage: float = LEVERAGE,
    max_deploy: float = MAX_DEPLOY,
) -> dict[str, Any]:
    risk_ps = entry - stop
    reward_ps = target - entry
    if entry <= 0 or risk_ps <= 0:
        return {
            "qty": 0, "risk_inr": 0.0, "target_pnl": 0.0, "rr": 0.0,
            "notional": 0.0, "risk_ps": round(risk_ps, 4) if risk_ps else 0.0,
        }
    qty_risk = int((equity * risk_pct / 100.0) / risk_ps)
    qty_cap = int((equity * leverage * max_deploy) / entry)
    qty = max(min(qty_risk, qty_cap), 0)
    risk_inr = round(qty * risk_ps, 2)
    target_pnl = round(qty * reward_ps, 2)
    rr = round(reward_ps / risk_ps, 2) if risk_ps else 0.0
    return {
        "qty": qty,
        "risk_inr": risk_inr,
        "target_pnl": target_pnl,
        "rr": rr,
        "notional": round(qty * entry, 2),
        "risk_ps": round(risk_ps, 4),
    }


def build_setup(
    symbol: str,
    open_px: float,
    pdc: float,
    atr: float,
    rsi: float,
    session: date | str,
    equity: float = CAPITAL,
    *,
    entry: float | None = None,
    pending: bool = False,
    entry_time: str | None = None,
) -> Optional[dict[str, Any]]:
    if open_px < MIN_PRICE or pdc <= 0:
        return None
    gap = open_px / pdc - 1.0
    if not passes_gap_down(gap) or not passes_rsi_band(rsi):
        return None
    fill = float(entry) if entry is not None else float(open_px)
    if fill < MIN_PRICE:
        return None
    if atr is None or atr <= 0 or (isinstance(atr, float) and np.isnan(atr)):
        atr = fill * 0.008
    stop = fill - SL_ATR * float(atr)
    target = compute_long_target(fill, pdc, float(atr), stop, TP_KIND)
    if not (target > fill > stop):
        return None
    sized = size_long(fill, stop, target, equity=equity)
    if sized["qty"] <= 0:
        return None
    sess = str(session)
    hhmm = entry_time or ENTRY_HHMM
    return {
        "symbol": symbol,
        "side": "long",
        "session": sess,
        "gap": round(gap, 4),
        "gap_pct": round(gap * 100, 2),
        "rsi": round(float(rsi), 1),
        "entry": round(float(fill), 2),
        "gap_open": round(float(open_px), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "pdc": round(float(pdc), 2),
        "atr": round(float(atr), 3),
        "qty": sized["qty"],
        "risk_inr": sized["risk_inr"],
        "target_pnl": sized["target_pnl"],
        "rr": sized["rr"],
        "notional": sized["notional"],
        "entry_time": hhmm,
        "pending": bool(pending),
        "tp_kind": TP_KIND,
        "exit_time": "15:15 if not filled",
    }


def select_trades(
    candidates: list[dict[str, Any]],
    top_k: int = TOP_K,
    max_pos: int = MAX_POS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(candidates, key=lambda c: -abs(float(c.get("gap") or 0)))
    take_n = max(min(int(top_k), int(max_pos)), 0)
    taken = ranked[:take_n]
    watch = ranked[take_n:]
    for i, row in enumerate(taken, 1):
        row["rank"] = i
        row["status"] = "trade"
    for i, row in enumerate(watch, take_n + 1):
        row["rank"] = i
        row["status"] = "watch"
    return taken, watch


def load_gap_hunt() -> dict[str, Any]:
    if not HUNT.exists():
        return {}
    data = json.loads(HUNT.read_text(encoding="utf-8"))
    trades: list[dict] = []
    if TRADES.exists():
        raw = json.loads(TRADES.read_text(encoding="utf-8"))
        trades = raw if isinstance(raw, list) else []
        trades = sorted(trades, key=lambda t: str(t.get("exit_ts", "")), reverse=True)
    data["trades"] = trades
    return data


def nifty200_symbols() -> list[str]:
    return list(
        Stock.objects.filter(is_active=True, is_nifty200=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )


def _now_ist() -> datetime:
    return datetime.now(IST)


def _daily_frames(symbols: list[str], lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    cache_key = f"gap:daily:{date.today().isoformat()}:{len(symbols)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    cutoff = date.today() - timedelta(days=lookback_days)
    rows = list(
        DailyPrice.objects.filter(stock_id__in=symbols, date__gte=cutoff)
        .order_by("stock_id", "date")
        .values("stock_id", "date", "open", "high", "low", "close")
    )
    out: dict[str, pd.DataFrame] = {}
    if not rows:
        cache.set(cache_key, out, 120)
        return out
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    for sym, g in df.groupby("stock_id"):
        g = g.sort_values("date").set_index("date")
        if len(g) < 20:
            continue
        g = g.copy()
        g["rsi14"] = _rsi(g["close"], 14)
        g["atr14"] = _atr(g["high"], g["low"], g["close"], 14)
        out[str(sym)] = g
    cache.set(cache_key, out, 300)
    return out


def _prior_daily(frame: pd.DataFrame, session: date) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    prior = frame[frame.index < session]
    if prior.empty:
        return None
    return prior.iloc[-1]


def _atr_from_5m(df: pd.DataFrame, session: date) -> float:
    if df is None or df.empty:
        return 0.0
    work = df.copy()
    idx = work.index
    if idx.tz is None:
        work.index = idx.tz_localize("Asia/Kolkata")
    else:
        work.index = idx.tz_convert("Asia/Kolkata")
    sess = pd.Series(work.index.date, index=work.index)
    prev = work[sess < session]
    use = prev if len(prev) >= 20 else work
    if use.empty:
        return 0.0
    atr = _atr(use["high"], use["low"], use["close"], 14)
    val = float(atr.iloc[-1]) if len(atr) else 0.0
    if np.isnan(val) or val <= 0:
        return 0.0
    return val


def _first_bar(df: pd.DataFrame) -> Optional[tuple[date, float]]:
    if df is None or df.empty:
        return None
    work = df.copy()
    idx = work.index
    if idx.tz is None:
        work.index = idx.tz_localize("Asia/Kolkata")
    else:
        work.index = idx.tz_convert("Asia/Kolkata")
    last_ts = work.index[-1]
    session = last_ts.date()
    day = work[pd.Series(work.index.date, index=work.index) == session]
    if day.empty:
        return None
    return session, float(day.iloc[0]["open"])


def _load_pickle(symbol: str) -> pd.DataFrame:
    path = DATA_5M / f"{symbol}.pkl"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _as_date(val: date | datetime | str | None) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def _tz_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    idx = work.index
    if idx.tz is None:
        work.index = idx.tz_localize("Asia/Kolkata")
    else:
        work.index = idx.tz_convert("Asia/Kolkata")
    return work


def _session_bars(df: pd.DataFrame, session: date) -> pd.DataFrame:
    work = _tz_ist(df)
    if work.empty:
        return work
    mask = pd.Series(work.index.date, index=work.index) == session
    return work.loc[mask].sort_index()


def _session_dates(df: pd.DataFrame) -> set[date]:
    work = _tz_ist(df)
    if work.empty:
        return set()
    return set(pd.Series(work.index.date, index=work.index).unique())


def _hhmm_time(val: str | None = None) -> time:
    raw = (val or ENTRY_HHMM or "09:15").strip()
    parts = raw.split(":")
    return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _session_open(df: pd.DataFrame, session: date) -> Optional[dict[str, Any]]:
    """9:15 open for the gap filter, plus the configured entry bar (9:30)."""
    day = _session_bars(df, session)
    if day.empty:
        return None
    first = day.iloc[0]
    open915 = float(first["open"])
    close915 = float(first["close"])
    want = _hhmm_time(ENTRY_HHMM)
    entry_px = None
    for ts, row in day.iterrows():
        bt = _bar_time(ts)
        if bt.hour == want.hour and bt.minute == want.minute:
            entry_px = float(row["open"])
            break
    last_t = _bar_time(day.index[-1])
    waiting = (last_t.hour, last_t.minute) < (want.hour, want.minute)
    pending = entry_px is None and waiting
    fill = entry_px if entry_px is not None else open915
    bounce = None if pending else bool(fill >= close915)
    return {
        "session": session,
        "open": open915,
        "close915": close915,
        "entry": fill,
        "pending": pending,
        "bounce": bounce,
        "atr": _atr_from_5m(df, session),
        "entry_time": ENTRY_HHMM,
    }


def previous_session(as_of: date, known: Optional[set[date]] = None) -> date:
    if known:
        prior = [d for d in known if d < as_of]
        if prior:
            return max(prior)
    d = as_of - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _bar_time(ts) -> time:
    if getattr(ts, "tzinfo", None) is not None:
        if hasattr(ts, "tz_convert"):
            ts = ts.tz_convert(IST)
        else:
            ts = ts.astimezone(IST)
    return ts.time() if hasattr(ts, "time") else FORCE_EXIT


def _session_still_open(session: date, now: datetime) -> bool:
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)
    if now.date() != session:
        return now.date() < session
    return now.time() < FORCE_EXIT


def evaluate_gap_trade(
    setup: dict[str, Any],
    df: Optional[pd.DataFrame],
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Walk 5-minute bars and mark the setup win / loss / still open."""
    out = dict(setup)
    session = _as_date(setup.get("session"))
    now = now or _now_ist()
    try:
        entry = float(setup["entry"])
        stop = float(setup["stop"])
        target = float(setup["target"])
        qty = int(setup.get("qty") or 0)
    except (TypeError, ValueError, KeyError):
        out.update(
            result="no_data",
            outcome="none",
            reason="no_data",
            reason_label=REASON_LABELS["no_data"],
            pnl=None,
            exit=None,
            win=None,
        )
        return out

    if setup.get("pending"):
        out.update(
            result="open",
            outcome="open",
            reason="open",
            reason_label=f"Wait {setup.get('entry_time') or ENTRY_HHMM}",
            pnl=None,
            exit=None,
            win=None,
            note=f"Gap is on; buy the {setup.get('entry_time') or ENTRY_HHMM} open",
        )
        return out

    day = _session_bars(df, session) if df is not None and session else pd.DataFrame()
    if day.empty:
        out.update(
            result="no_data",
            outcome="none",
            reason="no_data",
            reason_label=REASON_LABELS["no_data"],
            pnl=None,
            exit=None,
            win=None,
            note="No 5-minute bars",
        )
        return out

    def _close(px: float, ts, reason: str, *, still_open: bool = False) -> dict[str, Any]:
        pnl = round((float(px) - entry) * qty, 2)
        if still_open:
            result = "open"
            win: Optional[bool] = None
        elif pnl > 0:
            result = "win"
            win = True
        elif pnl < 0:
            result = "loss"
            win = False
        else:
            result = "flat"
            win = False
        out.update(
            result=result,
            outcome=result,
            reason=reason,
            reason_label=REASON_LABELS.get(reason, reason),
            pnl=pnl,
            exit=None if still_open else round(float(px), 2),
            last=round(float(px), 2),
            exit_ts=str(ts),
            win=win,
        )
        return out

    now_cmp = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
    entry_t = _hhmm_time(str(setup.get("entry_time") or ENTRY_HHMM))
    last_seen: Optional[tuple[Any, float]] = None
    for ts, row in day.iterrows():
        ts_ist = ts
        if getattr(ts, "tzinfo", None) is not None:
            ts_ist = ts.tz_convert(IST) if hasattr(ts, "tz_convert") else ts.astimezone(IST)
        if ts_ist > now_cmp:
            break
        bar_t = _bar_time(ts)
        if (bar_t.hour, bar_t.minute) < (entry_t.hour, entry_t.minute):
            continue
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        last_seen = (ts, close)
        if low <= stop:
            return _close(stop, ts, "sl")
        if high >= target:
            return _close(target, ts, "target")
        if bar_t >= FORCE_EXIT:
            return _close(close, ts, "eod")

    if last_seen is None:
        out.update(
            result="no_data",
            outcome="none",
            reason="no_data",
            reason_label=REASON_LABELS["no_data"],
            pnl=None,
            exit=None,
            win=None,
            note="No 5-minute bars yet",
        )
        return out

    last_ts, last_px = last_seen
    if session and _session_still_open(session, now):
        return _close(last_px, last_ts, "open", still_open=True)
    return _close(last_px, last_ts, "eod")


def _load_symbol_frames(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], set[date]]:
    frames: dict[str, pd.DataFrame] = {}
    sessions: set[date] = set()
    for sym in symbols:
        df = _load_pickle(sym)
        if df is None or df.empty:
            continue
        frames[sym] = df
        sessions.update(_session_dates(df))
    return frames, sessions


def _opens_from_frames(
    frames: dict[str, pd.DataFrame],
    session: date,
) -> dict[str, dict[str, Any]]:
    opens: dict[str, dict[str, Any]] = {}
    for sym, df in frames.items():
        row = _session_open(df, session)
        if row is not None:
            opens[sym] = row
    return opens


def _day_has_bars(frames: dict[str, pd.DataFrame], session: date) -> bool:
    return any(not _session_bars(df, session).empty for df in frames.values())


def _summarize_gap_day(
    title: str,
    session: date,
    scan: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    market,
    now: datetime,
) -> dict[str, Any]:
    taken = list(scan.get("taken") or [])
    evaluated = [evaluate_gap_trade(t, frames.get(t.get("symbol")), now=now) for t in taken]
    has_bars = _day_has_bars(frames, session)
    wins = sum(1 for t in evaluated if t.get("result") == "win")
    losses = sum(1 for t in evaluated if t.get("result") == "loss")
    open_count = sum(1 for t in evaluated if t.get("result") == "open")
    closed_pnl = round(
        sum(float(t.get("pnl") or 0) for t in evaluated if t.get("result") in ("win", "loss", "flat")),
        2,
    )
    open_pnl = round(
        sum(float(t.get("pnl") or 0) for t in evaluated if t.get("result") == "open"),
        2,
    )

    if session.weekday() >= 5:
        status, outcome, note = "weekend", "none", "Market closed (weekend)"
    elif not has_bars:
        status, outcome, note = "no_data", "none", "No 5-minute bars for this session yet"
    elif not evaluated:
        status = "flat"
        outcome = "none"
        note = "No trade — no Nifty 200 name gapped down 2–6% with RSI 45–70"
    elif open_count:
        status, outcome, note = "open", "open", f"{len(evaluated)} trade(s) in play"
    else:
        status = "traded"
        if wins and not losses:
            outcome = "win"
        elif losses and not wins:
            outcome = "loss"
        elif wins and losses:
            outcome = "mixed"
        else:
            outcome = "flat"
        note = f"{len(evaluated)} trade(s)"

    headline = {
        "win": "WIN",
        "loss": "LOSS",
        "open": "OPEN",
        "mixed": "MIXED",
        "flat": "FLAT",
        "none": "NO TRADE",
    }[outcome]
    if status == "no_data":
        headline = "NO DATA"
    elif status == "weekend":
        headline = "CLOSED"

    return {
        "title": title,
        "label": session.strftime("%a %d %b"),
        "date": session.isoformat(),
        "status": status,
        "outcome": outcome,
        "headline": headline,
        "note": note,
        "count": len(evaluated),
        "wins": wins,
        "losses": losses,
        "open_count": open_count,
        "pnl": closed_pnl,
        "open_pnl": open_pnl,
        "has_bars": has_bars,
        "trades": evaluated,
        "is_today": str(session) == market.session_date,
    }


def recent_gap_days(
    *,
    session: Optional[date] = None,
    symbols: Optional[list[str]] = None,
    daily: Optional[dict[str, pd.DataFrame]] = None,
    frames: Optional[dict[str, pd.DataFrame]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Today + previous session: was there a trade, and did it win or lose."""
    now_arg = now
    now = now or _now_ist()
    market = get_market_status(now=now)
    today = session or date.fromisoformat(market.session_date)
    use_cache = (
        frames is None
        and daily is None
        and symbols is None
        and session is None
        and now_arg is None
    )
    cache_key = f"gap:recent:{today.isoformat()}"
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            live = dict(cached.get("live") or {})
            live["market"] = {
                "status": market.status,
                "message": market.message,
                "now_ist": market.now_ist,
                "session_date": market.session_date,
            }
            live["is_today"] = str(live.get("session")) == market.session_date
            out = dict(cached)
            out["live"] = live
            return out

    symbols = symbols or nifty200_symbols()
    daily = daily if daily is not None else _daily_frames(symbols)
    if frames is None:
        frames, known_sessions = _load_symbol_frames(symbols)
    else:
        known_sessions = set()
        for df in frames.values():
            known_sessions.update(_session_dates(df))

    yesterday = previous_session(today, known_sessions or None)
    today_opens = _opens_from_frames(frames, today)
    yest_opens = _opens_from_frames(frames, yesterday)
    today_scan = scan_gap_setups(
        session=today, equity=CAPITAL, symbols=symbols, daily=daily, opens=today_opens,
    )
    yest_scan = scan_gap_setups(
        session=yesterday, equity=CAPITAL, symbols=symbols, daily=daily, opens=yest_opens,
    )

    last_sess = max(known_sessions) if known_sessions else today
    if last_sess == today:
        live = today_scan
    elif last_sess == yesterday:
        live = yest_scan
    else:
        live = scan_gap_setups(
            session=last_sess,
            equity=CAPITAL,
            symbols=symbols,
            daily=daily,
            opens=_opens_from_frames(frames, last_sess),
        )

    today_day = _summarize_gap_day("Today", today, today_scan, frames, market, now)
    yest_day = _summarize_gap_day("Yesterday", yesterday, yest_scan, frames, market, now)
    payload = {
        "as_of": today.isoformat(),
        "today": today_day,
        "yesterday": yest_day,
        "cards": [yest_day, today_day],
        "live": live,
    }
    if use_cache:
        cache.set(cache_key, payload, 120)
    return payload


def scan_gap_setups(
    *,
    session: Optional[date] = None,
    equity: float = CAPITAL,
    symbols: Optional[list[str]] = None,
    daily: Optional[dict[str, pd.DataFrame]] = None,
    opens: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Find today's (or last cached session's) gap-down bounce trades.

    `opens` maps symbol -> {session, open, atr} for tests / live inject.
    """
    market = get_market_status()
    use_cache = daily is None and opens is None and session is None and symbols is None
    cache_key = f"gap:live:{date.today().isoformat()}"
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            cached = dict(cached)
            cached["market"] = {
                "status": market.status,
                "message": market.message,
                "now_ist": market.now_ist,
                "session_date": market.session_date,
            }
            cached["is_today"] = str(cached.get("session")) == market.session_date
            return cached

    symbols = symbols or nifty200_symbols()
    daily = daily if daily is not None else _daily_frames(symbols)

    pickle_session: Optional[date] = None
    if opens is None:
        open_rows: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            df = _load_pickle(sym)
            first = _first_bar(df)
            if first is None:
                continue
            sess, o = first
            pickle_session = sess if pickle_session is None else max(pickle_session, sess)
            info = _session_open(df, sess)
            if info is None:
                open_rows[sym] = {"session": sess, "open": o, "atr": _atr_from_5m(df, sess)}
            else:
                open_rows[sym] = info
    else:
        open_rows = dict(opens)
        for row in open_rows.values():
            sess = row.get("session")
            if sess is None:
                continue
            if not isinstance(sess, date):
                sess = _as_date(sess)
            if sess:
                pickle_session = sess if pickle_session is None else max(pickle_session, sess)

    if session is None:
        if open_rows:
            session = max(r["session"] for r in open_rows.values() if r.get("session"))
        else:
            session = date.fromisoformat(market.session_date)

    candidates: list[dict[str, Any]] = []
    scanned = 0
    rsi_ok = 0
    for sym in symbols:
        frame = daily.get(sym)
        prior = _prior_daily(frame, session) if frame is not None else None
        if prior is None:
            continue
        scanned += 1
        rsi = prior.get("rsi14")
        pdc = float(prior["close"])
        if passes_rsi_band(rsi):
            rsi_ok += 1
        row = open_rows.get(sym)
        if not row or row.get("session") != session:
            continue
        atr = float(row.get("atr") or 0) or float(prior.get("atr14") or 0) * 0.25
        fill = row.get("entry")
        try:
            fill_px = float(fill) if fill is not None else float(row["open"])
        except (TypeError, ValueError):
            fill_px = float(row["open"])
        pending = bool(row.get("pending"))
        bounce = row.get("bounce")
        if REQUIRE_BOUNCE and not pending and bounce is False:
            continue
        setup = build_setup(
            sym,
            float(row["open"]),
            pdc,
            atr,
            float(rsi) if pd.notna(rsi) else float("nan"),
            session,
            equity,
            entry=fill_px,
            pending=pending,
            entry_time=str(row.get("entry_time") or ENTRY_HHMM),
        )
        if setup:
            candidates.append(setup)

    taken, watch = select_trades(candidates)
    source = "injected" if opens is not None else "5m_cache"
    payload = {
        "session": str(session),
        "is_today": str(session) == market.session_date,
        "market": {
            "status": market.status,
            "message": market.message,
            "now_ist": market.now_ist,
            "session_date": market.session_date,
        },
        "taken": taken,
        "watch": watch,
        "candidates": candidates,
        "counts": {
            "universe": len(symbols),
            "scanned": scanned,
            "rsi_band": rsi_ok,
            "qualified": len(candidates),
            "trades": len(taken),
        },
        "source": source,
        "pickle_session": str(pickle_session) if pickle_session else None,
        "strategy": STRATEGY,
        "entry_hhmm": ENTRY_HHMM,
        "tp_kind": TP_KIND,
    }
    if use_cache:
        cache.set(cache_key, payload, 120)
    return payload


def load_tp_search() -> dict[str, Any]:
    path = Path(settings.BASE_DIR) / "data" / "intraday_gap_tp_search.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_gap_page() -> dict[str, Any]:
    hunt = load_gap_hunt()
    recent = recent_gap_days()
    live = recent.get("live") or scan_gap_setups()
    story = hunt.get("story") or hunt.get("result") or {}
    trades = hunt.get("trades") or []
    for t in trades:
        t.setdefault("stop", None)
        t.setdefault("target", None)
        t.setdefault("rsi", None)
        t.setdefault("pdc", None)
        try:
            t["win"] = float(t.get("pnl") or 0) > 0
        except (TypeError, ValueError):
            t["win"] = False
    search = load_tp_search()
    wanted = {
        ("915", "fill"): "Old · 9:15 + yesterday’s close",
        ("915", "pct1.0"): "9:15 + 1% target",
        ("930", "pct1.0"): "9:30 + 1% (no bounce)",
        ("930_alive", "pct1.0"): "9:30 only if 9:15 stop untouched",
        ("930", "fill"): "9:30 + yesterday’s close",
    }
    compare = []
    for row in search.get("rows") or []:
        key = (row.get("entry"), row.get("tp"))
        if key in wanted:
            compare.append({**row, "label": wanted[key], "live": False})
    pro_path = Path(settings.BASE_DIR) / "data" / "intraday_gap_pro_hunt.json"
    if pro_path.exists():
        try:
            pro = json.loads(pro_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pro = {}
        for row in pro.get("rows") or []:
            if row.get("label") == "T bounce 1.0% size-up":
                compare.insert(0, {
                    **row,
                    "label": "Live · 9:30 bounce + 1%",
                    "live": True,
                })
                break
    compare.sort(key=lambda r: 0 if r.get("live") else 1)
    return {
        "hunt": hunt,
        "story": story,
        "months": story.get("months") or [],
        "window": hunt.get("window") or {},
        "fill_stats": hunt.get("fill_stats") or [],
        "trades": trades,
        "live": live,
        "recent_days": recent,
        "tp_search": search,
        "compare": compare,
        "strategy": {**(hunt.get("strategy") or {}), **STRATEGY},
    }
