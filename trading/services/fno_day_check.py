"""
Post-market one-click check: replay today's session with the live F&O strategy
and report any trades (win / loss) so the user does not need the laptop open all day.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from django.utils import timezone

from trading.services.fno_engine import (
    CAPITAL,
    FEATURE_COLS,
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    SLIPPAGE_PTS,
    STRATEGY,
    TARGET_R,
    TRADES_PER_DAY,
    enrich_features,
    loose_short_signal,
    lots_for_risk,
    max_lots,
    passes_strategy_filters,
    sig_ema_short,
)
from trading.services.fno_live import fetch_instrument_bars, predict_win_prob
from trading.services.intraday_data import get_market_status

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


def _now_ist() -> datetime:
    return datetime.now(IST)


def _ts_key(ts) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _simulate_short_trade(
    df: pd.DataFrame,
    entry_idx: int,
    stop: float,
    target: float,
) -> dict:
    """Simulate short from entry bar open; exit on stop / target / force-exit."""
    empty = {
        "exit_reason": "no_fill",
        "pnl_pts": 0.0,
        "entry_price": None,
        "exit_price": None,
        "entry_time": None,
        "exit_time": None,
    }
    if entry_idx >= len(df):
        return empty

    entry_row = df.iloc[entry_idx]
    entry = float(entry_row["open"]) - SLIPPAGE_PTS
    entry_ts = df.index[entry_idx]

    for j in range(entry_idx, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

        if hi >= stop:
            exit_px = stop + SLIPPAGE_PTS
            return {
                "exit_reason": "stop_hit",
                "pnl_pts": entry - exit_px,
                "entry_price": round(entry, 2),
                "exit_price": round(exit_px, 2),
                "entry_time": entry_ts.isoformat(),
                "exit_time": ts.isoformat(),
            }
        if lo <= target:
            exit_px = target + SLIPPAGE_PTS
            return {
                "exit_reason": "target_hit",
                "pnl_pts": entry - exit_px,
                "entry_price": round(entry, 2),
                "exit_price": round(exit_px, 2),
                "entry_time": entry_ts.isoformat(),
                "exit_time": ts.isoformat(),
            }
        if ts.time() >= FORCE_EXIT:
            exit_px = cl + SLIPPAGE_PTS
            return {
                "exit_reason": "eod_exit",
                "pnl_pts": entry - exit_px,
                "entry_price": round(entry, 2),
                "exit_price": round(exit_px, 2),
                "entry_time": entry_ts.isoformat(),
                "exit_time": ts.isoformat(),
            }
        if j > entry_idx and ts.date() != entry_ts.date():
            break

    return empty


def _paper_trades_for_day(session: date, instrument: str) -> list[dict]:
    """Any paper fills recorded for this calendar day (if paper module is used)."""
    try:
        from trading.models import PaperTrade

        qs = (
            PaperTrade.objects.filter(
                instrument=instrument.upper(),
                session_date=session,
            )
            .order_by("entry_time")
        )
        out = []
        for t in qs:
            pnl = float(t.pnl) if t.pnl is not None else None
            out.append({
                "source": "paper",
                "side": t.side,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": float(t.entry_price) if t.entry_price is not None else None,
                "exit_price": float(t.exit_price) if t.exit_price is not None else None,
                "lots": t.lots,
                "exit_reason": t.exit_reason or "",
                "pnl_inr": round(pnl, 2) if pnl is not None else None,
                "result": (
                    "WIN" if pnl is not None and pnl > 0
                    else "LOSS" if pnl is not None and pnl <= 0
                    else "OPEN"
                ),
            })
        return out
    except Exception:
        logger.debug("Paper trades lookup skipped", exc_info=True)
        return []


def check_today_trades(
    instrument: str = "NIFTY",
    session: Optional[date] = None,
    force_fetch: bool = True,
) -> dict:
    """
    Fetch latest 5m bars and replay the live strategy for one session day.

    Returns a summary + trade list. Safe to call after market close once.
    """
    key = (instrument or "NIFTY").upper()
    if key not in INSTRUMENTS:
        return {"ok": False, "error": f"Unknown instrument: {instrument}"}

    inst = INSTRUMENTS[key]
    session = session or _now_ist().date()
    market = get_market_status()
    strategy = STRATEGY
    threshold = float(strategy.get("ml_threshold", 0.58))

    try:
        df = enrich_features(fetch_instrument_bars(key, force=force_fetch))
    except Exception as exc:
        logger.exception("Day check fetch failed for %s", key)
        return {
            "ok": False,
            "error": f"Failed to fetch bars: {exc}",
            "instrument": key,
            "session_date": session.isoformat(),
        }

    if df is None or df.empty or len(df) < 55:
        return {
            "ok": False,
            "error": "Not enough 5m history. Click “Fetch history to today” first.",
            "instrument": key,
            "session_date": session.isoformat(),
        }

    day_mask = df.index.date == session
    day_bars = int(day_mask.sum())
    if day_bars == 0:
        last = df.index.max()
        last_d = last.date().isoformat() if hasattr(last, "date") else str(last)[:10]
        return {
            "ok": True,
            "instrument": key,
            "name": inst["name"],
            "session_date": session.isoformat(),
            "market": {
                "status": market.status,
                "message": market.message,
                "now_ist": market.now_ist,
            },
            "bars_today": 0,
            "data_last_bar": last_d,
            "trades": [],
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "message": (
                f"No {key} bars for {session.isoformat()}. "
                f"Latest stored bar is {last_d}. "
                "If the market already closed, try again in a minute or use Fetch history."
            ),
            "paper_trades": _paper_trades_for_day(session, key),
            "checked_at": timezone.now().isoformat(),
            "strategy": strategy.get("name", "Elite ML Short v2"),
        }

    # Indices into full df for today's bars (need prior days for indicators)
    day_ilocs = [i for i, is_day in enumerate(day_mask) if is_day]
    trades: list[dict] = []

    equity = float(CAPITAL)
    busy_until_idx = -1  # while a trade is open, skip new entries

    for i in day_ilocs:
        if len(trades) >= TRADES_PER_DAY:
            break
        if i <= busy_until_idx:
            continue
        if i + 1 >= len(df):
            continue

        ts = df.index[i]
        # Only signal on completed bars: if this is the last row and bar still forming, skip
        now = _now_ist()
        if i == len(df) - 1:
            bar_end = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if getattr(bar_end, "tzinfo", None) is None:
                bar_end = bar_end.replace(tzinfo=IST)
            else:
                bar_end = bar_end.astimezone(IST)
            from datetime import timedelta

            if now < bar_end + timedelta(minutes=5) and now.date() == session:
                # forming bar — skip for entry (same as live)
                continue

        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue

        # Entry next bar must be same session
        if df.index[i + 1].date() != session:
            continue

        row = df.iloc[i]
        if not loose_short_signal(row):
            continue

        features = [float(row.get(c, float("nan"))) for c in FEATURE_COLS]
        if any(pd.isna(v) for v in features):
            continue

        prob, _model = predict_win_prob(features)
        if prob is None:
            continue

        base_sig = sig_ema_short(row, TARGET_R)
        if base_sig:
            stop = float(base_sig["stop"])
            target = float(base_sig["target"])
        else:
            stop = float(row["ema_21"])
            risk_pts_est = stop - float(row["close"])
            if risk_pts_est <= 0:
                continue
            target = float(row["close"] - risk_pts_est * TARGET_R)

        entry_est = float(df.iloc[i + 1]["open"])
        risk_pts = stop - entry_est
        if risk_pts <= 0:
            continue

        if not passes_strategy_filters(row, ts, prob, risk_pts, strategy):
            continue

        outcome = _simulate_short_trade(df, i + 1, stop, target)
        if outcome["exit_reason"] == "no_fill":
            continue

        max_l = max_lots(equity, inst["mis_margin"])
        lots = lots_for_risk(equity, RISK_PCT, risk_pts, inst["lot_size"], max_l)
        if lots <= 0:
            continue

        pnl_pts = float(outcome["pnl_pts"])
        pnl_inr = pnl_pts * inst["lot_size"] * lots
        equity += pnl_inr
        result = "WIN" if pnl_inr > 0 else "LOSS"

        # Mark bars occupied until exit so we don't stack overlapping positions
        exit_ts = outcome["exit_time"]
        busy_until_idx = i + 1
        if exit_ts:
            for j in range(i + 1, len(df)):
                if _ts_key(df.index[j]) == exit_ts or df.index[j].isoformat() == exit_ts:
                    busy_until_idx = j
                    break
                # also match without microsecond quirks
                if str(df.index[j])[:19] == str(exit_ts)[:19]:
                    busy_until_idx = j
                    break

        sig_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
        entry_time_s = outcome["entry_time"]
        exit_time_s = outcome["exit_time"]
        try:
            entry_hm = datetime.fromisoformat(entry_time_s).strftime("%H:%M") if entry_time_s else "—"
            exit_hm = datetime.fromisoformat(exit_time_s).strftime("%H:%M") if exit_time_s else "—"
        except Exception:
            entry_hm, exit_hm = entry_time_s or "—", exit_time_s or "—"

        trades.append({
            "trade_no": len(trades) + 1,
            "source": "replay",
            "instrument": key,
            "side": "SHORT",
            "signal_time": sig_time,
            "entry_time": entry_hm,
            "exit_time": exit_hm,
            "entry_price": outcome["entry_price"],
            "exit_price": outcome["exit_price"],
            "stop": round(stop, 2),
            "target": round(target, 2),
            "risk_pts": round(risk_pts, 2),
            "lots": lots,
            "lot_size": inst["lot_size"],
            "ml_prob": round(float(prob), 3),
            "exit_reason": outcome["exit_reason"],
            "pnl_pts": round(pnl_pts, 2),
            "pnl_inr": round(pnl_inr, 2),
            "result": result,
        })

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    net = round(sum(t["pnl_inr"] for t in trades), 2)
    paper = _paper_trades_for_day(session, key)

    if not trades:
        msg = (
            f"No {key} strategy trades on {session.isoformat()} "
            f"({day_bars} bars replayed). Rules: Elite ML Short v2."
        )
    else:
        msg = (
            f"{len(trades)} trade(s) on {session.isoformat()}: "
            f"{wins} win / {losses} loss · P&L ₹{net:,.0f}"
        )

    return {
        "ok": True,
        "instrument": key,
        "name": inst["name"],
        "session_date": session.isoformat(),
        "market": {
            "status": market.status,
            "message": market.message,
            "now_ist": market.now_ist,
        },
        "bars_today": day_bars,
        "data_first_bar": str(df.index[day_ilocs[0]]) if day_ilocs else None,
        "data_last_bar": str(df.index[day_ilocs[-1]]) if day_ilocs else None,
        "trades": trades,
        "trade_count": len(trades),
        "wins": wins,
        "losses": losses,
        "net_pnl": net,
        "message": msg,
        "paper_trades": paper,
        "paper_count": len(paper),
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "ml_threshold": threshold,
        "strategy": strategy.get("name", "Elite ML Short v2"),
        "checked_at": timezone.now().isoformat(),
        "note": (
            "Replay uses the saved ML model + live filters on completed 5m bars "
            "(same rules as F&O Live ACTIVE). Index proxy, not live order fills."
        ),
    }
