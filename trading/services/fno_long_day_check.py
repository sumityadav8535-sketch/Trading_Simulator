"""After-hours check: replay today's bars with Elite ML Long v1 rules."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from trading.services.fno_engine import (
    CAPITAL,
    FEATURE_COLS,
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    SLIPPAGE_PTS,
    TRADES_PER_DAY,
    enrich_features,
    lots_for_risk,
    max_lots,
)
from trading.services.fno_live import fetch_instrument_bars
from trading.services.fno_long_engine import (
    LONG_TARGET_R,
    loose_long_signal,
    passes_long_filters,
    sig_ema_long,
)
from trading.services.fno_long_live import active_long_strategy, predict_long_win_prob

IST = ZoneInfo("Asia/Kolkata")


def _simulate_long_trade(df, entry_idx, stop, target):
    if entry_idx >= len(df):
        return None
    entry = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
    entry_ts = df.index[entry_idx]
    for j in range(entry_idx, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        if lo <= stop:
            exit_px = stop - SLIPPAGE_PTS
            return {
                "win": False, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "stop_hit", "exit_ts": ts, "entry_ts": entry_ts,
            }
        if hi >= target:
            exit_px = target - SLIPPAGE_PTS
            return {
                "win": True, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "target_hit", "exit_ts": ts, "entry_ts": entry_ts,
            }
        if ts.time() >= FORCE_EXIT:
            exit_px = cl - SLIPPAGE_PTS
            return {
                "win": exit_px > entry, "pnl_pts": exit_px - entry, "entry": entry,
                "exit": exit_px, "reason": "eod_exit", "exit_ts": ts, "entry_ts": entry_ts,
            }
        if j > entry_idx and ts.date() != entry_ts.date():
            break
    return None


def check_today_long_trades(
    instrument: str = "NIFTY",
    session: Optional[date] = None,
    force_fetch: bool = True,
) -> dict:
    instrument = instrument.upper()
    inst = INSTRUMENTS.get(instrument, INSTRUMENTS["NIFTY"])
    strategy = active_long_strategy()
    target_r = float(strategy.get("target_r", LONG_TARGET_R))
    session = session or datetime.now(IST).date()

    df = enrich_features(fetch_instrument_bars(instrument, force=force_fetch))
    if df.empty:
        return {"ok": False, "error": "No bar data", "instrument": instrument}

    day = df[df.index.date == session]
    if day.empty:
        return {
            "ok": True,
            "instrument": instrument,
            "session": session.isoformat(),
            "trades": [],
            "message": f"No 5m bars for {session.isoformat()}",
            "net_pnl": 0,
            "wins": 0,
            "losses": 0,
            "strategy": strategy.get("name", "Elite ML Long v1"),
        }

    # Need lookback for indicators — use full df but only signal on session
    trades = []
    day_count = 0
    equity = CAPITAL

    session_ilocs = [i for i, ts in enumerate(df.index) if ts.date() == session]
    for i in session_ilocs:
        if i < 50 or i + 1 >= len(df):
            continue
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        if day_count >= int(strategy.get("max_trades_per_day", TRADES_PER_DAY)):
            break
        if not loose_long_signal(row):
            continue

        features = [float(row.get(c, float("nan"))) for c in FEATURE_COLS]
        if any(pd.isna(v) for v in features):
            continue
        prob, _ = predict_long_win_prob(features)
        if prob is None:
            continue

        base = sig_ema_long(row, target_r)
        if base:
            stop, risk_pts = base["stop"], base["risk_pts"]
        else:
            stop = float(row["ema_21"])
            risk_pts = float(row["close"]) - stop
        if risk_pts <= 0:
            continue
        if not passes_long_filters(row, ts, prob, risk_pts, strategy):
            continue

        entry_idx = i + 1
        if df.index[entry_idx].date() != session:
            continue
        entry0 = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
        sp = entry0 - stop
        if sp <= 0:
            continue
        true_target = entry0 + sp * target_r
        out = _simulate_long_trade(df, entry_idx, stop, true_target)
        if not out:
            continue

        max_l = max_lots(equity, inst["mis_margin"])
        lots = lots_for_risk(equity, RISK_PCT, sp, inst["lot_size"], max_l)
        if lots <= 0:
            continue
        pnl = out["pnl_pts"] * inst["lot_size"] * lots
        equity += pnl
        day_count += 1
        trades.append({
            "trade_no": day_count,
            "result": "WIN" if pnl > 0 else "LOSS",
            "entry_time": out["entry_ts"].strftime("%H:%M"),
            "exit_time": out["exit_ts"].strftime("%H:%M"),
            "entry_price": round(out["entry"], 2),
            "exit_price": round(out["exit"], 2),
            "stop": round(stop, 2),
            "target": round(true_target, 2),
            "lots": lots,
            "ml_prob": round(prob, 3),
            "exit_reason": out["reason"],
            "pnl_inr": round(pnl, 2),
            "side": "LONG",
        })

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = len(trades) - wins
    net = sum(t["pnl_inr"] for t in trades)
    return {
        "ok": True,
        "instrument": instrument,
        "session": session.isoformat(),
        "strategy": strategy.get("name", "Elite ML Long v1"),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "net_pnl": round(net, 2),
        "message": (
            f"{len(trades)} long trade(s) · W{wins}/L{losses} · ₹{net:,.0f}"
            if trades else
            f"No Elite ML Long trades on {session.isoformat()} "
            f"({len(day)} bars). Rules: stack + ML."
        ),
        "note": "Replay uses completed-bar long rules (same engine as Live).",
        "paper_trades": [],
    }
