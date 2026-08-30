"""
Live Elite ML Long v1 dashboard: signals, charts, backtest stats.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from django.core.cache import cache
from django.utils import timezone

from trading.services.fno_engine import (
    CAPITAL,
    FEATURE_COLS,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    enrich_features,
    lots_for_risk,
    max_lots,
)
from trading.services.fno_live import (
    fetch_instrument_bars,
    last_completed_bar_iloc,
)
from trading.services.fno_long_engine import (
    LONG_TARGET_R,
    STRATEGY_LONG,
    loose_long_signal,
    merge_strategy_from_meta,
    passes_long_filters,
    sig_ema_long,
)
from trading.services.fno_long_results import (
    get_recent_day_results,
    get_trades,
    load_model_meta,
    load_strategy_results,
    load_strategy_trades_data,
)
from trading.services.intraday_data import get_market_status

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_long_model.pkl"
CACHE_TTL = 30


def _now_ist() -> datetime:
    return datetime.now(IST)


@lru_cache(maxsize=1)
def _load_ml_bundle() -> Optional[dict]:
    if not MODEL_PKL.exists():
        return None
    try:
        with MODEL_PKL.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("Failed to load long ML model: %s", exc)
        return None


def clear_ml_cache() -> None:
    _load_ml_bundle.cache_clear()


def predict_long_win_prob(features: list[float]) -> tuple[Optional[float], str]:
    bundle = _load_ml_bundle()
    meta = load_model_meta()
    model_type = meta.get("model_type", "ensemble_rf_gb_hgb")
    if bundle is None:
        return None, model_type
    try:
        import numpy as np
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.warning("scikit-learn not installed; long ML unavailable")
        return None, model_type

    X = np.array([features], dtype=float)

    def _proba(model, x):
        return float(model.predict_proba(x)[0, 1])

    models = bundle.get("models")
    if models:
        probs = [_proba(m, X) for m in models.values()]
        return float(sum(probs) / len(probs)), model_type

    model = bundle.get("model")
    if model is None:
        return None, model_type
    return _proba(model, X), model_type


def active_long_strategy() -> dict:
    return merge_strategy_from_meta(load_model_meta())


def evaluate_long_signal(key: str, force: bool = False) -> dict:
    key = key.upper()
    inst = INSTRUMENTS.get(key, INSTRUMENTS["NIFTY"])
    market = get_market_status()
    strategy = active_long_strategy()
    threshold = float(strategy.get("ml_threshold", 0.72))
    target_r = float(strategy.get("target_r", LONG_TARGET_R))

    df = enrich_features(fetch_instrument_bars(key, force=force))
    if df.empty or len(df) < 55:
        return {
            "instrument": key,
            "name": inst["name"],
            "status": "no_data",
            "message": "Insufficient bar history for indicators",
            "market": asdict(market),
            "strategy_name": strategy.get("name", "Elite ML Long v1"),
        }

    live_row = df.iloc[-1]
    live_ts = df.index[-1]
    sig_i = last_completed_bar_iloc(df)
    if sig_i < 0:
        return {
            "instrument": key,
            "name": inst["name"],
            "status": "no_data",
            "message": "No completed 5m bar yet",
            "market": asdict(market),
            "strategy_name": strategy.get("name", "Elite ML Long v1"),
        }
    row = df.iloc[sig_i]
    ts = df.index[sig_i]
    bar_time = ts.strftime("%H:%M IST") if hasattr(ts, "strftime") else str(ts)
    bar_forming = sig_i < (len(df) - 1)

    ema_aligned = bool(row["ema_9"] > row["ema_21"] > row["ema_50"])
    above_vwap = bool(pd.isna(row.get("vwap")) or row["close"] > row["vwap"])
    base_sig = sig_ema_long(row, target_r)
    is_loose = loose_long_signal(row)
    now_t = _now_ist().time()
    in_entry_window = MARKET_OPEN <= now_t <= NO_ENTRY_AFTER
    bar_in_window = MARKET_OPEN <= ts.time() <= NO_ENTRY_AFTER
    # champion prefers hour_end from strategy
    h_end = int(strategy.get("hour_end", 12))
    h_start = int(strategy.get("hour_start", 9))
    bar_hour_ok = h_start <= ts.hour <= h_end

    features = [float(row.get(c, float("nan"))) for c in FEATURE_COLS]
    has_nan = any(pd.isna(v) for v in features)
    prob, model_type = (
        predict_long_win_prob(features) if not has_nan else (None, "unknown")
    )

    stop = target = risk_pts = 0.0
    lots = 0
    if base_sig:
        stop = base_sig["stop"]
        target = base_sig["target"]
        risk_pts = base_sig["risk_pts"]
    elif is_loose:
        stop = float(row["ema_21"])
        risk_pts = float(row["close"]) - stop
        if risk_pts > 0:
            target = float(row["close"] + risk_pts * target_r)

    if risk_pts > 0:
        max_l = max_lots(CAPITAL, inst["mis_margin"])
        lots = lots_for_risk(CAPITAL, RISK_PCT, risk_pts, inst["lot_size"], max_l)

    opt_pass = (
        prob is not None
        and risk_pts > 0
        and bar_in_window
        and bar_hour_ok
        and passes_long_filters(row, ts, prob, risk_pts, strategy)
    )
    ml_pass = prob is not None and prob >= threshold
    base_pass = base_sig is not None

    if not in_entry_window:
        status, message = "closed_window", "Outside entry window (9:15–14:45 IST)"
    elif not bar_in_window:
        status, message = "closed_window", f"Signal bar {bar_time} outside entry window"
    elif not bar_hour_ok:
        status, message = (
            "watch",
            f"Bar hour {ts.hour}:xx outside long window {h_start}–{h_end}",
        )
    elif is_loose and opt_pass:
        status, message = (
            "active",
            f"{strategy['name']} — P(win) {prob:.0%}, risk {risk_pts:.1f}pts"
            + (" · completed bar" if bar_forming else ""),
        )
    elif is_loose and ml_pass and risk_pts > strategy.get("max_risk_pts", 35):
        status, message = (
            "watch",
            f"ML OK but stop too wide ({risk_pts:.1f}pts > {strategy.get('max_risk_pts')})",
        )
    elif is_loose and prob is not None and prob < threshold:
        status, message = "watch", f"ML prob {prob:.0%} < {threshold:.0%} threshold"
    elif base_pass and prob is not None:
        status, message = "watch", f"EMA long setup valid but filters not met (prob {prob:.0%})"
    elif is_loose and prob is not None:
        status, message = "watch", "Loose long — awaiting filter pass"
    elif ema_aligned and above_vwap:
        status, message = "setup", "Bullish structure forming — awaiting ML confirmation"
    else:
        status, message = "neutral", "No long setup on last completed bar"

    est_risk_inr = round(risk_pts * inst["lot_size"] * lots, 0) if lots else 0
    est_target_inr = round(risk_pts * target_r * inst["lot_size"] * lots, 0) if lots else 0
    ltp = round(float(live_row["close"]), 2)
    live_open = round(float(live_row["open"]), 2)
    entry_ref = live_open if bar_forming else round(float(row["close"]), 2)

    return {
        "instrument": key,
        "name": inst["name"],
        "ticker": inst["ticker"],
        "lot_size": inst["lot_size"],
        "mis_margin": inst["mis_margin"],
        "status": status,
        "message": message,
        "side": "LONG" if status == "active" else None,
        "bar_time": bar_time,
        "bar_complete": True,
        "bar_forming_skipped": bar_forming,
        "live_bar_time": live_ts.strftime("%H:%M IST") if hasattr(live_ts, "strftime") else str(live_ts),
        "ltp": ltp,
        "live_open": live_open,
        "entry_ref": entry_ref,
        "signal_close": round(float(row["close"]), 2),
        "open": round(float(row["open"]), 2),
        "high": round(float(row["high"]), 2),
        "low": round(float(row["low"]), 2),
        "vwap": round(float(row["vwap"]), 2) if pd.notna(row.get("vwap")) else None,
        "ema_9": round(float(row["ema_9"]), 2),
        "ema_21": round(float(row["ema_21"]), 2),
        "ema_50": round(float(row["ema_50"]), 2),
        "rsi": round(float(row["rsi"]), 1),
        "adx": round(float(row["adx"]), 1) if pd.notna(row.get("adx")) else None,
        "di_diff": round(float(row["di_diff"]), 1) if pd.notna(row.get("di_diff")) else None,
        "macd_hist": round(float(row["macd_hist"]), 2) if pd.notna(row.get("macd_hist")) else None,
        "range_pos": round(float(row["range_pos"]), 2) if pd.notna(row.get("range_pos")) else None,
        "pct_from_open": round(float(row["pct_from_open"]), 2) if pd.notna(row.get("pct_from_open")) else None,
        "ema_aligned": ema_aligned,
        "above_vwap": above_vwap,
        "base_signal": base_pass,
        "loose_candidate": is_loose,
        "ml_prob": round(prob, 3) if prob is not None else None,
        "ml_threshold": threshold,
        "strategy": strategy,
        "strategy_name": strategy.get("name", "Elite ML Long v1"),
        "strategy_pass": opt_pass,
        "ml_model": model_type,
        "ml_available": prob is not None,
        "stop": round(stop, 2) if stop else None,
        "target": round(target, 2) if target else None,
        "risk_pts": round(risk_pts, 2) if risk_pts else None,
        "target_r": target_r,
        "lots": lots,
        "risk_inr": est_risk_inr,
        "target_inr": est_target_inr,
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "in_entry_window": in_entry_window,
        "market": asdict(market),
        "updated_at": timezone.now().strftime("%H:%M:%S IST"),
    }


def build_fno_long_chart_json(key: str, force: bool = False) -> str:
    import plotly.graph_objects as go

    key = key.upper()
    cache_key = f"fno_long:chart:{key}:{date.today().isoformat()}"
    if not force:
        cached = cache.get(cache_key)
        if cached:
            return cached

    df = enrich_features(fetch_instrument_bars(key, force=force))
    today = _now_ist().date()
    sess = df[df.index.date == today] if not df.empty else df
    if sess.empty and not df.empty:
        sess = df.tail(80)
    if sess.empty:
        payload = json.dumps({"data": [], "layout": {"title": f"No F&O data for {key}"}})
        cache.set(cache_key, payload, CACHE_TTL)
        return payload

    inst = INSTRUMENTS[key]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=sess.index, open=sess["open"], high=sess["high"],
            low=sess["low"], close=sess["close"], name=key,
        )
    )
    for col, color, name in [
        ("ema_9", "#34d399", "EMA 9"),
        ("ema_21", "#fbbf24", "EMA 21"),
        ("ema_50", "#f87171", "EMA 50"),
        ("vwap", "#60a5fa", "VWAP/TWAP"),
    ]:
        if col in sess.columns:
            fig.add_trace(
                go.Scatter(x=sess.index, y=sess[col], name=name, line=dict(width=1.2, color=color))
            )

    sig = evaluate_long_signal(key, force=force)
    if sig.get("stop"):
        fig.add_hline(y=sig["stop"], line_dash="dash", line_color="#ef4444", annotation_text="Stop")
    if sig.get("target"):
        fig.add_hline(y=sig["target"], line_dash="dot", line_color="#22c55e", annotation_text="Target")

    fig.update_layout(
        title=f"{inst['name']} LONG — 5m intraday",
        height=420,
        template="plotly_white",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(title="Points", side="left"),
        legend=dict(orientation="h", y=1.12),
    )
    payload = fig.to_json()
    cache.set(cache_key, payload, CACHE_TTL)
    return payload


def get_fno_long_dashboard(key: str = "NIFTY", force: bool = False, trade_period: str = "full") -> dict:
    key = key.upper()
    results = load_strategy_results()
    trades_data = load_strategy_trades_data()
    strategy_trades = get_trades(trade_period)
    strategy_trades = sorted(
        strategy_trades,
        key=lambda t: (
            str(t.get("session_date") or ""),
            str(t.get("signal_time") or ""),
            int(t.get("trade_no") or 0),
        ),
        reverse=True,
    )
    signal = evaluate_long_signal(key, force=force)
    recent_days = get_recent_day_results(trade_period, instrument=key)
    strategy = active_long_strategy()

    instruments = []
    for k in INSTRUMENTS:
        snap = evaluate_long_signal(k, force=force) if k != key else signal
        instruments.append({
            "key": k,
            "name": INSTRUMENTS[k]["name"],
            "ltp": snap.get("ltp"),
            "status": snap.get("status"),
            "ml_prob": snap.get("ml_prob"),
            "side": snap.get("side"),
        })

    wins = sum(1 for t in strategy_trades if t.get("result") == "WIN")
    losses = len(strategy_trades) - wins
    total_pnl = sum(float(t.get("pnl_inr") or 0) for t in strategy_trades)

    # Summary from results file if present
    champ = results.get("champion") or {}
    full_sum = {}
    if isinstance(champ.get("full"), dict):
        full_sum = champ["full"]
    elif results.get("full_period", {}).get("summary"):
        full_sum = results["full_period"]["summary"]
    oos_sum = {}
    if isinstance(champ.get("oos"), dict):
        oos_sum = champ["oos"]
    elif results.get("oos_period", {}).get("summary"):
        oos_sum = results["oos_period"]["summary"]

    filters = results.get("filters") or strategy
    return {
        "signal": signal,
        "instruments": instruments,
        "results": results,
        "trades_data": trades_data,
        "strategy_trades": strategy_trades,
        "trade_period": trade_period,
        "recent_days": recent_days,
        "trade_summary": {
            "count": len(strategy_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(strategy_trades) * 100, 1) if strategy_trades else 0,
            "total_pnl": round(total_pnl, 2),
            "backtest": full_sum,
            "oos": oos_sum,
            "avg_daily": full_sum.get("avg_daily"),
        },
        "strategy_name": results.get("selected") or strategy.get("name", "Elite ML Long v1"),
        "strategy_filters": filters,
        "top_features": load_model_meta().get("top_features", {}),
        "updated_at": signal.get("updated_at"),
        "oos_summary": oos_sum,
        "full_summary": full_sum,
    }
