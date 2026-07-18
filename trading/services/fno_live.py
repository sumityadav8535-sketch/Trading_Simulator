"""
Live F&O dashboard: real-time ML signals, charts, and backtest stats.
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
import yfinance as yf
from django.core.cache import cache
from django.utils import timezone

from trading.services.fno_checklist import evaluate_soft_checklist, paper_losses_today
from trading.services.fno_engine import (
    CAPITAL,
    FEATURE_COLS,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    STRATEGY,
    TARGET_R,
    enrich_features,
    loose_short_signal,
    lots_for_risk,
    max_lots,
    normalize_df,
    passes_strategy_filters,
    sig_ema_short,
)
from trading.services.fno_results import (
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
DATA_DIR = ROOT / "data" / "intraday_fno"
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_model.pkl"

CACHE_TTL = 30


def _now_ist() -> datetime:
    return datetime.now(IST)


def fetch_instrument_bars(key: str, force: bool = False) -> pd.DataFrame:
    key = key.upper()
    if key not in INSTRUMENTS:
        return pd.DataFrame()

    cache_key = f"fno:bars:{key}:{date.today().isoformat()}"
    if not force:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{key}.pkl"
    hist = pd.DataFrame()
    if path.exists():
        try:
            hist = pd.read_pickle(path)
        except Exception:
            hist = pd.DataFrame()

    ticker = INSTRUMENTS[key]["ticker"]
    try:
        raw = yf.download(ticker, interval="5m", period="5d", progress=False, auto_adjust=False)
        live = normalize_df(raw)
    except Exception as exc:
        logger.warning("F&O live fetch failed for %s: %s", key, exc)
        live = pd.DataFrame()

    if not live.empty and not hist.empty:
        combined = pd.concat([hist, live])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    elif not live.empty:
        combined = live
    else:
        combined = hist

    if not combined.empty:
        combined.to_pickle(path)

    cache.set(cache_key, combined, CACHE_TTL)
    return combined


@lru_cache(maxsize=1)
def _load_ml_bundle() -> Optional[dict]:
    if not MODEL_PKL.exists():
        return None
    try:
        with MODEL_PKL.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("Failed to load ML model: %s", exc)
        return None


def predict_win_prob(features: list[float]) -> tuple[Optional[float], str]:
    bundle = _load_ml_bundle()
    meta = load_model_meta()
    threshold = float(meta.get("threshold", 0.35))
    model_type = meta.get("model_type", "unknown")

    if bundle is None:
        return None, model_type

    try:
        import numpy as np
        from sklearn.pipeline import Pipeline
    except ImportError:
        # Free-tier deploys may omit scikit-learn to save disk space.
        logger.warning("scikit-learn not installed; ML win probability unavailable")
        return None, model_type

    X = np.array([features], dtype=float)

    def _proba(model, x):
        if isinstance(model, Pipeline):
            return float(model.predict_proba(x)[0, 1])
        return float(model.predict_proba(x)[0, 1])

    model_type = bundle.get("model_type", model_type)
    if model_type == "ensemble" and "models" in bundle:
        probs = [_proba(m, X) for m in bundle["models"].values()]
        return float(sum(probs) / len(probs)), model_type

    model = bundle.get("model")
    if model is None:
        return None, model_type
    return _proba(model, X), model_type


def _session_bars_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    today = _now_ist().date()
    sess = df[df.index.date == today]
    return sess if not sess.empty else df.tail(80)


def _to_ist(ts) -> datetime:
    """Normalize a bar timestamp to timezone-aware IST."""
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=IST)
        return ts.astimezone(IST)
    return _now_ist()


def last_completed_bar_iloc(df: pd.DataFrame, now: Optional[datetime] = None, bar_minutes: int = 5) -> int:
    """
    Index of the last *fully closed* 5m bar.

    Yahoo/live feeds expose the currently forming bar as the last row. Paper and
    backtest must not trade on that incomplete candle (OHLC/RSI/ML change until close).
    A bar labeled T covers [T, T+bar_minutes); it is complete once now >= T+bar_minutes.
    """
    if df is None or df.empty:
        return -1
    now = now or _now_ist()
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    last_i = len(df) - 1
    ts = _to_ist(df.index[last_i])
    bar_end = ts + timedelta(minutes=bar_minutes)
    if now < bar_end and last_i > 0:
        return last_i - 1
    return last_i


def evaluate_signal(key: str, force: bool = False) -> dict:
    key = key.upper()
    inst = INSTRUMENTS.get(key, INSTRUMENTS["NIFTY"])
    market = get_market_status()
    strategy = STRATEGY
    threshold = float(strategy.get("ml_threshold", 0.58))

    df = enrich_features(fetch_instrument_bars(key, force=force))
    if df.empty or len(df) < 55:
        return {
            "instrument": key,
            "name": inst["name"],
            "status": "no_data",
            "message": "Insufficient bar history for indicators",
            "market": asdict(market),
        }

    # Live LTP from newest bar; strategy decision from last completed bar only
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
        }
    row = df.iloc[sig_i]
    ts = df.index[sig_i]
    bar_time = ts.strftime("%H:%M IST") if hasattr(ts, "strftime") else str(ts)
    bar_forming = sig_i < (len(df) - 1)

    ema_aligned = bool(row["ema_9"] < row["ema_21"] < row["ema_50"])
    below_vwap = bool(pd.notna(row.get("vwap")) and row["close"] < row["vwap"])
    base_sig = sig_ema_short(row, TARGET_R)
    is_loose = loose_short_signal(row)
    now_t = _now_ist().time()
    # Entry window uses wall-clock time (not the signal bar's clock alone)
    in_entry_window = MARKET_OPEN <= now_t <= NO_ENTRY_AFTER
    # Also require the signal bar itself to fall inside the strategy session
    bar_in_window = MARKET_OPEN <= ts.time() <= NO_ENTRY_AFTER

    features = [float(row.get(c, float("nan"))) for c in FEATURE_COLS]
    has_nan = any(pd.isna(v) for v in features)
    meta = load_model_meta()
    prob, model_type = predict_win_prob(features) if not has_nan else (None, meta.get("model_type", "random_forest"))

    stop = target = risk_pts = lots = 0.0
    if base_sig:
        stop = base_sig["stop"]
        target = base_sig["target"]
        risk_pts = base_sig["risk_pts"]
    elif is_loose:
        stop = float(row["ema_21"])
        risk_pts = stop - float(row["close"])
        if risk_pts > 0:
            target = float(row["close"] - risk_pts * TARGET_R)

    if risk_pts > 0:
        max_l = max_lots(CAPITAL, inst["mis_margin"])
        lots = lots_for_risk(CAPITAL, RISK_PCT, risk_pts, inst["lot_size"], max_l)

    opt_pass = (
        prob is not None
        and risk_pts > 0
        and bar_in_window
        and passes_strategy_filters(row, ts, prob, risk_pts, strategy)
    )
    ml_pass = prob is not None and prob >= threshold
    base_pass = base_sig is not None

    # Soft checklist (score ≥ 6) — gates ACTIVE for live + paper
    losses_today = paper_losses_today()
    checklist = evaluate_soft_checklist(row, ts, prob, losses_today=losses_today)
    checklist_pass = bool(checklist.take)

    if not in_entry_window:
        status, message = "closed_window", "Outside entry window (9:15–14:45 IST)"
    elif not bar_in_window:
        status, message = "closed_window", f"Signal bar {bar_time} outside entry window"
    elif is_loose and opt_pass and checklist_pass:
        status, message = (
            "active",
            f"{strategy['name']} — checklist {checklist.score}/6 PASS · "
            f"P(win) {prob:.0%}, risk {risk_pts:.1f}pts"
            + (" · completed bar" if bar_forming else ""),
        )
    elif is_loose and opt_pass and not checklist_pass:
        status, message = (
            "watch",
            f"Base setup OK but soft checklist fail ({checklist.score}/6)"
            + (f" — {checklist.reasons[0]}" if checklist.reasons else ""),
        )
    elif is_loose and ml_pass and risk_pts > strategy.get("max_risk_pts", 22):
        status, message = "watch", f"ML OK but stop too wide ({risk_pts:.1f}pts > {strategy['max_risk_pts']})"
    elif is_loose and prob is not None and prob < threshold:
        status, message = "watch", f"ML prob {prob:.0%} < {threshold:.0%} threshold"
    elif base_pass and prob is not None:
        status, message = "watch", f"EMA setup valid but filters not met (prob {prob:.0%})"
    elif is_loose and prob is not None:
        status, message = "watch", f"Loose short — awaiting optimized filter pass"
    elif ema_aligned and below_vwap:
        status, message = "setup", "Bearish structure forming — awaiting ML confirmation"
    else:
        status, message = "neutral", "No short setup on last completed bar"

    est_risk_inr = round(risk_pts * inst["lot_size"] * lots, 0) if lots else 0
    est_target_inr = round(risk_pts * TARGET_R * inst["lot_size"] * lots, 0) if lots else 0
    ltp = round(float(live_row["close"]), 2)
    live_open = round(float(live_row["open"]), 2)
    # Entry reference matches backtest: next bar open after the signal bar when available
    entry_ref = live_open if bar_forming else round(float(row["close"]), 2)

    return {
        "instrument": key,
        "name": inst["name"],
        "ticker": inst["ticker"],
        "lot_size": inst["lot_size"],
        "mis_margin": inst["mis_margin"],
        "status": status,
        "message": message,
        "side": "SHORT" if status == "active" else None,
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
        "macd_hist": round(float(row["macd_hist"]), 2) if pd.notna(row.get("macd_hist")) else None,
        "bb_pct": round(float(row["bb_pct"]), 2) if pd.notna(row.get("bb_pct")) else None,
        "range_pos": round(float(row["range_pos"]), 2) if pd.notna(row.get("range_pos")) else None,
        "ema_aligned": ema_aligned,
        "below_vwap": below_vwap,
        "base_signal": base_pass,
        "loose_candidate": is_loose,
        "ml_prob": round(prob, 3) if prob is not None else None,
        "ml_threshold": threshold,
        "strategy": strategy,
        "strategy_pass": opt_pass,
        "checklist_pass": checklist_pass,
        "checklist": checklist.as_dict(),
        "ml_model": model_type,
        "ml_available": prob is not None,
        "stop": round(stop, 2) if stop else None,
        "target": round(target, 2) if target else None,
        "risk_pts": round(risk_pts, 2) if risk_pts else None,
        "target_r": TARGET_R,
        "lots": lots,
        "risk_inr": est_risk_inr,
        "target_inr": est_target_inr,
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "in_entry_window": in_entry_window,
        "market": asdict(market),
        "updated_at": timezone.now().strftime("%H:%M:%S IST"),
    }


def build_fno_chart_json(key: str, force: bool = False) -> str:
    import plotly.graph_objects as go

    key = key.upper()
    cache_key = f"fno:chart:{key}:{date.today().isoformat()}"
    if not force:
        cached = cache.get(cache_key)
        if cached:
            return cached

    df = enrich_features(fetch_instrument_bars(key, force=force))
    sess = _session_bars_today(df)
    if sess.empty:
        payload = json.dumps({"data": [], "layout": {"title": f"No F&O data for {key}"}})
        cache.set(cache_key, payload, CACHE_TTL)
        return payload

    inst = INSTRUMENTS[key]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=sess.index,
            open=sess["open"],
            high=sess["high"],
            low=sess["low"],
            close=sess["close"],
            name=key,
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

    sig = evaluate_signal(key, force=force)
    if sig.get("stop"):
        fig.add_hline(y=sig["stop"], line_dash="dash", line_color="#ef4444", annotation_text="Stop")
    if sig.get("target"):
        fig.add_hline(y=sig["target"], line_dash="dot", line_color="#22c55e", annotation_text="Target")

    fig.update_layout(
        title=f"{inst['name']} — 5m intraday (index proxy)",
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


def get_fno_dashboard(key: str = "NIFTY", force: bool = False, trade_period: str = "full") -> dict:
    key = key.upper()
    results = load_strategy_results()
    trades_data = load_strategy_trades_data()
    strategy_trades = get_trades(trade_period)
    # Newest first so yesterday/today are at the top of the live trade log
    strategy_trades = sorted(
        strategy_trades,
        key=lambda t: (
            str(t.get("session_date") or ""),
            str(t.get("signal_time") or ""),
            int(t.get("trade_no") or 0),
        ),
        reverse=True,
    )
    signal = evaluate_signal(key, force=force)
    recent_days = get_recent_day_results(trade_period, instrument=key)

    instruments = []
    for k in INSTRUMENTS:
        snap = evaluate_signal(k, force=force) if k != key else signal
        instruments.append({
            "key": k,
            "name": INSTRUMENTS[k]["name"],
            "ltp": snap.get("ltp"),
            "status": snap.get("status"),
            "ml_prob": snap.get("ml_prob"),
            "side": snap.get("side"),
        })

    fp = trades_data.get("full_period", {})
    trade_summary = fp.get("summary", {})
    wins = sum(1 for t in strategy_trades if t.get("result") == "WIN")
    losses = len(strategy_trades) - wins
    total_pnl = sum(t.get("pnl_inr", 0) for t in strategy_trades)

    filters = results.get("filters", STRATEGY)
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
            "backtest": trade_summary,
            "avg_daily": trade_summary.get("avg_daily"),
        },
        "strategy_name": results.get("selected", STRATEGY["name"]),
        "strategy_filters": filters,
        "top_features": load_model_meta().get("top_features", {}),
        "updated_at": signal.get("updated_at"),
    }