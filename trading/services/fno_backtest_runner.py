"""
One-click Elite ML Short v2 F&O backtest using stored 5m index data through today.
Writes the same JSON artifacts the F&O Live page reads.
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from django.utils import timezone

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "intraday_fno_optimized.json"
TRADES_OUT = ROOT / "data" / "intraday_fno_optimized_trades.json"
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_model.pkl"
MODEL_META = ROOT / "data" / "intraday_fno_ml_model.json"
FNO_DIR = ROOT / "data" / "intraday_fno"

_lock = threading.Lock()
_status_lock = threading.Lock()
_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "message": "Idle",
    "progress": {"step": 0, "total": 5, "label": ""},
    "result_summary": None,
    "data_range": None,
    "error": None,
}


def get_backtest_status() -> dict:
    with _status_lock:
        return dict(_status)


def _set_status(**kwargs) -> None:
    with _status_lock:
        _status.update(kwargs)


def _step(step: int, total: int, label: str) -> None:
    _set_status(
        message=label,
        progress={"step": step, "total": total, "label": label},
    )


def _data_range_info() -> dict:
    info = {"instruments": {}, "through": None}
    lasts = []
    for key in ("NIFTY", "BANKNIFTY"):
        path = FNO_DIR / f"{key}.pkl"
        if not path.exists():
            info["instruments"][key] = {"bars": 0, "last_date": None, "first_date": None}
            continue
        try:
            import pandas as pd

            df = pd.read_pickle(path)
            if df is None or df.empty:
                info["instruments"][key] = {"bars": 0, "last_date": None, "first_date": None}
                continue
            first = df.index.min()
            last = df.index.max()
            first_d = first.date().isoformat() if hasattr(first, "date") else str(first)[:10]
            last_d = last.date().isoformat() if hasattr(last, "date") else str(last)[:10]
            lasts.append(last.date() if hasattr(last, "date") else None)
            info["instruments"][key] = {
                "bars": int(len(df)),
                "first_date": first_d,
                "last_date": last_d,
            }
        except Exception as exc:
            info["instruments"][key] = {"error": str(exc)}
    valid = [d for d in lasts if d]
    info["through"] = max(valid).isoformat() if valid else None
    return info


def _execute_backtest(end_date: Optional[date] = None) -> dict:
    """Run Elite ML Short v2 backtest. Caller must hold `_lock`."""
    # Local imports keep Django startup light and match script paths
    from scripts.intraday_fno_ml_enhance import (  # noqa: WPS433
        FEATURE_COLS,
        backtest_signals,
        build_dataset,
        enrich_features,
        feature_importance,
        load_instrument,
        model_proba,
        time_split,
        train_models,
    )
    from scripts.intraday_fno_optimize import (  # noqa: WPS433
        ELITE_CFG,
        build_filtered_signals,
    )
    from trading.services.fno_live import _load_ml_bundle

    total_steps = 5
    end_date = end_date or datetime.now(IST).date()

    _set_status(
        running=True,
        started_at=timezone.now().isoformat(),
        finished_at=None,
        message="Starting F&O backtest…",
        progress={"step": 0, "total": total_steps, "label": "Starting"},
        result_summary=None,
        data_range=None,
        error=None,
    )

    try:
        _step(1, total_steps, "Loading NIFTY 5m history…")
        df = load_instrument("NIFTY", force=False)
        if df is None or df.empty:
            raise RuntimeError("No NIFTY 5m data found. Use “Fetch history to today” first.")

        # Clip to end_date (inclusive) so backtest matches “through today”
        df = df[df.index.date <= end_date]
        if df.empty:
            raise RuntimeError(f"No NIFTY bars on or before {end_date.isoformat()}")

        first_d = df.index.min().date().isoformat()
        last_d = df.index.max().date().isoformat()
        data_range = {
            "instrument": "NIFTY",
            "first_date": first_d,
            "last_date": last_d,
            "bars": int(len(df)),
            "target_end": end_date.isoformat(),
        }
        _set_status(data_range=data_range)

        _step(2, total_steps, f"Building features & labels ({first_d} → {last_d})…")
        df = enrich_features(df)
        dset = build_dataset(df, use_loose=True).dropna(subset=FEATURE_COLS)
        if dset.empty or len(dset) < 30:
            raise RuntimeError(
                f"Not enough labeled samples ({len(dset)}). Need more 5m history."
            )

        train, test, train_dates, test_dates = time_split(dset, 0.7)
        if train.empty:
            raise RuntimeError("Train split is empty — not enough sessions.")

        _step(3, total_steps, "Training random forest model…")
        X_tr = train[FEATURE_COLS].values[: int(len(train) * 0.8)]
        y_tr = train["label"].values[: int(len(train) * 0.8)]
        if len(X_tr) < 20 or len(set(y_tr)) < 2:
            raise RuntimeError("Insufficient training labels for ML model.")

        models = train_models(X_tr, y_tr)
        rf = models["random_forest"]
        all_probs = model_proba(rf, dset[FEATURE_COLS].values)

        _step(4, total_steps, "Running Elite ML Short v2 backtest…")
        cfg = ELITE_CFG
        sigs = build_filtered_signals(df, dset, all_probs, cfg)
        stats, trades = backtest_signals(df, sigs, cfg.name, "full", record_trades=True)
        losses = sum(1 for t in trades if t["result"] == "LOSS")

        oos_stats = None
        oos_trades: list = []
        oos_losses = 0
        if test_dates:
            test_df = df[df["session_date"] >= min(test_dates)]
            test_dset = dset[dset["session_date"].isin(set(test_dates))].copy()
            if not test_dset.empty:
                oos_sigs = build_filtered_signals(
                    test_df,
                    test_dset,
                    model_proba(rf, test_dset[FEATURE_COLS].values),
                    cfg,
                )
                oos_stats, oos_trades = backtest_signals(
                    test_df, oos_sigs, f"{cfg.name} OOS", "OOS", record_trades=True
                )
                oos_losses = sum(1 for t in oos_trades if t["result"] == "LOSS")

        _step(5, total_steps, "Saving results…")
        filters = {
            "ml_threshold": cfg.ml_th,
            "adx_min": cfg.adx_min,
            "range_pos_min": cfg.range_pos_min,
            "max_risk_pts": cfg.max_risk_pts,
            "min_risk_pts": cfg.min_risk_pts,
            "require_ema_stack": cfg.require_ema_stack,
            "hour_start": cfg.hour_start,
            "hour_end": cfg.hour_end,
            "rsi_min": cfg.rsi_min,
            "rsi_max": cfg.rsi_max,
        }

        payload = {
            "selected": cfg.name,
            "generated_at": timezone.now().isoformat(),
            "data_range": data_range,
            "filters": filters,
            "full_period": {
                "summary": stats.__dict__,
                "losses": losses,
                "wins": stats.trades - losses,
                "trades": trades,
            },
            "oos": {
                "summary": oos_stats.__dict__ if oos_stats else {},
                "losses": oos_losses,
                "trades": oos_trades,
            },
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        TRADES_OUT.write_text(
            json.dumps(
                {
                    "strategy": cfg.name,
                    "filters": filters,
                    "generated_at": payload["generated_at"],
                    "data_range": data_range,
                    "full_period": payload["full_period"],
                    "oos": payload["oos"],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # Persist model so live signals match this backtest run
        imp = feature_importance(rf, FEATURE_COLS)
        meta = {
            "model_type": "random_forest",
            "threshold": cfg.ml_th,
            "features": FEATURE_COLS,
            "strategy_name": cfg.name,
            "strategy_filters": filters,
            "trained_at": payload["generated_at"],
            "data_range": data_range,
            "top_features": imp,
        }
        MODEL_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        with MODEL_PKL.open("wb") as fh:
            pickle.dump(
                {
                    "model_type": "random_forest",
                    "model": rf,
                    "threshold": cfg.ml_th,
                    "features": FEATURE_COLS,
                    "strategy_name": cfg.name,
                    "strategy_filters": filters,
                },
                fh,
            )

        # Drop cached ML bundle so next live poll loads the new model
        try:
            _load_ml_bundle.cache_clear()
        except Exception:
            pass

        summary = {
            "strategy": cfg.name,
            "trades": stats.trades,
            "win_rate": stats.win_rate,
            "net_pnl": stats.net_pnl,
            "avg_daily": stats.avg_daily,
            "profit_factor": stats.profit_factor,
            "max_dd_pct": stats.max_dd_pct,
            "wins": stats.trades - losses,
            "losses": losses,
            "oos_trades": oos_stats.trades if oos_stats else 0,
            "oos_avg_daily": oos_stats.avg_daily if oos_stats else None,
            "data_range": data_range,
        }

        msg = (
            f"Done {first_d} → {last_d}: {stats.trades} trades, "
            f"WR {stats.win_rate}%, ₹{stats.avg_daily:,.0f}/day"
        )
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=msg,
            progress={"step": total_steps, "total": total_steps, "label": "Complete"},
            result_summary=summary,
            data_range=data_range,
            error=None,
        )
        return {"ok": True, "summary": summary, "status": get_backtest_status()}

    except Exception as exc:
        logger.exception("F&O backtest failed")
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=f"Failed: {exc}",
            error=str(exc),
        )
        return {"ok": False, "error": str(exc), "status": get_backtest_status()}


def run_fno_backtest(end_date: Optional[date] = None) -> dict:
    """Synchronous backtest; rejects if already running."""
    if not _lock.acquire(blocking=False):
        return {
            "ok": False,
            "error": "A backtest is already running",
            "status": get_backtest_status(),
        }
    try:
        return _execute_backtest(end_date)
    finally:
        _lock.release()


def start_fno_backtest_async(end_date: Optional[date] = None) -> dict:
    """Kick off backtest in a daemon thread."""
    end_date = end_date or datetime.now(IST).date()
    if not _lock.acquire(blocking=False):
        return {
            "ok": False,
            "started": False,
            "error": "A backtest is already running",
            "status": get_backtest_status(),
        }

    _set_status(
        running=True,
        started_at=timezone.now().isoformat(),
        finished_at=None,
        message="Queued…",
        progress={"step": 0, "total": 5, "label": "Queued"},
        result_summary=None,
        data_range=_data_range_info(),
        error=None,
    )

    def _worker():
        try:
            _execute_backtest(end_date)
        finally:
            _lock.release()

    threading.Thread(target=_worker, name="fno-backtest", daemon=True).start()
    return {
        "ok": True,
        "started": True,
        "end_date": end_date.isoformat(),
        "message": f"Running Elite ML Short v2 backtest through {end_date.isoformat()}…",
        "status": get_backtest_status(),
        "data_range": _data_range_info(),
    }
