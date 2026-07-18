"""
Elite ML Short v2 F&O backtest utilities.

The live model is FROZEN: the web UI cannot retrain or overwrite
``intraday_fno_ml_model.pkl``. Historical backtest JSON is left as-is for the
dashboard; live signals always load the frozen pickle.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# When True, public run APIs refuse to retrain / rewrite the model or trade log.
MODEL_FROZEN = True

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_model.pkl"
MODEL_META = ROOT / "data" / "intraday_fno_ml_model.json"
FNO_DIR = ROOT / "data" / "intraday_fno"

_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "message": "Model frozen — retrain disabled",
    "progress": {"step": 0, "total": 5, "label": "Frozen"},
    "result_summary": None,
    "data_range": None,
    "error": None,
    "frozen": True,
}


def get_backtest_status() -> dict:
    st = dict(_status)
    st["frozen"] = MODEL_FROZEN
    if MODEL_FROZEN and not st.get("running"):
        st["message"] = st.get("message") or "Model frozen — retrain disabled"
    if MODEL_META.exists():
        try:
            meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
            st["model_trained_at"] = meta.get("trained_at")
            st["model_data_range"] = meta.get("data_range")
            st["model_threshold"] = meta.get("threshold")
        except Exception:
            pass
    return st


def _frozen_reject(action: str = "backtest") -> dict:
    msg = (
        "F&O ML model is frozen. Retrain / rewrite of model and backtest "
        "artifacts is disabled so live signals stay stable. "
        "Use paper trading for forward results."
    )
    logger.info("Rejected %s: model frozen", action)
    return {
        "ok": False,
        "started": False,
        "frozen": True,
        "error": msg,
        "message": msg,
        "status": get_backtest_status(),
    }


def run_fno_backtest(end_date: Optional[date] = None) -> dict:
    """Synchronous backtest — blocked while model is frozen."""
    return _frozen_reject("run_fno_backtest")


def start_fno_backtest_async(end_date: Optional[date] = None) -> dict:
    """Kick off backtest — blocked while model is frozen."""
    return _frozen_reject("start_fno_backtest_async")
