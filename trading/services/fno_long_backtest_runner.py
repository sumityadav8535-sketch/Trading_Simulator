"""
One-click Elite ML Long v1 backtest using stored 5m NIFTY data.
Writes data/intraday_fno_ml_long_* artifacts for the F&O Long Live page.
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
OUT = ROOT / "data" / "intraday_fno_ml_long_results.json"
TRADES_OUT = ROOT / "data" / "intraday_fno_ml_long_trades.json"
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_long_model.pkl"
MODEL_META = ROOT / "data" / "intraday_fno_ml_long_model.json"
FNO_DIR = ROOT / "data" / "intraday_fno"

_lock = threading.Lock()
_status_lock = threading.Lock()
_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "message": "Idle",
    "progress": {"step": 0, "total": 6, "label": ""},
    "result_summary": None,
    "data_range": None,
    "error": None,
}


def get_long_backtest_status() -> dict:
    with _status_lock:
        st = dict(_status)
    if MODEL_META.exists():
        try:
            meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
            st["model_trained_at"] = meta.get("trained_at")
            st["model_data_range"] = meta.get("data_range")
            st["model_threshold"] = meta.get("threshold") or (
                (meta.get("strategy_filters") or {}).get("ml_threshold")
            )
        except Exception:
            pass
    return st


def _set_status(**kwargs) -> None:
    with _status_lock:
        _status.update(kwargs)


def _step(step: int, total: int, label: str) -> None:
    _set_status(message=label, progress={"step": step, "total": total, "label": label})


def _execute_long_backtest(end_date: Optional[date] = None) -> dict:
    """Train long ML + backtest champion filters. Caller holds _lock."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
        normalize_df,
    )
    from trading.services.fno_long_checklist import evaluate_soft_checklist_long
    from trading.services.fno_long_engine import (
        STRATEGY_LONG,
        loose_long_signal,
        merge_strategy_from_meta,
        passes_long_filters,
        sig_ema_long,
    )
    from trading.services.fno_long_live import clear_ml_cache

    LOT_SIZE = INSTRUMENTS["NIFTY"]["lot_size"]
    MARGIN = INSTRUMENTS["NIFTY"]["mis_margin"]
    total_steps = 6
    end_date = end_date or datetime.now(IST).date()

    _set_status(
        running=True,
        started_at=timezone.now().isoformat(),
        finished_at=None,
        message="Starting Elite ML Long backtest…",
        progress={"step": 0, "total": total_steps, "label": "Starting"},
        result_summary=None,
        data_range=None,
        error=None,
    )

    def simulate_long(df, entry_idx, stop, target):
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
                    "exit": exit_px, "reason": "stop_hit", "exit_ts": ts,
                    "entry_ts": entry_ts,
                }
            if hi >= target:
                exit_px = target - SLIPPAGE_PTS
                return {
                    "win": True, "pnl_pts": exit_px - entry, "entry": entry,
                    "exit": exit_px, "reason": "target_hit", "exit_ts": ts,
                    "entry_ts": entry_ts,
                }
            if ts.time() >= FORCE_EXIT:
                exit_px = cl - SLIPPAGE_PTS
                return {
                    "win": exit_px > entry, "pnl_pts": exit_px - entry, "entry": entry,
                    "exit": exit_px, "reason": "eod_exit", "exit_ts": ts,
                    "entry_ts": entry_ts,
                }
            if j > entry_idx and ts.date() != entry_ts.date():
                break
        return None

    try:
        _step(1, total_steps, "Loading NIFTY 5m history…")
        path = FNO_DIR / "NIFTY.pkl"
        if not path.exists():
            raise RuntimeError("No NIFTY 5m data. Use “Fetch history to today” on F&O Live first.")
        df = normalize_df(pd.read_pickle(path))
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

        _step(2, total_steps, f"Building long features ({first_d} → {last_d})…")
        df = enrich_features(df)

        # Load prior filters if any
        prior_meta = {}
        if MODEL_META.exists():
            try:
                prior_meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
            except Exception:
                prior_meta = {}
        strategy = merge_strategy_from_meta(prior_meta)
        target_r = float(strategy.get("target_r", 1.8))

        _step(3, total_steps, "Labeling long candidates…")
        rows = []
        for i in range(50, len(df) - 2):
            row = df.iloc[i]
            ts = df.index[i]
            if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
                continue
            if not loose_long_signal(row):
                continue
            if pd.isna(row.get("ema_21")):
                continue
            stop = float(row["ema_21"])
            risk = float(row["close"]) - stop
            if risk <= 0:
                continue
            entry_idx = i + 1
            if entry_idx >= len(df) or df.index[entry_idx].date() != ts.date():
                continue
            entry0 = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
            stop_pts = entry0 - stop
            if stop_pts <= 0:
                continue
            true_target = entry0 + stop_pts * target_r
            out = simulate_long(df, entry_idx, stop, true_target)
            if not out:
                continue
            feat = {c: row.get(c, np.nan) for c in FEATURE_COLS}
            if any(pd.isna(feat[c]) for c in FEATURE_COLS):
                continue
            feat["label"] = 1 if out["win"] else 0
            feat["pnl_pts"] = out["pnl_pts"]
            feat["timestamp"] = ts
            feat["session_date"] = row["session_date"]
            feat["stop"] = stop
            feat["target"] = true_target
            feat["risk_pts"] = stop_pts
            feat["i"] = i
            rows.append(feat)

        dset = pd.DataFrame(rows)
        if dset.empty or len(dset) < 30:
            raise RuntimeError(f"Not enough long samples ({len(dset)}). Need more 5m history.")

        dates = sorted(dset["session_date"].unique())
        cut = max(1, int(len(dates) * 0.7))
        train_dates = set(dates[:cut])
        test_dates = set(dates[cut:])
        train = dset[dset["session_date"].isin(train_dates)]
        test = dset[dset["session_date"].isin(test_dates)]

        _step(4, total_steps, "Training long ensemble (RF/GB/HGB)…")
        X_tr = train[FEATURE_COLS].values
        y_tr = train["label"].values
        if len(X_tr) < 20 or len(set(y_tr)) < 2:
            raise RuntimeError("Insufficient training labels for long ML model.")

        models = {
            "gradient_boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=12, random_state=42,
                )),
            ]),
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=350, max_depth=7, min_samples_leaf=10,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ]),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_depth=5, learning_rate=0.06, max_iter=280,
                min_samples_leaf=15, l2_regularization=0.08, random_state=42,
            ),
        }
        for m in models.values():
            m.fit(X_tr, y_tr)

        def ens_proba(X):
            return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)

        dset = dset.copy()
        dset["prob"] = ens_proba(dset[FEATURE_COLS].values)

        # Map probs onto df rows by timestamp for filter walk
        prob_by_ts = {r["timestamp"]: float(r["prob"]) for _, r in dset.iterrows()}
        stop_by_ts = {r["timestamp"]: float(r["stop"]) for _, r in dset.iterrows()}
        risk_by_ts = {r["timestamp"]: float(r["risk_pts"]) for _, r in dset.iterrows()}

        def run_backtest(session_set: set | None, period: str):
            equity = CAPITAL
            peak = equity
            max_dd = 0.0
            wins = gp = gl = 0.0
            trades = 0
            day_count: dict = {}
            daily: dict = {}
            losses_today: dict = {}
            log = []

            for i in range(50, len(df) - 2):
                row = df.iloc[i]
                ts = df.index[i]
                if session_set is not None and row["session_date"] not in session_set:
                    continue
                if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
                    continue
                if not loose_long_signal(row):
                    continue
                if ts not in prob_by_ts:
                    continue
                prob = prob_by_ts[ts]
                stop = stop_by_ts[ts]
                risk_pts = risk_by_ts[ts]
                if not passes_long_filters(row, ts, prob, risk_pts, strategy):
                    continue
                sess = row["session_date"]
                if day_count.get(sess, 0) >= int(strategy.get("max_trades_per_day", TRADES_PER_DAY)):
                    continue
                if strategy.get("soft_checklist", True):
                    cl = evaluate_soft_checklist_long(
                        row, ts, prob, losses_today=losses_today.get(sess, 0)
                    )
                    if not cl.take:
                        continue

                entry_idx = i + 1
                if entry_idx >= len(df) or df.index[entry_idx].date() != ts.date():
                    continue
                entry0 = float(df.iloc[entry_idx]["open"]) + SLIPPAGE_PTS
                sp = entry0 - stop
                if sp <= 0:
                    continue
                true_target = entry0 + sp * target_r
                out = simulate_long(df, entry_idx, stop, true_target)
                if not out:
                    continue

                max_l = max_lots(equity, MARGIN)
                lots = lots_for_risk(equity, RISK_PCT, sp, LOT_SIZE, max_l)
                if lots <= 0:
                    continue
                pnl = out["pnl_pts"] * LOT_SIZE * lots
                equity += pnl
                trades += 1
                day_count[sess] = day_count.get(sess, 0) + 1
                daily[sess] = daily.get(sess, 0.0) + pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
                if pnl > 0:
                    wins += 1
                    gp += pnl
                else:
                    gl += abs(pnl)
                    losses_today[sess] = losses_today.get(sess, 0) + 1

                log.append({
                    "trade_no": trades,
                    "strategy": strategy.get("name", "Elite ML Long v1"),
                    "period": period,
                    "instrument": "NIFTY",
                    "side": "LONG",
                    "session_date": str(sess),
                    "signal_time": ts.isoformat(),
                    "entry_time": out["entry_ts"].isoformat(),
                    "entry_price": round(out["entry"], 2),
                    "stop": round(stop, 2),
                    "target": round(true_target, 2),
                    "risk_pts": round(sp, 2),
                    "target_r": target_r,
                    "lots": lots,
                    "lot_size": LOT_SIZE,
                    "ml_prob": round(prob, 3),
                    "exit_time": out["exit_ts"].isoformat(),
                    "exit_price": round(out["exit"], 2),
                    "exit_reason": out["reason"],
                    "pnl_pts": round(out["pnl_pts"], 2),
                    "pnl_inr": round(pnl, 2),
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "equity_after": round(equity, 2),
                })

            wr = 100.0 * wins / trades if trades else 0.0
            pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
            return {
                "trades": trades,
                "wins": int(wins),
                "losses": trades - int(wins),
                "win_rate": round(wr, 2),
                "net_pnl": round(equity - CAPITAL, 2),
                "profit_factor": round(pf, 2),
                "max_dd_pct": round(max_dd, 2),
                "final_equity": round(equity, 2),
                "avg_daily": round((equity - CAPITAL) / max(len(daily), 1), 2),
                "trade_log": log,
            }

        _step(5, total_steps, "Running full + OOS long backtests…")
        full_bt = run_backtest(None, "full")
        oos_bt = run_backtest(test_dates, "oos") if test_dates else {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "net_pnl": 0,
            "profit_factor": 0, "max_dd_pct": 0, "final_equity": CAPITAL,
            "avg_daily": 0, "trade_log": [],
        }

        _step(6, total_steps, "Saving model & results…")
        filters = {
            "ml_threshold": strategy.get("ml_threshold", 0.72),
            "require_ema_stack": strategy.get("require_ema_stack", True),
            "adx_min": strategy.get("adx_min", 0),
            "max_risk_pts": strategy.get("max_risk_pts", 35),
            "min_risk_pts": strategy.get("min_risk_pts", 1.5),
            "rsi_min": strategy.get("rsi_min", 40),
            "rsi_max": strategy.get("rsi_max", 75),
            "hour_start": strategy.get("hour_start", 9),
            "hour_end": strategy.get("hour_end", 12),
            "range_pos_min": strategy.get("range_pos_min", 0),
            "max_path_pct": strategy.get("max_path_pct", 0.8),
            "min_path_pct": strategy.get("min_path_pct", -0.55),
            "require_di_bull": strategy.get("require_di_bull", True),
            "require_macd_hist_pos": strategy.get("require_macd_hist_pos", False),
            "soft_checklist": strategy.get("soft_checklist", True),
            "soft_checklist_min_score": 6,
            "target_r": target_r,
            "max_trades_per_day": strategy.get("max_trades_per_day", 4),
        }

        with MODEL_PKL.open("wb") as fh:
            pickle.dump({
                "models": models,
                "feature_cols": FEATURE_COLS,
                "strategy_name": strategy.get("name", "Elite ML Long v1"),
                "filters": filters,
                "target_r": target_r,
            }, fh)

        meta = {
            "model_type": "ensemble_rf_gb_hgb",
            "threshold": filters["ml_threshold"],
            "features": FEATURE_COLS,
            "strategy_name": strategy.get("name", "Elite ML Long v1"),
            "strategy_filters": filters,
            "trained_at": timezone.now().isoformat(),
            "data_range": {
                **data_range,
                "train_sessions": len(train_dates),
                "oos_sessions": len(test_dates),
            },
            "champion": {
                "name": strategy.get("name", "Elite ML Long v1"),
                "full": {k: v for k, v in full_bt.items() if k != "trade_log"},
                "oos": {k: v for k, v in oos_bt.items() if k != "trade_log"},
            },
        }
        MODEL_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        payload = {
            "selected": strategy.get("name", "Elite ML Long v1"),
            "generated_at": meta["trained_at"],
            "selection_method": "web_backtest_retrain",
            "data_range": meta["data_range"],
            "filters": filters,
            "champion": meta["champion"],
            "full_period": {
                "summary": meta["champion"]["full"],
                "wins": full_bt["wins"],
                "losses": full_bt["losses"],
                "trades": full_bt["trade_log"],
            },
            "oos_period": {
                "summary": meta["champion"]["oos"],
                "trades": oos_bt["trade_log"],
            },
        }
        OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        TRADES_OUT.write_text(
            json.dumps(full_bt["trade_log"], indent=2, default=str), encoding="utf-8"
        )

        clear_ml_cache()

        summary = {
            "name": strategy.get("name", "Elite ML Long v1"),
            "trades": full_bt["trades"],
            "win_rate": full_bt["win_rate"],
            "net_pnl": full_bt["net_pnl"],
            "avg_daily": full_bt["avg_daily"],
            "profit_factor": full_bt["profit_factor"],
            "max_dd_pct": full_bt["max_dd_pct"],
            "oos_trades": oos_bt["trades"],
            "oos_win_rate": oos_bt["win_rate"],
            "oos_net_pnl": oos_bt["net_pnl"],
        }
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=(
                f"Done: {full_bt['trades']} trades · WR {full_bt['win_rate']}% · "
                f"₹{full_bt['net_pnl']:,.0f} · OOS WR {oos_bt['win_rate']}%"
            ),
            progress={"step": total_steps, "total": total_steps, "label": "Complete"},
            result_summary=summary,
            data_range=data_range,
            error=None,
        )
        return {"ok": True, "summary": summary, "data_range": data_range}

    except Exception as exc:
        logger.exception("Long F&O backtest failed")
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=f"Failed: {exc}",
            error=str(exc),
        )
        return {"ok": False, "error": str(exc)}


def start_fno_long_backtest_async(end_date: Optional[date] = None) -> dict:
    if not _lock.acquire(blocking=False):
        return {"ok": False, "started": False, "error": "Backtest already running"}
    if get_long_backtest_status().get("running"):
        _lock.release()
        return {"ok": False, "started": False, "error": "Backtest already running"}

    def _worker():
        try:
            _execute_long_backtest(end_date=end_date)
        finally:
            _lock.release()

    t = threading.Thread(target=_worker, name="fno-long-backtest", daemon=True)
    t.start()
    return {
        "ok": True,
        "started": True,
        "message": f"Running Elite ML Long v1 backtest through {(end_date or datetime.now(IST).date()).isoformat()}…",
    }
