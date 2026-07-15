"""
Optimize ML F&O short strategy — analyze loss drivers and backtest tighter filters.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import time as dt_time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.intraday_fno_ml_enhance import (  # noqa: E402
    CAPITAL,
    FEATURE_COLS,
    LOT_SIZE,
    MARGIN,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    TARGET_DAILY,
    TARGET_R,
    TRADES_PER_DAY,
    _ts_key,
    backtest_signals,
    build_dataset,
    enrich_features,
    load_instrument,
    model_proba,
    simulate_short_trade,
    time_split,
    train_models,
)
from scripts.intraday_fno_search import lots_for_risk, max_lots  # noqa: E402

OUT = ROOT / "data" / "intraday_fno_optimized.json"
TRADES_OUT = ROOT / "data" / "intraday_fno_optimized_trades.json"
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_model.pkl"


@dataclass
class FilterConfig:
    name: str
    ml_th: float = 0.35
    adx_min: float = 0.0
    range_pos_min: float = 0.0
    max_risk_pts: float = 999.0
    min_risk_pts: float = 0.0
    hour_start: int = 9
    hour_end: int = 14
    require_ema_stack: bool = False
    rsi_min: float = 0.0
    rsi_max: float = 100.0
    max_trades_per_day: int = 4


def apply_filters(row, ts, prob: float, cfg: FilterConfig, risk_pts: float) -> bool:
    if prob < cfg.ml_th:
        return False
    if cfg.require_ema_stack:
        if not (row["ema_9"] < row["ema_21"] < row["ema_50"]):
            return False
        if pd.notna(row.get("vwap")) and row["close"] >= row["vwap"]:
            return False
        if row["rsi"] > 50:
            return False
    adx = row.get("adx")
    if cfg.adx_min > 0 and (pd.isna(adx) or adx < cfg.adx_min):
        return False
    rp = row.get("range_pos")
    if cfg.range_pos_min > 0 and (pd.isna(rp) or rp < cfg.range_pos_min):
        return False
    if risk_pts > cfg.max_risk_pts or risk_pts < cfg.min_risk_pts:
        return False
    h = ts.hour
    if h < cfg.hour_start or h > cfg.hour_end:
        return False
    if not (cfg.rsi_min <= row["rsi"] <= cfg.rsi_max):
        return False
    return True


def build_filtered_signals(
    df: pd.DataFrame,
    dset: pd.DataFrame,
    probs: np.ndarray,
    cfg: FilterConfig,
) -> pd.DataFrame:
    dset = dset.copy()
    dset["prob"] = probs
    rows = []
    ts_to_idx = {_ts_key(ts): i for i, ts in enumerate(df.index)}

    for _, sig in dset.iterrows():
        prob = sig["prob"]
        ts = sig["timestamp"]
        key = _ts_key(ts)
        if key not in ts_to_idx:
            continue
        i = ts_to_idx[key]
        row = df.iloc[i]
        stop, target = float(sig["stop"]), float(sig["target"])
        entry_i = i + 1
        if entry_i >= len(df):
            continue
        entry_est = float(df.iloc[entry_i]["open"])
        risk_pts = stop - entry_est
        if risk_pts <= 0:
            continue
        if not apply_filters(row, df.index[i], prob, cfg, risk_pts):
            continue
        rows.append({
            "timestamp": ts,
            "stop": stop,
            "target": target,
            "prob": prob,
        })
    return pd.DataFrame(rows)


ELITE_CFG = FilterConfig("Elite ML Short v2", ml_th=0.58, max_risk_pts=22)


def main():
    print("Elite ML Short v2 — backtest & trade log")
    print(f"Capital Rs {CAPITAL:,.0f}\n")

    df = enrich_features(load_instrument("NIFTY"))
    dset = build_dataset(df, use_loose=True).dropna(subset=FEATURE_COLS)
    train, test, train_dates, test_dates = time_split(dset, 0.7)

    X_tr = train[FEATURE_COLS].values[: int(len(train) * 0.8)]
    y_tr = train["label"].values[: int(len(train) * 0.8)]

    print("Training random_forest model...")
    models = train_models(X_tr, y_tr)
    rf = models["random_forest"]
    all_probs = model_proba(rf, dset[FEATURE_COLS].values)

    cfg = ELITE_CFG
    sigs = build_filtered_signals(df, dset, all_probs, cfg)
    stats, trades = backtest_signals(df, sigs, cfg.name, "full", record_trades=True)
    losses = sum(1 for t in trades if t["result"] == "LOSS")

    test_df = df[df["session_date"] >= min(test_dates)]
    test_dset = dset[dset["session_date"].isin(set(test_dates))].copy()
    oos_sigs = build_filtered_signals(test_df, test_dset, model_proba(rf, test_dset[FEATURE_COLS].values), cfg)
    oos_stats, oos_trades = backtest_signals(test_df, oos_sigs, f"{cfg.name} OOS", "OOS", record_trades=True)
    oos_losses = sum(1 for t in oos_trades if t["result"] == "LOSS")

    print(f"Full: {stats.trades} trades, {losses} losses, WR {stats.win_rate}%")
    print(f"      Rs {stats.avg_daily:,.0f}/day | Net Rs {stats.net_pnl:,.0f} | PF {stats.profit_factor}")
    print(f"OOS:  {oos_stats.trades} trades, {oos_losses} losses, Rs {oos_stats.avg_daily:,.0f}/day")

    filters = {
        "ml_threshold": cfg.ml_th,
        "adx_min": cfg.adx_min,
        "range_pos_min": cfg.range_pos_min,
        "max_risk_pts": cfg.max_risk_pts,
        "require_ema_stack": cfg.require_ema_stack,
        "hour_start": cfg.hour_start,
        "hour_end": cfg.hour_end,
    }
    payload = {
        "selected": cfg.name,
        "filters": filters,
        "full_period": {
            "summary": stats.__dict__,
            "losses": losses,
            "wins": stats.trades - losses,
            "trades": trades,
        },
        "oos": {
            "summary": oos_stats.__dict__,
            "losses": oos_losses,
            "trades": oos_trades,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    TRADES_OUT.write_text(json.dumps({
        "strategy": cfg.name,
        "filters": filters,
        "full_period": payload["full_period"],
        "oos": payload["oos"],
    }, indent=2, default=str), encoding="utf-8")

    if MODEL_PKL.exists():
        with MODEL_PKL.open("rb") as fh:
            bundle = pickle.load(fh)
        bundle["strategy_filters"] = filters
        bundle["strategy_name"] = cfg.name
        with MODEL_PKL.open("wb") as fh:
            pickle.dump(bundle, fh)

    print(f"\nSaved {OUT}")
    print(f"Saved {TRADES_OUT}")


if __name__ == "__main__":
    main()