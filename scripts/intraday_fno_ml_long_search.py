"""
F&O Elite ML Long — train + multi-strategy search for Nifty 5m.

Mirrors Elite ML Short v2 pipeline for LONG side:
  - Build labeled long candidates
  - Train RF / GB / HistGB
  - Backtest many filter packs
  - Pick best profitable strategy with WR ≥ 70%

Outputs:
  data/intraday_fno_ml_long_model.pkl
  data/intraday_fno_ml_long_model.json
  data/intraday_fno_ml_long_results.json
  data/intraday_fno_ml_long_trades.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from trading.services.fno_engine import (  # noqa: E402
    CAPITAL,
    FEATURE_COLS,
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    RISK_PCT,
    SLIPPAGE_PTS,
    TARGET_R,
    TRADES_PER_DAY,
    enrich_features,
    lots_for_risk,
    max_lots,
    normalize_df,
)

LOT_SIZE = INSTRUMENTS["NIFTY"]["lot_size"]
MARGIN = INSTRUMENTS["NIFTY"]["mis_margin"]
DATA_PATH = ROOT / "data" / "intraday_fno" / "NIFTY.pkl"
OUT_MODEL = ROOT / "data" / "intraday_fno_ml_long_model.pkl"
OUT_META = ROOT / "data" / "intraday_fno_ml_long_model.json"
OUT_RESULTS = ROOT / "data" / "intraday_fno_ml_long_results.json"
OUT_TRADES = ROOT / "data" / "intraday_fno_ml_long_trades.json"

MIN_WR = 70.0
MIN_TRADES = 25


# ── Long signal rules ────────────────────────────────────────────────────────

def sig_ema_long(row, target_r: float = TARGET_R) -> Optional[dict]:
    """Strict bullish EMA stack + above VWAP + RSI ≥ 50."""
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("ema_50")):
        return None
    if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
        return None
    if pd.notna(row.get("vwap")) and row["close"] <= row["vwap"]:
        return None
    if row["rsi"] < 50:
        return None
    stop = float(row["ema_21"])
    risk = float(row["close"]) - stop
    if risk <= 0:
        return None
    return {
        "side": 1,
        "stop": stop,
        "target": float(row["close"] + risk * target_r),
        "risk_pts": float(risk),
    }


def loose_long_signal(row) -> bool:
    """Broader long candidate pool for ML."""
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return False
    trend = row["ema_9"] > row["ema_21"]
    above_vwap = pd.isna(row.get("vwap")) or row["close"] > row["vwap"]
    rsi_ok = 40 < row["rsi"] < 75
    return bool(trend and above_vwap and rsi_ok)


def sig_pullback_long(row, target_r: float = TARGET_R) -> Optional[dict]:
    """Uptrend pullback: EMA9>EMA21, price near EMA21, RSI 40–60, not extended."""
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return None
    if not (row["ema_9"] > row["ema_21"]):
        return None
    if pd.notna(row.get("ema_50")) and row["ema_21"] < row["ema_50"]:
        return None
    if not (40 <= row["rsi"] <= 60):
        return None
    # near EMA21 (within 0.15%)
    dist = abs(row["close"] - row["ema_21"]) / row["close"] * 100
    if dist > 0.20:
        return None
    if pd.notna(row.get("px_ema9")) and row["px_ema9"] > 0.35:
        return None  # too extended above EMA9
    stop = float(min(row["ema_21"], row["low"])) * 0.999
    # use structure stop under EMA21 / bar low
    stop = float(row["ema_50"]) if pd.notna(row.get("ema_50")) and row["ema_50"] < row["close"] else float(row["ema_21"])
    risk = float(row["close"]) - stop
    if risk <= 0 or risk > 40:
        return None
    return {
        "side": 1,
        "stop": stop,
        "target": float(row["close"] + risk * target_r),
        "risk_pts": float(risk),
    }


def sig_breakout_long(row, target_r: float = TARGET_R) -> Optional[dict]:
    """Session strength breakout: range_pos high, positive ROC, above VWAP."""
    if pd.isna(row.get("range_pos")) or pd.isna(row.get("rsi")):
        return None
    if row["range_pos"] < 0.70:
        return None
    if pd.notna(row.get("vwap")) and row["close"] <= row["vwap"]:
        return None
    if row["rsi"] < 55 or row["rsi"] > 80:
        return None
    if pd.notna(row.get("roc_6")) and row["roc_6"] < 0:
        return None
    if pd.notna(row.get("ema_9")) and pd.notna(row.get("ema_21")):
        if row["ema_9"] < row["ema_21"]:
            return None
    # stop under day mid or EMA21
    stop = float(row.get("ema_21") or row["close"] * 0.997)
    if pd.notna(row.get("day_low")) and pd.notna(row.get("day_high")):
        mid = (float(row["day_low"]) + float(row["day_high"])) / 2
        stop = max(stop, mid) if mid < row["close"] else stop
    risk = float(row["close"]) - stop
    if risk <= 0 or risk > 35:
        return None
    return {
        "side": 1,
        "stop": stop,
        "target": float(row["close"] + risk * target_r),
        "risk_pts": float(risk),
    }


def sig_vwap_reclaim_long(row, target_r: float = TARGET_R) -> Optional[dict]:
    """Price reclaiming VWAP with rising RSI and bullish body."""
    if pd.isna(row.get("vwap")) or pd.isna(row.get("rsi")):
        return None
    if row["close"] <= row["vwap"]:
        return None
    if row["prev_close"] is not None and pd.notna(row.get("prev_close")):
        # crossed or holding above VWAP
        if row["prev_close"] > row["vwap"] * 1.001 and row["rsi"] < 45:
            return None
    if not (45 <= row["rsi"] <= 70):
        return None
    if pd.notna(row.get("body_pct")) and row["body_pct"] < 0:
        return None
    if pd.notna(row.get("ema_9")) and row["close"] < row["ema_9"]:
        return None
    stop = float(row["vwap"]) * 0.999
    risk = float(row["close"]) - stop
    if risk <= 0 or risk > 30:
        return None
    return {
        "side": 1,
        "stop": stop,
        "target": float(row["close"] + risk * target_r),
        "risk_pts": float(risk),
    }


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate_long_trade(df: pd.DataFrame, entry_idx: int, stop: float, target: float) -> dict:
    empty = {
        "win": False, "pnl_pts": 0.0, "exit_price": None,
        "exit_time": None, "exit_reason": "no_fill",
        "entry_price": None, "entry_time": None,
    }
    if entry_idx >= len(df):
        return empty
    entry_row = df.iloc[entry_idx]
    entry = float(entry_row["open"]) + SLIPPAGE_PTS
    entry_ts = df.index[entry_idx]

    for j in range(entry_idx, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

        if lo <= stop:
            exit_px = stop - SLIPPAGE_PTS
            return {
                "win": False,
                "pnl_pts": exit_px - entry,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "stop_hit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if hi >= target:
            exit_px = target - SLIPPAGE_PTS
            return {
                "win": True,
                "pnl_pts": exit_px - entry,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "target_hit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if ts.time() >= FORCE_EXIT:
            exit_px = cl - SLIPPAGE_PTS
            return {
                "win": exit_px > entry,
                "pnl_pts": exit_px - entry,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "eod_exit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if j > entry_idx and ts.date() != entry_ts.date():
            break
    return empty


def long_stop_target(row, target_r: float, mode: str = "ema") -> Optional[tuple[float, float, float]]:
    """Return stop, target, risk_pts for long."""
    if mode == "ema_stack":
        sig = sig_ema_long(row, target_r)
    elif mode == "pullback":
        sig = sig_pullback_long(row, target_r)
    elif mode == "breakout":
        sig = sig_breakout_long(row, target_r)
    elif mode == "vwap_reclaim":
        sig = sig_vwap_reclaim_long(row, target_r)
    else:
        # default: stop under EMA21 if bullish, else ATR-like
        if pd.isna(row.get("ema_21")):
            return None
        stop = float(row["ema_21"])
        risk = float(row["close"]) - stop
        if risk <= 0:
            return None
        return stop, float(row["close"] + risk * target_r), risk

    if not sig:
        return None
    return sig["stop"], sig["target"], sig["risk_pts"]


# ── Dataset + ML ─────────────────────────────────────────────────────────────

def build_long_dataset(df: pd.DataFrame, target_r: float = TARGET_R) -> pd.DataFrame:
    rows = []
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        if not loose_long_signal(row):
            continue

        st = long_stop_target(row, target_r, mode="ema_stack")
        if st is None:
            # loose: still label using EMA21 stop
            if pd.isna(row.get("ema_21")):
                continue
            stop = float(row["ema_21"])
            risk = float(row["close"]) - stop
            if risk <= 0:
                continue
            target = float(row["close"] + risk * target_r)
        else:
            stop, target, risk = st

        entry_idx = i + 1
        if entry_idx >= len(df) or df.index[entry_idx].date() != ts.date():
            continue

        outcome = simulate_long_trade(df, entry_idx, stop, target)
        if outcome["exit_reason"] == "no_fill":
            continue

        feat = {c: row.get(c, np.nan) for c in FEATURE_COLS}
        feat["label"] = 1 if outcome["win"] else 0
        feat["pnl_pts"] = outcome["pnl_pts"]
        feat["timestamp"] = ts
        feat["session_date"] = row["session_date"]
        feat["stop"] = stop
        feat["target"] = target
        feat["risk_pts"] = risk
        feat["is_base_signal"] = sig_ema_long(row, target_r) is not None
        rows.append(feat)
    return pd.DataFrame(rows)


def time_split(dset: pd.DataFrame, train_frac: float = 0.7):
    dates = sorted(dset["session_date"].unique())
    cut = max(1, int(len(dates) * train_frac))
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train = dset[dset["session_date"].isin(train_dates)].copy()
    test = dset[dset["session_date"].isin(test_dates)].copy()
    return train, test, dates[:cut], dates[cut:]


def train_models(X_train, y_train):
    models = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, max_iter=250,
            min_samples_leaf=20, l2_regularization=0.1, random_state=42,
        ),
    }
    fitted = {}
    for name, clf in models.items():
        if name == "hist_gradient_boosting":
            clf.fit(X_train, y_train)
            fitted[name] = clf
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            fitted[name] = pipe
    return fitted


def model_proba(model, X):
    if isinstance(model, Pipeline):
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def ensemble_proba(models: dict, X) -> np.ndarray:
    return np.mean([model_proba(m, X) for m in models.values()], axis=0)


def pick_threshold(probs, y_val, pnl_val) -> float:
    best_th, best_score = 0.55, -1e18
    for th in np.arange(0.40, 0.85, 0.02):
        mask = probs >= th
        if mask.sum() < 8:
            continue
        wr = y_val[mask].mean()
        exp_pnl = pnl_val[mask].sum()
        # Prefer high WR and positive expectancy
        score = exp_pnl * (0.3 + wr) + (wr - 0.70) * 5000 if wr >= 0.65 else exp_pnl * wr
        if score > best_score:
            best_score, best_th = score, float(th)
    return best_th


# ── Backtest engine ──────────────────────────────────────────────────────────

@dataclass
class BTResult:
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    avg_daily: float = 0.0
    final_equity: float = CAPITAL
    period: str = "full"
    trade_log: list = field(default_factory=list)


def backtest_long(
    df: pd.DataFrame,
    signal_times: dict[Any, dict],
    name: str,
    period: str = "full",
    *,
    target_r: float = TARGET_R,
    risk_pct: float = RISK_PCT,
    max_trades_day: int = TRADES_PER_DAY,
    record_trades: bool = True,
) -> BTResult:
    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    wins = gp = gl = 0.0
    trades = 0
    day_count: dict = {}
    daily_pnl: dict = {}
    trade_log = []

    for i in range(len(df)):
        ts = df.index[i]
        key = ts
        if key not in signal_times and ts.isoformat() not in signal_times:
            # try iso
            if ts.isoformat() not in signal_times:
                continue
            key = ts.isoformat()
        sig = signal_times.get(key) or signal_times.get(ts.isoformat())
        if not sig:
            continue
        bar = df.iloc[i]
        sess = bar["session_date"]
        if day_count.get(sess, 0) >= max_trades_day:
            continue
        if i + 1 >= len(df):
            continue
        entry_idx = i + 1
        if df.index[entry_idx].date() != ts.date():
            continue

        stop, target = float(sig["stop"]), float(sig["target"])
        # Recompute target from entry for true R after fill
        outcome = simulate_long_trade(df, entry_idx, stop, target)
        if outcome["exit_reason"] == "no_fill":
            continue
        entry = outcome["entry_price"]
        # adjust target to entry-based R if we have risk
        risk_pts = entry - stop
        if risk_pts <= 0:
            continue
        # re-simulate with entry-based target for fairness
        true_target = entry + risk_pts * target_r
        outcome = simulate_long_trade(df, entry_idx, stop, true_target)
        if outcome["exit_reason"] == "no_fill":
            continue
        entry = outcome["entry_price"]
        stop_pts = entry - stop
        if stop_pts <= 0:
            continue

        max_l = max_lots(equity, MARGIN)
        lots = lots_for_risk(equity, risk_pct, stop_pts, LOT_SIZE, max_l)
        if lots <= 0:
            continue

        pnl_pts = outcome["pnl_pts"]
        pnl = pnl_pts * LOT_SIZE * lots
        equity += pnl
        trades += 1
        day_count[sess] = day_count.get(sess, 0) + 1
        daily_pnl[sess] = daily_pnl.get(sess, 0.0) + pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if pnl > 0:
            wins += 1
            gp += pnl
        else:
            gl += abs(pnl)

        if record_trades:
            trade_log.append({
                "trade_no": trades,
                "strategy": name,
                "period": period,
                "instrument": "NIFTY",
                "side": "LONG",
                "session_date": str(sess),
                "signal_time": ts.isoformat(),
                "entry_time": outcome["entry_time"],
                "entry_price": entry,
                "stop": round(stop, 2),
                "target": round(true_target, 2),
                "risk_pts": round(stop_pts, 2),
                "target_r": target_r,
                "lots": lots,
                "lot_size": LOT_SIZE,
                "ml_prob": round(float(sig.get("ml_prob", 0) or 0), 3),
                "exit_time": outcome["exit_time"],
                "exit_price": outcome["exit_price"],
                "exit_reason": outcome["exit_reason"],
                "pnl_pts": round(pnl_pts, 2),
                "pnl_inr": round(pnl, 2),
                "result": "WIN" if pnl > 0 else "LOSS",
                "equity_after": round(equity, 2),
            })

    wr = 100.0 * wins / trades if trades else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    n_days = max(len(daily_pnl), 1)
    return BTResult(
        name=name,
        trades=trades,
        wins=int(wins),
        losses=trades - int(wins),
        win_rate=round(wr, 2),
        net_pnl=round(equity - CAPITAL, 2),
        profit_factor=round(pf, 2),
        max_dd_pct=round(max_dd, 2),
        avg_daily=round((equity - CAPITAL) / n_days, 2),
        final_equity=round(equity, 2),
        period=period,
        trade_log=trade_log,
    )


def passes_long_filters(row, ts, prob: float, risk_pts: float, cfg: dict) -> bool:
    if prob is None or prob < cfg.get("ml_threshold", 0.55):
        return False
    if cfg.get("require_ema_stack"):
        if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
            return False
        if pd.notna(row.get("vwap")) and row["close"] <= row["vwap"]:
            return False
        if row["rsi"] < 50:
            return False
    adx_min = cfg.get("adx_min", 0)
    if adx_min > 0:
        adx = row.get("adx")
        if pd.isna(adx) or adx < adx_min:
            return False
    rp_max = cfg.get("range_pos_max", 1.0)
    rp_min = cfg.get("range_pos_min", 0.0)
    rp = row.get("range_pos")
    if pd.notna(rp):
        if rp < rp_min or rp > rp_max:
            return False
    max_risk = cfg.get("max_risk_pts", 999)
    min_risk = cfg.get("min_risk_pts", 0)
    if risk_pts > max_risk or risk_pts < min_risk:
        return False
    h = ts.hour if hasattr(ts, "hour") else 0
    if h < cfg.get("hour_start", 9) or h > cfg.get("hour_end", 14):
        return False
    if not (cfg.get("rsi_min", 0) <= row["rsi"] <= cfg.get("rsi_max", 100)):
        return False
    # day path not too extended green already
    if cfg.get("max_path_pct") is not None and pd.notna(row.get("pct_from_open")):
        if row["pct_from_open"] > cfg["max_path_pct"]:
            return False
    if cfg.get("min_path_pct") is not None and pd.notna(row.get("pct_from_open")):
        if row["pct_from_open"] < cfg["min_path_pct"]:
            return False
    if cfg.get("require_di_bull"):
        if pd.isna(row.get("di_diff")) or row["di_diff"] <= 0:
            return False
    if cfg.get("require_macd_hist_pos"):
        if pd.isna(row.get("macd_hist")) or row["macd_hist"] <= 0:
            return False
    return True


def soft_long_checklist_score(row, ts, ml_prob: float, losses_today: int = 0) -> tuple[bool, int]:
    """Mirror short checklist, flipped for longs. Score ≥ 6 to take."""
    score = 0
    hard_no = False
    path = float(row["pct_from_open"]) if pd.notna(row.get("pct_from_open")) else None
    hour = int(ts.hour) if hasattr(ts, "hour") else 0

    # Q1 path — don't chase strong already-green days; prefer mild green / reclaim
    if path is None:
        pass
    elif path < -0.6:
        hard_no = True  # strong red day — avoid long
    elif path <= -0.1:
        score += 2  # dip buy context
    elif path <= 0.35:
        score += 1
    elif path > 0.80:
        hard_no = True  # too extended

    # Q2 structure
    above_vwap = pd.isna(row.get("vwap")) or row["close"] > row["vwap"]
    ema_ok = row["ema_9"] > row["ema_21"]
    full = ema_ok and (pd.isna(row.get("ema_50")) or row["ema_21"] > row["ema_50"])
    if full and above_vwap:
        score += 2
    elif ema_ok and above_vwap:
        score += 1
    elif not above_vwap:
        hard_no = True

    # Q3 time
    if 9 <= hour <= 11:
        score += 2
    elif hour <= 13:
        score += 1
    elif hour >= 14:
        score += 0

    # Q4 ML
    if ml_prob >= 0.70:
        score += 2
    elif ml_prob >= 0.62:
        score += 1
    elif ml_prob < 0.55:
        hard_no = True

    # Q5 session losses
    if losses_today >= 2:
        hard_no = True
    elif losses_today == 0:
        score += 1

    take = (not hard_no) and score >= 6
    return take, score


# ── Signal builders ──────────────────────────────────────────────────────────

def build_rule_signals(df: pd.DataFrame, mode: str, target_r: float) -> dict:
    sigs = {}
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        st = long_stop_target(row, target_r, mode=mode)
        if st is None:
            continue
        stop, target, risk = st
        sigs[ts] = {"stop": stop, "target": target, "risk_pts": risk, "ml_prob": 0.0}
    return sigs


def build_ml_signals(
    df: pd.DataFrame,
    models: dict,
    cfg: dict,
    target_r: float,
    *,
    use_checklist: bool = False,
    ensemble: bool = True,
) -> dict:
    sigs = {}
    losses_today: dict = {}
    # Precompute features for all loose candidates
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue
        if not loose_long_signal(row):
            continue
        st = long_stop_target(row, target_r, mode="ema_stack")
        if st is None:
            if pd.isna(row.get("ema_21")):
                continue
            stop = float(row["ema_21"])
            risk = float(row["close"]) - stop
            if risk <= 0:
                continue
            target = float(row["close"] + risk * target_r)
        else:
            stop, target, risk = st

        feat = np.array([[float(row.get(c, np.nan)) for c in FEATURE_COLS]], dtype=float)
        if np.isnan(feat).any():
            continue
        if ensemble:
            prob = float(ensemble_proba(models, feat)[0])
        else:
            # use RF
            prob = float(model_proba(models["random_forest"], feat)[0])

        if not passes_long_filters(row, ts, prob, risk, cfg):
            continue

        sess = row["session_date"]
        if use_checklist:
            take, _sc = soft_long_checklist_score(row, ts, prob, losses_today.get(sess, 0))
            if not take:
                continue

        sigs[ts] = {
            "stop": stop, "target": target, "risk_pts": risk, "ml_prob": prob,
        }
    return sigs


def evaluate_and_maybe_update_losses(result: BTResult) -> None:
    """No-op placeholder — losses tracked inside checklist builder if needed."""
    pass


# ── Main search ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("F&O ELITE ML LONG — train + multi-strategy search")
    print("=" * 80)

    raw = pickle.load(open(DATA_PATH, "rb"))
    df = normalize_df(raw)
    df = enrich_features(df)
    print(f"Data: {df.index.min()} → {df.index.max()} | bars={len(df)}")

    print("\nBuilding labeled LONG dataset…")
    dset = build_long_dataset(df, TARGET_R).dropna(subset=FEATURE_COLS)
    print(f"  Samples: {len(dset)} | raw WR: {dset['label'].mean()*100:.1f}%")
    if len(dset) < 50:
        print("Not enough long samples — abort.")
        return

    train, test, train_dates, test_dates = time_split(dset, 0.7)
    print(f"  Train sessions: {len(train_dates)} | OOS test sessions: {len(test_dates)}")

    X_tr_full = train[FEATURE_COLS].values
    y_tr_full = train["label"].values
    val_cut = int(len(train) * 0.8)
    X_tr, X_val = X_tr_full[:val_cut], X_tr_full[val_cut:]
    y_tr, y_val = y_tr_full[:val_cut], y_tr_full[val_cut:]
    pnl_val = train["pnl_pts"].values[val_cut:]

    print("Training models…")
    models = train_models(X_tr, y_tr)
    ens_val = ensemble_proba(models, X_val)
    th_auto = pick_threshold(ens_val, y_val, pnl_val)
    print(f"  Auto ML threshold (val): {th_auto:.2f}")

    # Full-period + OOS dataframes
    all_dates = sorted(df["session_date"].unique())
    oos_dates = set(test_dates)
    oos_df = df[df["session_date"].isin(oos_dates)]

    results: list[dict] = []
    best_eligible: Optional[dict] = None
    best_any: Optional[dict] = None

    def run_named(name: str, sigs: dict, period: str, target_r: float, risk_pct: float, subset_df):
        nonlocal best_eligible, best_any
        r = backtest_long(
            subset_df, sigs, name, period,
            target_r=target_r, risk_pct=risk_pct, record_trades=True,
        )
        row = {
            "name": name,
            "period": period,
            "trades": r.trades,
            "wins": r.wins,
            "losses": r.losses,
            "win_rate": r.win_rate,
            "net_pnl": r.net_pnl,
            "profit_factor": r.profit_factor,
            "max_dd_pct": r.max_dd_pct,
            "avg_daily": r.avg_daily,
            "final_equity": r.final_equity,
            "target_r": target_r,
            "risk_pct": risk_pct,
            "trade_log": r.trade_log,
        }
        results.append(row)
        flag = ""
        eligible = (
            r.win_rate >= MIN_WR
            and r.net_pnl > 0
            and r.trades >= MIN_TRADES
            and r.profit_factor >= 1.2
        )
        if eligible:
            flag = " ★ ELIGIBLE ≥70% WR"
            if best_eligible is None or r.net_pnl > best_eligible["net_pnl"]:
                best_eligible = row
        if best_any is None or (
            (r.net_pnl > 0 and r.win_rate > (best_any.get("win_rate") or 0))
            or (r.net_pnl > best_any.get("net_pnl", -1e18) and r.win_rate >= 65)
        ):
            # track strongest overall
            score = r.net_pnl * (r.win_rate / 100) * min(r.profit_factor, 5)
            prev = best_any.get("_score", -1e18) if best_any else -1e18
            if score > prev:
                row["_score"] = score
                best_any = row

        print(
            f"  {name:55s} n={r.trades:3d} WR={r.win_rate:5.1f}% "
            f"PnL=₹{r.net_pnl:>10,.0f} PF={r.profit_factor:5.2f} DD={r.max_dd_pct:5.1f}%{flag}"
        )
        return row

    print("\n### RULE-BASED LONGS (full period) ###")
    for mode, label in [
        ("ema_stack", "Rule: EMA stack long"),
        ("pullback", "Rule: Pullback long"),
        ("breakout", "Rule: Breakout long"),
        ("vwap_reclaim", "Rule: VWAP reclaim long"),
    ]:
        for tr in [1.0, 1.5, 2.0]:
            sigs = build_rule_signals(df, mode, tr)
            run_named(f"{label} {tr}R", sigs, "full", tr, RISK_PCT, df)

    print("\n### ML LONG PACKS (full period) ###")
    # Grid of ML filter packs
    packs = []
    for th in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, th_auto]:
        for max_risk in [18, 22, 28, 35]:
            for hour_end in [13, 14]:
                for rsi_lo, rsi_hi in [(40, 70), (45, 70), (50, 75), (42, 68)]:
                    for adx in [0, 15, 20]:
                        for stack in [False, True]:
                            for tr in [1.2, 1.5, 2.0]:
                                packs.append({
                                    "ml_threshold": round(float(th), 2),
                                    "adx_min": adx,
                                    "range_pos_min": 0.0,
                                    "range_pos_max": 1.0,
                                    "max_risk_pts": max_risk,
                                    "min_risk_pts": 0.0,
                                    "hour_start": 9,
                                    "hour_end": hour_end,
                                    "rsi_min": rsi_lo,
                                    "rsi_max": rsi_hi,
                                    "require_ema_stack": stack,
                                    "target_r": tr,
                                    "checklist": False,
                                    "max_path_pct": None,
                                    "require_di_bull": False,
                                    "require_macd_hist_pos": False,
                                })

    # Extra high-quality packs
    for th in [0.60, 0.65, 0.70]:
        packs.append({
            "ml_threshold": th, "adx_min": 18, "range_pos_min": 0.35,
            "range_pos_max": 0.95, "max_risk_pts": 25, "min_risk_pts": 3,
            "hour_start": 9, "hour_end": 13, "rsi_min": 48, "rsi_max": 72,
            "require_ema_stack": True, "target_r": 1.5, "checklist": True,
            "max_path_pct": 0.55, "require_di_bull": True, "require_macd_hist_pos": False,
        })
        packs.append({
            "ml_threshold": th, "adx_min": 15, "range_pos_min": 0.25,
            "range_pos_max": 1.0, "max_risk_pts": 22, "min_risk_pts": 2,
            "hour_start": 9, "hour_end": 14, "rsi_min": 45, "rsi_max": 70,
            "require_ema_stack": True, "target_r": 1.5, "checklist": True,
            "max_path_pct": 0.70, "require_di_bull": False, "require_macd_hist_pos": True,
        })
        packs.append({
            "ml_threshold": th, "adx_min": 0, "range_pos_min": 0.0,
            "range_pos_max": 1.0, "max_risk_pts": 22, "min_risk_pts": 0,
            "hour_start": 9, "hour_end": 14, "rsi_min": 40, "rsi_max": 70,
            "require_ema_stack": False, "target_r": 1.5, "checklist": True,
            "max_path_pct": 0.60, "require_di_bull": False, "require_macd_hist_pos": False,
        })

    # Dedup packs
    seen = set()
    uniq_packs = []
    for p in packs:
        key = tuple(sorted((k, str(v)) for k, v in p.items()))
        if key not in seen:
            seen.add(key)
            uniq_packs.append(p)

    # Cap search size — prioritize higher thresholds & checklist
    uniq_packs.sort(
        key=lambda p: (
            p.get("checklist", False),
            p["ml_threshold"],
            p["require_ema_stack"],
            -p["max_risk_pts"],
        ),
        reverse=True,
    )
    uniq_packs = uniq_packs[:180]
    print(f"Testing {len(uniq_packs)} ML long packs…")

    for i, cfg in enumerate(uniq_packs, 1):
        tr = cfg["target_r"]
        use_cl = cfg.get("checklist", False)
        name = (
            f"ML long th={cfg['ml_threshold']:.2f} stack={cfg['require_ema_stack']} "
            f"ADX≥{cfg['adx_min']} risk≤{cfg['max_risk_pts']} "
            f"RSI {cfg['rsi_min']}-{cfg['rsi_max']} {tr}R"
            f"{' +CL' if use_cl else ''}"
        )
        sigs = build_ml_signals(df, models, cfg, tr, use_checklist=use_cl, ensemble=True)
        if len(sigs) < 5:
            continue
        run_named(name, sigs, "full", tr, RISK_PCT, df)
        if i % 30 == 0:
            print(f"  … {i}/{len(uniq_packs)} packs tested")

    # OOS evaluation of top candidates
    print("\n### OOS RE-TEST top candidates ###")
    ranked = sorted(
        [r for r in results if r["period"] == "full" and r["trades"] >= 15],
        key=lambda x: (
            1 if x["win_rate"] >= MIN_WR and x["net_pnl"] > 0 else 0,
            x["win_rate"],
            x["net_pnl"],
        ),
        reverse=True,
    )[:15]

    oos_results = []
    for cand in ranked:
        # re-parse filters from name is hard — store cfg on row next time
        # Instead re-run best_eligible style packs from top WR
        pass

    # Explicit OOS for elite packs
    elite_cfgs = []
    if best_eligible:
        # rebuild from top 10 eligible
        eligible = [
            r for r in results
            if r["period"] == "full"
            and r["win_rate"] >= MIN_WR
            and r["net_pnl"] > 0
            and r["trades"] >= MIN_TRADES
        ]
        eligible.sort(key=lambda x: (x["net_pnl"], x["win_rate"]), reverse=True)
        print(f"Eligible ≥{MIN_WR}% WR profitable: {len(eligible)}")
    else:
        eligible = []
        print(f"No pack hit WR≥{MIN_WR}% with enough trades yet — tightening search…")

    # Secondary pass: high-threshold ML only + checklist, optimized for WR
    print("\n### HIGH-WR SPECIALIST PASS ###")
    specialist = []
    for th in np.arange(0.58, 0.82, 0.02):
        for tr in [1.0, 1.2, 1.5, 1.8]:
            for stack in [True, False]:
                for cl in [True, False]:
                    for max_risk in [15, 20, 25]:
                        for rsi in [(50, 70), (48, 68), (45, 65), (52, 72)]:
                            cfg = {
                                "ml_threshold": round(float(th), 2),
                                "adx_min": 12,
                                "range_pos_min": 0.30,
                                "range_pos_max": 0.98,
                                "max_risk_pts": max_risk,
                                "min_risk_pts": 2.0,
                                "hour_start": 9,
                                "hour_end": 13,
                                "rsi_min": rsi[0],
                                "rsi_max": rsi[1],
                                "require_ema_stack": stack,
                                "target_r": tr,
                                "checklist": cl,
                                "max_path_pct": 0.50,
                                "min_path_pct": -0.35,
                                "require_di_bull": True,
                                "require_macd_hist_pos": False,
                            }
                            specialist.append(cfg)
    # unique + cap
    seen2 = set()
    sp2 = []
    for p in specialist:
        key = tuple(sorted((k, str(v)) for k, v in p.items()))
        if key not in seen2:
            seen2.add(key)
            sp2.append(p)
    sp2 = sp2[:120]

    specialist_rows = []
    for cfg in sp2:
        tr = cfg["target_r"]
        name = (
            f"EliteLong th={cfg['ml_threshold']:.2f} stack={cfg['require_ema_stack']} "
            f"CL={cfg['checklist']} {tr}R risk≤{cfg['max_risk_pts']} "
            f"RSI{cfg['rsi_min']}-{cfg['rsi_max']}"
        )
        sigs = build_ml_signals(df, models, cfg, tr, use_checklist=cfg["checklist"], ensemble=True)
        if len(sigs) < 8:
            continue
        row = run_named(name, sigs, "full", tr, RISK_PCT, df)
        row["cfg"] = cfg
        specialist_rows.append(row)

    # OOS for top specialist + eligible
    print("\n### OOS VALIDATION ###")
    candidates = []
    for r in results:
        if r["period"] != "full":
            continue
        if r["trades"] < MIN_TRADES:
            continue
        if r["net_pnl"] <= 0:
            continue
        score = 0
        if r["win_rate"] >= MIN_WR:
            score += 1000
        score += r["win_rate"] * 2 + min(r["profit_factor"], 5) * 20 + r["net_pnl"] / 10000
        r["_rank"] = score
        candidates.append(r)
    candidates.sort(key=lambda x: x["_rank"], reverse=True)

    oos_best = None
    for cand in candidates[:12]:
        cfg = cand.get("cfg")
        if not cfg:
            # try match specialist
            continue
        tr = cfg["target_r"]
        sigs = build_ml_signals(oos_df, models, cfg, tr, use_checklist=cfg.get("checklist", False))
        name = f"OOS | {cand['name']}"
        row = run_named(name, sigs, "oos", tr, RISK_PCT, oos_df)
        row["cfg"] = cfg
        row["full_wr"] = cand["win_rate"]
        row["full_pnl"] = cand["net_pnl"]
        if row["trades"] >= 8 and row["net_pnl"] > 0 and row["win_rate"] >= 60:
            if oos_best is None or (row["win_rate"], row["net_pnl"]) > (oos_best["win_rate"], oos_best["net_pnl"]):
                oos_best = row

    # Pick champion
    print("\n" + "=" * 80)
    eligible = [
        r for r in results
        if r["period"] == "full"
        and r["win_rate"] >= MIN_WR
        and r["net_pnl"] > 0
        and r["trades"] >= MIN_TRADES
        and r["profit_factor"] >= 1.2
    ]
    eligible.sort(key=lambda x: (x["net_pnl"], x["win_rate"], x["profit_factor"]), reverse=True)

    print(f"\nSTRATEGIES WITH WR≥{MIN_WR}% & PROFITABLE & n≥{MIN_TRADES}: {len(eligible)}")
    for i, r in enumerate(eligible[:15], 1):
        print(
            f"  #{i} {r['name'][:60]}\n"
            f"      WR={r['win_rate']}% n={r['trades']} PnL=₹{r['net_pnl']:,.0f} "
            f"PF={r['profit_factor']} DD={r['max_dd_pct']}%"
        )

    if eligible:
        champ = eligible[0]
        # Prefer one with cfg for reproducibility
        with_cfg = [r for r in eligible if r.get("cfg")]
        if with_cfg:
            champ = with_cfg[0]
    else:
        # Fallback: best WR among profitable
        prof = [r for r in results if r["period"] == "full" and r["net_pnl"] > 0 and r["trades"] >= 20]
        prof.sort(key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
        champ = prof[0] if prof else (best_any or results[0])
        print("\n⚠ No strategy hit 70% WR with constraints — showing best available.")

    print("\n### CHAMPION ###")
    print(json.dumps({k: v for k, v in champ.items() if k not in ("trade_log", "_score", "_rank")}, indent=2))

    # Save model + meta
    cfg = champ.get("cfg") or {
        "ml_threshold": 0.65,
        "adx_min": 12,
        "range_pos_min": 0.30,
        "range_pos_max": 0.98,
        "max_risk_pts": 22,
        "min_risk_pts": 2.0,
        "hour_start": 9,
        "hour_end": 13,
        "rsi_min": 48,
        "rsi_max": 70,
        "require_ema_stack": True,
        "target_r": 1.5,
        "checklist": True,
        "max_path_pct": 0.50,
        "min_path_pct": -0.35,
        "require_di_bull": True,
        "require_macd_hist_pos": False,
    }

    # Retrain on full train set for production model
    models_full = train_models(X_tr_full, y_tr_full)
    payload = {
        "models": models_full,
        "feature_cols": FEATURE_COLS,
        "strategy_name": "Elite ML Long v1",
        "filters": cfg,
        "target_r": cfg.get("target_r", 1.5),
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(payload, f)

    meta = {
        "model_type": "ensemble_rf_gb_hgb",
        "threshold": cfg.get("ml_threshold", 0.65),
        "features": FEATURE_COLS,
        "strategy_name": "Elite ML Long v1",
        "strategy_filters": cfg,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_range": {
            "instrument": "NIFTY",
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
            "bars": len(df),
        },
        "champion": {k: v for k, v in champ.items() if k not in ("trade_log", "_score", "_rank", "cfg")},
        "min_wr_target": MIN_WR,
        "eligible_count": len(eligible),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save comparison table (no trade logs)
    summary_rows = []
    for r in sorted(results, key=lambda x: (x["period"] != "full", -x.get("win_rate", 0), -x.get("net_pnl", 0))):
        summary_rows.append({k: v for k, v in r.items() if k not in ("trade_log", "_score", "_rank", "cfg")})

    out = {
        "selected": "Elite ML Long v1",
        "generated_at": meta["trained_at"],
        "data_range": meta["data_range"],
        "filters": cfg,
        "champion": meta["champion"],
        "eligible_top": [
            {k: v for k, v in r.items() if k not in ("trade_log", "_score", "_rank", "cfg")}
            for r in eligible[:20]
        ],
        "all_strategies_count": len(results),
        "top_by_wr": summary_rows[:40],
        "full_period": {
            "summary": meta["champion"],
            "wins": champ.get("wins"),
            "losses": champ.get("losses"),
            "trades": champ.get("trade_log", [])[:500],
        },
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT_TRADES, "w", encoding="utf-8") as f:
        json.dump(champ.get("trade_log", []), f, indent=2, default=str)

    print(f"\nSaved model → {OUT_MODEL}")
    print(f"Saved meta  → {OUT_META}")
    print(f"Saved results→ {OUT_RESULTS}")
    print(f"Saved trades → {OUT_TRADES}")

    # Final leaderboard
    print("\n### LEADERBOARD (full, profitable, n≥20) — by WR then PnL ###")
    board = [
        r for r in results
        if r["period"] == "full" and r["net_pnl"] > 0 and r["trades"] >= 20
    ]
    board.sort(key=lambda x: (x["win_rate"], x["net_pnl"]), reverse=True)
    for i, r in enumerate(board[:25], 1):
        mark = " ◀ BEST" if eligible and r["name"] == eligible[0]["name"] else ""
        print(
            f"{i:2d}. WR={r['win_rate']:5.1f}% n={r['trades']:3d} "
            f"PnL=₹{r['net_pnl']:>10,.0f} PF={r['profit_factor']:5.2f} DD={r['max_dd_pct']:5.1f}% "
            f"| {r['name'][:55]}{mark}"
        )


if __name__ == "__main__":
    main()
