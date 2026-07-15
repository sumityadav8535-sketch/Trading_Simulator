"""
ML-enhanced Nifty F&O EMA Short strategy.
Engineers 40+ indicators, trains GradientBoosting + RandomForest on time-split data,
filters trades by predicted win probability, compares OOS vs baseline.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys_path = ROOT / "scripts"
import sys

sys.path.insert(0, str(ROOT))

from scripts.intraday_fno_search import (  # noqa: E402
    CAPITAL,
    DATA_DIR,
    FORCE_EXIT,
    INSTRUMENTS,
    MARKET_OPEN,
    NO_ENTRY_AFTER,
    SLIPPAGE_PTS,
    TARGET_DAILY,
    add_indicators,
    load_instrument,
    lots_for_risk,
    max_lots,
    sig_ema_short,
)

OUT = ROOT / "data" / "intraday_fno_ml_results.json"
MODEL_META = ROOT / "data" / "intraday_fno_ml_model.json"
MODEL_PKL = ROOT / "data" / "intraday_fno_ml_model.pkl"
TRADES_OUT = ROOT / "data" / "intraday_fno_ml_trades.json"

LOT_SIZE = 75
MARGIN = 65_000
RISK_PCT = 1.5
TARGET_R = 1.5
TRADES_PER_DAY = 4


@dataclass
class BacktestStats:
    name: str
    trades: int
    win_rate: float
    net_pnl: float
    avg_daily: float
    profit_factor: float
    max_dd_pct: float
    days_above_1k: int
    period: str


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """40+ technical features for ML — no lookahead."""
    d = add_indicators(df)

    # EMA structure
    d["ema9_ema21"] = (d["ema_9"] - d["ema_21"]) / d["close"] * 100
    d["ema21_ema50"] = (d["ema_21"] - d["ema_50"]) / d["close"] * 100
    d["px_ema9"] = (d["close"] - d["ema_9"]) / d["close"] * 100
    d["px_ema21"] = (d["close"] - d["ema_21"]) / d["close"] * 100
    d["px_ema50"] = (d["close"] - d["ema_50"]) / d["close"] * 100
    d["px_vwap"] = (d["close"] - d["vwap"]) / d["vwap"] * 100

    # ADX / DI
    up = d["high"].diff()
    down = -d["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["di_plus"] = plus_di
    d["di_minus"] = minus_di
    d["di_diff"] = plus_di - minus_di

    # Bollinger
    d["bb_mid"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"] * 100
    d["bb_pct"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # Stochastic
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = (d["close"] - low14) / (high14 - low14).replace(0, np.nan) * 100
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # Momentum / ROC
    for n in (3, 6, 12, 24):
        d[f"roc_{n}"] = d["close"].pct_change(n) * 100

    # Volume (index proxies often have zero volume — use range-based activity)
    if (d["volume"] > 0).any():
        d["vol_ratio"] = d["volume"] / d["vol_sma"].replace(0, np.nan)
        d["vol_z"] = (d["volume"] - d["vol_sma"]) / d["volume"].rolling(20).std().replace(0, np.nan)
    else:
        bar_range = (d["high"] - d["low"]).replace(0, np.nan)
        range_sma = bar_range.rolling(20).mean()
        d["vol_ratio"] = bar_range / range_sma.replace(0, np.nan)
        d["vol_z"] = (bar_range - range_sma) / bar_range.rolling(20).std().replace(0, np.nan)

    # Candle anatomy
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_pct"] = (d["close"] - d["open"]) / rng * 100
    d["upper_wick"] = (d["high"] - d[["open", "close"]].max(axis=1)) / rng * 100
    d["lower_wick"] = (d[["open", "close"]].min(axis=1) - d["low"]) / rng * 100

    # Session context
    d["day_high"] = d.groupby("session_date")["high"].cummax()
    d["day_low"] = d.groupby("session_date")["low"].cummin()
    d["day_range"] = d["day_high"] - d["day_low"]
    d["range_pos"] = (d["close"] - d["day_low"]) / d["day_range"].replace(0, np.nan)
    d["pct_from_open"] = (d["close"] - d["day_open"]) / d["day_open"] * 100
    d["session_ret"] = d.groupby("session_date")["close"].pct_change().groupby(d["session_date"]).cumsum() * 100

    # Time features
    d["hour"] = d.index.hour + d.index.minute / 60
    d["mins_from_open"] = (d.index.hour - 9) * 60 + d.index.minute - 15
    d["mins_to_close"] = (15 * 60 + 30) - (d.index.hour * 60 + d.index.minute)

    # Lags
    d["rsi_chg"] = d["rsi"] - d["prev_rsi"]
    d["atr_pct"] = d["atr"] / d["close"] * 100

    # CCI / Williams %R
    tp = (d["high"] + d["low"] + d["close"]) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda s: np.abs(s - s.mean()).mean(), raw=True)
    d["cci"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    d["williams_r"] = (high14 - d["close"]) / (high14 - low14).replace(0, np.nan) * -100

    # Trend slopes & volatility regime
    d["ema9_slope"] = d["ema_9"].pct_change(3) * 100
    d["ema21_slope"] = d["ema_21"].pct_change(3) * 100
    d["vwap_slope"] = d["vwap"].pct_change(3) * 100
    d["atr_expansion"] = d["atr"] / d["atr"].rolling(20).mean().replace(0, np.nan)

    # Price structure
    d["dist_day_high"] = (d["day_high"] - d["close"]) / d["day_range"].replace(0, np.nan)
    d["bearish_3"] = (
        (d["close"] < d["open"])
        & (d["close"].shift(1) < d["open"].shift(1))
        & (d["close"].shift(2) < d["open"].shift(2))
    ).astype(int)
    d["rsi_macd_div"] = d["roc_6"] - d["macd_hist"]

    # Ichimoku-lite baseline
    high26 = d["high"].rolling(26).max()
    low26 = d["low"].rolling(26).min()
    d["px_kijun"] = (d["close"] - (high26 + low26) / 2) / d["close"] * 100

    return d


FEATURE_COLS = [
    "ema9_ema21", "ema21_ema50", "px_ema9", "px_ema21", "px_ema50", "px_vwap",
    "rsi", "prev_rsi", "rsi_chg", "adx", "di_plus", "di_minus", "di_diff",
    "bb_width", "bb_pct", "macd", "macd_sig", "macd_hist",
    "stoch_k", "stoch_d", "roc_3", "roc_6", "roc_12", "roc_24",
    "vol_ratio", "vol_z", "body_pct", "upper_wick", "lower_wick",
    "range_pos", "pct_from_open", "session_ret", "atr_pct",
    "cci", "williams_r", "ema9_slope", "ema21_slope", "vwap_slope",
    "atr_expansion", "dist_day_high", "bearish_3", "rsi_macd_div", "px_kijun",
    "hour", "mins_from_open", "mins_to_close", "strong_open_reject",
]


def base_short_signal(row) -> bool:
    sig = sig_ema_short(row, None, TARGET_R)
    return sig is not None


def loose_short_signal(row) -> bool:
    """Broader candidate pool for ML to learn from."""
    if pd.isna(row.get("ema_9")) or pd.isna(row.get("ema_21")) or pd.isna(row.get("rsi")):
        return False
    trend = row["ema_9"] < row["ema_21"]
    below_vwap = pd.isna(row.get("vwap")) or row["close"] < row["vwap"]
    rsi_ok = 40 < row["rsi"] < 65
    return trend and below_vwap and rsi_ok


def simulate_short_trade(
    df: pd.DataFrame, entry_idx: int, stop: float, target: float
) -> dict:
    """Simulate short from next-bar open; return exit details and PnL points."""
    empty = {
        "win": False,
        "pnl_pts": 0.0,
        "exit_price": None,
        "exit_time": None,
        "exit_reason": "no_fill",
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
                "win": False,
                "pnl_pts": entry - exit_px,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "stop_hit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if lo <= target:
            exit_px = target + SLIPPAGE_PTS
            return {
                "win": True,
                "pnl_pts": entry - exit_px,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "target_hit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if ts.time() >= FORCE_EXIT:
            exit_px = cl + SLIPPAGE_PTS
            return {
                "win": entry > cl,
                "pnl_pts": entry - exit_px,
                "exit_price": round(exit_px, 2),
                "exit_time": ts.isoformat(),
                "exit_reason": "eod_exit",
                "entry_price": round(entry, 2),
                "entry_time": entry_ts.isoformat(),
            }
        if j > entry_idx and ts.date() != entry_ts.date():
            break
    return empty


def simulate_short_outcome(df: pd.DataFrame, entry_idx: int, stop: float, target: float) -> tuple[int, float]:
    outcome = simulate_short_trade(df, entry_idx, stop, target)
    return (1 if outcome["win"] else 0), outcome["pnl_pts"]


def build_dataset(df: pd.DataFrame, use_loose: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        ts = df.index[i]
        if ts.time() < MARKET_OPEN or ts.time() > NO_ENTRY_AFTER:
            continue

        is_candidate = loose_short_signal(row) if use_loose else base_short_signal(row)
        if not is_candidate:
            continue

        sig = sig_ema_short(row, None, TARGET_R)
        if sig:
            stop, target = float(sig["stop"]), float(sig["target"])
        else:
            stop = float(row["ema_21"])
            risk = stop - row["close"]
            if risk <= 0:
                continue
            target = row["close"] - risk * TARGET_R

        entry_idx = i + 1
        if entry_idx >= len(df) or df.index[entry_idx].date() != ts.date():
            continue

        label, pnl_pts = simulate_short_outcome(df, entry_idx, stop, target)
        feat = {c: row.get(c, np.nan) for c in FEATURE_COLS}
        feat["label"] = label
        feat["pnl_pts"] = pnl_pts
        feat["timestamp"] = ts
        feat["session_date"] = row["session_date"]
        feat["stop"] = stop
        feat["target"] = target
        feat["is_base_signal"] = base_short_signal(row)
        rows.append(feat)

    return pd.DataFrame(rows)


def time_split(dset: pd.DataFrame, train_frac: float = 0.7):
    dates = sorted(dset["session_date"].unique())
    cut = int(len(dates) * train_frac)
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train = dset[dset["session_date"].isin(train_dates)].copy()
    test = dset[dset["session_date"].isin(test_dates)].copy()
    return train, test, dates[:cut], dates[cut:]


def train_models(X_train, y_train):
    models = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, max_iter=250,
            min_samples_leaf=25, l2_regularization=0.1, random_state=42,
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
    probs = [model_proba(m, X) for m in models.values()]
    return np.mean(probs, axis=0)


def pick_threshold_from_probs(probs, y_val, pnl_val) -> float:
    best_th, best_score = 0.5, -1e18
    for th in np.arange(0.35, 0.85, 0.05):
        mask = probs >= th
        if mask.sum() < 10:
            continue
        exp_pnl = pnl_val[mask].sum()
        wr = y_val[mask].mean()
        score = exp_pnl * (0.5 + wr)
        if score > best_score:
            best_score, best_th = score, th
    return best_th


def pick_threshold(model, X_val, y_val, pnl_val) -> float:
    return pick_threshold_from_probs(model_proba(model, X_val), y_val, pnl_val)


def _ts_key(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    name: str,
    period: str,
    *,
    record_trades: bool = False,
) -> BacktestStats | tuple[BacktestStats, list[dict]]:
    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    wins = 0
    gp = gl = 0.0
    daily_pnl: dict = {}
    trades = 0
    day_count: dict = {}
    trade_log: list[dict] = []

    sig_by_ts = {}
    for _, row in signals.iterrows():
        key = _ts_key(row["timestamp"])
        sig_by_ts[key] = row

    for i in range(len(df)):
        ts = df.index[i]
        key = _ts_key(ts)
        if key not in sig_by_ts:
            continue
        sig = sig_by_ts[key]
        bar = df.iloc[i]
        sess = bar["session_date"]
        if day_count.get(sess, 0) >= TRADES_PER_DAY:
            continue
        if i + 1 >= len(df):
            continue
        entry_idx = i + 1
        if df.index[entry_idx].date() != ts.date():
            continue

        stop, target = float(sig["stop"]), float(sig["target"])
        outcome = simulate_short_trade(df, entry_idx, stop, target)
        if outcome["exit_reason"] == "no_fill":
            continue

        entry = outcome["entry_price"]
        stop_pts = stop - entry
        max_l = max_lots(equity, MARGIN)
        lots = lots_for_risk(equity, RISK_PCT, stop_pts, LOT_SIZE, max_l)
        if lots <= 0:
            continue

        pnl_pts = outcome["pnl_pts"]
        pnl = pnl_pts * LOT_SIZE * lots
        equity += pnl
        trades += 1
        if pnl > 0:
            wins += 1
            gp += pnl
        else:
            gl += abs(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0) + pnl
        day_count[sess] = day_count.get(sess, 0) + 1

        if record_trades:
            prob = sig.get("prob")
            trade_log.append({
                "trade_no": trades,
                "strategy": name,
                "period": period,
                "instrument": "NIFTY",
                "side": "SHORT",
                "session_date": str(sess),
                "signal_time": key,
                "signal_close": round(float(bar["close"]), 2),
                "entry_time": outcome["entry_time"],
                "entry_price": entry,
                "stop": round(stop, 2),
                "target": round(target, 2),
                "risk_pts": round(stop_pts, 2),
                "target_r": TARGET_R,
                "lots": lots,
                "lot_size": LOT_SIZE,
                "ml_prob": round(float(prob), 3) if prob is not None and not pd.isna(prob) else None,
                "exit_time": outcome["exit_time"],
                "exit_price": outcome["exit_price"],
                "exit_reason": outcome["exit_reason"],
                "pnl_pts": round(pnl_pts, 2),
                "pnl_inr": round(pnl, 2),
                "result": "WIN" if pnl > 0 else "LOSS",
                "equity_after": round(equity, 2),
                "ema_9": round(float(bar["ema_9"]), 2),
                "ema_21": round(float(bar["ema_21"]), 2),
                "ema_50": round(float(bar["ema_50"]), 2),
                "rsi": round(float(bar["rsi"]), 1),
                "vwap": round(float(bar["vwap"]), 2) if pd.notna(bar.get("vwap")) else None,
                "adx": round(float(bar["adx"]), 1) if pd.notna(bar.get("adx")) else None,
                "range_pos": round(float(bar["range_pos"]), 3) if pd.notna(bar.get("range_pos")) else None,
            })

    days = len({ix.date() for ix in df.index if MARKET_OPEN <= ix.time() <= dt_time(15, 30)})
    pf = gp / gl if gl else (999 if gp else 0)
    stats = BacktestStats(
        name=name,
        trades=trades,
        win_rate=round(wins / trades * 100 if trades else 0, 2),
        net_pnl=round(equity - CAPITAL, 2),
        avg_daily=round((equity - CAPITAL) / max(days, 1), 2),
        profit_factor=round(pf, 2),
        max_dd_pct=round(max_dd, 2),
        days_above_1k=sum(1 for p in daily_pnl.values() if p >= TARGET_DAILY),
        period=period,
    )
    if record_trades:
        return stats, trade_log
    return stats


def feature_importance(model, feature_names) -> dict:
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
    else:
        clf = model
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        return dict(sorted(zip(feature_names, imp), key=lambda x: -x[1])[:15])
    return {}


def main():
    print("ML-Enhanced Nifty F&O Short Strategy")
    print(f"Capital Rs {CAPITAL:,.0f} | Risk {RISK_PCT}% | Target Rs {TARGET_DAILY:,.0f}/day\n")

    df = enrich_features(load_instrument("NIFTY"))
    if df.empty:
        print("No NIFTY data")
        return

    print("Building labeled dataset (loose short candidates)...")
    dset = build_dataset(df, use_loose=True)
    if dset.empty:
        print("  No training samples — check signal filters and data.")
        return
    dset = dset.dropna(subset=FEATURE_COLS)
    print(f"  Samples: {len(dset)} | Win rate (raw): {dset['label'].mean()*100:.1f}%")

    train, test, train_dates, test_dates = time_split(dset, 0.7)
    print(f"  Train: {len(train_dates)} sessions | Test: {len(test_dates)} sessions (OOS)")

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["label"].values

    # Validation split from tail of train for threshold
    val_cut = int(len(train) * 0.8)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]
    pnl_val = train["pnl_pts"].values[val_cut:]

    print("\nTraining models...")
    models = train_models(X_tr, y_tr)

    best_name, best_model, best_th, best_auc = "", None, 0.5, 0
    for name, model in models.items():
        auc = roc_auc_score(y_test, model_proba(model, X_test)) if len(set(y_test)) > 1 else 0
        print(f"  {name}: OOS ROC-AUC = {auc:.3f}")
        th = pick_threshold(model, X_val, y_val, pnl_val)
        if auc >= best_auc:
            best_auc, best_name, best_model, best_th = auc, name, model, th

    ens_val = ensemble_proba(models, X_val)
    ens_test = ensemble_proba(models, X_test)
    ens_auc = roc_auc_score(y_test, ens_test) if len(set(y_test)) > 1 else 0
    print(f"  ensemble: OOS ROC-AUC = {ens_auc:.3f}")

    ens_th = pick_threshold_from_probs(ens_val, y_val, pnl_val)
    if ens_auc >= best_auc:
        best_auc, best_name, best_th = ens_auc, "ensemble", ens_th
        best_probs_test = ens_test
        best_probs_all = ensemble_proba(models, dset[FEATURE_COLS].values)
    else:
        best_probs_test = model_proba(best_model, X_test)
        best_probs_all = model_proba(best_model, dset[FEATURE_COLS].values)

    print(f"\nSelected: {best_name} | threshold={best_th:.2f} | OOS AUC={best_auc:.3f}")

    # Build signal sets for backtest on TEST period only
    test_start = min(test_dates)
    test_df = df[df["session_date"] >= test_start]

    # Baseline: base EMA short only on test period
    base_test = test[test["is_base_signal"]].copy()
    base_signals = base_test[["timestamp", "stop", "target"]]

    # ML enhanced: loose candidates with prob >= threshold
    test = test.copy()
    test["prob"] = best_probs_test
    ml_test = test[test["prob"] >= best_th]
    ml_signals = ml_test[["timestamp", "stop", "target", "prob"]]

    # ML + base intersection (highest precision)
    hybrid = test[(test["is_base_signal"]) & (test["prob"] >= best_th)]
    hybrid_signals = hybrid[["timestamp", "stop", "target", "prob"]]

    print("\n--- OUT-OF-SAMPLE BACKTEST (last 30% sessions) ---")
    results = []
    for sigs, label in [
        (base_signals, "Baseline EMA Short"),
        (ml_signals, f"ML Enhanced ({best_name})"),
        (hybrid_signals, f"ML + EMA Hybrid"),
    ]:
        stats = backtest_signals(test_df, sigs, label, f"OOS {test_dates[0]}→{test_dates[-1]}")
        results.append(stats)
        print(
            f"{stats.name:<30} Trades={stats.trades:>3} WR={stats.win_rate:>5.1f}% "
            f"P&L=Rs {stats.net_pnl:>9,.0f} Rs/day={stats.avg_daily:>7,.0f} "
            f"PF={stats.profit_factor:.2f} DD={stats.max_dd_pct:.1f}% Days>1k={stats.days_above_1k}"
        )

    # Full period ML backtest for deployment estimate
    dset["prob"] = best_probs_all
    full_ml = dset[dset["prob"] >= best_th]
    ml_strategy_name = f"ML Enhanced ({best_name})"
    full_stats, full_trades = backtest_signals(
        df,
        full_ml[["timestamp", "stop", "target", "prob"]],
        ml_strategy_name,
        "full",
        record_trades=True,
    )
    _, oos_trades = backtest_signals(
        test_df,
        ml_signals,
        f"{ml_strategy_name} OOS",
        f"OOS {test_dates[0]}→{test_dates[-1]}",
        record_trades=True,
    )
    print(
        f"\n--- FULL PERIOD (ML, for reference) ---\n"
        f"{full_stats.name}: Trades={full_stats.trades} WR={full_stats.win_rate}% "
        f"P&L=Rs {full_stats.net_pnl:,.0f} Rs/day=Rs {full_stats.avg_daily:,.0f}"
    )
    print(f"  Trade log: {len(full_trades)} entries (full) | {len(oos_trades)} entries (OOS)")

    imp = feature_importance(
        models.get("random_forest", next(iter(models.values()))),
        FEATURE_COLS,
    )
    print("\nTop ML features:")
    for k, v in list(imp.items())[:10]:
        print(f"  {k}: {v:.4f}")

    improvement = results[1].avg_daily - results[0].avg_daily if len(results) >= 2 else 0
    print(f"\nOOS improvement vs baseline: Rs {improvement:+,.0f}/day")

    payload = {
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "model": best_name,
        "threshold": best_th,
        "oos_auc": round(best_auc, 4),
        "feature_cols": FEATURE_COLS,
        "top_features": imp,
        "oos_results": [r.__dict__ for r in results],
        "full_period_ml": full_stats.__dict__,
        "train_sessions": [str(d) for d in train_dates],
        "test_sessions": [str(d) for d in test_dates],
        "strategy_rules": {
            "instrument": "NIFTY Futures",
            "side": "SHORT",
            "base_filter": "EMA9<EMA21<EMA50, close<VWAP, RSI>50",
            "ml_filter": f"P(win) >= {best_th:.2f} using {best_name}",
            "stop": "EMA21",
            "target": f"{TARGET_R}R",
            "exit": "15:15 IST",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trades_payload = {
        "strategy": ml_strategy_name,
        "instrument": "NIFTY",
        "capital": CAPITAL,
        "risk_pct": RISK_PCT,
        "threshold": best_th,
        "lot_size": LOT_SIZE,
        "target_r": TARGET_R,
        "model": best_name,
        "full_period": {
            "summary": full_stats.__dict__,
            "trades": full_trades,
        },
        "oos": {
            "period": f"{test_dates[0]}→{test_dates[-1]}",
            "trades": oos_trades,
        },
    }
    TRADES_OUT.write_text(json.dumps(trades_payload, indent=2), encoding="utf-8")
    MODEL_META.write_text(json.dumps({
        "model_type": best_name,
        "threshold": best_th,
        "features": FEATURE_COLS,
        "top_features": imp,
    }, indent=2), encoding="utf-8")

    import pickle

    deploy_model = models.get("random_forest", best_model)
    if best_name == "ensemble":
        deploy_model = models
    with MODEL_PKL.open("wb") as fh:
        pickle.dump({
            "model_type": best_name,
            "model": deploy_model if best_name != "ensemble" else None,
            "models": models if best_name == "ensemble" else None,
            "threshold": best_th,
            "feature_cols": FEATURE_COLS,
        }, fh)
    print(f"\nSaved {OUT}")
    print(f"Saved {TRADES_OUT}")
    print(f"Saved {MODEL_PKL}")


if __name__ == "__main__":
    main()