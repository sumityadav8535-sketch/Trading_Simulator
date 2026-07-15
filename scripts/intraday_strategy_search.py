"""
Nifty 100 intraday 5-minute strategy tournament.

Downloads max available 5m history (yfinance limit: ~60 calendar days, not full 90d),
tests multiple strategy/filter combinations on Rs 1,00,000 capital, ranks by
profitability with win-rate and drawdown constraints.

Usage:
    python scripts/intraday_strategy_search.py
    python scripts/intraday_strategy_search.py --skip-download
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import time as dt_time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from trading.constants import NIFTY100_INDEX_TICKER  # noqa: E402
from trading.services.nifty100 import get_nifty100_symbols  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402

DATA_DIR = ROOT / "data" / "intraday_5m"
CAPITAL = 100_000.0
SLIPPAGE_PCT = 0.0005  # 0.05% per side
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
NO_ENTRY_AFTER = dt_time(14, 30)
FORCE_EXIT = dt_time(15, 15)
ORB_END = dt_time(9, 45)
ORB_END_15 = dt_time(9, 30)


@dataclass
class StrategyResult:
    name: str
    trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    net_pnl: float
    max_drawdown_pct: float
    avg_r: float
    final_equity: float
    params: dict = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    qty: int
    pnl: float
    reason: str
    strategy: str


def _normalize_df(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    sub = df.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        if ticker and ticker in sub.columns.get_level_values(0):
            sub = sub[ticker]
        elif ticker and ticker in sub.columns.get_level_values(1):
            sub = sub.xs(ticker, axis=1, level=1)
        else:
            sub.columns = sub.columns.get_level_values(0)

    rename = {}
    for col in sub.columns:
        key = str(col).lower().replace(" ", "_")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = key
    sub = sub.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    if not keep:
        return pd.DataFrame()
    sub = sub[keep].dropna(subset=["close"])
    if sub.index.tz is None:
        sub.index = sub.index.tz_localize("Asia/Kolkata")
    else:
        sub.index = sub.index.tz_convert("Asia/Kolkata")
    return sub.sort_index()


def download_symbol_5m(symbol: str, period: str = "60d") -> pd.DataFrame:
    ticker = yfinance_ticker(symbol)
    raw = yf.download(ticker, interval="5m", period=period, progress=False, auto_adjust=False)
    return _normalize_df(raw, ticker)


def load_or_fetch_data(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols, 1):
        path = DATA_DIR / f"{sym}.pkl"
        if path.exists() and not force:
            df = pd.read_pickle(path)
            if not df.empty:
                cache[sym] = df
                continue

        print(f"  [{i}/{len(symbols)}] {sym}...", flush=True)
        df = download_symbol_5m(sym)
        if not df.empty:
            df.to_pickle(path)
            cache[sym] = df
        time.sleep(0.15)

    idx_path = DATA_DIR / "_NIFTY100_INDEX.pkl"
    if idx_path.exists() and not force:
        cache["_INDEX"] = pd.read_pickle(idx_path)
    else:
        print("  [index] Nifty 100 (^CNX100)...", flush=True)
        raw = yf.download(NIFTY100_INDEX_TICKER, interval="5m", period="60d", progress=False, auto_adjust=False)
        idx_df = _normalize_df(raw, NIFTY100_INDEX_TICKER)
        if not idx_df.empty:
            idx_df.to_pickle(idx_path)
            cache["_INDEX"] = idx_df

    return cache


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df

    out = df.copy()
    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["vol_sma_20"] = out["volume"].rolling(20).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    prev = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev).abs(),
        (out["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up = out["high"].diff()
    down = -out["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = out["atr_14"]
    plus_di = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["di_plus"] = plus_di
    out["di_minus"] = minus_di

    out["bb_mid"] = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    out["session_date"] = out.index.date
    typical = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = (typical * out["volume"]).groupby(out["session_date"]).cumsum() / out["volume"].groupby(
        out["session_date"]
    ).cumsum().replace(0, np.nan)

    rng = out["high"] - out["low"]
    out["strong_close"] = (out["close"] - out["low"]) / rng.replace(0, np.nan) >= 0.7
    out["bar_range_pct"] = rng / out["close"]
    out["low_3"] = out["low"].rolling(3).min()
    out["low_5"] = out["low"].rolling(5).min()
    out["prev_rsi"] = out["rsi_14"].shift(1)

    return out


def prepare_universe(cache: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    stocks = {}
    for sym, df in cache.items():
        if sym.startswith("_"):
            continue
        enriched = add_intraday_indicators(df)
        if len(enriched) >= 100:
            stocks[sym] = enriched

    index_df = add_intraday_indicators(cache.get("_INDEX", pd.DataFrame()))
    if not index_df.empty:
        index_df["bullish"] = index_df["close"] > index_df["vwap"]
    return stocks, index_df


def _in_session(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def _can_enter(ts: pd.Timestamp) -> bool:
    return MARKET_OPEN <= ts.time() <= NO_ENTRY_AFTER


def _apply_slippage(price: float, side: str) -> float:
    if side == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def position_qty(equity: float, risk_pct: float, entry: float, stop: float) -> int:
    if any(pd.isna(x) for x in (equity, entry, stop)):
        return 0
    risk_per_share = entry - stop
    if risk_per_share <= 0 or pd.isna(risk_per_share):
        return 0
    risk_amount = equity * (risk_pct / 100)
    qty = int(risk_amount / risk_per_share)
    if qty * entry > equity * 0.25:
        qty = int((equity * 0.25) / entry)
    return max(qty, 0)


# ── Signal generators (return dict with stop, target, tag or None) ──────────

def sig_orb(row, hist, orb_high, orb_low, vol_mult: float, min_range_pct: float):
    if orb_high is None or orb_low is None:
        return None
    rng = orb_high - orb_low
    if rng <= 0 or rng / row["close"] < min_range_pct:
        return None
    if row["close"] <= orb_high:
        return None
    if pd.isna(row.get("vol_sma_20")) or row["volume"] < row["vol_sma_20"] * vol_mult:
        return None
    stop = orb_low
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * 1.5, "tag": "orb"}


def sig_vwap_pullback(row, hist, target_r: float, rsi_lo: float, rsi_hi: float):
    if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
        return None
    if row["ema_9"] <= row["ema_21"]:
        return None
    if not (rsi_lo <= row["rsi_14"] <= rsi_hi):
        return None
    touch = abs(row["low"] - row["ema_9"]) / row["ema_9"] <= 0.003
    if not touch:
        return None
    if row["adx_14"] < 18:
        return None
    stop = min(row["low"], float(row.get("low_3", row["low"])))
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "vwap_pb"}


def sig_rsi_reversal(row, prev, hist, target_r: float):
    if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
        return None
    prev_rsi = row.get("prev_rsi")
    if pd.isna(prev_rsi):
        return None
    if not (prev_rsi < 35 and row["rsi_14"] > 40):
        return None
    if row["ema_21"] < row["ema_50"]:
        return None
    stop = float(row.get("low_5", row["low"]))
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "rsi_rev"}


def sig_bb_breakout(row, hist, vol_mult: float, width_max: float):
    if pd.isna(row.get("bb_width")) or row["bb_width"] > width_max:
        return None
    if row["close"] <= row["bb_upper"]:
        return None
    if row["volume"] < row["vol_sma_20"] * vol_mult:
        return None
    if row["adx_14"] < 16:
        return None
    stop = row["bb_mid"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * 2.0, "tag": "bb_sq"}


def sig_ema_momentum(row, hist, target_r: float):
    if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
        return None
    if row["close"] <= row["vwap"]:
        return None
    if not row.get("strong_close", False):
        return None
    if row["di_plus"] <= row["di_minus"]:
        return None
    if row["adx_14"] < 20:
        return None
    stop = row["ema_21"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "ema_mom"}


def sig_combo_elite(row, prev, hist, target_r: float):
    """VWAP trend + RSI dip + strong close + ADX."""
    if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
        return None
    if not (row["ema_9"] > row["ema_21"]):
        return None
    if prev is None or prev["rsi_14"] >= 42:
        return None
    if row["rsi_14"] < 42 or row["rsi_14"] > 58:
        return None
    if not row.get("strong_close", False):
        return None
    if row["adx_14"] < 22 or row["di_plus"] <= row["di_minus"]:
        return None
    if row["volume"] < row["vol_sma_20"] * 1.1:
        return None
    stop = min(float(row.get("low_3", row["low"])), row["ema_21"])
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "combo"}


def _index_bullish_map(index_df: pd.DataFrame) -> dict[pd.Timestamp, bool]:
    if index_df.empty:
        return {}
    return {ts: bool(row.get("bullish", False)) for ts, row in index_df.iterrows()}


def build_timeline(stocks: dict[str, pd.DataFrame], orb_minutes: int = 30) -> list[tuple]:
    """Pre-build chronological events once. Tuple: ts, sym, sess, row, prev_row, orb, next_ts."""
    timeline: list[tuple] = []
    for sym, df in stocks.items():
        prev_row = None
        session_orb: dict = {}
        index_list = list(df.index)
        for i, ts in enumerate(index_list):
            row = df.iloc[i]
            if not _in_session(ts):
                prev_row = row
                continue
            sess = row["session_date"]
            if sess not in session_orb:
                session_orb[sess] = {"high": None, "low": None, "done": False}
            orb = session_orb[sess]
            t = ts.time()
            orb_cutoff = ORB_END if orb_minutes == 30 else ORB_END_15
            if t <= orb_cutoff:
                orb["high"] = row["high"] if orb["high"] is None else max(orb["high"], row["high"])
                orb["low"] = row["low"] if orb["low"] is None else min(orb["low"], row["low"])
            elif not orb["done"]:
                orb["done"] = True

            next_ts = None
            if i + 1 < len(index_list):
                nxt = index_list[i + 1]
                if nxt.date() == ts.date():
                    next_ts = nxt

            timeline.append((
                ts, sym, sess, row, prev_row,
                {"high": orb["high"], "low": orb["low"], "done": orb["done"]},
                next_ts,
            ))
            prev_row = row

    timeline.sort(key=lambda x: (x[0], x[1]))
    return timeline


def run_backtest(
    stocks: dict[str, pd.DataFrame],
    timeline: list[tuple],
    index_bullish: dict[pd.Timestamp, bool],
    signal_fn: Callable,
    name: str,
    risk_pct: float = 1.5,
    max_positions: int = 4,
    market_filter: bool = False,
) -> StrategyResult:
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    trades: list[Trade] = []
    open_positions: list[dict] = []
    pending_entries: list[dict] = []
    traded_today: set[tuple] = set()

    def close_position(pos: dict, ts: pd.Timestamp, exit_raw: float, reason: str) -> None:
        nonlocal equity, peak, max_dd
        exit_p = _apply_slippage(exit_raw, "sell")
        pnl = (exit_p - pos["entry"]) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        trades.append(Trade(
            symbol=pos["sym"],
            entry_time=pos["entry_ts"],
            exit_time=ts,
            entry=pos["entry"],
            exit=exit_p,
            qty=pos["qty"],
            pnl=pnl,
            reason=reason,
            strategy=name,
        ))

    for ts, sym, sess, row, prev_row, orb, next_ts in timeline:
        # Fill pending entries scheduled for this bar
        still_pending = []
        for pe in pending_entries:
            if pe["entry_ts"] != ts or pe["sym"] != sym:
                still_pending.append(pe)
                continue
            if len(open_positions) >= max_positions:
                still_pending.append(pe)
                continue
            entry = _apply_slippage(float(row["open"]), "buy")
            if entry <= pe["stop"]:
                continue
            qty = position_qty(equity, risk_pct, entry, pe["stop"])
            if qty <= 0:
                continue
            open_positions.append({
                "sym": sym,
                "entry_ts": ts,
                "entry": entry,
                "stop": pe["stop"],
                "target": pe["target"],
                "qty": qty,
                "tag": pe["tag"],
            })
        pending_entries = still_pending

        # Update open positions for this symbol
        remaining = []
        for pos in open_positions:
            if pos["sym"] != sym:
                remaining.append(pos)
                continue
            low, high, close = float(row["low"]), float(row["high"]), float(row["close"])
            if low <= pos["stop"]:
                close_position(pos, ts, pos["stop"], "sl")
            elif high >= pos["target"]:
                close_position(pos, ts, pos["target"], "target")
            elif ts.time() >= FORCE_EXIT:
                close_position(pos, ts, close, "eod")
            else:
                remaining.append(pos)
        open_positions = remaining

        if not _can_enter(ts):
            continue
        if len(open_positions) + len(pending_entries) >= max_positions:
            continue
        if (sym, sess) in traded_today:
            continue
        if market_filter and not index_bullish.get(ts, False):
            continue

        sig = signal_fn(row, prev_row, None, orb)
        if sig is None or next_ts is None:
            continue
        stop, target = sig["stop"], sig["target"]
        if any(pd.isna(x) for x in (stop, target)) or target <= row["close"]:
            continue

        pending_entries.append({
            "sym": sym,
            "entry_ts": next_ts,
            "stop": float(stop),
            "target": float(target),
            "tag": sig["tag"],
        })
        traded_today.add((sym, sess))

    for pos in open_positions:
        sym_df = stocks[pos["sym"]]
        last = sym_df.iloc[-1]
        close_position(pos, sym_df.index[-1], float(last["close"]), "final")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(trades) * 100 if trades else 0.0

    return StrategyResult(
        name=name,
        trades=len(trades),
        win_rate=round(wr, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        net_pnl=round(equity - CAPITAL, 2),
        max_drawdown_pct=round(max_dd, 2),
        avg_r=0.0,
        final_equity=round(equity, 2),
    )


def build_strategies() -> list[tuple[str, Callable, dict]]:
    strategies = []

    for vol in (1.2, 1.5):
        for min_rng in (0.002, 0.003):
            strategies.append((
                f"ORB30 vol{vol} rng{min_rng}",
                lambda row, prev, hist, orb, v=vol, m=min_rng: sig_orb(
                    row, hist, orb.get("high"), orb.get("low"), v, m
                ),
                {"family": "orb30"},
            ))

    for tr in (1.0, 1.5, 2.0):
        strategies.append((
            f"VWAP Pullback {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_vwap_pullback(row, hist, r, 45, 60),
            {"family": "vwap"},
        ))

    for tr in (1.0, 1.5):
        strategies.append((
            f"RSI Reversal {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_rsi_reversal(row, prev, hist, r),
            {"family": "rsi"},
        ))

    for vol in (1.3, 1.6):
        strategies.append((
            f"BB Squeeze vol{vol}",
            lambda row, prev, hist, orb, v=vol: sig_bb_breakout(row, hist, v, 0.04),
            {"family": "bb"},
        ))

    for tr in (1.5, 2.0):
        strategies.append((
            f"EMA Momentum {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_ema_momentum(row, hist, r),
            {"family": "ema"},
        ))

    for tr in (1.5, 2.0):
        strategies.append((
            f"Combo Elite {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_combo_elite(row, prev, hist, r),
            {"family": "combo"},
        ))

    return strategies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    symbols = get_nifty100_symbols()
    print(f"Nifty 100 intraday 5m strategy search | capital Rs {CAPITAL:,.0f}")
    print(f"Note: yfinance 5m data limited to ~60 days (not full 90d / 3 months)\n")

    if not args.skip_download:
        print("Downloading 5m data...")
        cache = load_or_fetch_data(symbols, force=args.force_download)
    else:
        cache = load_or_fetch_data(symbols, force=False)

    stocks, index_df = prepare_universe(cache)
    if not stocks:
        print("No stock data loaded.")
        return

    dates = set()
    for df in stocks.values():
        dates.update(df.index.date)
    print(f"Loaded {len(stocks)} symbols | {min(dates)} → {max(dates)} | {len(dates)} sessions")
    print("Building timeline...", flush=True)
    timeline = build_timeline(stocks)
    index_bullish = _index_bullish_map(index_df)
    print(f"Timeline events: {len(timeline):,}\n")

    base_strategies = build_strategies()
    results: list[StrategyResult] = []

    configs = [
        (1.0, 3, False),
        (1.5, 4, False),
        (1.5, 4, True),
        (2.0, 5, True),
    ]

    total = len(base_strategies) * len(configs)
    n = 0
    for strat_name, sig_fn, meta in base_strategies:
        for risk_pct, max_pos, mkt_filter in configs:
            n += 1
            suffix = f" r{risk_pct}% pos{max_pos}" + (" +idx" if mkt_filter else "")
            full_name = strat_name + suffix
            print(f"[{n}/{total}] {full_name}...", flush=True)
            res = run_backtest(
                stocks, timeline, index_bullish, sig_fn, full_name,
                risk_pct=risk_pct, max_positions=max_pos, market_filter=mkt_filter,
            )
            res.params = {**meta, "risk_pct": risk_pct, "max_positions": max_pos, "market_filter": mkt_filter}
            results.append(res)

    # Rank: profitable first, then PF, then WR, then return
    profitable = [r for r in results if r.net_pnl > 0 and r.trades >= 15]
    profitable.sort(key=lambda r: (-r.profit_factor, -r.win_rate, -r.total_return_pct))

    all_sorted = sorted(results, key=lambda r: (-r.net_pnl, -r.profit_factor, -r.win_rate))

    print("\n" + "=" * 100)
    print("TOP PROFITABLE STRATEGIES (min 15 trades, net P&L > 0)")
    print("=" * 100)
    print(f"{'Strategy':<42} {'Trades':>6} {'WR%':>6} {'Return%':>8} {'PF':>5} {'MaxDD%':>7} {'Net P&L':>10}")
    print("-" * 100)
    for r in profitable[:20]:
        print(
            f"{r.name:<42} {r.trades:>6} {r.win_rate:>5.1f}% "
            f"{r.total_return_pct:>7.2f}% {r.profit_factor:>5.2f} "
            f"{r.max_drawdown_pct:>6.2f}% {r.net_pnl:>10,.0f}"
        )

    high_wr = [r for r in profitable if r.win_rate >= 55]
    high_wr.sort(key=lambda r: (-r.total_return_pct, -r.profit_factor))
    print("\n" + "=" * 100)
    print("HIGH WIN-RATE PROFITABLE (WR >= 55%)")
    print("=" * 100)
    for r in high_wr[:10]:
        print(
            f"{r.name:<42} {r.trades:>6} {r.win_rate:>5.1f}% "
            f"{r.total_return_pct:>7.2f}% PF={r.profit_factor:.2f} DD={r.max_drawdown_pct:.1f}%"
        )

    print("\n" + "=" * 100)
    print("BEST OVERALL (including marginal)")
    print("=" * 100)
    for r in all_sorted[:15]:
        print(
            f"{r.name:<42} {r.trades:>6} {r.win_rate:>5.1f}% "
            f"{r.total_return_pct:>7.2f}% PF={r.profit_factor:.2f} P&L={r.net_pnl:,.0f}"
        )

    out_path = ROOT / "data" / "intraday_strategy_results.json"
    payload = {
        "capital": CAPITAL,
        "symbols": len(stocks),
        "date_range": [str(min(dates)), str(max(dates))],
        "data_note": "yfinance 5m max ~60 calendar days",
        "top_profitable": [r.__dict__ for r in profitable[:20]],
        "high_win_rate": [r.__dict__ for r in high_wr[:10]],
        "best_overall": [r.__dict__ for r in all_sorted[:15]],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    if profitable:
        best = profitable[0]
        print("\n" + "=" * 100)
        print("RECOMMENDED FOR IMPLEMENTATION")
        print("=" * 100)
        print(f"  Strategy : {best.name}")
        print(f"  Trades   : {best.trades}")
        print(f"  Win rate : {best.win_rate}%")
        print(f"  Return   : {best.total_return_pct}% on Rs {CAPITAL:,.0f}")
        print(f"  Net P&L  : Rs {best.net_pnl:,.0f}")
        print(f"  PF       : {best.profit_factor}")
        print(f"  Max DD   : {best.max_drawdown_pct}%")
        print(f"  Params   : {best.params}")
    else:
        print("\nNo profitable configuration met thresholds. Review best_overall for least-bad options.")


if __name__ == "__main__":
    main()