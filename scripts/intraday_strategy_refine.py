"""
Phase-2 intraday search: relaxed filters, fixed index alignment, win-rate focus.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_strategy_search import (  # noqa: E402
    CAPITAL,
    DATA_DIR,
    StrategyResult,
    Trade,
    _apply_slippage,
    _can_enter,
    _in_session,
    add_intraday_indicators,
    build_timeline,
    position_qty,
    prepare_universe,
    sig_bb_breakout,
    sig_combo_elite,
    sig_ema_momentum,
    sig_orb,
    sig_rsi_reversal,
    sig_vwap_pullback,
)
from scripts.intraday_strategy_search import FORCE_EXIT  # noqa: E402

RESULT_PATH = ROOT / "data" / "intraday_strategy_refine.json"


def aligned_index_bullish(index_df: pd.DataFrame, timeline_ts: list) -> dict:
    """Forward-fill index bullish flag onto stock bar timestamps."""
    if index_df.empty:
        return {}
    idx = index_df[["close", "vwap"]].copy()
    idx["bullish"] = idx["close"] > idx["vwap"]
    idx = idx.reset_index()
    ts_col = idx.columns[0]
    idx = idx.rename(columns={ts_col: "ts"})
    stock_ts = pd.DataFrame({"ts": sorted(set(timeline_ts))})
    merged = pd.merge_asof(
        stock_ts.sort_values("ts"),
        idx[["ts", "bullish"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged["bullish"] = merged["bullish"].fillna(False)
    return dict(zip(merged["ts"], merged["bullish"].astype(bool)))


def run_backtest_fast(
    stocks,
    timeline,
    index_bullish,
    signal_fn,
    name: str,
    risk_pct=1.5,
    max_positions=5,
    market_filter=False,
) -> StrategyResult:
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    trades: list[Trade] = []
    open_positions: list[dict] = []
    pending: list[dict] = []
    traded_today: set[tuple] = set()

    def close_pos(pos, ts, exit_raw, reason):
        nonlocal equity, peak, max_dd
        exit_p = _apply_slippage(exit_raw, "sell")
        pnl = (exit_p - pos["entry"]) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        trades.append(Trade(pos["sym"], pos["entry_ts"], ts, pos["entry"], exit_p, pos["qty"], pnl, reason, name))

    for ts, sym, sess, row, prev_row, orb, next_ts in timeline:
        still = []
        for pe in pending:
            if pe["entry_ts"] != ts or pe["sym"] != sym:
                still.append(pe)
                continue
            if len(open_positions) >= max_positions:
                still.append(pe)
                continue
            entry = _apply_slippage(float(row["open"]), "buy")
            if entry <= pe["stop"]:
                continue
            qty = position_qty(equity, risk_pct, entry, pe["stop"])
            if qty <= 0:
                continue
            open_positions.append({
                "sym": sym, "entry_ts": ts, "entry": entry,
                "stop": pe["stop"], "target": pe["target"], "qty": qty,
            })
        pending = still

        remaining = []
        for pos in open_positions:
            if pos["sym"] != sym:
                remaining.append(pos)
                continue
            low, high, close = float(row["low"]), float(row["high"]), float(row["close"])
            if low <= pos["stop"]:
                close_pos(pos, ts, pos["stop"], "sl")
            elif high >= pos["target"]:
                close_pos(pos, ts, pos["target"], "target")
            elif ts.time() >= FORCE_EXIT:
                close_pos(pos, ts, close, "eod")
            else:
                remaining.append(pos)
        open_positions = remaining

        if not _can_enter(ts) or len(open_positions) + len(pending) >= max_positions:
            continue
        if (sym, sess) in traded_today:
            continue
        if market_filter and not index_bullish.get(ts, True):
            continue

        sig = signal_fn(row, prev_row, None, orb)
        if sig is None or next_ts is None:
            continue
        stop, target = float(sig["stop"]), float(sig["target"])
        if np.isnan(stop) or np.isnan(target) or target <= row["close"]:
            continue
        pending.append({"sym": sym, "entry_ts": next_ts, "stop": stop, "target": target})
        traded_today.add((sym, sess))

    for pos in open_positions:
        df = stocks[pos["sym"]]
        close_pos(pos, df.index[-1], float(df.iloc[-1]["close"]), "final")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    return StrategyResult(
        name=name, trades=len(trades), win_rate=round(wr, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        net_pnl=round(equity - CAPITAL, 2),
        max_drawdown_pct=round(max_dd, 2), avg_r=0.0, final_equity=round(equity, 2),
    )


def main():
    cache = {}
    for p in DATA_DIR.glob("*.pkl"):
        sym = p.stem
        cache[sym if not sym.startswith("_") else "_INDEX"] = pd.read_pickle(p)
    stocks, index_df = prepare_universe(cache)
    timeline = build_timeline(stocks)
    ts_list = [t[0] for t in timeline]
    index_bullish = aligned_index_bullish(index_df, ts_list)

    strategies = []

    # Relaxed VWAP pullback
    for tr in (1.0, 1.25, 1.5):
        for adx in (15, 18):
            for rsi_hi in (55, 62):
                strategies.append((
                    f"VWAP-PB {tr}R adx{adx} rsi{rsi_hi}",
                    lambda row, prev, hist, orb, r=tr, a=adx, rh=rsi_hi: sig_vwap_pullback(
                        row, hist, r, 40, rh
                    ) if row["adx_14"] >= a else None,
                    "vwap",
                ))

    # Relaxed combo
    for tr in (1.0, 1.25, 1.5):
        for adx in (18, 20):
            strategies.append((
                f"Combo {tr}R adx{adx}",
                lambda row, prev, hist, orb, r=tr, a=adx: (
                    sig_combo_elite(row, prev, hist, r)
                    if row["adx_14"] >= a else None
                ),
                "combo",
            ))

    # Scalp: VWAP bounce 1R (high WR design)
    def sig_vwap_scalp(row, prev, hist, orb):
        if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
            return None
        if row["ema_9"] < row["ema_21"]:
            return None
        if row["low"] > row["vwap"] * 0.998:
            return None
        if row["rsi_14"] < 38 or row["rsi_14"] > 52:
            return None
        if not row.get("strong_close", False):
            return None
        stop = min(row["low"], row["vwap"] * 0.997)
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * 1.0, "tag": "scalp"}

    for label in ("VWAP Scalp 1R",):
        strategies.append((label, sig_vwap_scalp, "scalp"))

    # ORB relaxed
    for vol in (1.0, 1.2):
        for rng in (0.0015, 0.002):
            strategies.append((
                f"ORB {vol}/{rng}",
                lambda row, prev, hist, orb, v=vol, m=rng: sig_orb(
                    row, hist, orb.get("high"), orb.get("low"), v, m
                ),
                "orb",
            ))

    # EMA momentum relaxed
    for tr in (1.0, 1.25, 1.5):
        strategies.append((
            f"EMA-Mom {tr}R adx18",
            lambda row, prev, hist, orb, r=tr: (
                sig_ema_momentum(row, hist, r) if row["adx_14"] >= 18 else None
            ),
            "ema",
        ))

    # RSI reversal relaxed
    for tr in (1.0, 1.25):
        strategies.append((
            f"RSI-Rev {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_rsi_reversal(row, prev, hist, r),
            "rsi",
        ))

    configs = [
        (1.0, 5, False),
        (1.5, 5, False),
        (1.5, 5, True),
        (2.0, 6, True),
    ]

    results = []
    total = len(strategies) * len(configs)
    n = 0
    for sname, fn, family in strategies:
        for risk, pos, mfilter in configs:
            n += 1
            name = f"{sname} r{risk}% p{pos}" + (" +idx" if mfilter else "")
            print(f"[{n}/{total}] {name}", flush=True)
            r = run_backtest_fast(
                stocks, timeline, index_bullish, fn, name,
                risk_pct=risk, max_positions=pos, market_filter=mfilter,
            )
            r.params = {"family": family, "risk_pct": risk, "max_positions": pos, "market_filter": mfilter}
            results.append(r)

    profitable = [r for r in results if r.net_pnl > 0 and r.trades >= 8]
    profitable.sort(key=lambda r: (-r.win_rate, -r.profit_factor, -r.total_return_pct))

    high_wr = [r for r in profitable if r.win_rate >= 50]
    high_wr.sort(key=lambda r: (-r.total_return_pct, -r.profit_factor))

    best_return = sorted([r for r in results if r.trades >= 5], key=lambda r: -r.net_pnl)[:15]

    print("\n=== HIGH WIN-RATE + PROFITABLE (WR>=50%, trades>=8) ===")
    for r in high_wr[:12]:
        print(f"{r.name:<45} T={r.trades:>3} WR={r.win_rate:>5.1f}% Ret={r.total_return_pct:>6.2f}% PF={r.profit_factor:.2f} P&L={r.net_pnl:,.0f}")

    print("\n=== BEST PROFITABLE (trades>=8) ===")
    for r in profitable[:12]:
        print(f"{r.name:<45} T={r.trades:>3} WR={r.win_rate:>5.1f}% Ret={r.total_return_pct:>6.2f}% PF={r.profit_factor:.2f}")

    print("\n=== TOP BY P&L (trades>=5) ===")
    for r in best_return:
        print(f"{r.name:<45} T={r.trades:>3} WR={r.win_rate:>5.1f}% Ret={r.total_return_pct:>6.2f}% PF={r.profit_factor:.2f} P&L={r.net_pnl:,.0f}")

    payload = {
        "high_win_rate": [r.__dict__ for r in high_wr[:15]],
        "profitable": [r.__dict__ for r in profitable[:15]],
        "best_pnl": [r.__dict__ for r in best_return],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULT_PATH}")


if __name__ == "__main__":
    main()