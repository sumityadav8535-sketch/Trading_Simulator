"""
Aggressive intraday search targeting higher daily P&L.
Tests elevated risk, relaxed filters, more positions, multi-entry per day.
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

from scripts.intraday_strategy_refine import (  # noqa: E402
    aligned_index_bullish,
    run_backtest_fast,
)
from scripts.intraday_strategy_search import (  # noqa: E402
    CAPITAL,
    DATA_DIR,
    StrategyResult,
    add_intraday_indicators,
    build_timeline,
    prepare_universe,
    sig_combo_elite,
    sig_ema_momentum,
    sig_vwap_pullback,
)

OUT = ROOT / "data" / "intraday_aggressive_results.json"
TARGET_DAILY = 1000.0


def run_backtest_aggressive(
    stocks, timeline, index_bullish, signal_fn, name,
    risk_pct=5.0, max_positions=10, market_filter=False,
    trades_per_day=2, target_r=1.25,
) -> StrategyResult:
    """Variant allowing multiple entries per symbol per session."""
    from scripts.intraday_strategy_search import (
        Trade, _apply_slippage, _can_enter, _in_session, position_qty, FORCE_EXIT,
    )

    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    trades: list[Trade] = []
    open_positions: list[dict] = []
    pending: list[dict] = []
    daily_trade_count: dict[tuple, int] = {}
    daily_pnl: dict = {}

    def close_pos(pos, ts, exit_raw, reason):
        nonlocal equity, peak, max_dd
        exit_p = _apply_slippage(exit_raw, "sell")
        pnl = (exit_p - pos["entry"]) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        trades.append(Trade(pos["sym"], pos["entry_ts"], ts, pos["entry"], exit_p, pos["qty"], pnl, reason, name))
        d = ts.date()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl

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
        key = (sym, sess)
        if daily_trade_count.get(key, 0) >= trades_per_day:
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
        daily_trade_count[key] = daily_trade_count.get(key, 0) + 1

    for pos in open_positions:
        df = stocks[pos["sym"]]
        close_pos(pos, df.index[-1], float(df.iloc[-1]["close"]), "final")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(trades) * 100 if trades else 0.0

    active_days = len([d for d, p in daily_pnl.items() if p != 0]) or 1
    avg_daily = sum(daily_pnl.values()) / active_days
    days_above_1k = sum(1 for p in daily_pnl.values() if p >= TARGET_DAILY)
    max_daily = max(daily_pnl.values()) if daily_pnl else 0.0

    r = StrategyResult(
        name=name, trades=len(trades), win_rate=round(wr, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        net_pnl=round(equity - CAPITAL, 2),
        max_drawdown_pct=round(max_dd, 2), avg_r=round(avg_daily, 2),
        final_equity=round(equity, 2),
    )
    r.params = {
        "avg_daily_pnl": round(avg_daily, 2),
        "days_above_1000": days_above_1k,
        "max_daily_pnl": round(max_daily, 2),
        "trading_days": active_days,
    }
    return r


# ── Aggressive signal variants ─────────────────────────────────────────────

def sig_combo_relaxed(row, prev, hist, orb, adx_min=16, target_r=1.25):
    if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
        return None
    if row["ema_9"] <= row["ema_21"]:
        return None
    if not (38 <= row["rsi_14"] <= 62):
        return None
    if row["adx_14"] < adx_min:
        return None
    if row["volume"] < row["vol_sma_20"] * 0.9:
        return None
    stop = min(float(row.get("low_3", row["low"])), row["ema_21"])
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "combo_rel"}


def sig_momentum_burst(row, prev, hist, orb, target_r=1.5):
    if pd.isna(row.get("vwap")) or row["close"] <= row["vwap"]:
        return None
    if not (row["ema_9"] > row["ema_21"]):
        return None
    if row["rsi_14"] < 50 or row["rsi_14"] > 72:
        return None
    if row["volume"] < row["vol_sma_20"] * 1.3:
        return None
    if not row.get("strong_close", False):
        return None
    stop = row["ema_9"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "mom_burst"}


def sig_vwap_reclaim(row, prev, hist, orb, target_r=1.0):
    if pd.isna(row.get("vwap")):
        return None
    if prev is None:
        return None
    if not (prev["close"] < prev["vwap"] and row["close"] > row["vwap"]):
        return None
    if row["ema_9"] < row["ema_21"]:
        return None
    stop = row["low"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "vwap_rec"}


def sig_power_hour(row, prev, hist, orb, target_r=1.25):
    t = row.name.time() if hasattr(row, "name") else None
    if t is None:
        return None
    if not (pd.Timestamp("14:00").time() <= t <= pd.Timestamp("15:00").time()):
        return None
    if row["close"] <= row["vwap"]:
        return None
    if row["ema_9"] <= row["ema_21"]:
        return None
    if row["rsi_14"] < 52:
        return None
    if row["volume"] < row["vol_sma_20"]:
        return None
    stop = min(row["low"], row["ema_21"])
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"stop": stop, "target": row["close"] + risk * target_r, "tag": "power"}


def main():
    cache = {}
    for p in DATA_DIR.glob("*.pkl"):
        sym = p.stem
        cache[sym if not sym.startswith("_") else "_INDEX"] = pd.read_pickle(p)
    stocks, index_df = prepare_universe(cache)
    timeline = build_timeline(stocks)
    index_bullish = aligned_index_bullish(index_df, [t[0] for t in timeline])
    trading_days = len({t[0].date() for t in timeline})

    print(f"Aggressive search | Rs {CAPITAL:,.0f} | target Rs {TARGET_DAILY:,.0f}/day")
    print(f"Data: {trading_days} sessions | 100 Nifty stocks\n")

    strategies = []

    for tr in (1.0, 1.25, 1.5, 2.0):
        for adx in (16, 18, 20):
            strategies.append((
                f"ComboRel {tr}R adx{adx}",
                lambda row, prev, hist, orb, r=tr, a=adx: sig_combo_relaxed(row, prev, hist, orb, a, r),
            ))

    for tr in (1.25, 1.5, 2.0):
        strategies.append((f"MomBurst {tr}R", lambda row, prev, hist, orb, r=tr: sig_momentum_burst(row, prev, hist, orb, r)))
        strategies.append((f"VWAP-Reclaim {tr}R", lambda row, prev, hist, orb, r=tr: sig_vwap_reclaim(row, prev, hist, orb, r)))
        strategies.append((f"PowerHour {tr}R", lambda row, prev, hist, orb, r=tr: sig_power_hour(row, prev, hist, orb, r)))

    for tr in (1.25, 1.5):
        strategies.append((
            f"ComboElite {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_combo_elite(row, prev, hist, r),
        ))
        strategies.append((
            f"VWAP-PB {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_vwap_pullback(row, hist, r, 38, 65),
        ))
        strategies.append((
            f"EMA-Mom {tr}R",
            lambda row, prev, hist, orb, r=tr: sig_ema_momentum(row, hist, r),
        ))

    configs = [
        # risk%, max_pos, trades_per_day, multi_entry
        (3.0, 8, 1, False),
        (5.0, 10, 1, False),
        (5.0, 10, 2, True),
        (7.0, 12, 2, True),
        (10.0, 15, 3, True),
    ]

    results = []
    total = len(strategies) * len(configs)
    n = 0
    for sname, fn in strategies:
        for risk, pos, tpd, multi in configs:
            n += 1
            label = f"{sname} r{risk}% p{pos}" + (f" x{tpd}/day" if multi else "")
            print(f"[{n}/{total}] {label}", flush=True)

            if multi:
                r = run_backtest_aggressive(
                    stocks, timeline, index_bullish, fn, label,
                    risk_pct=risk, max_positions=pos, trades_per_day=tpd,
                )
            else:
                r = run_backtest_fast(
                    stocks, timeline, index_bullish, fn, label,
                    risk_pct=risk, max_positions=pos,
                )
                # compute daily stats from trades not available in fast - approximate
                r.params = {"avg_daily_pnl": round(r.net_pnl / trading_days, 2), "trading_days": trading_days}

            r.params.update({"risk_pct": risk, "max_positions": pos, "trades_per_day": tpd})
            results.append(r)

    # Rank by avg daily P&L
    for r in results:
        if "avg_daily_pnl" not in r.params:
            r.params["avg_daily_pnl"] = round(r.net_pnl / trading_days, 2)

    by_daily = sorted(results, key=lambda x: -x.params.get("avg_daily_pnl", 0))
    profitable = [r for r in results if r.net_pnl > 0]
    hit_1k_days = [r for r in results if r.params.get("days_above_1000", 0) > 0]

    print("\n" + "=" * 110)
    print(f"TOP 20 BY AVERAGE DAILY P&L (target Rs {TARGET_DAILY:,.0f}/day)")
    print("=" * 110)
    print(f"{'Strategy':<50} {'Trades':>6} {'WR%':>6} {'TotP&L':>9} {'Avg/Day':>9} {'MaxDay':>8} {'Days>1k':>7} {'DD%':>6}")
    for r in by_daily[:20]:
        print(
            f"{r.name:<50} {r.trades:>6} {r.win_rate:>5.1f}% {r.net_pnl:>9,.0f} "
            f"{r.params.get('avg_daily_pnl', 0):>9,.0f} {r.params.get('max_daily_pnl', 0):>8,.0f} "
            f"{r.params.get('days_above_1000', 0):>7} {r.max_drawdown_pct:>5.1f}%"
        )

    best = by_daily[0] if by_daily else None
    print("\n" + "=" * 110)
    if best and best.params.get("avg_daily_pnl", 0) >= TARGET_DAILY:
        print(f"TARGET MET: {best.name}")
    else:
        gap = TARGET_DAILY - (best.params.get("avg_daily_pnl", 0) if best else 0)
        print(f"TARGET NOT MET on Rs {CAPITAL:,.0f} capital.")
        if best:
            print(f"  Best avg/day : Rs {best.params.get('avg_daily_pnl', 0):,.0f} ({best.name})")
            print(f"  Gap          : Rs {gap:,.0f}/day still needed")
            capital_needed = CAPITAL * (TARGET_DAILY / max(best.params.get("avg_daily_pnl", 1), 1))
            print(f"  Rough capital needed at same edge: Rs {capital_needed:,.0f}")

    payload = {
        "target_daily": TARGET_DAILY,
        "capital": CAPITAL,
        "trading_days": trading_days,
        "target_met": bool(best and best.params.get("avg_daily_pnl", 0) >= TARGET_DAILY),
        "top_20": [{**r.__dict__, "params": r.params} for r in by_daily[:20]],
        "any_day_above_1000": [{**r.__dict__, "params": r.params} for r in hit_1k_days[:10]],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()