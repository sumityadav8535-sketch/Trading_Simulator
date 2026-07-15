"""
₹5L capital intraday search — institutional / pro trader strategy styles.
Target: > ₹1,000 average daily P&L across Nifty 100, 5m bars.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_strategy_search import (  # noqa: E402
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
)
from scripts.intraday_strategy_refine import aligned_index_bullish  # noqa: E402

CAPITAL = 500_000.0
TARGET_DAILY = 1_000.0
SLIPPAGE = 0.0004
FORCE_EXIT = dt_time(15, 15)
ORB_15 = dt_time(9, 30)
ORB_30 = dt_time(9, 45)
OUT = ROOT / "data" / "intraday_5L_pro_results.json"


@dataclass
class ProResult(StrategyResult):
    daily_stats: dict = field(default_factory=dict)


def position_qty_pro(equity, risk_pct, entry, stop, max_deploy_pct=0.3):
    if any(pd.isna(x) for x in (equity, entry, stop)):
        return 0
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    risk_amount = equity * (risk_pct / 100)
    qty = int(risk_amount / risk_per_share)
    cap_qty = int((equity * max_deploy_pct) / entry)
    return max(min(qty, cap_qty), 0)


def run_pro_backtest(
    stocks: dict[str, pd.DataFrame],
    timeline: list,
    index_bullish: dict,
    signal_fn,
    name: str,
    risk_pct: float = 2.0,
    max_positions: int = 8,
    market_filter: bool = False,
    target_r: float = 1.5,
    trades_per_symbol_day: int = 1,
) -> ProResult:
    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    trades: list[Trade] = []
    open_pos: list[dict] = []
    pending: list[dict] = []
    sym_day_count: dict[tuple, int] = {}
    daily_pnl: dict = {}

    def slip_buy(p):
        return p * (1 + SLIPPAGE)

    def slip_sell(p):
        return p * (1 - SLIPPAGE)

    def close(pos, ts, raw, reason):
        nonlocal equity, peak, max_dd
        exit_p = slip_sell(raw)
        pnl = (exit_p - pos["entry"]) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        trades.append(Trade(pos["sym"], pos["entry_ts"], ts, pos["entry"], exit_p, pos["qty"], pnl, reason, name))
        daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0.0) + pnl

    for ts, sym, sess, row, prev_row, orb, next_ts in timeline:
        still = []
        for pe in pending:
            if pe["entry_ts"] != ts or pe["sym"] != sym:
                still.append(pe)
                continue
            if len(open_pos) >= max_positions:
                still.append(pe)
                continue
            entry = slip_buy(float(row["open"]))
            if entry <= pe["stop"]:
                continue
            qty = position_qty_pro(equity, risk_pct, entry, pe["stop"])
            if qty <= 0:
                continue
            open_pos.append({
                "sym": sym, "entry_ts": ts, "entry": entry,
                "stop": pe["stop"], "target": pe["target"], "qty": qty,
            })
        pending = still

        rem = []
        for pos in open_pos:
            if pos["sym"] != sym:
                rem.append(pos)
                continue
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
            if lo <= pos["stop"]:
                close(pos, ts, pos["stop"], "sl")
            elif hi >= pos["target"]:
                close(pos, ts, pos["target"], "target")
            elif ts.time() >= FORCE_EXIT:
                close(pos, ts, cl, "eod")
            else:
                rem.append(pos)
        open_pos = rem

        if not _can_enter(ts) or len(open_pos) + len(pending) >= max_positions:
            continue
        key = (sym, sess)
        if sym_day_count.get(key, 0) >= trades_per_symbol_day:
            continue
        if market_filter and not index_bullish.get(ts, True):
            continue

        sig = signal_fn(row, prev_row, orb, ts, sym, sess)
        if sig is None or next_ts is None:
            continue
        stop, tgt = float(sig["stop"]), float(sig["target"])
        if np.isnan(stop) or np.isnan(tgt) or tgt <= row["close"]:
            continue
        pending.append({"sym": sym, "entry_ts": next_ts, "stop": stop, "target": tgt})
        sym_day_count[key] = sym_day_count.get(key, 0) + 1

    for pos in open_pos:
        df = stocks[pos["sym"]]
        close(pos, df.index[-1], float(df.iloc[-1]["close"]), "final")

    wins = [t for t in trades if t.pnl > 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf = gp / gl if gl else (999 if gp > 0 else 0)
    wr = len(wins) / len(trades) * 100 if trades else 0

    all_days = len({t[0].date() for t in timeline})
    avg_daily = (equity - CAPITAL) / all_days
    days_1k = sum(1 for p in daily_pnl.values() if p >= TARGET_DAILY)

    return ProResult(
        name=name, trades=len(trades), win_rate=round(wr, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        net_pnl=round(equity - CAPITAL, 2),
        max_drawdown_pct=round(max_dd, 2), avg_r=round(avg_daily, 2),
        final_equity=round(equity, 2),
        daily_stats={
            "avg_daily_pnl": round(avg_daily, 2),
            "days_above_1000": days_1k,
            "max_daily_pnl": round(max(daily_pnl.values()) if daily_pnl else 0, 2),
            "trading_days": all_days,
        },
    )


# ── Precompute per-stock session context ───────────────────────────────────

def enrich_stocks(stocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in stocks.items():
        d = add_intraday_indicators(df)
        d["prev_rsi"] = d["rsi_14"].shift(1)
        d["prev_close"] = d["close"].shift(1)
        d["vol_z"] = (d["volume"] - d["vol_sma_20"]) / d["vol_sma_20"].replace(0, np.nan)
        d["day_open"] = d.groupby("session_date")["open"].transform("first")
        d["day_high"] = d.groupby("session_date")["high"].cummax()
        d["day_low"] = d.groupby("session_date")["low"].cummin()
        d["pct_from_open"] = (d["close"] - d["day_open"]) / d["day_open"] * 100
        d["range_pos"] = (d["close"] - d["day_low"]) / (d["day_high"] - d["day_low"]).replace(0, np.nan)
        out[sym] = d
    return out


# ── Pro / institutional strategy signals ───────────────────────────────────

def make_orb(orb_end: dt_time, vol_mult: float, target_r: float, min_rng_pct: float):
    def fn(row, prev, orb, ts, sym, sess):
        if ts.time() <= orb_end or not orb.get("done"):
            return None
        hi, lo = orb.get("high"), orb.get("low")
        if hi is None or lo is None:
            return None
        rng = hi - lo
        if rng <= 0 or rng / row["close"] < min_rng_pct:
            return None
        if row["close"] <= hi:
            return None
        if row["volume"] < row["vol_sma_20"] * vol_mult:
            return None
        if row["close"] <= row["vwap"]:
            return None
        stop = lo
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_vwap_institutional(target_r: float, z_entry: float = -1.5):
    """Institutional VWAP fade: stretch below VWAP then reclaim (liquidity grab)."""
    def fn(row, prev, orb, ts, sym, sess):
        if pd.isna(row.get("vwap")):
            return None
        dist = (row["close"] - row["vwap"]) / row["vwap"] * 100
        if prev is None:
            return None
        prev_dist = (prev["close"] - prev["vwap"]) / prev["vwap"] * 100
        # Was stretched below VWAP, now reclaimed
        if not (prev_dist < z_entry * 0.15 and dist > -0.05):
            return None
        if row["ema_9"] < row["ema_21"]:
            return None
        if row["rsi_14"] < 40 or row["rsi_14"] > 58:
            return None
        stop = row["day_low"] if not pd.isna(row.get("day_low")) else row["low"]
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_relative_strength(rs_min: float, target_r: float):
    """Pro: long leaders outperforming open when index bullish."""
    def fn(row, prev, orb, ts, sym, sess):
        if row["pct_from_open"] < rs_min:
            return None
        if row["close"] <= row["vwap"]:
            return None
        if row["ema_9"] <= row["ema_21"]:
            return None
        if row["volume"] < row["vol_sma_20"] * 1.2:
            return None
        if row["rsi_14"] > 72:
            return None
        stop = max(row["vwap"], row["ema_21"])
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_first_hour_momentum(target_r: float):
    """Trade strong first-hour trend continuation (institutions build positions AM)."""
    def fn(row, prev, orb, ts, sym, sess):
        t = ts.time()
        if not (dt_time(10, 0) <= t <= dt_time(11, 30)):
            return None
        if row["pct_from_open"] < 0.4:
            return None
        if row["close"] <= row["vwap"] or row["ema_9"] <= row["ema_21"]:
            return None
        if row["adx_14"] < 20:
            return None
        if not row.get("strong_close", False):
            return None
        stop = row["ema_21"]
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_volume_breakout(target_r: float, vol_z_min: float = 2.0):
    """Big volume bar breakout — institutional block print proxy."""
    def fn(row, prev, orb, ts, sym, sess):
        if pd.isna(row.get("vol_z")) or row["vol_z"] < vol_z_min:
            return None
        if row["close"] <= row["open"]:
            return None
        if row["close"] <= row["vwap"]:
            return None
        if row["ema_9"] <= row["ema_21"]:
            return None
        body = row["close"] - row["open"]
        stop = row["open"] - body * 0.3
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_pdh_breakout(target_r: float):
    """Previous day high breakout — classic momentum funds approach."""
    # Use rolling max of prior session highs
    def fn(row, prev, orb, ts, sym, sess):
        # proxy: break above day_high established before this bar in prior sessions
        # intraday: break session high after 10am with volume
        t = ts.time()
        if t < dt_time(10, 0):
            return None
        if row["close"] < row["day_high"] * 0.999:
            return None
        if row["volume"] < row["vol_sma_20"] * 1.5:
            return None
        if row["close"] <= row["vwap"]:
            return None
        stop = row["day_low"]
        risk = row["close"] - stop
        if risk <= 0 or risk / row["close"] > 0.025:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_liquidity_sweep(target_r: float):
    """Stop hunt below day low then reclaim — SMC / institutional style."""
    def fn(row, prev, orb, ts, sym, sess):
        if prev is None:
            return None
        if not (prev["low"] <= prev["day_low"] * 1.001 and row["close"] > prev["day_low"]):
            return None
        if row["close"] <= row["vwap"]:
            return None
        if row["rsi_14"] < 38:
            return None
        stop = row["low"]
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_afternoon_ramp(target_r: float):
    """2:00–3:00 PM institutional ramp with index alignment."""
    def fn(row, prev, orb, ts, sym, sess):
        t = ts.time()
        if not (dt_time(14, 0) <= t <= dt_time(15, 0)):
            return None
        if row["close"] <= row["vwap"] or row["pct_from_open"] < 0.2:
            return None
        if row["ema_9"] <= row["ema_21"]:
            return None
        if row["volume"] < row["vol_sma_20"]:
            return None
        stop = row["ema_9"]
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def make_combo_pro(target_r: float):
    """Best-of-breed combo: VWAP trend + RS + volume."""
    def fn(row, prev, orb, ts, sym, sess):
        if row["close"] <= row["vwap"] or row["pct_from_open"] < 0.25:
            return None
        if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
            return None
        if row["rsi_14"] < 48 or row["rsi_14"] > 65:
            return None
        if row["adx_14"] < 18 or row["di_plus"] <= row["di_minus"]:
            return None
        if row["volume"] < row["vol_sma_20"] * 1.1:
            return None
        if not row.get("strong_close", False):
            return None
        stop = min(row.get("low_3", row["low"]), row["ema_21"])
        risk = row["close"] - stop
        if risk <= 0:
            return None
        return {"stop": stop, "target": row["close"] + risk * target_r}
    return fn


def main():
    cache = {}
    for p in DATA_DIR.glob("*.pkl"):
        sym = p.stem
        cache[sym if not sym.startswith("_") else "_INDEX"] = pd.read_pickle(p)

    stocks, index_df = prepare_universe(cache)
    stocks = enrich_stocks(stocks)
    timeline = build_timeline(stocks)
    index_bullish = aligned_index_bullish(index_df, [t[0] for t in timeline])
    all_days = len({t[0].date() for t in timeline})

    print(f"Pro trader search | Rs {CAPITAL:,.0f} | target > Rs {TARGET_DAILY:,.0f}/day")
    print(f"{len(stocks)} symbols | {all_days} sessions\n")

    strategies = [
        ("ORB-15 1.5R", make_orb(ORB_15, 1.0, 1.5, 0.0015)),
        ("ORB-15 2R", make_orb(ORB_15, 1.2, 2.0, 0.0015)),
        ("ORB-30 1.5R", make_orb(ORB_30, 1.0, 1.5, 0.002)),
        ("ORB-30 2R", make_orb(ORB_30, 1.2, 2.0, 0.002)),
        ("VWAP-Inst 1.25R", make_vwap_institutional(1.25)),
        ("VWAP-Inst 1.5R", make_vwap_institutional(1.5)),
        ("RelStrength 0.5%", make_relative_strength(0.5, 1.5)),
        ("RelStrength 0.8%", make_relative_strength(0.8, 1.5)),
        ("RelStrength 1.0%", make_relative_strength(1.0, 2.0)),
        ("1stHour-Mom 1.5R", make_first_hour_momentum(1.5)),
        ("1stHour-Mom 2R", make_first_hour_momentum(2.0)),
        ("Vol-Breakout 1.5R", make_volume_breakout(1.5, 2.0)),
        ("Vol-Breakout 2R", make_volume_breakout(2.0, 2.5)),
        ("DayHigh-Break 1.5R", make_pdh_breakout(1.5)),
        ("DayHigh-Break 2R", make_pdh_breakout(2.0)),
        ("Liquidity-Sweep 1.5R", make_liquidity_sweep(1.5)),
        ("Liquidity-Sweep 2R", make_liquidity_sweep(2.0)),
        ("PM-Ramp 1.25R", make_afternoon_ramp(1.25)),
        ("PM-Ramp 1.5R", make_afternoon_ramp(1.5)),
        ("Combo-Pro 1.25R", make_combo_pro(1.25)),
        ("Combo-Pro 1.5R", make_combo_pro(1.5)),
        ("Combo-Pro 2R", make_combo_pro(2.0)),
    ]

    configs = [
        (1.5, 6, 1, False),
        (2.0, 8, 1, False),
        (2.5, 10, 2, False),
        (3.0, 12, 2, True),
        (4.0, 15, 3, True),
    ]

    results: list[ProResult] = []
    total = len(strategies) * len(configs)
    n = 0
    for sname, fn in strategies:
        for risk, pos, tpd, mkt in configs:
            n += 1
            label = f"{sname} r{risk}% p{pos}" + (f" x{tpd}" if tpd > 1 else "") + (" +idx" if mkt else "")
            print(f"[{n}/{total}] {label}", flush=True)
            r = run_pro_backtest(
                stocks, timeline, index_bullish, fn, label,
                risk_pct=risk, max_positions=pos, market_filter=mkt,
                trades_per_symbol_day=tpd,
            )
            r.params = {"risk_pct": risk, "max_positions": pos, "trades_per_day": tpd, "market_filter": mkt}
            results.append(r)

    by_daily = sorted(results, key=lambda x: -x.daily_stats.get("avg_daily_pnl", 0))
    profitable = [r for r in results if r.net_pnl > 0]
    hit_target = [r for r in results if r.daily_stats.get("avg_daily_pnl", 0) >= TARGET_DAILY]

    print("\n" + "=" * 115)
    print(f"TOP 25 — Rs {CAPITAL:,.0f} capital (target avg > Rs {TARGET_DAILY:,.0f}/day)")
    print("=" * 115)
    print(f"{'Strategy':<52} {'Trd':>5} {'WR%':>6} {'Total':>10} {'Avg/Day':>9} {'Days>1k':>7} {'DD%':>6}")
    for r in by_daily[:25]:
        ds = r.daily_stats
        print(
            f"{r.name:<52} {r.trades:>5} {r.win_rate:>5.1f}% {r.net_pnl:>10,.0f} "
            f"{ds.get('avg_daily_pnl', 0):>9,.0f} {ds.get('days_above_1000', 0):>7} {r.max_drawdown_pct:>5.1f}%"
        )

    print("\n" + "=" * 115)
    if hit_target:
        print(f"TARGET MET — {len(hit_target)} configs averaged >= Rs {TARGET_DAILY:,.0f}/day")
        for r in hit_target[:5]:
            print(f"  * {r.name}: avg Rs {r.daily_stats['avg_daily_pnl']:,.0f}/day, total Rs {r.net_pnl:,.0f}")
    else:
        best = by_daily[0] if by_daily else None
        print(f"TARGET NOT MET on Rs {CAPITAL:,.0f}")
        if best:
            print(f"  Best avg/day : Rs {best.daily_stats.get('avg_daily_pnl', 0):,.0f} ({best.name})")
            print(f"  Best total   : Rs {best.net_pnl:,.0f} over {all_days} days ({best.total_return_pct}%)")
            print(f"  Gap          : Rs {TARGET_DAILY - best.daily_stats.get('avg_daily_pnl', 0):,.0f}/day")
            cap_need = CAPITAL * TARGET_DAILY / max(best.daily_stats.get("avg_daily_pnl", 1), 1)
            print(f"  Capital needed at same edge: Rs {cap_need:,.0f}")

    print(f"\nProfitable configs: {len(profitable)} / {len(results)}")

    OUT.write_text(json.dumps({
        "capital": CAPITAL,
        "target_daily": TARGET_DAILY,
        "trading_days": all_days,
        "target_met": len(hit_target) > 0,
        "hit_target": [{**r.__dict__, "daily_stats": r.daily_stats} for r in hit_target[:10]],
        "top_25": [{**r.__dict__, "daily_stats": r.daily_stats} for r in by_daily[:25]],
        "profitable_count": len(profitable),
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()