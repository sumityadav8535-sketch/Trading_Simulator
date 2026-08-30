"""
Gap Open: replace full-fill TP and compare 9:15 vs 9:30 entry.

Same universe as the live pack (Nifty 200, gap-down ≥2% ≤15%, RSI 45–65).

    python scripts/_gap_tp_entry_search.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_n200_search import nifty200_symbols  # noqa: E402
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts.intraday_gap_hunt import (  # noqa: E402
    STORY_SIZE,
    attach_rsi,
    build_sim_book,
    collect_events,
    months_from_trades,
    prepare,
    simulate_open,
)
from trading.services.intraday_gap import (  # noqa: E402
    GAP_MAX,
    GAP_MIN,
    RSI_HI,
    RSI_LO,
    SL_ATR,
    compute_long_target,
    passes_gap_down,
    passes_rsi_band,
    tp_kind_label,
)

OUT = ROOT / "data" / "intraday_gap_tp_search.json"
ENTRY_930 = dtime(9, 30)
TP_KINDS = [
    "fill",
    "half",
    "qtr",
    "fill75",
    "r0.8",
    "r1",
    "r1.2",
    "r1.5",
    "r2",
    "atr0.8",
    "atr1",
    "atr1.2",
    "atr1.5",
    "pct0.8",
    "pct1.0",
    "pct1.2",
    "pct1.5",
    "eod",
]


def _time_of(ts) -> dtime:
    t = ts.tz_convert("Asia/Kolkata").time() if getattr(ts, "tzinfo", None) else ts.time()
    return t.replace(microsecond=0)


def session_lookup(stocks: dict[str, pd.DataFrame]) -> dict[tuple, pd.DataFrame]:
    out = {}
    for sym, df in stocks.items():
        for sess, g in df.groupby("session", sort=False):
            out[(sym, sess)] = g
    return out


def enrich_events(events: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> list[dict]:
    days = session_lookup(stocks)
    rows = []
    for r in events.itertuples(index=False):
        if not passes_gap_down(r.gap) or not passes_rsi_band(getattr(r, "rsi14", None)):
            continue
        day = days.get((r.symbol, r.session))
        if day is None or day.empty:
            continue
        o = float(r.open)
        pdc = float(r.pdc)
        atr = float(r.atr) if float(r.atr) > 0 else o * 0.008
        stop915 = o - SL_ATR * atr
        row930 = None
        ts930 = None
        low_until_930 = None
        close915 = float(r.close1) if hasattr(r, "close1") else None
        for ts, bar in day.iterrows():
            t = _time_of(ts)
            if t < ENTRY_930:
                lo = float(bar["low"])
                low_until_930 = lo if low_until_930 is None else min(low_until_930, lo)
                if t.hour == 9 and t.minute == 15:
                    close915 = float(bar["close"])
            elif t == ENTRY_930:
                ts930 = ts
                row930 = bar
                break
        open930 = float(row930["open"]) if row930 is not None else None
        rec = {
            "symbol": r.symbol,
            "session": r.session,
            "ts915": r.ts,
            "open915": o,
            "close915": close915,
            "pdc": pdc,
            "gap": float(r.gap),
            "atr": atr,
            "rsi": float(r.rsi14) if r.rsi14 is not None else None,
            "stop915": stop915,
            "ts930": ts930,
            "open930": open930,
            "alive930": bool(
                ts930 is not None
                and (low_until_930 is None or low_until_930 > stop915)
            ),
            "bounce930": bool(
                open930 is not None and close915 is not None and open930 >= close915
            ),
        }
        rows.append(rec)
    return rows


def make_entry_signals(rows: list[dict], entry_mode: str, tp_kind: str) -> pd.DataFrame:
    out = []
    for r in rows:
        if entry_mode == "915":
            ts, entry = r["ts915"], r["open915"]
            stop = r["stop915"]
        else:
            if r["ts930"] is None or r["open930"] is None:
                continue
            if "alive" in entry_mode and not r["alive930"]:
                continue
            if "bounce" in entry_mode and not r["bounce930"]:
                continue
            ts, entry = r["ts930"], r["open930"]
            stop = entry - SL_ATR * r["atr"]
        if entry < 60:
            continue
        tgt = compute_long_target(entry, r["pdc"], r["atr"], stop, tp_kind)
        if not (tgt > entry > stop):
            continue
        rec = {
            "ts": ts,
            "symbol": r["symbol"],
            "side": "long",
            "entry": entry,
            "stop": stop,
            "target": tgt,
            "score": abs(r["gap"]),
            "gap": r["gap"],
            "pdc": r["pdc"],
            "rsi": r["rsi"],
        }
        out.append(rec)
    if not out:
        return pd.DataFrame(columns=["ts", "symbol", "side", "entry", "stop", "target", "score", "gap", "pdc", "rsi"])
    return pd.DataFrame(out)


def pack_result(res, tdf, entry_mode, tp_kind) -> dict:
    reasons = {}
    if tdf is not None and not tdf.empty:
        reasons = {str(k): int(v) for k, v in tdf["reason"].value_counts().to_dict().items()}
        first_bar_sl = int(
            (
                (tdf["reason"] == "sl")
                & (
                    pd.to_datetime(tdf["exit_ts"]) - pd.to_datetime(tdf["entry_ts"])
                    <= pd.Timedelta(minutes=5)
                )
            ).sum()
        )
    else:
        first_bar_sl = 0
    return {
        "entry": entry_mode,
        "tp": tp_kind,
        "tp_label": tp_kind_label(tp_kind),
        "trades": res.trades,
        "win_rate": res.win_rate,
        "profit_factor": res.profit_factor,
        "net_pnl": res.net_pnl,
        "total_return_pct": res.total_return_pct,
        "max_dd_pct": res.max_dd_pct,
        "median_daily_pnl": res.median_daily_pnl,
        "avg_daily_pnl": res.avg_daily_pnl,
        "worst_day": res.worst_day,
        "oos_pnl": res.oos_pnl,
        "oos_wr": res.oos_wr,
        "reasons": reasons,
        "first_bar_sl": first_bar_sl,
        "months": months_from_trades(tdf),
    }


def score_row(row: dict) -> tuple:
    """Prefer typical-day quality, then OOS, then total return."""
    med = float(row["median_daily_pnl"] or 0)
    return (
        1 if row["oos_pnl"] > 0 else 0,
        1 if med > 0 else 0,
        float(row["win_rate"] or 0),
        float(row["oos_pnl"] or 0),
        float(row["total_return_pct"] or 0),
        -float(row["max_dd_pct"] or 99),
    )


def main():
    symbols = nifty200_symbols()
    print(f"TP × entry search | {len(symbols)} names", flush=True)
    cache = load_frames(symbols)
    stocks = prepare(cache)
    book = build_sim_book(stocks)
    events = attach_rsi(collect_events(stocks))
    rows = enrich_events(events, stocks)
    n915 = len(rows)
    n930 = sum(1 for r in rows if r["open930"] is not None)
    n_alive = sum(1 for r in rows if r["alive930"])
    print(f"qualified 9:15 events {n915}  with 9:30 bar {n930}  still alive at 9:30 {n_alive}", flush=True)

    book_rows = []
    modes = ["915", "930", "930_alive"]
    total = len(modes) * len(TP_KINDS)
    n = 0
    for mode in modes:
        for kind in TP_KINDS:
            n += 1
            sigs = make_entry_signals(rows, mode, kind)
            label = f"{mode} {kind}"
            res, tdf = simulate_open(stocks, sigs, label, book=book, **STORY_SIZE)
            packed = pack_result(res, tdf, mode, kind)
            book_rows.append(packed)
            print(
                f"{n:2d}/{total}  {mode:10s} {kind:7s}  {packed['total_return_pct']:7.1f}%  "
                f"WR {packed['win_rate']:5.1f}  PF {packed['profit_factor']:5.2f}  "
                f"n={packed['trades']:3d}  med {packed['median_daily_pnl']:8.0f}  "
                f"OOS {packed['oos_pnl']:8.0f}  DD {packed['max_dd_pct']:4.1f}  "
                f"{packed['reasons']}",
                flush=True,
            )

    ranked_915 = sorted((r for r in book_rows if r["entry"] == "915"), key=score_row, reverse=True)
    ranked_all = sorted(book_rows, key=score_row, reverse=True)
    baseline = next((r for r in book_rows if r["entry"] == "915" and r["tp"] == "fill"), None)
    best_915 = ranked_915[0] if ranked_915 else None
    best_930 = next((r for r in ranked_all if r["entry"].startswith("930")), None)

    payload = {
        "universe": "nifty200",
        "filter": f"gap-down {GAP_MIN:.0%}-{GAP_MAX:.0%} RSI {RSI_LO:.0f}-{RSI_HI:.0f} sl{SL_ATR}",
        "size": STORY_SIZE,
        "events_915": n915,
        "events_930": n930,
        "events_930_alive": n_alive,
        "baseline": baseline,
        "best_915": best_915,
        "best_930": best_930,
        "rows": book_rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT)
    print("\n===== 9:15 ranking (top 8) =====")
    for r in ranked_915[:8]:
        print(
            f"  {r['tp']:7s}  {r['total_return_pct']:6.1f}% WR {r['win_rate']:5.1f}  "
            f"PF {r['profit_factor']:5.2f} med {r['median_daily_pnl']:8.0f}  "
            f"OOS {r['oos_pnl']:8.0f}  {r['tp_label']}"
        )
    print("\n===== 9:30 vs 9:15 (same TP) =====")
    for kind in TP_KINDS:
        a = next(r for r in book_rows if r["entry"] == "915" and r["tp"] == kind)
        b = next(r for r in book_rows if r["entry"] == "930" and r["tp"] == kind)
        c = next(r for r in book_rows if r["entry"] == "930_alive" and r["tp"] == kind)
        print(
            f"  {kind:7s}  9:15 {a['total_return_pct']:6.1f}%/{a['win_rate']:4.1f}WR  "
            f"9:30 {b['total_return_pct']:6.1f}%/{b['win_rate']:4.1f}WR  "
            f"alive {c['total_return_pct']:6.1f}%/{c['win_rate']:4.1f}WR"
        )
    print("\nRecommend 9:15 TP:", best_915["tp"] if best_915 else None, best_915)
    print("Best 9:30:", best_930["tp"] if best_930 else None, best_930.get("entry") if best_930 else None)


if __name__ == "__main__":
    main()
