"""Phase 2: RSI 45-70, ranking, faster Supertrend, 5m ST. Keep bounce + gap 2-8%."""
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

from scripts._gap_more_trades import (  # noqa: E402
    BASELINE_N,
    BASELINE_RET,
    ENTRY_930,
    SIZE,
    SL_ATR,
    _time_of,
    daily_lookup,
    enrich_all,
    gap_ok,
    rsi_ok,
)
from scripts._gap_tp_entry_search import session_lookup  # noqa: E402
from scripts.intraday_15m_n200_search import nifty200_symbols  # noqa: E402
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts.intraday_gap_hunt import (  # noqa: E402
    attach_rsi,
    build_sim_book,
    collect_events,
    months_from_trades,
    prepare,
    simulate_open,
)
from trading.services.indicators import _supertrend  # noqa: E402
from trading.services.intraday_gap import compute_long_target  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_more_trades_p2.json"


def extra_daily(symbols):
    """Faster / tighter Supertrend variants, shifted 1 day."""
    out = {}
    for sym in symbols:
        d = load_price_dataframe(sym)
        if d.empty or len(d) < 30:
            continue
        x = d.copy()
        variants = {}
        for period, mult, key in (
            (5, 2.0, "st52"),
            (7, 2.0, "st72"),
            (10, 2.0, "st102"),
            (5, 3.0, "st53"),
        ):
            _st, st_dir = _supertrend(x["high"], x["low"], x["close"], period, mult)
            variants[key] = (st_dir > 0).shift(1)
        frame = pd.DataFrame(variants)
        frame.index = pd.to_datetime(x.index).date
        out[sym] = frame
    return out


def last_5m_st(stocks: dict[str, pd.DataFrame]) -> dict[tuple, bool]:
    """(symbol, session) -> prior session last-bar 5m Supertrend bull."""
    flag = {}
    for sym, df in stocks.items():
        if "st_dir" not in df.columns:
            st, st_dir = _supertrend(df["high"], df["low"], df["close"], 10, 3.0)
            work = df
            work = work.copy()
            work["st_dir"] = st_dir
        else:
            work = df
        last = work.groupby("session", sort=True).tail(1)
        prev_bull = None
        prev_sess = None
        for sess, row in last.iterrows():
            sess_d = row["session"]
            if prev_sess is not None:
                flag[(sym, sess_d)] = bool(prev_bull)
            prev_bull = float(row["st_dir"]) > 0 if pd.notna(row["st_dir"]) else False
            prev_sess = sess_d
    return flag


def make_sigs(rows, pred, score_fn, tp="pct1.0"):
    out = []
    for r in rows:
        if not pred(r):
            continue
        if r["ts930"] is None or r["open930"] is None or not r["bounce930"]:
            continue
        entry_px = r["open930"]
        stop = entry_px - SL_ATR * r["atr"]
        if entry_px < 60:
            continue
        tgt = compute_long_target(entry_px, r["pdc"], r["atr"], stop, tp)
        if not (tgt > entry_px > stop):
            continue
        out.append({
            "ts": r["ts930"], "symbol": r["symbol"], "side": "long",
            "entry": entry_px, "stop": stop, "target": tgt,
            "score": float(score_fn(r)), "gap": r["gap"], "pdc": r["pdc"], "rsi": r["rsi"],
        })
    if not out:
        return pd.DataFrame(columns=["ts", "symbol", "side", "entry", "stop", "target", "score", "gap", "pdc", "rsi"])
    return pd.DataFrame(out)


def main():
    symbols = nifty200_symbols()
    print("P2 load…", flush=True)
    cache = load_frames(symbols)
    stocks = prepare(cache)
    events = attach_rsi(collect_events(stocks))
    daily = daily_lookup(list(stocks.keys()))
    rows = enrich_all(events, stocks, daily)
    extra = extra_daily(list(stocks.keys()))
    print("5m Supertrend (prior session)…", flush=True)
    # compute 5m ST once per stock
    for sym, df in list(stocks.items()):
        st, st_dir = _supertrend(df["high"], df["low"], df["close"], 10, 3.0)
        df["st_dir"] = st_dir
    st5 = last_5m_st(stocks)
    for r in rows:
        sess = r["session"]
        extra_row = extra.get(r["symbol"])
        if extra_row is not None and sess in extra_row.index:
            er = extra_row.loc[sess]
            if isinstance(er, pd.DataFrame):
                er = er.iloc[-1]
            r["st52"] = bool(er.get("st52")) if pd.notna(er.get("st52")) else False
            r["st72"] = bool(er.get("st72")) if pd.notna(er.get("st72")) else False
            r["st102"] = bool(er.get("st102")) if pd.notna(er.get("st102")) else False
            r["st53"] = bool(er.get("st53")) if pd.notna(er.get("st53")) else False
        else:
            r["st52"] = r["st72"] = r["st102"] = r["st53"] = False
        r["st5m_prev"] = bool(st5.get((r["symbol"], sess), False))

    book = build_sim_book(stocks)
    gap_abs = lambda r: abs(r["gap"])
    gap_small = lambda r: 0.08 - abs(r["gap"])  # prefer 2% over 8%
    st_then_gap = lambda r: (10.0 if r.get("st7_bull") else 0.0) + abs(r["gap"])
    st5_then_gap = lambda r: (10.0 if r.get("st5m_prev") else 0.0) + abs(r["gap"])
    mid_gap = lambda r: -abs(abs(r["gap"]) - 0.03)

    packs = [
        ("RSI45-70 g2-8 largest", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70), gap_abs),
        ("RSI45-72 g2-8 largest", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 72), gap_abs),
        ("RSI42-70 g2-8 largest", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 42, 70), gap_abs),
        ("RSI45-70 g2-6 largest", lambda r: gap_ok(r, 0.02, 0.06) and rsi_ok(r, 45, 70), gap_abs),
        ("RSI45-70 g2-5 largest", lambda r: gap_ok(r, 0.02, 0.05) and rsi_ok(r, 45, 70), gap_abs),
        ("RSI45-65 g2-8 smallest", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 65), gap_small),
        ("RSI45-70 g2-8 smallest", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70), gap_small),
        ("RSI45-70 g2-8 midgap", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70), mid_gap),
        ("RSI45-70 + ST7 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70) and r.get("st7_bull"), gap_abs),
        ("RSI45-70 or ST7 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and (rsi_ok(r, 45, 70) or r.get("st7_bull")), gap_abs),
        ("ST52 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r.get("st52"), gap_abs),
        ("ST72 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r.get("st72"), gap_abs),
        ("ST102 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r.get("st102"), gap_abs),
        ("ST53 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r.get("st53"), gap_abs),
        ("5m ST prev g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r.get("st5m_prev"), gap_abs),
        ("RSI45-70 + 5mST g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70) and r.get("st5m_prev"), gap_abs),
        ("RSI45-70 or 5mST g2-8", lambda r: gap_ok(r, 0.02, 0.08) and (rsi_ok(r, 45, 70) or r.get("st5m_prev")), gap_abs),
        ("RSI45-70 rank ST7 then gap", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70), st_then_gap),
        ("RSI45-70 rank 5mST then gap", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70), st5_then_gap),
        ("SMA20 + RSI45-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70) and r.get("above_sma20"), gap_abs),
        ("RSI45-70 + above SMA50 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70) and r.get("above_sma50"), gap_abs),
        ("BASE RSI45-65", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 65), gap_abs),
    ]

    sizes = [
        ("k3", SIZE),
        ("k4", dict(risk_pct=8.0, max_pos=4, top_k=4, max_deploy=0.50)),
    ]
    results = []
    print(f"{'pack':<42} {'n':>4} {'WR':>6} {'ret%':>7} {'PF':>5} {'DD':>5} keep")
    for label, pred, scorer in packs:
        for sname, size in sizes:
            sigs = make_sigs(rows, pred, scorer)
            name = f"{label} {sname}"
            res, tdf = simulate_open(stocks, sigs, name, book=book, **size)
            keep = res.trades > BASELINE_N and res.total_return_pct >= BASELINE_RET - 0.01
            print(
                f"{name:<42} {res.trades:4d} {res.win_rate:6.1f} {res.total_return_pct:7.2f} "
                f"{res.profit_factor:5.2f} {res.max_dd_pct:5.1f} {'YES' if keep else ''}",
                flush=True,
            )
            results.append({
                "label": name,
                "trades": res.trades,
                "win_rate": res.win_rate,
                "total_return_pct": res.total_return_pct,
                "profit_factor": res.profit_factor,
                "max_dd_pct": res.max_dd_pct,
                "oos_pnl": res.oos_pnl,
                "keep": keep,
                "months": months_from_trades(tdf),
                "size": size,
            })
    winners = sorted(
        [r for r in results if r["keep"]],
        key=lambda r: (r["total_return_pct"], r["trades"]),
        reverse=True,
    )
    print("\nWINNERS")
    for r in winners:
        print(f"  {r['label']}  n={r['trades']} ret={r['total_return_pct']}% WR={r['win_rate']}% PF={r['profit_factor']}")
    OUT.write_text(json.dumps({"winners": winners, "all": results}, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
