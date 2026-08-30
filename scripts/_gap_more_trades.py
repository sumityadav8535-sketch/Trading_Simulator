"""Hunt Gap Open filters that add trades without cutting return.

Baseline live pack: gap 2–8%, RSI 45–65, 9:30 bounce, 1% target → 21 trades, +26%.

    python scripts/_gap_more_trades.py
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
    attach_rsi,
    build_sim_book,
    collect_events,
    months_from_trades,
    prepare,
    simulate_open,
)
from scripts._gap_tp_entry_search import session_lookup  # noqa: E402
from trading.services.indicators import _adx, _rsi, _supertrend  # noqa: E402
from trading.services.intraday_gap import compute_long_target  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_more_trades.json"
ENTRY_930 = dtime(9, 30)
SL_ATR = 0.6
SIZE = dict(risk_pct=8.0, max_pos=4, top_k=3, max_deploy=0.50)
BASELINE_N = 21
BASELINE_RET = 26.07


def _time_of(ts) -> dtime:
    t = ts.tz_convert("Asia/Kolkata").time() if getattr(ts, "tzinfo", None) else ts.time()
    return t.replace(microsecond=0)


def daily_lookup(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for i, sym in enumerate(symbols, 1):
        d = load_price_dataframe(sym)
        if d.empty or len(d) < 30:
            continue
        x = d.copy()
        x["sma20"] = x["close"].rolling(20).mean()
        x["sma50"] = x["close"].rolling(50).mean()
        x["sma150"] = x["close"].rolling(150).mean()
        x["rsi14"] = _rsi(x["close"], 14)
        adx, di_p, di_m = _adx(x["high"], x["low"], x["close"], 14)
        x["adx"] = adx
        x["di_plus"] = di_p
        x["di_minus"] = di_m
        st, st_dir = _supertrend(x["high"], x["low"], x["close"], 10, 3.0)
        x["st"] = st
        x["st_dir"] = st_dir
        st7, st7_dir = _supertrend(x["high"], x["low"], x["close"], 7, 3.0)
        x["st7_dir"] = st7_dir
        # Yesterday only — no same-day look-ahead.
        shifted = x[[
            "rsi14", "sma20", "sma50", "sma150", "adx", "di_plus", "di_minus",
            "st", "st_dir", "st7_dir", "close",
        ]].shift(1)
        shifted.index = pd.to_datetime(shifted.index).date
        shifted["st_bull"] = (shifted["st_dir"] > 0) & (shifted["close"] > shifted["st"])
        shifted["st7_bull"] = shifted["st7_dir"] > 0
        shifted["above_sma20"] = shifted["close"] > shifted["sma20"]
        shifted["above_sma50"] = shifted["close"] > shifted["sma50"]
        shifted["above_sma150"] = shifted["close"] > shifted["sma150"]
        shifted["htf_up"] = shifted["above_sma20"] & (shifted["sma20"] > shifted["sma50"])
        out[sym] = shifted
        if i % 50 == 0:
            print(f"  daily {i}/{len(symbols)}", flush=True)
    return out


def enrich_all(events: pd.DataFrame, stocks: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame]) -> list[dict]:
    days = session_lookup(stocks)
    rows = []
    for r in events.itertuples(index=False):
        gap = float(r.gap)
        if gap > -0.01 or gap < -0.12:
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
        bar0 = day.iloc[0]
        ema_stack_5m = bool(float(bar0.get("ema9") or 0) > float(bar0.get("ema21") or 0))
        rsi5 = float(bar0["rsi"]) if pd.notna(bar0.get("rsi")) else None
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
        drow = None
        dframe = daily.get(r.symbol)
        if dframe is not None and r.session in dframe.index:
            drow = dframe.loc[r.session]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[-1]
        def _b(key, default=False):
            if drow is None:
                return default
            val = drow.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return bool(val)

        def _f(key):
            if drow is None:
                return None
            val = drow.get(key)
            try:
                x = float(val)
            except (TypeError, ValueError):
                return None
            if np.isnan(x):
                return None
            return x

        rsi = _f("rsi14")
        if rsi is None:
            rsi = float(r.rsi14) if getattr(r, "rsi14", None) is not None else None
        rec = {
            "symbol": r.symbol,
            "session": r.session,
            "ts915": r.ts,
            "open915": o,
            "close915": close915,
            "pdc": pdc,
            "gap": gap,
            "atr": atr,
            "rsi": rsi,
            "stop915": stop915,
            "ts930": ts930,
            "open930": open930,
            "alive930": bool(ts930 is not None and (low_until_930 is None or low_until_930 > stop915)),
            "bounce930": bool(open930 is not None and close915 is not None and open930 >= close915),
            "st_bull": _b("st_bull"),
            "st7_bull": _b("st7_bull"),
            "above_sma20": _b("above_sma20"),
            "above_sma50": _b("above_sma50"),
            "above_sma150": _b("above_sma150"),
            "htf_up": _b("htf_up"),
            "adx": _f("adx"),
            "di_plus": _f("di_plus"),
            "di_minus": _f("di_minus"),
            "ema_stack_5m": ema_stack_5m,
            "rsi5": rsi5,
        }
        rows.append(rec)
    return rows


def make_sigs(rows: list[dict], pred, *, tp="pct1.0", entry="930_bounce") -> pd.DataFrame:
    out = []
    for r in rows:
        if not pred(r):
            continue
        if entry.startswith("930"):
            if r["ts930"] is None or r["open930"] is None:
                continue
            if "bounce" in entry and not r["bounce930"]:
                continue
            if "alive" in entry and not r["alive930"]:
                continue
            ts, entry_px = r["ts930"], r["open930"]
            stop = entry_px - SL_ATR * r["atr"]
        else:
            ts, entry_px = r["ts915"], r["open915"]
            stop = r["stop915"]
        if entry_px < 60:
            continue
        tgt = compute_long_target(entry_px, r["pdc"], r["atr"], stop, tp)
        if not (tgt > entry_px > stop):
            continue
        out.append({
            "ts": ts, "symbol": r["symbol"], "side": "long",
            "entry": entry_px, "stop": stop, "target": tgt,
            "score": abs(r["gap"]), "gap": r["gap"], "pdc": r["pdc"], "rsi": r["rsi"],
        })
    if not out:
        return pd.DataFrame(columns=["ts", "symbol", "side", "entry", "stop", "target", "score", "gap", "pdc", "rsi"])
    return pd.DataFrame(out)


def rsi_ok(r, lo, hi):
    v = r.get("rsi")
    return v is not None and lo <= v <= hi


def gap_ok(r, lo, hi):
    g = r["gap"]
    return (-hi) <= g <= (-lo)


def main():
    symbols = nifty200_symbols()
    print(f"Gap more-trades hunt | {len(symbols)} names", flush=True)
    cache = load_frames(symbols)
    stocks = prepare(cache)
    events = collect_events(stocks)
    print(f"events {len(events)}  RSI…", flush=True)
    events = attach_rsi(events)
    print("daily Supertrend / SMA / ADX…", flush=True)
    daily = daily_lookup(list(stocks.keys()))
    rows = enrich_all(events, stocks, daily)
    print(f"enriched gap-downs 1–12%: {len(rows)}", flush=True)
    book = build_sim_book(stocks)

    packs = [
        ("BASE RSI45-65 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 65)),
        ("RSI40-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 40, 70)),
        ("RSI40-75 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 40, 75)),
        ("RSI35-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 35, 70)),
        ("RSI45-65 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and rsi_ok(r, 45, 65)),
        ("RSI40-70 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and rsi_ok(r, 40, 70)),
        ("RSI40-70 g1-8", lambda r: gap_ok(r, 0.01, 0.08) and rsi_ok(r, 40, 70)),
        ("ST bull g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["st_bull"]),
        ("ST bull g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["st_bull"]),
        ("ST bull g1-8", lambda r: gap_ok(r, 0.01, 0.08) and r["st_bull"]),
        ("ST7 bull g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["st7_bull"]),
        ("ST bull + RSI40-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["st_bull"] and rsi_ok(r, 40, 70)),
        ("ST bull + RSI40-70 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["st_bull"] and rsi_ok(r, 40, 70)),
        ("ST OR RSI45-65 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and (r["st_bull"] or rsi_ok(r, 45, 65))),
        ("ST OR RSI45-65 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and (r["st_bull"] or rsi_ok(r, 45, 65))),
        ("SMA20 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["above_sma20"]),
        ("SMA50 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["above_sma50"]),
        ("htf_up g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["htf_up"]),
        ("htf_up g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["htf_up"]),
        ("ST + SMA20 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["st_bull"] and r["above_sma20"]),
        ("ST + SMA20 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["st_bull"] and r["above_sma20"]),
        ("ST + htf_up g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["st_bull"] and r["htf_up"]),
        ("ADX>=20 + RSI40-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 40, 70) and (r.get("adx") or 0) >= 20),
        ("DI+ > DI- g2-8", lambda r: gap_ok(r, 0.02, 0.08) and (r.get("di_plus") or 0) > (r.get("di_minus") or 0)),
        ("DI+ > DI- g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and (r.get("di_plus") or 0) > (r.get("di_minus") or 0)),
        ("5m EMA stack g2-8", lambda r: gap_ok(r, 0.02, 0.08) and r["ema_stack_5m"]),
        ("ST OR htf_up g2-8", lambda r: gap_ok(r, 0.02, 0.08) and (r["st_bull"] or r["htf_up"])),
        ("ST OR htf_up g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and (r["st_bull"] or r["htf_up"])),
        ("RSI40-70 + SMA20 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and rsi_ok(r, 40, 70) and r["above_sma20"]),
        ("no RSI g2-8 bounce", lambda r: gap_ok(r, 0.02, 0.08)),
        ("no RSI g1.5-8 bounce", lambda r: gap_ok(r, 0.015, 0.08)),
        ("ST bull + RSI 30-80 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and r["st_bull"] and rsi_ok(r, 30, 80)),
        ("ST bull g1.5-10", lambda r: gap_ok(r, 0.015, 0.10) and r["st_bull"]),
        ("RSI45-70 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 45, 70)),
        ("RSI40-65 g2-8", lambda r: gap_ok(r, 0.02, 0.08) and rsi_ok(r, 40, 65)),
        ("RSI40-65 g1.5-8", lambda r: gap_ok(r, 0.015, 0.08) and rsi_ok(r, 40, 65)),
    ]

    results = []
    extra_sizes = [
        ("top3", SIZE),
        ("top4", dict(risk_pct=8.0, max_pos=4, top_k=4, max_deploy=0.50)),
    ]
    print(f"{'pack':<36} {'n':>4} {'WR':>6} {'ret%':>7} {'PF':>5} {'DD':>5} {'OOS':>8} keep?")
    for label, pred in packs:
        n_cand = sum(1 for r in rows if pred(r) and r["bounce930"] and r["ts930"] is not None)
        for size_name, size in extra_sizes:
            if size_name == "top4" and "g1-8" not in label and "g1.5" not in label and "ST" not in label and "BASE" not in label:
                continue
            sigs = make_sigs(rows, pred)
            name = f"{label} {size_name}"
            res, tdf = simulate_open(stocks, sigs, name, book=book, **size)
            keep = (
                res.trades > BASELINE_N
                and res.total_return_pct >= BASELINE_RET - 0.05
            )
            flag = "YES" if keep else ""
            print(
                f"{name:<36} {res.trades:4d} {res.win_rate:6.1f} {res.total_return_pct:7.2f} "
                f"{res.profit_factor:5.2f} {res.max_dd_pct:5.1f} {res.oos_pnl:8.0f} {flag}  cand={n_cand}",
                flush=True,
            )
            results.append({
                "label": name,
                "filter": label,
                "size": size,
                "candidates_bounce": n_cand,
                "trades": res.trades,
                "win_rate": res.win_rate,
                "total_return_pct": res.total_return_pct,
                "profit_factor": res.profit_factor,
                "max_dd_pct": res.max_dd_pct,
                "oos_pnl": res.oos_pnl,
                "oos_wr": res.oos_wr,
                "keep": keep,
                "months": months_from_trades(tdf),
            })

    winners = [r for r in results if r["keep"]]
    winners.sort(key=lambda r: (r["total_return_pct"], r["trades"]), reverse=True)
    print("\n===== packs with MORE trades and return ≥ baseline =====")
    for r in winners[:15]:
        print(
            f"{r['label']:<36} n={r['trades']} WR {r['win_rate']:.1f}% "
            f"ret {r['total_return_pct']:.2f}% PF {r['profit_factor']:.2f} DD {r['max_dd_pct']:.1f}"
        )
    OUT.write_text(json.dumps({
        "baseline": {"trades": BASELINE_N, "return": BASELINE_RET},
        "winners": winners,
        "all": results,
    }, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
