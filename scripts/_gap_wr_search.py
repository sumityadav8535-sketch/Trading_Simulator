"""
Search Gap Open packs for WR >= 70% after the 13:10 close change.

    python scripts/_gap_wr_search.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import time as dtime
from pathlib import Path

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
from trading.services.intraday_gap import compute_long_target  # noqa: E402

OUT = ROOT / "data" / "_gap_wr_search.json"
ENTRY_930 = dtime(9, 30)
SIZE = dict(risk_pct=8.0, max_pos=4, top_k=4, max_deploy=0.50)


def _time_of(ts) -> dtime:
    t = ts.tz_convert("Asia/Kolkata").time() if getattr(ts, "tzinfo", None) else ts.time()
    return t.replace(microsecond=0)


def enrich_all(events: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> list[dict]:
    days = session_lookup(stocks)
    rows = []
    for r in events.itertuples(index=False):
        day = days.get((r.symbol, r.session))
        if day is None or day.empty:
            continue
        o = float(r.open)
        atr = float(r.atr) if float(r.atr) > 0 else o * 0.008
        rsi = getattr(r, "rsi14", None)
        pdh = float(getattr(r, "pdh", 0) or 0)
        close915 = float(r.close1) if hasattr(r, "close1") else None
        row930 = None
        ts930 = None
        low_until_930 = None
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
        close_at_high = bool(pdh > 0 and float(r.pdc) >= pdh * 0.998)
        rows.append({
            "symbol": r.symbol,
            "session": r.session,
            "ts915": r.ts,
            "open915": o,
            "close915": close915,
            "pdc": float(r.pdc),
            "pdh": pdh,
            "gap": float(r.gap),
            "atr": atr,
            "rsi": float(rsi) if rsi is not None and pd.notna(rsi) else None,
            "ts930": ts930,
            "open930": open930,
            "bounce930": bool(open930 is not None and close915 is not None and open930 >= close915),
            "close_at_high": close_at_high,
        })
    return rows


def make_sigs(rows: list[dict], *, bounce: bool, tp: str, sl_atr: float) -> pd.DataFrame:
    out = []
    for r in rows:
        if r["ts930"] is None or r["open930"] is None:
            continue
        if bounce and not r["bounce930"]:
            continue
        entry = float(r["open930"])
        stop = entry - sl_atr * float(r["atr"])
        if entry < 60:
            continue
        tgt = compute_long_target(entry, r["pdc"], r["atr"], stop, tp)
        if not (tgt > entry > stop):
            continue
        out.append({
            "ts": r["ts930"],
            "symbol": r["symbol"],
            "side": "long",
            "entry": entry,
            "stop": stop,
            "target": tgt,
            "score": abs(r["gap"]),
            "gap": r["gap"],
            "pdc": r["pdc"],
            "rsi": r["rsi"],
        })
    if not out:
        return pd.DataFrame(columns=["ts", "symbol", "side", "entry", "stop", "target", "score", "gap", "pdc", "rsi"])
    return pd.DataFrame(out)


def filt(rows: list[dict], gmin, gmax, rlo, rhi, skip_auction: bool) -> list[dict]:
    out = []
    for r in rows:
        gap = r["gap"]
        rsi = r["rsi"]
        if gap > -gmin or gap < -gmax:
            continue
        if rsi is None or rsi < rlo or rsi > rhi:
            continue
        if skip_auction and r["close_at_high"]:
            continue
        out.append(r)
    return out


def main():
    symbols = nifty200_symbols()
    print(f"WR search | {len(symbols)} names", flush=True)
    stocks = prepare(load_frames(symbols))
    book = build_sim_book(stocks)
    print(f"prepared {len(stocks)}", flush=True)

    pdc_modes = ["13:10", "13:15", "15:10", "official"]
    tps = ["pct0.4", "pct0.5", "pct0.6", "pct0.8", "pct1.0"]
    sls = [0.6, 1.0, 1.5]
    gaps = [(0.02, 0.06), (0.02, 0.04), (0.015, 0.05)]
    rsis = [(45, 70), (50, 70), (40, 65)]
    bounces = [True]
    skip_flags = [False, True]

    enriched = {}
    for mode in pdc_modes:
        ev = attach_rsi(collect_events(stocks, pdc_mode=mode))
        enriched[mode] = enrich_all(ev, stocks)
        print(f"  events {mode}: {len(ev)} enriched {len(enriched[mode])}", flush=True)

    rows_out = []
    n = 0
    total = len(pdc_modes) * len(tps) * len(sls) * len(gaps) * len(rsis) * len(bounces) * len(skip_flags)
    for mode, base in enriched.items():
        for gmin, gmax in gaps:
            for rlo, rhi in rsis:
                for skip in skip_flags:
                    subset = filt(base, gmin, gmax, rlo, rhi, skip)
                    if len(subset) < 8:
                        n += len(tps) * len(sls)
                        continue
                    for sl in sls:
                        for tp in tps:
                            n += 1
                            sigs = make_sigs(subset, bounce=True, tp=tp, sl_atr=sl)
                            label = (
                                f"{mode} {tp} sl{sl} g{gmin:.0%}-{gmax:.0%} "
                                f"rsi{rlo:.0f}-{rhi:.0f}{' noAH' if skip else ''}"
                            )
                            res, tdf = simulate_open(stocks, sigs, label, book=book, **SIZE)
                            rec = {
                                "pdc": mode,
                                "tp": tp,
                                "sl_atr": sl,
                                "gap_min": gmin,
                                "gap_max": gmax,
                                "rsi_lo": rlo,
                                "rsi_hi": rhi,
                                "skip_auction_high": skip,
                                "bounce": True,
                                "trades": res.trades,
                                "win_rate": res.win_rate,
                                "profit_factor": res.profit_factor,
                                "net_pnl": res.net_pnl,
                                "total_return_pct": res.total_return_pct,
                                "max_dd_pct": res.max_dd_pct,
                                "oos_pnl": res.oos_pnl,
                                "oos_wr": res.oos_wr,
                                "median_daily_pnl": res.median_daily_pnl,
                                "trading_days": res.trading_days,
                                "months": months_from_trades(tdf),
                            }
                            rows_out.append(rec)
                            if n % 40 == 0 or n == total:
                                print(
                                    f"  {n}/{total}  WR {res.win_rate:5.1f}  n={res.trades:3d}  "
                                    f"ret {res.total_return_pct:6.1f}  {label}",
                                    flush=True,
                                )

    rows_out.sort(
        key=lambda r: (
            1 if r["win_rate"] >= 70 and r["profit_factor"] >= 1 and r["trades"] >= 15 else 0,
            1 if r["oos_pnl"] > 0 else 0,
            r["win_rate"],
            r["profit_factor"],
            r["total_return_pct"],
            r["trades"],
        ),
        reverse=True,
    )
    hits = [
        r for r in rows_out
        if r["win_rate"] >= 70 and r["profit_factor"] >= 1 and r["trades"] >= 15
    ]
    print(f"\nPacks WR>=70 PF>=1 n>=15: {len(hits)} / {len(rows_out)}")
    print(f"{'pdc':<10} {'tp':<8} {'sl':>4} {'gap':<8} {'rsi':<8} {'AH':>3} {'n':>4} {'WR':>6} {'PF':>5} {'ret%':>7} {'OOS':>8}")
    show = hits[:20] if hits else rows_out[:20]
    for r in show:
        print(
            f"{r['pdc']:<10} {r['tp']:<8} {r['sl_atr']:4.1f} "
            f"{r['gap_min']*100:.0f}-{r['gap_max']*100:.0f}%   "
            f"{r['rsi_lo']:.0f}-{r['rsi_hi']:.0f}  "
            f"{'Y' if r['skip_auction_high'] else 'n':>3} "
            f"{r['trades']:4d} {r['win_rate']:6.1f} {r['profit_factor']:5.2f} "
            f"{r['total_return_pct']:7.1f} {r['oos_pnl']:8.0f}"
        )

    payload = {
        "goal": "WR >= 70, PF >= 1, n >= 15",
        "size": SIZE,
        "hits": hits[:40],
        "top": rows_out[:40],
        "n_tested": len(rows_out),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
