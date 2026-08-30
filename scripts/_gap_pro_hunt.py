"""
Professional gap-down playbooks: confirmation, scale-out, breakeven, gap cap.

    python scripts/_gap_pro_hunt.py
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

from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    COST,
    FORCE_EXIT,
    SimResult,
    _summarize,
    nifty200_symbols,
)
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts.intraday_gap_hunt import (  # noqa: E402
    LEVERAGE,
    attach_rsi,
    build_sim_book,
    collect_events,
    months_from_trades,
    prepare,
)
from scripts._gap_tp_entry_search import enrich_events  # noqa: E402
from trading.services.intraday_gap import SL_ATR, compute_long_target  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_pro_hunt.json"


def _bar_t(ts) -> dtime:
    t = ts.tz_convert("Asia/Kolkata").time() if getattr(ts, "tzinfo", None) else ts.time()
    return t.replace(microsecond=0)


def make_signals(rows, *, mode, gap_min, gap_max, tp1, tp2):
    out = []
    for r in rows:
        gap = r["gap"]
        if gap > -gap_min or gap < -gap_max:
            continue
        if mode == "915":
            ts, entry, stop = r["ts915"], r["open915"], r["stop915"]
        else:
            if r["ts930"] is None or r["open930"] is None:
                continue
            if "alive" in mode and not r["alive930"]:
                continue
            if "bounce" in mode and not r["bounce930"]:
                continue
            ts, entry = r["ts930"], r["open930"]
            stop = entry - SL_ATR * r["atr"]
        if entry < 60:
            continue
        t1 = compute_long_target(entry, r["pdc"], r["atr"], stop, tp1)
        t2 = compute_long_target(entry, r["pdc"], r["atr"], stop, tp2)
        t2 = max(t2, t1)
        if not (t1 > entry > stop):
            continue
        out.append({
            "ts": ts, "symbol": r["symbol"], "side": "long",
            "entry": entry, "stop": stop, "target": t1, "target2": t2,
            "score": abs(gap), "gap": gap, "pdc": r["pdc"], "rsi": r["rsi"],
        })
    return pd.DataFrame(out)


def simulate_managed(
    signals,
    book,
    name,
    *,
    risk_pct=5.0,
    max_pos=4,
    top_k=2,
    max_deploy=0.35,
    scale=0.5,
    breakeven=True,
    time_stop=None,
    leverage=LEVERAGE,
):
    if signals is None or signals.empty:
        return SimResult(name=name), pd.DataFrame()

    sigs = signals.copy()
    sigs["day"] = pd.to_datetime(sigs["ts"]).dt.date
    sigs["rank"] = sigs.groupby("day")["score"].rank(method="first", ascending=False)
    sigs = sigs[sigs["rank"] <= top_k]

    ohlc = book["ohlc"]
    sess_bars = book["sess_bars"]
    last_bar = book["last_bar"]

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    cash = []
    taken = set()

    def _px(raw, side, exit=True):
        if side == "long":
            return raw * (1 - COST) if exit else raw * (1 + COST)
        return raw * (1 + COST) if exit else raw * (1 - COST)

    def record(sym, side, gap, entry_ts, exit_ts, entry, exit_p, stop, target, qty, pnl, reason, rsi, pdc):
        nonlocal equity, peak, max_dd
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        cash.append({
            "symbol": sym, "side": side, "gap": round(gap * 100, 2),
            "entry_ts": str(entry_ts), "exit_ts": str(exit_ts),
            "entry": round(entry, 2), "exit": round(exit_p, 2),
            "stop": round(float(stop), 2), "target": round(float(target), 2),
            "qty": qty, "pnl": round(pnl, 2), "reason": reason,
            "rsi": None if rsi is None else round(float(rsi), 1),
            "pdc": None if pdc is None else round(float(pdc), 2),
        })

    grouped = {}
    for rec in sigs.itertuples(index=False):
        grouped.setdefault(rec.day, []).append(rec)

    for day, recs in sorted(grouped.items(), key=lambda kv: kv[0]):
        recs = sorted(recs, key=lambda r: -r.score)
        used = 0
        for rec in recs:
            if used >= max_pos:
                break
            key = (rec.symbol, day)
            if key in taken:
                continue
            bars = sess_bars.get((rec.symbol, day))
            if not bars:
                continue
            raw = rec.entry
            entry = _px(raw, "long", exit=False)
            stop = float(rec.stop)
            t1 = float(rec.target)
            t2 = float(getattr(rec, "target2", rec.target))
            if not (t1 > entry > stop):
                continue
            risk_ps = entry - stop
            if risk_ps <= 0:
                continue
            bp = equity * leverage
            qty = int((equity * risk_pct / 100.0) / risk_ps)
            cap = int((bp * max_deploy) / entry)
            qty = max(min(qty, cap), 0)
            if qty <= 0:
                continue
            qty1 = max(int(round(qty * scale)), 0) if scale < 0.999 else qty
            qty1 = min(qty1, qty)
            qty2 = qty - qty1
            taken.add(key)
            used += 1

            scaled = False
            cur_stop = stop
            remaining = qty
            rsi = getattr(rec, "rsi", None)
            pdc = getattr(rec, "pdc", None)
            closed = False
            entry_t = _bar_t(rec.ts)
            for ts in bars:
                if _bar_t(ts) < entry_t:
                    continue
                bar = ohlc.get((rec.symbol, ts))
                if bar is None:
                    continue
                _o, high, low, close = bar
                # stop first
                if low <= cur_stop:
                    exit_p = _px(cur_stop, "long", True)
                    pnl = (exit_p - entry) * remaining
                    why = "be" if scaled and abs(cur_stop - entry) < 1e-6 else "sl"
                    record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, cur_stop, t1, remaining, pnl, why, rsi, pdc)
                    remaining = 0
                    closed = True
                    break
                if (not scaled) and high >= t1:
                    if qty1 > 0:
                        exit_p = _px(t1, "long", True)
                        pnl = (exit_p - entry) * qty1
                        record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, stop, t1, qty1, pnl, "t1", rsi, pdc)
                        remaining -= qty1
                    scaled = True
                    if breakeven:
                        cur_stop = max(cur_stop, entry)
                    if remaining <= 0:
                        closed = True
                        break
                    if high >= t2 and qty2 > 0:
                        exit_p = _px(t2, "long", True)
                        pnl = (exit_p - entry) * remaining
                        record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, cur_stop, t2, remaining, pnl, "t2", rsi, pdc)
                        remaining = 0
                        closed = True
                        break
                    continue
                if scaled and high >= t2:
                    exit_p = _px(t2, "long", True)
                    pnl = (exit_p - entry) * remaining
                    record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, cur_stop, t2, remaining, pnl, "t2", rsi, pdc)
                    remaining = 0
                    closed = True
                    break
                t = _bar_t(ts)
                if time_stop and (not scaled) and t >= time_stop:
                    exit_p = _px(close, "long", True)
                    pnl = (exit_p - entry) * remaining
                    record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, cur_stop, t1, remaining, pnl, "time", rsi, pdc)
                    remaining = 0
                    closed = True
                    break
                if t >= FORCE_EXIT:
                    exit_p = _px(close, "long", True)
                    pnl = (exit_p - entry) * remaining
                    record(rec.symbol, "long", rec.gap, rec.ts, ts, entry, exit_p, cur_stop, t1, remaining, pnl, "eod", rsi, pdc)
                    remaining = 0
                    closed = True
                    break
            if remaining > 0 and not closed:
                ts_last, last_c = last_bar[rec.symbol]
                exit_p = _px(last_c, "long", True)
                pnl = (exit_p - entry) * remaining
                record(rec.symbol, "long", rec.gap, rec.ts, ts_last, entry, exit_p, cur_stop, t1, remaining, pnl, "final", rsi, pdc)

    tdf = pd.DataFrame(cash)
    # One row per fill; summarize by original trades for WR of the *idea*
    if tdf is None or tdf.empty:
        res = _summarize(name, tdf, equity, max_dd)
        return res, tdf
    idea = tdf.groupby(["symbol", "entry_ts"], as_index=False).agg(
        pnl=("pnl", "sum"),
        qty=("qty", "sum"),
        gap=("gap", "first"),
        entry=("entry", "first"),
        side=("side", "first"),
        exit=("exit", "last"),
        stop=("stop", "first"),
        target=("target", "first"),
        entry_ts=("entry_ts", "first"),
        exit_ts=("exit_ts", "last"),
        reason=("reason", "last"),
        rsi=("rsi", "first"),
        pdc=("pdc", "first"),
        fills=("pnl", "count"),
    )
    res = _summarize(name, idea, equity, max_dd)
    res.params = dict(fills=len(tdf), ideas=len(idea))
    return res, idea


def pack(res, tdf, label, spec):
    reasons = {}
    if tdf is not None and not tdf.empty:
        reasons = {str(k): int(v) for k, v in tdf["reason"].value_counts().to_dict().items()}
    return {
        "label": label,
        **spec,
        "trades": res.trades,
        "win_rate": res.win_rate,
        "profit_factor": res.profit_factor,
        "net_pnl": res.net_pnl,
        "total_return_pct": res.total_return_pct,
        "max_dd_pct": res.max_dd_pct,
        "median_daily_pnl": res.median_daily_pnl,
        "oos_pnl": res.oos_pnl,
        "oos_wr": res.oos_wr,
        "reasons": reasons,
        "months": months_from_trades(tdf) if tdf is not None else [],
    }


def main():
    print("Professional gap hunt", flush=True)
    cache = load_frames(nifty200_symbols())
    stocks = prepare(cache)
    book = build_sim_book(stocks)
    rows = enrich_events(attach_rsi(collect_events(stocks)), stocks)
    print(f"events {len(rows)}  alive {sum(1 for r in rows if r['alive930'])}  bounce {sum(1 for r in rows if r['bounce930'])}", flush=True)

    playbooks = [
        dict(label="LIVE 9:30 1%", mode="930", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="OLD 9:15 fill", mode="915", gap_min=0.02, gap_max=0.15, tp1="fill", tp2="fill", scale=1.0, breakeven=False, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="A 9:30 alive 1%", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="B scale 1% then 2% BE", mode="930", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="C alive scale 1/2 BE", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="D alive bounce scale", mode="930_alive_bounce", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="E gap2-4 alive scale", mode="930_alive", gap_min=0.02, gap_max=0.04, tp1="pct1.0", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="F 9:15 scale 1% then fill", mode="915", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="fill", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="G 9:30 scale 1% then fill", mode="930", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="fill", scale=0.5, breakeven=True, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="H alive all names 1%", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=None, risk_pct=5.0, top_k=4, max_pos=4, max_deploy=0.35),
        dict(label="I alive all scale 0.8/1.5 BE", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct0.8", tp2="pct1.5", scale=0.6, breakeven=True, time_stop=None, risk_pct=5.0, top_k=4, max_pos=4, max_deploy=0.35),
        dict(label="J alive all 0.8% size-up", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct0.8", tp2="pct0.8", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=4, max_pos=4, max_deploy=0.50),
        dict(label="K alive all 1% size-up", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=4, max_pos=4, max_deploy=0.50),
        dict(label="L alive scale size-up", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=8.0, top_k=4, max_pos=4, max_deploy=0.50),
        dict(label="M 9:30 0.8% size-up k3", mode="930", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct0.8", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
        dict(label="N 9:30 scale 0.8/2 fill-cap size", mode="930", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct2.0", scale=0.6, breakeven=True, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
        dict(label="O time-stop 11:00 1%", mode="930_alive", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=dtime(11, 0), risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="P 9:15 scale 1% fill size-up", mode="915", gap_min=0.02, gap_max=0.15, tp1="pct1.0", tp2="fill", scale=0.5, breakeven=True, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.45),
        dict(label="Q bounce 0.8% size-up", mode="930_bounce", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct0.8", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
        dict(label="R alive bounce 0.8 size k4", mode="930_alive_bounce", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct1.5", scale=0.6, breakeven=True, time_stop=None, risk_pct=8.0, top_k=4, max_pos=4, max_deploy=0.50),
        dict(label="S bounce 0.8% k2 r5", mode="930_bounce", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct0.8", scale=1.0, breakeven=False, time_stop=None, risk_pct=5.0, top_k=2, max_pos=4, max_deploy=0.35),
        dict(label="T bounce 1.0% size-up", mode="930_bounce", gap_min=0.02, gap_max=0.08, tp1="pct1.0", tp2="pct1.0", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
        dict(label="U bounce 0.6% size-up", mode="930_bounce", gap_min=0.02, gap_max=0.08, tp1="pct0.6", tp2="pct0.6", scale=1.0, breakeven=False, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
        dict(label="V bounce scale 0.8/2 size", mode="930_bounce", gap_min=0.02, gap_max=0.08, tp1="pct0.8", tp2="pct2.0", scale=0.5, breakeven=True, time_stop=None, risk_pct=8.0, top_k=3, max_pos=4, max_deploy=0.50),
    ]

    book_rows = []
    for i, spec in enumerate(playbooks, 1):
        sigs = make_signals(
            rows, mode=spec["mode"], gap_min=spec["gap_min"], gap_max=spec["gap_max"],
            tp1=spec["tp1"], tp2=spec["tp2"],
        )
        res, tdf = simulate_managed(
            sigs, book, spec["label"],
            risk_pct=spec["risk_pct"], max_pos=spec["max_pos"], top_k=spec["top_k"],
            max_deploy=spec["max_deploy"], scale=spec["scale"], breakeven=spec["breakeven"],
            time_stop=spec["time_stop"],
        )
        row = pack(res, tdf, spec["label"], spec)
        book_rows.append(row)
        print(
            f"{i:2d}/{len(playbooks)}  {row['total_return_pct']:7.1f}%  WR {row['win_rate']:5.1f}  "
            f"PF {row['profit_factor']:5.2f}  n={row['trades']:3d}  med {row['median_daily_pnl']:8.0f}  "
            f"OOS {row['oos_pnl']:8.0f}  DD {row['max_dd_pct']:4.1f}  {spec['label']}",
            flush=True,
        )

    ranked = sorted(
        book_rows,
        key=lambda r: (
            min((r["win_rate"] or 0) / 70.0, 1.5) + min((r["total_return_pct"] or 0) / 70.0, 1.5),
            r["oos_pnl"] or 0,
            r["profit_factor"] or 0,
        ),
        reverse=True,
    )
    payload = {"rows": book_rows, "best": ranked[0] if ranked else None}
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT)
    print("\n===== ranked vs 70/70 =====")
    for r in ranked[:10]:
        print(
            f"  {r['total_return_pct']:6.1f}% WR {r['win_rate']:5.1f}  n={r['trades']:3d}  "
            f"OOS {r['oos_pnl']:8.0f}  {r['label']}"
        )


if __name__ == "__main__":
    main()
