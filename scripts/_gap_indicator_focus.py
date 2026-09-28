"""Finer target and size check around the gap-down bounce. Prints trades."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

from scripts._gap_indicator_opt import (  # noqa: E402
    CAPITAL,
    EXITS,
    LEVERAGE,
    build_book,
    eval_book,
    nifty_context,
)
from scripts.intraday_15m_n200_search import nifty200_symbols  # noqa: E402
from scripts.intraday_5m_hunt import load_frames  # noqa: E402


def add_exit(name, **spec):
    EXITS[name] = spec


def main():
    for pct, label in ((0.006, "p06"), (0.008, "p08"), (0.010, "p10"), (0.012, "p12"), (0.015, "p15")):
        for sl, slname in ((1.0, "a10"), (1.5, "a15"), (2.0, "a20")):
            add_exit(f"{label}_{slname}", tp="pct", pct=pct, stop="atr", sl=sl)

    symbols = nifty200_symbols()
    cache = load_frames(symbols)
    stocks = {k: v for k, v in cache.items() if not k.startswith("_")}
    feat, paths, calendar = build_book(stocks, nifty_context())
    oos_cut = calendar[int(len(calendar) * 0.70)]
    june = [d for d in calendar if d.month >= 6 or d.year > 2026]
    print(f"sessions {len(calendar)}  from June {len(june)}  oos {oos_cut}", flush=True)

    live = (feat["gap"] <= -0.02) & (feat["gap"] >= -0.06) & feat["rsi"].between(45, 70) & feat["bounce"]
    flat = live & (feat["nifty_gap"] > -0.004)
    ok = live & (feat["nifty_gap"] > -0.008)
    masks = {"live": live.fillna(False), "nifty>-0.4": flat.fillna(False), "nifty>-0.8": ok.fillna(False)}
    sizes = [
        ("top1", dict(risk_pct=8, max_deploy=1.0, top_k=1, cap_total=True)),
        ("top2", dict(risk_pct=8, max_deploy=0.5, top_k=2, cap_total=True)),
    ]
    exits = [f"p{p}_{s}" for p in ("06", "08", "10", "12", "15") for s in ("a10", "a15", "a20")]

    rows = []
    for mname, mask in masks.items():
        for ex in exits:
            for sname, sz in sizes:
                res = eval_book(
                    feat, paths, mask, ex, calendar, oos_cut,
                    name=f"{mname} {ex} {sname}", **sz,
                )
                res["june_sessions"] = len(june)
                rows.append(res)
    rows.sort(key=lambda r: (r["red_months"] == 0, r["oos_pnl"] > 0, r["net_pnl"]), reverse=True)
    print("\nGreen every month, OOS > 0, sorted by net")
    shown = 0
    for r in rows:
        if r["red_months"] or r["oos_pnl"] <= 0 or r["trades"] < 12:
            continue
        mo = " ".join(f"{k}:{v['pnl']:.0f}/{v['trades']}" for k, v in sorted(r["months"].items()))
        per_june = r["net_pnl"] / len(june)
        print(
            f"{r['name']:<28} n={r['trades']:3d} WR {r['win_rate']:5.1f} PF {r['profit_factor']:4.2f} "
            f"net {r['net_pnl']:8.0f} tday {r['avg_trade_day']:7.0f} june/day {per_june:6.0f} "
            f"5k {r['days_ge_5k']} worst {r['worst_day']:8.0f} DD {r['max_dd_pct']:4.1f} "
            f"oos {r['oos_pnl']:7.0f} {mo}"
        )
        shown += 1
        if shown >= 18:
            break

    # Trade tape for the reference book: 1% / 1.5 ATR / largest gap / full 5x.
    print("\n===== TRADES top1 p10 a15 =====")
    # Re-run inline to list trades. eval_book does not return them, so walk here.
    from scripts._gap_indicator_opt import _levels, _walk

    spec = EXITS["p10_a15"]
    by = {}
    idx = live.fillna(False).to_numpy().nonzero()[0]
    for i in idx:
        row = feat.iloc[int(i)]
        entry = float(row["open930"])
        levels = _levels(row, spec, "long", entry)
        if not levels:
            continue
        walked = _walk(paths[int(i)], 0, entry, levels[0], levels[1], "long", spec)
        if not walked:
            continue
        by.setdefault(row["session"], []).append((abs(float(row["gap"])), row, walked, levels))
    equity = CAPITAL
    print(f"{'date':<12} {'sym':<12} {'gap':>6} {'nifty':>7} {'qty':>6} {'pnl':>9} why")
    for sess in sorted(by):
        cands = sorted(by[sess], key=lambda x: -x[0])
        score, row, walked, levels = cands[0]
        pnl_ps, risk, entry_f, reason = walked
        bp = equity * LEVERAGE
        qty = int((equity * 8 / 100) / risk) if risk > 0 else 0
        cap = int(bp / entry_f) if entry_f else 0
        qty = max(min(qty, cap), 0)
        pnl = pnl_ps * qty
        ng = row["nifty_gap"]
        ng_s = f"{ng * 100:5.2f}%" if ng == ng else "  n/a"
        print(
            f"{sess} {row['symbol']:<12} {row['gap'] * 100:6.2f} {ng_s} {qty:6d} {pnl:9.0f} {reason}"
        )
        equity += pnl
    print(f"end equity {equity:.0f}")


def export():
    """Write the live 1% / 2 ATR / largest-gap tape and refresh the page story."""
    import json

    from scripts._gap_indicator_opt import _levels, _walk

    add_exit("p10_a20", tp="pct", pct=0.010, stop="atr", sl=2.0)
    symbols = nifty200_symbols()
    cache = load_frames(symbols)
    stocks = {k: v for k, v in cache.items() if not k.startswith("_")}
    feat, paths, calendar = build_book(stocks, nifty_context())
    oos_cut = calendar[int(len(calendar) * 0.70)]
    live = (
        (feat["gap"] <= -0.02) & (feat["gap"] >= -0.06) & feat["rsi"].between(45, 70) & feat["bounce"]
    ).fillna(False)
    spec = EXITS["p10_a20"]
    by = {}
    for i in live.to_numpy().nonzero()[0]:
        row = feat.iloc[int(i)]
        entry = float(row["open930"])
        levels = _levels(row, spec, "long", entry)
        if not levels:
            continue
        walked = _walk(paths[int(i)], 0, entry, levels[0], levels[1], "long", spec)
        if not walked:
            continue
        by.setdefault(row["session"], []).append((abs(float(row["gap"])), row, walked, levels, entry))

    equity = CAPITAL
    trades = []
    for sess in sorted(by):
        _score, row, walked, levels, entry = sorted(by[sess], key=lambda x: -x[0])[0]
        pnl_ps, risk, entry_f, reason = walked
        bp = equity * LEVERAGE
        qty = int((equity * 8 / 100) / risk) if risk > 0 else 0
        cap = int(bp / entry_f) if entry_f else 0
        qty = max(min(qty, cap), 0)
        if qty <= 0:
            continue
        pnl = pnl_ps * qty
        exit_f = entry_f + pnl_ps
        hh = 15
        mm = 10
        trades.append({
            "symbol": row["symbol"],
            "side": "long",
            "gap": round(float(row["gap"]) * 100, 2),
            "rsi": None if row["rsi"] != row["rsi"] else round(float(row["rsi"]), 1),
            "entry_ts": f"{sess} 09:30:00+05:30",
            "exit_ts": f"{sess} {hh:02d}:{mm:02d}:00+05:30",
            "entry": round(entry, 2),
            "stop": round(float(levels[0]), 2),
            "target": round(float(levels[1]), 2),
            "exit": round(exit_f, 2),
            "qty": int(qty),
            "pnl": round(pnl, 2),
            "reason": reason,
            "pdc": round(float(row["pdc"]), 2),
        })
        equity += pnl

    res = eval_book(
        feat, paths, live, "p10_a20", calendar, oos_cut,
        name="live p10 a20 top1", risk_pct=8, max_deploy=1.0, top_k=1, cap_total=True,
    )
    months = [
        {"month": k, "trades": v["trades"], "pnl": v["pnl"], "wr": v["wr"]}
        for k, v in sorted(res["months"].items())
    ]
    by_day = {}
    for t in trades:
        by_day[t["entry_ts"][:10]] = by_day.get(t["entry_ts"][:10], 0.0) + t["pnl"]
    day_vals = sorted(by_day.values())
    median_day = day_vals[len(day_vals) // 2] if day_vals else 0.0
    story = {
        "name": "Largest gap-down · 9:30 bounce · +1% · stop 2× ATR",
        "trades": res["trades"],
        "win_rate": res["win_rate"],
        "profit_factor": res["profit_factor"],
        "net_pnl": res["net_pnl"],
        "total_return_pct": round(res["net_pnl"] / CAPITAL * 100, 2),
        "max_dd_pct": res["max_dd_pct"],
        "avg_daily_pnl": res["avg_trade_day"],
        "median_daily_pnl": round(median_day, 2),
        "days_ge_5k": res["days_ge_5k"],
        "trading_days": res["trade_days"],
        "worst_day": res["worst_day"],
        "best_day": res["best_day"],
        "oos_pnl": res["oos_pnl"],
        "oos_wr": None,
        "months": months,
        "params": {
            "mode": "down_bounce",
            "target": "pct1.0",
            "entry": "09:30",
            "bounce": True,
            "prior_close": "15:10",
            "gap_min": 0.02,
            "gap_max": 0.06,
            "rsi_lo": 45,
            "rsi_hi": 70,
            "sl_atr": 2.0,
            "risk_pct": 8.0,
            "max_pos": 1,
            "top_k": 1,
            "max_deploy": 1.0,
        },
    }
    hunt_path = ROOT / "data" / "intraday_gap_hunt.json"
    hunt = json.loads(hunt_path.read_text(encoding="utf-8"))
    hunt["story"] = story
    hunt["result"] = story
    hunt["window"] = {
        "start": "2026-06-05",
        "end": "2026-09-25",
        "stocks": 204,
        "note": "5-minute bars. One name per day, largest gap, +1% target, stop 2× prior-session 5-minute ATR, full 5× book.",
    }
    hunt["idea"] = (
        "Buy the largest 2–6% gap-down in Nifty 200 at 9:30 if it is holding the 9:15 close "
        "and yesterday’s RSI is 45–70. Target +1%. Stop 2× the 5-minute ATR. One name, full 5× book."
    )
    hunt["strategy"] = {
        "name": "Gap-down bounce · RSI 45–70",
        "tp_kind": "pct1.0",
        "sl_atr": 2.0,
        "top_k": 1,
        "max_deploy": 1.0,
    }
    hunt_path.write_text(json.dumps(hunt, indent=2, default=str), encoding="utf-8")
    (ROOT / "data" / "intraday_gap_trades.json").write_text(
        json.dumps(trades, indent=2), encoding="utf-8"
    )
    print(
        f"exported {len(trades)} trades  net {res['net_pnl']:.0f}  "
        f"end {equity:.0f}  months {months}"
    )


if __name__ == "__main__":
    main()

