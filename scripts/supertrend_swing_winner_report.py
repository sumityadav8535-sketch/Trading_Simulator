"""Detailed report for the recommended Supertrend swing winner."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from scripts.supertrend_swing_search import (
    CAPITAL,
    FILTER_PACKS,
    START,
    END,
    StockPack,
    _preload_frames,
    collect_daily_signals,
)
from scripts.supertrend_swing_validate import simulate_capped
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def report(name, r):
    print(f"\n{'='*90}")
    print(name)
    print(
        f"n={r['n']} wins={r['wins']} WR={r['wr']}% ret={r['ret']}% "
        f"PF={r['pf']} DD={r['dd']}% final=₹{r['final']:,.0f}"
    )
    print(
        f"avg pos={r['avg_pos_frac']}% med pos={r['med_pos_frac']}% "
        f"max pos={r['max_pos_frac']}%  med stop={r['med_stop_pct']}% avg stop={r['avg_stop_pct']}%"
    )
    monthly = defaultdict(lambda: {"pnl": 0.0, "n": 0, "w": 0})
    by_reason = Counter()
    by_sym = defaultdict(float)
    for t in r["trades"]:
        m = str(t["exit"])[:7]
        monthly[m]["pnl"] += t["pnl"]
        monthly[m]["n"] += 1
        if t["pnl"] > 0:
            monthly[m]["w"] += 1
        by_reason[t["reason"]] += 1
        by_sym[t["symbol"]] += t["pnl"]
    print("exits:", dict(by_reason))
    print("monthly:")
    eq = CAPITAL
    for m in sorted(monthly):
        d = monthly[m]
        eq += d["pnl"]
        wr = d["w"] / d["n"] * 100 if d["n"] else 0
        print(f"  {m}  pnl={d['pnl']:+10,.0f}  n={d['n']:3d} WR={wr:5.0f}%  eq=₹{eq:,.0f}")
    mid = sorted(r["trades"], key=lambda t: str(t["exit"]))
    if mid:
        half = str(START)[:7]
        # split by exit date around Feb 2026
        h1 = sum(t["pnl"] for t in r["trades"] if str(t["exit"]) < "2026-02-19")
        h2 = sum(t["pnl"] for t in r["trades"] if str(t["exit"]) >= "2026-02-19")
        print(f"H1 (to 2026-02-18): {h1:+,.0f}  ({h1/CAPITAL*100:+.1f}%)")
        print(f"H2 (from 2026-02-19): {h2:+,.0f}  ({h2/CAPITAL*100:+.1f}%)")
    print("top 8 stocks by PnL:")
    for s, p in sorted(by_sym.items(), key=lambda x: -x[1])[:8]:
        print(f"  {s:<12} {p:+,.0f}")
    print("top 5 winners:")
    for t in sorted(r["trades"], key=lambda x: -x["pnl"])[:5]:
        print(
            f"  {t['symbol']:<12} {str(t['entry'])[:10]}→{str(t['exit'])[:10]} "
            f"{t['entry_px']:.1f}→{t['exit_px']:.1f} pos={t['pos_frac']*100:.0f}% "
            f"pnl={t['pnl']:+,.0f} {t['reason']} d={t['hold']}"
        )
    print("worst 5:")
    for t in sorted(r["trades"], key=lambda x: x["pnl"])[:5]:
        print(
            f"  {t['symbol']:<12} {str(t['entry'])[:10]}→{str(t['exit'])[:10]} "
            f"{t['entry_px']:.1f}→{t['exit_px']:.1f} pos={t['pos_frac']*100:.0f}% "
            f"pnl={t['pnl']:+,.0f} {t['reason']} d={t['hold']}"
        )
    return monthly


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    st_params = [(14, 3.0), (21, 3.0)]
    packs = {sym: StockPack(sym, df, st_params) for sym, df in frames.items()}
    nifty = StockPack("NIFTY50", nifty_df, st_params)
    st_ts, et_ts = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st_ts) & (df.index <= et_ts)].tolist()
    })
    fmap = dict(FILTER_PACKS)
    raw14 = collect_daily_signals(packs, nifty, (14, 3.0), "pullback", st_ts, et_ts)
    raw21 = collect_daily_signals(packs, nifty, (21, 3.0), "pullback", st_ts, et_ts)

    jobs = [
        ("A  ST14,3 pullback + quality  risk3 cap50 trail90", raw14, (14, 3.0), "quality", 3.0, 0.50, 0.0),
        ("A+ ST14,3 pullback + quality  risk3 cap50 trail90 +0.1% cost", raw14, (14, 3.0), "quality", 3.0, 0.50, 0.10),
        ("B  ST14,3 pullback + trend_rsi risk4 cap50 trail90", raw14, (14, 3.0), "trend_rsi", 4.0, 0.50, 0.0),
        ("B+ ST14,3 pullback + trend_rsi risk4 cap50 trail90 +0.1% cost", raw14, (14, 3.0), "trend_rsi", 4.0, 0.50, 0.10),
        ("C  ST14,3 pullback + quality  risk3 cap75 trail90", raw14, (14, 3.0), "quality", 3.0, 0.75, 0.0),
        ("D  ST21,3 pullback + ADX20    risk3 cap50 trail90", raw21, (21, 3.0), "adx20", 3.0, 0.50, 0.0),
    ]
    out = []
    for name, raw, st_key, filt, risk, cap, cost in jobs:
        r = simulate_capped(
            packs, calendar, raw, capital=CAPITAL, risk_pct=risk,
            need_flags=fmap[filt], stop_mode="st", exit_mode="st_trail",
            target_rr=0, max_hold=90, st_key=st_key,
            max_pos_pct=cap, cost_pct=cost,
        )
        report(name, r)
        out.append({"name": name, "ret": r["ret"], "wr": r["wr"], "pf": r["pf"],
                    "dd": r["dd"], "n": r["n"], "final": r["final"]})

    path = os.path.join(ROOT, "data", "supertrend_swing_winner.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"window": [str(START), str(END)], "jobs": out}, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
