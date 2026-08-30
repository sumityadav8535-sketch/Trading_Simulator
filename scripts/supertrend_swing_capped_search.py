"""Find Supertrend pullback configs that still clear 100% with a position cap."""
from __future__ import annotations

import os
import sys

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


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    st_params = [(10, 2.0), (10, 3.0), (14, 2.0), (14, 3.0), (21, 3.0), (7, 3.0)]
    packs = {sym: StockPack(sym, df, st_params) for sym, df in frames.items()}
    nifty = StockPack("NIFTY50", nifty_df, st_params)
    st_ts, et_ts = pd.Timestamp(START), pd.Timestamp(END)
    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st_ts) & (df.index <= et_ts)].tolist()
    })

    entries = ["pullback", "breakout", "reclaim"]
    filters = [
        "none", "adx20", "adx25", "rsi_healthy", "quality", "trend_rsi",
        "ema_trend", "above_200", "rs_outperform", "mkt_st", "volume",
        "quality_mkt", "not_extended",
    ]
    fmap = dict(FILTER_PACKS)
    raw_cache = {}
    rows = []

    print(f"Capped hunt {START}→{END}  N={len(packs)}", flush=True)
    for st_key in st_params:
        for entry in entries:
            raw = collect_daily_signals(packs, nifty, st_key, entry, st_ts, et_ts)
            raw_cache[(st_key, entry)] = raw
            print(f"  ST{st_key} {entry} sig={len(raw)}", flush=True)
            for fname in filters:
                need = fmap[fname]
                for risk in (3.0, 4.0, 5.0, 6.0, 8.0):
                    for cap in (0.33, 0.50, 0.75, 1.00):
                        for exit_m, hold in (("st_trail", 60), ("st_trail", 90), ("st_flip", 60)):
                            r = simulate_capped(
                                packs, calendar, raw, capital=CAPITAL, risk_pct=risk,
                                need_flags=need, stop_mode="st", exit_mode=exit_m,
                                target_rr=0, max_hold=hold, st_key=st_key,
                                max_pos_pct=cap, cost_pct=0.0,
                            )
                            r.update({
                                "st": f"{st_key[0]},{st_key[1]:g}",
                                "entry": entry, "filter": fname, "risk": risk,
                                "cap": int(cap * 100), "exit": exit_m, "hold": hold,
                            })
                            rows.append(r)

    usable = [r for r in rows if r["n"] >= 15]
    print(f"\nSims={len(rows)} usable={len(usable)}")
    print(f">=100% any cap: {sum(1 for r in usable if r['ret']>=100)}")
    print(f">=100% cap<=50: {sum(1 for r in usable if r['ret']>=100 and r['cap']<=50)}")
    print(f">=100% cap<=75: {sum(1 for r in usable if r['ret']>=100 and r['cap']<=75)}")

    def show(title, subset, k=12):
        print(f"\n{title}")
        for r in sorted(subset, key=lambda x: (x["ret"], x["pf"]), reverse=True)[:k]:
            print(
                f"  ST{r['st']:<6} {r['entry']:<9} {r['filter']:<14} "
                f"risk={r['risk']:g}% cap={r['cap']:>3}% {r['exit']:<8} h={r['hold']} "
                f"n={r['n']:3d} WR={r['wr']:5.1f}% ret={r['ret']:+7.1f}% "
                f"PF={r['pf']:.2f} DD={r['dd']:5.1f}% pos={r['avg_pos_frac']:.0f}%"
            )

    show("TOP UNCAPPED (cap 100)", [r for r in usable if r["cap"] == 100])
    show("TOP CAP 75%", [r for r in usable if r["cap"] == 75])
    show("TOP CAP 50%", [r for r in usable if r["cap"] == 50])
    show("TOP CAP 33%", [r for r in usable if r["cap"] == 33])
    hits50 = [r for r in usable if r["ret"] >= 100 and r["cap"] <= 50]
    show("HITS >=100% WITH CAP<=50", hits50, 15)
    hits75 = [r for r in usable if r["ret"] >= 100 and r["cap"] <= 75]
    show("HITS >=100% WITH CAP<=75", hits75, 12)

    # recommend: cap<=50, maximize ret, require PF>=2, n>=20, dd<=20
    cand = [
        r for r in usable
        if r["cap"] <= 50 and r["n"] >= 20 and r["pf"] >= 2.0 and r["dd"] <= 20
    ]
    show("BEST ROBUST (cap<=50, n>=20, PF>=2, DD<=20)", cand, 10)


if __name__ == "__main__":
    main()
