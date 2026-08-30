"""Scale the only profitable 15m family until 100% or the MIS wall."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_hunt import (  # noqa: E402
    load_frames,
    months_from_trades,
    prepare,
    simulate_sized,
)
from scripts.intraday_15m_n200_phase3 import attach_index, sig_vwap_fade_v2  # noqa: E402
from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    build_exec_index,
    collect_signals,
    nifty200_symbols,
)

OUT = ROOT / "data" / "intraday_scale_fade.json"
TRADES = ROOT / "data" / "intraday_scale_fade_trades.json"


def main():
    cache = load_frames(nifty200_symbols())
    stocks, idx = prepare(cache)
    stocks = attach_index(stocks, idx)
    exec_idx = build_exec_index(stocks)
    variants = [
        ("fade 1.8% VWAP", dict(ext_min=0.018, adx_max=26, target_mode="vwap", idx_abs_max=0.008)),
        ("fade 1.8% ATR", dict(ext_min=0.018, adx_max=26, target_mode="atr", idx_abs_max=0.008)),
        ("fade 1.8% half", dict(ext_min=0.018, adx_max=26, target_mode="half", idx_abs_max=0.008)),
        ("fade 1.8% VWAP idx1.2", dict(ext_min=0.018, adx_max=28, target_mode="vwap", idx_abs_max=0.012)),
    ]
    sigs = {n: collect_signals(stocks, sig_vwap_fade_v2, **kw) for n, kw in variants}
    for n, s in sigs.items():
        print(f"  {n:28s} {len(s):4d}")

    combos = []
    for lev in (5.0, 8.0, 12.0):
        for deploy in (0.8, 1.0, 2.0, 4.0):
            for max_pos in (6, 8, 12):
                for top_k in (2, 4, 8):
                    combos.append(dict(
                        mode="risk", risk_pct=25.0, alloc_pct=0.0, leverage=lev,
                        max_pos=max_pos, top_k=top_k, max_deploy=deploy,
                        daily_lock=0.0, daily_halt=0.0,
                    ))
    # unconstrained risk (cap off)
    combos.append(dict(
        mode="risk", risk_pct=25.0, alloc_pct=0.0, leverage=5.0,
        max_pos=8, top_k=8, max_deploy=20.0,
        daily_lock=0.0, daily_halt=0.0,
    ))

    book = []
    total = len(variants) * len(combos)
    n = 0
    for fam, kw in variants:
        for sz in combos:
            n += 1
            label = (
                f"{fam} | lev{sz['leverage']:g} d{sz['max_deploy']:g} "
                f"p{sz['max_pos']} k{sz['top_k']}"
            )
            res, tdf = simulate_sized(exec_idx, sigs[fam], label, **sz)
            res.params.update(kw)
            res.params["family"] = fam
            book.append((res, tdf, months_from_trades(tdf)))
            if n % 40 == 0 or n == total:
                print(f"  {n}/{total}  {res.total_return_pct:7.1f}%  DD {res.max_dd_pct:5.1f}  {label[:70]}", flush=True)

    ranked = sorted(book, key=lambda x: (x[0].total_return_pct, -x[0].max_dd_pct), reverse=True)
    print("\n===== HIGHEST RETURN =====")
    for res, _, months in ranked[:15]:
        print(
            f"{res.total_return_pct:7.1f}%  DD {res.max_dd_pct:5.1f}  WR {res.win_rate:5.1f}  "
            f"OOS {res.oos_pnl:8.0f}  {res.name[:72]}"
        )
    hits = [x for x in ranked if x[0].total_return_pct >= 100 and x[0].oos_pnl > 0]
    print(f"\nHits >=100% with OOS>0: {len(hits)}")
    wres, wtdf, wmonths = hits[0] if hits else ranked[0]
    payload = {
        "hit_100": bool(hits),
        "n_hits": len(hits),
        "winner": {**asdict(wres), "months": wmonths},
        "hits": [{**asdict(r), "months": m} for r, _, m in hits[:10]],
        "top": [{**asdict(r), "months": m} for r, _, m in ranked[:15]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if wtdf is not None and not wtdf.empty:
        wtdf.to_json(TRADES, orient="records", date_format="iso")
    print("WINNER", wres.name, wres.total_return_pct, "DD", wres.max_dd_pct, "months", wmonths)


if __name__ == "__main__":
    main()
