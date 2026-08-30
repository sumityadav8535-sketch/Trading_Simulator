"""
Rebuild the Gap Open pack: down-bounce 2–6% with yesterday RSI(14) 45–70.

    python scripts/intraday_gap_rsi_pack.py
"""
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

from scripts.intraday_15m_n200_search import nifty200_symbols  # noqa: E402
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from scripts._gap_tp_entry_search import enrich_events, make_entry_signals  # noqa: E402
from scripts.intraday_gap_hunt import (  # noqa: E402
    OUT,
    TRADES,
    attach_rsi,
    build_sim_book,
    collect_events,
    months_from_trades,
    prepare,
    simulate_open,
)
from trading.services.intraday_gap import (  # noqa: E402
    ENTRY_HHMM,
    GAP_MAX,
    GAP_MIN,
    MAX_DEPLOY,
    MAX_POS,
    REQUIRE_BOUNCE,
    RISK_PCT,
    RSI_HI,
    RSI_LO,
    SL_ATR,
    STRATEGY,
    TOP_K,
    TP_KIND,
    load_gap_hunt,
    scan_gap_setups,
    tp_kind_label,
)
def main():
    symbols = nifty200_symbols()
    print(f"RSI 45–70 gap pack | {len(symbols)} names", flush=True)
    cache = load_frames(symbols)
    stocks = prepare(cache)
    sample = next(iter(stocks.values()))
    events = collect_events(stocks)
    print(f"events {len(events)}  attaching daily RSI…", flush=True)
    events = attach_rsi(events)
    rows = enrich_events(events, stocks)
    if ENTRY_HHMM.startswith("09:30") and REQUIRE_BOUNCE:
        entry_mode = "930_bounce"
    elif ENTRY_HHMM.startswith("09:30"):
        entry_mode = "930"
    else:
        entry_mode = "915"
    size = dict(risk_pct=RISK_PCT, max_pos=MAX_POS, top_k=TOP_K, max_deploy=MAX_DEPLOY)
    print(
        f"qualified {len(rows)}  entry {ENTRY_HHMM} ({entry_mode})  TP {TP_KIND} ({tp_kind_label()})",
        flush=True,
    )

    sigs = make_entry_signals(rows, entry_mode, TP_KIND)
    name = f"down_bounce {entry_mode} {TP_KIND} g{GAP_MIN:.0%}-{GAP_MAX:.0%} sl{SL_ATR}"
    res, tdf = simulate_open(stocks, sigs, name, book=build_sim_book(stocks), **size)
    months = months_from_trades(tdf)
    res.params = dict(
        mode="down_bounce",
        target=TP_KIND,
        entry=ENTRY_HHMM,
        bounce=REQUIRE_BOUNCE,
        gap_min=GAP_MIN,
        gap_max=GAP_MAX,
        rsi_lo=RSI_LO,
        rsi_hi=RSI_HI,
        sl_atr=SL_ATR,
        **size,
    )
    print(
        f"{res.total_return_pct:.1f}%  WR {res.win_rate:.1f}  PF {res.profit_factor:.2f}  "
        f"n={res.trades}  DD {res.max_dd_pct:.1f}  OOS {res.oos_pnl:.0f}"
    )
    print("months", months)

    prev = load_gap_hunt()
    payload = {
        "capital": 100_000.0,
        "timeframe": "5m",
        "universe": "nifty200",
        "window": {
            "start": str(sample.index.min()),
            "end": str(sample.index.max()),
            "stocks": len(stocks),
        },
        "idea": (
            f"Qualify 9:15 gap-downs {GAP_MIN:.0%}-{GAP_MAX:.0%} with RSI 45–70, "
            f"buy {ENTRY_HHMM} only if it holds the 9:15 close, target {tp_kind_label()}."
        ),
        "strategy": STRATEGY,
        "fill_stats": prev.get("fill_stats") or [],
        "story": {**asdict(res), "months": months},
        "result": {**asdict(res), "months": months},
        "winner": prev.get("winner") or {},
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if tdf is not None and not tdf.empty:
        tdf.to_json(TRADES, orient="records", date_format="iso")
    print("Wrote", OUT)
    print("Wrote", TRADES, f"({0 if tdf is None else len(tdf)} trades)")

    live = scan_gap_setups()
    print(
        f"live session {live['session']}  taken {live['counts']['trades']}  "
        f"qualified {live['counts']['qualified']}"
    )
    for t in live["taken"]:
        print(
            f"  TRADE {t['symbol']}  gap {t['gap_pct']}%  RSI {t['rsi']}  "
            f"entry {t['entry']}  sl {t['stop']}  tgt {t['target']}  qty {t['qty']}"
        )


if __name__ == "__main__":
    main()
