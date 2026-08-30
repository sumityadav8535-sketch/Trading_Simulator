"""Refine winning tech filter combos for Stage 2.0."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from scripts._stage_v2_tech_rank_search import (
    apply_pred,
    collect_rich_signals,
    simulate,
)
from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import _preload_frames
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.tech_filters import enrich_daily_tech
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)
    capital = 1_000_000.0
    risk_pct = float(getattr(StrategyConfig.get_active(), "risk_pct", 2.0) or 2.0)
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nw = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    st, et = pd.Timestamp(start), pd.Timestamp(end)
    wb, tb = {}, {}
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < 40:
            continue
        wb[sym] = w
        tb[sym] = enrich_daily_tech(d)
    sigs = collect_rich_signals(frames, wb, tb, nw, st, et)
    cal = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })

    def run(name, pred, rank="quality"):
        bd = apply_pred(sigs, pred)
        r = simulate(frames, wb, bd, cal, capital, risk_pct, start, rank_key=rank)
        print(
            f"{name:<55} n={r['n']:3d} WR={r['wr']:5.1f}% "
            f"ret={r['ret']:+6.1f}% PF={r['pf']:.2f} DD={r['dd']:.1f}%"
        )
        return r

    print("Winner combos:")
    run("BASE", lambda s: True)
    run("daily S1|S2", lambda s: s["daily_stage"] in (1, 2))
    run("not ext 8%", lambda s: s["not_extended"])
    run("daily S1|S2 + not ext", lambda s: s["daily_stage"] in (1, 2) and s["not_extended"])
    run("daily S1|S2 + BB mid", lambda s: s["daily_stage"] in (1, 2) and s["bb_above_mid"])
    run("daily S1|S2 + EMA stack", lambda s: s["daily_stage"] in (1, 2) and s["ema_stack_bull"])
    run("daily S1|S2 + FT", lambda s: s["daily_stage"] in (1, 2) and s["follow_through"])
    run("daily S1|S2 + RSI not OB", lambda s: s["daily_stage"] in (1, 2) and s["rsi_not_overbought"])
    run(
        "daily S1|S2 + BB + not ext",
        lambda s: s["daily_stage"] in (1, 2) and s["bb_above_mid"] and s["not_extended"],
    )
    run("daily S1|S2 + rank", lambda s: s["daily_stage"] in (1, 2), "rank")
    run("BB+FT", lambda s: s["bb_above_mid"] and s["follow_through"])
    run(
        "daily S1|S2 + BB + FT",
        lambda s: s["daily_stage"] in (1, 2) and s["bb_above_mid"] and s["follow_through"],
    )
    run(
        "daily S1|S2 + EMA + not ext",
        lambda s: s["daily_stage"] in (1, 2) and s["ema_stack_bull"] and s["not_extended"],
    )
    run(
        "daily S1|S2 + not ext + RSI ok",
        lambda s: s["daily_stage"] in (1, 2) and s["not_extended"] and s["rsi_not_overbought"],
    )


if __name__ == "__main__":
    main()
