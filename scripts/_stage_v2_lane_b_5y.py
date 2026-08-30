"""
Lane B 5-year research backtest (not wired into the app).

Stage 2.0 entries, daily Stage 1/2, RS≥70, not extended ≤8% EMA20.
Exit: +8% take-profit or −15% hard stop. No Stage 4 force-close. 65-day time stop.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

import pandas as pd

from _stage_v2_high_wr_search import (
    CAPITAL,
    apply_pred,
    collect,
    simulate,
)
from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import MIN_WEEKLY_BARS, _preload_frames
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.tech_filters import enrich_daily_tech
from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def _cagr(start_eq: float, end_eq: float, years: float) -> float:
    if start_eq <= 0 or end_eq <= 0 or years <= 0:
        return 0.0
    return (end_eq / start_eq) ** (1.0 / years) * 100.0 - 100.0


def main() -> None:
    end = date.today()
    start = end - timedelta(days=int(5 * 365))
    years = (end - start).days / 365.25
    capital = CAPITAL
    risk_pct = float(getattr(StrategyConfig.get_active(), "risk_pct", 2.0) or 2.0)
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]

    print("=" * 68)
    print("LANE B — 5 year backtest (research only)")
    print("=" * 68)
    print(f"Period:     {start} → {end}  ({years:.2f} years)")
    print(f"Universe:   Nifty 200  ({len(symbols)} symbols)  capital Rs {capital:,.0f}")
    print("Entry:      weekly Stage 2 transition + daily Stage 1/2")
    print("Filters:    RS ≥ 70  AND  not extended (≤8% above EMA20)")
    print("Target:     +8% of entry")
    print("Stop:       −15% hard stop (intraday low)")
    print("Stage exit: OFF")
    print("Time stop:  65 trading days")
    print("Sizing:     2% equity risk, cash-constrained")
    print("Loading data...")

    frames = _preload_frames(symbols)
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nw = add_weekly_indicators(daily_to_weekly(nifty)) if not nifty.empty else pd.DataFrame()
    st, et = pd.Timestamp(start), pd.Timestamp(end)
    weekly_by_sym, tech_by_sym = {}, {}
    print("Weekly + tech...")
    for sym, d in frames.items():
        w = add_weekly_indicators(daily_to_weekly(d))
        if len(w) < MIN_WEEKLY_BARS + 2:
            continue
        weekly_by_sym[sym] = w
        tech_by_sym[sym] = enrich_daily_tech(d)
    print(f"Stocks with history: {len(weekly_by_sym)}")
    print("Collecting Stage 2 signals (this takes a few minutes)...")
    sigs = collect(frames, weekly_by_sym, tech_by_sym, nw, st, et)
    raw = len(sigs)
    lane = [
        s for s in sigs
        if s["daily_stage"] in (1, 2) and s["rs_rating"] >= 70 and s["not_extended"]
    ]
    print(f"Stage 2 transitions in window: {raw}")
    print(f"After Lane B filters:          {len(lane)}")

    calendar = sorted({
        ts for df in frames.values()
        for ts in df.index[(df.index >= st) & (df.index <= et)].tolist()
    })
    bd = apply_pred(
        sigs,
        lambda s: s["daily_stage"] in (1, 2) and s["rs_rating"] >= 70 and s["not_extended"],
    )
    r = simulate(
        frames, weekly_by_sym, bd, calendar,
        capital=capital, risk_pct=risk_pct, start_date=start,
        profit_pct=0.08, stop_mode="pct", stop_pct=0.15, stage_exit=False,
    )
    trades = r["trades"]
    cagr = _cagr(capital, r["final"], years)

    print("")
    print("=" * 68)
    print("RESULTS — Lane B last 5 years")
    print("=" * 68)
    print(f"Trades executed:   {r['n']}")
    print(f"Win rate:          {r['wr']}%")
    print(f"Profit factor:     {r['pf']}")
    print(f"Total return:      {r['ret']}%")
    print(f"CAGR:              {cagr:.2f}%")
    print(f"Max drawdown:      {r['dd']}%")
    print(f"Avg hold (days):   {r['avg_hold']}")
    print(f"Hit +8% target:    {r['hits']} / {r['n']} ({(r['hits'] / r['n'] * 100) if r['n'] else 0:.1f}%)")
    print(f"Hard stops (−15%): {r['stops']}")
    print(f"Final equity:      Rs {r['final']:,.0f}")
    print(f"Gross profit:      Rs {r['gp']:,.0f}")
    print(f"Gross loss:        Rs {r['gl']:,.0f}")
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        print(f"Avg win:           Rs {(r['gp'] / len(wins)) if wins else 0:,.0f}  ({len(wins)} wins)")
        print(f"Avg loss:          Rs {(r['gl'] / len(losses)) if losses else 0:,.0f}  ({len(losses)} losses)")
        pcts = [t.pnl_pct for t in trades]
        print(f"Avg trade %:       {sum(pcts) / len(pcts):+.2f}%")

    print("")
    print("Exit breakdown:")
    by_reason: dict[str, list] = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t)
    for reason, ts in sorted(by_reason.items(), key=lambda kv: -sum(x.pnl for x in kv[1])):
        pnl = sum(x.pnl for x in ts)
        wr = sum(1 for x in ts if x.pnl > 0) / len(ts) * 100
        hold = sum(x.days_held for x in ts) / len(ts)
        print(
            f"  {reason:12} n={len(ts):3d}  WR={wr:5.1f}%  "
            f"PnL {pnl:+,.0f}  avg hold {hold:.1f}d"
        )

    print("")
    print("Yearly (by exit date):")
    yearly: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0, "hits": 0, "stops": 0})
    for t in trades:
        y = t.exit_date[:4]
        yearly[y]["pnl"] += t.pnl
        yearly[y]["n"] += 1
        yearly[y]["wins"] += int(t.pnl > 0)
        yearly[y]["hits"] += int(t.exit_reason == "target")
        yearly[y]["stops"] += int(str(t.exit_reason).startswith("stop"))
    for y in sorted(yearly):
        row = yearly[y]
        wr = row["wins"] / row["n"] * 100 if row["n"] else 0
        print(
            f"  {y}: n={row['n']:3d}  WR={wr:5.1f}%  "
            f"ret {row['pnl'] / capital * 100:+6.1f}%  "
            f"PnL {row['pnl']:+,.0f}  "
            f"TP {row['hits']}  SL {row['stops']}"
        )

    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        monthly[t.exit_date[:7]] += t.pnl
    if monthly:
        print("")
        print("Monthly P&L:")
        red = 0
        for m in sorted(monthly):
            pnl = monthly[m]
            if pnl < 0:
                red += 1
            sign = "+" if pnl >= 0 else ""
            print(f"  {m}: {sign}{pnl:,.0f} ({sign}{pnl / capital * 100:.2f}%)")
        print(f"  Red months: {red} / {len(monthly)}")

    if trades:
        print("")
        print("Top 15 winners:")
        for t in sorted(trades, key=lambda x: x.pnl, reverse=True)[:15]:
            print(
                f"  {t.symbol:12} {t.entry_date}→{t.exit_date}  "
                f"{t.entry_price:.2f}→{t.exit_price:.2f}  "
                f"{t.pnl_pct:+6.1f}%  PnL {t.pnl:+,.0f}  "
                f"{t.days_held:3d}d  {t.exit_reason}"
            )
        print("")
        print("Top 15 losers:")
        for t in sorted(trades, key=lambda x: x.pnl)[:15]:
            print(
                f"  {t.symbol:12} {t.entry_date}→{t.exit_date}  "
                f"{t.entry_price:.2f}→{t.exit_price:.2f}  "
                f"{t.pnl_pct:+6.1f}%  PnL {t.pnl:+,.0f}  "
                f"{t.days_held:3d}d  {t.exit_reason}"
            )


if __name__ == "__main__":
    main()
