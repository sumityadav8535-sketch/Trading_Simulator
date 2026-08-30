"""
Analyze Stage Analysis 2.0 losses in Oct 2024 – Feb 2025 window.

Compares:
  - Nifty price path + weekly stage (market regime)
  - Default backtest trades entering/exiting in that window
  - Impact of market_filter (Nifty Stage 1/2 only)
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from stage_analysis_v2.services.backtester import (
    DEFAULT_EXIT_MODE,
    DEFAULT_TECH_FILTER,
    run_stage_v2_backtest,
)
from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.indicators import add_weekly_indicators
from stage_analysis_v2.services.stage_engine import detect_weekly_stage
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe


WINDOW_START = date(2024, 10, 1)
WINDOW_END = date(2025, 2, 28)
# Slightly wider for context
CTX_START = date(2024, 9, 1)
CTX_END = date(2025, 3, 15)


def nifty_regime() -> None:
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    if nifty is None or nifty.empty:
        print("ERROR: No Nifty price data")
        return

    weekly = add_weekly_indicators(daily_to_weekly(nifty))
    w = weekly.loc[
        (weekly.index >= pd.Timestamp(CTX_START))
        & (weekly.index <= pd.Timestamp(CTX_END))
    ]

    print("=" * 70)
    print("NIFTY 50 WEEKLY STAGE (Sep 2024 – mid Mar 2025)")
    print("=" * 70)
    for ts, _row in w.iterrows():
        window = weekly.loc[weekly.index <= ts]
        try:
            st, _, metrics = detect_weekly_stage(window)
            price = float(metrics.get("price") or 0)
            ma = float(metrics.get("ma") or 0)
            fav = st in (1, 2)
            flag = "OK" if fav else "RISK"
            print(
                f"  {ts.date()}  Stage {st}  close={price:,.0f}  "
                f"ma30w={ma:,.0f}  favorable={fav}  [{flag}]"
            )
        except Exception as e:
            print(f"  {ts.date()}  err: {e}")

    d = nifty.loc[
        (nifty.index >= pd.Timestamp(CTX_START))
        & (nifty.index <= pd.Timestamp(CTX_END))
    ]
    if d.empty:
        return

    peak_idx = d["close"].idxmax()
    after = d.loc[peak_idx:]
    trough_idx = after["close"].idxmin()
    peak = float(d["close"].max())
    trough = float(after["close"].min())
    start_px = float(d.iloc[0]["close"])
    end_px = float(d.iloc[-1]["close"])
    oct_start = d.loc[d.index >= pd.Timestamp("2024-10-01")]
    feb_end = d.loc[d.index <= pd.Timestamp("2025-02-28")]
    oct_px = float(oct_start.iloc[0]["close"]) if len(oct_start) else start_px
    feb_px = float(feb_end.iloc[-1]["close"]) if len(feb_end) else end_px

    print()
    print("=" * 70)
    print("NIFTY PRICE PATH")
    print("=" * 70)
    print(f"  Sep 2024 open-ish: {start_px:,.0f}")
    print(f"  Peak: {peak_idx.date()} @ {peak:,.0f}")
    print(f"  Trough after peak: {trough_idx.date()} @ {trough:,.0f}")
    print(f"  Peak → trough drawdown: {(trough / peak - 1) * 100:.1f}%")
    print(f"  Oct 1 ~ close: {oct_px:,.0f}  →  Feb 28 ~ close: {feb_px:,.0f}")
    print(f"  Oct–Feb change: {(feb_px / oct_px - 1) * 100:.1f}%")

    m = d["close"].resample("ME").last()
    print()
    print("  Monthly close / return:")
    prev = None
    for ts, val in m.items():
        v = float(val)
        ret = ((v / prev - 1) * 100) if prev else 0.0
        print(f"    {ts.strftime('%Y-%m')}: {v:,.0f}  ({ret:+.1f}%)")
        prev = v


def summarize_trades(trades, label: str) -> None:
    # Trades that overlap the pain window (entry or exit inside window)
    win_s = WINDOW_START.isoformat()
    win_e = WINDOW_END.isoformat()

    def in_window(d: str) -> bool:
        return win_s <= d <= win_e

    entered = [t for t in trades if in_window(t.entry_date)]
    exited = [t for t in trades if in_window(t.exit_date)]
    # Active during window: entered before end and exited after start
    active = [
        t
        for t in trades
        if t.entry_date <= win_e and t.exit_date >= win_s
    ]

    print()
    print("=" * 70)
    print(f"STRATEGY TRADES — {label}")
    print("=" * 70)
    print(f"  Total trades in full run: {len(trades)}")
    print(f"  Entered in window: {len(entered)}")
    print(f"  Exited in window:  {len(exited)}")
    print(f"  Active anytime in window: {len(active)}")

    for name, subset in (
        ("ENTERED Oct24–Feb25", entered),
        ("EXITED Oct24–Feb25", exited),
        ("ACTIVE in window", active),
    ):
        if not subset:
            print(f"\n  [{name}] none")
            continue
        wins = [t for t in subset if t.pnl > 0]
        losses = [t for t in subset if t.pnl <= 0]
        pnl = sum(t.pnl for t in subset)
        wr = 100.0 * len(wins) / len(subset) if subset else 0
        print(f"\n  [{name}] n={len(subset)}  WR={wr:.0f}%  PnL=₹{pnl:,.0f}")
        print(f"    wins={len(wins)} losses={len(losses)}")
        print(f"    exit reasons: {dict(Counter(t.exit_reason for t in subset))}")
        if losses:
            avg_q_l = sum(t.quality_score for t in losses) / len(losses)
            avg_hold_l = sum(t.days_held for t in losses) / len(losses)
            avg_r_l = sum(t.rr_achieved for t in losses) / len(losses)
            print(
                f"    losers avg Q={avg_q_l:.0f} hold={avg_hold_l:.0f}d R={avg_r_l:.2f}"
            )
        if wins:
            avg_q_w = sum(t.quality_score for t in wins) / len(wins)
            print(f"    winners avg Q={avg_q_w:.0f}")

        # By entry month
        by_m: dict[str, list] = defaultdict(list)
        for t in subset:
            by_m[t.entry_date[:7]].append(t)
        print("    by entry month:")
        for m in sorted(by_m):
            ts = by_m[m]
            p = sum(x.pnl for x in ts)
            w = sum(1 for x in ts if x.pnl > 0)
            print(f"      {m}: n={len(ts)} W={w} L={len(ts)-w} PnL=₹{p:,.0f}")

        # Worst 10 losers
        worst = sorted(subset, key=lambda t: t.pnl)[:10]
        print("    worst trades:")
        for t in worst:
            print(
                f"      {t.symbol:12} entry={t.entry_date} exit={t.exit_date} "
                f"PnL=₹{t.pnl:,.0f} ({t.pnl_pct}%) R={t.rr_achieved} "
                f"Q={t.quality_score} hold={t.days_held}d  {t.exit_reason}"
            )


def main() -> None:
    nifty_regime()

    symbols = get_universe_symbols(nifty200_only=True)
    # Full 3y context like user (2023–2026) so portfolio state is realistic
    full_start = date(2023, 1, 1)
    full_end = date(2026, 2, 28)

    print()
    print("=" * 70)
    print(
        f"Running default backtest {full_start} → {full_end} "
        f"({len(symbols)} symbols) — may take a few minutes…"
    )
    print("=" * 70)

    r = run_stage_v2_backtest(
        symbols=symbols,
        start_date=full_start,
        end_date=full_end,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print(
        f"  Full run: signals={r.stage2_entries} trades={r.total_trades} "
        f"return={r.total_return_pct}% WR={r.win_rate}% "
        f"PF={r.profit_factor} maxDD={r.max_drawdown_pct}%"
    )
    print(f"  Exit breakdown: {r.exit_breakdown}")
    summarize_trades(r.trades, "DEFAULT (no market filter)")

    # Monthly PnL for full run (by exit month) around the window
    monthly: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "n": 0, "w": 0})
    for t in r.trades:
        m = t.exit_date[:7]
        monthly[m]["pnl"] += t.pnl
        monthly[m]["n"] += 1
        if t.pnl > 0:
            monthly[m]["w"] += 1
    print()
    print("=" * 70)
    print("MONTHLY PnL BY EXIT MONTH (2024-07 → 2025-06)")
    print("=" * 70)
    for m in sorted(monthly):
        if "2024-07" <= m <= "2025-06":
            d = monthly[m]
            wr = 100 * d["w"] / d["n"] if d["n"] else 0
            print(f"  {m}: n={d['n']:3d} WR={wr:5.1f}%  PnL=₹{d['pnl']:>12,.0f}")

    # Compare with market filter ON
    print()
    print("=" * 70)
    print("Re-run WITH market_filter=True (Nifty Stage 1/2 only)…")
    print("=" * 70)
    r2 = run_stage_v2_backtest(
        symbols=symbols,
        start_date=full_start,
        end_date=full_end,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=True,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print(
        f"  Market-filter run: signals={r2.stage2_entries} trades={r2.total_trades} "
        f"return={r2.total_return_pct}% WR={r2.win_rate}% "
        f"PF={r2.profit_factor} maxDD={r2.max_drawdown_pct}%"
    )
    summarize_trades(r2.trades, "MARKET FILTER ON")

    # Window-only standalone backtest (fresh capital) for purity
    print()
    print("=" * 70)
    print("Standalone window backtest (fresh capital, default settings)")
    print("=" * 70)
    r3 = run_stage_v2_backtest(
        symbols=symbols,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print(
        f"  Window-only: signals={r3.stage2_entries} trades={r3.total_trades} "
        f"return={r3.total_return_pct}% WR={r3.win_rate}% "
        f"PF={r3.profit_factor} maxDD={r3.max_drawdown_pct}%"
    )
    print(f"  Exit breakdown: {r3.exit_breakdown}")
    if r3.trades:
        wins = sum(1 for t in r3.trades if t.pnl > 0)
        print(f"  Wins={wins} Losses={len(r3.trades)-wins}")
        print("  Top losers:")
        for t in sorted(r3.trades, key=lambda x: x.pnl)[:15]:
            print(
                f"    {t.symbol:12} {t.entry_date}→{t.exit_date} "
                f"₹{t.pnl:,.0f} ({t.pnl_pct}%) Q={t.quality_score} {t.exit_reason}"
            )

    r4 = run_stage_v2_backtest(
        symbols=symbols,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=True,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print()
    print(
        f"  Window-only + market filter: signals={r4.stage2_entries} "
        f"trades={r4.total_trades} return={r4.total_return_pct}% WR={r4.win_rate}% "
        f"PF={r4.profit_factor} maxDD={r4.max_drawdown_pct}%"
    )

    # Higher quality filter on window
    r5 = run_stage_v2_backtest(
        symbols=symbols,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
        capital=1_000_000.0,
        min_quality_score=75,
        market_filter=False,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print(
        f"  Window-only + Q≥75: signals={r5.stage2_entries} "
        f"trades={r5.total_trades} return={r5.total_return_pct}% WR={r5.win_rate}% "
        f"PF={r5.profit_factor} maxDD={r5.max_drawdown_pct}%"
    )

    r6 = run_stage_v2_backtest(
        symbols=symbols,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
        capital=1_000_000.0,
        min_quality_score=75,
        market_filter=True,
        exit_mode=DEFAULT_EXIT_MODE,
        tech_filter=DEFAULT_TECH_FILTER,
    )
    print(
        f"  Window-only + Q≥75 + mkt filter: signals={r6.stage2_entries} "
        f"trades={r6.total_trades} return={r6.total_return_pct}% WR={r6.win_rate}% "
        f"PF={r6.profit_factor} maxDD={r6.max_drawdown_pct}%"
    )


if __name__ == "__main__":
    main()
