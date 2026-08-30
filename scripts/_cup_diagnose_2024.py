"""Why Cup Breakout was ~flat in 2024 vs +172% in 2023."""
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

from stage_analysis_v2.services.cup_breakout import (
    CUP_EXIT_EMA20,
    CUP_EXIT_MEASURED,
    CUP_EXIT_TARGET_R,
    CupParams,
    run_cup_breakout_backtest,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe

PACK = CupParams(
    cup_min_days=20,
    cup_max_days=180,
    min_depth_pct=12.0,
    max_depth_pct=45.0,
    min_bottom_days=5,
    min_left_days=7,
    min_recovery_days=5,
    pivot_width=5,
    require_sma200_rising=False,
    require_rs_vs_nifty=False,
    require_close_strength=False,
    require_trend_stack=True,
    vol_mult=1.1,
    rsi_min=40.0,
    rsi_max=85.0,
    max_new_per_day=10,
    cup_exit_mode=CUP_EXIT_EMA20,
)


def run(start, end, **kw):
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    params = kw.pop("params", PACK)
    return run_cup_breakout_backtest(
        symbols=symbols,
        start_date=start,
        end_date=end,
        capital=1_000_000.0,
        risk_pct=kw.get("risk_pct", 10.0),
        max_hold_days=kw.get("max_hold_days", 40),
        cooldown_days=kw.get("cooldown_days", 0),
        max_pos_pct=kw.get("max_pos_pct", 100.0),
        params=params,
    )


def summarize(label, r):
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    print(f"\n===== {label} {r.start_date} → {r.end_date} =====")
    print(
        f"ret={r.total_return_pct:+.1f}% n={r.total_trades} sig={r.stage2_entries} "
        f"WR={r.win_rate}% PF={r.profit_factor} DD={r.max_drawdown_pct}% "
        f"avg_hold={r.avg_hold_days} avg_R={r.avg_rr} avg_trade={r.avg_trade:,.0f}"
    )
    print(f"exits={r.exit_breakdown}")
    if wins:
        print(
            f"  wins  n={len(wins)}  sum={sum(t.pnl for t in wins):,.0f}  "
            f"avg={sum(t.pnl for t in wins)/len(wins):,.0f}  "
            f"avg_R={sum(t.rr_achieved for t in wins)/len(wins):.2f}  "
            f"avg_hold={sum(t.days_held for t in wins)/len(wins):.1f}  "
            f"avg_pnl_pct={sum(t.pnl_pct for t in wins)/len(wins):.1f}%"
        )
    if losses:
        print(
            f"  loss  n={len(losses)}  sum={sum(t.pnl for t in losses):,.0f}  "
            f"avg={sum(t.pnl for t in losses)/len(losses):,.0f}  "
            f"avg_R={sum(t.rr_achieved for t in losses)/len(losses):.2f}  "
            f"avg_hold={sum(t.days_held for t in losses)/len(losses):.1f}  "
            f"avg_pnl_pct={sum(t.pnl_pct for t in losses)/len(losses):.1f}%"
        )

    by_m = defaultdict(lambda: {"pnl": 0.0, "n": 0, "w": 0, "ex": Counter()})
    for t in r.trades:
        m = t.exit_date[:7]
        by_m[m]["pnl"] += t.pnl
        by_m[m]["n"] += 1
        by_m[m]["w"] += int(t.pnl > 0)
        by_m[m]["ex"][t.exit_reason] += 1
    print("  monthly (exit month):")
    for m in sorted(by_m):
        d = by_m[m]
        wr = 100.0 * d["w"] / d["n"] if d["n"] else 0
        print(f"    {m}  {d['pnl']:+8,.0f}  n={d['n']:2d} WR={wr:4.0f}%  {dict(d['ex'])}")

    print("  worst 8 losses:")
    for t in sorted(r.trades, key=lambda x: x.pnl)[:8]:
        s = t.setup or {}
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.days_held:2d}d  "
            f"{t.exit_reason:12s}  {t.pnl:+8,.0f} ({t.pnl_pct:+5.1f}%  {t.rr_achieved:+.2f}R)  "
            f"depth={s.get('cup_depth_pct')} dur={s.get('cup_duration')} vol={s.get('volume_multiple')} "
            f"rsi={s.get('rsi')} Q={t.quality_score}"
        )
    print("  best 5 wins:")
    for t in sorted(r.trades, key=lambda x: x.pnl, reverse=True)[:5]:
        s = t.setup or {}
        print(
            f"    {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.days_held:2d}d  "
            f"{t.exit_reason:12s}  {t.pnl:+8,.0f} ({t.pnl_pct:+5.1f}%  {t.rr_achieved:+.2f}R)  "
            f"depth={s.get('cup_depth_pct')} dur={s.get('cup_duration')} vol={s.get('volume_multiple')}"
        )
    return r


def nifty_regime():
    df = load_price_dataframe(NIFTY50_SYMBOL)
    if df.empty:
        print("No Nifty data")
        return
    c = df["close"]
    print("\n===== Nifty 50 regime =====")
    for year in (2023, 2024, 2025):
        sl = c[(c.index >= f"{year}-01-01") & (c.index <= f"{year}-12-31")]
        if len(sl) < 20:
            continue
        ret = (float(sl.iloc[-1]) / float(sl.iloc[0]) - 1) * 100
        # max dd
        peak = sl.iloc[0]
        dd = 0.0
        worst_m = None
        monthly = sl.resample("ME").last().pct_change() * 100
        for px in sl:
            peak = max(peak, px)
            dd = min(dd, (px / peak - 1) * 100)
        print(f"  {year}: {float(sl.iloc[0]):.0f} → {float(sl.iloc[-1]):.0f}  ret={ret:+.1f}%  maxDD={dd:.1f}%")
        print("    months:", " ".join(f"{i.strftime('%b')}={v:+.1f}%" for i, v in monthly.dropna().items()))


def main():
    nifty_regime()
    r23 = summarize("2023 pack", run(date(2023, 1, 1), date(2023, 12, 31)))
    r24 = summarize("2024 pack", run(date(2024, 1, 1), date(2024, 12, 31)))

    print("\n===== 2024 same signals, other exits =====")
    from dataclasses import replace
    alts = [
        ("2024 meas r8 h45", 8.0, 45, replace(PACK, cup_exit_mode=CUP_EXIT_MEASURED)),
        ("2024 2R r10 h90", 10.0, 90, replace(PACK, cup_exit_mode=CUP_EXIT_TARGET_R, target_rr=2.0)),
        ("2024 3R r10 h90", 10.0, 90, replace(PACK, cup_exit_mode=CUP_EXIT_TARGET_R, target_rr=3.0)),
        ("2024 ema20 h90", 10.0, 90, PACK),
        ("2024 ema20 r6", 6.0, 40, PACK),
        ("2024 ema20 + 5d cd", 10.0, 40, PACK),
    ]
    for label, risk, hold, params in alts:
        cd = 5 if "cd" in label else 0
        r = run(date(2024, 1, 1), date(2024, 12, 31), risk_pct=risk, max_hold_days=hold, cooldown_days=cd, params=params)
        print(
            f"  {label:22s} ret={r.total_return_pct:+6.1f}% WR={r.win_rate:5.1f}% "
            f"PF={r.profit_factor:5.2f} DD={r.max_drawdown_pct:5.1f}% n={r.total_trades:3d} "
            f"hold={r.avg_hold_days} exits={r.exit_breakdown}"
        )

    print("\n===== 2023 with those same alts (don't break the 100% year) =====")
    for label, risk, hold, params in alts:
        cd = 5 if "cd" in label else 0
        r = run(date(2023, 1, 1), date(2023, 12, 31), risk_pct=risk, max_hold_days=hold, cooldown_days=cd, params=params)
        print(
            f"  {label.replace('2024','2023'):22s} ret={r.total_return_pct:+6.1f}% WR={r.win_rate:5.1f}% "
            f"PF={r.profit_factor:5.2f} DD={r.max_drawdown_pct:5.1f}% n={r.total_trades:3d}"
        )


if __name__ == "__main__":
    main()
