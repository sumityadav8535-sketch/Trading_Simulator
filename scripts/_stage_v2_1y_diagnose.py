"""Diagnose why Stage Analysis 2.0 last-1-year returns are weak."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

import pandas as pd  # noqa: E402

from stage_analysis.services.stage_detector import daily_to_weekly  # noqa: E402
from stage_analysis_v2.services.backtester import (  # noqa: E402
    DEFAULT_TECH_FILTER,
    run_stage_v2_backtest,
)
from stage_analysis_v2.services.indicators import add_weekly_indicators  # noqa: E402
from stage_analysis_v2.services.stage_engine import detect_weekly_stage  # noqa: E402
from trading.constants import NIFTY50_SYMBOL  # noqa: E402
from trading.services.market_data import get_universe_symbols, load_price_dataframe  # noqa: E402


def nifty_stage_map(nifty: pd.DataFrame) -> list[dict]:
    weekly = add_weekly_indicators(daily_to_weekly(nifty))
    rows = []
    for ts in weekly.index:
        window = weekly.loc[weekly.index <= ts]
        try:
            st, _, metrics = detect_weekly_stage(window)
        except ValueError:
            continue
        rows.append({
            "date": ts.date(),
            "stage": int(st),
            "close": float(metrics.get("price") or weekly.loc[ts, "close"]),
            "ma": float(metrics.get("ma") or 0),
            "favorable": int(st) in (1, 2),
        })
    return rows


def stage_at(rows: list[dict], d: date) -> dict | None:
    last = None
    for r in rows:
        if r["date"] <= d:
            last = r
        else:
            break
    return last


def summarize(r, label: str) -> None:
    print(f"\n===== {label} =====")
    print(
        f"return={r.total_return_pct}%  WR={r.win_rate}%  PF={r.profit_factor}  "
        f"maxDD={r.max_drawdown_pct}%  trades={r.total_trades}  "
        f"signals={r.stage2_entries}  skipped_cash={r.signals_skipped_cash}  "
        f"peak_parallel={r.peak_parallel}  avg_hold={r.avg_hold_days}  avg_R={r.avg_rr}"
    )
    print(f"exits: {r.exit_breakdown}")


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)
    symbols = get_universe_symbols(nifty200_only=True)
    print(f"Window {start} → {end}  universe={len(symbols)}  tech={DEFAULT_TECH_FILTER}")

    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    nframe = nifty.loc[(nifty.index >= str(start)) & (nifty.index <= str(end))]
    n_open = float(nframe["close"].iloc[0])
    n_close = float(nframe["close"].iloc[-1])
    n_ret = (n_close / n_open - 1) * 100
    n_dd = float((nframe["close"] / nframe["close"].cummax() - 1).min() * 100)
    print(f"Nifty 50: {n_open:.0f} → {n_close:.0f}  {n_ret:+.2f}%  maxDD {n_dd:.1f}%")

    stages = nifty_stage_map(nifty)
    in_win = [s for s in stages if start <= s["date"] <= end]
    fav_weeks = sum(1 for s in in_win if s["favorable"])
    print(f"Nifty weekly stages in window: {len(in_win)} weeks, {fav_weeks} favorable (S1/S2) "
          f"({fav_weeks / len(in_win) * 100 if in_win else 0:.0f}%)")
    print("Week-by-week Nifty stage:")
    for s in in_win:
        flag = "OK" if s["favorable"] else "CHOP/BEAR"
        print(f"  {s['date']}  S{s['stage']}  {s['close']:.0f}  vsMA {((s['close']-s['ma'])/s['ma']*100 if s['ma'] else 0):+.1f}%  [{flag}]")

    print("\nRunning default Stage 2.0 (market_filter OFF, daily_mtf, stage_4_only)...")
    r = run_stage_v2_backtest(
        symbols=symbols,
        start_date=start,
        end_date=end,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
    )
    summarize(r, "DEFAULT (market filter OFF)")

    print("\nRunning WITH Nifty Stage 1/2 market filter...")
    r_mkt = run_stage_v2_backtest(
        symbols=symbols,
        start_date=start,
        end_date=end,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=True,
    )
    summarize(r_mkt, "MARKET FILTER ON")

    trades = r.trades
    print("\n===== ENTRY REGIME (default book) =====")
    by_reg = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    unfav_entries = []
    for t in trades:
        ed = date.fromisoformat(t.entry_date)
        st = stage_at(stages, ed)
        key = f"S{st['stage']}" if st else "unknown"
        by_reg[key]["n"] += 1
        by_reg[key]["pnl"] += t.pnl
        if t.pnl > 0:
            by_reg[key]["wins"] += 1
        if st and not st["favorable"]:
            unfav_entries.append((t, st))
    for k in sorted(by_reg):
        d = by_reg[k]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"  entered while Nifty {k}: n={d['n']} WR={wr:.0f}% PnL={d['pnl']:+,.0f}")
    print(f"  trades entered in unfavorable Nifty (S3/S4): {len(unfav_entries)}")

    print("\n===== EXIT REASON =====")
    by_reason = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0, "days": 0.0, "r": 0.0})
    for t in trades:
        d = by_reason[t.exit_reason]
        d["n"] += 1
        d["pnl"] += t.pnl
        d["days"] += t.days_held
        d["r"] += t.rr_achieved
        if t.pnl > 0:
            d["wins"] += 1
    for reason, d in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
        n = d["n"]
        wr = d["wins"] / n * 100 if n else 0
        print(
            f"  {reason:12s} n={n:3d} WR={wr:5.1f}% PnL={d['pnl']:+10,.0f} "
            f"avg={d['pnl']/n:+8,.0f} avg_days={d['days']/n:5.1f} avg_R={d['r']/n:+.2f}"
        )

    print("\n===== MONTHLY (by EXIT) vs Nifty =====")
    monthly = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0, "exits": Counter()})
    for t in trades:
        m = t.exit_date[:7]
        monthly[m]["pnl"] += t.pnl
        monthly[m]["n"] += 1
        monthly[m]["exits"][t.exit_reason] += 1
        if t.pnl > 0:
            monthly[m]["wins"] += 1
    mclose = nframe["close"].resample("ME").last()
    mret = mclose.pct_change() * 100
    nifty_m = {}
    for ts, ret in mret.items():
        nifty_m[ts.strftime("%Y-%m")] = None if ret != ret else float(ret)
    for m in sorted(set(list(monthly) + list(nifty_m))):
        d = monthly.get(m) or {"pnl": 0, "n": 0, "wins": 0, "exits": Counter()}
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        nr = nifty_m.get(m)
        nr_s = f"{nr:+.1f}%" if nr is not None else "—"
        print(
            f"  {m}: strat {d['pnl']:+8,.0f} ({d['n']}t WR={wr:.0f}%)  "
            f"Nifty {nr_s:>7}  exits={dict(d['exits'])}"
        )

    print("\n===== ENTRY MONTH =====")
    entry_m = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        m = t.entry_date[:7]
        entry_m[m]["n"] += 1
        entry_m[m]["pnl"] += t.pnl
        if t.pnl > 0:
            entry_m[m]["wins"] += 1
    for m in sorted(entry_m):
        d = entry_m[m]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"  {m}: entries={d['n']:3d} WR={wr:5.0f}% PnL={d['pnl']:+,.0f}")

    print("\n===== QUALITY / RS =====")
    buckets = [(0, 49, "Q0-49"), (50, 69, "Q50-69"), (70, 84, "Q70-84"), (85, 100, "Q85-100")]
    for lo, hi, name in buckets:
        ts = [t for t in trades if lo <= t.quality_score <= hi]
        if not ts:
            print(f"  {name}: none")
            continue
        wins = sum(1 for t in ts if t.pnl > 0)
        pnl = sum(t.pnl for t in ts)
        print(f"  {name}: n={len(ts)} WR={wins/len(ts)*100:.0f}% PnL={pnl:+,.0f}")
    rs_b = [(0, 49, "RS<50"), (50, 69, "RS50-69"), (70, 100, "RS>=70")]
    for lo, hi, name in rs_b:
        ts = [t for t in trades if lo <= t.rs_rating <= hi]
        if not ts:
            continue
        wins = sum(1 for t in ts if t.pnl > 0)
        pnl = sum(t.pnl for t in ts)
        print(f"  {name}: n={len(ts)} WR={wins/len(ts)*100:.0f}% PnL={pnl:+,.0f}")

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    print("\n===== WIN vs LOSS =====")
    for label, ts in (("Winners", winners), ("Losers", losers)):
        if not ts:
            continue
        print(
            f"  {label}: n={len(ts)} avgQ={sum(t.quality_score for t in ts)/len(ts):.1f} "
            f"avgRS={sum(t.rs_rating for t in ts)/len(ts):.1f} "
            f"avgDays={sum(t.days_held for t in ts)/len(ts):.1f} "
            f"avgR={sum(t.rr_achieved for t in ts)/len(ts):.2f} "
            f"avgPnL={sum(t.pnl for t in ts)/len(ts):+,.0f} "
            f"exits={dict(Counter(t.exit_reason for t in ts))}"
        )
    if winners and losers:
        gp = sum(t.pnl for t in winners)
        gl = abs(sum(t.pnl for t in losers))
        print(f"  gross profit {gp:,.0f} vs gross loss {gl:,.0f}  (avg win {gp/len(winners):,.0f} / avg loss {gl/len(losers):,.0f})")
        print(f"  payoff ratio (avg win/avg loss) {(gp/len(winners)) / (gl/len(losers) or 1e-9):.2f}")

    print("\n===== WORST 12 LOSSES =====")
    for t in sorted(losers, key=lambda x: x.pnl)[:12]:
        st = stage_at(stages, date.fromisoformat(t.entry_date))
        ns = f"Nifty S{st['stage']}" if st else ""
        print(
            f"  {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.pnl:8,.0f}  "
            f"{t.pnl_pct:+6.1f}%  R={t.rr_achieved:+.2f}  Q{t.quality_score} RS{t.rs_rating:.0f}  "
            f"{t.exit_reason:12s} {t.days_held}d  {ns}"
        )

    print("\n===== BEST 8 WINS =====")
    for t in sorted(winners, key=lambda x: -x.pnl)[:8]:
        print(
            f"  {t.symbol:12s} {t.entry_date}→{t.exit_date}  {t.pnl:8,.0f}  "
            f"{t.pnl_pct:+6.1f}%  R={t.rr_achieved:+.2f}  {t.exit_reason} {t.days_held}d"
        )

    # time-stop: were they still working or already dead?
    time_ex = [t for t in trades if t.exit_reason == "time_exit"]
    if time_ex:
        tw = sum(1 for t in time_ex if t.pnl > 0)
        print(
            f"\n===== TIME STOP (65d) ===== n={len(time_ex)} WR={tw/len(time_ex)*100:.0f}% "
            f"PnL={sum(t.pnl for t in time_ex):+,.0f}"
        )
        print(
            f"  positive time-exits: {sum(1 for t in time_ex if t.pnl>0)}  "
            f"still below 1R: {sum(1 for t in time_ex if t.rr_achieved < 1)}"
        )

    print("\n===== CASH / CAPACITY =====")
    print(f"  signals {r.stage2_entries}  taken {r.total_trades}  skipped no-cash {r.signals_skipped_cash}")
    if r.stage2_entries:
        print(f"  take rate {r.total_trades / r.stage2_entries * 100:.0f}%")

    print("\n===== FILTER IMPACT =====")
    print(
        f"  default (no mkt filter): {r.total_return_pct}% WR {r.win_rate}% "
        f"on {r.total_trades} trades"
    )
    print(
        f"  Nifty S1/S2 only:        {r_mkt.total_return_pct}% WR {r_mkt.win_rate}% "
        f"on {r_mkt.total_trades} trades"
    )
    print(f"  Nifty buy-and-hold:      {n_ret:+.2f}%")


if __name__ == "__main__":
    main()
