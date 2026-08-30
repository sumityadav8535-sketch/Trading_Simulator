"""Stage Analysis 2.0 — 1-year backtest deep dive (monthly failure analysis)."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols, load_price_dataframe


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)
    symbols = get_universe_symbols(nifty200_only=True)

    print(f"Running Stage 2.0 backtest | {start} → {end} | {len(symbols)} symbols")
    r = run_stage_v2_backtest(
        symbols=symbols,
        start_date=start,
        end_date=end,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
    )
    print(
        f"Done: scanned={r.stocks_scanned} signals={r.stage2_entries} "
        f"trades={r.total_trades} return={r.total_return_pct}% wr={r.win_rate}% "
        f"PF={r.profit_factor} maxDD={r.max_drawdown_pct}%"
    )
    print(f"Exit breakdown: {r.exit_breakdown}")
    print(f"Peak parallel: {r.peak_parallel} | Skipped no cash: {r.signals_skipped_cash}")

    # ── Monthly by EXIT month ──
    monthly: dict[str, dict] = defaultdict(
        lambda: {
            "pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "by_exit": Counter(),
            "losers": [],
            "avg_q_win": [],
            "avg_q_loss": [],
            "avg_hold_win": [],
            "avg_hold_loss": [],
            "avg_r_win": [],
            "avg_r_loss": [],
        }
    )
    for t in r.trades:
        m = t.exit_date[:7]
        d = monthly[m]
        d["pnl"] += t.pnl
        d["trades"] += 1
        d["by_exit"][t.exit_reason] += 1
        if t.pnl > 0:
            d["wins"] += 1
            d["avg_q_win"].append(t.quality_score)
            d["avg_hold_win"].append(t.days_held)
            d["avg_r_win"].append(t.rr_achieved)
        else:
            d["losses"] += 1
            d["losers"].append(t)
            d["avg_q_loss"].append(t.quality_score)
            d["avg_hold_loss"].append(t.days_held)
            d["avg_r_loss"].append(t.rr_achieved)

    print("\n=== MONTHLY DETAIL (by exit month) ===")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        qwin = sum(d["avg_q_win"]) / len(d["avg_q_win"]) if d["avg_q_win"] else 0
        qloss = sum(d["avg_q_loss"]) / len(d["avg_q_loss"]) if d["avg_q_loss"] else 0
        sign = "+" if d["pnl"] >= 0 else ""
        print(
            f"{m}: PnL {sign}{d['pnl']:,.0f} | trades={d['trades']} "
            f"W/L={d['wins']}/{d['losses']} WR={wr:.0f}% | Qwin={qwin:.0f} Qloss={qloss:.0f}"
        )
        print(f"     exits: {dict(d['by_exit'])}")
        if d["losers"]:
            for t in sorted(d["losers"], key=lambda x: x.pnl)[:4]:
                print(
                    f"     LOSS {t.symbol:12s} entry={t.entry_date} exit={t.exit_date} "
                    f"Q{t.quality_score} RS{t.rs_rating:.0f} pnl={t.pnl:,.0f} "
                    f"reason={t.exit_reason} days={t.days_held} R={t.rr_achieved}"
                )

    # ── Exit reason performance ──
    print("\n=== EXIT REASON PERFORMANCE ===")
    by_reason: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in r.trades:
        by_reason[t.exit_reason]["n"] += 1
        by_reason[t.exit_reason]["pnl"] += t.pnl
        if t.pnl > 0:
            by_reason[t.exit_reason]["wins"] += 1
    for reason, d in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(
            f"{reason:15s} n={d['n']:3d} WR={wr:5.1f}% "
            f"PnL={d['pnl']:+,.0f} avg={d['pnl'] / d['n']:+,.0f}"
        )

    # ── Quality buckets ──
    print("\n=== QUALITY BUCKET PERFORMANCE ===")
    buckets = [(0, 49, "Q0-49"), (50, 74, "Q50-74"), (75, 84, "Q75-84"), (85, 100, "Q85-100")]
    for lo, hi, name in buckets:
        ts = [t for t in r.trades if lo <= t.quality_score <= hi]
        if not ts:
            print(f"{name}: no trades")
            continue
        wins = sum(1 for t in ts if t.pnl > 0)
        pnl = sum(t.pnl for t in ts)
        print(
            f"{name}: n={len(ts)} WR={wins / len(ts) * 100:.0f}% "
            f"PnL={pnl:+,.0f} avg={pnl / len(ts):+,.0f}"
        )

    # ── Entry month ──
    print("\n=== ENTRY MONTH PERFORMANCE (when trade opened) ===")
    entry_m: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "wins": 0, "stage_exits": 0, "stops": 0, "time": 0}
    )
    for t in r.trades:
        m = t.entry_date[:7]
        entry_m[m]["n"] += 1
        entry_m[m]["pnl"] += t.pnl
        if t.pnl > 0:
            entry_m[m]["wins"] += 1
        if t.exit_reason == "stage_exit":
            entry_m[m]["stage_exits"] += 1
        if t.exit_reason == "stop_loss":
            entry_m[m]["stops"] += 1
        if t.exit_reason == "time_exit":
            entry_m[m]["time"] += 1
    for m in sorted(entry_m.keys()):
        d = entry_m[m]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(
            f"{m}: entries={d['n']} WR={wr:.0f}% PnL={d['pnl']:+,.0f} "
            f"stage_exits={d['stage_exits']} stops={d['stops']} time_exits={d['time']}"
        )

    # ── Nifty monthly ──
    print("\n=== NIFTY MONTHLY CONTEXT ===")
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    if not nifty.empty:
        mask = (nifty.index >= str(start)) & (nifty.index <= str(end))
        n = nifty.loc[mask]
        if not n.empty:
            mclose = n["close"].resample("ME").last()
            mret = mclose.pct_change() * 100
            for ts, ret in mret.items():
                mon = ts.strftime("%Y-%m")
                if ret != ret:  # NaN
                    print(f"Nifty {mon}: first month (baseline close {mclose.loc[ts]:.0f})")
                else:
                    print(f"Nifty {mon}: {ret:+.1f}% (close {mclose.loc[ts]:.0f})")

    # ── Winner vs loser stats ──
    print("\n=== WINNER vs LOSER CHARACTERISTICS ===")
    losers = [t for t in r.trades if t.pnl <= 0]
    winners = [t for t in r.trades if t.pnl > 0]

    def stats(label, ts):
        if not ts:
            print(f"{label}: none")
            return
        print(
            f"{label}: n={len(ts)} avg_Q={sum(t.quality_score for t in ts) / len(ts):.1f} "
            f"avg_RS={sum(t.rs_rating for t in ts) / len(ts):.1f} "
            f"avg_days={sum(t.days_held for t in ts) / len(ts):.1f} "
            f"avg_R={sum(t.rr_achieved for t in ts) / len(ts):.2f} "
            f"avg_pnl={sum(t.pnl for t in ts) / len(ts):+,.0f}"
        )
        print(f"  exit reasons: {dict(Counter(t.exit_reason for t in ts))}")

    stats("Winners", winners)
    stats("Losers", losers)

    # ── Stage-exit losses (failed advances) ──
    print("\n=== STAGE-EXIT LOSSES (failed Stage 2 advances) — worst 15 ===")
    stage_losses = sorted(
        [t for t in losers if t.exit_reason == "stage_exit"], key=lambda x: x.pnl
    )[:15]
    for t in stage_losses:
        print(
            f"  {t.symbol:12s} Q{t.quality_score:3d} RS{t.rs_rating:5.0f} "
            f"held={t.days_held:2d}d entry={t.entry_date} exit={t.exit_date} "
            f"pnl={t.pnl:+,.0f} R={t.rr_achieved}"
        )

    # ── Stop losses ──
    print("\n=== STOP-LOSS TRADES ===")
    stops = sorted([t for t in r.trades if t.exit_reason == "stop_loss"], key=lambda x: x.pnl)
    for t in stops:
        print(
            f"  {t.symbol:12s} Q{t.quality_score:3d} RS{t.rs_rating:5.0f} "
            f"entry={t.entry_date} exit={t.exit_date} pnl={t.pnl:+,.0f} "
            f"entry_px={t.entry_price} stop={t.stop_loss}"
        )

    # ── Worst months ──
    print("\n=== WORST MONTHS (exit-month P&L) ===")
    worst = sorted(monthly.items(), key=lambda x: x[1]["pnl"])[:5]
    for m, d in worst:
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(
            f"{m}: {d['pnl']:+,.0f} trades={d['trades']} WR={wr:.0f}% "
            f"stage_exits={d['by_exit'].get('stage_exit', 0)} "
            f"stops={d['by_exit'].get('stop_loss', 0)} "
            f"time={d['by_exit'].get('time_exit', 0)} "
            f"target={d['by_exit'].get('target_2.5r', 0)}"
        )

    # ── Best months ──
    print("\n=== BEST MONTHS ===")
    best = sorted(monthly.items(), key=lambda x: x[1]["pnl"], reverse=True)[:5]
    for m, d in best:
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(
            f"{m}: {d['pnl']:+,.0f} trades={d['trades']} WR={wr:.0f}% "
            f"targets={d['by_exit'].get('target_2.5r', 0)} "
            f"time={d['by_exit'].get('time_exit', 0)}"
        )

    # ── Hold-to-loss pattern: short-lived stage exits ──
    print("\n=== FAST FAILURES (held <= 10 days, loss) ===")
    fast = sorted(
        [t for t in losers if t.days_held <= 10], key=lambda x: x.pnl
    )
    print(f"Count: {len(fast)}")
    for t in fast[:12]:
        print(
            f"  {t.symbol:12s} Q{t.quality_score:3d} held={t.days_held}d "
            f"{t.entry_date}→{t.exit_date} pnl={t.pnl:+,.0f} {t.exit_reason}"
        )

    # ── Capital / parallel stress ──
    print("\n=== NOTES ===")
    print(
        f"Signals={r.stage2_entries} but trades={r.total_trades} "
        f"(skipped cash={r.signals_skipped_cash}, cooldown/already open also filter)"
    )
    print("Done.")


if __name__ == "__main__":
    main()
