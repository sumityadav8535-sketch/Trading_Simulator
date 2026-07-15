"""Analyze ML F&O trade losses and search improved filters."""
from __future__ import annotations

import json
import statistics as stats
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADES_PATH = ROOT / "data" / "intraday_fno_ml_trades.json"


def hour_of(signal_time: str) -> int:
    return int(signal_time[11:13])


def main():
    data = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
    trades = data["full_period"]["trades"]
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    print(f"Total: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Net P&L: Rs {sum(t['pnl_inr'] for t in trades):,.0f}")
    print()
    print("Exit reasons (wins):", Counter(t["exit_reason"] for t in wins))
    print("Exit reasons (losses):", Counter(t["exit_reason"] for t in losses))
    print()
    for label, subset in [("WIN", wins), ("LOSS", losses)]:
        print(f"--- {label} averages ---")
        print(f"  ml_prob: {stats.mean(t['ml_prob'] for t in subset if t['ml_prob']):.3f}")
        print(f"  rsi: {stats.mean(t['rsi'] for t in subset):.1f}")
        print(f"  adx: {stats.mean(t['adx'] for t in subset if t['adx']):.1f}")
        print(f"  range_pos: {stats.mean(t['range_pos'] for t in subset if t['range_pos']):.3f}")
        print(f"  risk_pts: {stats.mean(t['risk_pts'] for t in subset):.1f}")
        print(f"  signal_close vs ema21: {stats.mean(t['signal_close'] - t['ema_21'] for t in subset):.1f}")

    print("\n--- ML prob buckets ---")
    for lo, hi in [(0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.65), (0.65, 1.0)]:
        bucket = [t for t in trades if t["ml_prob"] and lo <= t["ml_prob"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        print(f"  {lo:.2f}-{hi:.2f}: n={len(bucket):3d} WR={wr:5.1f}% P&L=Rs {pnl:>9,.0f}")

    print("\n--- Hour buckets (IST) ---")
    for h in range(9, 15):
        bucket = [t for t in trades if hour_of(t["signal_time"]) == h]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        losses_h = sum(1 for t in bucket if t["result"] == "LOSS")
        print(f"  {h:02d}xx: n={len(bucket):3d} WR={wr:5.1f}% losses={losses_h:3d} P&L=Rs {pnl:>9,.0f}")

    print("\n--- ADX buckets ---")
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 35), (35, 100)]:
        bucket = [t for t in trades if t["adx"] and lo <= t["adx"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        print(f"  ADX {lo}-{hi}: n={len(bucket):3d} WR={wr:5.1f}% P&L=Rs {pnl:>9,.0f}")

    print("\n--- range_pos buckets ---")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        bucket = [t for t in trades if t["range_pos"] is not None and lo <= t["range_pos"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        print(f"  pos {lo}-{hi}: n={len(bucket):3d} WR={wr:5.1f}% P&L=Rs {pnl:>9,.0f}")

    print("\n--- RSI buckets ---")
    for lo, hi in [(40, 45), (45, 50), (50, 55), (55, 65)]:
        bucket = [t for t in trades if lo <= t["rsi"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        print(f"  RSI {lo}-{hi}: n={len(bucket):3d} WR={wr:5.1f}% P&L=Rs {pnl:>9,.0f}")

    print("\n--- risk_pts buckets ---")
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 100)]:
        bucket = [t for t in trades if lo <= t["risk_pts"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket) * 100
        pnl = sum(t["pnl_inr"] for t in bucket)
        losses_b = sum(1 for t in bucket if t["result"] == "LOSS")
        print(f"  risk {lo}-{hi}: n={len(bucket):3d} WR={wr:5.1f}% losses={losses_b} P&L=Rs {pnl:>9,.0f}")

    # Combo filters on historical trades
    print("\n--- Filter simulations on executed trades ---")
    filters = [
        ("baseline", lambda t: True),
        ("ml>=0.40", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.40),
        ("ml>=0.45", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.45),
        ("ml>=0.50", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.50),
        ("adx>=20", lambda t: t["adx"] and t["adx"] >= 20),
        ("adx>=25", lambda t: t["adx"] and t["adx"] >= 25),
        ("range_pos>=0.5", lambda t: t["range_pos"] is not None and t["range_pos"] >= 0.5),
        ("range_pos<=0.5", lambda t: t["range_pos"] is not None and t["range_pos"] <= 0.5),
        ("hour 10-13", lambda t: 10 <= hour_of(t["signal_time"]) <= 13),
        ("no hour 14", lambda t: hour_of(t["signal_time"]) < 14),
        ("risk<=15", lambda t: t["risk_pts"] <= 15),
        ("risk<=20", lambda t: t["risk_pts"] <= 20),
        ("ema50 below", lambda t: t["ema_21"] < t["ema_50"]),
        ("rsi 42-52", lambda t: 42 <= t["rsi"] <= 52),
        ("ml>=0.45+adx>=20", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.45 and t["adx"] and t["adx"] >= 20),
        ("ml>=0.45+risk<=15", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.45 and t["risk_pts"] <= 15),
        ("ml>=0.45+hour10-13", lambda t: t["ml_prob"] and t["ml_prob"] >= 0.45 and 10 <= hour_of(t["signal_time"]) <= 13),
        ("ml>=0.50+adx>=20+risk<=15", lambda t: (
            t["ml_prob"] and t["ml_prob"] >= 0.50 and t["adx"] and t["adx"] >= 20 and t["risk_pts"] <= 15
        )),
    ]
    rows = []
    for name, fn in filters:
        sub = [t for t in trades if fn(t)]
        if len(sub) < 5:
            continue
        w = sum(1 for t in sub if t["result"] == "WIN")
        l = len(sub) - w
        pnl = sum(t["pnl_inr"] for t in sub)
        wr = w / len(sub) * 100
        rows.append((pnl, wr, l, len(sub), name))
    rows.sort(reverse=True)
    for pnl, wr, l, n, name in rows[:20]:
        print(f"  {name:<30} n={n:3d} W={n-l:3d} L={l:3d} WR={wr:5.1f}% P&L=Rs {pnl:>9,.0f}")


if __name__ == "__main__":
    main()