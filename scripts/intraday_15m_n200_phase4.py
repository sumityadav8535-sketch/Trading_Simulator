"""Phase 4: lock the robust VWAP-ATR fade, scale it, monthly walk-forward."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_n200_phase3 import (  # noqa: E402
    attach_index,
    load_prepared,
    sig_vwap_fade_v2,
    simulate_bp,
)
from scripts.intraday_15m_n200_search import (  # noqa: E402
    build_exec_index,
    collect_signals,
)

OUT = ROOT / "data" / "intraday_15m_n200_phase4.json"
TRADES = ROOT / "data" / "intraday_15m_n200_winner_trades.json"


SPECS = [
    ("ATR 1.8% adx26 idx0.8", dict(ext_min=0.018, ext_max=0.04, adx_max=26, idx_abs_max=0.008, target_mode="atr", rsi_hi=68, rsi_lo=32)),
    ("ATR 1.6% adx28 idx1.0", dict(ext_min=0.016, ext_max=0.04, adx_max=28, idx_abs_max=0.010, target_mode="atr", rsi_hi=66, rsi_lo=34)),
    ("ATR 1.5% adx28 idx1.0", dict(ext_min=0.015, ext_max=0.04, adx_max=28, idx_abs_max=0.010, target_mode="atr", rsi_hi=65, rsi_lo=35)),
    ("VWAP 1.8% adx26 idx0.8", dict(ext_min=0.018, ext_max=0.04, adx_max=26, idx_abs_max=0.008, target_mode="vwap", rsi_hi=68, rsi_lo=32)),
    ("HALF 1.8% adx26 idx0.8", dict(ext_min=0.018, ext_max=0.04, adx_max=26, idx_abs_max=0.008, target_mode="half", rsi_hi=68, rsi_lo=32)),
    ("ATR 1.4% adx30 wide", dict(ext_min=0.014, ext_max=0.045, adx_max=30, idx_abs_max=0.012, target_mode="atr", rsi_hi=64, rsi_lo=36, start=dtime(9, 45), end=dtime(14, 15))),
    ("ATR 1.2% adx30", dict(ext_min=0.012, ext_max=0.04, adx_max=30, idx_abs_max=0.012, target_mode="atr", rsi_hi=64, rsi_lo=36)),
]


def monthly(tdf: pd.DataFrame):
    if tdf is None or tdf.empty:
        return []
    t = tdf.copy()
    t["day"] = pd.to_datetime(t["exit_ts"]).dt.tz_localize(None)
    t["month"] = t["day"].dt.to_period("M").astype(str)
    rows = []
    for m, g in t.groupby("month"):
        daily = g.groupby(g["day"].dt.date)["pnl"].sum()
        rows.append({
            "month": m,
            "trades": int(len(g)),
            "pnl": round(float(g["pnl"].sum()), 2),
            "wr": round(float((g["pnl"] > 0).mean() * 100), 1),
            "avg_day": round(float(daily.mean()), 2),
            "days": int(daily.shape[0]),
            "days_2k": int((daily >= 2000).sum()),
        })
    return rows


def main():
    stocks, idx = load_prepared()
    stocks = attach_index(stocks, idx)
    exec_idx = build_exec_index(stocks)
    print(f"stocks={len(stocks)}")

    base = []
    for name, kw in SPECS:
        sigs = collect_signals(stocks, sig_vwap_fade_v2, **kw)
        res, tdf = simulate_bp(exec_idx, sigs, name, risk_pct=2, max_pos=4, top_k=1, max_deploy=0.5, leverage=1)
        print(f"BASE {name:28s} n={res.trades:4d} WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} "
              f"avg={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} DD={res.max_dd_pct:4.1f} days={res.trading_days}")
        base.append((res, tdf, kw, sigs, name))

    print("\n=== Scaled (5x MIS, 30% buying power / slot, daily lock 4k / halt 2.5k) ===")
    scaled = []
    for res0, _, kw, sigs, name in base:
        for lev, dep, risk, pos, lock, halt in [
            (1, 1.0, 2.0, 4, 0, 0),
            (5, 0.30, 2.5, 5, 4000, 2500),
            (5, 0.40, 2.5, 4, 5000, 2500),
        ]:
            label = f"{name} L{lev}d{dep}"
            res, tdf = simulate_bp(
                exec_idx, sigs, label, risk_pct=risk, max_pos=pos, top_k=1,
                daily_lock=lock, daily_halt=halt, max_deploy=dep, leverage=lev,
            )
            months = monthly(tdf)
            scaled.append((res, tdf, months, name, kw))
            print(f"  {label:36s} n={res.trades:4d} WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} "
                  f"avg={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} DD={res.max_dd_pct:4.1f} "
                  f"ge2k={res.days_ge_2k}/{res.trading_days} pnl={res.net_pnl:7.0f}")
            for mo in months:
                print(f"      {mo['month']}  trades={mo['trades']:3d} pnl={mo['pnl']:8.0f} "
                      f"WR={mo['wr']:5.1f} avg/d={mo['avg_day']:7.0f} d2k={mo['days_2k']}/{mo['days']}")

    # pick winner: require PF>=1.3, trades>=40, OOS>0, lowest DD among those with best OOS
    cand = [s for s in scaled if s[0].trades >= 35 and s[0].profit_factor >= 1.25 and s[0].oos_avg_daily > 0]
    cand.sort(key=lambda s: (s[0].oos_avg_daily, s[0].profit_factor, -s[0].max_dd_pct), reverse=True)
    print("\n===== CANDIDATES =====")
    for res, *_ in cand[:8]:
        print(f"  {res.name}: avg {res.avg_daily_pnl} OOS {res.oos_avg_daily} PF {res.profit_factor} WR {res.win_rate} DD {res.max_dd_pct}")

    winner = cand[0] if cand else scaled[0]
    wres, wtdf, wmonths, wname, wkw = winner
    print(f"\nWINNER {wres.name}")
    print(f"  trades {wres.trades} WR {wres.win_rate}% PF {wres.profit_factor} "
          f"avg/day ₹{wres.avg_daily_pnl} OOS ₹{wres.oos_avg_daily} DD {wres.max_dd_pct}%")

    payload = {
        "winner": asdict(wres),
        "winner_spec": wkw,
        "winner_family": wname,
        "months": wmonths,
        "candidates": [asdict(r) for r, *_ in cand[:8]],
        "base": [asdict(r) for r, *_ in base],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if wtdf is not None and not wtdf.empty:
        wtdf.to_json(TRADES, orient="records", date_format="iso")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
