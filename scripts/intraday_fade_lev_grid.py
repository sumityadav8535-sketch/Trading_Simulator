"""Fine grid: what leverage/deploy first crosses 100% on the 15m ATR fade."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from scripts.intraday_15m_hunt import load_frames, months_from_trades, prepare, simulate_sized
from scripts.intraday_15m_n200_phase3 import attach_index, sig_vwap_fade_v2
from scripts.intraday_15m_n200_search import build_exec_index, collect_signals, nifty200_symbols

cache = load_frames(nifty200_symbols())
stocks, idx = prepare(cache)
stocks = attach_index(stocks, idx)
exec_idx = build_exec_index(stocks)
sigs = collect_signals(
    stocks, sig_vwap_fade_v2,
    ext_min=0.018, adx_max=26, target_mode="atr", idx_abs_max=0.008,
)
print("signals", len(sigs))
print(f"{'lev':>4} {'dep':>5} {'ret%':>7} {'DD':>6} {'WR':>5} {'PnL':>9} months")
rows = []
for lev in (5.0, 6.0, 7.0, 8.0, 10.0):
    for dep in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        sz = dict(
            mode="risk", risk_pct=25.0, alloc_pct=0.0, leverage=lev,
            max_pos=6, top_k=2, max_deploy=dep, daily_lock=0.0, daily_halt=0.0,
        )
        res, tdf = simulate_sized(exec_idx, sigs, f"l{lev} d{dep}", **sz)
        months = months_from_trades(tdf)
        mtxt = " | ".join(f"{m['month'][-2:]} {m['pnl']:.0f}" for m in months)
        print(f"{lev:4g} {dep:5g} {res.total_return_pct:7.1f} {res.max_dd_pct:6.1f} {res.win_rate:5.1f} {res.net_pnl:9.0f}  {mtxt}")
        rows.append((lev, dep, res.total_return_pct, res.max_dd_pct, res.win_rate, res.net_pnl, months))

cross = [r for r in rows if r[2] >= 100]
print("\nFirst 100% crossings:")
for r in cross:
    print(f"  lev={r[0]} deploy={r[1]} ret={r[2]:.1f}% DD={r[3]:.1f}%")
