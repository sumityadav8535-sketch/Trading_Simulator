"""Save the 100% 15m fade pack (lev 5, deploy 2.5)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from scripts.intraday_15m_hunt import load_frames, months_from_trades, prepare, simulate_sized
from scripts.intraday_15m_n200_phase3 import attach_index, sig_vwap_fade_v2
from scripts.intraday_15m_n200_search import CAPITAL, build_exec_index, collect_signals, nifty200_symbols

OUT = ROOT / "data" / "intraday_100pct.json"
TRADES = ROOT / "data" / "intraday_100pct_trades.json"

cache = load_frames(nifty200_symbols())
stocks, idx = prepare(cache)
stocks = attach_index(stocks, idx)
exec_idx = build_exec_index(stocks)
sigs = collect_signals(
    stocks, sig_vwap_fade_v2,
    ext_min=0.018, adx_max=26, target_mode="atr", idx_abs_max=0.008,
)
sz = dict(
    mode="risk", risk_pct=25.0, alloc_pct=0.0, leverage=5.0,
    max_pos=6, top_k=2, max_deploy=2.5, daily_lock=0.0, daily_halt=0.0,
)
res, tdf = simulate_sized(exec_idx, sigs, "VWAP fade 1.8% ATR | 100pct pack", **sz)
months = months_from_trades(tdf)
payload = {
    "capital": CAPITAL,
    "timeframe": "15m",
    "universe": "nifty200",
    "note": (
        "Same VWAP fade as the 5% pack. 100% comes from size: 5x leverage * 2.5 deploy "
        "≈ 12.5x notional per name. Cash MIS is typically 5x, so this is oversized vs a "
        "normal equity intraday account. Win rate 55%, max DD ~30%, July-heavy."
    ),
    "notional_per_trade_vs_equity": 12.5,
    "winner": {**asdict(res), "months": months, "params": {**res.params, "ext_min": 0.018, "adx_max": 26, "target_mode": "atr", "idx_abs_max": 0.008}},
    "mis5x_compare": {"deploy": 1.0, "return_pct": 38.4, "dd_pct": 12.8},
}
OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
if tdf is not None and not tdf.empty:
    tdf.to_json(TRADES, orient="records", date_format="iso")
print(res.total_return_pct, res.max_dd_pct, res.win_rate, res.net_pnl, months)
print("wrote", OUT)
