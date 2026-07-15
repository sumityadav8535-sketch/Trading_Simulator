"""Targeted EMA20 strong/engulf combos — fast follow-up to coarse search."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from scripts.ema20_enhance_backtest import START, END, build_cache, backtest, make_ema20_strong
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols

TARGETS = [
    ("E: ADX20 (winner)", dict(adx_min=20)),
    ("C: tol 1.5%", dict(ema_tol=0.015)),
    ("E+C: ADX20 + tol1.5%", dict(adx_min=20, ema_tol=0.015)),
    ("E+C+3HL", dict(adx_min=20, ema_tol=0.015, hl_bars=3)),
    ("E+C+3HL+RSI47-57", dict(adx_min=20, ema_tol=0.015, hl_bars=3, rsi_lo=47, rsi_hi=57)),
    ("E+C+3HL+RSI48-57", dict(adx_min=20, ema_tol=0.015, hl_bars=3, rsi_lo=48, rsi_hi=57)),
    ("ADX21+tol1.4+3HL", dict(adx_min=21, ema_tol=0.014, hl_bars=3)),
    ("ADX21+tol1.5+3HL", dict(adx_min=21, ema_tol=0.015, hl_bars=3)),
    ("ADX22+tol1.5+3HL", dict(adx_min=22, ema_tol=0.015, hl_bars=3)),
    ("ADX20+tol1.4+3HL", dict(adx_min=20, ema_tol=0.014, hl_bars=3)),
    ("ADX20+tol1.3+3HL", dict(adx_min=20, ema_tol=0.013, hl_bars=3)),
    ("ADX20+tol1.5+3HL+RSI49-57", dict(adx_min=20, ema_tol=0.015, hl_bars=3, rsi_lo=49, rsi_hi=57)),
    ("ADX20+tol1.5+4HL", dict(adx_min=20, ema_tol=0.015, hl_bars=4)),
    ("BASE", dict()),
]

def main():
    config = StrategyConfig.get_active()
    cache = build_cache(get_universe_symbols(nifty200_only=True))
    print(f"Window {START} to {END} | {len(cache)} stocks\n", flush=True)
    results = []
    for name, kw in TARGETS:
        r = backtest(name, make_ema20_strong(**kw), cache, config)
        results.append(r)
        print(f"{name:30s} n={r.trades:3d} WR={r.win_rate:5.1f}% PF={r.profit_factor:.2f} ret={r.return_pct:5.2f}% {r.exits}", flush=True)
    print("\nWR>=50%:", flush=True)
    for r in sorted([x for x in results if x.win_rate >= 50 and x.trades >= 5], key=lambda x: -x.trades):
        print(f"  {r.name}: n={r.trades} WR={r.win_rate}% PF={r.profit_factor}", flush=True)

if __name__ == "__main__":
    main()