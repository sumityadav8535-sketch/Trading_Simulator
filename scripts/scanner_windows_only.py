"""Scanner elite history across 9m/1y/2y/3y — fast report."""
import os, sys
from copy import copy
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from trading.models import StrategyConfig
from trading.services.elite_scanner_history import run_elite_scanner_history
from trading.services.market_data import get_universe_symbols

CAPITAL = 100_000.0
END = date.today()
WINDOWS = {
    "9m": END - timedelta(days=274),
    "1y": END - timedelta(days=365),
    "2y": END - timedelta(days=730),
    "3y": END - timedelta(days=1095),
}

symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
cfg = copy(StrategyConfig.get_active())
cfg.risk_pct = 2.0

print(f"Scanner Elite | Rs {CAPITAL:,.0f} | 2% risk | 2R | {len(symbols)} symbols\n")
print(f"{'Window':<6} {'Trades':>7} {'WR%':>7} {'Return':>9} {'PF':>6}")
for label, start in WINDOWS.items():
    h = run_elite_scanner_history(symbols, start, END, capital=CAPITAL, config=cfg)
    s = h.summary
    wins = [e.pnl for e in h.entries if e.pnl > 0]
    losses = [e.pnl for e in h.entries if e.pnl <= 0]
    gp, gl = sum(wins), abs(sum(losses)) or 1e-9
    pf = round(gp / gl, 2)
    print(f"{label:<6} {s['trades']:>7} {s['win_rate']:>6.1f}% {s['total_return_pct']:>8.2f}% {pf:>6.2f}")