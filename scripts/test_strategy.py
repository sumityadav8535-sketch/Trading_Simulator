"""Quick strategy backtest analysis script."""
import os
import sys
from collections import Counter
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django
django.setup()

from trading.services.backtester import run_backtest
from trading.services.market_data import get_universe_symbols

end = date(2025, 6, 1)
start = end - timedelta(days=365)
symbols = get_universe_symbols(nifty200_only=True)
print(f"Symbols: {len(symbols)}, range: {start} to {end}")

bt = run_backtest(symbols, start, end, 500_000)
print(f"trades={bt.total_trades} wr={bt.win_rate}% pf={bt.profit_factor} ret={bt.total_return_pct}% avg_rr={bt.avg_rr}")
print("Exit reasons:", Counter(t.exit_reason for t in bt.trades))