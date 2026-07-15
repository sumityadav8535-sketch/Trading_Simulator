"""Quick tune for signal count vs win rate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from datetime import date, timedelta
from trading.services.signal_backtester import run_signal_backtest
from trading.services.market_data import get_universe_symbols

end = date(2025, 6, 1)
start = end - timedelta(days=365)
r = run_signal_backtest(get_universe_symbols(), start, end, 500_000)
print(f"signals={r.total_signals} trades={r.total_trades} wr={r.win_rate}% pf={r.profit_factor} ret={r.total_return_pct}% dd={r.max_drawdown_pct}% exp={r.expectancy_r}R")
print("exits", r.exit_breakdown)