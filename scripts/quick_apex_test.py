import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from datetime import date, timedelta
from scripts.optimize_strategy import Params, backtest_params
from trading.services.market_data import get_universe_symbols

end = date(2025, 6, 1)
start = end - timedelta(days=365)
symbols = get_universe_symbols(nifty200_only=True)

for label, p in [
    ("apex_default", Params()),
    ("apex_strict", Params(adx_min=28, min_score=9, cooldown_days=20, rsi_low=48, rsi_high=56)),
    ("apex_no_market", Params(min_score=8)),
]:
    n, wr, pf, ret = backtest_params(p, symbols, start, end)
    print(f"{label}: trades={n} wr={wr:.1f}% pf={pf:.2f} ret={ret:.1f}%")