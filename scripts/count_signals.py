import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from datetime import date, timedelta
import pandas as pd
from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from scripts.strategy_search2 import sig_ema20_bounce_v2

end = date(2025, 6, 1)
start = end - timedelta(days=365)
config = StrategyConfig.get_active()
capital = 500_000
n = 0
for sym in get_universe_symbols():
    df = load_price_dataframe(sym)
    if df.empty:
        continue
    df = compute_indicators(df)
    if not has_sufficient_history(df):
        continue
    dates = df.index[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    for ts in dates:
        if sig_ema20_bounce_v2(df.loc[:ts], config, capital):
            n += 1
print("total signals", n)