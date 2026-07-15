import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from datetime import date, timedelta
import pandas as pd
from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.swing_strategy import evaluate_swing_signal
from scripts.strategy_search2 import sig_ema20_bounce_v2

end = date(2025, 6, 1)
start = end - timedelta(days=365)
config = StrategyConfig.get_active()
capital = 500_000
n_new = n_old = 0
for sym in get_universe_symbols():
    df = load_price_dataframe(sym)
    if df.empty:
        continue
    df = compute_indicators(df)
    if not has_sufficient_history(df):
        continue
    for ts in df.index[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]:
        hist = df.loc[:ts]
        if evaluate_swing_signal(sym, ts.date(), config, capital, indicator_df=hist).is_valid:
            n_new += 1
        if sig_ema20_bounce_v2(hist, config, capital):
            n_old += 1
print("new strategy signals", n_new)
print("ema20_v2 signals", n_old)