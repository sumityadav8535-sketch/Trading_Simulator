import os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

import scripts.ema20_enhance_backtest as m
from scripts.ema20_enhance_backtest import build_cache, backtest, make_ema20_strong
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import run_signal_backtest

START, END = date(2022, 6, 2), date(2025, 6, 1)
m.CAPITAL = 100_000
cache = build_cache(get_universe_symbols(nifty200_only=True))
cfg = StrategyConfig.get_active()

r1 = backtest("elite-no-regime", make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3), cache, cfg, start=START, end=END)
live = run_signal_backtest(get_universe_symbols(nifty200_only=True), START, END, capital=100_000, config=cfg)

print("Script (elite, NO regime filter):", r1.trades, "trades", r1.win_rate, "% WR", r1.return_pct, "% ret")
print("Live app backtest (Nifty50 regime + elite + v2):", live.total_signals, "signals", live.total_trades, "trades", live.win_rate, "% WR", live.total_return_pct, "% ret")