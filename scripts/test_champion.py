import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from scripts.strategy_tournament import (
    START, END, CAPITAL, COOLDOWN, MAX_HOLD, build_cache, backtest_strategy,
    s_ema20_strong, s_ema20_v2, s_ema50_stack, s_ema20_hl3, s_dual_pullback,
)
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols

def champion_priority(hist, config, capital):
    for fn in (s_ema20_strong, s_ema20_hl3, s_ema20_v2, s_ema50_stack):
        sig = fn(hist, config, capital)
        if sig:
            return sig
    return None

config = StrategyConfig.get_active()
cache = build_cache(get_universe_symbols())
print(f"stocks={len(cache)}")
for name, fn in [
    ("Champion priority", champion_priority),
    ("Dual 20+50", s_dual_pullback),
    ("EMA20 strong", s_ema20_strong),
    ("EMA20 3bar HL", s_ema20_hl3),
]:
    r = backtest_strategy(name, fn, cache, config)
    print(f"{name}: sig={r.signals} trades={r.trades} WR={r.win_rate}% PF={r.profit_factor} ret={r.return_pct}%")