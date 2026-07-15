import os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import run_signal_backtest

r = run_signal_backtest(
    get_universe_symbols(nifty200_only=True),
    date(2026, 1, 1), date(2026, 7, 2),
    capital=100_000,
    config=StrategyConfig.get_active(),
)
print(f"trades={r.total_trades} signals={r.total_signals} wr={r.win_rate}% ret={r.total_return_pct}%")
for t in r.trades:
    print(f"\n{t.symbol} signal={t.signal_date} entry={t.entry_date}@{t.entry_price} exit={t.exit_date}@{t.exit_price}")
    print(f"  stop={t.stop_loss} target={t.target} reason={t.exit_reason} pnl={t.pnl} rr={t.rr_achieved}")
    print(f"  path reasons: {t.reasons}")
for s in r.signal_history:
    print(f"\nSIG {s.symbol} {s.signal_date} path={s.entry_path} outcome={s.outcome} days={s.days_held}")
    print(f"  DETAIL: {s.outcome_detail}")