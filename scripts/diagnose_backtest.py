"""Diagnose why 2022-2025 backtest has few signals."""
import os, sys
from datetime import date
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

import pandas as pd
from trading.models import StrategyConfig, DailyPrice, Stock
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.signal_backtester import run_signal_backtest
from trading.services.market_regime import load_proxy_frames, is_market_bullish
from trading.services.swing_strategy import evaluate_swing_signal
from django.db.models import Min, Max

START, END = date(2022, 6, 2), date(2025, 6, 1)
CAPITAL = 100_000

print("=== DATA ===")
print("Date range in DB:", DailyPrice.objects.aggregate(mn=Min("date"), mx=Max("date")))
print("Nifty200 stocks:", Stock.objects.filter(is_nifty200=True, is_active=True).count())

syms = get_universe_symbols(nifty200_only=True)
cache = {}
for sym in syms:
    df = load_price_dataframe(sym)
    if df.empty:
        continue
    df = compute_indicators(df)
    if has_sufficient_history(df):
        cache[sym] = df
print(f"Stocks with sufficient history: {len(cache)}")

# Regime ON days in period
proxies = load_proxy_frames()
regime_days = 0
total_days = 0
for sym, full in list(cache.items())[:1]:
    dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
    total_days = len(dates)
    for ts in dates:
        if is_market_bullish(proxies, ts):
            regime_days += 1
    break
print(f"Nifty50 regime ON days: {regime_days}/{total_days} ({regime_days/total_days*100:.1f}%)")

# Count rejection reasons (sample scan)
config = StrategyConfig.get_active()
rejections = Counter()
valid = 0
elite = 0
v2 = 0
for sym, full in cache.items():
    dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
    for ts in dates:
        hist = full.loc[:ts]
        row = hist.iloc[-1]
        if pd.isna(row.get("ema_200")) or float(row["close"]) <= float(row["ema_200"]):
            rejections["below_200ema"] += 1
            continue
        r = evaluate_swing_signal(sym, eval_date=ts.date(), config=config, capital=CAPITAL,
                                  indicator_df=hist, proxy_frames=proxies, require_market_filter=True)
        if r.is_valid:
            valid += 1
            if r.entry_path == "Elite 20 EMA":
                elite += 1
            else:
                v2 += 1
        else:
            reason = r.rejection_reasons[0] if r.rejection_reasons else "unknown"
            rejections[reason] += 1

print(f"\n=== RAW SIGNAL SCAN (no cooldown/position limits) ===")
print(f"Valid signals: {valid} (Elite={elite}, v2={v2})")
print("Top rejections:")
for k, v in rejections.most_common(10):
    print(f"  {k}: {v}")

r = run_signal_backtest(syms, START, END, capital=CAPITAL, config=config)
print(f"\n=== FULL BACKTEST (with cooldown, 1 pos/symbol) ===")
print(f"Signals: {r.total_signals}  Trades: {r.total_trades}  WR: {r.win_rate}%")
print(f"Return: {r.total_return_pct}%  PF: {r.profit_factor}")
print("Exit breakdown:", r.exit_breakdown)
if r.signal_history:
    paths = Counter(s.entry_path for s in r.signal_history)
    print("Paths:", dict(paths))