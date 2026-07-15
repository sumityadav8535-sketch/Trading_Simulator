"""Check market regime on Jan 2026 losing trade signal dates — read-only analysis."""
import os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from trading.services.market_regime import get_market_regime_status, load_proxy_frames, is_market_bullish
import pandas as pd

SIGNAL_DATES = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
TRADES = [
    ("ASIANPAINT", date(2026, 1, 5)),
    ("SUNPHARMA", date(2026, 1, 6)),
    ("INFY", date(2026, 1, 7)),
]

proxies = load_proxy_frames()
print("=== REGIME ON SIGNAL DATES ===\n")
for d in SIGNAL_DATES:
    ts = pd.Timestamp(d)
    bullish = sum(
        1 for sym, df in proxies.items()
        if not (h := df.loc[:ts]).empty
        and h.iloc[-1].get("ema_50")
        and float(h.iloc[-1]["close"]) > float(h.iloc[-1]["ema_50"])
    )
    print(f"{d}: {bullish}/5 proxies above 50 EMA — {'ON' if bullish >= 3 else 'OFF'}")
    for sym, df in proxies.items():
        h = df.loc[:ts]
        if h.empty:
            continue
        r = h.iloc[-1]
        c, e50 = float(r["close"]), float(r["ema_50"])
        e20 = float(r["ema_20"]) if pd.notna(r.get("ema_20")) else 0
        above = c > e50
        print(f"  {sym:10s} close={c:8.1f} 50EMA={e50:8.1f} {'ABOVE' if above else 'BELOW'}  20EMA={e20:8.1f}")

print("\n=== PER-TRADE STOCK TREND ON SIGNAL DAY ===\n")
from trading.services.market_data import load_price_dataframe
from trading.services.indicators import compute_indicators

for sym, d in TRADES:
    df = compute_indicators(load_price_dataframe(sym))
    row = df.loc[:pd.Timestamp(d)].iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = float(row["ema_20"]), float(row["ema_50"]), float(row["ema_200"])
    adx = float(row["adx_14"])
    print(f"{sym} on {d}:")
    print(f"  close={c:.1f} 20EMA={e20:.1f} 50EMA={e50:.1f} 200EMA={e200:.1f}")
    print(f"  above 200EMA: {c > e200}  20>50: {e20 > e50}  ADX={adx:.1f}")
    print(f"  dist from 20EMA: {abs(c-e20)/e20*100:.2f}%")

print("\n=== CURRENT REGIME ===")
r = get_market_regime_status()
print(r.message)
for p in r.proxies:
    print(f"  {p['symbol']}: {p['status']} close={p.get('close')}")