import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from datetime import date, timedelta
import pandas as pd
from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe, get_universe_symbols
from trading.services.position_sizing import calculate_position_size
from scripts.strategy_search2 import run_bt
from scripts.optimize_strategy import market_bullish as market_ok

def sig_pro(hist, config, capital):
    row = hist.iloc[-1]
    c = float(row["close"])
    if len(hist) < 6:
        return None
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50)):
        return None
    if float(row["adx_14"]) < 23:
        return None
    if abs(c - float(e20)) / float(e20) > 0.012:
        return None
    lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
    if not (lows[1] > lows[0] and lows[2] > lows[1] and lows[3] > lows[2]):
        return None
    if not (49 <= float(row["rsi_14"]) <= 57):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    if not bool(row.get("strong_close")):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]) * 0.75)
    risk = c - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}


def run_with_market(min_mkt, max_hold):
    cache = {}
    for s in get_universe_symbols():
        df = load_price_dataframe(s)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            cache[s] = df

    def sig(hist, config, capital):
        if min_mkt:
            if not market_ok(cache, hist.index[-1], min_mkt):
                return None
        return sig_pro(hist, config, capital)

    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    return run_bt(list(cache.keys()), start, end, sig, next_day_entry=True, max_hold=max_hold)


for mkt in [0, 2, 3]:
    for hold in [35, 50, 999]:
        n, wr, pf, r = run_with_market(mkt, hold)
        star = " ***" if wr >= 60 and n >= 5 else ""
        print(f"mkt={mkt} hold={hold}: n={n} wr={wr:.1f}% pf={pf:.2f} {dict(r)}{star}")