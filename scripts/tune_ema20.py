import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

from scripts.strategy_search2 import run_bt, enrich
from trading.services.market_data import get_universe_symbols
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import load_price_dataframe
from trading.services.position_sizing import calculate_position_size
from trading.models import StrategyConfig
from datetime import date, timedelta
import pandas as pd
import numpy as np


def make_signal(adx_min=22, rsi_lo=48, rsi_hi=58, vol_mult=0, require_engulf=False, hl_bars=4):
    def fn(hist, config, capital):
        row = hist.iloc[-1]
        c, low = float(row["close"]), float(row["low"])
        if len(hist) < hl_bars + 2:
            return None
        e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
        if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
            return None
        if not (c > float(e200) and float(e20) > float(e50)):
            return None
        if float(row["adx_14"]) < adx_min:
            return None
        tol = 0.012
        if abs(c - float(e20)) / float(e20) > tol:
            return None
        lows = [float(hist.iloc[j]["low"]) for j in range(-hl_bars, 0)]
        if not all(lows[i] > lows[i-1] for i in range(1, len(lows))):
            return None
        if not (rsi_lo <= float(row["rsi_14"]) <= rsi_hi):
            return None
        di_p, di_m = row.get("di_plus"), row.get("di_minus")
        if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
            return None
        if vol_mult > 0:
            vs = row.get("vol_sma_20")
            if pd.isna(vs) or int(row["volume"]) < float(vs) * vol_mult:
                return None
        if require_engulf and not bool(row.get("bullish_engulfing")):
            return None
        if not (bool(row.get("bullish_engulfing")) or bool(row.get("hammer")) or bool(row.get("strong_close"))):
            return None
        stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
        entry = c
        risk = entry - stop
        if risk <= 0:
            return None
        pos = calculate_position_size(capital, config.risk_pct, entry, stop)
        if pos.quantity <= 0:
            return None
        return {"entry": entry, "stop": stop, "target": entry + risk * 2, "qty": pos.quantity}
    return fn


if __name__ == "__main__":
    end = date(2025, 6, 1)
    start = end - timedelta(days=365)
    syms = get_universe_symbols(nifty200_only=True)

    configs = [
        ("base", make_signal()),
        ("adx25", make_signal(adx_min=25)),
        ("adx28", make_signal(adx_min=28)),
        ("rsi50-56", make_signal(rsi_lo=50, rsi_hi=56)),
        ("vol1.2", make_signal(vol_mult=1.2)),
        ("engulf", make_signal(require_engulf=True)),
        ("adx25_rsi50", make_signal(adx_min=25, rsi_lo=50, rsi_hi=56)),
        ("adx25_vol", make_signal(adx_min=25, vol_mult=1.15)),
        ("strict", make_signal(adx_min=25, rsi_lo=50, rsi_hi=56, vol_mult=1.1, require_engulf=True)),
        ("hl3", make_signal(hl_bars=3)),
    ]
    for name, fn in configs:
        n, wr, pf, r = run_bt(syms, start, end, fn, next_day_entry=True, max_hold=25)
        if n > 0:
            print(f"{name}: n={n} wr={wr:.1f}% pf={pf:.2f} {dict(r)}")