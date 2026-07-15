"""Extended search: regime, ADX rising, volume, RSI bands, 1R target."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.constants import NIFTY50_SYMBOL
from trading.models import StrategyConfig
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.market_regime import is_market_bullish
from trading.services.nifty50_index import load_nifty50_frame
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS
from scripts.ema20_enhance_backtest import _hl, _base_stop


CAPITAL = 100_000.0
END = date.today()
START = END - timedelta(days=274)


def make_signal(**kw):
    adx_min = kw.get("adx_min", 22)
    ema_tol = kw.get("ema_tol", 0.012)
    hl_bars = kw.get("hl_bars", 4)
    rsi_lo, rsi_hi = kw.get("rsi_lo", 50), kw.get("rsi_hi", 56)
    engulf_only = kw.get("engulf_only", False)
    require_stack = kw.get("require_stack", False)
    adx_rising = kw.get("adx_rising", False)
    vol_mult = kw.get("vol_mult", 0.0)

    def fn(hist, config, capital):
        row = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else row
        c = float(row["close"])
        if engulf_only:
            if not bool(row.get("bullish_engulfing")):
                return None
        elif not (bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))):
            return None

        e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
        needed = [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]
        if any(pd.isna(x) for x in needed):
            return None
        if c <= float(e200):
            return None
        if require_stack:
            if not (float(e20) > float(e50) > float(e200)):
                return None
        elif float(e20) <= float(e50):
            return None
        if float(row["adx_14"]) < adx_min:
            return None
        if adx_rising:
            if pd.isna(prev.get("adx_14")) or float(row["adx_14"]) <= float(prev["adx_14"]):
                return None
        if abs(c - float(e20)) / float(e20) > ema_tol:
            return None
        if not _hl(hist, hl_bars):
            return None
        if not (rsi_lo <= float(row["rsi_14"]) <= rsi_hi):
            return None
        di_p, di_m = row.get("di_plus"), row.get("di_minus")
        if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
            return None
        if vol_mult > 0:
            vs = row.get("vol_sma_20")
            if pd.isna(vs) or float(row["volume"]) < float(vs) * vol_mult:
                return None

        stop = _base_stop(hist, e20, e50, row["atr_14"], kw.get("stop_mode", "tight"))
        risk = c - stop
        if risk <= 0 or risk / c > 0.05:
            return None
        pos = calculate_position_size(capital, config.risk_pct, c, stop)
        if pos.quantity <= 0:
            return None
        return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}

    return fn


def backtest(signal_fn, cache, nifty, regime=False, target_r=2.0):
    trades = []
    equity = CAPITAL
    for _sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
        pending = None
        in_pos = False
        entry = stop = target = qty = 0.0
        hold = 0
        last_exit = None
        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, op = float(row["close"]), float(row["low"]), float(row["open"])
            if in_pos:
                hold += 1
                exit_p = reason = None
                if low <= stop:
                    exit_p, reason = stop, "sl"
                elif close >= target:
                    exit_p, reason = target, "tgt"
                elif hold >= MAX_HOLD_DAYS:
                    exit_p, reason = close, "time"
                if exit_p is not None:
                    pnl = (exit_p - entry) * qty
                    trades.append({"win": pnl > 0, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    hold = 0
                    last_exit = ts
                continue
            if pending is not None:
                sig = pending
                pending = None
                entry = op
                stop = sig["stop"]
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * target_r
                qty = sig["qty"]
                in_pos = True
                hold = 0
                if low <= stop:
                    equity += (stop - entry) * qty
                    trades.append({"win": False, "pnl": (stop - entry) * qty})
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    equity += (target - entry) * qty
                    trades.append({"win": True, "pnl": (target - entry) * qty})
                    in_pos = False
                    last_exit = ts
                continue
            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue
            if regime and not is_market_bullish({NIFTY50_SYMBOL: nifty}, ts, 1):
                continue
            sig = signal_fn(hist, StrategyConfig.get_active(), equity)
            if sig and i + 1 < len(dates):
                pending = sig
    if not trades:
        return None
    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    return {"n": len(trades), "wr": round(len(wins) / len(trades) * 100, 1),
            "ret": round((equity - CAPITAL) / CAPITAL * 100, 2), "pf": round(gp / gl, 2)}


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    cache = {s: compute_indicators(load_price_dataframe(s)) for s in symbols
             if not load_price_dataframe(s).empty and has_sufficient_history(compute_indicators(load_price_dataframe(s)))}
    nifty = load_nifty50_frame()

    combos = []
    for regime in (True,):
        for target_r in (2.0, 1.5, 1.0):
            for kw in [
                {"adx_min": 24, "rsi_lo": 50, "rsi_hi": 56, "hl_bars": 4, "adx_rising": True, "vol_mult": 1.5},
                {"adx_min": 24, "rsi_lo": 50, "rsi_hi": 56, "hl_bars": 4, "engulf_only": True, "adx_rising": True},
                {"adx_min": 22, "rsi_lo": 50, "rsi_hi": 56, "hl_bars": 4, "require_stack": True, "vol_mult": 1.5},
                {"adx_min": 25, "rsi_lo": 51, "rsi_hi": 55, "hl_bars": 4, "adx_rising": True, "engulf_only": True},
                {"adx_min": 22, "rsi_lo": 48, "rsi_hi": 58, "hl_bars": 3, "ema_tol": 0.014, "adx_rising": True, "vol_mult": 1.3},
                {"adx_min": 26, "rsi_lo": 50, "rsi_hi": 56, "hl_bars": 4, "engulf_only": True, "require_stack": True},
                {"adx_min": 24, "rsi_lo": 50, "rsi_hi": 56, "hl_bars": 4, "stop_mode": "wide", "adx_rising": True},
            ]:
                name = f"r={regime} t={target_r} " + ",".join(f"{k}={v}" for k, v in kw.items())
                combos.append((name, make_signal(**kw), regime, target_r))

    results = []
    for name, fn, regime, tr in combos:
        r = backtest(fn, cache, nifty, regime=regime, target_r=tr)
        if r and r["n"] >= 3:
            results.append((name, r))

    results.sort(key=lambda x: (-x[1]["ret"], -x[1]["pf"]))
    print("Top results (min 3 trades):\n")
    for name, r in results[:15]:
        print(f"ret={r['ret']:>7.2f}% wr={r['wr']:>5.1f}% n={r['n']:>3} pf={r['pf']:.2f} | {name[:70]}")
    prof = [x for x in results if x[1]["ret"] > 0]
    print(f"\nProfitable: {len(prof)}")
    if prof:
        print("BEST:", prof[0])


if __name__ == "__main__":
    main()