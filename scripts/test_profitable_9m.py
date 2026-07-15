"""Fast cached test of combined elite improvements for 9-month profitability."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.models import StrategyConfig
from trading.services.nifty50_index import load_nifty50_frame
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS
from scripts.ema20_enhance_backtest import build_cache, _hl, _base_stop

CAPITAL = 100_000.0
END = date.today()
START = END - timedelta(days=274)


def nifty_regime(nifty, ts, need200=True):
    h = nifty.loc[:pd.Timestamp(ts)]
    if h.empty:
        return False
    r = h.iloc[-1]
    c = float(r["close"])
    e50 = float(r["ema_50"]) if pd.notna(r.get("ema_50")) else None
    e200 = float(r["ema_200"]) if pd.notna(r.get("ema_200")) else None
    if not e50 or c <= e50:
        return False
    if need200 and (not e200 or c <= e200):
        return False
    return True


def signal(hist, config, capital, opts):
    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else row
    c = float(row["close"])
    if not (bool(row.get("strong_close")) or bool(row.get("bullish_engulfing"))):
        return None
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if c <= float(e200) or float(e20) <= float(e50):
        return None
    if float(row["adx_14"]) < opts["adx_min"]:
        return None
    if opts.get("adx_rising") and float(row["adx_14"]) <= float(prev.get("adx_14") or 0):
        return None
    if abs(c - float(e20)) / float(e20) > opts["ema_tol"]:
        return None
    if not _hl(hist, opts["hl_bars"]):
        return None
    rsi = float(row["rsi_14"])
    if not (opts["rsi_lo"] <= rsi <= opts["rsi_hi"]):
        return None
    dp, dm = row.get("di_plus"), row.get("di_minus")
    if pd.isna(dp) or pd.isna(dm) or float(dp) <= float(dm):
        return None
    if opts.get("vol_mult"):
        vs = row.get("vol_sma_20")
        if pd.isna(vs) or float(row["volume"]) < float(vs) * opts["vol_mult"]:
            return None
    stop = _base_stop(hist, e20, e50, row["atr_14"], opts.get("stop_mode", "tight"))
    risk = c - stop
    if risk <= 0 or risk / c > opts.get("max_stop_pct", 0.05):
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"stop": stop, "qty": pos.quantity, "signal_close": c}


def backtest(cache, nifty, config, opts):
    trades = []
    equity = CAPITAL
    for _sym, full in cache.items():
        dates = full.index[(full.index >= pd.Timestamp(START)) & (full.index <= pd.Timestamp(END))]
        pending = None
        in_pos = False
        entry = stop = target = qty = signal_close = 0.0
        hold = 0
        last_exit = None

        for i, ts in enumerate(dates):
            hist = full.loc[:ts]
            row = hist.iloc[-1]
            close, low, op = float(row["close"]), float(row["low"]), float(row["open"])

            if in_pos:
                hold += 1
                risk = entry - stop
                if opts.get("trail_be_r") and close >= entry + risk * opts["trail_be_r"]:
                    stop = max(stop, entry)
                ex = reason = None
                if low <= stop:
                    ex, reason = stop, "sl"
                elif close >= target:
                    ex, reason = target, "tgt"
                elif hold >= opts.get("max_hold", MAX_HOLD_DAYS):
                    ex, reason = close, "time"
                if ex is not None:
                    pnl = (ex - entry) * qty
                    trades.append({"win": pnl > 0, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    hold = 0
                    last_exit = ts
                continue

            if pending is not None:
                sig = pending
                pending = None
                if opts.get("max_gap_pct"):
                    gap = (op - sig["signal_close"]) / sig["signal_close"]
                    if gap > opts["max_gap_pct"]:
                        last_exit = None
                        continue
                entry = op
                stop = sig["stop"]
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * opts["target_r"]
                qty = sig["qty"]
                in_pos = True
                hold = 0
                if low <= stop:
                    pnl = (stop - entry) * qty
                    trades.append({"win": False, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                elif close >= target:
                    pnl = (target - entry) * qty
                    trades.append({"win": True, "pnl": pnl})
                    equity += pnl
                    in_pos = False
                    last_exit = ts
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue
            if opts.get("regime") and not nifty_regime(nifty, ts, opts.get("regime_200", True)):
                continue

            sig = signal(hist, config, equity, opts)
            if sig and i + 1 < len(dates):
                pending = sig

    if not trades:
        return None
    wins = [t for t in trades if t["win"]]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if not t["win"])) or 1e-9
    return {
        "n": len(trades),
        "wr": round(len(wins) / len(trades) * 100, 1),
        "ret": round((equity - CAPITAL) / CAPITAL * 100, 2),
        "pf": round(gp / gl, 2),
        "pnl": round(equity - CAPITAL, 2),
    }


def main():
    from trading.services.market_data import get_universe_symbols

    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"]
    cache = build_cache(symbols)
    nifty = load_nifty50_frame()
    config = StrategyConfig.get_active()

    configs = {
        "current": {
            "adx_min": 20, "ema_tol": 0.014, "hl_bars": 3, "rsi_lo": 48, "rsi_hi": 58,
            "target_r": 2.0, "regime": False,
        },
        "regime200": {
            "adx_min": 20, "ema_tol": 0.014, "hl_bars": 3, "rsi_lo": 48, "rsi_hi": 58,
            "target_r": 2.0, "regime": True, "regime_200": True,
        },
        "enhanced_v1": {
            "adx_min": 22, "ema_tol": 0.012, "hl_bars": 4, "rsi_lo": 50, "rsi_hi": 56,
            "vol_mult": 1.2, "target_r": 2.0, "regime": True, "regime_200": True,
            "trail_be_r": 1.0, "max_gap_pct": 0.012, "max_stop_pct": 0.04,
        },
        "enhanced_v2": {
            "adx_min": 21, "ema_tol": 0.013, "hl_bars": 3, "rsi_lo": 50, "rsi_hi": 57,
            "vol_mult": 1.15, "target_r": 2.0, "regime": True, "regime_200": True,
            "trail_be_r": 0.75, "max_gap_pct": 0.015, "max_hold": 30,
        },
        "enhanced_v3": {
            "adx_min": 20, "ema_tol": 0.014, "hl_bars": 3, "rsi_lo": 49, "rsi_hi": 57,
            "adx_rising": True, "target_r": 2.0, "regime": True, "regime_200": True,
            "trail_be_r": 1.0, "max_gap_pct": 0.01,
        },
        "enhanced_v4_wide": {
            "adx_min": 22, "ema_tol": 0.012, "hl_bars": 4, "rsi_lo": 50, "rsi_hi": 56,
            "target_r": 2.0, "regime": True, "regime_200": True,
            "stop_mode": "wide", "trail_be_r": 1.0, "max_hold": 35,
        },
    }

    print(f"9m: {START} to {END} | {len(cache)} symbols\n")
    best = None
    for name, opts in configs.items():
        r = backtest(cache, nifty, config, opts)
        if not r:
            print(f"{name}: no trades")
            continue
        print(f"{name}: n={r['n']} wr={r['wr']}% ret={r['ret']}% pf={r['pf']} pnl=₹{r['pnl']}")
        if r["ret"] > 0 and (best is None or r["ret"] > best[1]["ret"]):
            best = (name, opts, r)

    if best:
        print(f"\n>>> PROFITABLE: {best[0]} ret={best[1]['ret']}%")
    else:
        print("\n>>> No config profitable on 9m — applying best loss-reducer (regime200)")


if __name__ == "__main__":
    main()