"""Check 30% annual target: full years, rolling 12mo, dual-path, risk sweep."""
import os, sys
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()

import pandas as pd
from scripts.ema20_enhance_backtest import COOLDOWN, MAX_HOLD, build_cache, make_ema20_strong, backtest, BtResult
from trading.models import StrategyConfig
from trading.services.market_data import get_universe_symbols
from trading.services.position_sizing import calculate_position_size


def dual_path_signal(hist, config, capital):
    """Elite enhanced OR v2 bounce — more trades."""
    elite = make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3)(hist, config, capital)
    if elite:
        return elite
    # v2 fallback
    row = hist.iloc[-1]
    c = float(row["close"])
    e20, e50, e200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    if any(pd.isna(x) for x in [e20, e50, e200, row.get("adx_14"), row.get("rsi_14"), row.get("atr_14")]):
        return None
    if not (c > float(e200) and float(e20) > float(e50) and float(row["adx_14"]) >= 22):
        return None
    if abs(c - float(e20)) / float(e20) > 0.012:
        return None
    lows = [float(hist.iloc[j]["low"]) for j in range(-4, 0)]
    if not all(lows[i] > lows[i-1] for i in range(1, 4)):
        return None
    if not (48 <= float(row["rsi_14"]) <= 58):
        return None
    di_p, di_m = row.get("di_plus"), row.get("di_minus")
    if pd.isna(di_p) or pd.isna(di_m) or float(di_p) <= float(di_m):
        return None
    stop = min(float(hist["low"].iloc[-5:].min()), float(e20) - float(row["atr_14"]))
    risk = c - stop
    if risk <= 0:
        return None
    pos = calculate_position_size(capital, config.risk_pct, c, stop)
    if pos.quantity <= 0:
        return None
    return {"entry": c, "stop": stop, "target": c + risk * 2, "qty": pos.quantity}


def run_period(name, fn, cache, capital, risk_pct, start, end):
    config = StrategyConfig.get_active()
    orig_risk = config.risk_pct
    config.risk_pct = risk_pct
    # patch backtest to use custom capital
    from scripts.ema20_enhance_backtest import backtest as bt
    import scripts.ema20_enhance_backtest as mod
    old_cap = mod.CAPITAL
    mod.CAPITAL = capital
    r = bt(name, fn, cache, config, start=start, end=end)
    mod.CAPITAL = old_cap
    config.risk_pct = orig_risk
    return r


def main():
    capital = 100_000.0
    cache = build_cache(get_universe_symbols(nifty200_only=True))
    enhanced = make_ema20_strong(adx_min=20, ema_tol=0.014, hl_bars=3)
    config = StrategyConfig.get_active()

    print(f"Rs {capital:,.0f} capital | Target: 30%+ per year (Rs 1L -> Rs 1.3L+)\n")

    # Full calendar years
    print("=== FULL CALENDAR YEARS ===")
    for yr in [2023, 2024]:
        start, end = date(yr, 1, 1), date(yr, 12, 31)
        for risk in [1.0, 1.5, 2.0, 2.5]:
            config.risk_pct = risk
            import scripts.ema20_enhance_backtest as mod
            mod.CAPITAL = capital
            r = mod.backtest(f"enhanced", enhanced, cache, config, start=start, end=end)
            flag = "PASS" if r.return_pct >= 30 else "fail"
            print(f"  {yr} risk={risk}%: {r.trades} trades WR={r.win_rate}% ret={r.return_pct:+.1f}% [{flag}] -> Rs {capital*(1+r.return_pct/100):,.0f}")

    # Rolling 12-month windows
    print("\n=== ROLLING 12-MONTH WINDOWS ===")
    windows = [
        ("Jun23-Jun24", date(2023, 6, 16), date(2024, 6, 15)),
        ("Jun24-Jun25", date(2024, 6, 16), date(2025, 6, 1)),
        ("Jan24-Dec24", date(2024, 1, 1), date(2024, 12, 31)),
    ]
    for label, start, end in windows:
        for risk in [1.5, 2.0, 2.5]:
            config.risk_pct = risk
            import scripts.ema20_enhance_backtest as mod
            mod.CAPITAL = capital
            r = mod.backtest(label, enhanced, cache, config, start=start, end=end)
            flag = "PASS" if r.return_pct >= 30 else "fail"
            print(f"  {label} risk={risk}%: {r.trades}t WR={r.win_rate}% ret={r.return_pct:+.1f}% [{flag}]")

    # Dual path
    print("\n=== DUAL PATH (Elite + v2 bounce) ===")
    for risk in [1.0, 1.5, 2.0]:
        config.risk_pct = risk
        import scripts.ema20_enhance_backtest as mod
        mod.CAPITAL = capital
        r = mod.backtest("dual", dual_path_signal, cache, config, start=date(2024,1,1), end=date(2024,12,31))
        flag = "PASS" if r.return_pct >= 30 else "fail"
        print(f"  2024 risk={risk}%: {r.trades}t WR={r.win_rate}% ret={r.return_pct:+.1f}% [{flag}] PF={r.profit_factor}")

    print("\n=== VERDICT ===")
    config.risk_pct = 2.0
    import scripts.ema20_enhance_backtest as mod
    mod.CAPITAL = capital
    r24 = mod.backtest("2024", enhanced, cache, config, start=date(2024,1,1), end=date(2024,12,31))
    r12 = mod.backtest("12mo", enhanced, cache, config, start=date(2023,6,16), end=date(2024,6,15))
    print(f"Best full year 2024 @ 2% risk: {r24.return_pct:+.1f}% ({r24.trades} trades, WR {r24.win_rate}%)")
    print(f"First 12mo window @ 2% risk: {r12.return_pct:+.1f}% ({r12.trades} trades)")
    if r24.return_pct >= 30:
        print(f"30% target MET in 2024 with 2% risk/trade: Rs 1L -> Rs {100000*(1+r24.return_pct/100):,.0f}")
    else:
        needed = 30 / r24.return_pct * 2.0 if r24.return_pct > 0 else 99
        print(f"2024 needs ~{needed:.1f}% risk/trade to hit 30% annual return")


if __name__ == "__main__":
    main()