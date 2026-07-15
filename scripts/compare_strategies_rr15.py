"""
Compare Confluence Trend Pullback vs EMA20 Elite (+ v2) over 3 years.
Params: Rs 1L capital, 2% risk/trade, 1.5R target (entry filter + exit).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from copy import copy
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

import pandas as pd

from trading.models import StrategyConfig
from trading.services.backtester import run_backtest
from trading.services.indicators import compute_indicators, has_sufficient_history
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.position_sizing import calculate_position_size
from trading.services.signal_backtester import COOLDOWN_DAYS, MAX_HOLD_DAYS, run_signal_backtest
from trading.services.strategy import evaluate_stock
from trading.services.swing_strategy import evaluate_swing_signal

CAPITAL = 100_000.0
RISK_PCT = 2.0
TARGET_R = 1.5
END = date.today()
START = END - timedelta(days=3 * 365)


def _make_config() -> StrategyConfig:
    cfg = StrategyConfig.get_active()
    cfg = copy(cfg)
    cfg.risk_pct = RISK_PCT
    cfg.min_risk_reward = TARGET_R
    return cfg


def run_confluence_rr15(symbols: list[str], config: StrategyConfig) -> dict:
    """Confluence backtest with 1.5R exit (same-day entry, optional 20 EMA trail after 1R)."""
    equity = CAPITAL
    trades = []
    equity_curve = [{"date": str(START), "equity": equity}]

    for symbol in symbols:
        full_df = load_price_dataframe(symbol)
        if full_df.empty:
            continue
        full_df = compute_indicators(full_df)
        if not has_sufficient_history(full_df):
            continue

        mask = (full_df.index >= pd.Timestamp(START)) & (full_df.index <= pd.Timestamp(END))
        eval_dates = full_df.index[mask]

        in_position = False
        entry_price = stop = target = qty = 0.0
        entry_dt = None
        trail_ema = False

        for ts in eval_dates:
            hist = full_df.loc[:ts]
            if len(hist) < 220:
                continue
            row = hist.iloc[-1]
            close = float(row["close"])
            low = float(row["low"])
            ema20 = float(row["ema_20"]) if pd.notna(row["ema_20"]) else None

            if in_position:
                exit_price = None
                exit_reason = ""
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif close >= target:
                    exit_price = target
                    exit_reason = f"target_{TARGET_R}r"
                elif trail_ema and ema20 and close < ema20:
                    exit_price = close
                    exit_reason = "trail_20ema"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * qty
                    risk = entry_price - stop
                    rr = (exit_price - entry_price) / risk if risk > 0 else 0
                    trades.append(
                        {
                            "symbol": symbol,
                            "entry_date": str(entry_dt.date()),
                            "exit_date": str(ts.date()),
                            "pnl": round(pnl, 2),
                            "rr_achieved": round(rr, 2),
                            "exit_reason": exit_reason,
                        }
                    )
                    equity += pnl
                    equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
                    in_position = False
                    trail_ema = False
                elif close >= entry_price + (entry_price - stop):
                    trail_ema = True
                continue

            ev = evaluate_stock(
                symbol,
                eval_date=ts.date(),
                config=config,
                capital=equity,
                indicator_df=hist,
            )
            if ev.is_valid and ev.entry_price and ev.stop_loss:
                pos = calculate_position_size(equity, config.risk_pct, ev.entry_price, ev.stop_loss)
                if pos.quantity <= 0:
                    continue
                in_position = True
                entry_price = ev.entry_price
                stop = ev.stop_loss
                risk = entry_price - stop
                target = entry_price + risk * TARGET_R
                qty = pos.quantity
                entry_dt = ts

    return _summarize("Confluence Trend Pullback", trades, equity_curve, equity)


def run_swing_rr15(symbols: list[str], config: StrategyConfig) -> dict:
    """EMA20 Elite + v2 backtest with 1.5R exit (next-day open entry, 8-day cooldown)."""
    equity = CAPITAL
    trades = []
    equity_curve = [{"date": str(START), "equity": equity}]
    elite_count = 0
    v2_count = 0

    frames = {}
    for sym in symbols:
        df = load_price_dataframe(sym)
        if df.empty:
            continue
        df = compute_indicators(df)
        if has_sufficient_history(df):
            frames[sym] = df

    for symbol, full_df in frames.items():
        mask = (full_df.index >= pd.Timestamp(START)) & (full_df.index <= pd.Timestamp(END))
        eval_dates = full_df.index[mask]

        in_position = False
        pending = None
        entry_price = stop = target = qty = 0.0
        entry_dt = None
        hold_days = 0
        last_exit = None
        entry_path = ""

        def _exit(exit_px: float, reason: str) -> None:
            nonlocal equity, in_position, last_exit, hold_days
            pnl = (exit_px - entry_price) * qty
            risk = entry_price - stop
            rr = (exit_px - entry_price) / risk if risk > 0 else 0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": str(entry_dt.date()),
                    "exit_date": str(ts.date()),
                    "pnl": round(pnl, 2),
                    "rr_achieved": round(rr, 2),
                    "exit_reason": reason,
                    "entry_path": entry_path,
                }
            )
            equity += pnl
            equity_curve.append({"date": str(ts.date()), "equity": round(equity, 2)})
            in_position = False
            last_exit = ts
            hold_days = 0

        for i, ts in enumerate(eval_dates):
            hist = full_df.loc[:ts]
            row = hist.iloc[-1]
            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            open_price = float(row["open"])

            if in_position:
                hold_days += 1
                if low <= stop:
                    _exit(stop, "stop_loss")
                elif close >= target:
                    _exit(target, f"target_{TARGET_R}r")
                elif hold_days >= MAX_HOLD_DAYS:
                    _exit(close, "time_exit")
                continue

            if pending is not None:
                sig = pending
                pending = None
                entry_price = open_price
                stop = sig["stop"]
                risk = entry_price - stop
                if risk <= 0:
                    continue
                target = entry_price + risk * TARGET_R
                qty = sig["qty"]
                entry_dt = ts
                entry_path = sig.get("entry_path", "")
                in_position = True
                hold_days = 0
                if low <= stop:
                    _exit(stop, "stop_loss")
                elif close >= target:
                    _exit(target, f"target_{TARGET_R}r")
                continue

            if last_exit and (ts - last_exit).days < COOLDOWN_DAYS:
                continue

            close = float(row["close"])
            ema200 = row.get("ema_200")
            if pd.isna(ema200) or close <= float(ema200):
                continue

            ev = evaluate_swing_signal(
                symbol,
                eval_date=ts.date(),
                config=config,
                capital=equity,
                indicator_df=hist,
                require_market_filter=False,
            )
            if ev.is_valid and ev.entry_price and ev.stop_loss:
                pos = calculate_position_size(equity, config.risk_pct, ev.entry_price, ev.stop_loss)
                if pos.quantity <= 0:
                    continue
                if ev.entry_path == "Elite 20 EMA":
                    elite_count += 1
                elif ev.entry_path == "20 EMA v2":
                    v2_count += 1
                if i + 1 < len(eval_dates):
                    pending = {
                        "stop": ev.stop_loss,
                        "qty": pos.quantity,
                        "entry_path": ev.entry_path,
                    }

    out = _summarize("EMA20 Elite (+ 20 EMA v2)", trades, equity_curve, equity)
    out["elite_signals"] = elite_count
    out["v2_signals"] = v2_count
    return out


def _summarize(name: str, trades: list, equity_curve: list, equity: float) -> dict:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses)) or 1e-9

    peak = CAPITAL
    max_dd = 0.0
    for pt in equity_curve:
        e = pt["equity"]
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)

    exits = dict(Counter(t["exit_reason"] for t in trades))

    return {
        "name": name,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if trades else 0,
        "total_return_pct": round((equity - CAPITAL) / CAPITAL * 100, 2),
        "final_equity": round(equity, 2),
        "avg_rr": round(sum(t["rr_achieved"] for t in trades) / len(trades), 2) if trades else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "exit_breakdown": exits,
    }


def main() -> None:
    symbols = get_universe_symbols(nifty200_only=True)
    symbols = [s for s in symbols if s != "NIFTY50"]
    config = _make_config()

    print(f"Period: {START} to {END}  |  Capital: Rs {CAPITAL:,.0f}")
    print(f"Risk: {RISK_PCT}%/trade  |  Target: {TARGET_R}R  |  Universe: {len(symbols)} Nifty 200 stocks\n")

    conf = run_confluence_rr15(symbols, config)
    swing = run_swing_rr15(symbols, config)

    for r in (conf, swing):
        print(f"=== {r['name']} ===")
        print(f"  Trades:        {r['trades']}")
        print(f"  Win rate:      {r['win_rate']}%  ({r['wins']}W / {r['losses']}L)")
        print(f"  Profit factor: {r['profit_factor']}")
        print(f"  Total return:  {r['total_return_pct']}%  (final Rs {r['final_equity']:,.0f})")
        print(f"  Max drawdown:  {r['max_drawdown_pct']}%")
        print(f"  Avg R/trade:   {r['avg_rr']}")
        print(f"  Exit mix:      {r['exit_breakdown']}")
        if "elite_signals" in r:
            print(f"  Signal paths:  Elite={r['elite_signals']}, v2={r['v2_signals']}")
        print()

    winner = conf if conf["total_return_pct"] > swing["total_return_pct"] else swing
    if conf["total_return_pct"] == swing["total_return_pct"]:
        winner_name = "Tie"
    else:
        winner_name = winner["name"]
    print(f"WINNER (by total return): {winner_name}")
    print(f"  Confluence: {conf['total_return_pct']}% vs EMA20: {swing['total_return_pct']}%")


if __name__ == "__main__":
    main()