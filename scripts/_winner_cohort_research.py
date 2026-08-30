"""One-year technical profile of Stage 2.0 winner cohort vs Nifty 200."""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import django
import numpy as np
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
django.setup()

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from stage_analysis_v2.services.indicators import add_daily_indicators, add_weekly_indicators
from stage_analysis_v2.services.stage_engine import detect_weekly_stage
from stage_analysis_v2.services.tech_filters import enrich_daily_tech
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import load_price_dataframe, get_universe_symbols

OUT = ROOT / "data" / "winner_cohort_research.json"

WINNERS = [
    "FEDERALBNK", "AUBANK", "SHRIRAMFIN", "UNIONBANK",
    "BHARATFORG", "JSWSTEEL", "SAIL",
    "BEL", "CUMMINSIND", "GVT&D", "POWERINDIA",
    "GLENMARK", "ADANIENSOL",
]
START = date(2025, 8, 27)
END = date(2026, 8, 27)


def _ret(df: pd.DataFrame, days: int, end_ts: pd.Timestamp) -> float | None:
    if df.empty:
        return None
    window = df.loc[:end_ts]
    if window.empty:
        return None
    last = float(window["close"].iloc[-1])
    past = window.loc[: end_ts - pd.Timedelta(days=days)]
    if past.empty:
        return None
    first = float(past["close"].iloc[-1])
    if first <= 0:
        return None
    return round((last / first - 1) * 100, 2)


def _max_dd(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    peak = close.cummax()
    dd = (close / peak - 1) * 100
    return round(float(dd.min()), 2)


def _pct_from_high(close: pd.Series) -> float | None:
    if close.empty:
        return None
    hi = float(close.max())
    last = float(close.iloc[-1])
    if hi <= 0:
        return None
    return round((last / hi - 1) * 100, 2)


def profile_symbol(sym: str, nifty: pd.DataFrame) -> dict:
    df = load_price_dataframe(sym)
    if df.empty:
        return {"symbol": sym, "error": "no data"}
    d = enrich_daily_tech(add_daily_indicators(df), include_supertrend=True)
    end_ts = pd.Timestamp(END)
    start_ts = pd.Timestamp(START)
    y = d.loc[(d.index >= start_ts) & (d.index <= end_ts)]
    if y.empty:
        y = d.tail(252)
    last = d.loc[:end_ts].iloc[-1]
    close = float(last["close"])
    yr_close = d.loc[:end_ts, "close"]
    look = yr_close.loc[yr_close.index >= start_ts]
    if look.empty:
        look = yr_close.tail(252)

    # 52w stats at start of window (would a buyer then have seen "cheap"?)
    pre = d.loc[:start_ts]
    pre_52 = pre.tail(252)
    start_close = float(pre["close"].iloc[-1]) if not pre.empty else close
    start_52_high = float(pre_52["high"].max()) if not pre_52.empty else start_close
    start_52_low = float(pre_52["low"].min()) if not pre_52.empty else start_close
    start_pos = None
    if start_52_high > start_52_low:
        start_pos = round((start_close - start_52_low) / (start_52_high - start_52_low) * 100, 1)

    # Nifty RS over the year
    n = nifty.loc[(nifty.index >= start_ts) & (nifty.index <= end_ts), "close"]
    s = look
    aligned = pd.concat([s, n], axis=1, join="inner")
    aligned.columns = ["s", "n"]
    rs_year = None
    if len(aligned) >= 2 and aligned["n"].iloc[0] > 0 and aligned["s"].iloc[0] > 0:
        rs_year = round(
            ((aligned["s"].iloc[-1] / aligned["s"].iloc[0]) / (aligned["n"].iloc[-1] / aligned["n"].iloc[0]) - 1) * 100,
            2,
        )

    weekly = add_weekly_indicators(daily_to_weekly(d.loc[:end_ts]))
    w_stage, w_reasons, w_metrics = (0, [], {})
    if len(weekly) >= 40:
        try:
            w_stage, w_reasons, w_metrics = detect_weekly_stage(weekly)
        except ValueError:
            pass

    st_dir = last.get("st_dir")
    ema20 = last.get("ema_20")
    ema50 = last.get("ema_50")
    sma50 = last.get("ma_50d")
    sma150 = last.get("ma_150d")
    sma200 = last.get("ema_200")
    rsi = last.get("rsi_14")
    adx = last.get("adx")
    atr = last.get("atr_14")

    def g(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(float(v), 2)
        except Exception:
            return None

    stack = None
    if ema20 and sma50 and sma150:
        stack = bool(close > float(ema20) > float(sma50) and close > float(sma150))

    vol_y = y["volume"] if "volume" in y else pd.Series(dtype=float)
    vol_prior = d.loc[:start_ts].tail(60)["volume"] if "volume" in d else pd.Series(dtype=float)
    vol_expansion = None
    if len(vol_y) and len(vol_prior):
        a, b = float(vol_prior.mean() or 0), float(vol_y.mean() or 0)
        if a > 0:
            vol_expansion = round(b / a, 2)

    # Biggest consecutive up-leg inside the year
    c = look.astype(float)
    peak_leg = 0.0
    trough = c.iloc[0]
    for px in c:
        if px < trough:
            trough = px
        peak_leg = max(peak_leg, (px / trough - 1) * 100 if trough else 0)

    return {
        "symbol": sym,
        "last_close": round(close, 2),
        "ret_1y": _ret(d, 365, end_ts),
        "ret_6m": _ret(d, 182, end_ts),
        "ret_3m": _ret(d, 91, end_ts),
        "max_dd_1y": _max_dd(look),
        "from_52w_high": _pct_from_high(d.loc[d.index >= end_ts - pd.Timedelta(days=365), "high"] if False else look),
        "from_52w_high_now": _pct_from_high(d.loc[d.index >= end_ts - pd.Timedelta(days=365), "close"]),
        "year_high": round(float(look.max()), 2),
        "year_low": round(float(look.min()), 2),
        "pct_above_year_low": round((close / float(look.min()) - 1) * 100, 2) if float(look.min()) else None,
        "start_close_20250827": round(start_close, 2),
        "start_52w_range_pos_pct": start_pos,  # 0=52w low, 100=52w high at window start
        "start_vs_52w_high_pct": round((start_close / start_52_high - 1) * 100, 2) if start_52_high else None,
        "start_vs_52w_low_pct": round((start_close / start_52_low - 1) * 100, 2) if start_52_low else None,
        "rs_vs_nifty_1y": rs_year,
        "weekly_stage_now": w_stage,
        "weekly_reasons": w_reasons[:4],
        "rsi": g(rsi),
        "adx": g(adx),
        "atr_pct": round(float(atr) / close * 100, 2) if atr and close else None,
        "ema20": g(ema20),
        "sma50": g(sma50),
        "sma150": g(sma150),
        "above_ema20": bool(close > float(ema20)) if ema20 == ema20 and ema20 else None,
        "above_sma50": bool(close > float(sma50)) if sma50 == sma50 and sma50 else None,
        "above_sma150": bool(close > float(sma150)) if sma150 == sma150 and sma150 else None,
        "ema_stack": stack,
        "st_bull": int(st_dir) > 0 if st_dir == st_dir and st_dir is not None else None,
        "vol_expansion_vs_pre": vol_expansion,
        "max_upleg_from_trough_pct": round(peak_leg, 2),
        "price_vs_sma150_pct": round((close / float(sma150) - 1) * 100, 2) if sma150 == sma150 and sma150 else None,
    }


def main():
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    universe = get_universe_symbols(nifty200_only=True)
    winner_set = set(WINNERS)

    profiles = [profile_symbol(s, nifty) for s in WINNERS]
    nifty_prof = {
        "symbol": "NIFTY50",
        "ret_1y": _ret(nifty, 365, pd.Timestamp(END)),
        "ret_6m": _ret(nifty, 182, pd.Timestamp(END)),
        "ret_3m": _ret(nifty, 91, pd.Timestamp(END)),
        "max_dd_1y": _max_dd(nifty.loc[(nifty.index >= pd.Timestamp(START)) & (nifty.index <= pd.Timestamp(END)), "close"]),
    }

    # Universe 1y returns for percentile
    univ_rets = []
    for s in universe:
        df = load_price_dataframe(s)
        r = _ret(df, 365, pd.Timestamp(END))
        if r is not None:
            univ_rets.append({"symbol": s, "ret_1y": r, "winner": s in winner_set})
    univ_rets.sort(key=lambda x: x["ret_1y"], reverse=True)
    ranks = {row["symbol"]: i + 1 for i, row in enumerate(univ_rets)}
    n = len(univ_rets)
    for p in profiles:
        rnk = ranks.get(p["symbol"])
        p["nifty200_rank"] = rnk
        p["nifty200_pctile"] = round((n - rnk + 1) / n * 100, 1) if rnk else None

    winner_rets = [p["ret_1y"] for p in profiles if p.get("ret_1y") is not None]
    all_rets = [x["ret_1y"] for x in univ_rets]
    median_univ = float(np.median(all_rets)) if all_rets else None
    median_win = float(np.median(winner_rets)) if winner_rets else None

    # Stage 2 backtest last 1y on this cohort vs full nifty200 is too heavy;
    # just this cohort, then also scan losers among universe top-bottom.
    bt_w = run_stage_v2_backtest(
        symbols=WINNERS,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
    )
    signals = []
    for row in bt_w.signal_log or []:
        signals.append(row)
    trades = [
        {
            "symbol": t.symbol,
            "signal_date": t.signal_date,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "target": t.target,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "rr_achieved": t.rr_achieved,
            "exit_reason": t.exit_reason,
            "quality_score": t.quality_score,
            "rs_rating": t.rs_rating,
            "days_held": t.days_held,
            "status": "taken",
        }
        for t in bt_w.trades
    ]

    # At-signal snapshot (no look-ahead)
    snapshots = []
    for row in signals:
        sym = row["symbol"]
        sig_d = pd.Timestamp(row["signal_date"])
        df = load_price_dataframe(sym)
        if df.empty or sig_d not in df.index:
            # nearest
            if df.empty:
                continue
            i = df.index.get_indexer([sig_d], method="nearest")[0]
            bar_ts = df.index[i]
        else:
            bar_ts = sig_d
        hist = enrich_daily_tech(add_daily_indicators(df.loc[:bar_ts]), include_supertrend=True)
        if hist.empty:
            continue
        last = hist.iloc[-1]
        close = float(last["close"])
        look252 = hist["close"].tail(252)
        hi252 = float(hist["high"].tail(252).max())
        lo252 = float(hist["low"].tail(252).min())
        sma150 = last.get("ma_150d")
        ema20 = last.get("ema_20")
        rsi = last.get("rsi_14")
        adx = last.get("adx")
        vol_r = last.get("vol_ratio")
        st_dir = last.get("st_dir")
        range_pos = None
        if hi252 > lo252:
            range_pos = round((close - lo252) / (hi252 - lo252) * 100, 1)
        snapshots.append({
            "symbol": sym,
            "signal_date": row["signal_date"],
            "entry_date": row.get("entry_date"),
            "status": row.get("status"),
            "quality_score": row.get("quality_score"),
            "rs_rating": row.get("rs_rating"),
            "close": round(close, 2),
            "rsi": None if rsi is None or rsi != rsi else round(float(rsi), 1),
            "adx": None if adx is None or adx != adx else round(float(adx), 1),
            "vol_ratio": None if vol_r is None or vol_r != vol_r else round(float(vol_r), 2),
            "st_bull": int(st_dir) > 0 if st_dir == st_dir and st_dir is not None else None,
            "above_ema20": bool(close > float(ema20)) if ema20 == ema20 and ema20 else None,
            "pct_from_52w_high": round((close / hi252 - 1) * 100, 2) if hi252 else None,
            "pct_from_52w_low": round((close / lo252 - 1) * 100, 2) if lo252 else None,
            "range_pos_pct": range_pos,
            "vs_sma150_pct": round((close / float(sma150) - 1) * 100, 2) if sma150 == sma150 and sma150 else None,
            "stop_loss": row.get("stop_loss"),
            "target": row.get("target"),
            "pnl_pct": row.get("pnl_pct"),
            "exit_reason": row.get("exit_reason"),
        })

    bottom20 = univ_rets[-20:]
    top20 = univ_rets[:20]

    out = {
        "window": {"start": str(START), "end": str(END)},
        "nifty50": nifty_prof,
        "universe_n": n,
        "universe_median_1y": round(median_univ, 2) if median_univ is not None else None,
        "winner_median_1y": round(median_win, 2) if median_win is not None else None,
        "winner_mean_1y": round(float(np.mean(winner_rets)), 2) if winner_rets else None,
        "profiles": profiles,
        "top20_nifty200": top20,
        "bottom20_nifty200": bottom20,
        "cohort_backtest": {
            "signals": bt_w.total_signals,
            "trades": bt_w.total_trades,
            "skipped_cash": bt_w.signals_skipped_cash,
            "win_rate": bt_w.win_rate,
            "total_return_pct": bt_w.total_return_pct,
            "signal_log": signals,
            "trades_log": trades,
        },
        "at_signal_snapshots": snapshots,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"Nifty 1y {nifty_prof['ret_1y']}% | univ median {out['universe_median_1y']}% | winners median {out['winner_median_1y']}% mean {out['winner_mean_1y']}%")
    print("symbol ret1y rank pctile start_range_pos rs ema_stack vol_x")
    for p in sorted(profiles, key=lambda x: -(x.get("ret_1y") or -999)):
        print(
            f"{p['symbol']:12} {p.get('ret_1y')}%  rank={p.get('nifty200_rank')}  "
            f"pctile={p.get('nifty200_pctile')}  startPos={p.get('start_52w_range_pos_pct')}  "
            f"rs={p.get('rs_vs_nifty_1y')}  stack={p.get('ema_stack')}  volx={p.get('vol_expansion_vs_pre')}  "
            f"upleg={p.get('max_upleg_from_trough_pct')}  dd={p.get('max_dd_1y')}"
        )
    print("\n--- Stage 2 signals on cohort ---")
    for s in signals:
        print(s.get("symbol"), s.get("signal_date"), s.get("status"), "Q", s.get("quality_score"), "RS", s.get("rs_rating"), "pnl", s.get("pnl_pct"))


if __name__ == "__main__":
    main()
