"""Fundamentals + market-path autopsy for Stage 2.0 RS70 losers."""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from trading.models import DailyPrice, Stock  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402
from trading.constants import NIFTY50_SYMBOL  # noqa: E402

LOSERS = [
    dict(symbol="AXISBANK", signal="2026-01-30", entry="2026-02-02", exit="2026-03-13",
         entry_px=1337.97, exit_px=1196.38, pnl=-13026, ret=-10.6, r=-0.68, days=28, reason="stage_exit"),
    dict(symbol="JSWSTEEL", signal="2026-01-30", entry="2026-02-02", exit="2026-03-13",
         entry_px=1193.15, exit_px=1112.91, pnl=-11474, ret=-6.7, r=-0.59, days=28, reason="stage_exit"),
    dict(symbol="NYKAA", signal="2026-02-06", entry="2026-02-09", exit="2026-03-13",
         entry_px=277.0, exit_px=238.15, pnl=-11228, ret=-14.0, r=-0.85, days=23, reason="stage_exit"),
    dict(symbol="COROMANDEL", signal="2025-10-03", entry="2025-10-06", exit="2025-10-10",
         entry_px=2301.18, exit_px=2209.63, pnl=-9979, ret=-4.0, r=-0.55, days=4, reason="stage_exit"),
    dict(symbol="APLAPOLLO", signal="2026-01-30", entry="2026-02-02", exit="2026-05-01",
         entry_px=2064.0, exit_px=1905.0, pnl=-7632, ret=-7.7, r=-0.39, days=59, reason="stage_exit"),
    dict(symbol="ADANIPORTS", signal="2025-10-17", entry="2025-10-20", exit="2026-01-16",
         entry_px=1481.76, exit_px=1415.93, pnl=-6254, ret=-4.4, r=-0.32, days=61, reason="stage_exit"),
    dict(symbol="ONGC", signal="2026-02-20", entry="2026-02-23", exit="2026-06-01",
         entry_px=279.9, exit_px=264.3, pnl=-5912, ret=-5.6, r=-0.28, days=65, reason="time_exit"),
    dict(symbol="BIOCON", signal="2026-05-29", entry="2026-06-01", exit="2026-08-31",
         entry_px=430.68, exit_px=412.0, pnl=-5735, ret=-4.3, r=-0.28, days=65, reason="time_exit"),
    dict(symbol="BAJAJFINSV", signal="2025-09-19", entry="2025-09-22", exit="2025-12-26",
         entry_px=2065.54, exit_px=2015.88, pnl=-5115, ret=-2.4, r=-0.26, days=65, reason="time_exit"),
    dict(symbol="EXIDEIND", signal="2025-09-26", entry="2025-09-29", exit="2025-10-31",
         entry_px=387.64, exit_px=380.08, pnl=-5035, ret=-2.0, r=-0.26, days=22, reason="stage_exit"),
    dict(symbol="YESBANK", signal="2025-10-10", entry="2025-10-13", exit="2026-01-15",
         entry_px=24.2, exit_px=22.95, pnl=-4431, ret=-5.2, r=-0.23, days=65, reason="time_exit"),
    dict(symbol="SBIN", signal="2026-08-07", entry="2026-08-10", exit="2026-09-01",
         entry_px=1108.0, exit_px=1034.5, pnl=-3896, ret=-6.6, r=-0.64, days=16, reason="eod_force"),
    dict(symbol="EICHERMOT", signal="2026-08-14", entry="2026-08-17", exit="2026-09-01",
         entry_px=8058.0, exit_px=7970.0, pnl=-1056, ret=-1.1, r=-0.08, days=11, reason="eod_force"),
    dict(symbol="ASHOKLEY", signal="2026-04-10", entry="2026-04-13", exit="2026-05-01",
         entry_px=169.12, exit_px=159.37, pnl=-419, ret=-5.8, r=-0.78, days=13, reason="stage_exit"),
    dict(symbol="CHOLAFIN", signal="2026-08-14", entry="2026-08-17", exit="2026-09-01",
         entry_px=1888.0, exit_px=1814.7, pnl=-73, ret=-3.9, r=-0.23, days=11, reason="eod_force"),
]


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _sma(s: pd.Series, n: int) -> float | None:
    if s is None or len(s) < n:
        return None
    v = float(s.tail(n).mean())
    return v if v == v else None


def _ret(s: pd.Series, n: int) -> float | None:
    if s is None or len(s) < n + 1:
        return None
    a, b = float(s.iloc[-n - 1]), float(s.iloc[-1])
    if a <= 0:
        return None
    return (b / a - 1.0) * 100


def slice_to(df: pd.DataFrame, d: date) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index).date
    return df.loc[idx <= d]


def yf_fundamentals(symbol: str) -> dict:
    ticker = yfinance_ticker(symbol)
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        return {"error": str(exc)}
    keys = [
        "longName", "sector", "industry", "trailingPE", "forwardPE", "pegRatio",
        "priceToBook", "profitMargins", "operatingMargins", "returnOnEquity",
        "returnOnAssets", "debtToEquity", "currentRatio", "revenueGrowth",
        "earningsGrowth", "grossMargins", "heldPercentInsiders", "heldPercentInstitutions",
        "beta", "shortRatio", "enterpriseToEbitda", "trailingEps", "bookValue",
        "dividendYield", "payoutRatio", "recommendationKey", "numberOfAnalystOpinions",
        "targetMeanPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume",
        "marketCap", "floatShares", "sharesOutstanding",
    ]
    out = {k: info.get(k) for k in keys if info.get(k) is not None}
    out["ticker"] = ticker
    return out


def nifty_path(entry: date, exit_d: date) -> dict:
    n = load_price_dataframe(NIFTY50_SYMBOL)
    if n.empty:
        return {}
    a = slice_to(n, entry)
    b = slice_to(n, exit_d)
    if a.empty or b.empty:
        return {}
    e = float(a["close"].iloc[-1])
    x = float(b["close"].iloc[-1])
    hold = n.loc[(pd.to_datetime(n.index).date >= entry) & (pd.to_datetime(n.index).date <= exit_d)]
    dd = None
    if not hold.empty:
        c = hold["close"].astype(float)
        dd = float((c / c.cummax() - 1).min() * 100)
    return {
        "nifty_entry": round(e, 2),
        "nifty_exit": round(x, 2),
        "nifty_hold_pct": round((x / e - 1) * 100, 2) if e else None,
        "nifty_hold_dd": round(dd, 2) if dd is not None else None,
        "sma50": _sma(a["close"].astype(float), 50),
        "sma150": _sma(a["close"].astype(float), 150),
        "sma200": _sma(a["close"].astype(float), 200),
    }


def stock_path(symbol: str, signal: date, entry: date, exit_d: date) -> dict:
    df = load_price_dataframe(symbol)
    if df.empty:
        return {"error": "no prices"}
    pre = slice_to(df, signal)
    at_entry = slice_to(df, entry)
    if pre.empty:
        return {"error": "no pre-entry bars"}
    c = pre["close"].astype(float)
    px = float(c.iloc[-1])
    sma50 = _sma(c, 50)
    sma150 = _sma(c, 150)
    sma200 = _sma(c, 200)
    high20 = float(pre["high"].tail(20).max()) if len(pre) >= 20 else None
    high60 = float(pre["high"].tail(60).max()) if len(pre) >= 60 else None
    vol20 = float(pre["volume"].tail(20).mean()) if "volume" in pre and len(pre) >= 20 else None
    vol5 = float(pre["volume"].tail(5).mean()) if "volume" in pre and len(pre) >= 5 else None
    hold = df.loc[(pd.to_datetime(df.index).date >= entry) & (pd.to_datetime(df.index).date <= exit_d)]
    mfe = mae = None
    if not hold.empty and not at_entry.empty:
        ep = float(at_entry["close"].iloc[-1])
        mfe = float((hold["high"].astype(float).max() / ep - 1) * 100)
        mae = float((hold["low"].astype(float).min() / ep - 1) * 100)
    return {
        "bars": int(len(pre)),
        "close_signal": round(px, 2),
        "ret_1m": None if _ret(c, 21) is None else round(_ret(c, 21), 2),
        "ret_3m": None if _ret(c, 63) is None else round(_ret(c, 63), 2),
        "ret_6m": None if _ret(c, 126) is None else round(_ret(c, 126), 2),
        "ret_1y": None if _ret(c, 252) is None else round(_ret(c, 252), 2),
        "sma50": None if sma50 is None else round(sma50, 2),
        "sma150": None if sma150 is None else round(sma150, 2),
        "sma200": None if sma200 is None else round(sma200, 2),
        "pct_vs_sma50": None if not sma50 else round((px / sma50 - 1) * 100, 2),
        "pct_vs_sma150": None if not sma150 else round((px / sma150 - 1) * 100, 2),
        "pct_vs_sma200": None if not sma200 else round((px / sma200 - 1) * 100, 2),
        "ext_20d_high": None if not high20 else round((px / high20 - 1) * 100, 2),
        "ext_60d_high": None if not high60 else round((px / high60 - 1) * 100, 2),
        "vol5_vs_20": None if not vol20 or not vol5 else round(vol5 / vol20, 2),
        "mfe_pct": None if mfe is None else round(mfe, 2),
        "mae_pct": None if mae is None else round(mae, 2),
    }


def main():
    nifty = load_price_dataframe(NIFTY50_SYMBOL)
    rows = []
    for t in LOSERS:
        sym = t["symbol"]
        stock = Stock.objects.filter(pk=sym).first()
        db = {}
        if stock:
            db = {
                "name": stock.name,
                "sector": stock.sector,
                "sales_growth_yoy": stock.sales_growth_yoy,
                "profit_growth": stock.profit_growth,
                "roe_5yr_avg": stock.roe_5yr_avg,
                "debt_equity": stock.debt_equity,
                "peg": stock.peg,
                "institutional_interest": stock.institutional_interest,
            }
        print("fetch", sym, flush=True)
        yf_info = yf_fundamentals(sym)
        sig, ent, ex = _d(t["signal"]), _d(t["entry"]), _d(t["exit"])
        path = stock_path(sym, sig, ent, ex)
        mkt = nifty_path(ent, ex)
        n_pre = slice_to(nifty, sig)
        n_ret3 = _ret(n_pre["close"].astype(float), 63) if not n_pre.empty else None
        rs3 = None
        if path.get("ret_3m") is not None and n_ret3 is not None:
            rs3 = round(path["ret_3m"] - n_ret3, 2)
        row = {
            **t,
            "db": db,
            "yf": yf_info,
            "path": path,
            "mkt": mkt,
            "rs_3m_vs_nifty": rs3,
            "cluster": (
                "jan30_mar13" if t["signal"] in ("2026-01-30", "2026-02-06")
                else "aug_eod" if t["reason"] == "eod_force"
                else "time_65d" if t["reason"] == "time_exit"
                else "other_stage"
            ),
        }
        rows.append(row)

        pe = yf_info.get("trailingPE")
        roe = yf_info.get("returnOnEquity")
        de = yf_info.get("debtToEquity")
        pg = yf_info.get("profitMargins")
        rg = yf_info.get("revenueGrowth")
        eg = yf_info.get("earningsGrowth")
        print(
            f"  {sym:12} {db.get('sector') or yf_info.get('sector')}  "
            f"PE={pe} ROE={None if roe is None else round(roe*100,1)} "
            f"D/E={de} mgn={None if pg is None else round(pg*100,1)} "
            f"revG={rg} earnG={eg}  "
            f"sma150={path.get('pct_vs_sma150')} 3m={path.get('ret_3m')} rs3={rs3} "
            f"mfe={path.get('mfe_pct')} mae={path.get('mae_pct')} nifty={mkt.get('nifty_hold_pct')}"
        )

    out = ROOT / "data" / "_stage_rs70_loser_autopsy.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
