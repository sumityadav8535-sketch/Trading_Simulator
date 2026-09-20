"""Hunt a Nifty-200 fundamental + technical pack that can double in the last year.

Universe: Nifty 200. Fundamentals are point-in-time (FY + 90d lag).
Technicals: SMA50/150, EMA20, RSI, 3m/6m momentum, RS vs Nifty, 52w high.
No look-ahead: rank and enter on the rebalance close.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading.constants import NIFTY50_SYMBOL  # noqa: E402
from trading.services.fundamental_swing import (  # noqa: E402
    DEFAULT_CAPITAL,
    FundParams,
    load_cache,
    load_close_series,
    rank_at_date,
)
from trading.services.market_data import get_universe_symbols  # noqa: E402

END = date.today()
START_1Y = END - timedelta(days=365)
START_2Y = END - timedelta(days=365 * 2)
START_3Y = END - timedelta(days=365 * 3)
CAPITAL = DEFAULT_CAPITAL
TARGET = 100.0

LOOSE = FundParams(
    name="loose", min_roe=10, min_profit_margin=4, min_revenue_growth=5,
    min_earnings_growth=8, max_pe=80, max_peg=5, max_de=2.0, min_score=30, top_n=50,
)
GROWTH = FundParams(
    name="growth", min_roe=12, min_profit_margin=5, min_revenue_growth=15,
    min_earnings_growth=20, max_pe=70, max_peg=4, max_de=1.8, min_score=35, top_n=50,
)
TIGHT = FundParams(
    name="tight", min_roe=12, min_profit_margin=5, min_revenue_growth=20,
    min_earnings_growth=25, max_pe=60, max_peg=3, max_de=1.5, min_score=40, top_n=50,
)
FUND_PACKS = {"loose": LOOSE, "growth": GROWTH, "tight": TIGHT}


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def first_of_month(days: list[date], start: date, end: date) -> list[date]:
    out: list[date] = []
    last_ym = None
    for d in days:
        if d < start or d > end:
            continue
        ym = (d.year, d.month)
        if ym != last_ym:
            out.append(d)
            last_ym = ym
    if out and out[0] != start:
        out = [start] + out
    elif not out:
        out = [start]
    return out


def quarterly(days: list[date], start: date, end: date) -> list[date]:
    monthly = first_of_month(days, start, end)
    return [d for i, d in enumerate(monthly) if i == 0 or d.month in (1, 4, 7, 10)]


def every_n(days: list[date], start: date, end: date, n: int) -> list[date]:
    window = [d for d in days if start <= d <= end]
    if not window:
        return [start]
    return window[::n]


def diagnose_winners(closes: dict[str, pd.Series], nifty: pd.Series, start: date, end: date) -> None:
    rows = []
    for sym, s in closes.items():
        if sym == NIFTY50_SYMBOL:
            continue
        a = s.loc[: pd.Timestamp(start)]
        b = s.loc[: pd.Timestamp(end)]
        if a.empty or b.empty:
            continue
        pa, pb = float(a.iloc[-1]), float(b.iloc[-1])
        if pa <= 0:
            continue
        ret = (pb / pa - 1.0) * 100.0
        rows.append((ret, sym, pa, pb))
    rows.sort(reverse=True)
    na = nifty.loc[: pd.Timestamp(start)]
    nb = nifty.loc[: pd.Timestamp(end)]
    nret = (float(nb.iloc[-1]) / float(na.iloc[-1]) - 1.0) * 100.0 if not na.empty and not nb.empty else None
    print(f"\n===== Buy-hold {start} → {end}  Nifty={nret:.1f}% =====")
    print(f"{'ret':>8}  {'sym':<12}  {'start':>10}  {'end':>10}")
    for ret, sym, pa, pb in rows[:25]:
        print(f"{ret:8.1f}  {sym:<12}  {pa:10.2f}  {pb:10.2f}")
    n100 = sum(1 for r, *_ in rows if r >= 100)
    n50 = sum(1 for r, *_ in rows if r >= 50)
    print(f"doubled={n100}  >=50%={n50}  of {len(rows)}")


def build_tech(closes: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, s in closes.items():
        if s is None or s.empty or len(s) < 160:
            continue
        df = pd.DataFrame({"close": s.astype(float)})
        df["sma20"] = _sma(df["close"], 20)
        df["sma50"] = _sma(df["close"], 50)
        df["sma150"] = _sma(df["close"], 150)
        df["ema20"] = _ema(df["close"], 20)
        df["rsi"] = _rsi(df["close"], 14)
        df["ret63"] = df["close"].pct_change(63) * 100.0
        df["ret126"] = df["close"].pct_change(126) * 100.0
        df["high252"] = df["close"].rolling(252, min_periods=120).max()
        df["off_high"] = (df["close"] / df["high252"] - 1.0) * 100.0
        df["ext_ema"] = (df["close"] / df["ema20"] - 1.0) * 100.0
        out[sym] = df
    return out


def row_at(df: pd.DataFrame, d: date) -> pd.Series | None:
    sliced = df.loc[: pd.Timestamp(d)]
    if sliced.empty:
        return None
    return sliced.iloc[-1]


def passes_tech(row: pd.Series, nifty_row: pd.Series | None, kind: str) -> bool:
    close = float(row["close"])
    sma150 = row.get("sma150")
    sma50 = row.get("sma50")
    ema20 = row.get("ema20")
    rsi = row.get("rsi")
    off_high = row.get("off_high")
    ext = row.get("ext_ema")
    ret126 = row.get("ret126")
    if kind == "none":
        return True
    if pd.isna(sma150) or close <= float(sma150):
        return False
    if kind == "sma150":
        return True
    if kind in ("trend", "trend_rs", "pullback", "breakout", "trend_pull"):
        if pd.isna(sma50) or float(sma50) <= float(sma150):
            return False
        if close <= float(sma50):
            return False
    if kind in ("trend_rs",) and nifty_row is not None and not pd.isna(ret126) and not pd.isna(nifty_row.get("ret126")):
        if float(ret126) <= float(nifty_row["ret126"]):
            return False
    if kind in ("pullback", "trend_pull"):
        if pd.isna(rsi) or not (40 <= float(rsi) <= 68):
            return False
        if pd.isna(ext) or float(ext) > 8:
            return False
        if pd.isna(ema20) or close < float(ema20) * 0.97:
            return False
    if kind == "breakout":
        if pd.isna(off_high) or float(off_high) < -8:
            return False
        if pd.isna(rsi) or float(rsi) < 50:
            return False
    return True


def rank_value(row: pd.Series, nifty_row: pd.Series | None, fund_score: float, kind: str) -> float:
    mom3 = float(row["ret63"]) if not pd.isna(row.get("ret63")) else -999
    mom6 = float(row["ret126"]) if not pd.isna(row.get("ret126")) else -999
    rs6 = mom6
    if nifty_row is not None and not pd.isna(nifty_row.get("ret126")):
        rs6 = mom6 - float(nifty_row["ret126"])
    if kind == "score":
        return fund_score
    if kind == "mom3":
        return mom3
    if kind == "mom6":
        return mom6
    if kind == "rs6":
        return rs6
    if kind == "score_mom":
        return fund_score * 0.4 + max(mom6, 0) * 0.6
    if kind == "score_rs":
        return fund_score * 0.35 + max(rs6, 0) * 0.65
    return fund_score


def simulate(
    days: list[date],
    reb_dates: list[date],
    tech: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    fund_by_reb: dict[date, list[dict]],
    *,
    top_n: int,
    rank_kind: str,
    tech_kind: str,
    trail_sma: int,
    capital: float,
) -> dict:
    if not days:
        return {"total_return_pct": None, "error": "no days"}
    cash = float(capital)
    holdings: list[dict] = []
    peak = capital
    max_dd = 0.0
    trades: list[dict] = []
    reb_set = set(reb_dates)
    next_reb = 0

    def mark(d: date) -> float:
        eq = cash
        for h in holdings:
            df = tech.get(h["symbol"])
            row = row_at(df, d) if df is not None else None
            px = float(row["close"]) if row is not None else h["last_px"]
            h["last_px"] = px
            eq += h["qty"] * px
        return eq

    def close_one(h: dict, d: date, reason: str) -> float:
        df = tech.get(h["symbol"])
        row = row_at(df, d) if df is not None else None
        px = float(row["close"]) if row is not None else h["last_px"]
        proceeds = h["qty"] * px
        ret = (px / h["entry_px"] - 1.0) * 100.0 if h["entry_px"] else None
        trades.append({
            "symbol": h["symbol"], "entry_date": h["entry_date"], "exit_date": d.isoformat(),
            "return_pct": None if ret is None else round(ret, 2), "reason": reason,
            "doubled": bool(ret is not None and ret >= 100),
        })
        return proceeds

    def liquidate(d: date, reason: str) -> None:
        nonlocal cash, holdings
        for h in holdings:
            cash += close_one(h, d, reason)
        holdings = []

    def pick(d: date) -> list[dict]:
        fund_rows = fund_by_reb.get(d) or []
        nifty_row = row_at(nifty_df, d)
        scored = []
        for p in fund_rows:
            df = tech.get(p["symbol"])
            if df is None:
                continue
            row = row_at(df, d)
            if row is None:
                continue
            if not passes_tech(row, nifty_row, tech_kind):
                continue
            scored.append((rank_value(row, nifty_row, float(p.get("score") or 0), rank_kind), p, float(row["close"])))
        scored.sort(key=lambda t: (-t[0], t[1]["symbol"]))
        return scored[:top_n]

    def enter(d: date, scored) -> None:
        nonlocal cash, holdings
        priced = [(p, px) for _, p, px in scored if px > 0]
        if not priced or cash <= 0:
            holdings = []
            return
        slice_amt = cash / len(priced)
        new = []
        spent = 0.0
        for p, px in priced:
            qty = slice_amt / px
            spent += slice_amt
            new.append({
                "symbol": p["symbol"], "entry_date": d.isoformat(),
                "entry_px": px, "qty": qty, "last_px": px,
            })
        cash -= spent
        holdings = new

    trail_col = {20: "sma20", 50: "sma50", 150: "sma150"}.get(trail_sma)

    for d in days:
        while next_reb < len(reb_dates) and reb_dates[next_reb] <= d:
            if holdings:
                liquidate(d, "rebalance")
            enter(d, pick(reb_dates[next_reb]))
            next_reb += 1
        if trail_col and holdings:
            keep = []
            for h in holdings:
                df = tech.get(h["symbol"])
                row = row_at(df, d) if df is not None else None
                if row is None or pd.isna(row.get(trail_col)):
                    keep.append(h)
                    continue
                if float(row["close"]) < float(row[trail_col]):
                    cash += close_one(h, d, "trail")
                else:
                    keep.append(h)
            holdings = keep
        eq = mark(d)
        if eq > peak:
            peak = eq
        if peak > 0:
            max_dd = min(max_dd, eq / peak - 1.0)

    if holdings:
        liquidate(days[-1], "window_end")
    final = cash
    ret = (final / capital - 1.0) * 100.0
    rets = [t["return_pct"] for t in trades if t.get("return_pct") is not None]
    wins = sum(1 for r in rets if r > 0)
    return {
        "total_return_pct": round(ret, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "trades": len(trades),
        "win_rate": round(wins / len(rets) * 100.0, 1) if rets else None,
        "doublers": sum(1 for t in trades if t.get("doubled")),
        "hit_100": ret >= TARGET,
        "final_equity": round(final, 2),
        "holdings_history": trades,
    }


def nearest_reb(reb_dates: list[date], d: date) -> date:
    chosen = reb_dates[0]
    for r in reb_dates:
        if r <= d:
            chosen = r
        else:
            break
    return chosen


def main() -> None:
    t0 = time.time()
    cache = load_cache()
    symbols = get_universe_symbols(nifty200_only=True)
    cached = [s for s in symbols if (cache.get("stocks") or {}).get(s)]
    print(f"Universe {len(symbols)}  cached {len(cached)}", flush=True)
    closes = load_close_series(cached + [NIFTY50_SYMBOL])
    nifty = closes.get(NIFTY50_SYMBOL, pd.Series(dtype=float))
    diagnose_winners(closes, nifty, START_1Y, END)

    tech = build_tech(closes)
    nifty_df = tech.get(NIFTY50_SYMBOL)
    if nifty_df is None:
        raise SystemExit("No Nifty series")
    cal = sorted({ts.date() for df in tech.values() for ts in df.index})
    days_1y = [d for d in cal if START_1Y <= d <= END]
    print(f"tech frames={len(tech)}  1y days={len(days_1y)}  load {time.time()-t0:.1f}s", flush=True)

    # Precompute fund passers at monthly dates covering 3y.
    months = first_of_month(cal, START_3Y, END)
    fund_cache: dict[str, dict[date, list[dict]]] = {k: {} for k in FUND_PACKS}
    print(f"Scoring fundamentals on {len(months)} month-starts…", flush=True)
    for i, d in enumerate(months):
        for name, params in FUND_PACKS.items():
            passed, _ = rank_at_date(cache, cached, d, params, closes, use_info=False)
            fund_cache[name][d] = passed
        if (i + 1) % 6 == 0 or i == len(months) - 1:
            print(f"  {i+1}/{len(months)}", flush=True)

    def map_reb(reb_dates: list[date], pack: str) -> dict[date, list[dict]]:
        src = fund_cache[pack]
        keys = sorted(src)
        out = {}
        for d in reb_dates:
            chosen = keys[0]
            for k in keys:
                if k <= d:
                    chosen = k
                else:
                    break
            out[d] = src[chosen]
        return out

    freq_builders = {
        "monthly": lambda days, a, b: first_of_month(days, a, b),
        "quarterly": lambda days, a, b: quarterly(days, a, b),
        "21d": lambda days, a, b: every_n(days, a, b, 21),
    }
    grid = list(product(
        FUND_PACKS.keys(),
        freq_builders.keys(),
        (1, 2, 3),
        ("mom6", "rs6", "score_mom", "score_rs", "mom3"),
        ("sma150", "trend", "trend_rs", "pullback", "breakout", "trend_pull"),
        (0, 20, 50),
    ))
    print(f"Grid {len(grid)} on last 1y…", flush=True)

    results = []
    for i, (fp, freq, top_n, rank_k, tech_k, trail) in enumerate(grid):
        reb = freq_builders[freq](cal, START_1Y, END)
        fund_map = map_reb(reb, fp)
        run = simulate(
            days_1y, reb, tech, nifty_df, fund_map,
            top_n=top_n, rank_kind=rank_k, tech_kind=tech_k, trail_sma=trail, capital=CAPITAL,
        )
        rec = {
            "fund": fp, "freq": freq, "top_n": top_n, "rank": rank_k,
            "tech": tech_k, "trail": trail, **{k: run[k] for k in (
                "total_return_pct", "max_drawdown_pct", "trades", "win_rate", "doublers", "hit_100",
            )},
        }
        results.append(rec)
        if (i + 1) % 80 == 0:
            best = max(results, key=lambda r: r["total_return_pct"] or -999)
            print(f"  {i+1}/{len(grid)} best so far {best['total_return_pct']}% {best}", flush=True)

    results.sort(key=lambda r: (-(r["total_return_pct"] or -999), r["max_drawdown_pct"] or 0))
    print("\n===== TOP last-1y =====")
    for r in results[:25]:
        print(
            f"{r['total_return_pct']:7.1f}%  dd={r['max_drawdown_pct']:6.1f}  "
            f"n={r['top_n']} {r['freq']:<9} {r['fund']:<6} {r['rank']:<10} {r['tech']:<11} "
            f"trail{r['trail']:<3} wr={r['win_rate']}  trades={r['trades']}  dbl={r['doublers']}"
        )
    hits = [r for r in results if r.get("hit_100")]
    print(f"\nHits >=100%: {len(hits)} / {len(results)}")

    # Validate top hits (or top 8) on 2y/3y and prior 1y slices.
    candidates = hits[:8] or results[:8]
    print("\n===== Multi-window check =====")
    windows = [
        ("1y", START_1Y, END),
        ("2y", START_2Y, END),
        ("3y", START_3Y, END),
        ("23-24", date(2023, 9, 16), date(2024, 9, 16)),
        ("24-25", date(2024, 9, 16), date(2025, 9, 16)),
        ("25-26", date(2025, 9, 16), END),
    ]
    for c in candidates:
        parts = []
        for label, a, b in windows:
            days = [d for d in cal if a <= d <= b]
            reb = freq_builders[c["freq"]](cal, a, b)
            fund_map = map_reb(reb, c["fund"])
            run = simulate(
                days, reb, tech, nifty_df, fund_map,
                top_n=c["top_n"], rank_kind=c["rank"], tech_kind=c["tech"],
                trail_sma=c["trail"], capital=CAPITAL,
            )
            parts.append(f"{label}={run['total_return_pct']}%/{run['max_drawdown_pct']}dd")
        print(
            f"n={c['top_n']} {c['freq']} {c['fund']} {c['rank']} {c['tech']} trail{c['trail']}  "
            + "  ".join(parts),
            flush=True,
        )

    # Show last 1y holdings for the best hit (or best overall).
    best = (hits[0] if hits else results[0])
    reb = freq_builders[best["freq"]](cal, START_1Y, END)
    fund_map = map_reb(reb, best["fund"])
    run = simulate(
        days_1y, reb, tech, nifty_df, fund_map,
        top_n=best["top_n"], rank_kind=best["rank"], tech_kind=best["tech"],
        trail_sma=best["trail"], capital=CAPITAL,
    )
    print("\n===== Best pack last-1y trades =====")
    print(best)
    for t in run["holdings_history"]:
        print(f"  {t['symbol']:<12} {t['entry_date']} → {t['exit_date']}  {t['return_pct']}%  {t['reason']}")
    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
