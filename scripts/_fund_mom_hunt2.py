"""Phase 2: all-Nifty-200 momentum / breakout / let-winners-run hunt for >=100% last 1y."""
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
from trading.services.fundamental_swing import FundParams, load_cache, load_close_series, rank_at_date  # noqa: E402
from trading.services.market_data import get_universe_symbols  # noqa: E402

END = date.today()
START = END - timedelta(days=365)
CAPITAL = 1_000_000.0

LIGHT = FundParams(
    name="light", min_roe=8, min_profit_margin=0, min_revenue_growth=0,
    min_earnings_growth=0, max_pe=120, max_peg=10, max_de=5, min_score=0, top_n=200,
    min_current_ratio=0,
)


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def pct_change(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) <= n:
        return out
    prev = a[:-n]
    out[n:] = np.where(prev > 0, (a[n:] / prev - 1.0) * 100.0, np.nan)
    return out


def rolling_max(a: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(a)
    return s.rolling(n, min_periods=n).max().to_numpy()


def first_of_period(dates: list[date], start: date, end: date, n: int) -> list[int]:
    idxs = [i for i, d in enumerate(dates) if start <= d <= end]
    if not idxs:
        return []
    if n <= 0:
        # monthly
        out = []
        last = None
        for i in idxs:
            ym = (dates[i].year, dates[i].month)
            if ym != last:
                out.append(i)
                last = ym
        return out
    return idxs[::n]


def main() -> None:
    t0 = time.time()
    cache = load_cache()
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if (cache.get("stocks") or {}).get(s)]
    closes = load_close_series(symbols + [NIFTY50_SYMBOL])
    nifty = closes.get(NIFTY50_SYMBOL)
    cal = sorted({ts.date() for s in closes.values() for ts in s.index})
    cal = [d for d in cal if d >= START - timedelta(days=400) and d <= END]
    idx = {d: i for i, d in enumerate(cal)}
    n = len(cal)
    print(f"calendar {cal[0]} → {cal[-1]}  n={n}  symbols={len(symbols)}", flush=True)

    close_m = {}
    for sym, s in closes.items():
        arr = np.full(n, np.nan)
        for ts, px in s.items():
            j = idx.get(ts.date())
            if j is not None:
                arr[j] = float(px)
        # forward-fill small gaps
        srs = pd.Series(arr).ffill()
        close_m[sym] = srs.to_numpy(dtype=float)

    tech = {}
    for sym, a in close_m.items():
        tech[sym] = {
            "c": a,
            "sma20": sma(a, 20),
            "sma50": sma(a, 50),
            "sma150": sma(a, 150),
            "mom21": pct_change(a, 21),
            "mom63": pct_change(a, 63),
            "mom126": pct_change(a, 126),
            "high20": rolling_max(a, 20),
            "high55": rolling_max(a, 55),
        }

    i0 = next(i for i, d in enumerate(cal) if d >= START)
    i1 = len(cal) - 1
    print(f"window {cal[i0]} → {cal[i1]}  bars={i1-i0+1}", flush=True)

    # monthly light-fund sets
    months = first_of_period(cal, START - timedelta(days=40), END, 0)
    light_ok = {}
    none_ok = {}
    for i in months:
        d = cal[i]
        passed, _ = rank_at_date(cache, symbols, d, LIGHT, closes, use_info=False)
        light_ok[i] = {p["symbol"] for p in passed}
        none_ok[i] = set(symbols)
        print(f"  light {d}  {len(light_ok[i])} pass", flush=True)

    def universe_at(i: int, kind: str) -> set[str]:
        src = light_ok if kind == "light" else none_ok
        keys = sorted(src)
        chosen = keys[0]
        for k in keys:
            if k <= i:
                chosen = k
            else:
                break
        return src[chosen]

    def simulate(cfg) -> dict:
        top_n = cfg["top_n"]
        rank = cfg["rank"]
        gate = cfg["gate"]
        trail = cfg["trail"]
        freq = cfg["freq"]
        uni = cfg["uni"]
        entry = cfg["entry"]
        hold_mode = cfg["hold"]
        cash = CAPITAL
        holds: list[dict] = []
        peak = CAPITAL
        dd = 0.0
        trades = []
        reb = set(first_of_period(cal, cal[i0], cal[i1], freq))

        def px(sym, i):
            v = tech[sym]["c"][i]
            return None if np.isnan(v) or v <= 0 else float(v)

        def liq(i, reason):
            nonlocal cash, holds
            for h in holds:
                p = px(h["s"], i) or h["last"]
                ret = (p / h["epx"] - 1.0) * 100.0
                trades.append({"s": h["s"], "ret": round(ret, 2), "reason": reason,
                               "a": cal[h["ei"]].isoformat(), "b": cal[i].isoformat()})
                cash += h["qty"] * p
            holds = []

        def leaders(i):
            allowed = universe_at(i, uni)
            scored = []
            nifty_m = tech[NIFTY50_SYMBOL][rank][i] if NIFTY50_SYMBOL in tech else np.nan
            for sym in allowed:
                t = tech.get(sym)
                if t is None:
                    continue
                c = t["c"][i]
                if np.isnan(c) or c <= 0:
                    continue
                if gate == "sma150" and (np.isnan(t["sma150"][i]) or c <= t["sma150"][i]):
                    continue
                if gate == "sma50" and (np.isnan(t["sma50"][i]) or c <= t["sma50"][i]):
                    continue
                if gate == "trend" and (np.isnan(t["sma50"][i]) or np.isnan(t["sma150"][i]) or not (c > t["sma50"][i] > t["sma150"][i])):
                    continue
                if gate == "rs" and (np.isnan(t["mom63"][i]) or np.isnan(nifty_m) or t["mom63"][i] <= nifty_m):
                    continue
                if entry == "break20" and (np.isnan(t["high20"][i]) or c < t["high20"][i] * 0.999):
                    continue
                if entry == "break55" and (np.isnan(t["high55"][i]) or c < t["high55"][i] * 0.999):
                    continue
                score = t[rank][i]
                if np.isnan(score):
                    continue
                scored.append((float(score), sym, float(c)))
            scored.sort(reverse=True)
            return scored[:top_n]

        def enter(i, scored):
            nonlocal cash, holds
            if not scored or cash <= 0:
                holds = []
                return
            sl = cash / len(scored)
            new = []
            spent = 0.0
            for _, sym, p in scored:
                new.append({"s": sym, "epx": p, "qty": sl / p, "last": p, "ei": i})
                spent += sl
            cash -= spent
            holds = new

        for i in range(i0, i1 + 1):
            if hold_mode == "reb" and i in reb:
                if holds:
                    liq(i, "reb")
                enter(i, leaders(i))
            elif hold_mode == "run":
                # trail first
                if holds:
                    keep = []
                    for h in holds:
                        t = tech[h["s"]]
                        c = t["c"][i]
                        line = t[trail][i] if trail in t else np.nan
                        if np.isnan(c) or np.isnan(line) or c < line:
                            p = c if not np.isnan(c) else h["last"]
                            ret = (p / h["epx"] - 1.0) * 100.0
                            trades.append({"s": h["s"], "ret": round(ret, 2), "reason": "trail",
                                           "a": cal[h["ei"]].isoformat(), "b": cal[i].isoformat()})
                            cash += h["qty"] * p
                        else:
                            keep.append(h)
                    holds = keep
                if not holds:
                    enter(i, leaders(i))
                elif i in reb:
                    # rotate only if a new leader is clearly better
                    lead = leaders(i)
                    cur = {h["s"] for h in holds}
                    new_syms = [s for _, s, _ in lead]
                    if new_syms and set(new_syms) != cur:
                        liq(i, "rot")
                        enter(i, lead)
            eq = cash
            for h in holds:
                p = px(h["s"], i) or h["last"]
                h["last"] = p
                eq += h["qty"] * p
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = min(dd, eq / peak - 1.0)

        if holds:
            liq(i1, "end")
        ret = (cash / CAPITAL - 1.0) * 100.0
        rets = [t["ret"] for t in trades]
        wins = sum(1 for r in rets if r > 0)
        return {
            "ret": round(ret, 2),
            "dd": round(dd * 100.0, 2),
            "trades": len(trades),
            "wr": round(wins / len(rets) * 100.0, 1) if rets else None,
            "dbl": sum(1 for r in rets if r >= 100),
            "hit": ret >= 100,
            "trades_log": trades,
        }

    grid = []
    for uni, freq, top_n, rank, gate, entry, hold, trail in product(
        ("none", "light"),
        (5, 10, 21, 0),
        (1, 2),
        ("mom21", "mom63", "mom126"),
        ("sma50", "sma150", "trend"),
        ("close", "break20"),
        ("reb", "run"),
        ("sma20", "sma50"),
    ):
        if hold == "reb" and trail != "sma20":
            # trail unused in reb mode; keep one
            if trail != "sma20":
                continue
        grid.append({
            "uni": uni, "freq": freq, "top_n": top_n, "rank": rank, "gate": gate,
            "entry": entry, "hold": hold, "trail": trail if hold == "run" else "none",
        })

    print(f"Grid {len(grid)}", flush=True)
    results = []
    best = -999
    for i, cfg in enumerate(grid):
        run = simulate(cfg)
        rec = {**cfg, **{k: run[k] for k in ("ret", "dd", "trades", "wr", "dbl", "hit")}}
        rec["_log"] = run["trades_log"]
        results.append(rec)
        if run["ret"] > best:
            best = run["ret"]
        if (i + 1) % 60 == 0:
            top = max(results, key=lambda r: r["ret"])
            print(f"  {i+1}/{len(grid)} best {top['ret']}%  { {k: top[k] for k in top if k != '_log'} }", flush=True)

    results.sort(key=lambda r: (-r["ret"], r["dd"]))
    print("\n===== TOP =====")
    for r in results[:20]:
        print(
            f"{r['ret']:7.1f}% dd={r['dd']:6.1f} n={r['top_n']} f={r['freq']!s:<3} "
            f"{r['uni']:<5} {r['rank']:<7} {r['gate']:<7} {r['entry']:<7} {r['hold']:<3} "
            f"{r['trail']:<5} wr={r['wr']} tr={r['trades']} dbl={r['dbl']}"
        )
    hits = [r for r in results if r["hit"]]
    print(f"\nHits >=100%: {len(hits)}")
    show = hits[:5] or results[:3]
    for r in show:
        print("\n---", {k: r[k] for k in r if k != "_log"})
        for t in r["_log"]:
            print(f"  {t['s']:<12} {t['a']} → {t['b']}  {t['ret']}%  {t['reason']}")
    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
