"""
Nifty 200 gap backtest — last ~2 months of 5m bars.

Yesterday close 100, today opens 80–90 (gap down) or 110–120 (gap up).

Modes
  down_bounce   long gap-downs, target prior close (fill)
  up_fade       short gap-ups, target prior close (fill)
  both_fill     fade both ways toward prior close
  always_long   long any gap ("it will go up")
  gap_and_go    trade in the gap direction, 1.5R

Entry at the 9:15 open (gap is known). SL-first, costs on, flatten 15:15.

    python scripts/intraday_gap_hunt.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django  # noqa: E402

django.setup()

from scripts.intraday_15m_n200_search import (  # noqa: E402
    CAPITAL,
    COST,
    FORCE_EXIT,
    SimResult,
    _summarize,
    add_indicators,
    attach_daily_bias,
    nifty200_symbols,
)
from scripts.intraday_5m_hunt import load_frames  # noqa: E402
from trading.services.indicators import _rsi  # noqa: E402
from trading.services.market_data import load_price_dataframe  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_hunt.json"
TRADES = ROOT / "data" / "intraday_gap_trades.json"

LEVERAGE = 5.0
STORY_SIZE = dict(risk_pct=5.0, max_pos=4, top_k=2, max_deploy=0.35)


def attach_rsi(events: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame] = {}
    rsi_vals = []
    for r in events.itertuples(index=False):
        if r.symbol not in cache:
            d = load_price_dataframe(r.symbol)
            if d.empty or len(d) < 20:
                cache[r.symbol] = pd.DataFrame()
            else:
                d = d.copy()
                d["rsi14"] = _rsi(d["close"], 14)
                d.index = pd.to_datetime(d.index).date
                cache[r.symbol] = d[["rsi14"]]
        frame = cache[r.symbol]
        val = None
        if not frame.empty and r.session in frame.index:
            prior = frame[frame.index < r.session]
            if not prior.empty:
                val = float(prior["rsi14"].iloc[-1])
        rsi_vals.append(val)
    out = events.copy()
    out["rsi14"] = rsi_vals
    return out


def prepare(cache: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    stocks = {}
    items = [(k, v) for k, v in cache.items() if not k.startswith("_")]
    for i, (sym, df) in enumerate(items, 1):
        en = add_indicators(df)
        en = attach_daily_bias(en, sym)
        if len(en) >= 80:
            stocks[sym] = en
        if i % 50 == 0:
            print(f"  prepared {i}/{len(items)}", flush=True)
    return stocks


def session_first_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("session", sort=True).head(1)


def session_prior_close_at(df: pd.DataFrame, hour: int = 13, minute: int = 10) -> pd.Series:
    """Previous session close at hh:mm (else last bar at or before that time).

    Drops closing-auction 5m bars (wide range, close at the high).
    """
    if df is None or df.empty or "session" not in df.columns:
        return pd.Series(dtype=float)
    work = df
    close = work["close"].replace(0, np.nan)
    rng = (work["high"] - work["low"]) / close
    auction = (rng >= 0.012) & (work["close"] >= work["high"] * 0.997)
    work = work.loc[~auction.fillna(False)]
    if work.empty:
        return pd.Series(dtype=float)
    times = work.index
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert("Asia/Kolkata")
    hm = np.asarray(times.hour) * 60 + np.asarray(times.minute)
    cutoff = hour * 60 + minute
    exact = work.loc[hm == cutoff].groupby("session", sort=True)["close"].last()
    before = work.loc[hm <= cutoff].groupby("session", sort=True)["close"].last()
    closes = exact.combine_first(before).sort_index()
    return closes.shift(1)


def session_prior_1310_close(df: pd.DataFrame, hour: int = 13, minute: int = 10) -> pd.Series:
    """Previous session close at 13:10 (else last bar at or before 13:10). Ignores the auction."""
    return session_prior_close_at(df, hour, minute)


def collect_events(stocks: dict[str, pd.DataFrame], pdc_mode: str = "13:10") -> pd.DataFrame:
    """
    pdc_mode: 'official' uses yesterday's daily close; otherwise a 5m clock like '13:10'.
    """
    rows = []
    use_official = str(pdc_mode).lower() in {"official", "daily", "eod"}
    hhmm = None if use_official else str(pdc_mode)
    hour = minute = 13
    if hhmm:
        parts = hhmm.split(":")
        hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    for sym, df in stocks.items():
        prior_close = None if use_official else session_prior_close_at(df, hour, minute)
        first = session_first_rows(df)
        for ts, row in first.iterrows():
            sess = row["session"]
            if use_official:
                raw_pdc = row.get("pdc")
            else:
                raw_pdc = prior_close.get(sess) if prior_close is not None and len(prior_close) else None
            try:
                pdc = float(raw_pdc) if raw_pdc is not None and pd.notna(raw_pdc) else 0.0
            except (TypeError, ValueError):
                pdc = 0.0
            o = float(row["open"])
            if pdc <= 0 or o <= 0 or o < 60:
                continue
            # Same ATR as live: last 5m ATR of the previous session, not today's 9:15 bar.
            prev = df[df["session"] < sess]
            atr = 0.0
            if not prev.empty and "atr" in prev.columns:
                try:
                    atr = float(prev["atr"].iloc[-1])
                except (TypeError, ValueError):
                    atr = 0.0
            if atr <= 0:
                atr = float(row.get("atr") or 0)
            if atr <= 0:
                atr = o * 0.008
            gap = o / pdc - 1.0
            try:
                pdh = float(row.get("pdh") or 0)
            except (TypeError, ValueError):
                pdh = 0.0
            rows.append({
                "symbol": sym,
                "ts": ts,
                "session": row["session"],
                "open": o,
                "pdc": pdc,
                "pdh": pdh,
                "gap": gap,
                "atr": atr,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close1": float(row["close"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ts", "gap"])


def fill_stats(events: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> list[dict]:
    """How often a gap-down actually bounces / fills same day."""
    buckets = [(0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.15)]
    out = []
    for lo, hi in buckets:
        down = events[(events["gap"] <= -lo) & (events["gap"] > -hi)]
        up = events[(events["gap"] >= lo) & (events["gap"] < hi)]
        def _scan(ev, side):
            if ev.empty:
                return dict(n=0, fill_pct=0, bounce_pct=0, avg_mfe=0, avg_mae=0)
            fills = bounces = 0
            mfes, maes = [], []
            for r in ev.itertuples(index=False):
                df = stocks.get(r.symbol)
                if df is None:
                    continue
                day = df[df["session"] == r.session]
                if day.empty:
                    continue
                o, pdc = r.open, r.pdc
                if side == "down":
                    mfe = (day["high"].max() / o - 1.0) * 100
                    mae = (day["low"].min() / o - 1.0) * 100
                    filled = bool(day["high"].max() >= pdc)
                    bounced = bool(day["close"].iloc[-1] > o)
                else:
                    mfe = (1.0 - day["low"].min() / o) * 100
                    mae = (day["high"].max() / o - 1.0) * 100
                    filled = bool(day["low"].min() <= pdc)
                    bounced = bool(day["close"].iloc[-1] < o)
                fills += int(filled)
                bounces += int(bounced)
                mfes.append(mfe)
                maes.append(mae)
            n = max(len(mfes), 1)
            return dict(
                n=len(mfes),
                fill_pct=round(100.0 * fills / n, 1) if mfes else 0,
                bounce_pct=round(100.0 * bounces / n, 1) if mfes else 0,
                avg_mfe=round(float(np.mean(mfes)), 2) if mfes else 0,
                avg_mae=round(float(np.mean(maes)), 2) if mfes else 0,
            )
        d = _scan(down, "down")
        u = _scan(up, "up")
        out.append({
            "bucket": f"{int(lo*100)}–{int(hi*100)}%",
            "down_n": d["n"], "down_fill_pct": d["fill_pct"], "down_close_green_pct": d["bounce_pct"],
            "down_avg_mfe_pct": d["avg_mfe"], "down_avg_mae_pct": d["avg_mae"],
            "up_n": u["n"], "up_fill_pct": u["fill_pct"], "up_close_red_pct": u["bounce_pct"],
            "up_avg_mfe_pct": u["avg_mfe"], "up_avg_mae_pct": u["avg_mae"],
        })
    return out


def make_signals(events: pd.DataFrame, mode: str, gap_min: float, gap_max: float, sl_atr: float, target: str) -> pd.DataFrame:
    rows = []
    for r in events.itertuples(index=False):
        gap = r.gap
        if abs(gap) < gap_min or abs(gap) > gap_max:
            continue
        o, pdc, atr = r.open, r.pdc, r.atr
        half = o + 0.5 * (pdc - o)
        if mode == "down_bounce" or (mode == "both_fill" and gap < 0) or (mode == "always_long" and gap < 0):
            if gap >= 0:
                continue
            side = "long"
            stop = o - sl_atr * atr
            tgt = pdc if target == "fill" else (half if target == "half" else o + sl_atr * atr * 1.5)
            score = abs(gap)
        elif mode == "up_fade" or (mode == "both_fill" and gap > 0):
            if gap <= 0:
                continue
            side = "short"
            stop = o + sl_atr * atr
            tgt = pdc if target == "fill" else (half if target == "half" else o - sl_atr * atr * 1.5)
            score = abs(gap)
        elif mode == "always_long" and gap > 0:
            side = "long"
            stop = o - sl_atr * atr
            tgt = o + sl_atr * atr * 1.5 if target != "fill" else o + abs(o - pdc)
            score = abs(gap)
        elif mode == "gap_and_go":
            if gap > 0:
                side = "long"
                stop = min(r.low, o - sl_atr * atr)
                tgt = o + max(o - stop, sl_atr * atr) * 1.5
            else:
                side = "short"
                stop = max(r.high, o + sl_atr * atr)
                tgt = o - max(stop - o, sl_atr * atr) * 1.5
            score = abs(gap)
        else:
            continue
        if side == "long" and not (tgt > o > stop):
            continue
        if side == "short" and not (tgt < o < stop):
            continue
        rec = {
            "ts": r.ts, "symbol": r.symbol, "side": side,
            "entry": o, "stop": stop, "target": tgt, "score": score, "gap": gap,
            "pdc": pdc,
        }
        if hasattr(r, "rsi14"):
            rec["rsi"] = r.rsi14
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "side", "entry", "stop", "target", "score", "gap", "pdc", "rsi"])
    return pd.DataFrame(rows)


def build_sim_book(stocks: dict[str, pd.DataFrame]) -> dict:
    ohlc = {}
    calendar = set()
    last_bar = {}
    sess_bars: dict[tuple, list] = {}
    for sym, df in stocks.items():
        idx = df.index
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        last_bar[sym] = (idx[-1], float(c[-1]))
        for i, ts in enumerate(idx):
            calendar.add(ts)
            ohlc[(sym, ts)] = (o[i], h[i], l[i], c[i])
            d = ts.date() if hasattr(ts, "date") else ts
            sess_bars.setdefault((sym, d), []).append(ts)
    return {
        "ohlc": ohlc,
        "calendar": sorted(calendar),
        "last_bar": last_bar,
        "sess_bars": sess_bars,
    }


def simulate_open(
    stocks: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    name: str,
    risk_pct: float = 5.0,
    max_pos: int = 4,
    top_k: int = 2,
    max_deploy: float = 0.35,
    leverage: float = LEVERAGE,
    book: dict | None = None,
    cost: float | None = None,
) -> tuple[SimResult, pd.DataFrame]:
    if signals is None or signals.empty:
        return SimResult(name=name), pd.DataFrame()

    sigs = signals.copy()
    sigs["day"] = pd.to_datetime(sigs["ts"]).dt.date
    sigs["rank"] = sigs.groupby("day")["score"].rank(method="first", ascending=False)
    if top_k is not None and int(top_k) > 0:
        sigs = sigs[sigs["rank"] <= int(top_k)]

    packed = book or build_sim_book(stocks)
    ohlc = packed["ohlc"]
    calendar = packed["calendar"]
    last_bar = packed["last_bar"]
    sess_bars = packed["sess_bars"]
    fee = COST if cost is None else float(cost)

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    cash = []
    open_pos: list[dict] = []
    daily_pnl: dict = {}
    taken_day: set = set()

    by_day: dict = {}
    for rec in sigs.itertuples(index=False):
        by_day.setdefault(rec.day, []).append(rec)

    def close_pos(pos, ts, raw, reason):
        nonlocal equity, peak, max_dd
        exit_p = raw * (1 - fee) if pos["side"] == "long" else raw * (1 + fee)
        if pos["side"] == "long":
            pnl = (exit_p - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - exit_p) * pos["qty"]
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
        d = ts.date() if hasattr(ts, "date") else ts
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
        rsi = pos.get("rsi")
        try:
            rsi_out = None if rsi is None or (isinstance(rsi, float) and np.isnan(rsi)) else round(float(rsi), 1)
        except (TypeError, ValueError):
            rsi_out = None
        pdc = pos.get("pdc")
        try:
            pdc_out = None if pdc is None else round(float(pdc), 2)
        except (TypeError, ValueError):
            pdc_out = None
        cash.append({
            "symbol": pos["sym"], "side": pos["side"], "gap": round(pos["gap"] * 100, 2),
            "entry_ts": str(pos["entry_ts"]), "exit_ts": str(ts),
            "entry": round(pos["entry"], 2), "exit": round(exit_p, 2),
            "stop": round(float(pos["stop"]), 2),
            "target": round(float(pos["target"]), 2),
            "qty": pos["qty"], "pnl": round(pnl, 2), "reason": reason,
            "rsi": rsi_out, "pdc": pdc_out,
        })

    for ts in calendar:
        d = ts.date()
        still = []
        for pos in open_pos:
            bar = ohlc.get((pos["sym"], ts))
            if bar is None:
                still.append(pos)
                continue
            _o, high, low, close = bar
            stop, tgt = pos["stop"], pos["target"]
            if pos["side"] == "long":
                hit_sl, hit_tp = low <= stop, high >= tgt
            else:
                hit_sl, hit_tp = high >= stop, low <= tgt
            # First bar is the entry bar: use conservative SL-first after entry open.
            if ts == pos["entry_ts"]:
                # already in at open; SL/TP can hit later in the same bar
                pass
            if hit_sl:
                close_pos(pos, ts, stop, "sl")
            elif hit_tp:
                close_pos(pos, ts, tgt, "target")
            elif ts.time() >= FORCE_EXIT:
                close_pos(pos, ts, close, "eod")
            else:
                still.append(pos)
        open_pos = still
        held = {p["sym"] for p in open_pos}

        recs = by_day.get(d)
        if not recs:
            continue
        recs = sorted(recs, key=lambda r: -r.score)
        for rec in recs:
            if rec.ts != ts:
                continue
            if max_pos is not None and int(max_pos) > 0 and len(open_pos) >= int(max_pos):
                break
            if rec.symbol in held:
                continue
            if (rec.symbol, d) in taken_day:
                continue
            bars = sess_bars.get((rec.symbol, d))
            if not bars:
                continue
            raw_open = rec.entry
            side = rec.side
            entry = raw_open * (1 + fee) if side == "long" else raw_open * (1 - fee)
            stop = float(rec.stop)
            target = float(rec.target)
            if side == "long" and not (target > entry > stop):
                continue
            if side == "short" and not (target < entry < stop):
                continue
            risk_ps = (entry - stop) if side == "long" else (stop - entry)
            if risk_ps <= 0:
                continue
            bp = equity * leverage
            qty = int((equity * risk_pct / 100.0) / risk_ps)
            cap = int((bp * max_deploy) / entry)
            qty = max(min(qty, cap), 0)
            if qty <= 0:
                continue
            open_pos.append({
                "sym": rec.symbol, "side": side, "entry": entry, "stop": stop,
                "target": target, "qty": qty, "entry_ts": ts, "gap": rec.gap,
                "rsi": getattr(rec, "rsi", None),
                "pdc": getattr(rec, "pdc", None),
            })
            held.add(rec.symbol)
            taken_day.add((rec.symbol, d))

    for pos in open_pos:
        ts_last, last_c = last_bar[pos["sym"]]
        close_pos(pos, ts_last, last_c, "final")

    tdf = pd.DataFrame(cash)
    res = _summarize(name, tdf, equity, max_dd)
    return res, tdf


def months_from_trades(tdf: pd.DataFrame) -> list[dict]:
    if tdf is None or tdf.empty:
        return []
    t = tdf.copy()
    t["month"] = pd.to_datetime(t["exit_ts"]).dt.tz_localize(None).dt.to_period("M").astype(str)
    rows = []
    for m, g in t.groupby("month"):
        rows.append({
            "month": m,
            "trades": int(len(g)),
            "pnl": round(float(g["pnl"].sum()), 2),
            "wr": round(float((g["pnl"] > 0).mean() * 100), 1),
        })
    return rows


def main():
    symbols = nifty200_symbols()
    print(f"Gap hunt | Nifty 200 5m | ₹{CAPITAL:,.0f} | {len(symbols)} names")
    cache = load_frames(symbols)
    n_stocks = sum(1 for k in cache if not k.startswith("_"))
    print(f"Loaded {n_stocks} 5m frames")
    stocks = prepare(cache)
    sample = next(iter(stocks.values()))
    print(f"Window {sample.index.min()} → {sample.index.max()}  stocks={len(stocks)}")

    events = collect_events(stocks)
    print(f"Session opens with prior close: {len(events)}")
    stats = fill_stats(events, stocks)
    print("\n===== Gap fill diagnostics (same day) =====")
    for s in stats:
        print(
            f"  {s['bucket']:>8}  down n={s['down_n']:4d} fill {s['down_fill_pct']:5.1f}% "
            f"close>open {s['down_close_green_pct']:5.1f}%  MFE {s['down_avg_mfe_pct']:5.2f}% MAE {s['down_avg_mae_pct']:5.2f}% "
            f"| up n={s['up_n']:4d} fill {s['up_fill_pct']:5.1f}% close<open {s['up_close_red_pct']:5.1f}%"
        )

    modes = [
        ("down_bounce", "fill"),
        ("down_bounce", "half"),
        ("up_fade", "fill"),
        ("both_fill", "fill"),
        ("both_fill", "half"),
        ("always_long", "fill"),
        ("always_long", "r"),
        ("gap_and_go", "r"),
    ]
    gap_mins = (0.01, 0.02, 0.03, 0.05, 0.08)
    sls = (0.6, 1.0)
    sizes = [
        dict(risk_pct=5.0, max_pos=4, top_k=2, max_deploy=0.35),
        dict(risk_pct=10.0, max_pos=5, top_k=3, max_deploy=0.40),
    ]

    book = []
    total = len(modes) * len(gap_mins) * len(sls) * len(sizes)
    n = 0
    for mode, tgt in modes:
        for gmin in gap_mins:
            for sl in sls:
                sigs = make_signals(events, mode, gmin, 0.15, sl, tgt)
                for sz in sizes:
                    n += 1
                    label = f"{mode} {tgt} g{gmin:.0%} sl{sl} r{sz['risk_pct']:g} p{sz['max_pos']}"
                    res, tdf = simulate_open(stocks, sigs, label, **sz)
                    res.params = dict(mode=mode, target=tgt, gap_min=gmin, sl_atr=sl, **sz)
                    months = months_from_trades(tdf)
                    book.append((res, tdf, months, sigs if tdf is not None and len(tdf) else None))
                    if n % 15 == 0 or n == total:
                        print(f"  {n}/{total}  {res.total_return_pct:7.1f}%  WR {res.win_rate:5.1f}  {label}", flush=True)

    ranked = sorted(
        book,
        key=lambda x: (
            1 if x[0].oos_pnl > 0 else 0,
            x[0].total_return_pct,
            x[0].profit_factor,
            -x[0].max_dd_pct,
        ),
        reverse=True,
    )
    print("\n===== TOP 15 =====")
    print(f"{'name':<52} {'n':>4} {'WR':>5} {'PF':>5} {'ret%':>7} {'PnL':>9} {'OOS':>8} {'DD':>5}")
    for res, _, _, _ in ranked[:15]:
        print(
            f"{res.name[:52]:<52} {res.trades:4d} {res.win_rate:5.1f} {res.profit_factor:5.2f} "
            f"{res.total_return_pct:7.1f} {res.net_pnl:9.0f} {res.oos_pnl:8.0f} {res.max_dd_pct:5.1f}"
        )

    wres, wtdf, wmonths, _ = ranked[0]
    # pick a clean "user story" pack: down_bounce fill, 2% gap, 5% risk
    story = next(
        (x for x in ranked if x[0].params.get("mode") == "down_bounce" and x[0].params.get("target") == "fill"
         and x[0].params.get("gap_min") == 0.02 and x[0].params.get("risk_pct") == 5.0),
        ranked[0],
    )
    payload = {
        "capital": CAPITAL,
        "timeframe": "5m",
        "universe": "nifty200",
        "window": {"start": str(sample.index.min()), "end": str(sample.index.max()), "stocks": len(stocks)},
        "idea": (
            "If a stock closes 100 and opens 80–90 (gap down) or 110–120 (gap up), "
            "trade the gap on the 9:15 open."
        ),
        "fill_stats": stats,
        "winner": {**asdict(wres), "months": wmonths},
        "story": {**asdict(story[0]), "months": story[2]},
        "top": [{**asdict(r), "months": m} for r, _, m, _ in ranked[:12]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if wtdf is not None and not wtdf.empty:
        wtdf.to_json(TRADES, orient="records", date_format="iso")
    print(f"\nWinner {wres.name}")
    print(f"  {wres.total_return_pct:.1f}%  ₹{wres.net_pnl:,.0f}  WR {wres.win_rate:.1f}%  DD {wres.max_dd_pct:.1f}%")
    print(f"  months {wmonths}")
    print(f"Story pack {story[0].name}  {story[0].total_return_pct:.1f}%")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
