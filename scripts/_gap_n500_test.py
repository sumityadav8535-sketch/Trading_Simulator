"""
Fill missing Nifty 500 5-minute bars (Yahoo allows ~60 days) and run the live
gap book: largest 2–6% gap-down, RSI 45–70, 9:30 bounce, +1% target, 2× ATR stop.

    python scripts/_gap_n500_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

from scripts._gap_indicator_opt import (  # noqa: E402
    CAPITAL,
    EXITS,
    LEVERAGE,
    _levels,
    _walk,
    build_book,
    eval_book,
    nifty_context,
)
from scripts.intraday_15m_n200_search import MARKET_OPEN, _normalize  # noqa: E402
from scripts.intraday_5m_hunt import DATA_5M, download_5m_universe  # noqa: E402
from trading.models import Stock  # noqa: E402
from trading.services.nse_price_sync import yfinance_ticker  # noqa: E402

OUT = ROOT / "data" / "intraday_gap_n500.json"
EXITS["p10_a20"] = dict(tp="pct", pct=0.010, stop="atr", sl=2.0)
SIZE = dict(risk_pct=8.0, max_deploy=1.0, top_k=1, cap_total=True)


def universe() -> tuple[list[str], set[str]]:
    n500 = list(
        Stock.objects.filter(is_active=True, is_nifty500=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )
    n200 = set(
        Stock.objects.filter(is_active=True, is_nifty200=True).values_list("symbol", flat=True)
    )
    return n500, n200


def missing(symbols: list[str]) -> list[str]:
    return [s for s in symbols if not (DATA_5M / f"{s}.pkl").exists()]


def retry_one(symbols: list[str]) -> int:
    import yfinance as yf

    saved = 0
    for sym in symbols:
        path = DATA_5M / f"{sym}.pkl"
        if path.exists():
            continue
        ticker = yfinance_ticker(sym)
        try:
            raw = yf.download(ticker, interval="5m", period="60d", progress=False, auto_adjust=False)
            df = _normalize(raw, ticker)
        except Exception as exc:
            print(f"    retry fail {sym}: {exc}", flush=True)
            time.sleep(0.4)
            continue
        if df.empty or len(df) < 80:
            print(f"    empty {sym}", flush=True)
            time.sleep(0.2)
            continue
        t = pd.Series(df.index.tz_convert("Asia/Kolkata").time, index=df.index)
        df = df.loc[(t >= MARKET_OPEN) & (t <= dtime(15, 30))]
        if len(df) < 80:
            continue
        df.to_pickle(path)
        saved += 1
        time.sleep(0.25)
    return saved


def live_mask(feat: pd.DataFrame) -> pd.Series:
    return (
        (feat["gap"] <= -0.02) & (feat["gap"] >= -0.06) & feat["rsi"].between(45, 70) & feat["bounce"]
    ).fillna(False)


def tape(feat, paths, mask, calendar) -> list[dict]:
    spec = EXITS["p10_a20"]
    by = {}
    for i in mask.to_numpy().nonzero()[0]:
        row = feat.iloc[int(i)]
        entry = float(row["open930"])
        levels = _levels(row, spec, "long", entry)
        if not levels:
            continue
        walked = _walk(paths[int(i)], 0, entry, levels[0], levels[1], "long", spec)
        if not walked:
            continue
        by.setdefault(row["session"], []).append((abs(float(row["gap"])), row, walked))
    equity = CAPITAL
    trades = []
    allowed = set(calendar)
    for sess in calendar:
        cands = by.get(sess) or []
        if not cands or sess not in allowed:
            continue
        _score, row, walked = max(cands, key=lambda x: x[0])
        pnl_ps, risk, entry_f, reason = walked[:4]
        bp = equity * LEVERAGE
        qty = int((equity * 8 / 100.0) / risk) if risk > 0 else 0
        cap = int(bp / entry_f) if entry_f else 0
        qty = max(min(qty, cap), 0)
        if qty <= 0:
            continue
        pnl = pnl_ps * qty
        trades.append({
            "session": str(sess),
            "symbol": row["symbol"],
            "gap": round(float(row["gap"]) * 100, 2),
            "pnl": round(float(pnl), 2),
            "reason": reason,
            "in_nifty200": bool(row["in_nifty200"]),
        })
        equity += pnl
    return trades


def pack(name: str, trades: list[dict], sessions: int) -> dict:
    if not trades:
        return {"name": name, "trades": 0, "net_pnl": 0, "sessions": sessions}
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    months: dict[str, dict] = {}
    for t in trades:
        m = t["session"][:7]
        row = months.setdefault(m, {"trades": 0, "pnl": 0.0, "wins": 0})
        row["trades"] += 1
        row["pnl"] += t["pnl"]
        row["wins"] += int(t["pnl"] > 0)
    for row in months.values():
        row["pnl"] = round(row["pnl"], 2)
        row["wr"] = round(100 * row["wins"] / row["trades"], 1)
    outside = sum(1 for t in trades if not t["in_nifty200"])
    return {
        "name": name,
        "trades": len(trades),
        "win_rate": round(100 * len(wins) / len(trades), 1),
        "profit_factor": round(gp / gl, 2) if gl else None,
        "net_pnl": round(sum(pnl), 2),
        "avg_trade_day": round(sum(pnl) / len(trades), 2),
        "days_ge_5k": int(sum(p >= 5000 for p in pnl)),
        "worst_day": round(min(pnl), 2),
        "best_day": round(max(pnl), 2),
        "outside_nifty200": outside,
        "sessions": sessions,
        "months": months,
    }


def main():
    t0 = time.time()
    n500, n200 = universe()
    miss = missing(n500)
    print(f"Nifty 500 in database: {len(n500)}  already have 5m: {len(n500) - len(miss)}  missing: {len(miss)}", flush=True)
    if miss:
        print("Downloading 5-minute bars for the missing names (about 60 days, Yahoo's limit)…", flush=True)
        download_5m_universe(miss)
        still = missing(n500)
        if still:
            print(f"Retrying {len(still)} names one by one…", flush=True)
            retry_one(still)
        still = missing(n500)
        print(f"Still missing after download: {len(still)}", flush=True)
        if still:
            print("  " + ", ".join(still[:40]), flush=True)

    have = [s for s in n500 if (DATA_5M / f"{s}.pkl").exists()]
    print(f"5m files for Nifty 500: {len(have)}", flush=True)
    from scripts.intraday_5m_hunt import load_frames

    cache = load_frames(have)
    stocks = {k: v for k, v in cache.items() if not str(k).startswith("_")}
    spans = []
    for sym, df in stocks.items():
        if df is None or df.empty:
            continue
        idx = df.index
        spans.append((sym, idx.min().date(), idx.max().date(), len(df)))
    spans.sort(key=lambda r: r[1])
    print(f"Loaded {len(spans)} frames. Earliest start {spans[0][1] if spans else '—'}  latest end {max(s[2] for s in spans) if spans else '—'}", flush=True)
    # Day when most of the newly added names are present.
    new_starts = [a for sym, a, _b, _n in spans if sym not in n200]
    if new_starts:
        new_starts.sort()
        common = new_starts[int(len(new_starts) * 0.10)]
    else:
        common = spans[0][1]
    print(f"Common Nifty 500 window starts {common} (90% of the new names have bars by then)", flush=True)

    feat, paths, calendar = build_book(stocks, nifty_context())
    feat["in_nifty200"] = feat["symbol"].isin(n200)
    # Drop an unfinished session (today, if the download ran before the close).
    full_days = []
    for sess in calendar:
        day = feat[feat["session"] == sess]
        if day.empty:
            continue
        full_days.append(sess)
    # Prefer sessions that appear in the feature set; incomplete today is still a session
    # but build_book only keeps names with a 9:30 bar, so a mid-morning download can trade it.
    # Drop the last session when it is today and the clock is before 15:30.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if calendar and calendar[-1] == now.date() and now.time() < dtime(15, 30):
        print(f"Dropping unfinished session {calendar[-1]}", flush=True)
        calendar = [d for d in calendar if d != now.date()]

    base = live_mask(feat)
    common_cal = [d for d in calendar if d >= common]
    mask_500 = base & feat["session"].isin(common_cal)
    mask_200 = mask_500 & feat["in_nifty200"]
    # Same engine as the live page, for the headline numbers.
    oos_cut = common_cal[int(len(common_cal) * 0.70)] if common_cal else calendar[0]
    r500 = eval_book(feat, paths, mask_500, "p10_a20", common_cal, oos_cut, name="Nifty 500", **SIZE)
    r200 = eval_book(feat, paths, mask_200, "p10_a20", common_cal, oos_cut, name="Nifty 200", **SIZE)
    t500 = tape(feat, paths, mask_500, common_cal)
    t200 = tape(feat, paths, mask_200, common_cal)
    p500 = pack("Nifty 500", t500, len(common_cal))
    p200 = pack("Nifty 200", t200, len(common_cal))

    changed = []
    b200 = {t["session"]: t for t in t200}
    for t in t500:
        other = b200.get(t["session"])
        if other and other["symbol"] != t["symbol"]:
            changed.append({
                "session": t["session"],
                "n500": t["symbol"],
                "n500_pnl": t["pnl"],
                "n200": other["symbol"],
                "n200_pnl": other["pnl"],
            })
    only_500 = [t for t in t500 if t["session"] not in b200]

    print("\n===== SAME WINDOW =====", flush=True)
    for row in (p500, p200):
        mo = " ".join(f"{k}:{v['pnl']:.0f}/{v['trades']}" for k, v in sorted(row.get("months", {}).items()))
        print(
            f"{row['name']:<12} n={row['trades']:3d} WR {row.get('win_rate', 0):5.1f} "
            f"PF {row.get('profit_factor')} net {row['net_pnl']:8.0f} "
            f"tday {row.get('avg_trade_day', 0):7.0f} 5k {row.get('days_ge_5k', 0)} "
            f"worst {row.get('worst_day', 0):8.0f} outside200 {row.get('outside_nifty200', 0)}  {mo}",
            flush=True,
        )
    print(f"Days the 500 pick was a different stock: {len(changed)}", flush=True)
    print(f"Days only the 500 book traded: {len(only_500)}", flush=True)
    print(f"Engine check 500 net {r500['net_pnl']:.0f} vs tape {p500['net_pnl']:.0f}", flush=True)

    payload = {
        "universe_db": len(n500),
        "with_5m": len(have),
        "loaded": len(stocks),
        "still_missing": missing(n500),
        "common_start": str(common),
        "common_end": str(common_cal[-1]) if common_cal else None,
        "common_sessions": len(common_cal),
        "nifty500": p500,
        "nifty200": p200,
        "engine_500": {k: r500[k] for k in ("trades", "win_rate", "profit_factor", "net_pnl", "max_dd_pct", "oos_pnl", "worst_day", "days_ge_5k", "months")},
        "different_picks": changed,
        "only_on_500": only_500,
        "trades_500": t500,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
