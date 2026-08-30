"""
Phase 3: refine the only profitable family (VWAP extension fade)
and scale with realistic MIS buying power.

Also test a second uncorrelated add-on: failed-break fade with ATR floors.
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
    TARGET_LO,
    _base_cols,
    _can_trade_mask,
    add_indicators,
    attach_daily_bias,
    build_exec_index,
    collect_signals,
    load_universe,
    nifty200_symbols,
    simulate as simulate_base,
    _qty as qty_base,
    _summarize,
    SimResult,
)

OUT = ROOT / "data" / "intraday_15m_n200_phase3.json"


def load_prepared():
    cache = load_universe(nifty200_symbols(), skip_download=True)
    stocks = {}
    for sym, df in cache.items():
        if sym.startswith("_"):
            continue
        en = add_indicators(df)
        en = attach_daily_bias(en, sym)
        if len(en) >= 50:
            stocks[sym] = en
    idx = cache.get("_INDEX", pd.DataFrame())
    if not idx.empty:
        idx = add_indicators(idx)
        idx["idx_day"] = idx["close"] / idx.groupby("session")["open"].transform("first") - 1
    return stocks, idx


def attach_index(stocks, idx):
    if idx.empty:
        for df in stocks.values():
            df["idx_day"] = 0.0
            df["idx_adx"] = 20.0
        return stocks
    lookup_day = idx["idx_day"]
    lookup_adx = idx["adx"]
    for df in stocks.values():
        df["idx_day"] = lookup_day.reindex(df.index, method="ffill")
        df["idx_adx"] = lookup_adx.reindex(df.index, method="ffill")
    return stocks


def sig_vwap_fade_v2(
    df,
    ext_min=0.012,
    ext_max=0.035,
    rsi_hi=68,
    rsi_lo=32,
    adx_max=28,
    idx_abs_max=0.006,
    min_atr_pct=0.0025,
    target_mode="vwap",  # vwap | half | atr
    start=dtime(10, 0),
    end=dtime(14, 0),
    min_price=60,
):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 2) & (t >= start) & (t <= end)
    m &= df["close"] > min_price
    m &= df["atr"] / df["close"] >= min_atr_pct
    m &= df["vol_sma"] > 30_000
    if "idx_day" in df.columns:
        m &= df["idx_day"].abs() <= idx_abs_max
    m &= df["adx"] <= adx_max

    dist = (df["close"] - df["vwap"]) / df["vwap"]
    short = m & dist.between(ext_min, ext_max) & (df["rsi"] >= rsi_hi)
    lng = m & dist.between(-ext_max, -ext_min) & (df["rsi"] <= rsi_lo)

    # stop: beyond extreme + 0.35 ATR (needs room vs costs)
    sstop = df["close"] + df["atr"] * 0.45
    lstop = df["close"] - df["atr"] * 0.45
    if target_mode == "half":
        stgt = df["close"] - (df["close"] - df["vwap"]) * 0.55
        ltgt = df["close"] + (df["vwap"] - df["close"]) * 0.55
    elif target_mode == "atr":
        stgt = df["close"] - df["atr"] * 1.1
        ltgt = df["close"] + df["atr"] * 1.1
    else:
        stgt = df["vwap"]
        ltgt = df["vwap"]

    sscore = dist * 100 * (df["rsi"] / 70) * df["rel_vol"].clip(upper=4).fillna(1)
    lscore = (-dist) * 100 * ((40 - df["rsi"]).clip(lower=1) / 20) * df["rel_vol"].clip(upper=4).fillna(1)
    frames = [
        _base_cols(df, short, "short", sstop, stgt, sscore),
        _base_cols(df, lng, "long", lstop, ltgt, lscore),
    ]
    return pd.concat(frames, ignore_index=True)


def sig_orb_fade_v2(df, ext=0.007, adx_max=26):
    t = pd.Series(df.index.time, index=df.index)
    m = _can_trade_mask(df, 1) & (t >= dtime(9, 45)) & (t <= dtime(12, 15))
    m &= (df["adx"] <= adx_max) & (df["close"] > 60) & (df["vol_sma"] > 30_000)
    rng = (df["orb15_high"] - df["orb15_low"]) / df["close"]
    m &= rng.between(0.004, 0.02)
    short = m & (df["close"] > df["orb15_high"] * (1 + ext)) & (df["rsi"] > 66)
    lng = m & (df["close"] < df["orb15_low"] * (1 - ext)) & (df["rsi"] < 34)
    mid = (df["orb15_high"] + df["orb15_low"]) / 2
    sstop = df["close"] + df["atr"] * 0.45
    lstop = df["close"] - df["atr"] * 0.45
    stgt = np.maximum(df["vwap"], mid)
    ltgt = np.minimum(df["vwap"], mid)
    return pd.concat([
        _base_cols(df, short, "short", sstop, stgt, df["rsi"]),
        _base_cols(df, lng, "long", lstop, ltgt, 100 - df["rsi"]),
    ], ignore_index=True)


def simulate_bp(
    exec_idx,
    signals,
    name,
    risk_pct=2.0,
    max_pos=4,
    top_k=1,
    daily_lock=0.0,
    daily_halt=0.0,
    max_deploy=1.0,
    leverage=1.0,
):
    """Same as simulate() but with buying-power / max_deploy control."""
    # monkeypatch qty via local copy of simulate is messy — inline thin wrapper
    # by temporarily wrapping _qty is not possible cleanly; duplicate call with patched max
    from scripts import intraday_15m_n200_search as eng

    orig = eng._qty

    def qty(equity, risk_pct, entry, stop, side, max_deploy_pct=max_deploy):
        risk_ps = (entry - stop) if side == "long" else (stop - entry)
        if risk_ps <= 0 or entry <= 0:
            return 0
        bp = equity * leverage
        qty_r = int((equity * risk_pct / 100.0) / risk_ps)
        cap = int((bp * max_deploy_pct) / entry)
        return max(min(qty_r, cap), 0)

    eng._qty = qty
    try:
        res, tdf = simulate_base(
            exec_idx, signals, name,
            risk_pct=risk_pct, max_pos=max_pos, top_k=top_k,
            daily_lock=daily_lock, daily_halt=daily_halt,
        )
    finally:
        eng._qty = orig
    res.params = {
        "risk": risk_pct, "max_pos": max_pos, "top_k": top_k,
        "lock": daily_lock, "halt": daily_halt,
        "max_deploy": max_deploy, "leverage": leverage,
    }
    return res, tdf


def merge_signals(*frames):
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=["ts", "symbol", "side", "stop", "target", "score"])
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "score"], ascending=[True, False])


def main():
    print("Phase 3 — refine VWAP fade + scale")
    stocks, idx = load_prepared()
    stocks = attach_index(stocks, idx)
    exec_idx = build_exec_index(stocks)
    print(f"stocks={len(stocks)} bars={len(exec_idx['calendar'])}")

    variants = []
    for ext in (0.010, 0.012, 0.015, 0.018):
        for adx in (22, 26, 32):
            for tgt in ("vwap", "half", "atr"):
                for idxm in (0.004, 0.008, 0.02):
                    name = f"fade e{ext:.3f} adx{adx} {tgt} idx{idxm}"
                    variants.append((name, dict(ext_min=ext, adx_max=adx, target_mode=tgt, idx_abs_max=idxm)))

    print(f"\nScanning {len(variants)} fade variants @ r2 p4 k1 deploy0.5 lev1 ...")
    book = []
    for name, kw in variants:
        sigs = collect_signals(stocks, sig_vwap_fade_v2, **kw)
        res, tdf = simulate_bp(exec_idx, sigs, name, risk_pct=2, max_pos=4, top_k=1, max_deploy=0.5, leverage=1)
        res.params.update(kw)
        book.append((res, tdf, kw, sigs))

    book.sort(key=lambda x: (x[0].oos_avg_daily if x[0].oos_days >= 4 else -999, x[0].profit_factor, x[0].avg_daily_pnl), reverse=True)
    print("\nTop fade variants:")
    for res, _, _, _ in book[:12]:
        print(f"  {res.name:46s} n={res.trades:4d} WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} "
              f"avg={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} DD={res.max_dd_pct:4.1f} "
              f"ge2k={res.days_ge_2k}/{res.trading_days}")

    best_kw = book[0][3]
    best_name = book[0][0].name
    print(f"\nBest variant: {best_name}")

    # Scale the best 3 variants
    print("\n=== Scale best variants with MIS buying power ===")
    scaled = []
    for res0, _, kw, sigs in book[:3]:
        for lev, dep, risk, pos, topk, lock, halt in [
            (1, 0.5, 2.0, 4, 1, 0, 0),
            (1, 1.0, 2.0, 4, 1, 0, 0),
            (3, 0.35, 2.0, 4, 1, 3000, 2500),
            (4, 0.30, 2.5, 5, 1, 4000, 2500),
            (5, 0.25, 2.0, 4, 1, 3000, 2000),
            (5, 0.30, 3.0, 5, 2, 5000, 2500),
            (5, 0.35, 2.5, 6, 1, 4000, 2500),
        ]:
            label = f"{res0.name[:22]} L{lev} d{dep} r{risk} p{pos}"
            res, tdf = simulate_bp(
                exec_idx, sigs, label, risk_pct=risk, max_pos=pos, top_k=topk,
                daily_lock=lock, daily_halt=halt, max_deploy=dep, leverage=lev,
            )
            res.params.update(kw)
            res.params.update({"leverage": lev, "max_deploy": dep, "lock": lock})
            scaled.append((res, tdf))
            print(f"  {label:48s} n={res.trades:4d} WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} "
                  f"avg={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} DD={res.max_dd_pct:5.1f} "
                  f"ge2k={res.days_ge_2k}/{res.trading_days} worst={res.worst_day:7.0f}")

    # Combo: best fade + orb fade
    print("\n=== Combo fade + ORB fade ===")
    fade_kw = book[0][2]
    fade_sigs = collect_signals(stocks, sig_vwap_fade_v2, **fade_kw)
    orb_sigs = collect_signals(stocks, sig_orb_fade_v2)
    combo = merge_signals(fade_sigs, orb_sigs)
    print(f"combo signals {len(combo)}")
    for lev, dep, risk, pos in [(1, 0.8, 2, 5), (4, 0.3, 2.5, 5), (5, 0.3, 2.5, 6)]:
        res, tdf = simulate_bp(exec_idx, combo, f"combo L{lev}", risk_pct=risk, max_pos=pos,
                               top_k=1, daily_lock=4000, daily_halt=2500, max_deploy=dep, leverage=lev)
        scaled.append((res, tdf))
        print(f"  combo L{lev} d{dep} r{risk} p{pos}: n={res.trades} WR={res.win_rate} PF={res.profit_factor} "
              f"avg={res.avg_daily_pnl} OOS={res.oos_avg_daily} DD={res.max_dd_pct} ge2k={res.days_ge_2k}/{res.trading_days}")

    scaled.sort(key=lambda x: (x[0].oos_avg_daily if x[0].oos_days >= 4 else -9999, x[0].avg_daily_pnl), reverse=True)
    hits = [r for r, _ in scaled if r.avg_daily_pnl >= TARGET_LO and r.oos_avg_daily >= 800 and r.max_dd_pct <= 25]
    near = [r for r, _ in scaled if r.avg_daily_pnl >= 800 and r.oos_avg_daily >= 400]

    print("\n===== LEADERBOARD =====")
    for res, _ in scaled[:12]:
        print(f"  {res.name:48s} avg={res.avg_daily_pnl:7.0f} OOS={res.oos_avg_daily:7.0f} "
              f"WR={res.win_rate:5.1f} PF={res.profit_factor:5.2f} DD={res.max_dd_pct:5.1f} "
              f"ge2k={res.days_ge_2k}/{res.trading_days} pnl={res.net_pnl:8.0f}")

    payload = {
        "fade_top": [asdict(r) for r, *_ in book[:10]],
        "scaled_top": [asdict(r) for r, _ in scaled[:15]],
        "hits": [asdict(r) for r in hits],
        "near": [asdict(r) for r in near[:8]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {OUT}")
    print(f"Hits (>=2k/day, OOS ok, DD<=25%): {len(hits)}")
    print(f"Near (>=800/day): {len(near)}")


if __name__ == "__main__":
    main()
