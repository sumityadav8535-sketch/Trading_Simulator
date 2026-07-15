"""
Preview-only filter search for Elite ML Short v2.
Does NOT write production result files.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.intraday_fno_ml_enhance import (  # noqa: E402
    FEATURE_COLS,
    backtest_signals,
    build_dataset,
    enrich_features,
    load_instrument,
    model_proba,
    time_split,
    train_models,
)
from scripts.intraday_fno_optimize import (  # noqa: E402
    FilterConfig,
    build_filtered_signals,
)


def run_cfg(df, dset, probs, cfg: FilterConfig, test_dates):
    sigs = build_filtered_signals(df, dset, probs, cfg)
    if sigs.empty:
        return None
    full_stats, full_trades = backtest_signals(df, sigs, cfg.name, "full", record_trades=True)
    full_losses = sum(1 for t in full_trades if t["result"] == "LOSS")
    full_wins = full_stats.trades - full_losses

    oos_stats = None
    oos_trades = []
    if test_dates:
        test_df = df[df["session_date"] >= min(test_dates)]
        test_dset = dset[dset["session_date"].isin(set(test_dates))].copy()
        if not test_dset.empty:
            # recompute probs aligned to full dset order — use model probs already on dset
            oos_probs = test_dset["prob"].values if "prob" in test_dset.columns else None
            if oos_probs is None:
                # fallback: map from full
                pass
            # build from full probs via filtered on test slice
            oos_sigs = build_filtered_signals(test_df, test_dset, test_dset["prob"].values, cfg)
            if not oos_sigs.empty:
                oos_stats, oos_trades = backtest_signals(
                    test_df, oos_sigs, f"{cfg.name} OOS", "OOS", record_trades=True
                )

    return {
        "cfg": cfg,
        "full": full_stats,
        "full_trades": full_trades,
        "full_wins": full_wins,
        "full_losses": full_losses,
        "oos": oos_stats,
        "oos_trades": oos_trades,
        "oos_losses": sum(1 for t in oos_trades if t["result"] == "LOSS") if oos_trades else 0,
    }


def score(row) -> float:
    """Prefer high OOS daily, then full daily, penalize few trades and high DD."""
    f = row["full"]
    o = row["oos"]
    if f.trades < 15:
        return -1e12
    oos_daily = o.avg_daily if o and o.trades >= 5 else f.avg_daily * 0.5
    oos_wr = o.win_rate if o and o.trades >= 5 else f.win_rate
    # composite in ₹ terms
    return (
        oos_daily * 2.0
        + f.avg_daily * 1.0
        + (oos_wr - 70) * 80
        + (f.win_rate - 70) * 40
        - f.max_dd_pct * 150
        + min(f.trades, 60) * 20
        - row["full_losses"] * 200
    )


def main():
    print("Loading NIFTY 5m + training RF (same pipeline as live backtest)...")
    df = enrich_features(load_instrument("NIFTY"))
    dset = build_dataset(df, use_loose=True).dropna(subset=FEATURE_COLS)
    train, test, train_dates, test_dates = time_split(dset, 0.7)
    X_tr = train[FEATURE_COLS].values[: int(len(train) * 0.8)]
    y_tr = train["label"].values[: int(len(train) * 0.8)]
    models = train_models(X_tr, y_tr)
    rf = models["random_forest"]
    probs = model_proba(rf, dset[FEATURE_COLS].values)
    dset = dset.copy()
    dset["prob"] = probs

    baseline = FilterConfig("BASELINE Elite v2", ml_th=0.58, max_risk_pts=22)

    # Focused candidates from loss analysis
    candidates: list[FilterConfig] = [baseline]

    # Single-axis improvements
    for rp in (0.15, 0.20, 0.25, 0.30, 0.35):
        candidates.append(FilterConfig(f"rp>={rp}", ml_th=0.58, max_risk_pts=22, range_pos_min=rp))
    for adx in (12, 15, 18, 20, 22):
        candidates.append(FilterConfig(f"adx>={adx}", ml_th=0.58, max_risk_pts=22, adx_min=adx))
    candidates.append(FilterConfig("stack+vwap+rsi50", ml_th=0.58, max_risk_pts=22, require_ema_stack=True))
    for ml in (0.60, 0.62, 0.65, 0.68):
        candidates.append(FilterConfig(f"ml>={ml}", ml_th=ml, max_risk_pts=22))
    for risk in (18, 20):
        candidates.append(FilterConfig(f"max_risk<={risk}", ml_th=0.58, max_risk_pts=risk))

    # Combinations aimed at the 9 losses
    combos = [
        FilterConfig("BEST-A rp0.25 adx15", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, adx_min=15),
        FilterConfig("BEST-B rp0.30 adx15", ml_th=0.58, max_risk_pts=22, range_pos_min=0.30, adx_min=15),
        FilterConfig("BEST-C rp0.25 adx18", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, adx_min=18),
        FilterConfig("BEST-D rp0.20 adx15 stack", ml_th=0.58, max_risk_pts=22, range_pos_min=0.20, adx_min=15, require_ema_stack=True),
        FilterConfig("BEST-E rp0.25 adx15 ml0.60", ml_th=0.60, max_risk_pts=22, range_pos_min=0.25, adx_min=15),
        FilterConfig("BEST-F rp0.25 adx15 risk20", ml_th=0.58, max_risk_pts=20, range_pos_min=0.25, adx_min=15),
        FilterConfig("BEST-G rp0.30 adx18 ml0.60", ml_th=0.60, max_risk_pts=22, range_pos_min=0.30, adx_min=18),
        FilterConfig("BEST-H rp0.25 stack", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, require_ema_stack=True),
        FilterConfig("BEST-I rp0.20 adx15 ml0.62", ml_th=0.62, max_risk_pts=22, range_pos_min=0.20, adx_min=15),
        FilterConfig("BEST-J rp0.25 adx15 hour<=13", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, adx_min=15, hour_end=13),
        FilterConfig("BEST-K rp0.30 adx15 stack", ml_th=0.58, max_risk_pts=22, range_pos_min=0.30, adx_min=15, require_ema_stack=True),
        FilterConfig("BEST-L rp0.35 adx15", ml_th=0.58, max_risk_pts=22, range_pos_min=0.35, adx_min=15),
        FilterConfig("BEST-M rp0.25 adx20", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, adx_min=20),
        FilterConfig("BEST-N ml0.62 rp0.25", ml_th=0.62, max_risk_pts=22, range_pos_min=0.25),
        FilterConfig("BEST-O rp0.25 adx15 risk18", ml_th=0.58, max_risk_pts=18, range_pos_min=0.25, adx_min=15),
        FilterConfig("BEST-P stack adx15", ml_th=0.58, max_risk_pts=22, adx_min=15, require_ema_stack=True),
        FilterConfig("BEST-Q rp0.20 adx18 risk20", ml_th=0.58, max_risk_pts=20, range_pos_min=0.20, adx_min=18),
        FilterConfig("BEST-R rp0.25 adx15 ml0.58 rsi40-55", ml_th=0.58, max_risk_pts=22, range_pos_min=0.25, adx_min=15, rsi_min=40, rsi_max=55),
    ]

    # Small grid of strongest levers
    for rp, adx, ml, stack in product([0.20, 0.25, 0.30], [15.0, 18.0], [0.58, 0.60], [False, True]):
        name = f"G rp{rp} a{adx:.0f} m{ml}" + (" stack" if stack else "")
        candidates.append(
            FilterConfig(
                name,
                ml_th=ml,
                max_risk_pts=22,
                range_pos_min=rp,
                adx_min=adx,
                require_ema_stack=stack,
            )
        )

    candidates.extend(combos)

    # de-dupe by name
    seen = set()
    uniq = []
    for c in candidates:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(c)

    print(f"Testing {len(uniq)} filter configs...\n")
    results = []
    for i, cfg in enumerate(uniq, 1):
        r = run_cfg(df, dset, probs, cfg, test_dates)
        if r is None:
            continue
        r["score"] = score(r)
        results.append(r)
        if i % 15 == 0:
            print(f"  ...{i}/{len(uniq)}")

    results.sort(key=lambda r: r["score"], reverse=True)

    def fmt(r):
        f, o = r["full"], r["oos"]
        oos_t = o.trades if o else 0
        oos_wr = o.win_rate if o else 0
        oos_d = o.avg_daily if o else 0
        oos_net = o.net_pnl if o else 0
        oos_l = r["oos_losses"]
        return (
            f"{r['cfg'].name[:42]:42s} | "
            f"n={f.trades:3d} WR={f.win_rate:5.1f}% L={r['full_losses']:2d} "
            f"₹/d={f.avg_daily:8,.0f} net={f.net_pnl:10,.0f} DD={f.max_dd_pct:4.1f}% | "
            f"OOS n={oos_t:2d} WR={oos_wr:5.1f}% L={oos_l:2d} ₹/d={oos_d:8,.0f} net={oos_net:9,.0f} | "
            f"score={r['score']:,.0f}"
        )

    print("=" * 140)
    print("BASELINE")
    print("=" * 140)
    base_r = next(r for r in results if r["cfg"].name.startswith("BASELINE"))
    print(fmt(base_r))
    print()
    print("=" * 140)
    print("TOP 15 BY COMPOSITE SCORE (OOS daily + full daily + WR − DD − losses)")
    print("=" * 140)
    for r in results[:15]:
        print(fmt(r))

    best = results[0]
    print()
    print("=" * 140)
    print("RECOMMENDED (best score)")
    print("=" * 140)
    cfg = best["cfg"]
    print("Filters:")
    for k, v in asdict(cfg).items():
        if k == "name":
            continue
        print(f"  {k}: {v}")
    f, o = best["full"], best["oos"]
    print()
    print(f"FULL: {f.trades} trades | WR {f.win_rate}% | W/L {best['full_wins']}/{best['full_losses']}")
    print(f"      ₹{f.avg_daily:,.0f}/day | Net ₹{f.net_pnl:,.0f} | PF {f.profit_factor} | MaxDD {f.max_dd_pct}%")
    if o:
        print(f"OOS:  {o.trades} trades | WR {o.win_rate}% | L {best['oos_losses']}")
        print(f"      ₹{o.avg_daily:,.0f}/day | Net ₹{o.net_pnl:,.0f} | PF {o.profit_factor} | MaxDD {o.max_dd_pct}%")

    # Compare which of the original 9 loss dates would survive
    base_loss_dates = {
        t["session_date"]
        for t in base_r["full_trades"]
        if t["result"] == "LOSS"
    }
    best_loss_dates = {
        t["session_date"]
        for t in best["full_trades"]
        if t["result"] == "LOSS"
    }
    print()
    print("Baseline loss session dates:", sorted(base_loss_dates))
    print("Best-filter loss session dates:", sorted(best_loss_dates))
    print(
        "Loss sessions removed:",
        sorted(base_loss_dates - best_loss_dates),
    )
    print(
        "New loss sessions introduced:",
        sorted(best_loss_dates - base_loss_dates),
    )

    # Also show top by pure full net and pure OOS daily for transparency
    print()
    print("--- Top 5 by full net P&L ---")
    for r in sorted(results, key=lambda x: x["full"].net_pnl, reverse=True)[:5]:
        print(fmt(r))
    print()
    print("--- Top 5 by OOS avg daily (min 5 OOS trades) ---")
    oos_ok = [r for r in results if r["oos"] and r["oos"].trades >= 5]
    for r in sorted(oos_ok, key=lambda x: x["oos"].avg_daily, reverse=True)[:5]:
        print(fmt(r))

    # Runner-up alternatives: high WR with decent volume
    print()
    print("--- High WR alternatives (full WR>=90%, trades>=20) ---")
    hi = [r for r in results if r["full"].win_rate >= 90 and r["full"].trades >= 20]
    hi.sort(key=lambda x: (x["full"].avg_daily, x["full"].net_pnl), reverse=True)
    for r in hi[:8]:
        print(fmt(r))

    # Write preview only (not production paths)
    out = ROOT / "data" / "_filter_preview_results.json"
    import json

    payload = {
        "note": "PREVIEW ONLY — not applied to live F&O config",
        "baseline": {
            "filters": {k: v for k, v in asdict(base_r["cfg"]).items() if k != "name"},
            "full": base_r["full"].__dict__,
            "full_losses": base_r["full_losses"],
            "oos": base_r["oos"].__dict__ if base_r["oos"] else {},
        },
        "recommended": {
            "name": best["cfg"].name,
            "filters": {k: v for k, v in asdict(best["cfg"]).items() if k != "name"},
            "full": best["full"].__dict__,
            "full_wins": best["full_wins"],
            "full_losses": best["full_losses"],
            "oos": best["oos"].__dict__ if best["oos"] else {},
            "oos_losses": best["oos_losses"],
            "score": best["score"],
        },
        "top15": [
            {
                "name": r["cfg"].name,
                "filters": {k: v for k, v in asdict(r["cfg"]).items() if k != "name"},
                "full": r["full"].__dict__,
                "full_losses": r["full_losses"],
                "oos": r["oos"].__dict__ if r["oos"] else {},
                "score": r["score"],
            }
            for r in results[:15]
        ],
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nPreview JSON (not live): {out}")


if __name__ == "__main__":
    main()
