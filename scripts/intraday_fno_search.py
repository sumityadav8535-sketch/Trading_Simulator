"""
NSE Index F&O intraday backtest (Nifty 50 + Bank Nifty).
Uses index 5m data as futures proxy; lot-based P&L with MIS margin rules.
Capital: Rs 5,00,000 | Target: > Rs 1,000/day average.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "intraday_fno"
OUT = ROOT / "data" / "intraday_fno_results.json"

CAPITAL = 500_000.0
TARGET_DAILY = 1_000.0
SLIPPAGE_PTS = 0.5  # half-point slippage per side on index

MARKET_OPEN = dt_time(9, 15)
NO_ENTRY_AFTER = dt_time(14, 45)
FORCE_EXIT = dt_time(15, 15)
ORB_15_END = dt_time(9, 30)
ORB_30_END = dt_time(9, 45)

INSTRUMENTS = {
    "NIFTY": {
        "ticker": "^NSEI",
        "lot_size": 75,
        "mis_margin": 65_000,
        "name": "Nifty 50 Futures",
    },
    "BANKNIFTY": {
        "ticker": "^NSEBANK",
        "lot_size": 30,
        "mis_margin": 85_000,
        "name": "Bank Nifty Futures",
    },
}


@dataclass
class FnoResult:
    name: str
    instrument: str
    trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    net_pnl: float
    max_drawdown_pct: float
    final_equity: float
    avg_daily_pnl: float
    days_above_1000: int
    max_daily_pnl: float
    params: dict = field(default_factory=dict)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(0)
    rename = {c: str(c).lower() for c in sub.columns}
    sub = sub.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    sub = sub[keep].dropna(subset=["close"])
    if sub.index.tz is None:
        sub.index = sub.index.tz_localize("Asia/Kolkata")
    else:
        sub.index = sub.index.tz_convert("Asia/Kolkata")
    return sub.sort_index()


def load_instrument(key: str, force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{key}.pkl"
    if path.exists() and not force:
        return pd.read_pickle(path)

    ticker = INSTRUMENTS[key]["ticker"]
    raw = yf.download(ticker, interval="5m", period="60d", progress=False, auto_adjust=False)
    df = normalize_df(raw)
    if not df.empty:
        df.to_pickle(path)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema_21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema_50"] = d["close"].ewm(span=50, adjust=False).mean()
    if (d["volume"] > 0).any():
        d["vol_sma"] = d["volume"].rolling(20).mean()
    else:
        d["vol_sma"] = 1.0

    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    d["session_date"] = d.index.date
    typical = (d["high"] + d["low"] + d["close"]) / 3
    cum_pv = (typical * d["volume"]).groupby(d["session_date"]).cumsum()
    cum_vol = d["volume"].groupby(d["session_date"]).cumsum().replace(0, np.nan)
    d["vwap"] = cum_pv / cum_vol
    # Index tickers (^NSEI) often report zero volume — use session TWAP instead.
    if d["vwap"].isna().all():
        d["vwap"] = typical.groupby(d["session_date"]).transform(lambda s: s.expanding().mean())

    d["day_open"] = d.groupby("session_date")["open"].transform("first")
    d["prev_rsi"] = d["rsi"].shift(1)
    d["prev_close"] = d["close"].shift(1)
    rng = d["high"] - d["low"]
    d["strong_close"] = (d["close"] - d["low"]) / rng.replace(0, np.nan) >= 0.65
    d["strong_open_reject"] = (d["high"] - d["close"]) / rng.replace(0, np.nan) >= 0.65

    return d


def max_lots(capital: float, margin_per_lot: float, deploy_pct: float = 0.85) -> int:
    return max(int(capital * deploy_pct / margin_per_lot), 0)


def lots_for_risk(capital, risk_pct, stop_pts, lot_size, max_l):
    if stop_pts <= 0:
        return 0
    risk_amt = capital * risk_pct / 100
    lots = int(risk_amt / (stop_pts * lot_size))
    return max(min(lots, max_l), 0)


# ── F&O signal generators (long=1, short=-1) ───────────────────────────────

def sig_orb_long(row, orb, min_rng=8.0):
    if not orb.get("done"):
        return None
    hi, lo = orb.get("high"), orb.get("low")
    if hi is None or lo is None or hi - lo < min_rng:
        return None
    if row["close"] <= hi:
        return None
    if row["volume"] < row["vol_sma"] * 0.8:
        return None
    stop = lo
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"side": 1, "stop": stop, "target": row["close"] + risk * 1.5}


def sig_orb_short(row, orb, min_rng=8.0):
    if not orb.get("done"):
        return None
    hi, lo = orb.get("high"), orb.get("low")
    if hi is None or lo is None or hi - lo < min_rng:
        return None
    if row["close"] >= lo:
        return None
    if row["volume"] < row["vol_sma"] * 0.8:
        return None
    stop = hi
    risk = stop - row["close"]
    if risk <= 0:
        return None
    return {"side": -1, "stop": stop, "target": row["close"] - risk * 1.5}


def sig_vwap_long(row, orb=None, target_r=1.25):
    if pd.isna(row["vwap"]) or row["close"] <= row["vwap"]:
        return None
    if row["ema_9"] <= row["ema_21"]:
        return None
    if row["rsi"] < 45 or row["rsi"] > 62:
        return None
    if not row["strong_close"]:
        return None
    stop = row["low"]
    risk = row["close"] - stop
    if risk <= 0 or risk > row["atr"] * 2:
        return None
    return {"side": 1, "stop": stop, "target": row["close"] + risk * target_r}


def sig_vwap_short(row, orb=None, target_r=1.25):
    if pd.isna(row["vwap"]) or row["close"] >= row["vwap"]:
        return None
    if row["ema_9"] >= row["ema_21"]:
        return None
    if row["rsi"] < 38 or row["rsi"] > 55:
        return None
    if not row["strong_open_reject"]:
        return None
    stop = row["high"]
    risk = stop - row["close"]
    if risk <= 0 or risk > row["atr"] * 2:
        return None
    return {"side": -1, "stop": stop, "target": row["close"] - risk * target_r}


def sig_ema_long(row, orb=None, target_r=1.5):
    if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
        return None
    if row["close"] <= row["vwap"]:
        return None
    if row["rsi"] < 50:
        return None
    stop = row["ema_21"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"side": 1, "stop": stop, "target": row["close"] + risk * target_r}


def sig_ema_short(row, orb=None, target_r=1.5):
    if not (row["ema_9"] < row["ema_21"] < row["ema_50"]):
        return None
    if row["close"] >= row["vwap"]:
        return None
    if row["rsi"] > 50:
        return None
    stop = row["ema_21"]
    risk = stop - row["close"]
    if risk <= 0:
        return None
    return {"side": -1, "stop": stop, "target": row["close"] - risk * target_r}


def sig_rsi_long(row, orb=None, target_r=1.0):
    if pd.isna(row["prev_rsi"]) or row["prev_rsi"] >= 35 or row["rsi"] <= 40:
        return None
    if row["close"] <= row["vwap"]:
        return None
    stop = row["low"]
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"side": 1, "stop": stop, "target": row["close"] + risk * target_r}


def sig_rsi_short(row, orb=None, target_r=1.0):
    if pd.isna(row["prev_rsi"]) or row["prev_rsi"] <= 65 or row["rsi"] >= 60:
        return None
    if row["close"] >= row["vwap"]:
        return None
    stop = row["high"]
    risk = stop - row["close"]
    if risk <= 0:
        return None
    return {"side": -1, "stop": stop, "target": row["close"] - risk * target_r}


def sig_combo_long(row, orb=None, target_r=1.25):
    if row["close"] <= row["vwap"] or row["ema_9"] <= row["ema_21"]:
        return None
    if row["rsi"] < 42 or row["rsi"] > 58:
        return None
    if not row["strong_close"]:
        return None
    if row["volume"] < row["vol_sma"]:
        return None
    stop = min(row["low"], row["ema_21"])
    risk = row["close"] - stop
    if risk <= 0:
        return None
    return {"side": 1, "stop": stop, "target": row["close"] + risk * target_r}


def run_fno_backtest(
    df: pd.DataFrame,
    inst_key: str,
    signal_fn,
    name: str,
    risk_pct: float = 1.5,
    trades_per_day: int = 3,
    orb_end: dt_time = ORB_30_END,
    min_orb_rng: float = 8.0,
) -> FnoResult:
    meta = INSTRUMENTS[inst_key]
    lot_size = meta["lot_size"]
    margin = meta["mis_margin"]

    equity = CAPITAL
    peak = equity
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    daily_pnl: dict = {}

    in_pos = False
    side = 0
    entry = stop = target = 0.0
    lots = 0
    day_count: dict = {}
    pending = None

    session_orb: dict = {}
    prev_row = None

    for i, (ts, row) in enumerate(df.iterrows()):
        if ts.time() < MARKET_OPEN or ts.time() > dt_time(15, 30):
            prev_row = row
            continue

        sess = row["session_date"]
        if sess not in session_orb:
            session_orb[sess] = {"high": None, "low": None, "done": False}
        orb = session_orb[sess]
        if ts.time() <= orb_end:
            orb["high"] = row["high"] if orb["high"] is None else max(orb["high"], row["high"])
            orb["low"] = row["low"] if orb["low"] is None else min(orb["low"], row["low"])
        elif not orb["done"]:
            orb["done"] = True

        # fill pending
        if pending and pending["ts"] == ts:
            max_l = max_lots(equity, margin)
            stop_pts = abs(float(row["open"]) - pending["stop"])
            lots = lots_for_risk(equity, risk_pct, stop_pts, lot_size, max_l)
            if lots > 0:
                in_pos = True
                side = pending["side"]
                entry = float(row["open"]) + SLIPPAGE_PTS * side
                stop = pending["stop"]
                target = pending["target"]
            pending = None

        # manage position
        if in_pos:
            exit_p = reason = None
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            if side == 1:
                if lo <= stop:
                    exit_p, reason = stop - SLIPPAGE_PTS, "sl"
                elif hi >= target:
                    exit_p, reason = target - SLIPPAGE_PTS, "target"
                elif ts.time() >= FORCE_EXIT:
                    exit_p, reason = cl - SLIPPAGE_PTS, "eod"
            else:
                if hi >= stop:
                    exit_p, reason = stop + SLIPPAGE_PTS, "sl"
                elif lo <= target:
                    exit_p, reason = target + SLIPPAGE_PTS, "target"
                elif ts.time() >= FORCE_EXIT:
                    exit_p, reason = cl + SLIPPAGE_PTS, "eod"

            if exit_p is not None:
                pts = (exit_p - entry) * side
                pnl = pts * lot_size * lots
                equity += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0)
                daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0.0) + pnl
                in_pos = False

        if in_pos or ts.time() > NO_ENTRY_AFTER:
            prev_row = row
            continue
        if day_count.get(sess, 0) >= trades_per_day:
            prev_row = row
            continue

        sig = signal_fn(row, orb)
        if sig is None:
            prev_row = row
            continue
        if i + 1 >= len(df):
            prev_row = row
            continue
        next_ts = df.index[i + 1]
        if next_ts.date() != ts.date():
            prev_row = row
            continue

        pending = {"ts": next_ts, "side": sig["side"], "stop": float(sig["stop"]), "target": float(sig["target"])}
        day_count[sess] = day_count.get(sess, 0) + 1
        prev_row = row

    all_days = len({ix.date() for ix in df.index if MARKET_OPEN <= ix.time() <= dt_time(15, 30)})
    avg_daily = (equity - CAPITAL) / max(all_days, 1)
    pf = gross_profit / gross_loss if gross_loss else (999 if gross_profit else 0)
    wr = wins / trades * 100 if trades else 0

    return FnoResult(
        name=name,
        instrument=inst_key,
        trades=trades,
        win_rate=round(wr, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round((equity - CAPITAL) / CAPITAL * 100, 2),
        net_pnl=round(equity - CAPITAL, 2),
        max_drawdown_pct=round(max_dd, 2),
        final_equity=round(equity, 2),
        avg_daily_pnl=round(avg_daily, 2),
        days_above_1000=sum(1 for p in daily_pnl.values() if p >= TARGET_DAILY),
        max_daily_pnl=round(max(daily_pnl.values()) if daily_pnl else 0, 2),
        params={"risk_pct": risk_pct, "trades_per_day": trades_per_day, "lot_size": lot_size},
    )


def main():
    print(f"F&O intraday search | Rs {CAPITAL:,.0f} | target > Rs {TARGET_DAILY:,.0f}/day")
    print("Proxy: index 5m bars | lot P&L | MIS margin model\n")

    data = {}
    for key in INSTRUMENTS:
        df = load_instrument(key)
        if df.empty:
            print(f"  WARN: no data for {key}")
            continue
        data[key] = add_indicators(df)
        days = len({ix.date() for ix in df.index})
        print(f"  {INSTRUMENTS[key]['name']}: {len(df)} bars, {days} calendar days")

    strategies = []
    for inst in data:
        min_rng = 6.0 if inst == "NIFTY" else 12.0
        strategies.extend([
            (inst, f"{inst} ORB-30 Long", lambda r, o, m=min_rng: sig_orb_long(r, o, m), ORB_30_END),
            (inst, f"{inst} ORB-30 Short", lambda r, o, m=min_rng: sig_orb_short(r, o, m), ORB_30_END),
            (inst, f"{inst} ORB-15 Long", lambda r, o, m=min_rng * 0.7: sig_orb_long(r, o, m), ORB_15_END),
            (inst, f"{inst} ORB-15 Short", lambda r, o, m=min_rng * 0.7: sig_orb_short(r, o, m), ORB_15_END),
            (inst, f"{inst} VWAP Long", lambda r, o: sig_vwap_long(r), ORB_15_END),
            (inst, f"{inst} VWAP Short", lambda r, o: sig_vwap_short(r), ORB_15_END),
            (inst, f"{inst} EMA Long", lambda r, o: sig_ema_long(r), ORB_15_END),
            (inst, f"{inst} EMA Short", lambda r, o: sig_ema_short(r), ORB_15_END),
            (inst, f"{inst} RSI Long", lambda r, o: sig_rsi_long(r), ORB_15_END),
            (inst, f"{inst} RSI Short", lambda r, o: sig_rsi_short(r), ORB_15_END),
            (inst, f"{inst} Combo Long", lambda r, o: sig_combo_long(r), ORB_15_END),
        ])

    risks = [1.0, 1.5, 2.0, 3.0]
    results: list[FnoResult] = []

    total = len(strategies) * len(risks)
    n = 0
    for inst, sname, fn, orb_end in strategies:
        for risk in risks:
            n += 1
            label = f"{sname} r{risk}%"
            print(f"[{n}/{total}] {label}", flush=True)

            r = run_fno_backtest(
                data[inst], inst, fn, label,
                risk_pct=risk, trades_per_day=4,
                orb_end=orb_end,
                min_orb_rng=6 if inst == "NIFTY" else 12,
            )
            results.append(r)

    by_daily = sorted(results, key=lambda x: -x.avg_daily_pnl)
    profitable = [r for r in results if r.net_pnl > 0]
    hit = [r for r in results if r.avg_daily_pnl >= TARGET_DAILY]

    print("\n" + "=" * 110)
    print(f"TOP 20 F&O (Rs {CAPITAL:,.0f})")
    print("=" * 110)
    print(f"{'Strategy':<40} {'Trd':>5} {'WR%':>6} {'Total':>10} {'Rs/day':>9} {'D>1k':>5} {'DD%':>6}")
    for r in by_daily[:20]:
        print(
            f"{r.name:<40} {r.trades:>5} {r.win_rate:>5.1f}% {r.net_pnl:>10,.0f} "
            f"{r.avg_daily_pnl:>9,.0f} {r.days_above_1000:>5} {r.max_drawdown_pct:>5.1f}%"
        )

    print("\n" + "=" * 110)
    if hit:
        print(f"TARGET MET — {len(hit)} configs >= Rs {TARGET_DAILY:,.0f}/day avg")
        for r in hit[:8]:
            print(f"  * {r.name}: Rs {r.avg_daily_pnl:,.0f}/day | total Rs {r.net_pnl:,.0f} | {r.trades} trades")
    else:
        best = by_daily[0] if by_daily else None
        print("TARGET NOT MET")
        if best:
            print(f"  Best: {best.name}")
            print(f"  Avg/day : Rs {best.avg_daily_pnl:,.0f}")
            print(f"  Total   : Rs {best.net_pnl:,.0f} ({best.total_return_pct}%)")
            print(f"  Trades  : {best.trades} | WR {best.win_rate}% | PF {best.profit_factor}")
            print(f"  Max day : Rs {best.max_daily_pnl:,.0f} | Days>1k: {best.days_above_1000}")

    print(f"\nProfitable: {len(profitable)} / {len(results)}")

    OUT.write_text(json.dumps({
        "capital": CAPITAL,
        "target_daily": TARGET_DAILY,
        "instruments": INSTRUMENTS,
        "note": "Index 5m proxy for futures; lot-based P&L",
        "target_met": len(hit) > 0,
        "hit_target": [r.__dict__ for r in hit[:15]],
        "top_20": [r.__dict__ for r in by_daily[:20]],
        "profitable_count": len(profitable),
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()