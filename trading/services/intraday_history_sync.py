"""
Backfill / refresh stored 5m intraday pickle history for Nifty 100 equities
and F&O index proxies (NIFTY / BANKNIFTY) up through the click day (IST).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from django.utils import timezone

from trading.constants import NIFTY100_INDEX_TICKER
from trading.services.fno_engine import INSTRUMENTS, normalize_df
from trading.services.nifty100 import ensure_nifty100_marked
from trading.services.nse_price_sync import yfinance_ticker

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent.parent.parent
EQUITY_DIR = ROOT / "data" / "intraday_5m"
FNO_DIR = ROOT / "data" / "intraday_fno"
INDEX_PKL = "_NIFTY100_INDEX.pkl"

# yfinance 5m history is limited (~60 calendar days)
MAX_5M_LOOKBACK_DAYS = 59
BATCH_SIZE = 20
BATCH_PAUSE_SEC = 0.35

_sync_lock = threading.Lock()
_status_lock = threading.Lock()
_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "message": "Idle",
    "target_date": None,
    "progress": {"done": 0, "total": 0},
    "equities": {},
    "fno": {},
    "error": None,
}


def get_sync_status() -> dict:
    with _status_lock:
        return dict(_status)


def _set_status(**kwargs) -> None:
    with _status_lock:
        _status.update(kwargs)


def _now_ist() -> datetime:
    return datetime.now(IST)


def _target_date(now: Optional[datetime] = None) -> date:
    """Trading calendar date we aim to fill through when the button is clicked."""
    return (now or _now_ist()).date()


def _normalize_equity_frame(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    sub = df.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        if ticker and ticker in sub.columns.get_level_values(0):
            sub = sub[ticker]
        elif ticker and ticker in sub.columns.get_level_values(1):
            sub = sub.xs(ticker, axis=1, level=1)
        else:
            sub.columns = sub.columns.get_level_values(0)

    rename = {}
    for col in sub.columns:
        key = str(col).lower().replace(" ", "_")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = key
    sub = sub.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    if not keep:
        return pd.DataFrame()
    sub = sub[keep].dropna(subset=["close"])
    if sub.empty:
        return pd.DataFrame()

    if sub.index.tz is None:
        sub.index = sub.index.tz_localize("Asia/Kolkata")
    else:
        sub.index = sub.index.tz_convert("Asia/Kolkata")
    return sub.sort_index()


def _read_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        # Ensure lowercase OHLCV if older Title-case files appear
        if "close" not in df.columns and "Close" in df.columns:
            df = _normalize_equity_frame(df)
        return df
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _last_bar_ts(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    ts = df.index.max()
    if getattr(ts, "tzinfo", None) is None:
        ts = pd.Timestamp(ts).tz_localize("Asia/Kolkata")
    else:
        ts = pd.Timestamp(ts).tz_convert("Asia/Kolkata")
    return ts


def _last_bar_date(df: pd.DataFrame) -> Optional[date]:
    ts = _last_bar_ts(df)
    return ts.date() if ts is not None else None


def _fetch_start_for(hist: pd.DataFrame, target: date) -> date:
    """
    Start date for yfinance download. Overlap 1 day with existing history for clean merge.
    Cap lookback to yfinance 5m limit.
    """
    floor = target - timedelta(days=MAX_5M_LOOKBACK_DAYS)
    last = _last_bar_date(hist)
    if last is None:
        return floor
    start = last - timedelta(days=1)
    return max(start, floor)


def _needs_update(hist: pd.DataFrame, target: date, now: Optional[datetime] = None) -> bool:
    """True if history is missing target day, or target is today and bars look incomplete."""
    last_ts = _last_bar_ts(hist)
    if last_ts is None:
        return True
    last = last_ts.date()
    if last < target:
        return True
    if last > target:
        return False
    # Same calendar day as target — refresh if still short of session end / now
    now = now or _now_ist()
    session_end = datetime.combine(target, dt_time(15, 30), tzinfo=IST)
    cutoff = min(now, session_end)
    # Allow a small lag (one 5m bar) so we don't thrash
    return last_ts < (cutoff - timedelta(minutes=5))


def _merge_bars(hist: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if fresh is None or fresh.empty:
        return hist if hist is not None else pd.DataFrame()
    if hist is None or hist.empty:
        return fresh
    combined = pd.concat([hist, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _download_single(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download 5m bars; end is exclusive (yfinance convention)."""
    raw = yf.download(
        ticker,
        interval="5m",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return raw


def _download_batch(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    return yf.download(
        tickers,
        interval="5m",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        group_by="ticker",
        progress=False,
        auto_adjust=False,
        threads=True,
    )


def _extract_from_batch(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if ticker in level0:
            sub = raw[ticker].copy()
        elif ticker in level1:
            sub = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
        return _normalize_equity_frame(sub)
    return _normalize_equity_frame(raw, ticker)


def get_history_coverage() -> dict:
    """Snapshot of stored history date ranges (no network)."""
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    FNO_DIR.mkdir(parents=True, exist_ok=True)

    equity_files = sorted(EQUITY_DIR.glob("*.pkl"))
    eq_max = None
    eq_min = None
    for path in equity_files:
        df = _read_pickle(path)
        dmax = _last_bar_date(df)
        dmin = df.index.min().date() if not df.empty else None
        if dmax and (eq_max is None or dmax > eq_max):
            eq_max = dmax
        if dmin and (eq_min is None or dmin < eq_min):
            eq_min = dmin

    fno_info = {}
    for key in INSTRUMENTS:
        path = FNO_DIR / f"{key}.pkl"
        df = _read_pickle(path)
        fno_info[key] = {
            "last_date": _last_bar_date(df).isoformat() if _last_bar_date(df) else None,
            "last_bar": _last_bar_ts(df).isoformat() if _last_bar_ts(df) is not None else None,
            "bars": int(len(df)),
        }

    target = _target_date()
    return {
        "target_date": target.isoformat(),
        "equities": {
            "files": len(equity_files),
            "first_date": eq_min.isoformat() if eq_min else None,
            "last_date": eq_max.isoformat() if eq_max else None,
            "stale": (eq_max is None) or (eq_max < target),
        },
        "fno": {
            "instruments": fno_info,
            "stale": any(
                (info["last_date"] is None) or (date.fromisoformat(info["last_date"]) < target)
                for info in fno_info.values()
            )
            if fno_info
            else True,
        },
        "sync": get_sync_status(),
    }


def _sync_equities(target: date, progress_base: int, progress_total: int) -> dict:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    symbols = ensure_nifty100_marked()
    # Prefer symbols that already have pickles; still fetch missing universe members
    existing = {p.stem for p in EQUITY_DIR.glob("*.pkl") if not p.stem.startswith("_")}
    ordered = sorted(set(symbols) | existing)

    stats = {
        "checked": 0,
        "updated": 0,
        "skipped": 0,
        "bars_added": 0,
        "errors": [],
        "last_date_before": None,
        "last_date_after": None,
    }

    # Global last date sample
    sample_paths = list(EQUITY_DIR.glob("*.pkl"))
    before_dates = [_last_bar_date(_read_pickle(p)) for p in sample_paths[:5]]
    before_dates = [d for d in before_dates if d]
    if before_dates:
        stats["last_date_before"] = max(before_dates).isoformat()

    # Index first
    idx_path = EQUITY_DIR / INDEX_PKL
    try:
        hist = _read_pickle(idx_path)
        if _needs_update(hist, target):
            start = _fetch_start_for(hist, target)
            raw = _download_single(NIFTY100_INDEX_TICKER, start, target)
            fresh = _normalize_equity_frame(raw, NIFTY100_INDEX_TICKER)
            if not fresh.empty:
                before_n = len(hist)
                merged = _merge_bars(hist, fresh)
                # Drop bars beyond target calendar day
                merged = merged[merged.index.date <= target]
                merged.to_pickle(idx_path)
                stats["updated"] += 1
                stats["bars_added"] += max(0, len(merged) - before_n)
            else:
                stats["skipped"] += 1
        else:
            stats["skipped"] += 1
        stats["checked"] += 1
    except Exception as exc:
        logger.exception("Index sync failed")
        stats["errors"].append(f"_INDEX: {exc}")

    _set_status(
        message=f"Syncing equities… 0/{len(ordered)}",
        progress={"done": progress_base + 1, "total": progress_total},
    )

    for i in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[i : i + BATCH_SIZE]
        # Determine which need update and a common start
        need: list[tuple[str, Path, pd.DataFrame, date]] = []
        for sym in batch:
            path = EQUITY_DIR / f"{sym}.pkl"
            hist = _read_pickle(path)
            stats["checked"] += 1
            if not _needs_update(hist, target):
                stats["skipped"] += 1
                continue
            start = _fetch_start_for(hist, target)
            need.append((sym, path, hist, start))

        if need:
            batch_start = min(s for *_, s in need)
            tickers = [yfinance_ticker(sym) for sym, *_ in need]
            try:
                raw = _download_batch(tickers, batch_start, target)
            except Exception as exc:
                logger.exception("Batch download failed: %s", tickers)
                stats["errors"].append(f"batch {tickers[0]}…: {exc}")
                raw = pd.DataFrame()

            for sym, path, hist, _start in need:
                ticker = yfinance_ticker(sym)
                try:
                    if len(need) == 1 and not isinstance(getattr(raw, "columns", None), pd.MultiIndex):
                        fresh = _normalize_equity_frame(raw, ticker)
                    else:
                        fresh = _extract_from_batch(raw, ticker)
                    if fresh.empty:
                        # Fallback single-ticker download
                        single = _download_single(ticker, _start, target)
                        fresh = _normalize_equity_frame(single, ticker)
                    if fresh.empty:
                        stats["skipped"] += 1
                        continue
                    before_n = len(hist)
                    merged = _merge_bars(hist, fresh)
                    merged = merged[merged.index.date <= target]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    merged.to_pickle(path)
                    stats["updated"] += 1
                    stats["bars_added"] += max(0, len(merged) - before_n)
                except Exception as exc:
                    logger.warning("Equity sync failed for %s: %s", sym, exc)
                    stats["errors"].append(f"{sym}: {exc}")

            time.sleep(BATCH_PAUSE_SEC)

        done = progress_base + 1 + min(i + BATCH_SIZE, len(ordered))
        _set_status(
            message=f"Syncing equities… {min(i + BATCH_SIZE, len(ordered))}/{len(ordered)}",
            progress={"done": min(done, progress_total), "total": progress_total},
        )

    after_dates = []
    for p in EQUITY_DIR.glob("*.pkl"):
        d = _last_bar_date(_read_pickle(p))
        if d:
            after_dates.append(d)
    if after_dates:
        stats["last_date_after"] = max(after_dates).isoformat()

    return stats


def _sync_fno(target: date, progress_done: int, progress_total: int) -> dict:
    FNO_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "checked": 0,
        "updated": 0,
        "skipped": 0,
        "bars_added": 0,
        "errors": [],
        "instruments": {},
    }

    for key, meta in INSTRUMENTS.items():
        stats["checked"] += 1
        path = FNO_DIR / f"{key}.pkl"
        hist = _read_pickle(path)
        before = _last_bar_date(hist)
        ticker = meta["ticker"]
        try:
            if not _needs_update(hist, target):
                stats["skipped"] += 1
                stats["instruments"][key] = {
                    "last_date": before.isoformat() if before else None,
                    "bars_added": 0,
                    "skipped": True,
                }
                continue

            start = _fetch_start_for(hist, target)
            raw = _download_single(ticker, start, target)
            fresh = normalize_df(raw)
            if fresh.empty:
                stats["skipped"] += 1
                stats["instruments"][key] = {
                    "last_date": before.isoformat() if before else None,
                    "bars_added": 0,
                    "skipped": True,
                    "note": "no fresh data",
                }
                continue

            before_n = len(hist)
            merged = _merge_bars(hist, fresh)
            merged = merged[merged.index.date <= target]
            merged.to_pickle(path)
            added = max(0, len(merged) - before_n)
            stats["updated"] += 1
            stats["bars_added"] += added
            after = _last_bar_date(merged)
            stats["instruments"][key] = {
                "last_date": after.isoformat() if after else None,
                "bars_added": added,
                "skipped": False,
            }
        except Exception as exc:
            logger.exception("F&O sync failed for %s", key)
            stats["errors"].append(f"{key}: {exc}")
            stats["instruments"][key] = {
                "last_date": before.isoformat() if before else None,
                "error": str(exc),
            }

        progress_done += 1
        _set_status(
            message=f"Syncing F&O… {key}",
            progress={"done": progress_done, "total": progress_total},
        )

    return stats


def _execute_sync(target: date) -> dict:
    """Core sync body. Caller must hold `_sync_lock`."""
    symbols = ensure_nifty100_marked()
    existing = {p.stem for p in EQUITY_DIR.glob("*.pkl") if not p.stem.startswith("_")}
    equity_count = len(set(symbols) | existing)
    total = equity_count + 1 + len(INSTRUMENTS)

    _set_status(
        running=True,
        started_at=timezone.now().isoformat(),
        finished_at=None,
        message="Starting history sync…",
        target_date=target.isoformat(),
        progress={"done": 0, "total": total},
        equities={},
        fno={},
        error=None,
    )

    try:
        eq_stats = _sync_equities(target, progress_base=0, progress_total=total)
        fno_stats = _sync_fno(target, progress_done=equity_count + 1, progress_total=total)

        result = {
            "ok": True,
            "target_date": target.isoformat(),
            "equities": eq_stats,
            "fno": fno_stats,
            "coverage": get_history_coverage(),
        }
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=(
                f"Done through {target.isoformat()} — "
                f"equities +{eq_stats.get('bars_added', 0)} bars, "
                f"F&O +{fno_stats.get('bars_added', 0)} bars"
            ),
            progress={"done": total, "total": total},
            equities=eq_stats,
            fno=fno_stats,
            error=None,
        )
        return result
    except Exception as exc:
        logger.exception("Intraday history sync failed")
        _set_status(
            running=False,
            finished_at=timezone.now().isoformat(),
            message=f"Failed: {exc}",
            error=str(exc),
        )
        return {"ok": False, "error": str(exc), "status": get_sync_status()}


def run_intraday_history_sync(target: Optional[date] = None) -> dict:
    """
    Synchronously fill equity + F&O 5m pickles through `target` (default: today IST).
    Concurrent calls are rejected.
    """
    if not _sync_lock.acquire(blocking=False):
        return {
            "ok": False,
            "error": "A history sync is already running",
            "status": get_sync_status(),
        }
    try:
        return _execute_sync(target or _target_date())
    finally:
        _sync_lock.release()


def start_intraday_history_sync_async(target: Optional[date] = None) -> dict:
    """Kick off sync in a daemon thread; return immediately with status."""
    target = target or _target_date()
    if not _sync_lock.acquire(blocking=False):
        return {
            "ok": False,
            "started": False,
            "error": "A history sync is already running",
            "status": get_sync_status(),
        }

    _set_status(
        running=True,
        started_at=timezone.now().isoformat(),
        finished_at=None,
        message="Queued…",
        target_date=target.isoformat(),
        progress={"done": 0, "total": 0},
        equities={},
        fno={},
        error=None,
    )

    def _worker():
        try:
            _execute_sync(target)
        finally:
            _sync_lock.release()

    threading.Thread(
        target=_worker,
        name="intraday-history-sync",
        daemon=True,
    ).start()

    return {
        "ok": True,
        "started": True,
        "target_date": target.isoformat(),
        "status": get_sync_status(),
        "message": f"Fetching remaining 5m data through {target.isoformat()}…",
    }
