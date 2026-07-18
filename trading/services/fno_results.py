"""Load Elite ML Short v2 F&O backtest results and trade log."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from django.core.cache import cache

ROOT = Path(__file__).resolve().parent.parent.parent
IST = ZoneInfo("Asia/Kolkata")

RESULTS_PATH = ROOT / "data" / "intraday_fno_optimized.json"
TRADES_PATH = ROOT / "data" / "intraday_fno_optimized_trades.json"
MODEL_META_PATH = ROOT / "data" / "intraday_fno_ml_model.json"
FNO_DIR = ROOT / "data" / "intraday_fno"

_CACHE_TTL = 300


def _load_json(path: Path, cache_prefix: str) -> dict:
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cache_key = f"{cache_prefix}:{mtime}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    data = json.loads(path.read_text(encoding="utf-8"))
    cache.set(cache_key, data, _CACHE_TTL)
    return data


def load_strategy_results() -> dict:
    return _load_json(RESULTS_PATH, "fno:results")


def load_strategy_trades_data() -> dict:
    return _load_json(TRADES_PATH, "fno:trades")


def load_model_meta() -> dict:
    return _load_json(MODEL_META_PATH, "fno:model_meta")


def get_trades(period: str = "full") -> list[dict]:
    data = load_strategy_trades_data()
    if not data:
        return []
    if period == "oos":
        return data.get("oos", {}).get("trades", [])
    return data.get("full_period", {}).get("trades", [])


def _parse_session_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _session_dates_with_bars(instrument: str = "NIFTY") -> set[date]:
    """Trading session dates present in the local 5m pickle."""
    path = FNO_DIR / f"{instrument.upper()}.pkl"
    if not path.exists():
        return set()
    try:
        import pandas as pd

        df = pd.read_pickle(path)
        if df is None or df.empty:
            return set()
        return {d for d in df.index.date}
    except Exception:
        return set()


def _load_paper_trades_for_day(session: date, instrument: str | None = None) -> list[dict]:
    """Paper closed trades for a session (for F&O Live comparison)."""
    try:
        from trading.models import PaperTrade
    except Exception:
        return []

    qs = PaperTrade.objects.filter(session_date=session).order_by("entry_time")
    if instrument:
        qs = qs.filter(instrument=instrument.upper())
    out = []
    for t in qs:
        pnl = float(t.pnl or 0)
        out.append({
            "source": "paper",
            "instrument": t.instrument,
            "side": t.side,
            "session_date": t.session_date.isoformat(),
            "signal_time": (
                t.position.signal_bar_time
                if t.position_id and t.position and t.position.signal_bar_time
                else (t.entry_time.astimezone(IST).strftime("%H:%M") if t.entry_time else "")
            ),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "stop": float(t.stop_loss),
            "target": float(t.target),
            "lots": t.lots,
            "ml_prob": t.ml_prob,
            "pnl_inr": round(pnl, 2),
            "result": "WIN" if pnl > 0 else "LOSS",
            "exit_reason": t.exit_reason,
            "strategy": t.strategy_name,
        })
    return out


def _summarize_day(
    session: date,
    trades: list[dict],
    has_bars: bool,
    *,
    instrument: str | None = None,
) -> dict:
    day_trades = [t for t in trades if _parse_session_date(t.get("session_date")) == session]
    wins = sum(1 for t in day_trades if t.get("result") == "WIN")
    losses = len(day_trades) - wins
    pnl = round(sum(float(t.get("pnl_inr") or 0) for t in day_trades), 2)
    paper_trades = _load_paper_trades_for_day(session, instrument=instrument)
    paper_pnl = round(sum(float(t.get("pnl_inr") or 0) for t in paper_trades), 2)
    paper_wins = sum(1 for t in paper_trades if t.get("result") == "WIN")
    paper_losses = len(paper_trades) - paper_wins

    if day_trades:
        note = f"{len(day_trades)} backtest trade(s) · {wins}W / {losses}L"
        status = "traded"
    elif has_bars:
        note = "No backtest trades — setup/ML filters did not pass on completed bars"
        status = "flat"
    else:
        note = "No 5m history for this session"
        status = "no_data"

    if paper_trades and not day_trades:
        note += f" · Paper had {len(paper_trades)} trade(s) (live data may have differed)"
    elif paper_trades:
        note += f" · Paper {len(paper_trades)} trade(s)"

    return {
        "date": session.isoformat(),
        "label": session.strftime("%a %d %b"),
        "trades": day_trades,
        "count": len(day_trades),
        "wins": wins,
        "losses": losses,
        "pnl": pnl,
        "paper_trades": paper_trades,
        "paper_count": len(paper_trades),
        "paper_wins": paper_wins,
        "paper_losses": paper_losses,
        "paper_pnl": paper_pnl,
        "has_bars": has_bars,
        "status": status,
        "note": note,
    }


def get_recent_day_results(
    period: str = "full",
    *,
    instrument: str = "NIFTY",
    as_of: date | None = None,
) -> dict:
    """
    Yesterday + today backtest rollup for the F&O Live page.

    Always returns both days so a flat session is visible (not just missing rows
    in the full trade log). Also includes paper trades for side-by-side compare.
    """
    as_of = as_of or datetime.now(IST).date()
    today = as_of
    yesterday = as_of - timedelta(days=1)

    trades = get_trades(period)
    bar_dates = _session_dates_with_bars(instrument)
    data = load_strategy_trades_data() or load_strategy_results()
    data_range = data.get("data_range") or {}
    last_trade = max(
        (d for d in (_parse_session_date(t.get("session_date")) for t in trades) if d),
        default=None,
    )

    return {
        "as_of": as_of.isoformat(),
        "today": _summarize_day(today, trades, today in bar_dates, instrument=instrument),
        "yesterday": _summarize_day(yesterday, trades, yesterday in bar_dates, instrument=instrument),
        "data_range": data_range,
        "last_trade_date": last_trade.isoformat() if last_trade else None,
    }
