"""
Historical Stage Analysis 2.0 signals for the Signals chart page.

Uses the **same engine as the Backtest** (run_stage_v2_backtest) so signal counts
and stock lists match for identical date range + filters. Cards include portfolio
execution status (taken vs skipped for cash / cooldown / etc.).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from stage_analysis_v2.services.backtester import (
    STRATEGY_NAME,
    run_stage_v2_backtest,
)
from stage_analysis_v2.services.tech_filters import (
    DEFAULT_TECH_FILTER,
    TECH_FILTER_LABELS,
    normalize_tech_filter,
)
from trading.models import Stock
from trading.services.market_data import get_universe_symbols


@dataclass
class StageV2SignalHistory:
    strategy_name: str
    start_date: date
    end_date: date
    stocks_scanned: int = 0
    total_signals: int = 0
    total_trades: int = 0
    signals_skipped_cash: int = 0
    capital: float = 1_000_000.0
    min_quality_score: int = 0
    market_filter: bool = False
    tech_filter: str = DEFAULT_TECH_FILTER
    tech_filter_label: str = ""
    exit_mode: str = "stage_4_only"
    exit_mode_label: str = ""
    signals: list[dict[str, Any]] = field(default_factory=list)
    daily: list[dict[str, Any]] = field(default_factory=list)
    by_date: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _enrich_names(cards: list[dict[str, Any]]) -> None:
    symbols = list({c["symbol"] for c in cards})
    if not symbols:
        return
    name_map: dict[str, str] = {}
    sector_map: dict[str, str] = {}
    for s in Stock.objects.filter(symbol__in=symbols).only("symbol", "name", "sector"):
        name_map[s.symbol] = s.name
        sector_map[s.symbol] = s.sector
    for c in cards:
        c["company_name"] = name_map.get(c["symbol"], c.get("company_name") or "")
        c["sector"] = sector_map.get(c["symbol"], c.get("sector") or "")
        stop = float(c.get("stop_loss") or 0)
        close = float(c.get("signal_close") or 0)
        target = float(c.get("target") or 0)
        entry = c.get("entry_price")
        ref = float(entry) if entry else close
        risk = ref - stop if ref and stop and ref > stop else 0.0
        reward = target - ref if target and ref else 0.0
        c["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0.0


def collect_stage_v2_signal_history(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_quality_score: int = 0,
    market_filter: bool = False,
    tech_filter: str = DEFAULT_TECH_FILTER,
    capital: float = 1_000_000.0,
    exit_mode: str = "stage_4_only",
) -> StageV2SignalHistory:
    """
    Scan universe for Stage 2 entry signals using the backtest engine.

    Bars are keyed by **entry day** (next session after weekly Stage 2 confirm).
    Each card includes whether the shared-capital backtest would take the trade.
    """
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=183))  # ~6 months
    tech_filter = normalize_tech_filter(tech_filter)
    symbols = symbols or get_universe_symbols(nifty200_only=True)

    bt = run_stage_v2_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        min_quality_score=min_quality_score,
        market_filter=market_filter,
        exit_mode=exit_mode,
        tech_filter=tech_filter,
    )

    cards: list[dict[str, Any]] = list(bt.signal_log or [])
    _enrich_names(cards)

    result = StageV2SignalHistory(
        strategy_name=STRATEGY_NAME,
        start_date=start_date,
        end_date=end_date,
        stocks_scanned=bt.stocks_scanned,
        total_signals=bt.total_signals,
        total_trades=bt.total_trades,
        signals_skipped_cash=bt.signals_skipped_cash,
        capital=capital,
        min_quality_score=min_quality_score,
        market_filter=market_filter,
        tech_filter=bt.tech_filter,
        tech_filter_label=bt.tech_filter_label or TECH_FILTER_LABELS.get(tech_filter, tech_filter),
        exit_mode=bt.exit_mode,
        exit_mode_label=bt.exit_mode_label,
        signals=cards,
    )

    by_date: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_date[card["entry_date"]].append(card)

    daily: list[dict[str, Any]] = []
    for d in sorted(by_date.keys()):
        rows = by_date[d]
        rows.sort(key=lambda r: (-int(r.get("quality_score") or 0), r["symbol"]))
        by_date[d] = rows
        avg_q = sum(int(r.get("quality_score") or 0) for r in rows) / len(rows)
        avg_rs = sum(float(r.get("rs_rating") or 0) for r in rows) / len(rows)
        high_q = sum(1 for r in rows if int(r.get("quality_score") or 0) >= 75)
        taken = sum(1 for r in rows if r.get("status") == "taken")
        symbols_str = ", ".join(r["symbol"] for r in rows[:8])
        if len(rows) > 8:
            symbols_str += f" +{len(rows) - 8}"
        daily.append({
            "date": d,
            "count": len(rows),
            "taken": taken,
            "skipped": len(rows) - taken,
            "avg_quality": round(avg_q, 1),
            "avg_rs": round(avg_rs, 1),
            "high_quality": high_q,
            "symbols_preview": symbols_str,
            "pnl": 0.0,
            "wins": high_q,
            "losses": len(rows) - high_q,
        })

    result.daily = daily
    result.by_date = dict(by_date)

    taken_n = sum(1 for c in cards if c.get("status") == "taken")
    skipped_n = len(cards) - taken_n
    avg_quality = (
        round(sum(int(c.get("quality_score") or 0) for c in cards) / len(cards), 1)
        if cards else 0
    )
    avg_rs = (
        round(sum(float(c.get("rs_rating") or 0) for c in cards) / len(cards), 1)
        if cards else 0
    )
    result.summary = {
        "signals": len(cards),
        "trades": taken_n,
        "skipped": skipped_n,
        "skipped_cash": bt.signals_skipped_cash,
        "days_with_signals": len(daily),
        "avg_quality": avg_quality,
        "avg_rs": avg_rs,
        "high_quality": sum(1 for c in cards if int(c.get("quality_score") or 0) >= 75),
        "stocks_scanned": result.stocks_scanned,
        "total_return_pct": bt.total_return_pct,
        "win_rate": bt.win_rate,
        # Same numbers the Backtest page shows for this run
        "backtest_signals": bt.total_signals,
        "backtest_trades": bt.total_trades,
    }
    return result
