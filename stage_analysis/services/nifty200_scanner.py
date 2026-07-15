"""
Scan Nifty 200 universe for Stage 2 (Advancing) stocks.

Uses local DailyPrice data (fast) with yfinance fallback for missing history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from stage_analysis.models import Stage
from stage_analysis.services.stage_detector import (
    MA_PERIOD,
    MA_SLOPE_LOOKBACK,
    StageResult,
    analyze_ticker,
    daily_to_weekly,
    normalize_ticker,
    result_from_weekly_df,
)
from trading.models import Stock
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.nse_price_sync import yfinance_ticker


@dataclass
class Nifty200ScanResult:
    stage2: list[StageResult] = field(default_factory=list)
    scanned: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    total: int = 0

    @property
    def stage2_count(self) -> int:
        return len(self.stage2)


def _analyze_symbol(symbol: str, company_name: str) -> StageResult:
    """Analyze one NSE symbol using DB daily bars, falling back to yfinance."""
    daily = load_price_dataframe(symbol)
    ticker = yfinance_ticker(symbol)

    if not daily.empty:
        weekly = daily_to_weekly(daily)
        if len(weekly) >= MA_PERIOD + MA_SLOPE_LOOKBACK:
            return result_from_weekly_df(
                ticker,
                company_name,
                weekly,
                include_chart=False,
            )

    return analyze_ticker(ticker)


def scan_nifty200_stage2() -> Nifty200ScanResult:
    """Return all Nifty 200 stocks currently in Stage 2."""
    symbols = get_universe_symbols(nifty200_only=True)
    names = dict(
        Stock.objects.filter(symbol__in=symbols).values_list("symbol", "name")
    )

    result = Nifty200ScanResult(total=len(symbols))

    for symbol in symbols:
        company_name = names.get(symbol) or symbol
        try:
            analysis = _analyze_symbol(symbol, company_name)
            result.scanned += 1
            if analysis.stage == Stage.ADVANCING:
                result.stage2.append(analysis)
        except ValueError as exc:
            result.errors.append((symbol, str(exc)))
            result.skipped += 1

    result.stage2.sort(key=lambda r: r.price_vs_ma_pct, reverse=True)
    return result