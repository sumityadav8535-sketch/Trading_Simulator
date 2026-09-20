"""
Fundamental Compounders — fundamentals-only swing (12-month hold).

Ranks NSE names on quality, growth, balance sheet, valuation, and ownership.
No technical indicators. Rebalances on 1 July (≈90 days after a typical
Indian FY-end) using only statements that would already have been published.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Fundamental Compounders"
CACHE_NAME = "fundamental_swing_cache.json"
RESULTS_NAME = "fundamental_swing_results.json"
STATEMENT_LAG_DAYS = 90
REBALANCE_MONTH = 7
REBALANCE_DAY = 1
DEFAULT_CAPITAL = 1_000_000.0
MAX_FETCH_WORKERS = 6

INFO_KEYS = [
    "longName",
    "shortName",
    "sector",
    "industry",
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "profitMargins",
    "operatingMargins",
    "returnOnEquity",
    "returnOnAssets",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
    "grossMargins",
    "heldPercentInsiders",
    "heldPercentInstitutions",
    "enterpriseToEbitda",
    "trailingEps",
    "forwardEps",
    "bookValue",
    "dividendYield",
    "payoutRatio",
    "marketCap",
    "beta",
    "freeCashflow",
    "operatingCashflow",
    "totalRevenue",
    "totalCash",
    "totalDebt",
    "ebitda",
    "recommendationKey",
    "targetMeanPrice",
    "numberOfAnalystOpinions",
    "averageVolume",
    "sharesOutstanding",
    "enterpriseValue",
    "currentPrice",
    "regularMarketPrice",
    "financialCurrency",
    "currency",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
]

INCOME_ALIASES = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "net_income": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ),
    "operating_income": ("Operating Income", "Total Operating Income As Reported", "EBIT"),
    "gross_profit": ("Gross Profit",),
    "ebit": ("EBIT",),
    "interest_expense": ("Interest Expense",),
    "diluted_eps": ("Diluted EPS", "Basic EPS"),
    "shares": ("Diluted Average Shares", "Basic Average Shares"),
}
BALANCE_ALIASES = {
    "equity": ("Stockholders Equity", "Common Stock Equity", "Tangible Book Value"),
    "assets": ("Total Assets",),
    "debt": ("Total Debt",),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "shares": ("Ordinary Shares Number", "Share Issued"),
    "cash": ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
}
CASHFLOW_ALIASES = {
    "fcf": ("Free Cash Flow",),
    "ocf": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
}

FINANCIAL_SECTORS = {"financial services", "financials", "insurance"}
FINANCIAL_INDUSTRY_HINTS = (
    "bank",
    "insurance",
    "credit services",
    "asset management",
    "capital markets",
    "mortgage",
)

WINDOW_YEARS = (1, 2, 3, 4, 5)


@dataclass
class FundParams:
    name: str = "Aggressive Growth"
    min_roe: float = 12.0
    min_profit_margin: float = 5.0
    min_operating_margin: float = 0.0
    min_revenue_growth: float = 20.0
    min_earnings_growth: float = 25.0
    max_pe: float = 60.0
    max_peg: float = 3.0
    max_de: float = 1.5
    min_current_ratio: float = 1.0
    min_score: float = 40.0
    top_n: int = 3
    min_market_cap: float = 20_000_000_000  # ₹2,000 Cr
    exclude_financials: bool = False
    rebalance: str = "july"  # july | monthly | quarterly | 21d
    rank_by: str = "score"  # score | mom3 | mom6 | rs6 | score_mom
    tech: str = "none"  # none | sma150 | trend | trend_rs | pullback | breakout
    trail_sma: int = 0  # 0, 20, 50


AGGRESSIVE_GROWTH = FundParams(
    name="Aggressive Growth",
    min_roe=12.0,
    min_profit_margin=5.0,
    min_revenue_growth=20.0,
    min_earnings_growth=25.0,
    max_pe=60.0,
    max_peg=3.0,
    max_de=1.5,
    min_score=40.0,
    top_n=3,
)
QUALITY_GROWTH = FundParams(
    name="Quality Growth",
    min_roe=15.0,
    min_profit_margin=8.0,
    min_revenue_growth=8.0,
    min_earnings_growth=10.0,
    max_pe=45.0,
    max_peg=2.5,
    max_de=1.0,
    min_score=50.0,
    top_n=10,
)
GARP = FundParams(
    name="GARP (growth at a reasonable price)",
    min_roe=12.0,
    min_profit_margin=6.0,
    min_revenue_growth=5.0,
    min_earnings_growth=8.0,
    max_pe=28.0,
    max_peg=1.8,
    max_de=0.8,
    min_score=45.0,
    top_n=10,
)
MOMENTUM_SWING = FundParams(
    name="Momentum Swing",
    min_roe=12.0,
    min_profit_margin=5.0,
    min_revenue_growth=20.0,
    min_earnings_growth=25.0,
    max_pe=60.0,
    max_peg=3.0,
    max_de=1.5,
    min_score=40.0,
    top_n=1,
    rebalance="21d",
    rank_by="mom3",
    tech="sma150",
    trail_sma=0,
)
DEFAULT_PACKS = (MOMENTUM_SWING, AGGRESSIVE_GROWTH, QUALITY_GROWTH, GARP)
DEFAULT_PARAMS = MOMENTUM_SWING


def _in_tests() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return any(arg == "test" or arg.endswith("test") for arg in sys.argv)


def _cache_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / CACHE_NAME


def _results_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / RESULTS_NAME


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def as_pct(value: Any) -> float | None:
    """Normalise Yahoo fractions (0.32) and already-percent values (32)."""
    v = _to_float(value)
    if v is None:
        return None
    if abs(v) <= 1.5:
        return v * 100.0
    return v


def de_ratio_from_yahoo(value: Any) -> float | None:
    """Yahoo debtToEquity is percent (9.5 = 9.5%). Return a ratio (0.095)."""
    v = _to_float(value)
    if v is None:
        return None
    return v / 100.0


def is_financial(sector: str | None, industry: str | None) -> bool:
    sec = (sector or "").strip().lower()
    ind = (industry or "").strip().lower()
    if sec in FINANCIAL_SECTORS:
        return True
    return any(h in ind for h in FINANCIAL_INDUSTRY_HINTS)


def _r(value: Any, digits: int = 2) -> float | None:
    v = _to_float(value)
    if v is None:
        return None
    return round(v, digits)


def _pick(row: dict[str, Any], names: Iterable[str]) -> float | None:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        raw = row.get(name)
        if raw is None:
            raw = lower.get(name.lower())
        v = _to_float(raw)
        if v is not None:
            return v
    return None


def _band_score(value: float | None, tiers: list[tuple[float, int]], *, higher_is_better: bool = True) -> int:
    if value is None:
        return 0
    ordered = list(tiers)
    if higher_is_better:
        for threshold, pts in ordered:
            if value >= threshold:
                return pts
        return 0
    for threshold, pts in ordered:
        if value <= threshold:
            return pts
    return 0


def metrics_from_info(info: dict[str, Any]) -> dict[str, Any]:
    price = _to_float(info.get("currentPrice")) or _to_float(info.get("regularMarketPrice"))
    fcf = _to_float(info.get("freeCashflow"))
    mcap = _to_float(info.get("marketCap"))
    target = _to_float(info.get("targetMeanPrice"))
    fcf_yield = None
    if fcf is not None and mcap and mcap > 0:
        fcf_yield = (fcf / mcap) * 100.0
    target_upside = None
    if target is not None and price and price > 0:
        target_upside = (target / price - 1.0) * 100.0
    return {
        "name": info.get("longName") or info.get("shortName") or "",
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "financial_currency": info.get("financialCurrency") or "",
        "price": price,
        "pe": _to_float(info.get("trailingPE")),
        "forward_pe": _to_float(info.get("forwardPE")),
        "peg": _to_float(info.get("pegRatio")),
        "pb": _to_float(info.get("priceToBook")),
        "ev_ebitda": _to_float(info.get("enterpriseToEbitda")),
        "roe": as_pct(info.get("returnOnEquity")),
        "roa": as_pct(info.get("returnOnAssets")),
        "profit_margin": as_pct(info.get("profitMargins")),
        "operating_margin": as_pct(info.get("operatingMargins")),
        "gross_margin": as_pct(info.get("grossMargins")),
        "revenue_growth": as_pct(info.get("revenueGrowth")),
        "earnings_growth": as_pct(info.get("earningsGrowth")),
        "q_earnings_growth": as_pct(info.get("earningsQuarterlyGrowth")),
        "de_ratio": de_ratio_from_yahoo(info.get("debtToEquity")),
        "current_ratio": _to_float(info.get("currentRatio")),
        "quick_ratio": _to_float(info.get("quickRatio")),
        "insider_pct": as_pct(info.get("heldPercentInsiders")),
        "institution_pct": as_pct(info.get("heldPercentInstitutions")),
        "dividend_yield": as_pct(info.get("dividendYield")),
        "payout_ratio": as_pct(info.get("payoutRatio")),
        "beta": _to_float(info.get("beta")),
        "market_cap": mcap,
        "market_cap_cr": (mcap / 1e7) if mcap else None,
        "fcf_yield": fcf_yield,
        "eps": _to_float(info.get("trailingEps")),
        "target_price": target,
        "target_upside_pct": target_upside,
        "recommendation": info.get("recommendationKey") or "",
        "analysts": info.get("numberOfAnalystOpinions"),
        "source": "info",
    }


def _sorted_fy(statements: dict[str, dict]) -> list[str]:
    return sorted(k for k in statements.keys() if k)


def latest_statement_on_or_before(
    statements: dict[str, dict],
    asof: date,
    *,
    lag_days: int = STATEMENT_LAG_DAYS,
) -> tuple[str, dict] | None:
    best: tuple[date, str, dict] | None = None
    for raw, row in (statements or {}).items():
        try:
            fy = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        available = fy + timedelta(days=int(lag_days))
        if available > asof:
            continue
        if best is None or fy > best[0]:
            best = (fy, str(raw)[:10], row or {})
    if best is None:
        return None
    return best[1], best[2]


def previous_statement(statements: dict[str, dict], fy: str) -> tuple[str, dict] | None:
    keys = _sorted_fy(statements)
    if fy not in keys:
        return None
    i = keys.index(fy)
    if i <= 0:
        return None
    prev = keys[i - 1]
    return prev, statements.get(prev) or {}


def metrics_from_statements(
    income: dict[str, dict],
    balance: dict[str, dict],
    cashflow: dict[str, dict],
    asof: date,
    *,
    price: float | None = None,
    financial_currency: str = "",
    lag_days: int = STATEMENT_LAG_DAYS,
) -> dict[str, Any]:
    inc_pair = latest_statement_on_or_before(income or {}, asof, lag_days=lag_days)
    if inc_pair is None:
        return {"source": "statements", "asof": asof.isoformat()}
    fy, inc = inc_pair
    bal = (balance or {}).get(fy) or {}
    if not bal:
        bal_pair = latest_statement_on_or_before(balance or {}, asof, lag_days=lag_days)
        bal = bal_pair[1] if bal_pair else {}
    cf = (cashflow or {}).get(fy) or {}
    if not cf:
        cf_pair = latest_statement_on_or_before(cashflow or {}, asof, lag_days=lag_days)
        cf = cf_pair[1] if cf_pair else {}

    revenue = _pick(inc, INCOME_ALIASES["revenue"])
    net_income = _pick(inc, INCOME_ALIASES["net_income"])
    operating_income = _pick(inc, INCOME_ALIASES["operating_income"])
    gross_profit = _pick(inc, INCOME_ALIASES["gross_profit"])
    equity = _pick(bal, BALANCE_ALIASES["equity"])
    assets = _pick(bal, BALANCE_ALIASES["assets"])
    debt = _pick(bal, BALANCE_ALIASES["debt"])
    current_assets = _pick(bal, BALANCE_ALIASES["current_assets"])
    current_liab = _pick(bal, BALANCE_ALIASES["current_liabilities"])
    fcf = _pick(cf, CASHFLOW_ALIASES["fcf"])
    eps = _pick(inc, INCOME_ALIASES["diluted_eps"])

    prev = previous_statement(income or {}, fy)
    prev_rev = _pick(prev[1], INCOME_ALIASES["revenue"]) if prev else None
    prev_ni = _pick(prev[1], INCOME_ALIASES["net_income"]) if prev else None

    def _margin(num, den):
        if num is None or den in (None, 0):
            return None
        return (num / den) * 100.0

    def _growth(cur, prev_v):
        if cur is None or prev_v in (None, 0):
            return None
        return (cur / prev_v - 1.0) * 100.0

    pe = None
    ccy = (financial_currency or "").upper()
    if price and eps and eps > 0 and ccy in ("", "INR"):
        pe = price / eps

    fcf_yield = None
    if fcf is not None and price and eps and eps > 0 and ccy in ("", "INR"):
        shares = _pick(inc, INCOME_ALIASES["shares"]) or _pick(bal, BALANCE_ALIASES["shares"])
        if shares and shares > 0:
            mcap = price * shares
            if mcap > 0:
                fcf_yield = (fcf / mcap) * 100.0

    return {
        "source": "statements",
        "asof": asof.isoformat(),
        "fy": fy,
        "revenue": revenue,
        "net_income": net_income,
        "pe": pe,
        "roe": _margin(net_income, equity),
        "roa": _margin(net_income, assets),
        "profit_margin": _margin(net_income, revenue),
        "operating_margin": _margin(operating_income, revenue),
        "gross_margin": _margin(gross_profit, revenue),
        "revenue_growth": _growth(revenue, prev_rev),
        "earnings_growth": _growth(net_income, prev_ni),
        "de_ratio": (debt / equity) if debt is not None and equity not in (None, 0) else None,
        "current_ratio": (current_assets / current_liab) if current_assets is not None and current_liab not in (None, 0) else None,
        "fcf_yield": fcf_yield,
        "eps": eps,
        "price": price,
        "financial_currency": financial_currency,
    }


def score_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    sector = metrics.get("sector") or ""
    industry = metrics.get("industry") or ""
    financial = is_financial(sector, industry)

    growth = _band_score(metrics.get("earnings_growth"), [(40, 15), (25, 12), (15, 9), (8, 6), (0, 3)])
    growth += _band_score(metrics.get("revenue_growth"), [(25, 15), (15, 12), (8, 8), (0, 4)])
    quality = _band_score(metrics.get("roe"), [(25, 12), (18, 9), (12, 6), (8, 3)])
    quality += _band_score(metrics.get("profit_margin"), [(20, 10), (12, 7), (8, 4), (0, 2)])
    quality += _band_score(metrics.get("operating_margin"), [(20, 8), (12, 5), (8, 3)])
    if financial:
        balance = 8
        de_pts = 5
        cr_pts = 3
    else:
        de_pts = _band_score(metrics.get("de_ratio"), [(0.3, 8), (0.7, 6), (1.0, 4), (1.5, 2)], higher_is_better=False)
        cr_pts = _band_score(metrics.get("current_ratio"), [(2.0, 7), (1.5, 5), (1.2, 3), (1.0, 1)])
        balance = de_pts + cr_pts
    peg_pts = _band_score(metrics.get("peg"), [(0.8, 8), (1.2, 6), (1.8, 4), (2.5, 2)], higher_is_better=False)
    pe = metrics.get("pe")
    pe_pts = 0
    if pe is not None and pe > 0:
        if 8 <= pe <= 22:
            pe_pts = 7
        elif pe < 8:
            pe_pts = 3
        elif pe <= 35:
            pe_pts = 5
        elif pe <= 50:
            pe_pts = 2
    valuation = peg_pts + pe_pts
    insider_pts = _band_score(metrics.get("insider_pct"), [(20, 5), (10, 3), (5, 1)])
    inst_pts = _band_score(metrics.get("institution_pct"), [(40, 5), (20, 3), (10, 1)])
    ownership = insider_pts + inst_pts

    total = int(growth + quality + balance + valuation + ownership)
    reasons: list[str] = []
    if (metrics.get("earnings_growth") or 0) >= 25:
        reasons.append(f"Earnings growth {metrics['earnings_growth']:.0f}%")
    if (metrics.get("revenue_growth") or 0) >= 15:
        reasons.append(f"Revenue growth {metrics['revenue_growth']:.0f}%")
    if (metrics.get("roe") or 0) >= 18:
        reasons.append(f"ROE {metrics['roe']:.0f}%")
    if (metrics.get("profit_margin") or 0) >= 12:
        reasons.append(f"Profit margin {metrics['profit_margin']:.0f}%")
    if not financial and metrics.get("de_ratio") is not None and metrics["de_ratio"] <= 0.5:
        reasons.append(f"Low debt (D/E {metrics['de_ratio']:.2f})")
    if metrics.get("peg") is not None and 0 < metrics["peg"] <= 1.2:
        reasons.append(f"PEG {metrics['peg']:.2f}")
    if metrics.get("pe") is not None and 8 <= metrics["pe"] <= 25:
        reasons.append(f"PE {metrics['pe']:.1f}")

    return {
        "score": total,
        "breakdown": {
            "growth": growth,
            "quality": quality,
            "balance_sheet": balance,
            "valuation": valuation,
            "ownership": ownership,
        },
        "reasons": reasons,
        "financial": financial,
    }


def passes_filters(metrics: dict[str, Any], params: FundParams) -> tuple[bool, list[str]]:
    rejects: list[str] = []
    financial = is_financial(metrics.get("sector"), metrics.get("industry"))
    if params.exclude_financials and financial:
        return False, ["Financial sector excluded"]

    def _need(key: str, pred, label: str) -> None:
        val = metrics.get(key)
        if val is None:
            return
        if not pred(val):
            rejects.append(label)

    mcap = metrics.get("market_cap")
    if mcap is not None and params.min_market_cap and mcap < params.min_market_cap:
        rejects.append("Market cap too small")

    _need("roe", lambda v: v >= params.min_roe, f"ROE < {params.min_roe:.0f}%")
    _need("profit_margin", lambda v: v >= params.min_profit_margin, f"Margin < {params.min_profit_margin:.0f}%")
    if params.min_operating_margin:
        _need("operating_margin", lambda v: v >= params.min_operating_margin, "Operating margin weak")
    _need("revenue_growth", lambda v: v >= params.min_revenue_growth, f"Revenue growth < {params.min_revenue_growth:.0f}%")
    _need("earnings_growth", lambda v: v >= params.min_earnings_growth, f"Earnings growth < {params.min_earnings_growth:.0f}%")
    _need("pe", lambda v: 0 < v <= params.max_pe, f"PE > {params.max_pe:.0f}")
    _need("peg", lambda v: 0 < v <= params.max_peg, f"PEG > {params.max_peg:.1f}")
    if not financial:
        _need("de_ratio", lambda v: v <= params.max_de, f"D/E > {params.max_de:.1f}")
        if params.min_current_ratio:
            _need("current_ratio", lambda v: v >= params.min_current_ratio, "Current ratio weak")

    pm = metrics.get("profit_margin")
    ni_ok = metrics.get("net_income")
    if pm is not None and pm <= 0:
        rejects.append("Unprofitable")
    elif ni_ok is not None and ni_ok <= 0:
        rejects.append("Unprofitable")

    return (len(rejects) == 0), rejects


def _stock_payload(symbol: str, metrics: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": metrics.get("name") or symbol,
        "sector": metrics.get("sector") or "",
        "industry": metrics.get("industry") or "",
        "score": scored["score"],
        "breakdown": scored["breakdown"],
        "reasons": scored["reasons"],
        "financial": scored["financial"],
        "roe": _r(metrics.get("roe"), 1),
        "roa": _r(metrics.get("roa"), 1),
        "profit_margin": _r(metrics.get("profit_margin"), 1),
        "operating_margin": _r(metrics.get("operating_margin"), 1),
        "gross_margin": _r(metrics.get("gross_margin"), 1),
        "revenue_growth": _r(metrics.get("revenue_growth"), 1),
        "earnings_growth": _r(metrics.get("earnings_growth"), 1),
        "q_earnings_growth": _r(metrics.get("q_earnings_growth"), 1),
        "pe": _r(metrics.get("pe"), 1),
        "forward_pe": _r(metrics.get("forward_pe"), 1),
        "peg": _r(metrics.get("peg"), 2),
        "pb": _r(metrics.get("pb"), 2),
        "de_ratio": _r(metrics.get("de_ratio"), 2),
        "current_ratio": _r(metrics.get("current_ratio"), 2),
        "fcf_yield": _r(metrics.get("fcf_yield"), 1),
        "insider_pct": _r(metrics.get("insider_pct"), 1),
        "institution_pct": _r(metrics.get("institution_pct"), 1),
        "dividend_yield": _r(metrics.get("dividend_yield"), 2),
        "beta": _r(metrics.get("beta"), 2),
        "market_cap_cr": _r(metrics.get("market_cap_cr"), 0),
        "price": _r(metrics.get("price"), 2),
        "target_upside_pct": _r(metrics.get("target_upside_pct"), 1),
        "recommendation": metrics.get("recommendation") or "",
        "fy": metrics.get("fy"),
        "source": metrics.get("source"),
    }


def rank_metrics(
    rows: list[tuple[str, dict[str, Any]]],
    params: FundParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for symbol, metrics in rows:
        scored = score_metrics(metrics)
        ok, reasons = passes_filters(metrics, params)
        payload = _stock_payload(symbol, metrics, scored)
        payload["reject_reasons"] = reasons
        if not ok or scored["score"] < params.min_score:
            if scored["score"] < params.min_score and ok:
                payload["reject_reasons"] = reasons + [f"Score {scored['score']} < {params.min_score:.0f}"]
            rejected.append(payload)
            continue
        passed.append(payload)
    passed.sort(key=lambda r: (-r["score"], r["symbol"]))
    rejected.sort(key=lambda r: (-r["score"], r["symbol"]))
    return passed, rejected


def july1_on_or_after(year: int) -> date:
    d = date(year, REBALANCE_MONTH, REBALANCE_DAY)
    if d.weekday() >= 5:
        d += timedelta(days=7 - d.weekday())
    return d


def format_short_date(d: date | None) -> str:
    if d is None:
        return "—"
    return f"{d.day} {d.strftime('%b %Y')}"


def parse_iso_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw).strip()[:10])
    except ValueError:
        return None


def params_by_name(name: str | None) -> FundParams:
    if not name:
        return DEFAULT_PARAMS
    needle = str(name).strip().lower()
    for pack in DEFAULT_PACKS:
        if pack.name.lower() == needle:
            return pack
    aliases = {
        "aggressive": AGGRESSIVE_GROWTH,
        "aggressive growth": AGGRESSIVE_GROWTH,
        "quality": QUALITY_GROWTH,
        "quality growth": QUALITY_GROWTH,
        "garp": GARP,
        "garp (growth at a reasonable price)": GARP,
        "momentum": MOMENTUM_SWING,
        "momentum swing": MOMENTUM_SWING,
    }
    return aliases.get(needle, DEFAULT_PARAMS)


def last_rebalance_on_or_before(asof: date) -> date:
    d = july1_on_or_after(asof.year)
    if d > asof:
        d = july1_on_or_after(asof.year - 1)
    return d


def next_rebalance_after(asof: date) -> date:
    d = july1_on_or_after(asof.year)
    if d <= asof:
        d = july1_on_or_after(asof.year + 1)
    return d


def rebalance_schedule(start: date, end: date) -> list[date]:
    dates: list[date] = [start]
    year = start.year
    while True:
        d = july1_on_or_after(year)
        if d > end:
            break
        if d > start:
            dates.append(d)
        year += 1
        if year > end.year + 1:
            break
    return dates


def _month_starts(days: list[date], start: date, end: date) -> list[date]:
    out: list[date] = []
    last_ym = None
    for d in days:
        if d < start or d > end:
            continue
        ym = (d.year, d.month)
        if ym != last_ym:
            out.append(d)
            last_ym = ym
    if out and out[0] != start:
        out = [start] + [d for d in out if d > start]
    elif not out:
        out = [start]
    return out


def rebalance_dates(
    start: date,
    end: date,
    calendar: pd.DatetimeIndex | None,
    mode: str = "july",
) -> list[date]:
    mode = (mode or "july").lower()
    if mode in ("july", "annual", "year"):
        return rebalance_schedule(start, end)
    days = _trading_days(calendar, start, end) if calendar is not None else []
    if not days:
        return rebalance_schedule(start, end)
    if mode in ("monthly", "month"):
        return _month_starts(days, start, end)
    if mode in ("quarterly", "quarter"):
        months = _month_starts(days, start, end)
        return [d for i, d in enumerate(months) if i == 0 or d.month in (1, 4, 7, 10)]
    if mode in ("21d", "21", "monthly_21"):
        return [days[i] for i in range(0, len(days), 21)]
    return rebalance_schedule(start, end)


def build_close_tech(closes: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, series in (closes or {}).items():
        if series is None or getattr(series, "empty", True) or len(series) < 30:
            continue
        close = series.astype(float).sort_index()
        df = pd.DataFrame({"close": close})
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma150"] = df["close"].rolling(150).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["ret21"] = df["close"].pct_change(21) * 100.0
        df["ret63"] = df["close"].pct_change(63) * 100.0
        df["ret126"] = df["close"].pct_change(126) * 100.0
        df["high20"] = df["close"].rolling(20).max()
        df["high252"] = df["close"].rolling(252, min_periods=120).max()
        out[sym] = df
    return out


def _tech_row(df: pd.DataFrame | None, asof: date) -> pd.Series | None:
    if df is None or getattr(df, "empty", True):
        return None
    sliced = df.loc[: pd.Timestamp(asof)]
    if sliced.empty:
        return None
    return sliced.iloc[-1]


def passes_tech_row(row: pd.Series, nifty_row: pd.Series | None, kind: str) -> bool:
    kind = (kind or "none").lower()
    if kind in ("none", "", "off"):
        return True
    close = _to_float(row.get("close"))
    sma150 = _to_float(row.get("sma150"))
    sma50 = _to_float(row.get("sma50"))
    ema20 = _to_float(row.get("ema20"))
    rsi = _to_float(row.get("rsi"))
    ret63 = _to_float(row.get("ret63"))
    high20 = _to_float(row.get("high20"))
    if close is None:
        return False
    if kind in ("sma150", "trend", "trend_rs", "pullback", "breakout"):
        if sma150 is None or close <= sma150:
            return False
    if kind in ("trend", "trend_rs", "pullback"):
        if sma50 is None or sma150 is None or not (close > sma50 > sma150):
            return False
    if kind == "trend_rs":
        nifty_mom = _to_float(nifty_row.get("ret63")) if nifty_row is not None else None
        if ret63 is None or nifty_mom is None or ret63 <= nifty_mom:
            return False
    if kind == "pullback":
        if rsi is None or not (40 <= rsi <= 68):
            return False
        if ema20 is None or close > ema20 * 1.08:
            return False
    if kind == "breakout":
        if high20 is None or close < high20 * 0.999:
            return False
        if rsi is not None and rsi < 50:
            return False
    return True


def _rank_key(row: pd.Series | None, nifty_row: pd.Series | None, score: float, kind: str) -> float:
    kind = (kind or "score").lower()
    if row is None:
        return float(score or 0) if kind == "score" else -999.0
    mom3 = _to_float(row.get("ret63"))
    mom6 = _to_float(row.get("ret126"))
    nifty_m6 = _to_float(nifty_row.get("ret126")) if nifty_row is not None else None
    rs6 = None if mom6 is None or nifty_m6 is None else mom6 - nifty_m6
    if kind == "mom3":
        return mom3 if mom3 is not None else -999.0
    if kind == "mom6":
        return mom6 if mom6 is not None else -999.0
    if kind == "rs6":
        return rs6 if rs6 is not None else -999.0
    if kind == "score_mom":
        return float(score or 0) * 0.4 + max(mom6 or 0.0, 0.0) * 0.6
    return float(score or 0)


def apply_tech_rank(
    passed: list[dict[str, Any]],
    asof: date,
    params: FundParams,
    tech: dict[str, pd.DataFrame] | None,
    nifty_symbol: str | None = None,
) -> list[dict[str, Any]]:
    if not passed:
        return []
    tech = tech or {}
    from trading.constants import NIFTY50_SYMBOL

    bench = nifty_symbol or NIFTY50_SYMBOL
    nifty_row = _tech_row(tech.get(bench), asof)
    scored: list[tuple[float, dict[str, Any]]] = []
    for p in passed:
        row = _tech_row(tech.get(p["symbol"]), asof)
        if params.tech not in ("none", "", None) and (row is None or not passes_tech_row(row, nifty_row, params.tech)):
            continue
        key = _rank_key(row, nifty_row, float(p.get("score") or 0), params.rank_by)
        payload = dict(p)
        if row is not None:
            payload["mom3"] = _r(row.get("ret63"), 1)
            payload["mom6"] = _r(row.get("ret126"), 1)
            payload["rsi"] = _r(row.get("rsi"), 1)
            payload["above_sma150"] = bool(
                _to_float(row.get("close")) and _to_float(row.get("sma150"))
                and float(row["close"]) > float(row["sma150"])
            )
        payload["rank_key"] = _r(key, 2)
        scored.append((key, payload))
    scored.sort(key=lambda t: (-t[0], t[1]["symbol"]))
    return [p for _, p in scored]


def close_on_or_before(series: pd.Series, d: date) -> float | None:
    if series is None or series.empty:
        return None
    sliced = series.loc[: pd.Timestamp(d)]
    if sliced.empty:
        return None
    v = _to_float(sliced.iloc[-1])
    if v is None or v <= 0:
        return None
    return v


def period_return(series: pd.Series, start: date, end: date) -> float | None:
    a = close_on_or_before(series, start)
    b = close_on_or_before(series, end)
    if a is None or b is None:
        return None
    return (b / a - 1.0) * 100.0


def _df_to_records(df: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if df is None or getattr(df, "empty", True):
        return {}
    out: dict[str, dict[str, float]] = {}
    for col in df.columns:
        try:
            key = pd.Timestamp(col).date().isoformat()
        except Exception:
            key = str(col)[:10]
        row: dict[str, float] = {}
        try:
            series = df[col]
        except Exception:
            continue
        for idx, val in series.items():
            v = _to_float(val)
            if v is None:
                continue
            row[str(idx)] = v
        if row:
            out[key] = row
    return out


def fetch_symbol(symbol: str) -> dict[str, Any]:
    import yfinance as yf
    from trading.services.nse_price_sync import yfinance_ticker

    ticker = yfinance_ticker(symbol)
    yt = yf.Ticker(ticker)
    info_raw = {}
    try:
        info_raw = yt.info or {}
    except Exception as exc:
        return {"symbol": symbol, "ticker": ticker, "error": f"info: {exc}"}
    info = {k: info_raw.get(k) for k in INFO_KEYS if info_raw.get(k) is not None}
    try:
        income = _df_to_records(yt.income_stmt)
    except Exception:
        income = {}
    try:
        balance = _df_to_records(yt.balance_sheet)
    except Exception:
        balance = {}
    try:
        cashflow = _df_to_records(yt.cashflow)
    except Exception:
        cashflow = {}
    return {
        "symbol": symbol,
        "ticker": ticker,
        "info": info,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "error": None,
    }


def load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {"fetched_at": None, "stocks": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"fetched_at": None, "stocks": {}}
    if not isinstance(raw, dict):
        return {"fetched_at": None, "stocks": {}}
    raw.setdefault("stocks", {})
    return raw


def save_cache(cache: dict[str, Any]) -> Path:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    return path


def refresh_cache(symbols: list[str], *, force: bool = False, progress: bool = False) -> dict[str, Any]:
    cache = load_cache()
    stocks: dict[str, Any] = dict(cache.get("stocks") or {})
    todo = [s for s in symbols if force or s not in stocks or (stocks.get(s) or {}).get("error")]
    if todo and _in_tests():
        return cache
    if todo:
        workers = min(MAX_FETCH_WORKERS, len(todo))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fetch_symbol, sym): sym for sym in todo}
            for fut in as_completed(futs):
                sym = futs[fut]
                done += 1
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"symbol": sym, "error": str(exc), "info": {}, "income": {}, "balance": {}, "cashflow": {}}
                stocks[sym] = row
                if progress and (done % 10 == 0 or done == len(todo)):
                    print(f"  fetched {done}/{len(todo)}", flush=True)
        cache = {
            "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stocks": stocks,
        }
        save_cache(cache)
    return cache


def current_metrics_row(symbol: str, blob: dict[str, Any]) -> dict[str, Any] | None:
    info = blob.get("info") or {}
    if not info:
        return None
    metrics = metrics_from_info(info)
    metrics["symbol"] = symbol
    return metrics


def statement_metrics_row(
    symbol: str,
    blob: dict[str, Any],
    asof: date,
    *,
    price: float | None = None,
) -> dict[str, Any] | None:
    info = blob.get("info") or {}
    metrics = metrics_from_statements(
        blob.get("income") or {},
        blob.get("balance") or {},
        blob.get("cashflow") or {},
        asof,
        price=price,
        financial_currency=str(info.get("financialCurrency") or ""),
    )
    if not metrics.get("fy") and metrics.get("net_income") is None and metrics.get("revenue") is None:
        return None
    metrics["symbol"] = symbol
    metrics["name"] = info.get("longName") or info.get("shortName") or symbol
    metrics["sector"] = info.get("sector") or ""
    metrics["industry"] = info.get("industry") or ""
    metrics["market_cap"] = _to_float(info.get("marketCap"))
    if metrics.get("market_cap"):
        metrics["market_cap_cr"] = metrics["market_cap"] / 1e7
    return metrics


def load_close_series(symbols: list[str]) -> dict[str, pd.Series]:
    from trading.models import DailyPrice

    qs = (
        DailyPrice.objects.filter(stock_id__in=symbols)
        .order_by("stock_id", "date")
        .values_list("stock_id", "date", "close")
    )
    buckets: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for sym, d, close in qs:
        v = _to_float(close)
        if v is None:
            continue
        buckets.setdefault(sym, []).append((pd.Timestamp(d), v))
    out: dict[str, pd.Series] = {}
    for sym, rows in buckets.items():
        idx, vals = zip(*rows)
        out[sym] = pd.Series(vals, index=pd.DatetimeIndex(idx))
    return out


def _trading_days(index: pd.DatetimeIndex, start: date, end: date) -> list[date]:
    sliced = index[(index.date >= start) & (index.date <= end)]
    return [ts.date() for ts in sliced]


def _max_drawdown(equities: list[float]) -> float:
    peak = None
    dd = 0.0
    for eq in equities:
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = min(dd, eq / peak - 1.0)
    return dd * 100.0


def _cagr(total_return_pct: float, start: date, end: date) -> float | None:
    days = (end - start).days
    if days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    total = 1.0 + total_return_pct / 100.0
    if total <= 0:
        return None
    return (total ** (1.0 / years) - 1.0) * 100.0


def rank_at_date(
    cache: dict[str, Any],
    symbols: list[str],
    asof: date,
    params: FundParams,
    closes: dict[str, pd.Series] | None = None,
    *,
    use_info: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    stocks = cache.get("stocks") or {}
    empty = pd.Series(dtype=float)
    price_map = closes or {}
    for sym in symbols:
        blob = stocks.get(sym) or {}
        if use_info:
            metrics = current_metrics_row(sym, blob)
        else:
            px = close_on_or_before(price_map.get(sym, empty), asof)
            metrics = statement_metrics_row(sym, blob, asof, price=px)
        if not metrics:
            continue
        rows.append((sym, metrics))
    return rank_metrics(rows, params)


def pick_at_date(
    cache: dict[str, Any],
    symbols: list[str],
    asof: date,
    params: FundParams,
    closes: dict[str, pd.Series] | None = None,
    *,
    use_info: bool = False,
    tech: dict[str, pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    passed, _ = rank_at_date(cache, symbols, asof, params, closes, use_info=use_info)
    ranked = apply_tech_rank(passed, asof, params, tech)
    return ranked[: params.top_n]


def run_window(
    cache: dict[str, Any],
    symbols: list[str],
    start: date,
    end: date,
    params: FundParams,
    closes: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    *,
    capital: float = DEFAULT_CAPITAL,
) -> dict[str, Any]:
    need_tech = (params.tech not in ("none", "", None)) or (params.rank_by not in ("score", "", None)) or int(params.trail_sma or 0) > 0
    tech = build_close_tech(closes) if need_tech else {}
    reb_dates = rebalance_dates(start, end, calendar, params.rebalance)
    days = _trading_days(calendar, start, end)
    if not days:
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_return_pct": None,
            "error": "no trading days",
        }

    cash = float(capital)
    holdings: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    round_trips: list[dict[str, Any]] = []
    rebalance_log: list[dict[str, Any]] = []
    next_reb_i = 0

    def _mark(d: date) -> float:
        eq = cash
        for h in holdings:
            px = close_on_or_before(closes.get(h["symbol"], pd.Series(dtype=float)), d)
            if px is None:
                px = h.get("last_px") or h["entry_px"]
            h["last_px"] = px
            eq += h["qty"] * px
        return eq

    def _close_one(h: dict[str, Any], d: date, reason: str) -> float:
        px = close_on_or_before(closes.get(h["symbol"], pd.Series(dtype=float)), d) or h.get("last_px") or h["entry_px"]
        ret = (px / h["entry_px"] - 1.0) * 100.0 if h["entry_px"] else None
        round_trips.append({
            "symbol": h["symbol"],
            "name": h.get("name") or h["symbol"],
            "score": h.get("score"),
            "entry_date": h["entry_date"],
            "exit_date": d.isoformat(),
            "entry_px": _r(h["entry_px"], 2),
            "exit_px": _r(px, 2),
            "return_pct": _r(ret, 2),
            "doubled": bool(ret is not None and ret >= 100),
            "reason": reason,
        })
        return h["qty"] * px

    def _liquidate(d: date, reason: str) -> None:
        nonlocal cash, holdings
        for h in holdings:
            cash += _close_one(h, d, reason)
        holdings = []

    def _enter(d: date, picks: list[dict[str, Any]]) -> None:
        nonlocal cash, holdings
        priced = []
        for p in picks:
            px = close_on_or_before(closes.get(p["symbol"], pd.Series(dtype=float)), d)
            if px is None or px <= 0:
                continue
            priced.append((p, px))
        if not priced or cash <= 0:
            holdings = []
            return
        slice_amt = cash / len(priced)
        new_holdings = []
        spent = 0.0
        for p, px in priced:
            qty = slice_amt / px
            spent += slice_amt
            new_holdings.append({
                "symbol": p["symbol"],
                "name": p.get("name") or p["symbol"],
                "score": p.get("score"),
                "entry_date": d.isoformat(),
                "entry_px": px,
                "qty": qty,
                "last_px": px,
            })
        cash -= spent
        holdings = new_holdings

    for d in days:
        while next_reb_i < len(reb_dates) and reb_dates[next_reb_i] <= d:
            reb = reb_dates[next_reb_i]
            if holdings:
                _liquidate(d, "rebalance")
            picks = pick_at_date(cache, symbols, reb, params, closes, use_info=False, tech=tech)
            _enter(d, picks)
            rebalance_log.append({
                "date": d.isoformat(),
                "asof": reb.isoformat(),
                "picks": [
                    {
                        "symbol": p["symbol"],
                        "name": p.get("name"),
                        "score": p.get("score"),
                        "roe": p.get("roe"),
                        "earnings_growth": p.get("earnings_growth"),
                        "revenue_growth": p.get("revenue_growth"),
                        "profit_margin": p.get("profit_margin"),
                        "pe": p.get("pe"),
                    }
                    for p in picks
                ],
                "count": len(picks),
            })
            next_reb_i += 1
        trail_n = int(params.trail_sma or 0)
        if trail_n and holdings:
            col = {20: "sma20", 50: "sma50", 150: "sma150"}.get(trail_n)
            if col:
                keep: list[dict[str, Any]] = []
                for h in holdings:
                    row = _tech_row(tech.get(h["symbol"]), d)
                    px = _to_float(row.get("close")) if row is not None else None
                    line = _to_float(row.get(col)) if row is not None else None
                    if px is not None and line is not None and px < line:
                        cash += _close_one(h, d, "trail")
                    else:
                        keep.append(h)
                holdings = keep
        eq = _mark(d)
        if not equity_curve or equity_curve[-1]["date"] != d.isoformat():
            equity_curve.append({"date": d.isoformat(), "equity": round(eq, 2)})

    if holdings:
        _liquidate(end, "window_end")

    final_eq = equity_curve[-1]["equity"] if equity_curve else capital
    total_ret = (final_eq / capital - 1.0) * 100.0 if capital else None
    rets = [t["return_pct"] for t in round_trips if t.get("return_pct") is not None]
    wins = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r <= 0)
    doubled = [t for t in round_trips if t.get("doubled")]
    sampled = equity_curve
    if len(sampled) > 400:
        step = max(1, len(sampled) // 260)
        sampled = sampled[::step]
        if sampled[-1] != equity_curve[-1]:
            sampled.append(equity_curve[-1])

    one_year_holds = [t for t in round_trips if t.get("reason") in ("rebalance", "window_end")]
    best_stock_year = max(one_year_holds, key=lambda t: t.get("return_pct") or -999, default=None)

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": f"{start.isoformat()} → {end.isoformat()}",
        "capital": capital,
        "final_equity": round(final_eq, 2),
        "total_return_pct": _r(total_ret, 2),
        "cagr_pct": _r(_cagr(total_ret or 0.0, start, end), 2),
        "max_drawdown_pct": _r(_max_drawdown([p["equity"] for p in equity_curve]), 2),
        "trades": len(round_trips),
        "wins": wins,
        "losses": losses,
        "win_rate": _r((wins / len(rets) * 100.0) if rets else None, 1),
        "avg_hold_return_pct": _r(sum(rets) / len(rets) if rets else None, 2),
        "doublers": len(doubled),
        "hit_100": bool(total_ret is not None and total_ret >= 100),
        "best_stock_year": best_stock_year,
        "rebalances": rebalance_log,
        "holdings_history": round_trips,
        "equity_curve": sampled,
    }


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def yearly_slices(start: date, end: date) -> list[tuple[date, date, str]]:
    out: list[tuple[date, date, str]] = []
    cursor = start
    while cursor < end:
        nxt = _add_years(cursor, 1)
        chunk_end = min(nxt, end)
        if chunk_end <= cursor:
            break
        out.append((cursor, chunk_end, f"{cursor.isoformat()} → {chunk_end.isoformat()}"))
        cursor = chunk_end
    return out


def index_return(closes: pd.Series, start: date, end: date) -> float | None:
    return period_return(closes, start, end)


def params_to_dict(params: FundParams) -> dict[str, Any]:
    return asdict(params)


def current_ranking(cache: dict[str, Any], symbols: list[str], params: FundParams) -> tuple[list[dict], list[dict]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    stocks = cache.get("stocks") or {}
    for sym in symbols:
        metrics = current_metrics_row(sym, stocks.get(sym) or {})
        if metrics:
            rows.append((sym, metrics))
    return rank_metrics(rows, params)


def _cache_symbols(cache: dict[str, Any]) -> list[str]:
    return [str(s) for s in (cache.get("stocks") or {}).keys() if s]


def attach_mark_to_market(
    picks: list[dict[str, Any]],
    asof: date,
    *,
    entry_date: date | None = None,
) -> None:
    if not picks:
        return
    try:
        closes = load_close_series([p["symbol"] for p in picks if p.get("symbol")])
    except Exception:
        return
    for p in picks:
        series = closes.get(p["symbol"])
        last_px = close_on_or_before(series, asof) if series is not None else None
        if last_px is not None:
            p["last_px"] = _r(last_px, 2)
            if not p.get("price"):
                p["price"] = p["last_px"]
        if entry_date is None or series is None:
            continue
        entry_px = close_on_or_before(series, entry_date)
        if entry_px:
            p["entry_px"] = _r(entry_px, 2)
            p["entry_date"] = entry_date.isoformat()
            if last_px:
                p["since_entry_pct"] = _r((last_px / entry_px - 1.0) * 100.0, 1)


def live_books(
    cache: dict[str, Any],
    symbols: list[str] | None,
    params: FundParams,
    asof: date,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Today's held book (last rebalance) vs names ranked for the next rebalance."""
    from trading.constants import NIFTY50_SYMBOL

    today = today or date.today()
    asof = min(asof, today)
    symbols = symbols or _cache_symbols(cache)
    is_live = asof >= today
    closes: dict[str, pd.Series] = {}
    tech: dict[str, pd.DataFrame] = {}
    calendar = pd.DatetimeIndex([])
    need_px = (
        (params.tech not in ("none", "", None))
        or (params.rank_by not in ("score", "", None))
        or (params.rebalance not in ("july", "annual", "year", "", None))
        or int(params.trail_sma or 0) > 0
    )
    if need_px:
        try:
            closes = load_close_series(list(dict.fromkeys(list(symbols) + [NIFTY50_SYMBOL])))
            tech = build_close_tech(closes)
            nifty = closes.get(NIFTY50_SYMBOL)
            if nifty is not None and not nifty.empty:
                calendar = pd.DatetimeIndex(nifty.index)
        except Exception:
            closes, tech, calendar = {}, {}, pd.DatetimeIndex([])

    if params.rebalance not in ("july", "annual", "year", "", None) and len(calendar) > 0:
        past = rebalance_dates(asof - timedelta(days=365 * 3), asof, calendar, params.rebalance)
        last_reb = past[-1] if past else last_rebalance_on_or_before(asof)
        future = rebalance_dates(asof, asof + timedelta(days=90), calendar, params.rebalance)
        later = [d for d in future if d > asof]
        next_reb = later[0] if later else next_rebalance_after(asof)
    else:
        last_reb = last_rebalance_on_or_before(asof)
        next_reb = next_rebalance_after(asof)

    today_book = pick_at_date(cache, symbols, last_reb, params, closes or None, use_info=False, tech=tech)
    use_yahoo = is_live and params.rank_by in ("score", "", None) and params.tech in ("none", "", None)
    if use_yahoo:
        passed, _ = current_ranking(cache, symbols, params)
        passed = apply_tech_rank(passed, asof, params, tech)
    else:
        passed, _ = rank_at_date(cache, symbols, asof, params, closes or None, use_info=False)
        passed = apply_tech_rank(passed, asof, params, tech)
    upcoming_book = passed[: params.top_n]
    upcoming_bench = passed[params.top_n: params.top_n + 12]

    today_syms = {p["symbol"] for p in today_book}
    upcoming_syms = {p["symbol"] for p in upcoming_book}

    for p in today_book:
        staying = p["symbol"] in upcoming_syms
        p["status"] = "stay" if staying else "exit"
        p["status_label"] = "Staying" if staying else "Likely out next rebalance"
    for p in upcoming_book:
        staying = p["symbol"] in today_syms
        p["status"] = "stay" if staying else "enter"
        p["status_label"] = "Already held" if staying else "New next rebalance"
    for p in upcoming_bench:
        p["status"] = "bench"
        p["status_label"] = "On deck"

    attach_mark_to_market(today_book, asof, entry_date=last_reb)
    attach_mark_to_market(upcoming_book + upcoming_bench, asof)

    staying = [p for p in upcoming_book if p["symbol"] in today_syms]
    entering = [p for p in upcoming_book if p["symbol"] not in today_syms]
    leaving = [p for p in today_book if p["symbol"] not in upcoming_syms]

    return {
        "today_book": today_book,
        "upcoming_book": upcoming_book,
        "upcoming_bench": upcoming_bench,
        "book_diff": {
            "staying": [p["symbol"] for p in staying],
            "entering": [p["symbol"] for p in entering],
            "leaving": [p["symbol"] for p in leaving],
            "stay_count": len(staying),
            "enter_count": len(entering),
            "leave_count": len(leaving),
        },
        "last_rebalance": last_reb.isoformat(),
        "last_rebalance_label": format_short_date(last_reb),
        "next_rebalance": next_reb.isoformat(),
        "next_rebalance_label": format_short_date(next_reb),
        "asof": asof.isoformat(),
        "asof_label": format_short_date(asof),
        "asof_is_today": is_live,
        "today_title": "Today’s book" if is_live else f"Held on {format_short_date(asof)}",
        "today_sub": (
            f"Equal-weight names held since {format_short_date(last_reb)} · next rebalance {format_short_date(next_reb)}"
            if is_live
            else f"Point-in-time book from the {format_short_date(last_reb)} rebalance"
        ),
        "upcoming_title": (
            f"Upcoming · {format_short_date(next_reb)}"
            if is_live
            else f"Next in line as of {format_short_date(asof)}"
        ),
        "upcoming_sub": (
            f"Highest-ranked names now — these would enter at the next rebalance"
            if is_live
            else "Highest-ranked names on that date using statements already public then"
        ),
        "picks": today_book,
        "watch": upcoming_book + upcoming_bench,
    }


def empty_page(params: FundParams | None = None) -> dict[str, Any]:
    params = params or DEFAULT_PARAMS
    return {
        "strategy": STRATEGY_NAME,
        "needs_fetch": True,
        "picks": [],
        "watch": [],
        "today_book": [],
        "upcoming_book": [],
        "upcoming_bench": [],
        "book_diff": {"staying": [], "entering": [], "leaving": [], "stay_count": 0, "enter_count": 0, "leave_count": 0},
        "windows": [],
        "year_rows": [],
        "pack_runs": {},
        "doublers": [],
        "coverage": {"universe": 0, "cached": 0, "passed": 0, "fetched_at": None},
        "params": params_to_dict(params),
        "disclaimer": "",
        "blurb": "",
        "asof": date.today().isoformat(),
        "asof_is_today": True,
        "today_title": "Today’s book",
        "upcoming_title": "Upcoming",
        "today_sub": "",
        "upcoming_sub": "",
        "last_rebalance_label": "",
        "next_rebalance_label": "",
    }


def apply_live_books(
    payload: dict[str, Any],
    *,
    asof: date | None = None,
    params: FundParams | None = None,
) -> dict[str, Any]:
    cache = load_cache()
    if not (cache.get("stocks") or {}):
        payload.setdefault("today_book", payload.get("picks") or [])
        payload.setdefault("upcoming_book", [])
        payload.setdefault("upcoming_bench", [])
        payload.setdefault("book_diff", {"staying": [], "entering": [], "leaving": [], "stay_count": 0, "enter_count": 0, "leave_count": 0})
        payload.setdefault("today_title", "Today’s book")
        payload.setdefault("upcoming_title", "Upcoming")
        payload.setdefault("asof_is_today", True)
        return payload
    asof = asof or parse_iso_date(payload.get("asof")) or date.today()
    chosen = params or params_by_name((payload.get("params") or {}).get("name"))
    snap = live_books(cache, _cache_symbols(cache), chosen, asof)
    payload.update(snap)
    payload["params"] = params_to_dict(chosen)
    return payload


def run_custom_backtest(
    start: date,
    end: date,
    *,
    params: FundParams | None = None,
    capital: float = DEFAULT_CAPITAL,
    cache: dict[str, Any] | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    from trading.constants import NIFTY50_SYMBOL
    from trading.services.market_data import get_universe_symbols, load_price_dataframe

    params = params or DEFAULT_PARAMS
    cache = cache or load_cache()
    if end <= start:
        return {"error": "End date must be after start date.", "start": start.isoformat(), "end": end.isoformat()}
    if not (cache.get("stocks") or {}):
        return {"error": "Yahoo fundamentals are not cached yet.", "start": start.isoformat(), "end": end.isoformat()}

    cached_syms = _cache_symbols(cache)
    if symbols is None:
        try:
            universe = get_universe_symbols(nifty200_only=True)
        except Exception:
            universe = cached_syms
        cached_set = set(cached_syms)
        symbols = [s for s in universe if s in cached_set] or cached_syms

    closes = load_close_series(list(dict.fromkeys(symbols + [NIFTY50_SYMBOL])))
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    if nifty_df.empty:
        calendar = pd.DatetimeIndex([])
        nifty = pd.Series(dtype=float)
    else:
        calendar = pd.DatetimeIndex(pd.to_datetime(nifty_df.index))
        nifty = nifty_df["close"].astype(float)

    result = run_window(cache, symbols, start, end, params, closes, calendar, capital=capital)
    result["title"] = f"{format_short_date(start)} → {format_short_date(end)}"
    result["years"] = round((end - start).days / 365.25, 2)
    result["benchmark_pct"] = _r(index_return(nifty, start, end), 2)
    result["beat_benchmark"] = (
        result.get("total_return_pct") is not None
        and result.get("benchmark_pct") is not None
        and result["total_return_pct"] > result["benchmark_pct"]
    )
    result["params_name"] = params.name
    result["asof_book"] = pick_at_date(cache, symbols, start, params, closes, use_info=False)
    return result


def attach_forward_context(picks: list[dict[str, Any]], closes: dict[str, pd.Series], asof: date) -> None:
    for p in picks:
        series = closes.get(p["symbol"])
        if series is None:
            continue
        p["ret_1y_pct"] = _r(period_return(series, asof - timedelta(days=365), asof), 1)
        p["ret_3y_pct"] = _r(period_return(series, asof - timedelta(days=365 * 3), asof), 1)


def run_pack_windows(
    cache: dict[str, Any],
    symbols: list[str],
    params: FundParams,
    closes: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    nifty: pd.Series,
    end: date,
    *,
    capital: float = DEFAULT_CAPITAL,
) -> dict[str, Any]:
    windows = []
    for years in WINDOW_YEARS:
        start = end - timedelta(days=365 * years + 1)
        result = run_window(cache, symbols, start, end, params, closes, calendar, capital=capital)
        result["years"] = years
        result["title"] = f"Last {years} year" + ("s" if years != 1 else "")
        result["benchmark_pct"] = _r(index_return(nifty, start, end), 2)
        result["beat_benchmark"] = (
            result.get("total_return_pct") is not None
            and result.get("benchmark_pct") is not None
            and result["total_return_pct"] > result["benchmark_pct"]
        )
        windows.append(result)

    year_rows = []
    longest_start = end - timedelta(days=365 * 5 + 1)
    for a, b, label in yearly_slices(longest_start, end):
        chunk = run_window(cache, symbols, a, b, params, closes, calendar, capital=capital)
        year_rows.append({
            "label": label,
            "start": a.isoformat(),
            "end": b.isoformat(),
            "return_pct": chunk.get("total_return_pct"),
            "cagr_pct": chunk.get("cagr_pct"),
            "max_drawdown_pct": chunk.get("max_drawdown_pct"),
            "trades": chunk.get("trades"),
            "win_rate": chunk.get("win_rate"),
            "doublers": chunk.get("doublers"),
            "hit_100": chunk.get("hit_100"),
            "benchmark_pct": _r(index_return(nifty, a, b), 2),
            "picks": (chunk.get("rebalances") or [{}])[0].get("picks") if chunk.get("rebalances") else [],
            "best_stock_year": chunk.get("best_stock_year"),
        })
    return {"windows": windows, "years": year_rows}


def build_results(
    cache: dict[str, Any] | None = None,
    *,
    symbols: list[str] | None = None,
    params: FundParams | None = None,
    capital: float = DEFAULT_CAPITAL,
    end: date | None = None,
) -> dict[str, Any]:
    from trading.constants import NIFTY50_SYMBOL
    from trading.services.market_data import get_universe_symbols, load_price_dataframe

    params = params or DEFAULT_PARAMS
    cache = cache or load_cache()
    symbols = symbols or get_universe_symbols(nifty200_only=True)
    end = end or date.today()
    closes = load_close_series(symbols + [NIFTY50_SYMBOL])
    nifty_df = load_price_dataframe(NIFTY50_SYMBOL)
    if nifty_df.empty:
        calendar = pd.DatetimeIndex([])
        nifty = pd.Series(dtype=float)
    else:
        calendar = pd.DatetimeIndex(pd.to_datetime(nifty_df.index))
        nifty = nifty_df["close"].astype(float)

    passed, rejected = current_ranking(cache, symbols, params)
    attach_forward_context(passed, closes, end)
    portfolio = passed[: params.top_n]

    pack_runs = {}
    for pack in DEFAULT_PACKS:
        pack_runs[pack.name] = run_pack_windows(
            cache, symbols, pack, closes, calendar, nifty, end, capital=capital
        )

    primary = pack_runs[params.name]
    doublers = []
    for window in primary["windows"]:
        for t in window.get("holdings_history") or []:
            if t.get("doubled"):
                doublers.append({**t, "window": window.get("title")})

    coverage = {
        "universe": len(symbols),
        "cached": sum(1 for s in symbols if (cache.get("stocks") or {}).get(s)),
        "with_info": sum(1 for s in symbols if ((cache.get("stocks") or {}).get(s) or {}).get("info")),
        "passed": len(passed),
        "rejected": len(rejected),
        "fetched_at": cache.get("fetched_at"),
    }

    payload = {
        "strategy": STRATEGY_NAME,
        "blurb": (
            "Nifty 200, fundamentals first (ROE, growth, margins, PE), then a technical overlay. "
            "Default Momentum Swing: only names above SMA150, ranked by 3-month momentum, "
            "one name, rebalanced every 21 trading days. Older packs still use a 1 July "
            "12-month hold with no technicals."
        ),
        "disclaimer": (
            "Doubling in one year is rare. In Sep 2025–Sep 2026 only two Nifty 200 names "
            "doubled (LAURUSLABS, MCX) and Nifty 50 was down. A 3-name July hold made ~12–17%. "
            "Scanning all 200 with monthly/21-day entries, SMA/RSI/relative-strength filters "
            "and trailing stops peaked around 78–91% in-sample — not 100% without knowing the "
            "winners in advance. Max drawdown in the researched pack is around 20–30%. "
            "Past backtests are not a forecast."
        ),
        "params": params_to_dict(params),
        "packs": [params_to_dict(p) for p in DEFAULT_PACKS],
        "coverage": coverage,
        "picks": portfolio,
        "watch": passed[:25],
        "rejected_sample": rejected[:15],
        "windows": primary["windows"],
        "year_rows": primary["years"],
        "pack_runs": {
            name: {
                "windows": [
                    {k: w.get(k) for k in (
                        "title", "years", "start", "end", "total_return_pct", "cagr_pct",
                        "max_drawdown_pct", "win_rate", "trades", "doublers", "hit_100",
                        "benchmark_pct", "beat_benchmark",
                    )}
                    for w in run["windows"]
                ],
                "year_rows": run["years"],
            }
            for name, run in pack_runs.items()
        },
        "doublers": doublers[:40],
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "asof": end.isoformat(),
        "needs_fetch": coverage["cached"] == 0,
    }
    return payload


def save_results(payload: dict[str, Any]) -> Path:
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_results() -> dict[str, Any] | None:
    path = _results_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_page(
    *,
    rebuild: bool = False,
    asof: date | None = None,
    params: FundParams | None = None,
) -> dict[str, Any]:
    params = params or DEFAULT_PARAMS
    if not rebuild:
        existing = load_results()
        if existing:
            existing.setdefault("needs_fetch", False)
            return apply_live_books(existing, asof=asof, params=params)
    cache = load_cache()
    if not (cache.get("stocks") or {}):
        return empty_page(params)
    payload = build_results(cache, params=DEFAULT_PARAMS, end=date.today())
    if not _in_tests():
        save_results(payload)
    return apply_live_books(payload, asof=asof or date.today(), params=params)


def search_params(
    cache: dict[str, Any],
    symbols: list[str],
    closes: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    nifty: pd.Series,
    end: date,
    *,
    capital: float = DEFAULT_CAPITAL,
) -> list[dict[str, Any]]:
    """Small economically-sensible grid. Used by the build script, not the web request."""
    grid: list[FundParams] = []
    for top_n in (5, 8, 10):
        for min_roe in (12.0, 15.0, 18.0):
            for min_eg in (5.0, 15.0, 25.0):
                for min_rg in (5.0, 12.0):
                    for max_pe in (40.0, 55.0):
                        grid.append(FundParams(
                            name=f"n{top_n}_roe{min_roe:.0f}_eg{min_eg:.0f}_rg{min_rg:.0f}_pe{max_pe:.0f}",
                            min_roe=min_roe,
                            min_earnings_growth=min_eg,
                            min_revenue_growth=min_rg,
                            max_pe=max_pe,
                            top_n=top_n,
                            min_score=45.0,
                        ))
    scored = []
    for params in grid:
        start = end - timedelta(days=365 * 4 + 1)
        result = run_window(cache, symbols, start, end, params, closes, calendar, capital=capital)
        yearly = []
        for a, b, _label in yearly_slices(end - timedelta(days=365 * 5 + 1), end):
            chunk = run_window(cache, symbols, a, b, params, closes, calendar, capital=capital)
            yearly.append(chunk.get("total_return_pct"))
        valid_years = [y for y in yearly if y is not None]
        min_year = min(valid_years) if valid_years else None
        hit = sum(1 for y in valid_years if y >= 100)
        scored.append({
            "name": params.name,
            "params": params_to_dict(params),
            "four_year_pct": result.get("total_return_pct"),
            "cagr_pct": result.get("cagr_pct"),
            "max_drawdown_pct": result.get("max_drawdown_pct"),
            "min_year_pct": min_year,
            "years_hit_100": hit,
            "trades": result.get("trades"),
            "yearly": valid_years,
        })
    scored.sort(
        key=lambda r: (
            -(r["years_hit_100"] or 0),
            -(r["min_year_pct"] if r["min_year_pct"] is not None else -999),
            -(r["cagr_pct"] or -999),
        )
    )
    return scored
