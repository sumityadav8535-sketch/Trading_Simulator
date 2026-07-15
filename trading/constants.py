"""
Default strategy thresholds — all configurable via StrategyConfig model.
Prompt ranges interpreted as sensible defaults (e.g. 8-10% → 8%).
"""

# Fundamental defaults (optional filters in v1)
DEFAULT_SALES_GROWTH_MIN = 8.0
DEFAULT_PROFIT_GROWTH_MIN = 10.0
DEFAULT_ROE_MIN = 12.0
DEFAULT_DEBT_EQUITY_MAX = 1.0
DEFAULT_PEG_MAX = 2.0

# Technical defaults
DEFAULT_ADX_MIN = 20.0
DEFAULT_ADX_PREFERRED = 25.0
DEFAULT_RSI_LOW = 40.0
DEFAULT_RSI_HIGH = 65.0
DEFAULT_VOLUME_MULTIPLIER = 1.5
DEFAULT_EMA_PULLBACK_TOLERANCE_PCT = 2.0
DEFAULT_FIB_LOW = 0.382
DEFAULT_FIB_HIGH = 0.618

# Risk defaults
DEFAULT_RISK_PCT = 2.0
DEFAULT_MIN_RR = 2.0
DEFAULT_ATR_SL_BUFFER = 0.5

# Index symbols for market bias
NIFTY50_SYMBOL = "NIFTY50"
NIFTY100_INDEX_TICKER = "^CNX100"
NIFTY200_PROXY = "^CNX200"  # TODO: load actual index OHLCV

NSE_NIFTY100_CSV_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
)

MIN_BARS_FOR_STRATEGY = 220