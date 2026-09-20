from django import forms

from stage_analysis_v2.services.backtester import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_ENTRY_ON,
    DEFAULT_ENTRY_STAGE,
    DEFAULT_EXIT_MODE,
    DEFAULT_MA_PERIOD,
    DEFAULT_MA_TYPE,
    DEFAULT_MAX_HOLD_DAYS,
    DEFAULT_RISK_PCT,
    DEFAULT_STOP_MA_MULT,
    DEFAULT_TARGET_RR,
    DEFAULT_TRAIL_MA_MULT,
    ENTRY_ON_CHOICES,
    ENTRY_STAGE_CHOICES,
    EXIT_MODE_CHOICES,
    MA_CONDITION_CHOICES,
    MA_COND_NONE,
)
from stage_analysis_v2.services.cup_breakout import (
    CUP_ENTRY_CHOICES,
    CUP_ENTRY_NEXT_OPEN,
    CUP_EXIT_CHOICES,
    CUP_EXIT_EMA20,
)
from stage_analysis_v2.services.strategy_catalog import (
    BACKTEST_STRATEGY_CHOICES,
    DEFAULT_STAGE_MAX_POS_PCT,
    DEFAULT_STRATEGY,
)
from stage_analysis_v2.services.tech_filters import DEFAULT_TECH_FILTER, TECH_FILTER_CHOICES
from stage_analysis_v2.services.top_picks import get_sector_list


class AnalyzeForm(forms.Form):
    ticker = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            "placeholder": "RELIANCE.NS or AAPL",
            "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2 w-48",
        }),
    )


class WatchlistForm(forms.Form):
    ticker = forms.CharField(max_length=20, widget=forms.TextInput(attrs={
        "placeholder": "TCS.NS",
        "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2",
    }))
    notes = forms.CharField(required=False, max_length=255, widget=forms.TextInput(attrs={
        "placeholder": "Notes",
        "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2",
    }))


class ScreenerForm(forms.Form):
    min_quality_score = forms.IntegerField(
        initial=75, min_value=0, max_value=100,
        widget=forms.NumberInput(attrs={"class": "bg-slate-800 border border-slate-700 rounded px-3 py-2 w-20"}),
    )
    min_rs_rating = forms.FloatField(
        initial=60, min_value=0, max_value=100,
        widget=forms.NumberInput(attrs={"class": "bg-slate-800 border border-slate-700 rounded px-3 py-2 w-20"}),
    )
    daily_stage = forms.ChoiceField(
        choices=[("", "Any"), ("2", "Stage 2"), ("1", "Stage 1")],
        required=False,
        widget=forms.Select(attrs={"class": "bg-slate-800 border border-slate-700 rounded px-3 py-2"}),
    )
    breakout_type = forms.ChoiceField(
        choices=[("", "Any"), ("clean", "Clean"), ("weak", "Weak")],
        required=False,
        initial="clean",
        widget=forms.Select(attrs={"class": "bg-slate-800 border border-slate-700 rounded px-3 py-2"}),
    )
    sector = forms.ChoiceField(
        required=False,
        widget=forms.Select(attrs={"class": "bg-slate-800 border border-slate-700 rounded px-3 py-2"}),
    )
    market_favorable_only = forms.BooleanField(required=False, initial=False)
    highlight_only = forms.BooleanField(required=False, initial=False)
    strong_rs_only = forms.BooleanField(required=False, initial=True, label="Strong RS only")
    run_scan = forms.BooleanField(required=False, initial=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sectors = [("", "All Sectors")] + [(s, s) for s in get_sector_list()]
        self.fields["sector"].choices = sectors


class StageV2BacktestForm(forms.Form):
    UNIVERSE_CHOICES = [
        ("single", "Single stock"),
        ("custom", "Custom list"),
        ("nifty200", "Nifty 200"),
        ("nifty_smallcap250", "Nifty Smallcap 250"),
    ]

    strategy = forms.ChoiceField(
        choices=BACKTEST_STRATEGY_CHOICES,
        initial=DEFAULT_STRATEGY,
        label="Strategy",
        help_text="Which engine to backtest. Supertrend / cup use their own researched rules.",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-emerald-600 rounded px-3 py-2",
            "id": "strategy-select",
        }),
    )
    universe = forms.ChoiceField(choices=UNIVERSE_CHOICES, initial="nifty200")
    symbol = forms.CharField(
        max_length=20,
        initial="RELIANCE",
        required=False,
        widget=forms.TextInput(attrs={
            "placeholder": "RELIANCE or RELIANCE.NS",
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    symbols = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            "rows": 2,
            "placeholder": "TCS, INFY, HDFCBANK",
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    start_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date", "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2"}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date", "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2"}))
    capital = forms.DecimalField(
        initial=1_000_000,
        min_value=1000,
        decimal_places=2,
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2"}),
    )
    min_quality_score = forms.IntegerField(
        initial=0,
        min_value=0,
        max_value=100,
        label="Min quality score",
        help_text="Only include Stage 2 signals with quality ≥ this value (0 = any, try 75/85/90).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 w-24",
            "min": 0,
            "max": 100,
            "step": 1,
        }),
    )
    min_rs_rating = forms.FloatField(
        initial=0,
        min_value=0,
        max_value=100,
        required=False,
        label="Min RS rating",
        help_text="Only include Stage 2 signals with RS ≥ this vs Nifty (0 = any, researched pack uses 70).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 w-24",
            "min": 0,
            "max": 100,
            "step": 1,
            "id": "min_rs_rating",
        }),
    )
    market_filter = forms.BooleanField(
        required=False,
        initial=False,
        label="Nifty Stage 1/2 only",
    )
    shared_capital = forms.ChoiceField(
        choices=[("1", "Shared capital"), ("0", "Take every signal")],
        required=False,
        initial="1",
        label="Capital model",
        help_text="Charts follow this choice. Both books are always computed and shown.",
    )
    exit_mode = forms.ChoiceField(
        choices=EXIT_MODE_CHOICES,
        initial=DEFAULT_EXIT_MODE,
        label="Exit mode (stage rule)",
        help_text="How to leave when weekly stage deteriorates (plus stop/target/time).",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    target_rr = forms.FloatField(
        initial=DEFAULT_TARGET_RR,
        min_value=0.5,
        max_value=10,
        label="Target R:R",
        help_text="Reward multiple of risk (default 2.5). Target = entry + R×(entry−stop).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.1",
            "min": "0.5",
            "max": "10",
        }),
    )
    max_hold_days = forms.IntegerField(
        initial=DEFAULT_MAX_HOLD_DAYS,
        min_value=1,
        max_value=500,
        label="Max hold (days)",
        help_text="Time stop — exit at close after this many trading days (default 65).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "min": "1",
            "max": "500",
        }),
    )
    stop_ma_mult = forms.FloatField(
        initial=DEFAULT_STOP_MA_MULT,
        min_value=0.5,
        max_value=0.999,
        label="Stop = 30w MA ×",
        help_text="Stop under 30-week MA (default 0.95 = 5% below MA).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.01",
            "min": "0.5",
            "max": "0.999",
        }),
    )
    trail_ma_mult = forms.FloatField(
        initial=DEFAULT_TRAIL_MA_MULT,
        min_value=0.5,
        max_value=0.999,
        label="Trail stop = 30w MA ×",
        help_text="Only used when exit mode is Trail MA + Stage 4 (default 0.98).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.01",
            "min": "0.5",
            "max": "0.999",
        }),
    )
    risk_pct = forms.FloatField(
        initial=DEFAULT_RISK_PCT,
        min_value=0.1,
        max_value=50,
        label="Risk % per trade",
        help_text="Percent of equity risked per trade (default 2). Aggressive 300% pack uses ~5.",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.1",
            "min": "0.1",
            "max": "50",
        }),
    )
    cooldown_days = forms.IntegerField(
        initial=DEFAULT_COOLDOWN_DAYS,
        min_value=0,
        max_value=365,
        label="Cooldown cooldown (days)",
        help_text="Days after exit before re-buying same symbol (default 40). Aggressive pack uses 0–5.",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "min": "0",
            "max": "365",
        }),
    )
    tech_filter = forms.ChoiceField(
        choices=TECH_FILTER_CHOICES,
        initial=DEFAULT_TECH_FILTER,
        label="Tech filter",
        help_text="Extra technical confirmation on Stage 2 entries (EMA / BB / daily stage).",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    entry_stage = forms.TypedChoiceField(
        choices=ENTRY_STAGE_CHOICES,
        coerce=int,
        initial=DEFAULT_ENTRY_STAGE,
        label="Entry stage",
        help_text="Weekly Weinstein stage that generates an entry signal.",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    entry_on = forms.ChoiceField(
        choices=ENTRY_ON_CHOICES,
        initial=DEFAULT_ENTRY_ON,
        label="Entry timing",
        help_text="Transition = only when stage newly becomes the selected stage.",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    # Stock filters (0 / empty = off)
    min_price = forms.FloatField(
        initial=0,
        min_value=0,
        required=False,
        label="Min stock price ₹",
        help_text="0 = no minimum",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.01",
            "min": "0",
        }),
    )
    max_price = forms.FloatField(
        initial=0,
        min_value=0,
        required=False,
        label="Max stock price ₹",
        help_text="0 = no maximum",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.01",
            "min": "0",
        }),
    )
    min_volume = forms.FloatField(
        initial=0,
        min_value=0,
        required=False,
        label="Min volume (shares)",
        help_text="Absolute volume on signal day; 0 = off",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "1",
            "min": "0",
        }),
    )
    min_volume_ratio = forms.FloatField(
        initial=0,
        min_value=0,
        required=False,
        label="Min volume ratio (× 20d avg)",
        help_text="e.g. 1.5 = 1.5× average volume; 0 = off",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.1",
            "min": "0",
        }),
    )
    ma_condition = forms.ChoiceField(
        choices=MA_CONDITION_CHOICES,
        initial=MA_COND_NONE,
        label="Price vs MA",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    ma_period = forms.IntegerField(
        initial=DEFAULT_MA_PERIOD,
        min_value=0,
        max_value=500,
        required=False,
        label="MA period",
        help_text="Used when Price vs MA is not “No MA filter”",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "min": "0",
            "max": "500",
        }),
    )
    ma_type = forms.ChoiceField(
        choices=[("sma", "SMA"), ("ema", "EMA")],
        initial=DEFAULT_MA_TYPE,
        label="MA type",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    quality_overlay = forms.BooleanField(
        required=False,
        initial=False,
        label="Quality overlay",
        help_text="Nifty > SMA150, stock ≤15% above SMA150, PE ≤ 50, profit margin ≥ 8%. PE/margin use latest snapshot.",
    )
    max_pct_above_ma = forms.FloatField(
        initial=0,
        min_value=0,
        max_value=100,
        required=False,
        label="Max % above MA",
        help_text="0 = off. 15 = skip if close is more than 15% above the MA above.",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "1",
            "min": "0",
            "max": "100",
        }),
    )
    nifty_sma_period = forms.IntegerField(
        initial=0,
        min_value=0,
        max_value=400,
        required=False,
        label="Nifty above SMA",
        help_text="0 = off. 150 = skip new buys when Nifty close ≤ SMA150.",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "min": "0",
            "max": "400",
        }),
    )
    min_profit_margin = forms.FloatField(
        initial=0,
        min_value=0,
        max_value=100,
        required=False,
        label="Min profit margin %",
        help_text="0 = off. 8 = skip names with net margin below 8% (latest snapshot).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "0.5",
            "min": "0",
            "max": "100",
        }),
    )
    max_pe = forms.FloatField(
        initial=0,
        min_value=0,
        max_value=500,
        required=False,
        label="Max PE",
        help_text="0 = off. 50 = skip trailing PE above 50 (latest snapshot).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "1",
            "min": "0",
            "max": "500",
        }),
    )
    max_pos_pct = forms.FloatField(
        initial=DEFAULT_STAGE_MAX_POS_PCT,
        min_value=5,
        max_value=100,
        required=False,
        label="Max position % of equity",
        help_text="Cap notional per name. Supertrend/cup default 50. Stage 2.0 uses 100 (cash only).",
        widget=forms.NumberInput(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
            "step": "5",
            "min": "5",
            "max": "100",
        }),
    )
    # Cup-and-handle parameters (ignored by other engines)
    entry_mode = forms.ChoiceField(
        choices=CUP_ENTRY_CHOICES,
        initial=CUP_ENTRY_NEXT_OPEN,
        required=False,
        label="Cup entry",
        help_text="Default: buy next open after the breakout close. Retest waits for a hold above the rim.",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    cup_exit_mode = forms.ChoiceField(
        choices=CUP_EXIT_CHOICES,
        initial=CUP_EXIT_EMA20,
        required=False,
        label="Cup exit",
        help_text="2R/3R uses Target R:R. Measured move = cup high + cup depth. Always time-stops.",
        widget=forms.Select(attrs={
            "class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    cup_min_days = forms.IntegerField(
        initial=20, min_value=10, max_value=400, required=False,
        label="Cup min days",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "10", "max": "400"}),
    )
    cup_max_days = forms.IntegerField(
        initial=180, min_value=20, max_value=500, required=False,
        label="Cup max days",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "20", "max": "500"}),
    )
    min_depth_pct = forms.FloatField(
        initial=12, min_value=5, max_value=50, required=False,
        label="Min cup depth %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "1", "min": "5", "max": "50"}),
    )
    max_depth_pct = forms.FloatField(
        initial=45, min_value=10, max_value=70, required=False,
        label="Max cup depth %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "1", "min": "10", "max": "70"}),
    )
    recovery_pct = forms.FloatField(
        initial=90, min_value=70, max_value=100, required=False,
        label="Right-rim recovery %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "1", "min": "70", "max": "100"}),
    )
    handle_min_days = forms.IntegerField(
        initial=5, min_value=0, max_value=40, required=False,
        label="Handle min days",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "0", "max": "40"}),
    )
    handle_max_days = forms.IntegerField(
        initial=30, min_value=1, max_value=80, required=False,
        label="Handle max days",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "1", "max": "80"}),
    )
    handle_max_depth_pct = forms.FloatField(
        initial=15, min_value=3, max_value=30, required=False,
        label="Handle max depth %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.5", "min": "3", "max": "30"}),
    )
    require_handle = forms.BooleanField(required=False, initial=False, label="Require handle")
    breakout_buffer_pct = forms.FloatField(
        initial=0.5, min_value=0, max_value=5, required=False,
        label="Breakout buffer %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.1", "min": "0", "max": "5"}),
    )
    vol_mult = forms.FloatField(
        initial=1.1, min_value=0, max_value=5, required=False,
        label="Breakout volume × 20d",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.1", "min": "0", "max": "5"}),
    )
    rsi_min = forms.FloatField(
        initial=40, min_value=0, max_value=90, required=False,
        label="RSI min",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "1", "min": "0", "max": "90"}),
    )
    rsi_max = forms.FloatField(
        initial=85, min_value=10, max_value=100, required=False,
        label="RSI max",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "1", "min": "10", "max": "100"}),
    )
    max_gap_pct = forms.FloatField(
        initial=5, min_value=0, max_value=20, required=False,
        label="Max opening gap %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.5", "min": "0", "max": "20"}),
    )
    min_close_loc = forms.FloatField(
        initial=0.70, min_value=0, max_value=1, required=False,
        label="Min close location",
        help_text="0.70 = close in the upper 30% of the daily range.",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.05", "min": "0", "max": "1"}),
    )
    require_close_strength = forms.BooleanField(
        required=False, initial=False, label="Require upper-30% close",
    )
    require_sma200_rising = forms.BooleanField(
        required=False, initial=False, label="Require 200 SMA rising",
    )
    require_rs_vs_nifty = forms.BooleanField(
        required=False, initial=False, label="Stock 60d return > Nifty 50",
    )
    require_trend_stack = forms.BooleanField(
        required=False, initial=True, label="Require EMA20 / SMA50 / SMA200 stack",
    )
    require_nifty_sma200 = forms.BooleanField(
        required=False, initial=False, label="Nifty 50 > 200 SMA",
    )
    stop_atr_mult = forms.FloatField(
        initial=0.5, min_value=0, max_value=3, required=False,
        label="Stop ATR multiple",
        help_text="Stop = handle/pullback low − this × ATR(14).",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.1", "min": "0", "max": "3"}),
    )
    retest_tol_pct = forms.FloatField(
        initial=2, min_value=0.2, max_value=8, required=False,
        label="Retest tolerance %",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "step": "0.1", "min": "0.2", "max": "8"}),
    )
    retest_max_days = forms.IntegerField(
        initial=10, min_value=2, max_value=40, required=False,
        label="Retest window (days)",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "2", "max": "40"}),
    )
    min_bottom_days = forms.IntegerField(
        initial=5, min_value=2, max_value=40, required=False,
        label="Min days near cup low",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "2", "max": "40"}),
    )
    max_new_per_day = forms.IntegerField(
        initial=10, min_value=1, max_value=30, required=False,
        label="Max new positions / day",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "1", "max": "30"}),
    )
    nifty_ema_period = forms.IntegerField(
        initial=20, min_value=0, max_value=200, required=False,
        label="Nifty must be above EMA",
        help_text="0 = off. 20 = skip new buys when Nifty close ≤ EMA20 (signal day).",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "0", "max": "200"}),
    )
    loss_streak = forms.IntegerField(
        initial=3, min_value=0, max_value=10, required=False,
        label="Pause after N losses",
        help_text="0 = off. Halt new entries after this many consecutive losses.",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "0", "max": "10"}),
    )
    loss_streak_cooloff_days = forms.IntegerField(
        initial=10, min_value=0, max_value=60, required=False,
        label="Loss-streak pause (days)",
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2", "min": "0", "max": "60"}),
    )

    def clean(self):
        cleaned = super().clean()
        start = cleaned.get("start_date")
        end = cleaned.get("end_date")
        if start and end and end < start:
            raise forms.ValidationError("End date must be on or after start date.")
        universe = cleaned.get("universe")
        if universe == "single" and not (cleaned.get("symbol") or "").strip():
            self.add_error("symbol", "Enter a symbol for single-stock backtest.")
        if universe == "custom" and not (cleaned.get("symbols") or "").strip():
            self.add_error("symbols", "Enter at least one symbol for custom list.")
        min_p = cleaned.get("min_price") or 0
        max_p = cleaned.get("max_price") or 0
        if min_p and max_p and max_p < min_p:
            self.add_error("max_price", "Max price must be ≥ min price.")
        ma_cond = cleaned.get("ma_condition") or MA_COND_NONE
        ma_period = cleaned.get("ma_period") or 0
        if ma_cond != MA_COND_NONE and ma_period <= 0:
            cleaned["ma_period"] = DEFAULT_MA_PERIOD
        if cleaned.get("quality_overlay"):
            if not cleaned.get("max_pct_above_ma"):
                cleaned["max_pct_above_ma"] = 15
            if not cleaned.get("nifty_sma_period"):
                cleaned["nifty_sma_period"] = 150
            if not cleaned.get("min_profit_margin"):
                cleaned["min_profit_margin"] = 8
            if not cleaned.get("max_pe"):
                cleaned["max_pe"] = 50
            if ma_cond == MA_COND_NONE:
                cleaned["ma_condition"] = "above"
                if not cleaned.get("ma_period"):
                    cleaned["ma_period"] = 150
        cup_min = cleaned.get("cup_min_days") or 30
        cup_max = cleaned.get("cup_max_days") or 150
        if cup_max < cup_min:
            self.add_error("cup_max_days", "Cup max days must be ≥ cup min days.")
        dmin = cleaned.get("min_depth_pct") or 15
        dmax = cleaned.get("max_depth_pct") or 40
        if dmax < dmin:
            self.add_error("max_depth_pct", "Max cup depth must be ≥ min cup depth.")
        hmin = cleaned.get("handle_min_days") or 5
        hmax = cleaned.get("handle_max_days") or 30
        if hmax < hmin:
            self.add_error("handle_max_days", "Handle max days must be ≥ handle min days.")
        rmin = cleaned.get("rsi_min") or 50
        rmax = cleaned.get("rsi_max") or 80
        if rmax < rmin:
            self.add_error("rsi_max", "RSI max must be ≥ RSI min.")
        raw_shared = cleaned.get("shared_capital")
        if raw_shared in (None, ""):
            cleaned["shared_capital"] = True
        else:
            cleaned["shared_capital"] = str(raw_shared).lower() in ("1", "on", "true", "yes")
        return cleaned