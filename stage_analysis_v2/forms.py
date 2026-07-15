from django import forms

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
    ]

    universe = forms.ChoiceField(choices=UNIVERSE_CHOICES, initial="single")
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
        widget=forms.NumberInput(attrs={"class": "block w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 w-24"}),
    )
    market_filter = forms.BooleanField(
        required=False,
        initial=False,
        label="Nifty Stage 1/2 only",
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
        return cleaned