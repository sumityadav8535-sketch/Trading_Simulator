from django import forms

from trading.models import Stock, TradeJournalEntry


class ScreenerForm(forms.Form):
    nifty200_only = forms.BooleanField(initial=True, required=False)
    enable_fundamentals = forms.BooleanField(initial=False, required=False, label="Apply fundamental filters")
    min_adx = forms.FloatField(initial=20.0, min_value=0)
    above_200ema = forms.BooleanField(initial=True, required=False)
    min_confluence = forms.IntegerField(initial=5, min_value=0, max_value=10)


class ScannerForm(forms.Form):
    SCOPE_CHOICES = [
        ("nifty200", "Nifty 200"),
        ("nifty_smallcap250", "Nifty Smallcap 250"),
        ("watchlist", "Watchlist only"),
    ]
    scope = forms.ChoiceField(choices=SCOPE_CHOICES, initial="nifty200")
    min_score = forms.IntegerField(initial=7, min_value=0, max_value=10)
    capital = forms.DecimalField(initial=100_000, min_value=1000, decimal_places=2)


class BacktestForm(forms.Form):
    symbol = forms.CharField(max_length=20, initial="RELIANCE")
    symbols = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="Optional comma-separated list for multi-stock backtest",
    )
    start_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    capital = forms.DecimalField(initial=100_000, min_value=1000, decimal_places=2)


class RiskCalculatorForm(forms.Form):
    capital = forms.DecimalField(initial=100_000, min_value=1000, decimal_places=2)
    risk_pct = forms.FloatField(initial=2.0, min_value=0.1, max_value=5.0)
    entry_price = forms.FloatField(initial=1000.0, min_value=0.01)
    stop_loss = forms.FloatField(initial=980.0, min_value=0.01)


class SignalsForm(forms.Form):
    start_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    capital = forms.DecimalField(initial=100_000, min_value=1000, decimal_places=2)


class ChartForm(forms.Form):
    symbol = forms.ChoiceField(required=True)


class FnoForm(forms.Form):
    INSTRUMENT_CHOICES = [
        ("NIFTY", "Nifty 50 Futures"),
        ("BANKNIFTY", "Bank Nifty Futures"),
    ]
    instrument = forms.ChoiceField(choices=INSTRUMENT_CHOICES, initial="NIFTY")


class IntradayForm(forms.Form):
    INTERVAL_CHOICES = [
        ("1m", "1 minute"),
        ("5m", "5 minutes"),
        ("15m", "15 minutes"),
    ]
    SORT_CHOICES = [
        ("change_pct", "Change %"),
        ("symbol", "Symbol"),
        ("ltp", "LTP"),
        ("volume", "Volume"),
    ]

    interval = forms.ChoiceField(choices=INTERVAL_CHOICES, initial="5m")
    sort = forms.ChoiceField(choices=SORT_CHOICES, initial="change_pct")
    symbol = forms.CharField(required=False, max_length=20)


class WatchlistForm(forms.Form):
    symbol = forms.CharField(max_length=20)
    notes = forms.CharField(required=False, max_length=255)


class JournalForm(forms.ModelForm):
    class Meta:
        model = TradeJournalEntry
        fields = [
            "stock", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "stop_loss", "target", "status", "notes",
        ]
        widgets = {
            "entry_date": forms.DateInput(attrs={"type": "date"}),
            "exit_date": forms.DateInput(attrs={"type": "date"}),
            "notes": forms.Textarea(attrs={"rows": 3}),
        }