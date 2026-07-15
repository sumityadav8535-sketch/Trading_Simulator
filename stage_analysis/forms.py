from django import forms


class AnalyzeTickerForm(forms.Form):
    ticker = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            "placeholder": "AAPL or RELIANCE.NS",
            "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2 w-48",
        }),
    )


class WatchlistForm(forms.Form):
    ticker = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            "placeholder": "TCS.NS",
            "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )
    notes = forms.CharField(
        required=False,
        max_length=255,
        widget=forms.TextInput(attrs={
            "placeholder": "Notes (optional)",
            "class": "bg-slate-800 border border-slate-700 rounded px-3 py-2",
        }),
    )