"""Plotly chart builders for stock analysis."""
from __future__ import annotations

import json
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.services.indicators import compute_indicators
from trading.services.market_data import load_price_dataframe
from trading.services.strategy import evaluate_stock


def build_stock_chart(symbol: str, signal_date: Optional[str] = None) -> str:
    """Return Plotly figure JSON for price + indicators."""
    df = load_price_dataframe(symbol)
    if df.empty:
        return json.dumps({"data": [], "layout": {"title": f"No data for {symbol}"}})

    df = compute_indicators(df)
    eval_result = evaluate_stock(symbol)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=(f"{symbol} — Daily", "Volume", "RSI / ADX"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
        ),
        row=1, col=1,
    )
    for col, name, color in [
        ("ema_20", "EMA 20", "#f59e0b"),
        ("ema_50", "EMA 50", "#3b82f6"),
        ("ema_200", "EMA 200", "#8b5cf6"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name, line=dict(width=1, color=color)),
                row=1, col=1,
            )

    if eval_result.is_valid and eval_result.entry_price:
        fig.add_hline(y=eval_result.entry_price, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_hline(y=eval_result.stop_loss, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=eval_result.target_2r, line_dash="dash", line_color="blue", row=1, col=1)

    colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.6),
        row=2, col=1,
    )
    if "vol_sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["vol_sma_20"], name="Vol SMA 20", line=dict(color="#94a3b8")),
            row=2, col=1,
        )

    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14", line=dict(color="#06b6d4")),
            row=3, col=1,
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=40, line_dash="dot", line_color="green", row=3, col=1)
        fig.add_hline(y=65, line_dash="dot", line_color="red", row=3, col=1)

    if "adx_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["adx_14"], name="ADX 14", line=dict(color="#f97316")),
            row=3, col=1,
        )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Osc", row=3, col=1)

    return fig.to_json()


def build_equity_curve(curve: list[dict]) -> str:
    if not curve:
        return json.dumps({"data": [], "layout": {}})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["date"] for p in curve],
        y=[p["equity"] for p in curve],
        mode="lines",
        name="Equity",
        line=dict(color="#22c55e", width=2),
    ))
    fig.update_layout(
        title="Equity Curve",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig.to_json()


def build_win_loss_pie(wins: int, losses: int) -> str:
    fig = go.Figure(data=[go.Pie(
        labels=["Wins", "Losses"],
        values=[wins, losses],
        marker_colors=["#22c55e", "#ef4444"],
        hole=0.45,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="Win / Loss Distribution",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig.to_json()


def build_exit_breakdown_chart(breakdown: dict) -> str:
    if not breakdown:
        return json.dumps({"data": [], "layout": {}})
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = {
        "target_2r": "#22c55e",
        "target_2.5r": "#22c55e",
        "stop_loss": "#ef4444",
        "time_exit": "#f59e0b",
        "trail_20ema": "#3b82f6",
        "stage_exit": "#a855f7",
    }
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=[colors.get(l, "#94a3b8") for l in labels],
    )])
    fig.update_layout(
        title="Exit Reasons",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_json()


def build_monthly_returns_chart(monthly: list[dict]) -> str:
    if not monthly:
        return json.dumps({"data": [], "layout": {}})
    colors = ["#22c55e" if m["pnl"] >= 0 else "#ef4444" for m in monthly]
    fig = go.Figure(data=[go.Bar(
        x=[m["month"] for m in monthly],
        y=[m["pnl"] for m in monthly],
        marker_color=colors,
        name="Monthly PnL",
    )])
    fig.update_layout(
        title="Monthly P&L (₹)",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_json()


def build_scanner_signal_bars(daily: list[dict]) -> str:
    """Bar chart: signal count per day, colored by net P&L."""
    if not daily:
        return json.dumps({
            "data": [],
            "layout": {
                "title": "No signals in selected period",
                "template": "plotly_white",
                "height": 360,
            },
        })

    colors = [
        "#22c55e" if d["pnl"] > 0 else ("#ef4444" if d["pnl"] < 0 else "#64748b")
        for d in daily
    ]
    fig = go.Figure(data=[go.Bar(
        x=[d["date"] for d in daily],
        y=[d["count"] for d in daily],
        marker_color=colors,
        customdata=[
            [d["pnl"], d["wins"], d["losses"], d["count"]] for d in daily
        ],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Signals: %{y}<br>"
            "Net P&L: ₹%{customdata[0]:,.0f}<br>"
            "W/L: %{customdata[1]}/%{customdata[2]}<extra></extra>"
        ),
    )])
    fig.update_layout(
        title="EMA20 Elite signals by date (click a bar for trade details)",
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=20, t=50, b=80),
        xaxis_title="Signal date",
        yaxis_title="Signal count",
        bargap=0.15,
    )
    return fig.to_json()


def build_drawdown_chart(equity_curve: list[dict]) -> str:
    if not equity_curve:
        return json.dumps({"data": [], "layout": {}})
    peak = equity_curve[0]["equity"]
    dd = []
    for p in equity_curve:
        peak = max(peak, p["equity"])
        dd.append({"date": p["date"], "dd": round((peak - p["equity"]) / peak * 100, 2) if peak else 0})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in dd],
        y=[d["dd"] for d in dd],
        fill="tozeroy",
        name="Drawdown %",
        line=dict(color="#ef4444"),
    ))
    fig.update_layout(
        title="Drawdown",
        template="plotly_white",
        height=280,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Drawdown %",
    )
    return fig.to_json()