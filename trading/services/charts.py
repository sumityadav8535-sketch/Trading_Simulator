"""Plotly chart builders for stock analysis."""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
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


def build_stage_v2_signal_bars(daily: list[dict]) -> str:
    """Bar chart: Stage 2.0 signal count per entry day, colored by avg quality."""
    if not daily:
        return json.dumps({
            "data": [],
            "layout": {
                "title": "No Stage 2 signals in selected period",
                "template": "plotly_white",
                "height": 380,
            },
        })

    def _color(avg_q: float) -> str:
        if avg_q >= 75:
            return "#059669"  # high quality
        if avg_q >= 50:
            return "#2563eb"  # average
        return "#64748b"

    colors = [_color(float(d.get("avg_quality") or 0)) for d in daily]
    fig = go.Figure(data=[go.Bar(
        x=[d["date"] for d in daily],
        y=[d["count"] for d in daily],
        marker_color=colors,
        customdata=[
            [
                d.get("avg_quality", 0),
                d.get("avg_rs", 0),
                d.get("high_quality", 0),
                d.get("symbols_preview", ""),
                d["count"],
            ]
            for d in daily
        ],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Signals: %{y}<br>"
            "Avg quality: %{customdata[0]}<br>"
            "Avg RS: %{customdata[1]}<br>"
            "High quality (≥75): %{customdata[2]}<br>"
            "%{customdata[3]}<br>"
            "<extra>Click for cards</extra>"
        ),
    )])
    fig.update_layout(
        title="Stage Analysis 2.0 signals by entry day (click a bar for stock cards)",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=20, t=55, b=80),
        xaxis_title="Entry date",
        yaxis_title="Signal count",
        bargap=0.2,
        clickmode="event+select",
    )
    return fig.to_json()


def _nearest_index(df, ts) -> int:
    indexer = df.index.get_indexer([ts], method="nearest")
    if indexer.size == 0 or indexer[0] < 0:
        return 0
    return int(indexer[0])


def build_signal_review_chart(
    symbol: str,
    signal_date: str,
    entry_date: Optional[str] = None,
    exit_date: Optional[str] = None,
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    df=None,
) -> str:
    """Candlestick around a backtest signal: mark signal, entry, stop, target, exit."""
    if df is None:
        sig_ts = pd.Timestamp(signal_date)
        start = (sig_ts - pd.Timedelta(days=90)).date()
        end = (sig_ts + pd.Timedelta(days=180)).date()
        df = load_price_dataframe(symbol, start=start, end=end)
    if df is None or getattr(df, "empty", True):
        return json.dumps({
            "data": [],
            "layout": {
                "title": f"No price data for {symbol}",
                "template": "plotly_dark",
                "height": 520,
                "paper_bgcolor": "#0f172a",
                "plot_bgcolor": "#0f172a",
            },
        })

    sig_ts = pd.Timestamp(str(signal_date)[:10])
    entry_ts = pd.Timestamp(str(entry_date)[:10]) if entry_date else None
    exit_ts = pd.Timestamp(str(exit_date)[:10]) if exit_date else None

    sig_i = _nearest_index(df, sig_ts)
    end_i = sig_i
    for ts in (entry_ts, exit_ts):
        if ts is not None:
            end_i = max(end_i, _nearest_index(df, ts))
    end_i = max(end_i, min(len(df) - 1, sig_i + 60))
    start_i = max(0, sig_i - 40)
    end_i = min(len(df) - 1, end_i + 15)
    window = df.iloc[start_i : end_i + 1]
    if window.empty:
        window = df

    actual_sig = window.index[_nearest_index(window, sig_ts)] if len(window) else sig_ts
    last_x = window.index[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
        subplot_titles=(f"{symbol} — signal {str(signal_date)[:10]}", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=window.index,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(window["close"], window["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=window.index, y=window["volume"], name="Volume",
            marker_color=vol_colors, opacity=0.55, showlegend=False,
        ),
        row=2, col=1,
    )

    # Close path after the signal so the follow-through is obvious
    after = window.loc[window.index >= actual_sig]
    if len(after) >= 2:
        fig.add_trace(
            go.Scatter(
                x=after.index, y=after["close"],
                mode="lines",
                name="After signal",
                line=dict(color="#38bdf8", width=1.5),
            ),
            row=1, col=1,
        )

    if actual_sig is not None:
        sig_px = float(window.loc[actual_sig, "close"]) if actual_sig in window.index else float(window["close"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=[actual_sig], y=[sig_px],
                mode="markers+text",
                name="Signal",
                marker=dict(symbol="star", size=16, color="#fbbf24", line=dict(color="#0f172a", width=1)),
                text=["Signal"],
                textposition="top center",
                textfont=dict(color="#fbbf24", size=11),
            ),
            row=1, col=1,
        )

    if entry_ts is not None and entry_price:
        fig.add_trace(
            go.Scatter(
                x=[entry_ts], y=[float(entry_price)],
                mode="markers+text",
                name="Entry",
                marker=dict(symbol="triangle-up", size=14, color="#22c55e"),
                text=["Entry"],
                textposition="bottom center",
                textfont=dict(color="#86efac", size=11),
            ),
            row=1, col=1,
        )
    if exit_ts is not None and exit_price:
        win = (float(entry_price) if entry_price else 0) > 0 and float(exit_price) >= float(entry_price or exit_price)
        fig.add_trace(
            go.Scatter(
                x=[exit_ts], y=[float(exit_price)],
                mode="markers+text",
                name="Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color="#22c55e" if win else "#ef4444",
                ),
                text=["Exit"],
                textposition="top center",
                textfont=dict(color="#fda4af" if not win else "#86efac", size=11),
            ),
            row=1, col=1,
        )

    shapes = [
        dict(
            type="rect",
            xref="x", yref="paper",
            x0=actual_sig, x1=last_x,
            y0=0, y1=1,
            fillcolor="rgba(56,189,248,0.06)",
            line=dict(width=0),
            layer="below",
        ),
        dict(
            type="line",
            xref="x", yref="paper",
            x0=actual_sig, x1=actual_sig,
            y0=0, y1=1,
            line=dict(color="#fbbf24", width=2, dash="dot"),
        ),
    ]
    annotations = [
        dict(
            x=actual_sig, y=1.02, xref="x", yref="paper",
            text="Signal day", showarrow=False,
            font=dict(color="#fbbf24", size=11),
        ),
    ]
    if stop_loss:
        fig.add_hline(
            y=float(stop_loss), line_dash="dash", line_color="#ef4444",
            annotation_text=f"Stop {float(stop_loss):.2f}",
            annotation_position="bottom right",
            annotation_font_color="#fca5a5",
            row=1, col=1,
        )
    if target:
        fig.add_hline(
            y=float(target), line_dash="dash", line_color="#38bdf8",
            annotation_text=f"Target {float(target):.2f}",
            annotation_position="top right",
            annotation_font_color="#7dd3fc",
            row=1, col=1,
        )
    if entry_price:
        fig.add_hline(
            y=float(entry_price), line_dash="dot", line_color="#22c55e",
            annotation_text=f"Entry {float(entry_price):.2f}",
            annotation_position="top left",
            annotation_font_color="#86efac",
            row=1, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=540,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, font=dict(size=11)),
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1e293b")
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b")
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