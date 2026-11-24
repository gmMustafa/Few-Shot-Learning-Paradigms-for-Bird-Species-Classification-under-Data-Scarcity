"""
Reusable Plotly chart functions for the FSL thesis dashboard.
All functions return a Plotly Figure — pages control layout and display.
All colours are explicitly set so charts are readable on both light and dark hosts.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette ─────────────────────────────────────────────────────────────
FAU_BLUE   = "#003865"
FAU_ORANGE = "#FF6B35"
FAU_GRAY   = "#6B7280"
FAU_LIGHT  = "#EFF6FF"
PALETTE    = [FAU_BLUE, FAU_ORANGE, "#0a9396", "#ae2012", "#94d2bd",
              "#e9d8a6", "#9b2226", "#005f73", "#ee9b00", "#ca6702"]

BACKBONE_COLOR = {"ResNet-50": FAU_BLUE, "DeiT-S": FAU_ORANGE}

# ── Explicit colours — never inherit from theme ───────────────────────────────
_TEXT   = "#1e293b"   # dark slate — readable on white
_BG     = "#FFFFFF"   # plot area
_PAPER  = "#FAFBFC"   # figure background
_GRID   = "#E2E8F0"   # grid lines
_LINE   = "#CBD5E1"   # axis lines
_FONT   = dict(family="system-ui, -apple-system, sans-serif", size=13, color=_TEXT)

# Axis template — applied via update_xaxes / update_yaxes, NOT in _BASE_LAYOUT
_AXIS_STYLE = dict(
    title_font=dict(size=12, color=_TEXT),
    tickfont=dict(size=11, color=_TEXT),
    linecolor=_LINE,
    gridcolor=_GRID,
    zerolinecolor=_LINE,
)

# _BASE_LAYOUT must NOT contain 'xaxis' or 'yaxis' keys — pages also pass those
# in update_layout() and Python raises "multiple values for keyword argument"
_BASE_LAYOUT = dict(
    font=_FONT,
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=_LINE,
        borderwidth=1,
        font=dict(color=_TEXT, size=12),
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color=_TEXT,
        bordercolor=_LINE,
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color=_TEXT),
        title_font=dict(color=_TEXT),
    ),
)


def _force_axis_colours(fig: go.Figure) -> go.Figure:
    """Apply explicit dark text to every axis in the figure (all subplots)."""
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


def _apply_base(fig: go.Figure, title: str = None,
                xlab: str = None, ylab: str = None) -> go.Figure:
    updates = dict(**_BASE_LAYOUT)
    if title:
        updates["title"] = dict(text=title, font=dict(size=15, color=FAU_BLUE), x=0.02)
    fig.update_layout(**updates)
    _force_axis_colours(fig)
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


# ── 1. Grouped / stacked bar chart ────────────────────────────────────────────
def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    error_y: str = None,
    title: str = None,
    xlab: str = None,
    ylab: str = "Accuracy (%)",
    color_map: dict = None,
    barmode: str = "group",
    text_auto: bool = False,
    category_orders: dict = None,
    **kwargs,
) -> go.Figure:
    fig = px.bar(
        df, x=x, y=y, color=color,
        error_y=error_y,
        barmode=barmode,
        color_discrete_map=color_map or {},
        color_discrete_sequence=PALETTE,
        category_orders=category_orders or {},
        text_auto=text_auto,
        **kwargs,
    )
    fig.update_traces(
        error_y_color=FAU_GRAY,
        error_y_thickness=1.5,
        error_y_width=4,
        marker_line_width=0,
        textfont_color=_TEXT,
    )
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return _apply_base(fig, title, xlab or x, ylab)


# ── 2. Line chart ─────────────────────────────────────────────────────────────
def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    error_y: str = None,
    markers: bool = True,
    title: str = None,
    xlab: str = None,
    ylab: str = "Accuracy (%)",
    color_map: dict = None,
    dash_map: dict = None,
    category_orders: dict = None,
    **kwargs,
) -> go.Figure:
    fig = px.line(
        df, x=x, y=y, color=color,
        error_y=error_y,
        markers=markers,
        color_discrete_map=color_map or {},
        color_discrete_sequence=PALETTE,
        category_orders=category_orders or {},
        **kwargs,
    )
    if dash_map:
        for trace in fig.data:
            if trace.name in dash_map:
                trace.line.dash = dash_map[trace.name]
    fig.update_traces(
        marker_size=8,
        error_y_color=FAU_GRAY,
        error_y_thickness=1,
    )
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return _apply_base(fig, title, xlab or x, ylab)


# ── 3. Heatmap ────────────────────────────────────────────────────────────────
def heatmap(
    matrix: pd.DataFrame,
    title: str = None,
    colorscale: str = "RdYlGn",
    zmin: float = 0,
    zmax: float = 100,
    fmt: str = ".1f",
    xlab: str = "Species",
    ylab: str = "Method",
    **kwargs,
) -> go.Figure:
    z    = matrix.values.astype(float)
    xcols = [str(c) for c in matrix.columns]
    yrows = [str(r) for r in matrix.index]
    text_vals = [[f"{v:{fmt}}" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=xcols, y=yrows,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=text_vals,
        texttemplate="%{text}%",
        textfont=dict(color=_TEXT, size=11),
        hovertemplate="Method: %{y}<br>Species: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title="Acc (%)",
            title_font=dict(color=_TEXT, size=12),
            tickfont=dict(color=_TEXT, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=_LINE,
        ),
        **kwargs,
    ))
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        height=max(250, 60 * len(yrows) + 120),
    )
    fig.update_xaxes(
        tickangle=-30, side="bottom",
        tickfont=dict(color=_TEXT, size=11),
        title_font=dict(color=_TEXT),
        title_text=xlab,
        linecolor=_LINE,
    )
    fig.update_yaxes(
        tickfont=dict(color=_TEXT, size=11),
        title_font=dict(color=_TEXT),
        title_text=ylab,
        linecolor=_LINE,
    )
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color=FAU_BLUE), x=0.02)
        )
    return fig


# ── 4. Scatter with diagonal ──────────────────────────────────────────────────
def scatter_with_diagonal(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    color: str = None,
    title: str = None,
    xlab: str = None,
    ylab: str = None,
    color_map: dict = None,
    add_trendline: bool = False,
    **kwargs,
) -> go.Figure:
    kw = {"color": color} if color else {}
    fig = px.scatter(
        df, x=x, y=y, text=label_col,
        color_discrete_map=color_map or {},
        color_discrete_sequence=PALETTE,
        **kw, **kwargs,
    )
    fig.update_traces(textposition="top center", marker_size=10,
                      textfont=dict(color=_TEXT, size=11))

    all_vals = pd.concat([df[x], df[y]]).dropna()
    lo, hi = float(all_vals.min()) - 2, float(all_vals.max()) + 2
    fig.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color=FAU_GRAY, dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=hi - 1, y=hi - 3, text="Equal performance",
        showarrow=False, font=dict(size=10, color=FAU_GRAY),
    )
    if add_trendline:
        xs = df[x].dropna().values
        ys = df[y].dropna().values
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() > 1:
            m, b = np.polyfit(xs[mask], ys[mask], 1)
            xl = np.array([xs[mask].min(), xs[mask].max()])
            fig.add_trace(go.Scatter(
                x=xl, y=m * xl + b, mode="lines",
                line=dict(color=FAU_ORANGE, width=1.5, dash="dot"),
                name="Trend", showlegend=True,
            ))

    fig.update_layout(yaxis=dict(range=[lo, hi]), xaxis=dict(range=[lo, hi]))
    return _apply_base(fig, title, xlab or x, ylab or y)


# ── 5. Dual-panel k-shot line chart ──────────────────────────────────────────
def kshot_dual_panel(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    color: str,
    y: str = "accuracy",
    x: str = "k_shot",
    error_y: str = "ci",
    title_left: str = "B15/N5",
    title_right: str = "B10/N10",
    color_map: dict = None,
) -> go.Figure:
    color_discrete_map = color_map or {}
    methods = df_left[color].unique().tolist()
    cmap = {m: color_discrete_map.get(m, PALETTE[i % len(PALETTE)])
            for i, m in enumerate(methods)}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"<b>{title_left}</b>", f"<b>{title_right}</b>"],
        shared_yaxes=True,
    )
    added: set = set()
    for col_idx, df in enumerate([df_left, df_right], start=1):
        for method in methods:
            sub = df[df[color] == method].sort_values(x)
            if sub.empty:
                continue
            show = method not in added
            if show:
                added.add(method)
            err_arr = sub[error_y].values if error_y in sub.columns else None
            fig.add_trace(
                go.Scatter(
                    x=sub[x], y=sub[y],
                    mode="lines+markers",
                    name=method,
                    line=dict(color=cmap[method], width=2),
                    marker=dict(size=8, color=cmap[method]),
                    error_y=dict(
                        type="data", array=err_arr,
                        visible=err_arr is not None,
                        color=FAU_GRAY, thickness=1.5, width=4,
                    ),
                    showlegend=show,
                    legendgroup=method,
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(
        yaxis=dict(range=[0, 105], title="Accuracy (%)"),
        xaxis=dict(tickvals=[1, 3, 5], title="K (shots)"),
        xaxis2=dict(tickvals=[1, 3, 5], title="K (shots)"),
    )
    # Subplot title colour
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=13)
    _force_axis_colours(fig)
    return fig


# ── 6. Marginal gain bar ──────────────────────────────────────────────────────
def marginal_gain_bar(
    df: pd.DataFrame,
    method_col: str = "method",
    accuracy_col: str = "accuracy",
    shot_col: str = "k_shot",
    title: str = "Marginal Gain: 1→3 and 3→5 shots",
) -> go.Figure:
    def _scalar(v):
        if isinstance(v, pd.Series):
            return float(v.iloc[0]) if not v.empty else np.nan
        return float(v) if v is not None else np.nan

    methods = list(df[method_col].unique())
    gains_13, gains_35, labels = [], [], []
    for m in methods:
        sub = df[df[method_col] == m].set_index(shot_col)[accuracy_col]
        a1 = _scalar(sub.get(1.0, np.nan))
        a3 = _scalar(sub.get(3.0, np.nan))
        a5 = _scalar(sub.get(5.0, np.nan))
        gains_13.append((a3 - a1) if not (np.isnan(a1) or np.isnan(a3)) else 0.0)
        gains_35.append((a5 - a3) if not (np.isnan(a3) or np.isnan(a5)) else 0.0)
        labels.append(m)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="1→3 gain", x=labels, y=gains_13,
        marker_color=FAU_BLUE,
        text=[f"+{v:.1f}" for v in gains_13],
        textposition="inside",
        textfont=dict(color="white", size=11),
    ))
    fig.add_trace(go.Bar(
        name="3→5 gain", x=labels, y=gains_35,
        marker_color=FAU_ORANGE,
        text=[f"+{v:.1f}" for v in gains_35],
        textposition="inside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(barmode="stack", **_BASE_LAYOUT)
    return _apply_base(fig, title, ylab="Accuracy gain (pp)")


# ── 7. Distance gap bar chart ─────────────────────────────────────────────────
def distance_gap_chart(
    df: pd.DataFrame,
    group_col: str = "method",
    euc_col: str = "Euclidean",
    man_col: str = "Manhattan",
    title: str = "Euclidean vs Manhattan Gap",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Euclidean", x=df[group_col], y=df[euc_col],
                         marker_color=FAU_BLUE))
    fig.add_trace(go.Bar(name="Manhattan", x=df[group_col], y=df[man_col],
                         marker_color=FAU_ORANGE))
    for _, row in df.iterrows():
        gap = row[euc_col] - row[man_col]
        if abs(gap) > 0.05:
            fig.add_annotation(
                x=row[group_col],
                y=max(row[euc_col], row[man_col]) + 1.5,
                text=f"Δ{gap:+.1f}pp",
                showarrow=False,
                font=dict(size=10, color="red" if gap > 1 else FAU_GRAY),
            )
    fig.update_layout(barmode="group", **_BASE_LAYOUT)
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return _apply_base(fig, title, ylab="Accuracy (%)")
