"""Plotly Visualization Module for AI Analytics Engineer.

Builds beautiful, responsive, modern charts with tailored palettes,
informative hover tooltips, and clean layouts.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Design System Theme Tokens
COLOR_PRIMARY = "#3B82F6"      # Indigo / Blue
COLOR_ACCENT = "#10B981"       # Emerald Green
COLOR_WARM = "#F59E0B"         # Amber
COLOR_DANGER = "#EF4444"       # Rose Red
COLOR_PURPLE = "#8B5CF6"       # Violet
COLOR_PALETTE = [
    "#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899",
    "#06B6D4", "#6366F1", "#14B8A6", "#F97316", "#84CC16"
]

LAYOUT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13),
    margin=dict(l=40, r=40, t=60, b=40),
    hoverlabel=dict(
        bgcolor="#1E293B",
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#FFFFFF"
    ),
)


def apply_custom_layout(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply consistent premium styling to a Plotly figure."""
    fig.update_layout(
        **LAYOUT_THEME,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=16, color="#1E293B"),
            x=0.02,
            y=0.95
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=13, color="#475569")),
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zerolinecolor="#E2E8F0",
        ) if x_title or not fig.layout.xaxis.title.text else {},
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=13, color="#475569")),
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zerolinecolor="#E2E8F0",
        ) if y_title or not fig.layout.yaxis.title.text else {},
    )
    return fig


def plot_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    horizontal: bool = False
) -> go.Figure:
    """Create styled bar chart."""
    if horizontal:
        fig = px.bar(
            df,
            x=y,
            y=x,
            orientation="h",
            color=color if color else y,
            color_continuous_scale="Blues",
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color if color else x,
            color_discrete_sequence=COLOR_PALETTE,
        )
        
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>" if not horizontal else "<b>%{y}</b><br>Value: %{x:,.2f}<extra></extra>"
    )
    fig = apply_custom_layout(fig, title=title, x_title=x if not horizontal else y, y_title=y if not horizontal else x)
    fig.update_layout(showlegend=False if not color else True)
    return fig


def plot_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    """Create styled line chart with markers."""
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        markers=True,
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>" if not color else "<b>%{x}</b><br>%{y:,.2f}<extra></extra>"
    )
    fig = apply_custom_layout(fig, title=title, x_title=x, y_title=y)
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    """Create styled scatter plot with optional trendline."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.85,
    )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=1, color="white")),
        hovertemplate=f"<b>{x}</b>: %{{x:,.2f}}<br><b>{y}</b>: %{{y:,.2f}}<extra></extra>"
    )
    fig = apply_custom_layout(fig, title=title, x_title=x, y_title=y)
    return fig


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    nbins: int = 30,
    title: str = ""
) -> go.Figure:
    """Create styled histogram for numerical distribution."""
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        color_discrete_sequence=[COLOR_PRIMARY],
        marginal="box",
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="#FFFFFF")),
        opacity=0.85
    )
    fig = apply_custom_layout(fig, title=title or f"Distribution of {column}", x_title=column, y_title="Frequency")
    return fig


def plot_box(
    df: pd.DataFrame,
    y: str,
    x: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    """Create styled box plot for outlier detection and quartile analysis."""
    fig = px.box(
        df,
        x=x,
        y=y,
        color=x if x else None,
        color_discrete_sequence=COLOR_PALETTE,
        points="outliers",
    )
    fig = apply_custom_layout(fig, title=title or f"Box Plot & Outliers: {y}", x_title=x or "", y_title=y)
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Create annotated correlation matrix heatmap."""
    if corr_df is None or corr_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No correlation data available", showarrow=False)
        return fig

    z_vals = corr_df.values
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Pearson r", thickness=15, len=0.8),
        text=np.round(z_vals, 2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
    ))

    fig = apply_custom_layout(fig, title="Numerical Correlation Heatmap")
    fig.update_layout(
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed")
    )
    return fig


def create_chart_from_result(
    suggested_chart: Optional[Dict[str, Any]],
    data: Any,
    full_df: pd.DataFrame
) -> Optional[go.Figure]:
    """Dynamically generate appropriate Plotly chart from analytical result."""
    if not suggested_chart:
        return None
        
    chart_type = suggested_chart.get("type", "").lower()
    title = suggested_chart.get("title", "")
    
    try:
        if chart_type == "bar":
            if isinstance(data, pd.DataFrame) and not data.empty:
                x_col = suggested_chart.get("x")
                y_col = suggested_chart.get("y")
                if x_col in data.columns and y_col in data.columns:
                    return plot_bar_chart(data, x=x_col, y=y_col, title=title)
            elif isinstance(data, pd.Series):
                s_df = data.reset_index()
                return plot_bar_chart(s_df, x=s_df.columns[0], y=s_df.columns[1], title=title)
                
        elif chart_type == "line":
            if isinstance(data, pd.DataFrame) and not data.empty:
                x_col = suggested_chart.get("x")
                y_col = suggested_chart.get("y")
                color_col = suggested_chart.get("color")
                if x_col in data.columns and y_col in data.columns:
                    return plot_line_chart(data, x=x_col, y=y_col, color=color_col if color_col in data.columns else None, title=title)
                    
        elif chart_type == "scatter":
            x_col = suggested_chart.get("x")
            y_col = suggested_chart.get("y")
            color_col = suggested_chart.get("color")
            if x_col in full_df.columns and y_col in full_df.columns:
                return plot_scatter(full_df, x=x_col, y=y_col, color=color_col if color_col in full_df.columns else None, title=title)
                
        elif chart_type == "histogram":
            x_col = suggested_chart.get("x")
            if x_col in full_df.columns:
                return plot_histogram(full_df, column=x_col, title=title)
                
        elif chart_type == "box":
            y_col = suggested_chart.get("y")
            if y_col in full_df.columns:
                return plot_box(full_df, y=y_col, title=title)
                
    except Exception:
        return None
        
    return None
