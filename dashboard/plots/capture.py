import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import CHART_HEIGHT_HALF, TYPE_COLORS
from dashboard.plots.style import clean_plotly_layout

def fig_capture_distribution(capture_df: pd.DataFrame, bins: int) -> go.Figure:
    fig = px.histogram(
        capture_df,
        x="capture_difficulty_coef",
        nbins=bins,
        title="Distribution of Capture Difficulty Coefficient",
        labels={"capture_difficulty_coef": "Capture Difficulty Coefficient", "count": "Count"},
        color_discrete_sequence=["#4F6F82"],
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_capture_by_type(type_difficulty: pd.DataFrame, selected_types: list[str]) -> go.Figure:
    plot_df = type_difficulty.loc[type_difficulty["type1"].isin(selected_types)].sort_values("avg_difficulty", ascending=True)
    fig = px.bar(
        plot_df,
        x="avg_difficulty",
        y="type1",
        orientation="h",
        color="type1",
        color_discrete_map=TYPE_COLORS,
        hover_data={"avg_capture_rate": ":.1f", "count": True},
        title="Average Capture Difficulty by Primary Type",
        labels={"avg_difficulty": "Average Capture Difficulty", "type1": "Primary Type"},
    )
    fig.update_traces(text=plot_df["avg_difficulty"].round(3), textposition="outside", showlegend=False)
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_difficulty_vs_bst(plot_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        plot_df,
        x="capture_difficulty_coef",
        y="base_total",
        color="legendary_label",
        hover_name="name",
        hover_data={"type1": True, "generation": True, "capture_rate_num": True, "legendary_label": False},
        color_discrete_map={"Non-Legendary": "#4F6F82", "Legendary": "#9A4D4D"},
        title="Capture Difficulty vs Base Total",
        labels={"capture_difficulty_coef": "Capture Difficulty Coefficient", "base_total": "Base Total"},
    )
    x = plot_df["capture_difficulty_coef"]
    y = plot_df["base_total"]
    if len(plot_df) > 1 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Trend (r={x.corr(y):.3f})",
                line=dict(color="#333333", width=2, dash="dash"),
            )
        )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_capture_box(capture_df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        capture_df,
        x="legendary_label",
        y="capture_difficulty_coef",
        color="legendary_label",
        color_discrete_map={"Non-Legendary": "#4F6F82", "Legendary": "#9A4D4D"},
        points="outliers",
        title="Capture Difficulty by Legendary Status",
        labels={"legendary_label": "Legendary Status", "capture_difficulty_coef": "Capture Difficulty Coefficient"},
    )
    fig.update_layout(showlegend=False)
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_capture_by_generation(generation_difficulty: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        generation_difficulty,
        x=generation_difficulty["generation"].astype(str),
        y="avg_difficulty",
        text=generation_difficulty["avg_difficulty"].round(3),
        hover_data={"avg_capture_rate": ":.1f", "count": True},
        color_discrete_sequence=["#6E7469"],
        title="Average Capture Difficulty by Generation",
        labels={"x": "Generation", "avg_difficulty": "Average Capture Difficulty"},
    )
    fig.update_traces(textposition="outside")
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_top_capture_rank(plot_df: pd.DataFrame, title: str, color: str) -> go.Figure:
    plot_df = plot_df.sort_values("base_total", ascending=True)
    fig = px.bar(
        plot_df,
        x="base_total",
        y="name",
        orientation="h",
        hover_data={"type1": True, "type2_cleaned": True, "capture_rate_num": True, "capture_difficulty_coef": ":.3f"},
        title=title,
        labels={"base_total": "Base Stat Total", "name": "Pokemon"},
        color_discrete_sequence=[color],
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)
