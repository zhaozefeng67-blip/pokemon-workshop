import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import (
    CHART_HEIGHT_FULL,
    CHART_HEIGHT_HALF,
    PROFILE_LABELS,
    STAT_COLUMNS,
    STAT_MAP,
    TYPE_COLORS,
)
from dashboard.plots.style import clean_plotly_layout

def fig_all_type_radar(type_deviations: pd.DataFrame, full_stats: pd.DataFrame, selected_types: list[str]) -> go.Figure:
    fig = go.Figure()
    for type_name in selected_types:
        deviations = type_deviations.loc[type_name]
        stats_row = full_stats.loc[full_stats["type"] == type_name].iloc[0]
        values = deviations.tolist() + deviations.tolist()[:1]
        max_stat = deviations.idxmax()
        max_value = deviations.max()
        sign = "+" if max_value >= 0 else ""
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=PROFILE_LABELS + PROFILE_LABELS[:1],
                mode="lines",
                name=f"{type_name.capitalize()} | {STAT_MAP[max_stat]}{sign}{max_value:.1f} | BST {stats_row['average_bst']:.0f}",
                line=dict(color=TYPE_COLORS.get(type_name, "#777777"), width=2),
                opacity=0.82,
                hovertemplate="%{theta}: %{r:.1f} vs global average<extra></extra>",
            )
        )
    fig.update_layout(
        title="Pokémon Type Profiles — Deviation from Global Average",
        polar=dict(radialaxis=dict(visible=True)),
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_FULL)


def fig_bst_by_type(full_stats: pd.DataFrame, selected_types: list[str]) -> go.Figure:
    plot_df = full_stats.loc[full_stats["type"].isin(selected_types)].sort_values("average_bst", ascending=True)
    fig = px.bar(
        plot_df,
        x="average_bst",
        y="type",
        orientation="h",
        color="type",
        color_discrete_map=TYPE_COLORS,
        hover_data={"total_entries": True, "primary_count": True, "secondary_count": True, "secondary_ratio": ":.1%"},
        labels={"average_bst": "Average BST", "type": "Type"},
        title="Average Base Stat Total by Type",
    )
    fig.update_traces(text=plot_df["average_bst"].round(1), textposition="outside", showlegend=False)
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_single_type_profile(type_deviations: pd.DataFrame, type_name: str) -> go.Figure:
    deviations = type_deviations.loc[type_name]
    max_stat = deviations.idxmax()
    min_stat = deviations.idxmin()
    max_value = deviations.max()
    min_value = deviations.min()
    sign_max = "+" if max_value >= 0 else ""
    sign_min = "+" if min_value >= 0 else ""
    values = deviations.tolist() + deviations.tolist()[:1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=PROFILE_LABELS + PROFILE_LABELS[:1],
            fill="toself",
            name=type_name.capitalize(),
            line=dict(color=TYPE_COLORS.get(type_name, "#777777"), width=3),
            hovertemplate="%{theta}: %{r:.1f} vs global average<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[0] * len(values),
            theta=PROFILE_LABELS + PROFILE_LABELS[:1],
            mode="lines",
            name="Global Average",
            line=dict(color="#777777", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=(
            f"{type_name.capitalize()} Type Profile<br>"
            f"<sup>Strongest: {STAT_MAP[max_stat]}{sign_max}{max_value:.1f} | "
            f"Weakest: {STAT_MAP[min_stat]}{sign_min}{min_value:.1f}</sup>"
        ),
        polar=dict(radialaxis=dict(visible=True)),
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)
