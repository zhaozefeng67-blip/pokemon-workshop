import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import CHART_HEIGHT_FULL, CHART_HEIGHT_HALF, PROFILE_LABELS, TYPE_COLORS
from dashboard.plots.style import clean_plotly_layout

def fig_dual_frequency_heatmap(frequency_matrix: pd.DataFrame, selected_types: list[str]) -> go.Figure:
    matrix = frequency_matrix.loc[selected_types, selected_types]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="Blues",
            text=matrix.values,
            texttemplate="%{text}",
            hovertemplate="Primary type: %{y}<br>Secondary type: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(title="Dual Type Combination Frequency", xaxis_title="Secondary Type", yaxis_title="Primary Type")
    return clean_plotly_layout(fig, height=CHART_HEIGHT_FULL)


def fig_top_dual_bst(true_dual_stats: pd.DataFrame, top_n: int, min_count: int) -> go.Figure:
    plot_df = (
        true_dual_stats.loc[true_dual_stats["count"] >= min_count]
        .sort_values("avg_bst", ascending=False)
        .head(top_n)
        .sort_values("avg_bst", ascending=True)
    )
    colors = [TYPE_COLORS.get(name.split(" + ")[0], "#777777") for name in plot_df["dual_type"]]
    fig = go.Figure(
        go.Bar(
            x=plot_df["avg_bst"],
            y=plot_df["dual_type"],
            orientation="h",
            marker_color=colors,
            text=plot_df["avg_bst"].round(1),
            textposition="outside",
            hovertemplate="Dual type: %{y}<br>Avg BST: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Strongest Dual Type Combinations",
        xaxis_title="Average Base Stat Total",
        yaxis_title="Dual Type",
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_popular_dual_radar(true_dual_stats: pd.DataFrame, combinations: list[str]) -> go.Figure:
    fig = go.Figure()
    plot_df = true_dual_stats.loc[true_dual_stats["dual_type"].isin(combinations)]
    for _, row in plot_df.iterrows():
        primary = row["dual_type"].split(" + ")[0]
        values = [
            row["avg_hp"],
            row["avg_attack"],
            row["avg_defense"],
            row["avg_sp_attack"],
            row["avg_sp_defense"],
            row["avg_speed"],
        ]
        values = values + values[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=PROFILE_LABELS + PROFILE_LABELS[:1],
                fill="toself",
                name=f"{row['dual_type']} (BST {row['avg_bst']:.0f})",
                line=dict(color=TYPE_COLORS.get(primary, "#777777"), width=2),
                opacity=0.78,
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            )
        )
    fig.update_layout(title="Ability Profiles of Selected Dual Type Combinations", polar=dict(radialaxis=dict(visible=True)))
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_dual_tier_distribution(true_dual_stats: pd.DataFrame) -> go.Figure:
    tier_order = ["T0 (Top Tier)", "T1 (Strong Tier)", "T2 (Mid Tier)", "T3 (Basic Tier)"]
    tier_counts = true_dual_stats["tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
    tier_counts.columns = ["tier", "count"]
    fig = px.bar(
        tier_counts,
        x="tier",
        y="count",
        text="count",
        color="tier",
        color_discrete_sequence=["#8E3B46", "#B77742", "#B6A650", "#6E8F70"],
        title="Dual Type Combination Power Tier Distribution",
        labels={"tier": "Power Tier", "count": "Number of Combinations"},
    )
    fig.update_traces(textposition="outside", showlegend=False)
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


def fig_legendary_dual_ratio(true_dual_stats: pd.DataFrame, top_n: int) -> go.Figure:
    plot_df = (
        true_dual_stats.loc[true_dual_stats["legendary_count"] > 0]
        .sort_values("legendary_ratio", ascending=False)
        .head(top_n)
        .sort_values("legendary_ratio", ascending=True)
    )
    colors = [TYPE_COLORS.get(name.split(" + ")[0], "#777777") for name in plot_df["dual_type"]]
    ratio_pct = plot_df["legendary_ratio"] * 100
    fig = go.Figure(
        go.Bar(
            x=ratio_pct,
            y=plot_df["dual_type"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in ratio_pct],
            textposition="outside",
            hovertemplate="Dual type: %{y}<br>Legendary ratio: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Dual Types by Legendary Ratio",
        xaxis_title="Legendary Ratio (%)",
        yaxis_title="Dual Type",
    )
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)
