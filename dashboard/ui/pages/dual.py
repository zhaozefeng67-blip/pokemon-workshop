import pandas as pd
import streamlit as st

from dashboard.analysis.dual import prepare_dual_analysis
from dashboard.config import DEFAULT_SHOWCASE_TYPES, TYPE_COLORS
from dashboard.plots.dual import (
    fig_dual_frequency_heatmap,
    fig_dual_tier_distribution,
    fig_legendary_dual_ratio,
    fig_popular_dual_radar,
    fig_top_dual_bst,
)
from dashboard.ui.widgets import multiselect_popover, plot_chart, render_chart_expander, searchable_dataframe

def render_dual_section(df: pd.DataFrame) -> None:
    _, dual_type_stats, true_dual_stats, frequency_matrix = prepare_dual_analysis(df)
    max_combo_count = int(true_dual_stats["count"].max())
    default_combos = [
        combo
        for combo in ["normal + flying", "bug + flying", "grass + poison", "water + flying"]
        if combo in set(true_dual_stats["dual_type"])
    ]

    all_types = list(TYPE_COLORS.keys())
    default_heatmap_types = [t for t in DEFAULT_SHOWCASE_TYPES if t in all_types]
    if len(default_heatmap_types) < 2:
        default_heatmap_types = all_types[:3]

    with render_chart_expander("Dual Type Combination Frequency", expanded=True):
        selected_types = multiselect_popover(
            "Types in heatmap",
            all_types,
            key="dual_heatmap_types_v3",
            default=default_heatmap_types,
        )
        plot_chart(fig_dual_frequency_heatmap(frequency_matrix, selected_types))

    with render_chart_expander("Top Strongest Dual Type Combinations"):
        slider_cols = st.columns(2)
        top_n = slider_cols[0].slider("Top combinations", 5, 25, 15, key="dual_top_n")
        min_count = slider_cols[1].slider(
            "Min Pokémon per combination",
            1,
            max_combo_count,
            1,
            key="dual_min_count",
        )
        plot_chart(fig_top_dual_bst(true_dual_stats, top_n, min_count))

    with render_chart_expander("Ability Profiles of Selected Dual Type Combinations"):
        combo_options = sorted(true_dual_stats["dual_type"].unique())
        selected_combos = multiselect_popover(
            "Dual combinations",
            combo_options,
            key="dual_radar_combos",
            default=default_combos,
        )
        plot_chart(fig_popular_dual_radar(true_dual_stats, selected_combos))

    with render_chart_expander("Dual Type Combination Power Tier Distribution"):
        plot_chart(fig_dual_tier_distribution(true_dual_stats))

    with render_chart_expander("Top Dual Types by Legendary Ratio"):
        legendary_top_n = st.slider("Show top N", 5, 15, 10, key="dual_legendary_top")
        plot_chart(fig_legendary_dual_ratio(true_dual_stats, legendary_top_n))

    with st.expander("Dual type statistics table"):
        searchable_dataframe(dual_type_stats.sort_values("avg_bst", ascending=False), "dual_stats", 50)
