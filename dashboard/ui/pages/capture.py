import pandas as pd
import streamlit as st

from dashboard.analysis.capture import prepare_capture_analysis
from dashboard.plots.capture import (
    fig_capture_box,
    fig_capture_by_generation,
    fig_capture_by_type,
    fig_capture_distribution,
    fig_difficulty_vs_bst,
    fig_top_capture_rank,
)
from dashboard.ui.widgets import multiselect_popover, plot_chart, render_chart_expander, searchable_dataframe

def render_capture_section(df: pd.DataFrame) -> None:
    capture_df, type_difficulty, generation_difficulty, legendary_summary, top_hardest, top_easiest = prepare_capture_analysis(df)

    type_options = sorted(capture_df["type1"].dropna().unique())
    generations = sorted(capture_df["generation"].dropna().unique())

    def filtered_capture_df() -> pd.DataFrame:
        selected_types = st.session_state.get("capture_types", type_options) or type_options
        raw_generations = st.session_state.get("capture_generations", generations) or generations
        selected_generations = [int(value) for value in raw_generations]
        legendary_choice = st.session_state.get("capture_legendary", "All")
        preview = capture_df.loc[
            capture_df["type1"].isin(selected_types) & capture_df["generation"].isin(selected_generations)
        ].copy()
        if legendary_choice == "Legendary":
            preview = preview.loc[preview["is_legendary"] == 1]
        elif legendary_choice == "Non-Legendary":
            preview = preview.loc[preview["is_legendary"] == 0]
        return preview

    with render_chart_expander("Distribution of Capture Difficulty Coefficient", expanded=True):
        filter_cols = st.columns(2)
        with filter_cols[0]:
            multiselect_popover("Primary types", type_options, key="capture_types", default=type_options)
        with filter_cols[1]:
            multiselect_popover("Generation", [str(g) for g in generations], key="capture_generations", default=[str(g) for g in generations])
        bins = st.slider("Histogram bins", 10, 50, 20, key="capture_bins")
        plot_chart(fig_capture_distribution(filtered_capture_df(), bins))

    with render_chart_expander("Average Capture Difficulty by Primary Type"):
        selected_types = st.session_state.get("capture_types", type_options) or type_options
        plot_chart(fig_capture_by_type(type_difficulty, selected_types))

    with render_chart_expander("Capture Difficulty vs Base Total"):
        st.radio(
            "Legendary filter",
            ["All", "Legendary", "Non-Legendary"],
            horizontal=True,
            key="capture_legendary",
        )
        plot_chart(fig_difficulty_vs_bst(filtered_capture_df()))

    with render_chart_expander("Capture Difficulty by Legendary Status"):
        plot_chart(fig_capture_box(capture_df))

    with render_chart_expander("Average Capture Difficulty by Generation"):
        plot_chart(fig_capture_by_generation(generation_difficulty))

    with render_chart_expander("Top 15 Hardest Pokémon (minimum capture rate tier)"):
        plot_chart(
            fig_top_capture_rank(
                top_hardest,
                "Top 15 Hardest Pokémon: BST within the Minimum Capture Rate Tier",
                "#4F6F82",
            )
        )

    with render_chart_expander("Top 15 Easiest Pokémon (maximum capture rate tier)"):
        plot_chart(
            fig_top_capture_rank(
                top_easiest,
                "Top 15 Easiest Pokémon: BST within the Maximum Capture Rate Tier",
                "#6E7469",
            )
        )

    with st.expander("Capture summary tables"):
        tab1, tab2, tab3 = st.tabs(["Hardest", "Easiest", "Legendary Summary"])
        with tab1:
            searchable_dataframe(top_hardest, "capture_hardest", 15)
        with tab2:
            searchable_dataframe(top_easiest, "capture_easiest", 15)
        with tab3:
            st.dataframe(legendary_summary, use_container_width=True, hide_index=True)
