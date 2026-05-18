import pandas as pd
import streamlit as st

from dashboard.analysis.profile import prepare_personality_profiles
from dashboard.config import DEFAULT_SHOWCASE_TYPES
from dashboard.plots.profile import fig_all_type_radar, fig_bst_by_type, fig_single_type_profile
from dashboard.ui.widgets import multiselect_popover, plot_chart, render_chart_expander, searchable_dataframe

def render_profile_section(df: pd.DataFrame) -> None:
    type_deviations, full_stats, selected_types_all = prepare_personality_profiles(df)

    default_profile_types = [t for t in DEFAULT_SHOWCASE_TYPES if t in selected_types_all]
    if len(default_profile_types) < 2:
        default_profile_types = selected_types_all[:3]

    with render_chart_expander("Pokémon Type Profiles — Deviation from Global Average", expanded=True):
        selected_types = multiselect_popover(
            "Types to display",
            selected_types_all,
            key="profile_types_v3",
            default=default_profile_types,
        )
        plot_chart(fig_all_type_radar(type_deviations, full_stats, selected_types))

    with render_chart_expander("Average Base Stat Total by Type"):
        profile_types = st.session_state.get("profile_types_v3", default_profile_types) or default_profile_types
        plot_chart(fig_bst_by_type(full_stats, profile_types))

    with render_chart_expander("Single Type Profile"):
        single_type = st.selectbox(
            "Type",
            selected_types_all,
            index=selected_types_all.index("dragon") if "dragon" in selected_types_all else 0,
            key="profile_single_type",
        )
        plot_chart(fig_single_type_profile(type_deviations, single_type))

    with st.expander("Type personality statistics table"):
        searchable_dataframe(full_stats.sort_values("average_bst", ascending=False), "profile_stats", 50)
