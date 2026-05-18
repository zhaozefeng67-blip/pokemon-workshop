import pandas as pd
import streamlit as st

from dashboard.ui.widgets import multiselect_popover

def render_data_preview_section(df: pd.DataFrame) -> None:
    search = st.text_input("Search", key="data_preview_search", placeholder="Name, type…")
    type_options = sorted(df["type1"].dropna().unique())
    generations = sorted(df["generation"].dropna().unique())
    filter_cols = st.columns([1, 1, 1])
    with filter_cols[0]:
        selected_types = multiselect_popover("Primary type", type_options, key="data_preview_types", default=type_options)
    with filter_cols[1]:
        selected_generations = multiselect_popover(
            "Generation",
            [str(g) for g in generations],
            key="data_preview_generations",
            default=[str(g) for g in generations],
        )
        selected_generations = [int(value) for value in selected_generations]
    with filter_cols[2]:
        legendary_filter = st.radio(
            "Legendary",
            ["All", "Legendary", "Non-Legendary"],
            horizontal=True,
            key="data_preview_legendary",
        )

    preview = df.loc[df["type1"].isin(selected_types) & df["generation"].isin(selected_generations)].copy()
    if legendary_filter == "Legendary":
        preview = preview.loc[preview["is_legendary"] == 1]
    elif legendary_filter == "Non-Legendary":
        preview = preview.loc[preview["is_legendary"] == 0]
    if search:
        mask = preview.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        preview = preview.loc[mask]

    st.caption(f"{len(preview):,} of {len(df):,} Pokémon")
    display_cols = [
        "name",
        "type1",
        "type2",
        "generation",
        "is_legendary",
        "base_total",
        "capture_rate",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
    ]
    st.dataframe(preview[display_cols], use_container_width=True, hide_index=True)
