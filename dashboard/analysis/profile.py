import pandas as pd
import streamlit as st

from dashboard.config import STAT_COLUMNS, TYPE_COLORS
from dashboard.data.cleaning import normalize_type2

@st.cache_data
def prepare_personality_profiles(df: pd.DataFrame):
    profile_df = df[["type1", "type2"] + STAT_COLUMNS].dropna(subset=STAT_COLUMNS).copy()
    profile_df.columns = ["primary_type", "secondary_type"] + STAT_COLUMNS
    profile_df["primary_type"] = profile_df["primary_type"].astype(str).str.strip()
    profile_df["secondary_type"] = normalize_type2(profile_df["secondary_type"])

    global_means = profile_df[STAT_COLUMNS].mean()
    primary_profiles = profile_df[["primary_type"] + STAT_COLUMNS].rename(columns={"primary_type": "type"})
    secondary_profiles = profile_df.loc[
        profile_df["secondary_type"].ne("None"), ["secondary_type"] + STAT_COLUMNS
    ].rename(columns={"secondary_type": "type"})
    type_long = pd.concat([primary_profiles, secondary_profiles], ignore_index=True)
    type_means = type_long.groupby("type")[STAT_COLUMNS].mean()
    type_deviations = type_means - global_means

    rows = []
    for type_name in type_deviations.index:
        type_entries = type_long.loc[type_long["type"] == type_name]
        average_bst = type_entries[STAT_COLUMNS].sum(axis=1).mean()
        total_entries = len(type_entries)
        primary_count = int((profile_df["primary_type"] == type_name).sum())
        rows.append(
            {
                "type": type_name,
                "average_bst": round(average_bst, 1),
                "total_entries": total_entries,
                "primary_count": primary_count,
                "secondary_count": total_entries - primary_count,
            }
        )
    full_stats = pd.DataFrame(rows)
    full_stats["secondary_ratio"] = full_stats["secondary_count"] / full_stats["total_entries"]
    selected_types = [type_name for type_name in TYPE_COLORS if type_name in type_deviations.index]
    return type_deviations, full_stats, selected_types
