import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import TYPE_COLORS
from dashboard.data.cleaning import assign_tier, normalize_type2

@st.cache_data
def prepare_dual_analysis(df: pd.DataFrame):
    dual_df = df.copy()
    dual_df["type1"] = dual_df["type1"].astype(str).str.strip()
    dual_df["type2_cleaned"] = normalize_type2(dual_df["type2"])
    dual_df["dual_type"] = np.where(
        dual_df["type2_cleaned"].ne("None"),
        dual_df["type1"] + " + " + dual_df["type2_cleaned"],
        dual_df["type1"] + " (Single Type)",
    )

    dual_type_stats = (
        dual_df.groupby("dual_type")
        .agg(
            count=("name", "count"),
            legendary_count=("is_legendary", "sum"),
            avg_bst=("base_total", "mean"),
            avg_hp=("hp", "mean"),
            avg_attack=("attack", "mean"),
            avg_defense=("defense", "mean"),
            avg_sp_attack=("sp_attack", "mean"),
            avg_sp_defense=("sp_defense", "mean"),
            avg_speed=("speed", "mean"),
        )
        .reset_index()
    )
    dual_type_stats["legendary_ratio"] = dual_type_stats["legendary_count"] / dual_type_stats["count"]
    dual_type_stats["tier"] = dual_type_stats["avg_bst"].apply(assign_tier)

    all_types = list(TYPE_COLORS.keys())
    frequency_matrix = pd.DataFrame(0, index=all_types, columns=all_types)
    for _, row in dual_df.loc[dual_df["type2_cleaned"].ne("None")].iterrows():
        if row["type1"] in frequency_matrix.index and row["type2_cleaned"] in frequency_matrix.columns:
            frequency_matrix.loc[row["type1"], row["type2_cleaned"]] += 1

    true_dual_stats = dual_type_stats.loc[
        ~dual_type_stats["dual_type"].str.contains("Single Type", regex=False)
    ].copy()
    return dual_df, dual_type_stats, true_dual_stats, frequency_matrix
