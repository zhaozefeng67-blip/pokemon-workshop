import pandas as pd
import streamlit as st

from dashboard.data.cleaning import normalize_type2, parse_capture_rate

@st.cache_data
def prepare_capture_analysis(df: pd.DataFrame):
    capture_df = df.copy()
    capture_df["capture_rate_num"] = capture_df["capture_rate"].apply(parse_capture_rate)
    min_rate = capture_df["capture_rate_num"].min()
    max_rate = capture_df["capture_rate_num"].max()
    if max_rate == min_rate:
        capture_df["capture_difficulty_coef"] = 0.0
    else:
        capture_df["capture_difficulty_coef"] = (max_rate - capture_df["capture_rate_num"]) / (max_rate - min_rate)
    capture_df["type2_cleaned"] = normalize_type2(capture_df["type2"])
    capture_df["legendary_label"] = capture_df["is_legendary"].map({0: "Non-Legendary", 1: "Legendary"})

    type_difficulty = (
        capture_df.groupby("type1")
        .agg(avg_difficulty=("capture_difficulty_coef", "mean"), avg_capture_rate=("capture_rate_num", "mean"), count=("name", "count"))
        .reset_index()
        .sort_values("avg_difficulty", ascending=False)
    )
    generation_difficulty = (
        capture_df.groupby("generation")
        .agg(avg_difficulty=("capture_difficulty_coef", "mean"), avg_capture_rate=("capture_rate_num", "mean"), count=("name", "count"))
        .reset_index()
    )
    legendary_summary = (
        capture_df.groupby("legendary_label")
        .agg(avg_difficulty=("capture_difficulty_coef", "mean"), avg_capture_rate=("capture_rate_num", "mean"), count=("name", "count"))
        .reset_index()
    )
    top_hardest = (
        capture_df[["name", "type1", "type2_cleaned", "generation", "is_legendary", "capture_rate_num", "capture_difficulty_coef", "base_total"]]
        .sort_values(by=["capture_difficulty_coef", "base_total"], ascending=[False, False])
        .head(15)
    )
    top_easiest = (
        capture_df[["name", "type1", "type2_cleaned", "generation", "is_legendary", "capture_rate_num", "capture_difficulty_coef", "base_total"]]
        .sort_values(by=["capture_difficulty_coef", "base_total"], ascending=[True, True])
        .head(15)
    )
    return capture_df, type_difficulty, generation_difficulty, legendary_summary, top_hardest, top_easiest
