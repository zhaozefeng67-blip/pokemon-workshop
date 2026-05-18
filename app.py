from pathlib import Path
from typing import Optional
from math import pi
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "pokemon.csv"

SECTION_DUAL = "Analysis of Pokémon Dual Types"
SECTION_PROFILE = "Analysis of Pokémon Personality Profiles"
SECTION_CAPTURE = "Exploratory Analysis of Pokémon Capture Difficulty"
SECTION_ML = "Machine Learning"

TYPE_COLORS = {
    "normal": "#7C7C61",
    "fire": "#C7602B",
    "water": "#4B70B8",
    "electric": "#C89C24",
    "grass": "#5B9B49",
    "ice": "#6FA6A6",
    "fighting": "#9C332E",
    "poison": "#7E3F7F",
    "ground": "#B39552",
    "flying": "#7D6CB8",
    "psychic": "#BB4C73",
    "bug": "#7E8A2C",
    "rock": "#8C7B35",
    "ghost": "#594772",
    "dragon": "#5A45A5",
    "dark": "#5E5148",
    "steel": "#7F8396",
    "fairy": "#B66F84",
}

STAT_COLUMNS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
PROFILE_LABELS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
STAT_MAP = {
    "hp": "HP",
    "attack": "Atk",
    "defense": "Def",
    "sp_attack": "SpA",
    "sp_defense": "SpD",
    "speed": "Spe",
}
RADAR_ANGLES = [n / len(PROFILE_LABELS) * 2 * pi for n in range(len(PROFILE_LABELS))]
RADAR_ANGLES += RADAR_ANGLES[:1]


st.set_page_config(
    page_title="Pokémon Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f7f4;
        color: #252525;
    }
    [data-testid="stSidebar"] {
        background: #efeee9;
        border-right: 1px solid #dedbd1;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: #202020;
    }
    .hero {
        padding: 1.25rem 1.35rem;
        border: 1px solid #dedbd1;
        border-radius: 8px;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #555;
        margin-bottom: 0;
        line-height: 1.55;
    }
    .chart-note {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #666;
        font-size: 0.92rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_type2(series: pd.Series) -> pd.Series:
    return (
        series.fillna("None")
        .astype(str)
        .str.strip()
        .replace({"": "None", "nan": "None", "NaN": "None"})
    )


def parse_capture_rate(value) -> float:
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"\d+", str(value).strip())
    if not numbers:
        return np.nan
    return float(min(int(number) for number in numbers))


def assign_tier(avg_bst: float) -> str:
    if avg_bst >= 600:
        return "T0 (Top Tier)"
    if avg_bst >= 550:
        return "T1 (Strong Tier)"
    if avg_bst >= 500:
        return "T2 (Mid Tier)"
    return "T3 (Basic Tier)"


def clean_plotly_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial, sans-serif", size=13, color="#2b2b2b"),
        title_font=dict(size=18, color="#1f1f1f"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=62, b=38),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


# Notebook Part 1: Analysis of Pokémon dual types.
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
    return clean_plotly_layout(fig, height=720)


def fig_top_dual_bst(true_dual_stats: pd.DataFrame, top_n: int, min_count: int) -> go.Figure:
    plot_df = (
        true_dual_stats.loc[true_dual_stats["count"] >= min_count]
        .sort_values("avg_bst", ascending=False)
        .head(top_n)
        .sort_values("avg_bst", ascending=True)
    )
    colors = [TYPE_COLORS.get(name.split(" + ")[0], "#777777") for name in plot_df["dual_type"]]
    fig = px.bar(
        plot_df,
        x="avg_bst",
        y="dual_type",
        orientation="h",
        color="dual_type",
        color_discrete_sequence=colors,
        hover_data={"avg_bst": ":.1f", "count": True, "legendary_ratio": ":.1%"},
        labels={"avg_bst": "Average Base Stat Total", "dual_type": "Dual Type"},
        title=f"Top {top_n} Strongest Dual Type Combinations",
    )
    fig.update_traces(text=plot_df["avg_bst"].round(1), textposition="outside", showlegend=False)
    return clean_plotly_layout(fig, height=520)


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
    return clean_plotly_layout(fig, height=600)


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
    return clean_plotly_layout(fig, height=460)


def fig_legendary_dual_ratio(true_dual_stats: pd.DataFrame, top_n: int) -> go.Figure:
    plot_df = (
        true_dual_stats.loc[true_dual_stats["legendary_count"] > 0]
        .sort_values("legendary_ratio", ascending=False)
        .head(top_n)
        .sort_values("legendary_ratio", ascending=True)
    )
    fig = px.bar(
        plot_df,
        x=plot_df["legendary_ratio"] * 100,
        y="dual_type",
        orientation="h",
        color="dual_type",
        color_discrete_sequence=[TYPE_COLORS.get(name.split(" + ")[0], "#777777") for name in plot_df["dual_type"]],
        hover_data={"count": True, "legendary_count": True, "legendary_ratio": ":.1%"},
        title=f"Top {top_n} Dual Types by Legendary Ratio",
        labels={"x": "Legendary Ratio (%)", "dual_type": "Dual Type"},
    )
    fig.update_traces(text=[f"{v * 100:.1f}%" for v in plot_df["legendary_ratio"]], textposition="outside", showlegend=False)
    return clean_plotly_layout(fig, height=480)


# Notebook Part 2: Analysis of Pokémon Personality Profiles.
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
    fig.update_layout(title="All 18 Pokémon Types - Deviation from Global Average", polar=dict(radialaxis=dict(visible=True)))
    return clean_plotly_layout(fig, height=680)


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
    return clean_plotly_layout(fig, height=560)


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
    return clean_plotly_layout(fig, height=560)


# Notebook Part 3: Exploratory Analysis of Pokémon Capture Difficulty.
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


def fig_capture_distribution(capture_df: pd.DataFrame, bins: int) -> go.Figure:
    fig = px.histogram(
        capture_df,
        x="capture_difficulty_coef",
        nbins=bins,
        title="Distribution of Capture Difficulty Coefficient",
        labels={"capture_difficulty_coef": "Capture Difficulty Coefficient", "count": "Count"},
        color_discrete_sequence=["#4F6F82"],
    )
    return clean_plotly_layout(fig, height=480)


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
    return clean_plotly_layout(fig, height=560)


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
    return clean_plotly_layout(fig, height=540)


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
    return clean_plotly_layout(fig, height=480)


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
    return clean_plotly_layout(fig, height=460)


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
    return clean_plotly_layout(fig, height=520)


# Notebook Part 4: Machine Learning.
@st.cache_data
def train_legendary_model(df: pd.DataFrame, test_size: float = 0.30):
    model_vars = [
        "name",
        "is_legendary",
        "type1",
        "generation",
        "height_m",
        "weight_kg",
        "capture_rate",
        "base_egg_steps",
        "experience_growth",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
    ]
    pokemon_model = df[model_vars].copy()
    numeric_features = [
        "height_m",
        "weight_kg",
        "capture_rate",
        "base_egg_steps",
        "experience_growth",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
    ]
    categorical_features = ["type1", "generation"]

    for col in numeric_features:
        if col == "capture_rate":
            pokemon_model[col] = pokemon_model[col].apply(parse_capture_rate)
        pokemon_model[col] = pd.to_numeric(pokemon_model[col], errors="coerce")

    pokemon_model["generation"] = pokemon_model["generation"].astype(str)
    pokemon_model = pokemon_model.dropna().copy()

    x_data = pokemon_model[numeric_features + categorical_features]
    y_data = pokemon_model["is_legendary"]
    names = pokemon_model["name"]

    x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(
        x_data,
        y_data,
        names,
        test_size=test_size,
        random_state=123,
        stratify=y_data,
    )

    scaler = StandardScaler()
    x_train_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(x_train[numeric_features]),
        columns=numeric_features,
        index=x_train.index,
    )
    x_test_numeric_scaled = pd.DataFrame(
        scaler.transform(x_test[numeric_features]),
        columns=numeric_features,
        index=x_test.index,
    )
    x_train_categorical = pd.get_dummies(x_train[categorical_features], columns=categorical_features, drop_first=True)
    x_test_categorical = pd.get_dummies(x_test[categorical_features], columns=categorical_features, drop_first=True)
    x_test_categorical = x_test_categorical.reindex(columns=x_train_categorical.columns, fill_value=0)

    x_train_processed = pd.concat([x_train_numeric_scaled, x_train_categorical], axis=1)
    x_test_processed = pd.concat([x_test_numeric_scaled, x_test_categorical], axis=1)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        max_iter=2000,
        random_state=123,
        class_weight="balanced",
    )
    model.fit(x_train_processed, y_train)
    y_pred = model.predict(x_test_processed)
    y_prob = model.predict_proba(x_test_processed)[:, 1]

    predictions = pd.DataFrame(
        {
            "Pokemon": names_test.values,
            "Actual": y_test.values,
            "Predicted": y_pred,
            "Legendary_Probability": y_prob,
        }
    ).sort_values("Legendary_Probability", ascending=False)

    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Balanced Accuracy", "Recall", "Precision"],
            "Score": [
                accuracy_score(y_test, y_pred),
                balanced_accuracy_score(y_test, y_pred),
                recall_score(y_test, y_pred, pos_label=1),
                precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            ],
        }
    )
    coef_df = pd.DataFrame({"Feature": x_train_processed.columns, "Coefficient": model.coef_[0]})
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "pokemon_model": pokemon_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "coef_df": coef_df,
        "cm": cm,
    }


def fig_legendary_class_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["is_legendary"].value_counts().sort_index().reindex([0, 1], fill_value=0)
    percents = df["is_legendary"].value_counts(normalize=True).sort_index().reindex([0, 1], fill_value=0) * 100
    plot_df = pd.DataFrame(
        {"Class": ["Non-Legendary", "Legendary"], "Percentage": percents.values, "Count": counts.values}
    )
    fig = px.bar(
        plot_df,
        x="Class",
        y="Percentage",
        text=plot_df.apply(lambda row: f"{row['Percentage']:.1f}%<br>n={int(row['Count'])}", axis=1),
        color="Class",
        color_discrete_map={"Non-Legendary": "#4F6F82", "Legendary": "#9A4D4D"},
        title="Class Distribution of Legendary Status",
        hover_data={"Count": True, "Percentage": ":.1f"},
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_yaxes(range=[0, 100])
    return clean_plotly_layout(fig, height=440)


def fig_train_test_split(x_train: pd.DataFrame, x_test: pd.DataFrame) -> go.Figure:
    train_size = len(x_train)
    test_size = len(x_test)
    total = train_size + test_size
    plot_df = pd.DataFrame(
        {
            "Set": ["Training Set", "Testing Set"],
            "Percentage": [train_size / total * 100, test_size / total * 100],
            "Count": [train_size, test_size],
        }
    )
    fig = px.bar(
        plot_df,
        x="Set",
        y="Percentage",
        text=plot_df.apply(lambda row: f"{row['Percentage']:.1f}%<br>n={int(row['Count'])}", axis=1),
        color="Set",
        color_discrete_sequence=["#4F6F82", "#9A7B4F"],
        title="Train-Test Split",
        hover_data={"Count": True, "Percentage": ":.1f"},
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_yaxes(range=[0, 100])
    return clean_plotly_layout(fig, height=420)


def fig_train_test_class_distribution(y_train: pd.Series, y_test: pd.Series) -> go.Figure:
    train_percent = y_train.value_counts(normalize=True).sort_index().reindex([0, 1], fill_value=0) * 100
    test_percent = y_test.value_counts(normalize=True).sort_index().reindex([0, 1], fill_value=0) * 100
    plot_df = pd.DataFrame(
        {
            "Class": ["Non-Legendary", "Legendary"] * 2,
            "Set": ["Training Set", "Training Set", "Testing Set", "Testing Set"],
            "Percentage": list(train_percent.values) + list(test_percent.values),
        }
    )
    fig = px.bar(
        plot_df,
        x="Class",
        y="Percentage",
        color="Set",
        barmode="group",
        text=plot_df["Percentage"].map(lambda value: f"{value:.1f}%"),
        color_discrete_sequence=["#4F6F82", "#9A7B4F"],
        title="Class Distribution in Training and Testing Sets",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 100])
    return clean_plotly_layout(fig, height=440)


def fig_confusion_matrix(cm: np.ndarray) -> go.Figure:
    labels = ["Non-Legendary", "Legendary"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(title="Confusion Matrix")
    return clean_plotly_layout(fig, height=460)


def fig_model_metrics(metrics: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        metrics,
        x="Metric",
        y="Score",
        text=metrics["Score"].map(lambda value: f"{value:.3f}"),
        color="Metric",
        color_discrete_sequence=["#4F6F82", "#6E7469", "#9A7B4F", "#9A4D4D"],
        title="Model Performance Metrics",
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_yaxes(range=[0, 1.05])
    return clean_plotly_layout(fig, height=430)


def fig_top_coefficients(coef_df: pd.DataFrame, top_n: int) -> go.Figure:
    plot_df = coef_df.head(top_n).sort_values("Coefficient", ascending=True)
    fig = px.bar(
        plot_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"Top {top_n} Logistic Regression Coefficients",
        hover_data={"Abs_Coefficient": ":.3f"},
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#333333")
    return clean_plotly_layout(fig, height=max(460, top_n * 30))


def searchable_dataframe(df: pd.DataFrame, key_prefix: str, default_rows: int = 25) -> pd.DataFrame:
    query = st.text_input("Search table", key=f"{key_prefix}_search", placeholder="Search by name, type, or any visible value")
    visible = df.copy()
    if query:
        mask = visible.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        visible = visible.loc[mask]
    st.dataframe(visible.head(default_rows), use_container_width=True, hide_index=True)
    return visible


def render_chart(fig: go.Figure, explanation: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
    st.markdown(f"<div class='chart-note'>{explanation}</div>", unsafe_allow_html=True)


def render_project_header(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Pokémon Data Analysis Dashboard</h1>
            <p>
                This Streamlit application converts the combined project notebook into an interactive dashboard.
                It explores type combinations, battle-stat personality profiles, capture difficulty, and a
                machine learning model for predicting legendary status using the original Pokémon dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Pokémon", f"{len(df):,}")
    cols[1].metric("Columns", f"{df.shape[1]:,}")
    cols[2].metric("Types", f"{df['type1'].nunique():,}")
    cols[3].metric("Legendary", f"{int(df['is_legendary'].sum()):,}")

    with st.expander("Interactive data preview", expanded=False):
        preview_cols = st.columns([1.2, 1.2, 1.2, 1])
        search = preview_cols[0].text_input("Search Pokémon", key="global_search")
        type_options = sorted(df["type1"].dropna().unique())
        selected_types = preview_cols[1].multiselect("Primary type", type_options, default=type_options, key="global_types")
        generations = sorted(df["generation"].dropna().unique())
        selected_generations = preview_cols[2].multiselect("Generation", generations, default=generations, key="global_generations")
        legendary_filter = preview_cols[3].radio("Legendary", ["All", "Legendary", "Non-Legendary"], horizontal=False, key="global_legendary")

        preview = df.loc[df["type1"].isin(selected_types) & df["generation"].isin(selected_generations)].copy()
        if legendary_filter == "Legendary":
            preview = preview.loc[preview["is_legendary"] == 1]
        elif legendary_filter == "Non-Legendary":
            preview = preview.loc[preview["is_legendary"] == 0]
        if search:
            mask = preview.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            preview = preview.loc[mask]

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


def render_dual_section(df: pd.DataFrame) -> None:
    st.header(SECTION_DUAL)
    st.markdown(
        "This section follows the notebook's dual-type analysis: type-pair frequency, average base stat strength, "
        "ability profiles, power tiers, and legendary ratios."
    )
    _, dual_type_stats, true_dual_stats, frequency_matrix = prepare_dual_analysis(df)

    control_col1, control_col2, control_col3 = st.columns(3)
    selected_types = control_col1.multiselect(
        "Types in heatmap",
        list(TYPE_COLORS.keys()),
        default=list(TYPE_COLORS.keys()),
        key="dual_heatmap_types",
    )
    top_n = control_col2.slider("Top combinations", 5, 25, 15, key="dual_top_n")
    min_count = control_col3.slider("Minimum Pokémon per combination", 1, int(true_dual_stats["count"].max()), 1, key="dual_min_count")
    if not selected_types:
        selected_types = list(TYPE_COLORS.keys())

    with st.container(border=True):
        render_chart(
            fig_dual_frequency_heatmap(frequency_matrix, selected_types),
            "This heatmap shows counts for primary and secondary type pairs. It uses `type1` and cleaned `type2` values, helping identify common, rare, and absent dual-type combinations.",
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_top_dual_bst(true_dual_stats, top_n, min_count),
                "This chart ranks true dual-type combinations by average `base_total`. It uses the minimum-count control to reduce noise from very rare pairings.",
            )
    with right:
        default_combos = [combo for combo in ["normal + flying", "bug + flying", "grass + poison", "water + flying"] if combo in set(true_dual_stats["dual_type"])]
        selected_combos = st.multiselect(
            "Dual combinations in radar chart",
            sorted(true_dual_stats["dual_type"].unique()),
            default=default_combos,
            key="dual_radar_combos",
        )
        with st.container(border=True):
            render_chart(
                fig_popular_dual_radar(true_dual_stats, selected_combos),
                "This radar chart compares six average battle stats for selected dual-type combinations. It shows whether each pairing leans toward offense, defense, speed, or balanced stats.",
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_dual_tier_distribution(true_dual_stats),
                "This chart groups dual-type combinations into power tiers using average base stat total. It shows how many pairings fall into basic, mid, strong, or top-tier ranges.",
            )
    with right:
        legendary_top_n = st.slider("Legendary-ratio ranking size", 5, 15, 10, key="dual_legendary_top")
        with st.container(border=True):
            render_chart(
                fig_legendary_dual_ratio(true_dual_stats, legendary_top_n),
                "This chart ranks dual-type combinations that contain legendary Pokémon by legendary ratio. It uses `legendary_count`, total count, and the derived ratio.",
            )

    with st.expander("Dual type statistics table"):
        searchable_dataframe(dual_type_stats.sort_values("avg_bst", ascending=False), "dual_stats", 50)


def render_profile_section(df: pd.DataFrame) -> None:
    st.header(SECTION_PROFILE)
    st.markdown(
        "This section follows the updated `pokemon_project(1).py` personality-profile logic. Primary and secondary "
        "type appearances are both counted, and each type is compared with the global average stat profile."
    )
    type_deviations, full_stats, selected_types_all = prepare_personality_profiles(df)

    controls = st.columns([2, 1])
    selected_types = controls[0].multiselect(
        "Types to display",
        selected_types_all,
        default=selected_types_all,
        key="profile_types",
    )
    single_type = controls[1].selectbox("Single type profile", selected_types_all, index=selected_types_all.index("dragon") if "dragon" in selected_types_all else 0)
    if not selected_types:
        selected_types = selected_types_all

    with st.container(border=True):
        render_chart(
            fig_all_type_radar(type_deviations, full_stats, selected_types),
            "This radar chart shows each selected type's deviation from the global average for HP, Attack, Defense, Special Attack, Special Defense, and Speed. Positive values mean above-average stats.",
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_bst_by_type(full_stats, selected_types),
                "This bar chart ranks selected types by average base stat total. It uses all primary and secondary appearances, matching the updated personality-profile calculation.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_single_type_profile(type_deviations, single_type),
                "This selected-type radar chart isolates one type's six stat deviations from the global average. It highlights the type's strongest and weakest stat dimensions.",
            )

    with st.expander("Type personality statistics table"):
        searchable_dataframe(full_stats.sort_values("average_bst", ascending=False), "profile_stats", 50)


def render_capture_section(df: pd.DataFrame) -> None:
    st.header(SECTION_CAPTURE)
    st.markdown(
        "This section follows the notebook's capture analysis. Capture rates are converted into a difficulty coefficient "
        "where 0 is easiest and 1 is hardest."
    )
    capture_df, type_difficulty, generation_difficulty, legendary_summary, top_hardest, top_easiest = prepare_capture_analysis(df)

    controls = st.columns(4)
    bins = controls[0].slider("Histogram bins", 10, 50, 20, key="capture_bins")
    type_options = sorted(capture_df["type1"].dropna().unique())
    selected_types = controls[1].multiselect("Primary types", type_options, default=type_options, key="capture_types")
    generations = sorted(capture_df["generation"].dropna().unique())
    selected_generations = controls[2].multiselect("Generations", generations, default=generations, key="capture_generations")
    legendary_choice = controls[3].radio("Scatter filter", ["All", "Legendary", "Non-Legendary"], key="capture_legendary")
    if not selected_types:
        selected_types = type_options
    if not selected_generations:
        selected_generations = generations

    filtered_capture = capture_df.loc[
        capture_df["type1"].isin(selected_types) & capture_df["generation"].isin(selected_generations)
    ].copy()
    if legendary_choice == "Legendary":
        filtered_capture = filtered_capture.loc[filtered_capture["is_legendary"] == 1]
    elif legendary_choice == "Non-Legendary":
        filtered_capture = filtered_capture.loc[filtered_capture["is_legendary"] == 0]

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_capture_distribution(filtered_capture, bins),
                "This histogram shows the distribution of the derived capture difficulty coefficient. It uses parsed `capture_rate` values transformed so lower capture rates become harder.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_capture_by_type(type_difficulty, selected_types),
                "This chart averages capture difficulty by primary type. It uses `type1`, parsed capture rate, and the derived difficulty coefficient to compare type-level catch difficulty.",
            )

    with st.container(border=True):
        render_chart(
            fig_difficulty_vs_bst(filtered_capture),
            "This scatter plot compares capture difficulty with `base_total`. Hover details include Pokémon name, type, generation, and capture rate; the dashed trend line shows the overall relationship.",
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_capture_box(capture_df),
                "This box plot compares capture difficulty for legendary and non-legendary Pokémon. It shows whether legendary Pokémon are generally harder to catch.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_capture_by_generation(generation_difficulty),
                "This chart averages capture difficulty by generation. It uses `generation` and the derived coefficient to compare era-level capture patterns.",
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_top_capture_rank(top_hardest, "Top 15 Hardest Pokémon: BST within the Minimum Capture Rate Tier", "#4F6F82"),
                "This chart displays the hardest-to-catch Pokémon. Because many share the minimum capture rate, base stat total is used to distinguish them within that hardest tier.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_top_capture_rank(top_easiest, "Top 15 Easiest Pokémon: BST within the Maximum Capture Rate Tier", "#6E7469"),
                "This chart displays the easiest-to-catch Pokémon. Because many share the maximum capture rate, base stat total distinguishes them within that easiest tier.",
            )

    with st.expander("Capture summary tables"):
        tab1, tab2, tab3 = st.tabs(["Hardest", "Easiest", "Legendary Summary"])
        with tab1:
            searchable_dataframe(top_hardest, "capture_hardest", 15)
        with tab2:
            searchable_dataframe(top_easiest, "capture_easiest", 15)
        with tab3:
            st.dataframe(legendary_summary, use_container_width=True, hide_index=True)


def render_ml_section(df: pd.DataFrame) -> None:
    st.header(SECTION_ML)
    st.markdown(
        "The machine learning goal is to predict whether a Pokémon is legendary. The model follows the notebook: "
        "numeric features are standardized, `type1` and `generation` are one-hot encoded, and `SGDClassifier` with "
        "logistic loss is trained with class balancing."
    )

    test_size = st.slider("Test set size", 0.20, 0.40, 0.30, 0.05, help="0.30 matches the original notebook split.")
    result = train_legendary_model(df, test_size)

    feature_text = ", ".join(result["numeric_features"] + result["categorical_features"])
    st.markdown(f"**Selected features:** {feature_text}")
    st.markdown(
        f"**Train/test split:** {int((1 - test_size) * 100)}% training and {int(test_size * 100)}% testing, "
        "with stratification to preserve the legendary/non-legendary class ratio."
    )

    metric_cols = st.columns(4)
    for col, (_, row) in zip(metric_cols, result["metrics"].iterrows()):
        col.metric(row["Metric"], f"{row['Score']:.3f}")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_legendary_class_distribution(result["pokemon_model"]),
                "This chart shows the original legendary class distribution. It uses `is_legendary` and reveals the strong imbalance between legendary and non-legendary Pokémon.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_train_test_split(result["x_train"], result["x_test"]),
                "This chart confirms the train-test split size. It uses the row counts after model preprocessing and keeps the default notebook split at 70/30.",
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_train_test_class_distribution(result["y_train"], result["y_test"]),
                "This chart compares class proportions in the training and testing sets. It verifies that stratified splitting preserved similar legendary proportions.",
            )
    with right:
        with st.container(border=True):
            render_chart(
                fig_confusion_matrix(result["cm"]),
                "This confusion matrix shows correct and incorrect predictions for both classes. It uses the model's predictions on the held-out test set.",
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_chart(
                fig_model_metrics(result["metrics"]),
                "This bar chart summarizes accuracy, balanced accuracy, recall, and precision. Balanced accuracy and recall are especially useful because legendary Pokémon are rare.",
            )
    with right:
        coef_top_n = st.slider("Number of coefficients", 8, 25, 15, key="coef_top_n")
        with st.container(border=True):
            render_chart(
                fig_top_coefficients(result["coef_df"], coef_top_n),
                "This chart ranks model coefficients by absolute value. Positive coefficients push predictions toward legendary status; negative coefficients push away from it.",
            )

    best_metric = result["metrics"].sort_values("Score", ascending=False).iloc[0]
    st.info(
        f"Interpretation: the model's strongest metric here is {best_metric['Metric']} "
        f"({best_metric['Score']:.3f}). Review the confusion matrix together with recall and precision because the "
        "legendary class is much smaller than the non-legendary class."
    )

    with st.expander("Predicted legendary probabilities"):
        predictions = result["predictions"].copy()
        predictions["Actual"] = predictions["Actual"].map({0: "Non-Legendary", 1: "Legendary"})
        predictions["Predicted"] = predictions["Predicted"].map({0: "Non-Legendary", 1: "Legendary"})
        searchable_dataframe(predictions, "ml_predictions", 50)


def main() -> None:
    df = load_dataset()
    render_project_header(df)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose section",
        [SECTION_DUAL, SECTION_PROFILE, SECTION_CAPTURE, SECTION_ML],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        **Dataset**

        `{DATA_PATH.name}`

        <span class="small-muted">Loaded from the project directory only.</span>
        """,
        unsafe_allow_html=True,
    )

    if section == SECTION_DUAL:
        render_dual_section(df)
    elif section == SECTION_PROFILE:
        render_profile_section(df)
    elif section == SECTION_CAPTURE:
        render_capture_section(df)
    else:
        render_ml_section(df)


if __name__ == "__main__":
    main()
