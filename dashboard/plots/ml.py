import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import CHART_HEIGHT_HALF
from dashboard.plots.style import clean_plotly_layout

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
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


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
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


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
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


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
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


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
    return clean_plotly_layout(fig, height=CHART_HEIGHT_HALF)


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
    return clean_plotly_layout(fig, height=max(CHART_HEIGHT_HALF, top_n * 30))
