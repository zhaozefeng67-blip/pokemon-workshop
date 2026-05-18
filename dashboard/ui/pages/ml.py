import pandas as pd
import streamlit as st

from dashboard.analysis.ml import load_ml_results
from dashboard.plots.ml import (
    fig_confusion_matrix,
    fig_legendary_class_distribution,
    fig_model_metrics,
    fig_top_coefficients,
    fig_train_test_class_distribution,
    fig_train_test_split,
)
from dashboard.ui.widgets import plot_chart, render_chart_expander, searchable_dataframe

def render_ml_section(df: pd.DataFrame) -> None:
    result = load_ml_results(df)
    metrics = result["metrics"]

    metric_cols = st.columns(4)
    for col, (_, row) in zip(metric_cols, metrics.iterrows()):
        col.metric(row["Metric"], f"{row['Score']:.3f}")

    eval_left, eval_right = st.columns(2, gap="medium")
    with eval_left:
        with render_chart_expander("Confusion matrix", expanded=True):
            plot_chart(fig_confusion_matrix(result["cm"]))
    with eval_right:
        with render_chart_expander("Performance summary", expanded=True):
            plot_chart(fig_model_metrics(metrics))

    with render_chart_expander("Dataset & training split"):
        tab_balance, tab_split, tab_split_balance = st.tabs(
            ["Class balance (full data)", "Train vs test size", "Balance in each split"]
        )
        with tab_balance:
            plot_chart(fig_legendary_class_distribution(result["pokemon_model"]))
        with tab_split:
            plot_chart(fig_train_test_split(result["x_train"], result["x_test"]))
        with tab_split_balance:
            plot_chart(fig_train_test_class_distribution(result["y_train"], result["y_test"]))

    with render_chart_expander("Important features"):
        coef_top_n = st.slider("Show top coefficients", 8, 20, 12, key="coef_top_n")
        plot_chart(fig_top_coefficients(result["coef_df"], coef_top_n))

    with st.expander("Test set predictions"):
        predictions = result["predictions"].copy()
        predictions["Actual"] = predictions["Actual"].map({0: "Non-Legendary", 1: "Legendary"})
        predictions["Predicted"] = predictions["Predicted"].map({0: "Non-Legendary", 1: "Legendary"})
        searchable_dataframe(predictions, "ml_predictions", 50)
