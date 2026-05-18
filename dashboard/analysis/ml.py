import pickle

import pandas as pd
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

from dashboard.config import ARTIFACTS_DIR, ML_ARTIFACT_PATH, ML_TEST_SIZE
from dashboard.data.cleaning import parse_capture_rate

def _train_legendary_model_impl(df: pd.DataFrame, test_size: float = 0.30):
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
        "test_size": test_size,
    }


@st.cache_data
def train_legendary_model(df: pd.DataFrame, test_size: float = 0.30):
    return _train_legendary_model_impl(df, test_size)


@st.cache_data
def load_ml_results(_df: pd.DataFrame) -> dict:
    if ML_ARTIFACT_PATH.exists():
        with open(ML_ARTIFACT_PATH, "rb") as artifact_file:
            return pickle.load(artifact_file)
    result = _train_legendary_model_impl(_df, ML_TEST_SIZE)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ML_ARTIFACT_PATH, "wb") as artifact_file:
        pickle.dump(result, artifact_file)
    return result
