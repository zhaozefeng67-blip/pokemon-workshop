import pandas as pd
import streamlit as st

from dashboard.config import DATA_PATH

@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)
