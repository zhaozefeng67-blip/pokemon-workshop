import re

import numpy as np
import pandas as pd

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

