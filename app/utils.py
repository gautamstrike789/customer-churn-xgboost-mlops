# ============================================
# IMPORTING LIBRARIES
# ============================================

from pathlib import Path
import joblib
import pandas as pd
import numpy as np


# ============================================
# FILE PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "artifacts" / "feature_columns.joblib"


# ============================================
# LOADING SAVED ARTIFACTS
# ============================================

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


# ============================================
# FEATURE ENGINEERING FUNCTION
# ============================================

def prepare_input_data(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    # Recreate feature engineering exactly as in training
    df["AvgChargesPerMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    # Reorder columns to match training
    df = df[feature_columns]

    return df