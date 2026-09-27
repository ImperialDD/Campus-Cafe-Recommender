"""Data loading, cleaning, and matrix creation for the Campus Cafe Recommender.

This module handles:
  - Loading the survey CSV
  - Normalizing free-response item names to canonical names
  - Marking missing values (blanks, ".", "NA", "There's nothing else")
  - Feature-mean imputation for PCA (kept separate from observed ratings)
  - Building the Student x Item rating matrix
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


# Canonical feature order used throughout the project.
FEATURES: list[str] = [
    "Misal Pav",
    "Samosa Pav",
    "Vada Pav",
    "Pav Bhaji",
    "Sandwich",
    "Sprite",
    "Diet Coke",
    "Coca Cola",
    "Thums Up",
    "Lassi",
]

# Free-response -> canonical name mapping (documented, transparent).
NAME_NORMALIZATION: dict[str, str] = {
    "vadapav": "Vada Pav",
    "wadapav": "Vada Pav",
    "pav vada": "Vada Pav",
    "misal pav": "Misal Pav",
    "missal pav": "Misal Pav",
    "samosa pav": "Samosa Pav",
    "samsoa": "Samosa",
    "samosa": "Samosa Pav",
    "thumps up": "Thums Up",
    "thumsup": "Thums Up",
    "thumbs up": "Thums Up",
    "coke": "Coca Cola",
    "coca cola": "Coca Cola",
    "coca-cola": "Coca Cola",
    "coco cola": "Coca Cola",
    "spirit": "Sprite",
    "spite": "Sprite",
    "mrinda": "Mirinda",
    "chai": "Tea",
    "sandwiches": "Sandwich",
    "chicken shaurma": "Chicken Shawarma",
    "steam momos": "Momos",
}

# Tokens that represent a missing response (not a rating of 0).
MISSING_TOKENS = {"", ".", "na", "n/a", "there's nothing else", "theres nothing else"}

DEFAULT_DATA_PATH = os.path.join("data", "survey_data.csv")


class DataError(Exception):
    """Raised when the dataset is unusable."""


def normalize_name(raw: str) -> str:
    """Map a free-response item name to its canonical form."""
    key = str(raw).strip().lower()
    if key in MISSING_TOKENS:
        return ""
    return NAME_NORMALIZATION.get(key, str(raw).strip())


def is_missing(value) -> bool:
    """Return True for blank / '.' / 'NA' / "There's nothing else"."""
    if value is None:
        return True
    s = str(value).strip()
    return s.lower() in MISSING_TOKENS


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw survey CSV into a DataFrame.

    Expects the first column to be a student identifier and the remaining
    columns to be item names. Item columns are normalized to canonical names;
    duplicate/ambiguous columns are reported but the canonical 10 are kept.
    """
    if not os.path.exists(path):
        raise DataError(
            f"Dataset file not found at '{path}'. Please place the survey CSV "
            "in the data/ folder."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        raise DataError(f"Could not read '{path}': {exc}") from exc

    if df.shape[0] < 2:
        raise DataError("Dataset must contain at least 2 student responses.")

    # First column is the respondent name.
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "Respondent"})

    # Normalize the item column headers.
    raw_cols = [c for c in df.columns if c != "Respondent"]
    canonical_cols: dict[str, str] = {}
    for raw in raw_cols:
        canonical_cols[raw] = normalize_name(raw)

    df = df.rename(columns=canonical_cols)

    # Keep only the canonical features (in fixed order).
    available = [f for f in FEATURES if f in df.columns]
    if len(available) < 2:
        raise DataError(
            "Dataset has fewer than 2 usable features after normalization. "
            f"Found: {available}"
        )
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        # We tolerate missing features but warn; for the provided dataset all 10 exist.
        for f in missing_features:
            df[f] = np.nan

    df = df[["Respondent"] + FEATURES]
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw ratings to numeric and mark missing values.

    Returns
    -------
    observed : DataFrame with NaN where ratings are missing.
    imputed  : DataFrame with feature-mean imputation applied.
    """
    ratings = df.copy()
    for col in FEATURES:
        ratings[col] = ratings[col].apply(
            lambda v: np.nan if is_missing(v) else pd.to_numeric(v, errors="coerce")
        )
        # Out-of-range ratings are treated as invalid -> missing.
        ratings[col] = ratings[col].apply(
            lambda v: v if (not np.isnan(v) and 1 <= v <= 5) else np.nan
        )

    observed = ratings.copy()

    # Feature-mean imputation using only observed ratings.
    feature_means = observed[FEATURES].mean(skipna=True)
    imputed = observed.copy()
    for col in FEATURES:
        imputed[col] = observed[col].fillna(feature_means[col])

    return observed, imputed


def create_matrix(imputed: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """Return the (n_students x n_features) numpy matrix and respondent names."""
    student_ids = imputed["Respondent"].tolist()
    matrix = imputed[FEATURES].to_numpy(dtype=float)
    return matrix, student_ids


def feature_means(imputed: pd.DataFrame) -> pd.Series:
    """Observed (pre-imputation) mean per feature, for transparency."""
    return imputed[FEATURES].mean(skipna=True)
