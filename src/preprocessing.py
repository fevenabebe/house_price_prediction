"""Data cleaning and preprocessing functions."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import TARGET_COLUMN


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from object columns."""
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return cleaned


def normalize_text(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Normalize text columns to lowercase and collapse internal whitespace."""
    cleaned = df.copy()
    cols = columns or cleaned.select_dtypes(include=["object", "string"]).columns
    for col in cols:
        if col in cleaned.columns:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .replace({"nan": np.nan})
            )
    return cleaned


def parse_price(series: pd.Series) -> pd.Series:
    """
    Convert price strings like '$10,300' into numeric values.

    Invalid or missing values become NaN.
    """
    numeric = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(numeric, errors="coerce")


def parse_mileage(series: pd.Series) -> pd.Series:
    """
    Convert mileage strings like '51,000 mi.' into numeric miles.

    Also handles column name typo 'milage'.
    """
    numeric = (
        series.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(numeric, errors="coerce")


def parse_model_year(series: pd.Series) -> pd.Series:
    """Convert model year to integer."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def parse_clean_title(series: pd.Series) -> pd.Series:
    """
    Convert clean_title to binary (1 = yes, 0 = no/unknown).

    Accepts 'yes', 'y', '1', 'true' as positive.
    """
    normalized = series.astype(str).str.lower().str.strip()
    positive = {"yes", "y", "1", "true"}
    result = normalized.isin(positive).astype(int)
    result = result.where(series.notna(), np.nan)
    return result


def parse_accident(series: pd.Series) -> pd.Series:
    """
    Convert accident history to binary.

    1 = at least one accident reported, 0 = none reported or clean.
    """
    normalized = series.astype(str).str.lower().str.strip()
    has_accident = normalized.str.contains("at least", na=False) | normalized.str.contains(
        "accident", na=False
    ) & ~normalized.str.contains("none", na=False)
    no_accident = normalized.str.contains("none reported", na=False)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[has_accident] = 1.0
    result.loc[no_accident] = 0.0
    result.loc[series.isna() | (normalized == "nan")] = np.nan
    return result


def detect_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag and nullify invalid numeric values.

    Rules:
    - price <= 0 or > 500_000
    - mileage < 0 or > 500_000
    - model_year < 1980 or > current year + 1
    """
    cleaned = df.copy()
    current_year = pd.Timestamp.now().year + 1

    if TARGET_COLUMN in cleaned.columns:
        invalid_price = (cleaned[TARGET_COLUMN] <= 0) | (cleaned[TARGET_COLUMN] > 500_000)
        cleaned.loc[invalid_price, TARGET_COLUMN] = np.nan

    mileage_col = "mileage" if "mileage" in cleaned.columns else "milage"
    if mileage_col in cleaned.columns:
        invalid_mileage = (cleaned[mileage_col] < 0) | (cleaned[mileage_col] > 500_000)
        cleaned.loc[invalid_mileage, mileage_col] = np.nan

    if "model_year" in cleaned.columns:
        invalid_year = (cleaned["model_year"] < 1980) | (cleaned["model_year"] > current_year)
        cleaned.loc[invalid_year, "model_year"] = np.nan

    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with domain-aware imputation.

    Numeric: median imputation for model_year, mileage; drop rows with missing target.
    Categorical: fill with 'unknown'.
    """
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    for col in numeric_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical_cols:
        cleaned[col] = cleaned[col].fillna("unknown")

    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline returning a cleaned dataframe.

    Steps: trim, normalize, parse fields, detect invalids, handle missing, dedupe.
    """
    cleaned = df.copy()

    # Standardize mileage column name
    if "milage" in cleaned.columns and "mileage" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"milage": "mileage"})

    cleaned = trim_whitespace(cleaned)
    cleaned = normalize_text(cleaned)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = parse_price(cleaned[TARGET_COLUMN])

    if "mileage" in cleaned.columns:
        cleaned["mileage"] = parse_mileage(cleaned["mileage"])

    if "model_year" in cleaned.columns:
        cleaned["model_year"] = parse_model_year(cleaned["model_year"])

    if "clean_title" in cleaned.columns:
        cleaned["clean_title"] = parse_clean_title(cleaned["clean_title"])

    if "accident" in cleaned.columns:
        cleaned["accident"] = parse_accident(cleaned["accident"])

    cleaned = detect_invalid_values(cleaned)
    cleaned = handle_missing_values(cleaned)
    cleaned = remove_duplicates(cleaned)

    return cleaned


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split dataframe columns into numeric and categorical feature lists.

    Excludes target column.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        c for c in feature_cols if c not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer with imputation, scaling, and one-hot encoding.

    Parameters
    ----------
    numeric_features : list[str]
        Numeric column names.
    categorical_features : list[str]
        Categorical column names.
    scale_numeric : bool
        If True, apply StandardScaler to numeric features (for linear/SVR models).
        Tree models should set this to False.

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessing transformer.
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
