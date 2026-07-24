"""Inference utilities for used car price prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import clean_dataframe
from src.utils import MODELS_DIR, TARGET_COLUMN


def load_model(model_path: Path | None = None) -> dict[str, Any]:
    """
    Load saved model bundle from disk.

    Returns dict with 'model' (Pipeline) and 'metadata'.
    """
    path = model_path or (MODELS_DIR / "best_model.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def load_feature_info() -> dict[str, Any]:
    """Load feature column metadata saved during training."""
    import json
    info_path = MODELS_DIR / "feature_info.json"
    if not info_path.exists():
        raise FileNotFoundError("Feature info not found. Run training first.")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare a single prediction input from raw feature dict.

    Applies cleaning (without dropping target) and feature engineering.
    """
    df = pd.DataFrame([raw_input])

    # Apply parsing without full clean pipeline (no target to drop)
    from src.preprocessing import (
        parse_accident,
        parse_clean_title,
        parse_mileage,
        parse_model_year,
        normalize_text,
        trim_whitespace,
    )

    if "milage" in df.columns and "mileage" not in df.columns:
        df = df.rename(columns={"milage": "mileage"})

    df = trim_whitespace(df)
    df = normalize_text(df)

    if "mileage" in df.columns:
        df["mileage"] = parse_mileage(df["mileage"])
    if "model_year" in df.columns:
        df["model_year"] = parse_model_year(df["model_year"])
    if "clean_title" in df.columns:
        df["clean_title"] = parse_clean_title(df["clean_title"])
    if "accident" in df.columns:
        df["accident"] = parse_accident(df["accident"])

    df = engineer_features(df)

    feature_info = load_feature_info()
    all_features = feature_info["all_features"]
    for col in all_features:
        if col not in df.columns:
            df[col] = 0 if col in feature_info["numeric_features"] else "unknown"

    return df[all_features]


def predict_price(raw_input: dict[str, Any], model_path: Path | None = None) -> float:
    """
    Predict used car price from raw feature dictionary.

    Parameters
    ----------
    raw_input : dict
        Feature name to value mapping (same columns as training data minus price).
    model_path : Path, optional
        Path to saved model pickle.

    Returns
    -------
    float
        Predicted price in dollars.
    """
    bundle = load_model(model_path)
    model = bundle["model"]
    X = prepare_input(raw_input)
    prediction = model.predict(X)[0]
    return float(max(prediction, 0))


def predict_batch(df: pd.DataFrame, model_path: Path | None = None) -> pd.Series:
    """Predict prices for a batch of raw inputs."""
    bundle = load_model(model_path)
    model = bundle["model"]
    feature_info = load_feature_info()
    all_features = feature_info["all_features"]

    processed_rows = []
    for _, row in df.iterrows():
        processed = prepare_input(row.to_dict())
        processed_rows.append(processed.iloc[0])

    X = pd.DataFrame(processed_rows)[all_features]
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index, name="predicted_price")
