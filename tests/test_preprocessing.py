"""Basic tests for preprocessing and feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import engineer_features
from src.preprocessing import clean_dataframe, parse_mileage, parse_price


@pytest.fixture
def sample_raw_row() -> dict:
    return {
        "brand": "Ford",
        "model": "F-150 XLT",
        "model_year": 2020,
        "milage": "25,000 mi.",
        "fuel_type": "gasoline",
        "engine": "3.5L V6",
        "transmission": "automatic",
        "ext_col": "black",
        "int_col": "gray",
        "accident": "none reported",
        "clean_title": "yes",
        "price": "$35,000",
    }


def test_parse_price():
    series = pd.Series(["$10,300", "$1,234,567", "invalid"])
    result = parse_price(series)
    assert result.iloc[0] == 10300
    assert pd.isna(result.iloc[2])


def test_parse_mileage():
    series = pd.Series(["51,000 mi.", "1234", ""])
    result = parse_mileage(series)
    assert result.iloc[0] == 51000
    assert result.iloc[1] == 1234


def test_clean_dataframe(sample_raw_row):
    df = pd.DataFrame([sample_raw_row])
    cleaned = clean_dataframe(df)
    assert "mileage" in cleaned.columns
    assert cleaned["price"].iloc[0] == 35000
    assert cleaned["mileage"].iloc[0] == 25000
    assert cleaned["clean_title"].iloc[0] == 1


def test_engineer_features(sample_raw_row):
    df = pd.DataFrame([sample_raw_row])
    cleaned = clean_dataframe(df)
    engineered = engineer_features(cleaned)
    assert "car_age" in engineered.columns
    assert "mileage_per_year" in engineered.columns
    assert "is_luxury_brand" in engineered.columns
    assert engineered["car_age"].iloc[0] >= 0
