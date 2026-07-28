"""Tests for house price preprocessing and feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing import clean_dataframe
from src.feature_engineering import engineer_features



@pytest.fixture
def sample_house_data():

    return {

        "OverallQual": 7,

        "GrLivArea": 2000,

        "TotalBsmtSF": 1000,

        "1stFlrSF": 1200,

        "2ndFlrSF": 800,

        "YearBuilt": 2000,

        "GarageCars": 2,

        "FullBath": 2,

        "SalePrice": 250000

    }



def test_clean_dataframe(sample_house_data):

    df = pd.DataFrame(
        [sample_house_data]
    )


    cleaned = clean_dataframe(
        df
    )


    assert isinstance(
        cleaned,
        pd.DataFrame
    )


    assert "SalePrice" in cleaned.columns



def test_engineer_features(sample_house_data):

    df = pd.DataFrame(
        [sample_house_data]
    )


    cleaned = clean_dataframe(
        df
    )


    engineered = engineer_features(
        cleaned
    )


    assert isinstance(
        engineered,
        pd.DataFrame
    )


    assert "QualityLivingArea" in engineered.columns


    assert "TotalSF" in engineered.columns