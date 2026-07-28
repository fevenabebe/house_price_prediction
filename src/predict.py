"""
Prediction utilities for House Price Prediction web application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import engineer_features
from src.utils import MODELS_DIR



# ============================================================
# MODEL LOADING
# ============================================================


def load_model(
    model_path: Path | None = None
) -> dict[str, Any]:
    """
    Load trained regression model bundle.
    """

    path = (
        model_path
        if model_path
        else MODELS_DIR / "best_model.pkl"
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Model not found: {path}. "
            "Run training first."
        )


    return joblib.load(path)



# ============================================================
# FEATURE INFORMATION
# ============================================================


def load_feature_info() -> dict[str, Any]:
    """
    Load feature metadata saved during training.
    """

    path = (
        MODELS_DIR /
        "feature_info.json"
    )


    if not path.exists():

        raise FileNotFoundError(
            "feature_info.json not found. "
            "Run training first."
        )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)



# ============================================================
# INPUT PREPARATION
# ============================================================


def prepare_input(
    raw_input: dict[str, Any]
) -> pd.DataFrame:
    """
    Prepare one house input for prediction.

    Steps:
    1. Convert dictionary to dataframe
    2. Apply feature engineering
    3. Match training features
    """


    df = pd.DataFrame(
        [raw_input]
    )


    # Apply same feature engineering
    # used during training

    df = engineer_features(
        df
    )


    feature_info = load_feature_info()


    required_features = (
        feature_info["all_features"]
    )


    numerical_features = (
        feature_info["numerical_features"]
    )



    # Add missing columns

    for col in required_features:

        if col not in df.columns:

            if col in numerical_features:

                df[col] = 0

            else:

                df[col] = "None"



    # Keep same order as training

    df = df[
        required_features
    ]


    return df



# ============================================================
# SINGLE PREDICTION
# ============================================================


def predict_price(
    raw_input: dict[str, Any],
    model_path: Path | None = None
) -> float:
    """
    Predict house sale price.

    Parameters
    ----------
    raw_input:
        Dictionary containing house features.

    Returns
    -------
    float:
        Predicted SalePrice.
    """


    bundle = load_model(
        model_path
    )


    model = bundle["model"]


    X = prepare_input(
        raw_input
    )


    prediction = model.predict(
        X
    )[0]


    # Prevent negative prices

    prediction = max(
        prediction,
        0
    )


    return float(
        prediction
    )



# ============================================================
# BATCH PREDICTION
# ============================================================


def predict_batch(
    df: pd.DataFrame,
    model_path: Path | None = None
) -> pd.Series:
    """
    Predict prices for multiple houses.
    """


    bundle = load_model(
        model_path
    )


    model = bundle["model"]



    processed = []


    for _, row in df.iterrows():

        processed.append(

            prepare_input(
                row.to_dict()
            ).iloc[0]

        )



    X = pd.DataFrame(
        processed
    )



    predictions = model.predict(
        X
    )



    return pd.Series(
        predictions,
        index=df.index,
        name="Predicted_SalePrice"
    )
if __name__ == "__main__":

    sample_house = {

        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "GrLivArea": 1710,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Fireplaces": 0,
        "GarageCars": 2,
        "GarageArea": 548,
        "GarageType": "Attchd",
        "GarageYrBlt": 2003,
        "GarageFinish": "RFn",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "SaleType": "WD",
        "SaleCondition": "Normal"
    }


    prediction = predict_price(
        sample_house
    )


    print(
        f"Predicted House Price: ${prediction:,.2f}"
    )