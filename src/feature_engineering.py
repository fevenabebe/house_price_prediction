"""Feature engineering for Ames Housing price prediction."""

from __future__ import annotations

import pandas as pd



def add_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total square footage feature.

    Combines:
    - Total basement area
    - First floor area
    - Second floor area
    """

    engineered = df.copy()

    required_cols = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF"
    ]

    if all(col in engineered.columns for col in required_cols):

        engineered["TotalSF"] = (
            engineered["TotalBsmtSF"]
            + engineered["1stFlrSF"]
            + engineered["2ndFlrSF"]
        )

    return engineered



def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weighted total bathrooms.

    Full bathrooms have higher contribution than half bathrooms.
    """

    engineered = df.copy()

    required_cols = [
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath"
    ]

    if all(col in engineered.columns for col in required_cols):

        engineered["TotalBathrooms"] = (
            engineered["FullBath"]
            + 0.5 * engineered["HalfBath"]
            + engineered["BsmtFullBath"]
            + 0.5 * engineered["BsmtHalfBath"]
        )

    return engineered



def add_house_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add house age at time of sale.

    Older houses may have different pricing patterns.
    """

    engineered = df.copy()

    if "YearBuilt" in engineered.columns and "YrSold" in engineered.columns:

        engineered["HouseAge"] = (
            engineered["YrSold"]
            - engineered["YearBuilt"]
        )

    return engineered



def add_year_since_remodel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add years since last remodeling.
    """

    engineered = df.copy()

    if "YearRemodAdd" in engineered.columns and "YrSold" in engineered.columns:

        engineered["YearsSinceRemodel"] = (
            engineered["YrSold"]
            - engineered["YearRemodAdd"]
        )

    return engineered



def add_total_porch_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine outdoor living areas.
    """

    engineered = df.copy()

    porch_columns = [
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF"
    ]

    available_cols = [
        col for col in porch_columns
        if col in engineered.columns
    ]

    if available_cols:

        engineered["TotalPorchSF"] = (
            engineered[available_cols]
            .sum(axis=1)
        )

    return engineered



def add_total_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine important quality-related ordinal features.

    Uses:
    - OverallQual
    - OverallCond
    """

    engineered = df.copy()

    if (
        "OverallQual" in engineered.columns
        and "OverallCond" in engineered.columns
    ):

        engineered["OverallScore"] = (
            engineered["OverallQual"]
            * engineered["OverallCond"]
        )

    return engineered



def add_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine bedrooms and total rooms.
    """

    engineered = df.copy()

    if (
        "TotRmsAbvGrd" in engineered.columns
        and "BedroomAbvGr" in engineered.columns
    ):

        engineered["TotalRooms"] = (
            engineered["TotRmsAbvGrd"]
            + engineered["BedroomAbvGr"]
        )

    return engineered



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Features are created only from input variables
    to avoid target leakage.
    """

    engineered = df.copy()

    engineered = add_total_sf(engineered)

    engineered = add_total_bathrooms(engineered)

    engineered = add_house_age(engineered)

    engineered = add_year_since_remodel(engineered)

    engineered = add_total_porch_area(engineered)

    engineered = add_total_quality_score(engineered)

    engineered = add_total_rooms(engineered)

    return engineered