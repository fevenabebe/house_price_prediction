"""
Feature engineering functions for Ames Housing price prediction.
"""

from __future__ import annotations

import pandas as pd



# ============================================================
# TOTAL HOUSE AREA FEATURES
# ============================================================


def add_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total house square footage.

    Combines:
    - Basement area
    - First floor area
    - Second floor area
    """

    engineered = df.copy()

    columns = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF"
    ]

    if all(col in engineered.columns for col in columns):

        engineered["TotalSF"] = (
            engineered["TotalBsmtSF"]
            +
            engineered["1stFlrSF"]
            +
            engineered["2ndFlrSF"]
        )

    return engineered



def add_total_area_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine important living spaces.

    Includes:
    - Above-ground living area
    - Basement area
    - Garage area

    Larger usable space generally increases house value.
    """

    engineered = df.copy()

    columns = [
        "GrLivArea",
        "TotalBsmtSF",
        "GarageArea"
    ]

    if all(col in engineered.columns for col in columns):

        engineered["TotalAreaScore"] = (
            engineered["GrLivArea"]
            +
            engineered["TotalBsmtSF"]
            +
            engineered["GarageArea"]
        )

    return engineered



# ============================================================
# BATHROOM FEATURES
# ============================================================


def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weighted bathroom count.

    Full bathrooms contribute more than half bathrooms.
    """

    engineered = df.copy()

    columns = [
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath"
    ]

    if all(col in engineered.columns for col in columns):

        engineered["TotalBathrooms"] = (
            engineered["FullBath"]
            +
            0.5 * engineered["HalfBath"]
            +
            engineered["BsmtFullBath"]
            +
            0.5 * engineered["BsmtHalfBath"]
        )

    return engineered



# ============================================================
# AGE FEATURES
# ============================================================


def add_house_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate house age at the time of sale.
    """

    engineered = df.copy()

    if (
        "YearBuilt" in engineered.columns
        and "YrSold" in engineered.columns
    ):

        engineered["HouseAge"] = (
            engineered["YrSold"]
            -
            engineered["YearBuilt"]
        )

    return engineered



def add_year_since_remodel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate years since last remodeling.
    """

    engineered = df.copy()

    if (
        "YearRemodAdd" in engineered.columns
        and "YrSold" in engineered.columns
    ):

        engineered["YearsSinceRemodel"] = (
            engineered["YrSold"]
            -
            engineered["YearRemodAdd"]
        )

    return engineered



# ============================================================
# OUTDOOR FEATURES
# ============================================================


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

    available_columns = [
        col
        for col in porch_columns
        if col in engineered.columns
    ]

    if available_columns:

        engineered["TotalPorchSF"] = (
            engineered[available_columns]
            .sum(axis=1)
        )

    return engineered



# ============================================================
# QUALITY FEATURES
# ============================================================


def add_total_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine overall quality and condition.

    Higher values represent better quality houses.
    """

    engineered = df.copy()

    if (
        "OverallQual" in engineered.columns
        and "OverallCond" in engineered.columns
    ):

        engineered["OverallScore"] = (
            engineered["OverallQual"]
            *
            engineered["OverallCond"]
        )

    return engineered



def add_quality_living_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture interaction between house quality
    and living area.
    """

    engineered = df.copy()

    if (
        "OverallQual" in engineered.columns
        and "GrLivArea" in engineered.columns
    ):

        engineered["QualityLivingArea"] = (
            engineered["OverallQual"]
            *
            engineered["GrLivArea"]
        )

    return engineered



# ============================================================
# GARAGE FEATURES
# ============================================================


def add_garage_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine garage capacity and size.
    """

    engineered = df.copy()

    if (
        "GarageCars" in engineered.columns
        and "GarageArea" in engineered.columns
    ):

        engineered["GarageScore"] = (
            engineered["GarageCars"]
            *
            engineered["GarageArea"]
        )

    return engineered



# ============================================================
# ROOM FEATURES
# ============================================================


def add_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine total rooms and bedrooms.
    """

    engineered = df.copy()

    if (
        "TotRmsAbvGrd" in engineered.columns
        and "BedroomAbvGr" in engineered.columns
    ):

        engineered["TotalRooms"] = (
            engineered["TotRmsAbvGrd"]
            +
            engineered["BedroomAbvGr"]
        )

    return engineered



# ============================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering operations.

    Important:
    - Uses only input features.
    - Does not use SalePrice.
    - Prevents target leakage.
    """

    engineered = df.copy()


    engineered = add_total_sf(engineered)

    engineered = add_total_area_score(engineered)

    engineered = add_total_bathrooms(engineered)

    engineered = add_house_age(engineered)

    engineered = add_year_since_remodel(engineered)

    engineered = add_total_porch_area(engineered)

    engineered = add_total_quality_score(engineered)

    engineered = add_quality_living_area(engineered)

    engineered = add_garage_score(engineered)

    engineered = add_total_rooms(engineered)


    return engineered