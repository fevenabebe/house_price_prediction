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



# ============================================================
# BATHROOM FEATURES
# ============================================================

def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weighted bathroom count.
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
    Calculate house age at sale.
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
    Calculate years since remodeling.
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

    available = [
        col
        for col in porch_columns
        if col in engineered.columns
    ]

    if available:

        engineered["TotalPorchSF"] = (
            engineered[available]
            .sum(axis=1)
        )

    return engineered



# ============================================================
# INTERACTION FEATURES
# ============================================================

def add_quality_living_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction between quality and living area.
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



def add_garage_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine garage size and capacity.
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
# OPTIONAL OUTLIER HANDLING
# ============================================================

def remove_extreme_area_outliers(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove known Ames Housing extreme GrLivArea outliers.

    These are very large houses sold at unusually low prices.
    """

    engineered = df.copy()

    if (
        "GrLivArea" in engineered.columns
        and "SalePrice" in engineered.columns
    ):

        engineered = engineered[
            ~(
                (engineered["GrLivArea"] > 4000)
                &
                (engineered["SalePrice"] < 300000)
            )
        ]

    return engineered



# ============================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply all feature engineering operations.

    Does not use SalePrice for feature creation.
    """

    engineered = df.copy()


    engineered = add_total_sf(
        engineered
    )


    engineered = add_total_bathrooms(
        engineered
    )


    engineered = add_house_age(
        engineered
    )


    engineered = add_year_since_remodel(
        engineered
    )


    engineered = add_total_porch_area(
        engineered
    )


    engineered = add_quality_living_area(
        engineered
    )


    engineered = add_garage_score(
        engineered
    )


    return engineered