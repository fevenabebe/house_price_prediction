"""Feature engineering for used car price prediction."""

from __future__ import annotations

import pandas as pd

from src.utils import CURRENT_YEAR, LUXURY_BRANDS, PREMIUM_FUEL_TYPES


def add_car_age(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Add car_age = current_year - model_year."""
    engineered = df.copy()
    if "model_year" in engineered.columns:
        engineered["car_age"] = current_year - engineered["model_year"]
        engineered["car_age"] = engineered["car_age"].clip(lower=0)
    return engineered


def add_vehicle_age_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize vehicles by age.

    Categories: new (0-3), recent (4-7), mature (8-12), old (13+).
    """
    engineered = df.copy()
    if "car_age" not in engineered.columns:
        engineered = add_car_age(engineered)

    bins = [-1, 3, 7, 12, 100]
    labels = ["new", "recent", "mature", "old"]
    engineered["vehicle_age_category"] = pd.cut(
        engineered["car_age"], bins=bins, labels=labels
    ).astype(str)
    return engineered


def add_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add mileage_per_year = mileage / max(car_age, 1)."""
    engineered = df.copy()
    if "car_age" not in engineered.columns:
        engineered = add_car_age(engineered)
    if "mileage" in engineered.columns:
        age_denominator = engineered["car_age"].clip(lower=1)
        engineered["mileage_per_year"] = engineered["mileage"] / age_denominator
    return engineered


def add_luxury_brand_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary is_luxury_brand based on known luxury manufacturers."""
    engineered = df.copy()
    if "brand" in engineered.columns:
        brand_lower = engineered["brand"].astype(str).str.lower()
        luxury_lower = {b.lower() for b in LUXURY_BRANDS}
        engineered["is_luxury_brand"] = brand_lower.isin(luxury_lower).astype(int)
    return engineered


def add_premium_fuel_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary is_premium_fuel for premium/hybrid/electric fuel types."""
    engineered = df.copy()
    if "fuel_type" in engineered.columns:
        fuel_lower = engineered["fuel_type"].astype(str).str.lower().str.strip()
        engineered["is_premium_fuel"] = fuel_lower.isin(PREMIUM_FUEL_TYPES).astype(int)
        # Also flag hybrid/electric keywords not in exact set
        keyword_match = fuel_lower.str.contains(
            r"hybrid|electric|diesel|premium", na=False, regex=True
        )
        engineered.loc[keyword_match, "is_premium_fuel"] = 1
    return engineered


def engineer_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """
    Apply all feature engineering steps without target leakage.

    Only uses input features available at prediction time.
    """
    engineered = df.copy()
    engineered = add_car_age(engineered, current_year=current_year)
    engineered = add_vehicle_age_category(engineered)
    engineered = add_mileage_per_year(engineered)
    engineered = add_luxury_brand_flag(engineered)
    engineered = add_premium_fuel_flag(engineered)
    return engineered
