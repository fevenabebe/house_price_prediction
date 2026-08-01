"""
Data cleaning and preprocessing functions for Ames Housing dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import TARGET_COLUMN


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataframe.
    
    """

    return df.drop_duplicates().reset_index(drop=True)



def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme housing outliers identified during EDA.

    Removes properties with:
    - GrLivArea > 4000 sq ft
    - SalePrice < 300000

    These observations represent abnormal size-price
    relationships and negatively affect regression models.
    """

    cleaned = df.copy()

    if (
        TARGET_COLUMN in cleaned.columns
        and "GrLivArea" in cleaned.columns
    ):

        condition = (
            (cleaned["GrLivArea"] > 4000)
            &
            (cleaned[TARGET_COLUMN] < 300000)
        )

        cleaned = cleaned.loc[~condition]


    return cleaned.reset_index(drop=True)

def remove_high_missing_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove features with excessive missing values.

    Features removed:
    - Alley
    - PoolQC
    - Fence
    - MiscFeature

    These features contain more than 80% missing values
    and provide limited predictive value.
    """

    cleaned = df.copy()

    columns_to_remove = [
        "Alley",
        "PoolQC",
        "Fence",
        "MiscFeature"
    ]

    cleaned = cleaned.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    return cleaned

def remove_low_variance_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove categorical features with extremely low variation.

    Removed:
    - Street
    - Utilities
    - Condition2
    - RoofMatl

    These features contain very few minority examples,
    making them unreliable for model learning.
    """

    cleaned = df.copy()


    columns_to_remove = [
        "Street",
        "Utilities",
        "Condition2",
        "RoofMatl"
    ]


    cleaned = cleaned.drop(
        columns=[
            col
            for col in columns_to_remove
            if col in cleaned.columns
        ]
    )


    return cleaned



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using Ames Housing domain knowledge.

    Strategy:
    - Remove rows with missing target values.
    - Replace missing categorical features where absence
      represents no existing feature with 'None'.
    - Fill numerical missing values using median.
    - Fill remaining categorical missing values with 'None'.
    """

    cleaned = df.copy()


    # --------------------------------------------------------
    # Remove missing target values
    # --------------------------------------------------------

    if TARGET_COLUMN in cleaned.columns:

        cleaned = cleaned.dropna(
            subset=[TARGET_COLUMN]
        )


    # --------------------------------------------------------
    # Missing categorical values meaning "feature absent"
    # --------------------------------------------------------

    none_columns = [
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond"]   


    for col in none_columns:

        if col in cleaned.columns:

            cleaned[col] = cleaned[col].fillna("None")



    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    median_columns = [
        "LotFrontage",
        "MasVnrArea",
        "GarageYrBlt"
    ]


    for col in median_columns:

        if col in cleaned.columns:

            cleaned[col] = cleaned[col].fillna(
                cleaned[col].median()
            )



    # --------------------------------------------------------
    # Remaining categorical missing values
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Electrical
    # --------------------------------------------------------

    if "Electrical" in cleaned.columns:

        cleaned["Electrical"] = cleaned["Electrical"].fillna(
            cleaned["Electrical"].mode()[0]
        )

    categorical_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns


    for col in categorical_columns:

        cleaned[col] = cleaned[col].fillna("None")



    # --------------------------------------------------------
    # Remaining numerical missing values
    # --------------------------------------------------------

    numerical_columns = cleaned.select_dtypes(
        include=[np.number]
    ).columns


    for col in numerical_columns:

        if col != TARGET_COLUMN:

            cleaned[col] = cleaned[col].fillna(
                cleaned[col].median()
            )


    return cleaned



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete dataframe cleaning pipeline.

    Steps:
    1. Remove extreme outliers.
    2. Remove low-information categorical features.
    3. Handle missing values.
    4. Remove duplicates.
    """

    cleaned = df.copy()


    cleaned = remove_high_missing_features(
        cleaned
    )

    cleaned = remove_low_variance_features(
        cleaned
    )

    cleaned = remove_outliers(
        cleaned
    )

    cleaned = handle_missing_values(
        cleaned
    )

    cleaned = remove_duplicates(
        cleaned
    )


    return cleaned.reset_index(drop=True)



# ============================================================
# FEATURE TYPE DETECTION
# ============================================================


def get_feature_columns(
    df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Separate numerical and categorical features.

    Excludes:
    - Target variable SalePrice
    - Id column because it is only an identifier.
    """

    feature_columns = [
        col
        for col in df.columns
        if col not in [TARGET_COLUMN, "Id"]
    ]


    numerical_features = (
        df[feature_columns]
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )


    categorical_features = [
        col
        for col in feature_columns
        if col not in numerical_features
    ]


    return numerical_features, categorical_features



# ============================================================
# SKLEARN PREPROCESSING PIPELINE
# ============================================================


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    Numerical:
    - Median imputation
    - Optional StandardScaler

    Categorical:
    - Most frequent imputation
    - One-hot encoding

    Scaling is used for:
    - Linear Regression
    - Ridge
    - Lasso
    - SVR

    Tree-based models do not require scaling.
    """

    numerical_steps = [

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]


    if scale_numeric:

        numerical_steps.append(

            (
                "scaler",
                StandardScaler()
            )

        )


    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )


    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]

    )


    transformers = []


    if numerical_features:

        transformers.append(

            (
                "num",
                numerical_pipeline,
                numerical_features
            )

        )


    if categorical_features:

        transformers.append(

            (
                "cat",
                categorical_pipeline,
                categorical_features
            )

        )


    return ColumnTransformer(

        transformers=transformers,

        remainder="drop"

    )