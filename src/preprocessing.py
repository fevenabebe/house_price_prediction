"""Data cleaning and preprocessing functions for Ames Housing dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import TARGET_COLUMN


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    """
    return df.drop_duplicates().reset_index(drop=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove domain-specific outliers identified during EDA.

    During exploratory analysis, two houses were identified with:
    - Extremely large living area (>4500 sq ft)
    - Unusually low SalePrice (<300000)

    These observations were removed because they can negatively
    influence regression models.
    """

    cleaned = df.copy()

    if (
        TARGET_COLUMN in cleaned.columns
        and "GrLivArea" in cleaned.columns
    ):

        outliers = (
            (cleaned["GrLivArea"] > 4500)
            &
            (cleaned[TARGET_COLUMN] < 300000)
        )

        cleaned = cleaned.loc[~outliers]

    return cleaned.reset_index(drop=True)



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using domain knowledge from EDA.

    Categorical:
    - Missing values in features representing absence
      are replaced with "None".

    Numerical:
    - Missing values are replaced with median values.

    Target:
    - Rows with missing SalePrice are removed.
    """

    cleaned = df.copy()


    # Remove missing target rows
    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.dropna(
            subset=[TARGET_COLUMN]
        )


    # Missing categorical values indicating absence
    none_columns = [
        "Alley",
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
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature"
    ]


    for col in none_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("None")


    # Numerical columns where median is appropriate
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


    # Remaining categorical missing values
    categorical_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns


    for col in categorical_columns:
        cleaned[col] = cleaned[col].fillna("None")


    # Remaining numerical missing values
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
    Complete preprocessing cleaning pipeline.

    Steps:
    1. Remove EDA-identified outliers
    2. Handle missing values
    3. Remove duplicate rows
    """

    cleaned = df.copy()

    cleaned = remove_outliers(cleaned)

    cleaned = handle_missing_values(cleaned)

    cleaned = remove_duplicates(cleaned)

    return cleaned



def get_feature_columns(
    df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Split dataframe columns into numerical and categorical features.

    The target column is excluded.
    """

    feature_columns = [
        col for col in df.columns
        if col != TARGET_COLUMN
    ]


    numerical_features = (
        df[feature_columns]
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )


    categorical_features = [
        col for col in feature_columns
        if col not in numerical_features
    ]


    return numerical_features, categorical_features



def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    Numerical features:
    - Median imputation
    - Optional standard scaling

    Categorical features:
    - Most frequent imputation
    - One-hot encoding

    Parameters
    ----------
    numerical_features:
        List of numerical feature names.

    categorical_features:
        List of categorical feature names.

    scale_numeric:
        True for Linear Regression/SVR models.
        False for tree-based models.
    """


    numerical_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
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