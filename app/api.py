"""
FastAPI service for House Price Prediction.
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_price


# ============================================================
# CREATE API
# ============================================================

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using the Ames Housing dataset.",
    version="1.0.0",
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class HouseInput(BaseModel):
    Neighborhood: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    GrLivArea: float
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    FullBath: int
    GarageCars: int
    GarageArea: float
    KitchenQual: str
    GarageType: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running",
        "model": "CatBoost",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(house: HouseInput):

    try:

        input_data: dict[str, Any] = house.model_dump()

        prediction = predict_price(
            input_data
        )

        return {
            "predicted_price": round(prediction, 2),
            "currency": "USD",
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )