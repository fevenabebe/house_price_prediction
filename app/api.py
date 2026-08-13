"""
FastAPI service for House Price Prediction.
"""
import logging
from typing import Any
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_price


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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

    logger.info("Health check requested")

    return {
        "status": "healthy"
    }

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(house: HouseInput):
    start_time = time.perf_counter()

    logger.info(
        "Prediction request received for neighborhood=%s",
        house.Neighborhood,
    )

    try:

        input_data: dict[str, Any] = house.model_dump()

        prediction = predict_price(
            input_data
        )


        duration = time.perf_counter() - start_time

        logger.info(
            "Prediction completed successfully: %.2f USD | duration=%.2fs",
            prediction,
            duration,
        )

        return {
            "predicted_price": round(prediction, 2),
            "currency": "USD",
        }

    except Exception as e:

        duration = time.perf_counter() - start_time

        logger.exception(
            "Prediction failed | duration=%.2fs",
            duration,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )