"""Tests for the House Price Prediction FastAPI."""

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_check():
    """Test that the API health endpoint works."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }