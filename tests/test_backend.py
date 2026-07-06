"""
tests/test_backend.py

Integration tests for the headless FastAPI backend.
"""

from fastapi.testclient import TestClient
import pytest
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Checks that the health endpoint returns a successful response and valid JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "onnx_loaded" in data
    assert "pytorch_loaded" in data
