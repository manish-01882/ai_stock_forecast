"""
Basic API tests using pytest and httpx.
Run with: pytest tests/test_api.py -v
"""
import pytest
from httpx import AsyncClient
from api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
        assert "available_models" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root landing page."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_list_models():
    """Test listing all models."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_models" in data
        assert "models" in data
        assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_predict_invalid_ticker():
    """Test prediction with invalid ticker format."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"ticker": "INVALID@#$", "days": 7}
        )
        
        # Should return 400 for invalid format
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_predict_missing_model():
    """Test prediction for ticker without trained model."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"ticker": "XYZ", "days": 7}
        )
        
        # Should return 404 if no model exists
        # or 500 if data fetching fails
        assert response.status_code in [404, 500]


@pytest.mark.asyncio
async def test_predict_validation():
    """Test request validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test invalid days (too high)
        response = await client.post(
            "/api/v1/predict",
            json={"ticker": "AAPL", "days": 1000}
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_nonexistent_model():
    """Test getting info for nonexistent model."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/models/NONEXISTENT")
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_validation():
    """Test training request validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test invalid epochs
        response = await client.post(
            "/api/v1/train",
            json={"ticker": "AAPL", "epochs": 5}  # Below minimum
        )
        
        assert response.status_code == 422  # Validation error


# Integration tests (require actual models)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_predict_with_real_model():
    """Test prediction with an actual trained model (AAPL)."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"ticker": "AAPL", "days": 7}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            assert data["ticker"] == "AAPL"
            assert "predictions" in data
            assert len(data["predictions"]) == 7
            assert "current_price" in data
            assert "confidence_score" in data
            
            # Validate prediction structure
            for pred in data["predictions"]:
                assert "date" in pred
                assert "price" in pred
                assert pred["price"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_real_model_info():
    """Test getting info for a real model."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First, get list of available models
        list_response = await client.get("/api/v1/models")
        models = list_response.json()
        
        if models["total_models"] > 0:
            ticker = models["models"][0]["ticker"]
            
            # Get specific model info
            response = await client.get(f"/api/v1/models/{ticker}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["ticker"] == ticker
            assert "metrics" in data
            assert "model_path" in data
            assert "training_date" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
