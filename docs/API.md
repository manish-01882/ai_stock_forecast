# Stock Forecast API Documentation

## Overview

The Stock Forecast API provides programmatic access to AI-powered stock price predictions using LSTM neural networks. This RESTful API enables integration with trading bots, mobile apps, dashboards, and any application that needs stock forecasts.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. In production, implement API keys or OAuth2.

## Quick Start

### 1. Start the API

```bash
# Using Docker Compose (Recommended)
docker-compose up

# Or run directly with Python
uvicorn api.main:app --reload
```

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get predictions
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'
```

### 3. Explore Interactive Docs

Navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check

**GET** `/health`

Check API health and get system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-14T12:00:00",
  "version": "1.0.0",
  "models_loaded": 8,
  "available_models": ["AAPL", "GOOGL", "MSFT", "TSLA"]
}
```

---

### Generate Predictions

**POST** `/api/v1/predict`

Generate multi-day price predictions for a stock.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "days": 7,
  "include_history": false
}
```

**Parameters:**
- `ticker` (string, required): Stock ticker symbol (e.g., "AAPL")
- `days` (integer, optional): Number of days to predict (1-90). Default: 7
- `include_history` (boolean, optional): Include historical data. Default: false

**Response:**
```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "prediction_date": "2026-01-14",
  "predictions": [
    {"date": "2026-01-15", "price": 151.30},
    {"date": "2026-01-16", "price": 152.10},
    {"date": "2026-01-17", "price": 151.80}
  ],
  "model_metrics": {
    "rmse": 2.15,
    "mae": 1.75,
    "mse": 4.62,
    "mape": 1.8
  },
  "confidence_score": 0.89
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request parameters
- `404`: No model found for ticker
- `500`: Prediction failed

---

### List All Models

**GET** `/api/v1/models`

Get a list of all trained models with metrics.

**Response:**
```json
{
  "total_models": 3,
  "models": [
    {
      "ticker": "AAPL",
      "model_name": "lstm_paper_AAPL_20260114_1200",
      "model_path": "models/lstm_paper_AAPL_20260114_1200_best.h5",
      "training_date": "2026-01-14T12:00:00",
      "metrics": {
        "rmse": 2.15,
        "mae": 1.75,
        "mse": 4.62,
        "mape": 1.8
      },
      "lookback_window": 60,
      "total_epochs": 100,
      "architecture": "Stacked LSTM (4 layers, 96 units each)"
    }
  ]
}
```

---

### Get Model Info

**GET** `/api/v1/models/{ticker}`

Get detailed information about a specific model.

**Path Parameters:**
- `ticker` (string, required): Stock ticker symbol

**Example:**
```bash
curl http://localhost:8000/api/v1/models/AAPL
```

**Response:**
```json
{
  "ticker": "AAPL",
  "model_name": "lstm_paper_AAPL_20260114_1200",
  "model_path": "models/lstm_paper_AAPL_20260114_1200_best.h5",
  "training_date": "2026-01-14T12:00:00",
  "metrics": {
    "rmse": 2.15,
    "mae": 1.75,
    "mse": 4.62,
    "mape": 1.8
  },
  "lookback_window": 60,
  "total_epochs": 100,
  "architecture": "Stacked LSTM (4 layers, 96 units each)"
}
```

**Status Codes:**
- `200`: Success
- `404`: Model not found

---

### Train New Model

**POST** `/api/v1/train`

Start a background training job for a ticker.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "epochs": 100,
  "batch_size": 32
}
```

**Parameters:**
- `ticker` (string, required): Stock ticker to train
- `epochs` (integer, optional): Training epochs (10-500). Default: from config
- `batch_size` (integer, optional): Batch size (8-256). Default: from config

**Response:**
```json
{
  "job_id": "train_AAPL_20260114_120500",
  "ticker": "AAPL",
  "status": "pending",
  "message": "Training job started for AAPL",
  "estimated_duration": "5-10 minutes",
  "started_at": "2026-01-14T12:05:00"
}
```

**Status Code:** `202 Accepted`

---

### Get Training Status

**GET** `/api/v1/train/{job_id}`

Check the status of a training job.

**Path Parameters:**
- `job_id` (string, required): Job identifier from training response

**Example:**
```bash
curl http://localhost:8000/api/v1/train/train_AAPL_20260114_120500
```

**Response:**
```json
{
  "job_id": "train_AAPL_20260114_120500",
  "ticker": "AAPL",
  "status": "running",
  "progress": 65.5,
  "current_epoch": 65,
  "total_epochs": 100,
  "metrics": {
    "loss": 0.0012,
    "val_loss": 0.0015
  }
}
```

**Status Values:**
- `pending`: Job queued
- `running`: Currently training
- `completed`: Training finished successfully
- `failed`: Training encountered an error

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ErrorType",
  "message": "Detailed error message",
  "details": {"key": "value"},
  "timestamp": "2026-01-14T12:00:00"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service unhealthy

---

## Integration Examples

### Python

```python
import requests

# Get predictions
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"ticker": "AAPL", "days": 7}
)
data = response.json()
print(f"Current price: ${data['current_price']}")
for pred in data['predictions']:
    print(f"{pred['date']}: ${pred['price']:.2f}")
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

async function getPredictions(ticker, days = 7) {
  const response = await axios.post('http://localhost:8000/api/v1/predict', {
    ticker: ticker,
    days: days
  });
  return response.data;
}

getPredictions('AAPL').then(data => {
  console.log(`Current: $${data.current_price}`);
  data.predictions.forEach(pred => {
    console.log(`${pred.date}: $${pred.price.toFixed(2)}`);
  });
});
```

### curl

```bash
# Get predictions with formatted output
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}' \
  | python -m json.tool

# Train a model
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "epochs": 100}'

# Check training status
curl http://localhost:8000/api/v1/train/train_TSLA_20260114_120500
```

---

## Rate Limiting

Currently no rate limiting is enforced. For production deployment, consider:
- API key authentication
- Rate limiting (e.g., 100 requests/minute)
- Usage quotas per user

---

## Performance

- **Prediction latency**: <500ms (with cached model)
- **Training time**: 5-10 minutes per ticker
- **Concurrent requests**: Supports multiple simultaneous predictions

---

## Deployment

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

Services:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run in production with workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Support

For issues or questions:
- Check interactive docs: http://localhost:8000/docs
- View logs: `docker-compose logs api`
- GitHub Issues: [Your repo URL]

---

## Changelog

### v1.0.0 (2026-01-14)
- Initial API release
- Prediction endpoints
- Model management
- Training functionality
- Health monitoring
