# 📈 Stock Forecast - ML System

AI-powered stock price prediction system with **LSTM neural networks**, production-ready **REST API**, and interactive **dashboard**.

![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.127.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## ✨ Features

- 🤖 **LSTM Neural Networks** - Stacked 4-layer architecture trained on historical price data
- 🚀 **Production REST API** - FastAPI with automatic OpenAPI docs
- 📊 **Interactive Dashboard** - Streamlit web interface
- 🐳 **Docker Support** - One-command deployment with Docker Compose  
- 📈 **Multi-Stock Training** - Batch training for multiple tickers
- 📉 **Performance Metrics** - RMSE, MAE, MAPE tracking
- ⚡ **Model Caching** - Fast predictions (<500ms)
- 🔄 **Background Training** - Async model training jobs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│  (Browser, Mobile App, Trading Bot, Python Script)      │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
        ┌──────▼──────┐        ┌─────▼──────┐
        │  Dashboard  │        │    API     │
        │ (Streamlit) │        │ (FastAPI)  │
        │   :8501     │        │   :8000    │
        └──────┬──────┘        └─────┬──────┘
               │                     │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Shared Resources  │
               │  ├─ Models (.h5)    │
               │  ├─ Data (CSV)      │
               │  └─ Config (YAML)   │
               └─────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd stock_forecast

# Start both API and Dashboard
docker-compose up -d

# Access the services
# Dashboard: http://localhost:8501
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API (Terminal 1)
uvicorn api.main:app --reload

# Run the Dashboard (Terminal 2)
streamlit run app.py
```

---

## 📡 API Usage

### Get Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'
```

Response:
```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "predictions": [
    {"date": "2026-01-15", "price": 151.30},
    {"date": "2026-01-16", "price": 152.10}
  ],
  "model_metrics": {"rmse": 2.15, "mape": 1.8},
  "confidence_score": 0.89
}
```

### List Available Models

```bash
curl http://localhost:8000/api/v1/models
```

### Train New Model

```bash
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "epochs": 100}'
```

**Full API Documentation:** http://localhost:8000/docs

---

## 🎓 Training Models

### Train Single Stock

```bash
python train_model.py
```

### Batch Train Multiple Stocks

```bash
python train_batch.py
```

Results:
```
┌─────────┬─────────┬─────────┬────────┬──────────┐
│ Ticker  │ Status  │  RMSE   │  MAE   │   MAPE   │
├─────────┼─────────┼─────────┼────────┼──────────┤
│ AAPL    │ SUCCESS │  $2.15  │ $1.75  │   1.80%  │
│ GOOGL   │ SUCCESS │  $3.42  │ $2.85  │   2.10%  │
│ MSFT    │ SUCCESS │  $1.98  │ $1.62  │   1.65%  │
└─────────┴─────────┴─────────┴────────┴──────────┘
```

---

## 📁 Project Structure

```
stock_forecast/
├── api/                    # FastAPI REST API
│   ├── __init__.py
│   ├── main.py            # API application
│   ├── models.py          # Pydantic schemas
│   └── utils.py           # Helper functions
├── src/                   # Core ML modules
│   ├── data_ingestor.py   # Yahoo Finance data fetching
│   ├── feature_engineer.py # Data preprocessing
│   └── model_trainer.py   # LSTM training
├── config/                # Configuration
│   ├── config.yaml        # Training parameters
│   └── settings.py        # Python settings
├── models/                # Trained model files
├── data/                  # Dataset storage
│   └── raw/              # Raw CSV files
├── docs/                  # Documentation
│   ├── API.md            # API reference
│   └── api_examples.sh   # Example commands
├── tests/                 # Test suite
│   └── test_api.py       # API tests
├── app.py                 # Streamlit dashboard
├── train_batch.py         # Batch training script
├── Dockerfile.api         # API container
├── dockerfile             # Dashboard container
├── docker-compose.yml     # Multi-service orchestration
└── requirements.txt       # Python dependencies
```

---

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
tickers: ["AAPL", "GOOGL", "MSFT", "TSLA"]
look_back: 60
train_test_split: 0.8

models:
  production_path: "models/lstm_production.h5"
  registry_dir: "models"
```

Edit `config/settings.py` for advanced options:
- Epochs, batch size
- LSTM architecture
- Data sources
- Logging levels

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::test_health_endpoint -v

# Run with coverage
pytest tests/ --cov=api --cov-report=html
```

---

## 📊 Model Architecture

**Stacked LSTM Network:**
- **Layer 1:** LSTM (96 units, return_sequences=True)
- **Dropout:** 0.2
- **Layer 2:** LSTM (96 units, return_sequences=True)
- **Dropout:** 0.2
- **Layer 3:** LSTM (96 units, return_sequences=True)
- **Dropout:** 0.2
- **Layer 4:** LSTM (96 units)
- **Dropout:** 0.2
- **Output:** Dense (1 unit)

**Training:**
- Optimizer: Adam
- Loss: MSE
- Early stopping with patience=10
- ModelCheckpoint for best weights

---

## 📈 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | API landing page |
| **Predictions** |
| POST | `/api/v1/predict` | Generate predictions |
| **Models** |
| GET | `/api/v1/models` | List all models |
| GET | `/api/v1/models/{ticker}` | Get model info |
| **Training** |
| POST | `/api/v1/train` | Start training job |
| GET | `/api/v1/train/{job_id}` | Check training status |

---

## 🐳 Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Remove everything
docker-compose down -v
```

---

## 🎯 Integration Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"ticker": "AAPL", "days": 7}
)
predictions = response.json()
```

### JavaScript

```javascript
fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ticker: 'AAPL', days: 7})
})
.then(r => r.json())
.then(data => console.log(data));
```

### cURL

```bash
# Run all examples
./docs/api_examples.sh
```

---

## 🔒 Production Deployment

1. **Environment Variables:** Create `.env` file
2. **Authentication:** Add API key middleware
3. **Rate Limiting:** Implement request throttling
4. **HTTPS:** Use reverse proxy (nginx)
5. **Monitoring:** Add Prometheus/Grafana
6. **Scaling:** Deploy multiple Uvicorn workers

```bash
# Production command
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📝 TODO / Roadmap

- [ ] Add more ML models (GRU, Transformer)
- [ ] Implement model ensemble
- [ ] Add sentiment analysis from news
- [ ] Create mobile app
- [ ] Add backtesting framework
- [ ] Implement CI/CD pipeline
- [ ] Add Grafana dashboards
- [ ] Support cryptocurrency predictions

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Manish** - Stock Forecast ML System

- GitHub: [@your-github]
- LinkedIn: [your-linkedin]
- Portfolio: [your-portfolio]

---

## 🙏 Acknowledgments

- Research papers for LSTM architecture
- Yahoo Finance for market data
- FastAPI and Streamlit communities

---

## 📞 Support

- **Documentation:** [docs/API.md](docs/API.md)
- **Issues:** GitHub Issues
- **Email:** your-email@example.com

---

**⭐ Star this repository if you found it helpful!**
