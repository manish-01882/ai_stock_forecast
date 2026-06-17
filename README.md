<div align="center">

# 📈 AI Stock Forecast

### LSTM-powered stock price prediction, served as a real API + dashboard — not just a notebook

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[Quick Start](#quick-start) • [API Usage](#api-usage) • [Architecture](#architecture) • [Model Details](#model-architecture) • [Roadmap](#roadmap)

</div>

---

## What this is

A stacked LSTM forecasting engine wrapped in a FastAPI service and a Streamlit dashboard, packaged with Docker so it can actually be deployed rather than just run in a notebook. Train a model on any ticker, hit a REST endpoint to get predictions back in under 500ms, and watch the results in a live dashboard — all in one repo.

> [!TIP]
> **Add a screenshot or short GIF of the dashboard here.** A 10-second clip of the Streamlit UI predicting a real ticker is the single most effective thing you can add to this README — drop it in `docs/demo.gif` and reference it right below this line.

<a name="features"></a>
## ✨ Features

| | Feature | Details |
|---|---|---|
| 🤖 | **LSTM neural network** | Stacked 4-layer architecture trained on historical price data |
| 🚀 | **Production REST API** | FastAPI with automatic OpenAPI/Swagger docs |
| 📊 | **Interactive dashboard** | Streamlit UI for exploring predictions live |
| 🐳 | **One-command deploy** | Full stack via Docker Compose |
| 📈 | **Multi-stock training** | Batch-train across any list of tickers |
| 📉 | **Tracked metrics** | RMSE, MAE, MAPE logged per model |
| ⚡ | **Fast inference** | Model caching keeps predictions under 500ms |
| 🔄 | **Async training** | Long training jobs run in the background via the API |

<a name="architecture"></a>
## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│  (Browser, Mobile App, Trading Bot, Python Script)         │
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

<a name="quick-start"></a>
## 🚀 Quick Start

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/manish-01882/ai_stock_forecast.git
cd ai_stock_forecast
docker-compose up -d
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

### Option 2 — Manual setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run app.py
```

<a name="api-usage"></a>
## 📡 API Usage

**Get predictions**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'
```

```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "predictions": [
    {"date": "2026-06-18", "price": 151.30},
    {"date": "2026-06-19", "price": 152.10}
  ],
  "model_metrics": {"rmse": 2.15, "mape": 1.8},
  "confidence_score": 0.89
}
```

**List available models**

```bash
curl http://localhost:8000/api/v1/models
```

**Train a new model**

```bash
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "epochs": 100}'
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | API landing page |
| POST | `/api/v1/predict` | Generate predictions |
| GET | `/api/v1/models` | List all trained models |
| GET | `/api/v1/models/{ticker}` | Get model info |
| POST | `/api/v1/train` | Start a training job |
| GET | `/api/v1/train/{job_id}` | Check training job status |

<a name="model-architecture"></a>
## 🧠 Model Architecture

```
Input (60-day price window)
   │
   ▼
LSTM(96) — return_sequences=True  →  Dropout(0.2)
   │
   ▼
LSTM(96) — return_sequences=True  →  Dropout(0.2)
   │
   ▼
LSTM(96) — return_sequences=True  →  Dropout(0.2)
   │
   ▼
LSTM(96)                          →  Dropout(0.2)
   │
   ▼
Dense(1)  →  predicted price
```

**Training setup:** Adam optimizer, MSE loss, early stopping (patience=10), checkpointing on best validation weights.

## 🎓 Training

```bash
# Single stock
python train_model.py

# Batch across multiple tickers
python train_batch.py
```

```
┌─────────┬─────────┬─────────┬────────┬──────────┐
│ Ticker  │ Status  │  RMSE   │  MAE   │   MAPE   │
├─────────┼─────────┼─────────┼────────┼──────────┤
│ AAPL    │ SUCCESS │  $2.15  │ $1.75  │   1.80%  │
│ GOOGL   │ SUCCESS │  $3.42  │ $2.85  │   2.10%  │
│ MSFT    │ SUCCESS │  $1.98  │ $1.62  │   1.65%  │
└─────────┴─────────┴─────────┴────────┴──────────┘
```

## 📁 Project Structure

```
ai_stock_forecast/
├── api/                     # FastAPI REST API
│   ├── main.py
│   ├── models.py            # Pydantic schemas
│   └── utils.py
├── src/                     # Core ML modules
│   ├── data_ingestor.py     # Yahoo Finance data fetching
│   ├── feature_engineer.py  # Preprocessing
│   └── model_trainer.py     # LSTM training
├── config/
│   ├── config.yaml
│   └── settings.py
├── models/                  # Trained model artifacts
├── data/raw/                # Raw CSV data
├── docs/
│   ├── API.md
│   └── api_examples.sh
├── tests/
│   └── test_api.py
├── app.py                   # Streamlit dashboard
├── train_batch.py
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Configuration

`config/config.yaml` covers the basics:

```yaml
tickers: ["AAPL", "GOOGL", "MSFT", "TSLA"]
look_back: 60
train_test_split: 0.8

models:
  production_path: "models/lstm_production.h5"
  registry_dir: "models"
```

`config/settings.py` covers epochs, batch size, architecture tweaks, data sources, and logging.

## 🧪 Testing

```bash
pip install pytest httpx

pytest tests/ -v                                    # all tests
pytest tests/test_api.py::test_health_endpoint -v    # single test
pytest tests/ --cov=api --cov-report=html             # with coverage
```

<a name="roadmap"></a>
## 🗺️ Roadmap

- [ ] Additional architectures (GRU, Transformer)
- [ ] Model ensembling
- [ ] News sentiment as an auxiliary feature
- [ ] Backtesting framework
- [ ] CI/CD pipeline

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Manish Choudhary**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manish-choudhary-547b092b7/)
[![Email](https://img.shields.io/badge/Email-man01882%40outlook.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:man01882@outlook.com)

⭐ If this was useful as a reference, a star helps it get found

</div>
