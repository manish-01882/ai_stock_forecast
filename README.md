# 📈 AI Stock Forecast — LSTM Price Prediction System

AI-powered stock price prediction system with stacked LSTM neural networks, a production-style REST API, and an interactive dashboard — built to go past a single training script and look like something that could actually be deployed.

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.0-green)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)

## Features

- **LSTM neural networks** — stacked 4-layer architecture trained on historical price data
- **Production REST API** — FastAPI with automatic OpenAPI docs
- **Interactive dashboard** — Streamlit web interface for exploring predictions
- **Docker support** — one-command deployment via Docker Compose
- **Multi-stock training** — batch training across multiple tickers
- **Performance tracking** — RMSE, MAE, MAPE logged per model
- **Fast inference** — model caching keeps predictions under 500ms
- **Background training** — training jobs run async via the API

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│  (Browser, Mobile App, Trading Bot, Python Script)        │
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

## Quick start

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/manish-01882/ai_stock_forecast.git
cd ai_stock_forecast
docker-compose up -d
```

Then visit the dashboard at `http://localhost:8501` and the API docs at `http://localhost:8000/docs`.

### Option 2: Manual setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run app.py
```

## API usage

**Get predictions:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'
```

**List available models:**

```bash
curl http://localhost:8000/api/v1/models
```

**Train a new model:**

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
| GET | `/api/v1/models` | List all models |
| GET | `/api/v1/models/{ticker}` | Get model info |
| POST | `/api/v1/train` | Start a training job |
| GET | `/api/v1/train/{job_id}` | Check training job status |

## Model architecture

Stacked LSTM, 4 layers (96 units each) with 0.2 dropout between layers, ending in a single dense output unit. Trained with Adam and MSE loss, early stopping (patience=10), and checkpointing on best weights.

## Training

```bash
# Single stock
python train_model.py

# Batch across multiple tickers
python train_batch.py
```

Sample batch output:

```
┌─────────┬─────────┬─────────┬────────┬──────────┐
│ Ticker  │ Status  │  RMSE   │  MAE   │   MAPE   │
├─────────┼─────────┼─────────┼────────┼──────────┤
│ AAPL    │ SUCCESS │  $2.15  │ $1.75  │   1.80%  │
│ GOOGL   │ SUCCESS │  $3.42  │ $2.85  │   2.10%  │
│ MSFT    │ SUCCESS │  $1.98  │ $1.62  │   1.65%  │
└─────────┴─────────┴─────────┴────────┴──────────┘
```

## Project structure

```
ai_stock_forecast/
├── api/                    # FastAPI REST API
│   ├── main.py
│   ├── models.py           # Pydantic schemas
│   └── utils.py
├── src/                    # Core ML modules
│   ├── data_ingestor.py    # Yahoo Finance data fetching
│   ├── feature_engineer.py # Preprocessing
│   └── model_trainer.py    # LSTM training
├── config/
│   ├── config.yaml
│   └── settings.py
├── models/                 # Trained model artifacts
├── data/raw/                # Raw CSV data
├── docs/
│   ├── API.md
│   └── api_examples.sh
├── tests/
│   └── test_api.py
├── app.py                  # Streamlit dashboard
├── train_batch.py
├── docker-compose.yml
└── requirements.txt
```

## Configuration

Edit `config/config.yaml` for the basics:

```yaml
tickers: ["AAPL", "GOOGL", "MSFT", "TSLA"]
look_back: 60
train_test_split: 0.8

models:
  production_path: "models/lstm_production.h5"
  registry_dir: "models"
```

`config/settings.py` covers epochs, batch size, architecture tweaks, data sources, and logging.

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
pytest tests/test_api.py::test_health_endpoint -v   # single test
pytest tests/ --cov=api --cov-report=html             # with coverage
```

## Roadmap

- [ ] Additional model architectures (GRU, Transformer)
- [ ] Model ensembling
- [ ] News sentiment as an auxiliary feature
- [ ] Backtesting framework
- [ ] CI/CD pipeline

## License

MIT — see `LICENSE` for details.

## Author

**Manish Choudhary**

[LinkedIn](https://www.linkedin.com/in/manish-choudhary-547b092b7/) · [man01882@outlook.com](mailto:man01882@outlook.com)
