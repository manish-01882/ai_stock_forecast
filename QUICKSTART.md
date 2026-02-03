# Quick Start Guide - Stock Forecast API

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Option A: Using virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option B: System-wide install
pip install -r requirements.txt
```

### Step 2: Start the API

```bash
# Start FastAPI API
uvicorn api.main:app --reload

# The API will start on http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Step 3: Test It

```bash
# In another terminal - test health endpoint
curl http://localhost:8000/health

# Get predictions (if you have trained models for AAPL)
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'

# Or just open browser and go to:
# http://localhost:8000/docs
# Click "Try it out" on any endpoint!
```

---

## 🐳 Docker Quick Start

### Start Everything

```bash
# Build and start API + Dashboard
docker-compose up --build

# Services will be available at:
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

### Manage Services

```bash
# View logs
docker-compose logs -f api

# Stop everything
docker-compose down

# Restart
docker-compose restart
```

---

## 📝 Before First Use

### Train Some Models

If you don't have models yet:

```bash
# Train a single model
python train_model.py

# Or batch train multiple stocks
python train_batch.py
```

This will create model files in `models/` directory.

---

## 🧪 Run Tests

```bash
# Install test dependencies (if not already installed)
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::test_health_endpoint -v
```

---

## 📚 Available Endpoints

Once the API is running, visit:

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**API Endpoints:**
- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/api/v1/predict`
- Models: `GET http://localhost:8000/api/v1/models`
- Model Info: `GET http://localhost:8000/api/v1/models/AAPL`

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Docker permission denied
```bash
# Option 1: Use sudo
sudo docker-compose up

# Option 2: Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then log out and log back in
```

### Port already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml or when running uvicorn:
uvicorn api.main:app --port 8001
```

### No models found
```bash
# Train at least one model first
python train_model.py
```

---

## 💡 Quick Examples

### Python Integration

```python
import requests

# Get predictions
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"ticker": "AAPL", "days": 7}
)

if response.status_code == 200:
    data = response.json()
    print(f"Current price: ${data['current_price']}")
    print("Predictions:")
    for pred in data['predictions']:
        print(f"  {pred['date']}: ${pred['price']:.2f}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript Integration

```javascript
fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ticker: 'AAPL', days: 7})
})
.then(r => r.json())
.then(data => console.log('Predictions:', data.predictions))
.catch(err => console.error('Error:', err));
```

---

## 📖 Next Steps

1. **Read Full Documentation:** [docs/API.md](docs/API.md)
2. **Run Example Commands:** `./docs/api_examples.sh`
3. **Explore Interactive Docs:** http://localhost:8000/docs
4. **Customize Configuration:** Edit `config/config.yaml`
5. **Add More Models:** Train additional stock tickers

---

## 🎯 Common Tasks

```bash
# Start API for development (auto-reload)
uvicorn api.main:app --reload

# Start API for production (multiple workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start dashboard alongside API
streamlit run app.py  # In another terminal

# Train new model via API
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "epochs": 100}'

# Check training status
curl http://localhost:8000/api/v1/train/train_TSLA_20260114_120500
```

---

## ✅ Validation Checklist

- [ ] Dependencies installed successfully
- [ ] API starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Health endpoint returns 200
- [ ] At least one model trained
- [ ] Prediction endpoint works
- [ ] Tests pass with pytest

---

## 📞 Need Help?

- Check [README.md](README.md) for detailed info
- Read [docs/API.md](docs/API.md) for API reference
- Open GitHub issue
- Check logs: `docker-compose logs api`

---

**Ready to Deploy? 🚀**

Follow the production deployment section in [README.md](README.md)
