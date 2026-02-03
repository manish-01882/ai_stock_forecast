#!/bin/bash
# Stock Forecast API - Example curl commands
# Make this file executable: chmod +x api_examples.sh

API_URL="http://localhost:8000"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        Stock Forecast API - Example Commands             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Health Check
echo -e "${GREEN}1. Health Check${NC}"
echo -e "${BLUE}GET $API_URL/health${NC}"
curl -s "$API_URL/health" | python3 -m json.tool
echo -e "\n"

# 2. List All Models
echo -e "${GREEN}2. List All Available Models${NC}"
echo -e "${BLUE}GET $API_URL/api/v1/models${NC}"
curl -s "$API_URL/api/v1/models" | python3 -m json.tool
echo -e "\n"

# 3. Get Specific Model Info
echo -e "${GREEN}3. Get Model Info for AAPL${NC}"
echo -e "${BLUE}GET $API_URL/api/v1/models/AAPL${NC}"
curl -s "$API_URL/api/v1/models/AAPL" | python3 -m json.tool
echo -e "\n"

# 4. Get Predictions for AAPL (7 days)
echo -e "${GREEN}4. Get 7-Day Predictions for AAPL${NC}"
echo -e "${BLUE}POST $API_URL/api/v1/predict${NC}"
curl -s -X POST "$API_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}' \
  | python3 -m json.tool
echo -e "\n"

# 5. Get Extended Predictions (30 days)
echo -e "${GREEN}5. Get 30-Day Predictions for GOOGL${NC}"
echo -e "${BLUE}POST $API_URL/api/v1/predict${NC}"
curl -s -X POST "$API_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "GOOGL", "days": 30}' \
  | python3 -m json.tool
echo -e "\n"

# 6. Train a New Model
echo -e "${GREEN}6. Train New Model for TSLA${NC}"
echo -e "${BLUE}POST $API_URL/api/v1/train${NC}"
echo -e "${YELLOW}Note: This will start a background training job${NC}"
curl -s -X POST "$API_URL/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "epochs": 50, "batch_size": 32}' \
  | python3 -m json.tool
echo -e "\n"

# 7. Check Training Status (you'll need to replace JOB_ID)
echo -e "${GREEN}7. Check Training Status${NC}"
echo -e "${BLUE}GET $API_URL/api/v1/train/{job_id}${NC}"
echo -e "${YELLOW}Replace {job_id} with actual job ID from training response${NC}"
echo "Example: curl -s \"$API_URL/api/v1/train/train_TSLA_20260114_120500\" | python3 -m json.tool"
echo -e "\n"

# 8. Test Error Handling (Invalid Ticker)
echo -e "${GREEN}8. Test Error Handling - Invalid Ticker${NC}"
echo -e "${BLUE}POST $API_URL/api/v1/predict${NC}"
curl -s -X POST "$API_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "INVALID_TICKER_XYZ", "days": 7}' \
  | python3 -m json.tool
echo -e "\n"

# 9. Performance Test - Multiple Predictions
echo -e "${GREEN}9. Performance Test - Multiple Sequential Predictions${NC}"
echo -e "${YELLOW}Testing latency for AAPL predictions...${NC}"
for i in {1..3}; do
  echo -n "Request $i: "
  time_start=$(date +%s%N)
  curl -s -X POST "$API_URL/api/v1/predict" \
    -H "Content-Type: application/json" \
    -d '{"ticker": "AAPL", "days": 7}' > /dev/null
  time_end=$(date +%s%N)
  duration=$(( (time_end - time_start) / 1000000 ))
  echo "${duration}ms"
done
echo -e "\n"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   Tests Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📚 For more information, visit: $API_URL/docs"
echo ""
