FROM python:3.13-slim

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENV PORT=8501

EXPOSE 8501

# Run Streamlit dashboard
CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]





