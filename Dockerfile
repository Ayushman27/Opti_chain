# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for CBC solver
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    coinor-cbc \
    coinor-libcbc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY optichain/ ./optichain/
COPY pyproject.toml .
COPY pytest.ini .

EXPOSE 8000

CMD ["uvicorn", "optichain.agent.service:app", "--host", "0.0.0.0", "--port", "8000"]
