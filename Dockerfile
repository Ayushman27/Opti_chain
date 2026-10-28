# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Ensure Python output is unbuffered and disable bytecode creation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies including COIN-OR CBC solver and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    coinor-cbc \
    coinor-libcbc-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, benchmark artifacts, and test configs
COPY optichain/ ./optichain/
COPY docs/ ./docs/
COPY pyproject.toml pytest.ini ./

# Create non-root user for container security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Verified health check against actual FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "optichain.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
