# Dockerfile for AFA (Autonomous Financial Advisor)
# Cross-platform compatible (Windows, macOS, Linux)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Install system dependencies
# build-essential is needed for compiling pandas/numpy if wheels fail
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies including pyarrow for parquet support
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pyarrow

# Copy application code
COPY portfolio_agent/src/ ./src/
COPY portfolio_agent/main.py .
COPY portfolio_agent/config.yaml .
COPY portfolio_agent/tests/ ./tests/

# Create necessary directories inside the container
# These will be overlaid by host volumes in docker-compose, but ensures structure exists
RUN mkdir -p /app/data /app/output /app/logs

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
