# Dockerfile for AFA (Autonomous Financial Advisor)
# Portfolio Agent and Historical Backtest Engine
# Cross-platform compatible (Windows, macOS, Linux)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata \
    APP_HOME=/app \
    PYTHONPATH=/app

# Labels
LABEL project="afa" \
      component="agent-backtest" \
      purpose="portfolio agent and historical backtest engine"

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# Use COPY --follow-symlinks to handle symlinks properly (Docker BuildKit feature)
COPY --follow-symlinks main.py .
COPY --follow-symlinks run_backtest.py .
COPY --follow-symlinks config.yaml .
COPY --follow-symlinks requirements.txt .
COPY --follow-symlinks src/ ./src/
COPY --follow-symlinks tests/ ./tests/
COPY --follow-symlinks scripts/ ./scripts/

# Create runtime directories
RUN mkdir -p /app/data /app/data/market_data /app/output /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Give ownership of data, output, and logs directories to appuser
RUN chown -R appuser:appuser /app/data /app/output /app/logs

# Switch to non-root user
USER appuser

# Default command runs the live agent
CMD ["python", "main.py"]
