# Dockerfile for AFA (Autonomous Financial Advisor)
# Portfolio Agent and Historical Backtest Engine
# Modern, optimized with uv package manager and GPU support

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

# Labels
LABEL project="afa" \
      component="agent-backtest" \
      purpose="portfolio agent and historical backtest engine with GPU support"

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml .
COPY uv.lock* ./

# Install Python dependencies with GPU support (frozen lockfile)
RUN uv sync --frozen --no-dev --extra gpu || uv sync --no-dev --extra gpu

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Create runtime directories and set ownership
RUN mkdir -p /app/data /app/models /app/output /app/logs \
    && chown -R appuser:appuser /app/data /app/models /app/output /app/logs

# Copy application code
COPY portfolio_agent/ ./portfolio_agent/
COPY config/ ./config/
COPY main.py .
COPY run_backtest.py .
COPY cli.py . 2>/dev/null || true

# Switch to non-root user
USER appuser

# Default command runs the live agent
CMD ["python", "-m", "portfolio_agent.cli", "run-agent"]
