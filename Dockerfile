# Optional Dockerfile for AFA (Autonomous Financial Advisor).
#
# The primary workflow is local `uv sync` + `uv run portfolio-agent ...` (see
# README). This image exists for reproducibility on a fresh machine/GPU box,
# not as a required part of the workflow.
#
# GPU (torch) support is opt-in via a build arg, since most usage (rule-based
# backtesting, the live agent) doesn't need it:
#   docker build --build-arg INSTALL_GPU=true -t afa:gpu .

FROM python:3.11-slim

ARG INSTALL_GPU=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

LABEL project="afa" \
      component="agent-backtest" \
      purpose="portfolio agent and historical backtest engine"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock* ./

RUN if [ "$INSTALL_GPU" = "true" ]; then \
        uv sync --frozen --no-dev --extra gpu || uv sync --no-dev --extra gpu; \
    else \
        uv sync --frozen --no-dev || uv sync --no-dev; \
    fi

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

RUN mkdir -p /app/data /app/models /app/output /app/logs \
    && chown -R appuser:appuser /app/data /app/models /app/output /app/logs

COPY portfolio_agent/ ./portfolio_agent/
COPY config.yaml .

USER appuser

CMD ["uv", "run", "portfolio-agent", "run-agent"]
