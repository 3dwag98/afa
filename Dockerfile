# Production-safe Dockerfile for local Docker execution
FROM python:3.11-slim

LABEL project="portfolio_agent"
LABEL purpose="self-learning portfolio optimization agent"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Kolkata

# Set working directory
WORKDIR /app

# Install system packages (tzdata for timezone support)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY config.yaml .
COPY src/ ./src/
COPY tests/ ./tests/

# Create required directories
RUN mkdir -p /app/data /app/output /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Give ownership of data, output, and logs directories to appuser
RUN chown -R appuser:appuser /app/data /app/output /app/logs

# Switch to non-root user
USER appuser

# Default entrypoint
CMD ["python", "main.py"]
