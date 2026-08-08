# AFA Project Windows Setup Script
# PowerShell 5+ compatible
# Does not require administrator privileges (unless Docker itself requires it)

Write-Host "=== AFA Project Windows Setup ===" -ForegroundColor Cyan

# Check if docker is available
Write-Host "Checking Docker availability..." -NoNewline
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Docker is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# Check if docker info succeeds
Write-Host "Checking Docker daemon..." -NoNewline
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Yellow
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# Check if docker compose version succeeds
Write-Host "Checking Docker Compose..." -NoNewline
$composeVersion = docker compose version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Docker Compose is not available." -ForegroundColor Yellow
    Write-Host "Please ensure Docker Desktop is installed with Compose plugin." -ForegroundColor Yellow
    exit 1
}
Write-Host " OK ($composeVersion)" -ForegroundColor Green

# Create required directories
Write-Host "Creating required directories..." -ForegroundColor Cyan

$directories = @(
    "data",
    "data/market_data",
    "output",
    "logs",
    "airflow/dags",
    "scripts"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

# Copy .env.example to .env if .env does not exist
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  Created .env from .env.example" -ForegroundColor Green
    } else {
        Write-Host "  No .env.example found, skipping .env creation" -ForegroundColor Yellow
    }
} else {
    Write-Host "  .env already exists, skipping" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  Build images:       docker compose build" -ForegroundColor White
Write-Host "  Run agent:          docker compose run --rm agent" -ForegroundColor White
Write-Host "  Run backtest:       docker compose run --rm backtest" -ForegroundColor White
Write-Host "  Run tests:          docker compose run --rm test" -ForegroundColor White
Write-Host ""
Write-Host "Or use the PowerShell scripts:" -ForegroundColor Cyan
Write-Host "  .\scripts\run_agent.ps1" -ForegroundColor White
Write-Host "  .\scripts\run_backtest.ps1 --years 2 --universe-size 20" -ForegroundColor White
Write-Host "  .\scripts\run_tests.ps1" -ForegroundColor White
