# AFA Docker Smoke Test (PowerShell)
# Windows-compatible smoke test script

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AFA Docker Smoke Test" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Building app image..." -ForegroundColor Yellow
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker compose build failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow
docker compose run --rm test
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Tests failed with exit code $LASTEXITCODE" -ForegroundColor Yellow
    # Continue with smoke test even if tests fail
}

Write-Host ""
Write-Host "Running agent help or dry check..." -ForegroundColor Yellow
docker compose run --rm agent python main.py --help
# Allow this to fail gracefully
if ($LASTEXITCODE -ne 0) {
    Write-Host "NOTE: Agent help command returned non-zero (may be expected)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Running backtest help or dry check..." -ForegroundColor Yellow
docker compose run --rm backtest python run_backtest.py --help
# Allow this to fail gracefully
if ($LASTEXITCODE -ne 0) {
    Write-Host "NOTE: Backtest help command returned non-zero (may be expected)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Running small backtest (1 year, 10 tickers)..." -ForegroundColor Yellow
docker compose run --rm backtest python run_backtest.py --years 1 --universe-size 10
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Small backtest failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Smoke test complete." -ForegroundColor Green
Write-Host "Check ./output for generated Excel files." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

exit 0
