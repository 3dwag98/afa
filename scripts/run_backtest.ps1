# Run the backtest engine
# Usage: .\scripts\run_backtest.ps1 [--years N] [--universe-size N]
# Examples:
#   .\scripts\run_backtest.ps1 --years 2 --universe-size 20
#   .\scripts\run_backtest.ps1 --years 5
# Arguments are passed through to docker compose

docker compose run --rm backtest @args
exit $LASTEXITCODE
