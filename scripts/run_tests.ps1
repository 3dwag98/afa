# Run the test suite
# Usage: .\scripts\run_tests.ps1
# Arguments are passed through to docker compose

docker compose run --rm test @args
exit $LASTEXITCODE
