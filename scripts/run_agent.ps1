# Run the live agent
# Usage: .\scripts\run_agent.ps1
# Arguments are passed through to docker compose

docker compose run --rm agent @args
exit $LASTEXITCODE
