@echo off
REM AFA Agent Runner Script for Windows
REM Usage: run-agent.bat
REM 
REM Options can be set in config.yaml:
REM   force_refresh: true/false
REM   simulate_outcome: true/false
REM   update_outcomes: true/false

setlocal enabledelayedexpansion

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Build and run the agent
echo Starting AFA Agent...
docker compose --profile base run --rm agent

if errorlevel 0 (
    echo.
    echo Agent completed successfully.
    echo Check the 'output' folder for reports.
) else (
    echo.
    echo Agent failed. Check logs for details.
)

endlocal
