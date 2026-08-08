@echo off
REM AFA Agent Runner Script for Windows
REM Usage: run-agent.bat [OPTIONS]
REM Options:
REM   --force-refresh      Force refresh all data
REM   --simulate-outcome   Simulate trade outcomes
REM   --update-outcomes    Update actual trade outcomes

setlocal enabledelayedexpansion

REM Parse command line arguments
set ARGS=
for %%a in (%*) do (
    set ARGS=!ARGS! %%a
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Build and run the agent
echo Starting AFA Agent...
docker compose --profile base run --rm agent %ARGS%

if errorlevel 0 (
    echo.
    echo Agent completed successfully.
    echo Check the 'output' folder for reports.
) else (
    echo.
    echo Agent failed. Check logs for details.
)

endlocal
