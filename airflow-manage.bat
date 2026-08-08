@echo off
REM AFA Airflow Management Script for Windows
REM Usage: airflow-manage.bat [COMMAND]
REM Commands:
REM   start     - Start all Airflow services (webserver, scheduler, postgres)
REM   stop      - Stop all Airflow services
REM   restart   - Restart all Airflow services
REM   logs      - Show logs from all services
REM   init      - Initialize/reset the Airflow database
REM   status    - Check status of services

setlocal enabledelayedexpansion

if "%1"=="" goto usage
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="init" goto init
if "%1"=="status" goto status
goto usage

:start
    echo Starting Airflow services...
    docker compose --profile airflow up -d --build
    echo.
    echo Airflow is starting. Wait ~30 seconds for all services to be ready.
    echo Web UI will be available at: http://localhost:8080
    echo Login: admin / admin
    exit /b 0

:stop
    echo Stopping Airflow services...
    docker compose --profile airflow down
    echo.
    echo Airflow services stopped.
    exit /b 0

:restart
    echo Restarting Airflow services...
    docker compose --profile airflow down
    docker compose --profile airflow up -d --build
    echo.
    echo Airflow is restarting. Wait ~30 seconds for all services to be ready.
    exit /b 0

:logs
    docker compose --profile airflow logs -f
    exit /b 0

:init
    echo WARNING: This will reset the Airflow database!
    set /p CONFIRM="Are you sure? (y/n): "
    if /i not "!CONFIRM!"=="y" (
        echo Operation cancelled.
        exit /b 0
    )
    echo Initializing Airflow database...
    docker compose --profile airflow down
    docker volume rm afa_postgres_data
    docker compose --profile airflow up -d --build airflow-init
    echo.
    echo Airflow database initialized.
    exit /b 0

:status
    echo Checking Airflow service status...
    docker compose --profile airflow ps
    exit /b 0

:usage
    echo Usage: airflow-manage.bat [COMMAND]
    echo.
    echo Commands:
    echo   start     - Start all Airflow services
    echo   stop      - Stop all Airflow services
    echo   restart   - Restart all Airflow services
    echo   logs      - Show logs from all services
    echo   init      - Initialize/reset the Airflow database
    echo   status    - Check status of services
    echo.
    exit /b 1

endlocal
