"""
Airflow DAGs for Portfolio Agent

Three DAGs:
1. portfolio_daily_run - Daily portfolio execution (weekdays)
2. portfolio_update_outcomes - Update trade outcomes (weekdays)
3. portfolio_relearn - Manual relearning task
"""

import logging
from datetime import timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pendulum

from src.airflow_jobs import (
    run_daily_job,
    run_update_outcomes_job,
    run_relearn_job
)

logger = logging.getLogger(__name__)


def load_config_safe():
    """Load config safely with fallback to defaults."""
    try:
        from src.config import get_config
        config = get_config()
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return a mock config object with defaults
        class DefaultConfig:
            scheduler_enabled = False
            schedule_time_ist = "15:45"
            schedule_outcome_time_ist = "16:00"
        return DefaultConfig()


def time_to_cron(time_str: str) -> str:
    """Convert HH:MM time string to cron expression for weekdays."""
    try:
        parts = time_str.split(":")
        hour = parts[0]
        minute = parts[1]
        # Weekdays only (Monday=1 to Friday=5)
        return f"{minute} {hour} * * 1-5"
    except Exception as e:
        logger.error(f"Failed to parse time {time_str}: {e}")
        # Default fallback
        return "45 15 * * 1-5"


# Load configuration
config = load_config_safe()
scheduler_enabled = getattr(config, 'scheduler_enabled', False)
schedule_time_ist = getattr(config, 'schedule_time_ist', "15:45")
schedule_outcome_time_ist = getattr(config, 'schedule_outcome_time_ist', "16:00")

# Convert to cron expressions
daily_cron = time_to_cron(schedule_time_ist)
outcome_cron = time_to_cron(schedule_outcome_time_ist)

# Determine if DAGs should be paused
is_paused = not scheduler_enabled

# Default arguments for all DAGs
default_args = {
    'owner': 'portfolio_agent',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
}

# DAG 1: Daily Portfolio Run
dag_daily = DAG(
    dag_id='portfolio_daily_run',
    default_args=default_args,
    schedule=daily_cron,
    catchup=False,
    max_active_runs=1,
    tags=['portfolio', 'daily', 'india'],
    is_paused_upon_creation=is_paused,
)

daily_task = PythonOperator(
    task_id='daily_run',
    python_callable=run_daily_job,
    dag=dag_daily,
)

# DAG 2: Update Outcomes
dag_outcomes = DAG(
    dag_id='portfolio_update_outcomes',
    default_args=default_args,
    schedule=outcome_cron,
    catchup=False,
    max_active_runs=1,
    tags=['portfolio', 'outcomes', 'india'],
    is_paused_upon_creation=is_paused,
)

outcomes_task = PythonOperator(
    task_id='update_outcomes',
    python_callable=run_update_outcomes_job,
    dag=dag_outcomes,
)

# DAG 3: Relearn (Manual)
dag_relearn = DAG(
    dag_id='portfolio_relearn',
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=['portfolio', 'learning', 'manual'],
    is_paused_upon_creation=False,
)

relearn_task = PythonOperator(
    task_id='relearn',
    python_callable=run_relearn_job,
    dag=dag_relearn,
)
