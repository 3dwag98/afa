"""
AFA Operations DAGs

This module defines Airflow DAGs for:
1. Daily agent execution (weekday mornings at 15:45 IST)
2. Small backtest (1 year, 20 tickers) - manually triggered
3. Full backtest (5 years, all tickers) - manually triggered
"""

import sys
import logging
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project path for Docker volume mounts
project_path = '/opt/airflow/project'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    logger.info(f"Added {project_path} to sys.path")

# Import job functions from src.airflow_jobs
try:
    from src.airflow_jobs import (
        run_daily_agent_job,
        run_backtest_job,
        run_full_backtest_job
    )
except ImportError:
    from airflow_jobs import (
        run_daily_agent_job,
        run_backtest_job,
        run_full_backtest_job
    )


# Default arguments for daily agent DAG
daily_agent_default_args = {
    'owner': 'afa',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    'catchup': False,
}

# Default arguments for small backtest DAG
small_backtest_default_args = {
    'owner': 'afa',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'start_date': pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    'catchup': False,
}

# Default arguments for full backtest DAG
full_backtest_default_args = {
    'owner': 'afa',
    'retries': 0,
    'start_date': pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    'catchup': False,
}


# DAG 1: afa_daily_agent - Runs daily on weekdays at 15:45 IST
with DAG(
    dag_id='afa_daily_agent',
    default_args=daily_agent_default_args,
    description='AFA Daily Agent - Paper Trading Mode',
    schedule_interval='45 15 * * 1-5',  # 15:45 IST on weekdays (Mon-Fri)
    timezone='Asia/Kolkata',
    catchup=False,
    tags=['afa', 'finance', 'daily', 'agent'],
    max_active_runs=1,
) as afa_daily_agent:

    daily_agent_task = PythonOperator(
        task_id='daily_agent',
        python_callable=run_daily_agent_job,
        provide_context=True,
        op_kwargs={},
    )

    daily_agent_task


# DAG 2: afa_backtest_small - Manually triggered, 1 year, 20 tickers
with DAG(
    dag_id='afa_backtest_small',
    default_args=small_backtest_default_args,
    description='AFA Small Backtest - 1 Year, 20 Tickers',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['afa', 'finance', 'backtest', 'small'],
    max_active_runs=1,
) as afa_backtest_small:

    backtest_small_task = PythonOperator(
        task_id='backtest_small',
        python_callable=run_backtest_job,
        provide_context=True,
        op_kwargs={},
        execution_timeout=timedelta(hours=1),
        retries=1,
    )

    backtest_small_task


# DAG 3: afa_backtest_full - Manually triggered, 5 years, all tickers
with DAG(
    dag_id='afa_backtest_full',
    default_args=full_backtest_default_args,
    description='AFA Full Backtest - 5 Years, All Tickers',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['afa', 'finance', 'backtest', 'full'],
    max_active_runs=1,
) as afa_backtest_full:

    backtest_full_task = PythonOperator(
        task_id='backtest_full',
        python_callable=run_full_backtest_job,
        provide_context=True,
        op_kwargs={},
        execution_timeout=timedelta(hours=6),
        retries=0,
    )

    backtest_full_task
