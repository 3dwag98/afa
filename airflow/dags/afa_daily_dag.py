"""
AFA Daily Orchestrator DAG

This DAG runs the Autonomous Financial Advisor daily logic on weekdays at 15:45 IST.
It bridges the Airflow scheduler with the core AFA Python logic.
"""

import sys
import logging
from datetime import datetime

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'afa',
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
    'start_date': pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    'catchup': False,
}

def run_daily_agent(**context):
    """
    Wrapper function to execute the AFA daily orchestrator logic.
    
    This function:
    1. Adds the project path to sys.path for proper imports in Docker
    2. Imports the core orchestrator logic
    3. Executes the daily run
    4. Catches and logs exceptions without crashing the Airflow worker
    """
    try:
        # Crucial: Add project path for Docker volume mounts
        # In Docker, ./src is mounted to /opt/airflow/project/src
        # and PYTHONPATH is set to /opt/airflow/project
        project_path = '/opt/airflow/project'
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
            logger.info(f"Added {project_path} to sys.path")

        # Import the orchestrator logic
        # The src module is available via volume mount at /opt/airflow/project/src
        from src.orchestrator import run_orchestrator
        
        logger.info("Starting AFA daily orchestrator...")
        
        # Execute the daily loop
        # You can pass execution_date or other context variables if needed
        execution_date = context.get('execution_date')
        logger.info(f"Execution date: {execution_date}")
        
        result = run_orchestrator()
        
        logger.info(f"AFA daily orchestrator completed successfully: {result}")
        return result
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure src/orchestrator.py exists and contains run_orchestrator()")
        # Re-raise to mark task as failed in Airflow
        raise
        
    except Exception as e:
        logger.error(f"Error executing AFA daily orchestrator: {e}", exc_info=True)
        # Log the error but don't crash the worker unnecessarily
        # Re-raise to properly mark task as failed in Airflow UI
        raise

# Define the DAG
with DAG(
    dag_id='afa_daily_orchestrator',
    default_args=default_args,
    description='Autonomous Financial Advisor - Daily Orchestration',
    schedule_interval='45 15 * * 1-5',  # 15:45 IST on weekdays (Mon-Fri)
    catchup=False,
    tags=['afa', 'finance', 'daily'],
    max_active_runs=1,  # Prevent overlapping runs
) as dag:
    
    run_daily_agent_task = PythonOperator(
        task_id='run_daily_agent',
        python_callable=run_daily_agent,
        provide_context=True,
        op_kwargs={},
    )
    
    run_daily_agent_task
