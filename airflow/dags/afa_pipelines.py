"""
AFA Pipelines - Modern Airflow 2.x TaskFlow API DAGs

This module defines two DAGs for the Autonomous Financial Advisor:
1. afa_daily_pipeline - Daily weekday pipeline for running the agent
2. afa_training_pipeline - Manual trigger pipeline for model training and backtesting
"""

import pendulum
from airflow.sdk import dag, task
from airflow.providers.standard.operators.bash import BashOperator


def cli_command(args: str) -> BashOperator:
    """Helper function to create a BashOperator that runs CLI commands.
    
    Args:
        args: The CLI arguments to pass to portfolio_agent.cli
        
    Returns:
        BashOperator configured to run the CLI command
    """
    return BashOperator(
        task_id=args.replace(" ", "_").replace("-", "_"),
        bash_command=f"python -m portfolio_agent.cli {args}",
        env={"AFA_PAPER_TRADING_MODE": "true"}
    )


@dag(
    dag_id='afa_daily_pipeline',
    schedule='45 15 * * 1-5',  # 15:45 IST on weekdays (Mon-Fri)
    start_date=pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    catchup=False,
    max_active_runs=1,
    default_args={
        'retries': 1,
    },
    tags=['afa', 'finance', 'daily'],
    description='AFA Daily Pipeline - Download data and run agent on weekdays',
)
def afa_daily_pipeline():
    """Daily pipeline that downloads market data and runs the portfolio agent."""
    
    # Define tasks using the CLI helper
    download_data_task = cli_command("download-data")
    run_agent_task = cli_command("run-agent")
    
    # Set dependencies
    download_data_task >> run_agent_task


@dag(
    dag_id='afa_training_pipeline',
    schedule=None,  # Manual trigger only
    start_date=pendulum.datetime(2024, 1, 1, tz='Asia/Kolkata'),
    catchup=False,
    max_active_runs=1,
    default_args={
        'retries': 1,
    },
    tags=['afa', 'finance', 'training', 'ml'],
    description='AFA Training Pipeline - Download data, build features, train model, and run backtest',
)
def afa_training_pipeline():
    """Training pipeline for model development and backtesting."""
    
    # Define tasks using the CLI helper
    download_data_task = cli_command("download-data")
    build_features_task = cli_command("build-features")
    train_model_task = cli_command("train --model lstm --device auto")
    run_backtest_task = cli_command("backtest")
    
    # Set dependencies
    download_data_task >> build_features_task >> train_model_task >> run_backtest_task


# Instantiate the DAGs
daily_dag = afa_daily_pipeline()
training_dag = afa_training_pipeline()
