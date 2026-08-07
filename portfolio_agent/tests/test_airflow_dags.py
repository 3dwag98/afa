"""
Tests for Airflow DAGs

These tests verify:
- All three DAGs exist
- Schedule intervals are correct
- Tasks are properly configured
"""

import os
import pytest

# Set AIRFLOW_HOME before importing airflow modules
os.environ['AIRFLOW_HOME'] = '/workspace/portfolio_agent/airflow'


def get_dagbag():
    """Load the DAG bag for testing."""
    from airflow.models import DagBag
    return DagBag('/workspace/portfolio_agent/airflow/dags')


class TestPortfolioDAGs:
    """Test suite for Portfolio Agent DAGs."""

    @pytest.fixture(autouse=True)
    def setup_dagbag(self):
        """Setup dagbag for each test."""
        self.dagbag = get_dagbag()

    def test_daily_dag_exists(self):
        """Test that portfolio_daily_run DAG exists."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None, "portfolio_daily_run DAG should exist"

    def test_outcomes_dag_exists(self):
        """Test that portfolio_update_outcomes DAG exists."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None, "portfolio_update_outcomes DAG should exist"

    def test_relearn_dag_exists(self):
        """Test that portfolio_relearn DAG exists."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None, "portfolio_relearn DAG should exist"

    def test_daily_dag_schedule(self):
        """Test daily DAG schedule contains '45 15' for default config."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None
        schedule = dag.schedule
        # Default config is 15:45 -> "45 15 * * 1-5"
        assert "45 15" in schedule, f"Daily schedule should contain '45 15', got {schedule}"

    def test_outcome_dag_schedule(self):
        """Test outcome DAG schedule contains '0 16' for default config."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None
        schedule = dag.schedule
        # Default config is 16:00 -> "0 16 * * 1-5"
        assert "0 16" in schedule or "00 16" in schedule, f"Outcome schedule should contain '0 16', got {schedule}"

    def test_relearn_dag_no_schedule(self):
        """Test relearn DAG has no schedule (manual trigger)."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None
        schedule = dag.schedule
        assert schedule is None, f"Relearn DAG should have no schedule, got {schedule}"

    def test_daily_dag_has_task(self):
        """Test daily DAG has task daily_run."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None
        task_ids = [task.task_id for task in dag.tasks]
        assert "daily_run" in task_ids, f"daily_run task should exist, got {task_ids}"

    def test_outcome_dag_has_task(self):
        """Test outcome DAG has task update_outcomes."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None
        task_ids = [task.task_id for task in dag.tasks]
        assert "update_outcomes" in task_ids, f"update_outcomes task should exist, got {task_ids}"

    def test_relearn_dag_has_task(self):
        """Test relearn DAG has task relearn."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None
        task_ids = [task.task_id for task in dag.tasks]
        assert "relearn" in task_ids, f"relearn task should exist, got {task_ids}"

    def test_daily_dag_tags(self):
        """Test daily DAG has correct tags."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None
        tags = dag.tags
        assert "portfolio" in tags
        assert "daily" in tags
        assert "india" in tags

    def test_outcome_dag_tags(self):
        """Test outcome DAG has correct tags."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None
        tags = dag.tags
        assert "portfolio" in tags
        assert "outcomes" in tags
        assert "india" in tags

    def test_relearn_dag_tags(self):
        """Test relearn DAG has correct tags."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None
        tags = dag.tags
        assert "portfolio" in tags
        assert "learning" in tags
        assert "manual" in tags

    def test_daily_dag_catchup_false(self):
        """Test daily DAG has catchup=False."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None
        assert dag.catchup is False

    def test_outcome_dag_catchup_false(self):
        """Test outcome DAG has catchup=False."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None
        assert dag.catchup is False

    def test_relearn_dag_catchup_false(self):
        """Test relearn DAG has catchup=False."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None
        assert dag.catchup is False

    def test_daily_dag_max_active_runs(self):
        """Test daily DAG has max_active_runs=1."""
        dag = self.dagbag.get_dag("portfolio_daily_run")
        assert dag is not None
        assert dag.max_active_runs == 1

    def test_outcome_dag_max_active_runs(self):
        """Test outcome DAG has max_active_runs=1."""
        dag = self.dagbag.get_dag("portfolio_update_outcomes")
        assert dag is not None
        assert dag.max_active_runs == 1

    def test_relearn_dag_max_active_runs(self):
        """Test relearn DAG has max_active_runs=1."""
        dag = self.dagbag.get_dag("portfolio_relearn")
        assert dag is not None
        assert dag.max_active_runs == 1
