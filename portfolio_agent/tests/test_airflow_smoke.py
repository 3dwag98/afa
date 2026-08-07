"""Smoke tests for Airflow setup.

These tests verify that all required Airflow files and configurations exist.
They do not require Docker, network, or Postgres to run.
"""

import os
import pytest


class TestAirflowSetup:
    """Test suite for Airflow setup verification."""

    def test_dockerfile_airflow_exists(self):
        """Test that Dockerfile.airflow exists."""
        dockerfile_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'Dockerfile.airflow'
        )
        assert os.path.exists(dockerfile_path), \
            f"Dockerfile.airflow should exist at {dockerfile_path}"

    def test_portfolio_agent_dag_exists(self):
        """Test that airflow/dags/portfolio_agent_dag.py exists."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'airflow', 'dags', 'portfolio_agent_dag.py'
        )
        assert os.path.exists(dag_path), \
            f"portfolio_agent_dag.py should exist at {dag_path}"

    def test_airflow_init_script_exists(self):
        """Test that airflow/scripts/airflow-init.sh exists."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'airflow', 'scripts', 'airflow-init.sh'
        )
        assert os.path.exists(script_path), \
            f"airflow-init.sh should exist at {script_path}"

    def test_docker_compose_contains_required_services(self):
        """Test that docker-compose.yml contains all required Airflow services."""
        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'docker-compose.yml'
        )
        assert os.path.exists(compose_path), \
            f"docker-compose.yml should exist at {compose_path}"

        with open(compose_path, 'r') as f:
            content = f.read()

        required_services = [
            'postgres',
            'airflow-init',
            'airflow-webserver',
            'airflow-scheduler',
            'airflow-test'
        ]

        for service in required_services:
            assert service in content, \
                f"docker-compose.yml should contain service '{service}'"

    def test_config_yaml_contains_scheduler_enabled(self):
        """Test that config.yaml contains scheduler_enabled setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )
        assert os.path.exists(config_path), \
            f"config.yaml should exist at {config_path}"

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'scheduler_enabled' in content, \
            "config.yaml should contain 'scheduler_enabled' setting"

    def test_config_yaml_contains_schedule_time_ist(self):
        """Test that config.yaml contains schedule_time_ist setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'schedule_time_ist' in content, \
            "config.yaml should contain 'schedule_time_ist' setting"

    def test_config_yaml_contains_schedule_outcome_time_ist(self):
        """Test that config.yaml contains schedule_outcome_time_ist setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'schedule_outcome_time_ist' in content, \
            "config.yaml should contain 'schedule_outcome_time_ist' setting"

    def test_config_yaml_contains_airflow_ui_enabled(self):
        """Test that config.yaml contains airflow_ui_enabled setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'airflow_ui_enabled' in content, \
            "config.yaml should contain 'airflow_ui_enabled' setting"

    def test_config_yaml_contains_airflow_webserver_port(self):
        """Test that config.yaml contains airflow_webserver_port setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'airflow_webserver_port' in content, \
            "config.yaml should contain 'airflow_webserver_port' setting"

    def test_config_yaml_contains_airflow_timezone(self):
        """Test that config.yaml contains airflow_timezone setting."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'airflow_timezone' in content, \
            "config.yaml should contain 'airflow_timezone' setting"
