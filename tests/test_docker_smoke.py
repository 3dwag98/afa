"""
Docker smoke tests for AFA project.

These tests validate Docker configuration without requiring Docker daemon.
Use pytest markers to run Docker-dependent tests separately.
"""

import os
import re
from pathlib import Path

import pytest
import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestRequiredFilesExist:
    """Test that all required project files exist."""

    def test_required_files_exist(self):
        """Assert these files exist: Dockerfile, docker-compose.yml, main.py, run_backtest.py, requirements.txt, config.yaml."""
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            "main.py",
            "run_backtest.py",
            "requirements.txt",
            "config.yaml",
        ]

        for filename in required_files:
            filepath = PROJECT_ROOT / filename
            assert filepath.exists(), f"Required file '{filename}' does not exist at {filepath}"
            assert filepath.is_file(), f"'{filename}' exists but is not a file"

    def test_directories_exist(self):
        """Assert these directories exist: data, output, logs, scripts."""
        required_dirs = [
            "data",
            "output",
            "logs",
            "scripts",
        ]

        for dirname in required_dirs:
            dirpath = PROJECT_ROOT / dirname
            assert dirpath.exists(), f"Required directory '{dirname}' does not exist at {dirpath}"
            assert dirpath.is_dir(), f"'{dirname}' exists but is not a directory"


class TestDockerComposeConfiguration:
    """Test docker-compose.yml configuration."""

    @pytest.fixture
    def compose_config(self):
        """Load and parse docker-compose.yml."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path, "r") as f:
            return yaml.safe_load(f)

    def test_docker_compose_has_agent_and_backtest(self, compose_config):
        """Assert services include: agent, backtest, test."""
        assert "services" in compose_config, "docker-compose.yml must have 'services' key"

        services = compose_config["services"]
        required_services = ["agent", "backtest", "test"]

        for service_name in required_services:
            assert service_name in services, f"Required service '{service_name}' not found in docker-compose.yml"

    def test_docker_compose_volumes(self, compose_config):
        """Assert agent and backtest services mount: ./data:/app/data, ./output:/app/output, ./logs:/app/logs."""
        services = compose_config.get("services", {})

        # Check agent service volumes
        assert "agent" in services, "agent service not found"
        agent_volumes = services["agent"].get("volumes", [])
        self._assert_volume_mounts(agent_volumes, "agent")

        # Check backtest service volumes
        assert "backtest" in services, "backtest service not found"
        backtest_volumes = services["backtest"].get("volumes", [])
        self._assert_volume_mounts(backtest_volumes, "backtest")

    def _assert_volume_mounts(self, volumes, service_name):
        """Helper to assert required volume mounts exist."""
        required_mounts = [
            "./data:/app/data",
            "./output:/app/output",
            "./logs:/app/logs",
        ]

        # Normalize volumes (handle both string and dict formats)
        normalized_volumes = []
        for vol in volumes:
            if isinstance(vol, str):
                normalized_volumes.append(vol)
            elif isinstance(vol, dict):
                # Handle long-form volume syntax
                source = vol.get("source", "")
                target = vol.get("target", "")
                if source and target:
                    normalized_volumes.append(f"{source}:{target}")

        for mount in required_mounts:
            assert any(
                mount in v or v.startswith(mount.split(":")[0])
                for v in normalized_volumes
            ), f"Service '{service_name}' missing required volume mount: {mount}"


class TestMakefileTargets:
    """Test Makefile has required targets."""

    def test_makefile_has_required_targets(self):
        """Assert these targets exist: agent, backtest, test, build."""
        makefile_path = PROJECT_ROOT / "Makefile"
        assert makefile_path.exists(), "Makefile does not exist"

        with open(makefile_path, "r") as f:
            makefile_content = f.read()

        required_targets = ["agent", "backtest", "test", "build"]

        # Match target definitions (target name followed by colon)
        for target in required_targets:
            # Look for target pattern: target_name: (with optional dependencies)
            pattern = rf"^{re.escape(target)}\s*:"
            assert re.search(pattern, makefile_content, re.MULTILINE), (
                f"Required target '{target}' not found in Makefile"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
