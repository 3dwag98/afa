"""Tests for device resolution.

The bug these pin down: asking for `--device cuda` on a machine without a
usable CUDA build used to return `torch.device('cuda')` anyway. The CLI then
printed "Selected device: cuda", warned that CUDA was unavailable, warned
again that it was falling back, announced "Starting training with device:
cuda", and the trainer finally printed "Selected device: cpu" — four
contradictory lines for one decision.
"""

import pytest

torch = pytest.importorskip("torch")

from portfolio_agent.utils.device import (
    cuda_unavailable_reason,
    describe_devices,
    get_device,
    reset_announcements,
    resolve_device,
)


@pytest.fixture(autouse=True)
def _fresh_announcements():
    """Messages are printed once per process; reset between tests."""
    reset_announcements()
    yield
    reset_announcements()


class TestResolveDevice:
    def test_cpu_is_always_available(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_never_raises(self):
        assert resolve_device("auto").type in ("cuda", "mps", "cpu")

    def test_unavailable_cuda_resolves_to_cpu(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)
        assert resolve_device("cuda").type == "cpu"

    def test_available_cuda_resolves_to_cuda(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: True)
        assert resolve_device("cuda").type == "cuda"

    def test_unavailable_mps_resolves_to_cpu(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.mps_is_available", lambda: False)
        assert resolve_device("mps").type == "cpu"

    def test_auto_prefers_cuda_then_mps(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: True)
        monkeypatch.setattr("portfolio_agent.utils.device.mps_is_available", lambda: True)
        assert resolve_device("auto").type == "cuda"

        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)
        assert resolve_device("auto").type == "mps"

    def test_unknown_device_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_device("gpu0")

    def test_case_and_whitespace_tolerated(self):
        assert resolve_device(" CPU ").type == "cpu"


class TestGetDevice:
    def test_reports_the_device_it_returns(self, monkeypatch, capsys):
        """The printed device must be the device that is returned."""
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)

        device = get_device("cuda")
        out = capsys.readouterr().out

        assert device.type == "cpu"
        assert "Selected device: cpu" in out
        assert "Selected device: cuda" not in out

    def test_downgrade_explains_why_and_how_to_fix(self, monkeypatch, capsys):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)

        get_device("cuda")
        out = capsys.readouterr().out

        assert "falling back to CPU" in out
        assert "gpu-check" in out

    def test_messages_are_not_repeated(self, monkeypatch, capsys):
        """get_device() is called from CLI, trainer and dataloaders in one run."""
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)

        get_device("cuda")
        capsys.readouterr()

        get_device("cuda")
        get_device("cuda")
        assert capsys.readouterr().out == ""

    def test_verbose_false_is_silent(self, capsys):
        get_device("cpu", verbose=False)
        assert capsys.readouterr().out == ""

    def test_no_warning_when_request_is_satisfied(self, capsys):
        get_device("cpu")
        out = capsys.readouterr().out
        assert "Warning" not in out


class TestDiagnostics:
    def test_cpu_only_build_names_the_real_problem(self, monkeypatch):
        """The common Windows case: `uv sync --extra gpu` gives a CPU-only wheel."""
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)
        monkeypatch.setattr(torch.version, "cuda", None, raising=False)

        reason = "\n".join(cuda_unavailable_reason())

        assert "CPU-only build" in reason
        assert "download.pytorch.org" in reason

    def test_hidden_devices_are_called_out(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: False)
        monkeypatch.setattr(torch.version, "cuda", "12.6", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        reason = "\n".join(cuda_unavailable_reason())

        assert "CUDA_VISIBLE_DEVICES" in reason

    def test_no_reason_when_cuda_works(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.device.cuda_is_available", lambda: True)
        assert cuda_unavailable_reason() == []

    def test_describe_devices_reports_build_and_availability(self):
        lines = "\n".join(describe_devices())
        assert "PyTorch version" in lines
        assert "CUDA available" in lines
