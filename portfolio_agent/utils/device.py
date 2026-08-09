"""Device utilities for GPU/CPU selection in PyTorch.

Single source of truth for "which device is this process actually going to
use". Every caller (CLI, trainer, dataloader construction, ML strategy
inference) resolves through :func:`get_device`, so the device that gets
printed is always the device that gets used — a requested-but-unavailable
accelerator is downgraded *here*, once, rather than being reported as
selected and silently swapped later.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional

import torch

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")

# Messages are announced at most once per process: get_device() is called from
# the CLI, the trainer and the dataloader factory within a single run, and
# repeating the same banner (or the same warning) three times is what made the
# original CUDA fallback so confusing to read.
_ANNOUNCED: set = set()


def _announce(key: str, *lines: str) -> None:
    """Print a message once per process, keyed by `key`."""
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    for line in lines:
        print(line)


def reset_announcements() -> None:
    """Forget which messages have been printed (used by tests)."""
    _ANNOUNCED.clear()


def cuda_is_available() -> bool:
    """Whether a CUDA device is usable by this PyTorch build right now."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive: broken driver/runtime
        return False


def mps_is_available() -> bool:
    """Whether Apple Metal Performance Shaders are usable.

    `torch.backends.mps` does not exist on every build, so this is guarded
    rather than accessed directly (the old code raised AttributeError on
    builds without MPS while merely trying to pick a device).
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def _nvidia_smi_present() -> bool:
    """Whether an NVIDIA driver utility is on PATH and responds."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - defensive
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def cuda_unavailable_reason() -> List[str]:
    """Explain why CUDA is not usable, with the concrete fix.

    Returns a list of message lines (empty when CUDA *is* usable). The common
    case on Windows is by far the first one: the default PyPI `torch` wheel
    for Windows is CPU-only, so `uv sync --extra gpu` installs a build whose
    `torch.version.cuda` is None and which can never see a GPU no matter how
    good the card is.
    """
    if cuda_is_available():
        return []

    lines: List[str] = []
    build_cuda = getattr(torch.version, "cuda", None)

    if build_cuda is None:
        lines.append(
            f"  Installed PyTorch ({torch.__version__}) is a CPU-only build "
            "(torch.version.cuda is None), so it cannot use any GPU."
        )
        if _nvidia_smi_present():
            lines.append("  An NVIDIA driver IS present — only the PyTorch build is wrong.")
        lines.append("  Install a CUDA build of PyTorch, e.g. for CUDA 12.6:")
        lines.append(
            "    uv pip install --force-reinstall --index-url "
            "https://download.pytorch.org/whl/cu126 torch"
        )
        lines.append(
            "  Pick the index URL matching your driver at "
            "https://pytorch.org/get-started/locally/"
        )
        return lines

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        lines.append(
            f"  PyTorch was built with CUDA {build_cuda}, but CUDA_VISIBLE_DEVICES="
            f"{visible!r} hides every GPU from this process."
        )
        lines.append("  Unset CUDA_VISIBLE_DEVICES (or set it to a real device index).")
        return lines

    lines.append(
        f"  PyTorch was built with CUDA {build_cuda}, but no CUDA-capable GPU is "
        "visible to the driver."
    )
    if _nvidia_smi_present():
        lines.append(
            "  nvidia-smi works, so the driver is installed but is likely older than "
            f"the CUDA {build_cuda} runtime this PyTorch build needs — update the "
            "NVIDIA driver, or install a PyTorch build for an older CUDA version."
        )
    else:
        lines.append(
            "  nvidia-smi did not report a GPU: check that an NVIDIA GPU is present "
            "and its driver is installed (WSL users also need the Windows driver)."
        )
    return lines


def describe_devices() -> List[str]:
    """Human-readable summary of every compute device this process can see."""
    lines = [
        f"PyTorch version:  {torch.__version__}",
        f"Built with CUDA:  {getattr(torch.version, 'cuda', None) or 'no (CPU-only build)'}",
        f"CUDA available:   {cuda_is_available()}",
        f"MPS available:    {mps_is_available()}",
    ]

    if cuda_is_available():
        lines.append(f"CUDA devices:     {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            lines.append(
                f"  [{index}] {props.name} — {props.total_memory / (1024 ** 3):.2f} GB, "
                f"compute capability {props.major}.{props.minor}"
            )
    else:
        lines.extend(cuda_unavailable_reason())

    return lines


def resolve_device(config_device: str = "auto") -> torch.device:
    """Resolve a configured device string to a device that actually works.

    Unlike a bare ``torch.device(name)``, this never returns an accelerator
    that PyTorch cannot use: an unavailable request is downgraded to CPU.

    Args:
        config_device: One of "auto", "cuda", "mps", "cpu". Anything else
            raises ValueError rather than silently producing a broken device.

    Returns:
        A usable torch.device.
    """
    requested = (config_device or "auto").strip().lower()
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device {config_device!r}. Valid options: {', '.join(VALID_DEVICES)}"
        )

    if requested == "auto":
        if cuda_is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not cuda_is_available():
        return torch.device("cpu")
    if requested == "mps" and not mps_is_available():
        return torch.device("cpu")

    return torch.device(requested)


def get_device(config_device: str = "auto", verbose: bool = True) -> torch.device:
    """Get the device to run on, reporting the decision exactly once.

    Args:
        config_device: "auto", "cuda", "mps", or "cpu".
        verbose: Print the selected device (and, on a downgrade, why plus how
            to fix it). Messages are de-duplicated per process, so calling this
            from several modules in one run prints one banner, not several.

    Returns:
        torch.device that is guaranteed usable by this PyTorch build.
    """
    requested = (config_device or "auto").strip().lower()
    device = resolve_device(requested)

    if not verbose:
        return device

    downgraded = requested in ("cuda", "mps") and device.type != requested

    if downgraded:
        _announce(
            f"downgrade:{requested}",
            f"Warning: {requested.upper()} was requested but is not available — "
            f"falling back to {device.type.upper()}.",
            *(cuda_unavailable_reason() if requested == "cuda" else []),
            "  Run `portfolio-agent gpu-check` for full device diagnostics.",
        )

    _announce(f"selected:{device.type}", f"Selected device: {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        _announce(
            "cuda-details",
            f"GPU: {props.name}",
            f"Total GPU Memory: {props.total_memory / (1024 ** 3):.2f} GB",
            f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB",
        )

    return device
