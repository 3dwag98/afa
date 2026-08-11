"""Worker-count resolution that accounts for how processes start.

Why this exists. The platform asks for parallelism in two places — a
ProcessPoolExecutor that featurizes tickers, and PyTorch DataLoader workers —
and both defaulted to "as many as there are CPUs" with no reference to the
operating system. On Linux and macOS that is roughly free: `fork` gives each
child a copy-on-write view of the parent, so twelve workers do not cost twelve
copies of the interpreter.

**On Windows there is no fork.** Every worker is a fresh interpreter that
re-imports the parent module, and this project's workers import torch, pandas
and pyarrow — on the order of 300-800 MB of resident memory each before any
data is touched. Twelve of those on a 16 GB machine does not fail cleanly with
an out-of-memory error; it drives the system into the page file, where
"training" becomes disk thrashing and the run appears to hang rather than
crash. A 6 GB GPU is irrelevant to this: the exhausted resource is host RAM.

The same reasoning applies to DataLoader workers, which additionally receive a
copy of the dataset tensors rather than a shared view.

So: cap process pools hard on Windows, and use in-process loading for the
DataLoader there. The cost is some wall-clock time on a machine that was
otherwise going to page; the benefit is a run that finishes.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Above this, extra worker processes stop paying for themselves even where
# fork is available: the work is bounded by parquet reads and pandas, and each
# additional interpreter still costs memory and startup.
MAX_PROCESS_WORKERS = 8

# Windows spawns rather than forks, so each worker re-imports torch/pandas and
# pays for it in resident memory. Two is enough to overlap I/O with compute
# without putting a 16 GB machine into the page file.
MAX_PROCESS_WORKERS_WINDOWS = 2


def is_windows() -> bool:
    """Whether processes here are spawned rather than forked."""
    return sys.platform.startswith("win")


def cpu_count() -> int:
    """Usable CPU count, never below 1."""
    return max(1, os.cpu_count() or 1)


def resolve_process_workers(requested: Optional[int] = None) -> int:
    """Worker count for a ProcessPoolExecutor, capped for the platform.

    Args:
        requested: An explicit count from config, or None for "decide for me".
            An explicit request is still capped — the cap exists because the
            memory cost is real, not because the caller was assumed careless,
            and a config written on a Linux box gets copied to a Windows one.

    Returns:
        A worker count of at least 1.
    """
    ceiling = MAX_PROCESS_WORKERS_WINDOWS if is_windows() else MAX_PROCESS_WORKERS
    if requested is not None and requested > 0:
        return max(1, min(int(requested), ceiling))
    return max(1, min(cpu_count(), ceiling))


def resolve_dataloader_workers(requested: Optional[int] = None) -> int:
    """Worker count for a torch DataLoader.

    Zero on Windows, deliberately. A DataLoader worker there is a spawned
    interpreter that re-imports torch *and* receives a copy of the dataset
    tensors; with a stacked multi-ticker panel that is the difference between
    training and paging. In-process loading is slower per batch and finishes.
    """
    if is_windows():
        return 0
    if requested is None:
        return 2
    return max(0, min(int(requested), 4))


def describe_worker_plan(process_workers: int, dataloader_workers: int) -> str:
    """One line explaining the choice, for the run's startup output.

    Printed rather than logged because a user watching a run stall wants to
    know that the platform noticed the constraint, not to discover it later in
    a log file.
    """
    if not is_windows():
        return (
            f"Parallelism: {process_workers} data-loading process(es), "
            f"{dataloader_workers} DataLoader worker(s)"
        )
    return (
        f"Parallelism: {process_workers} data-loading process(es), "
        f"{dataloader_workers} DataLoader worker(s) — capped for Windows, where "
        f"each worker is a fresh interpreter that re-imports torch rather than a "
        f"copy-on-write fork"
    )
