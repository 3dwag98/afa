"""Data-quality invariants and store inventory.

    portfolio-agent data status      # what is actually on disk
    portfolio-agent data validate    # exits non-zero on a structural violation

Bad bars become bad labels, and a bad label is indistinguishable from a
hard-to-forecast day in every metric downstream — so the data layer is the only
place these can be caught. `invariants.py` holds the checks and the
structural/advisory split that decides which of them may fail a build;
`status.py` answers "what span, what coverage, what is missing".

Named `data_quality` rather than `validation` to keep it distinct from
`portfolio_agent/validation/`, which is cross-validation in the model-fitting
sense. The two words mean different things and sharing a package name would
make every import ambiguous at a glance.
"""

from .invariants import (
    ADVISORY,
    STRUCTURAL,
    IngestRejected,
    ValidationReport,
    Violation,
    assert_writable,
    check_adjustment_factor,
    check_calendar_coverage,
    check_duplicate_dates,
    check_extreme_returns,
    check_history_length,
    check_monotonic_index,
    check_ohlc_ordering,
    check_price_positivity,
    infer_trading_calendar,
    validate_frame,
    validate_store,
)
from .status import StoreStatus, SymbolStatus, collect_status, load_store

__all__ = [
    "ADVISORY",
    "STRUCTURAL",
    "IngestRejected",
    "StoreStatus",
    "SymbolStatus",
    "ValidationReport",
    "Violation",
    "assert_writable",
    "check_adjustment_factor",
    "check_calendar_coverage",
    "check_duplicate_dates",
    "check_extreme_returns",
    "check_history_length",
    "check_monotonic_index",
    "check_ohlc_ordering",
    "check_price_positivity",
    "collect_status",
    "infer_trading_calendar",
    "load_store",
    "validate_frame",
    "validate_store",
]
