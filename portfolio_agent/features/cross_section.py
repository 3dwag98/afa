"""A registry for features that need the whole universe, not one ticker.

`features/registry.py` binds one shape: ``Series = f(one_ticker_ohlcv)``. There
is no date argument and no universe, because a moving average does not need
one. That covers every feature the platform had — and it is why
`features/market_relative.py` was written *outside* the registry, is not
exported by `features/__init__.py`, re-implements the lag convention by hand,
and is reached by importing it directly inside a strategy method.

The shape a cross-sectional feature needs is different:

    DataFrame(date x symbol) = f(panel of date x symbol inputs)

Idiosyncratic volatility is the residual against *the cross-section's own*
market. Beta is measured against it. A characteristic ranked within its sector
needs every peer on the date. None of that is expressible one ticker at a time,
and every fundamental ratio worth having is peer-relative — so this registry is
the prerequisite for that work rather than a tidying of what exists.

Two things are enforced here rather than described in a comment
-------------------------------------------------------------
**The lag.** `technical.py` opens by declaring that every feature shifts its
input, and then each function does it individually — twenty-two chances to
forget. Here the decorator shifts the input frames before the function sees
them, so a feature *cannot* read the session it is used to decide. A feature
that genuinely needs the current bar passes ``lag=0``, which is a written-down
decision and greppable, not an omission.

**The warm-up.** Declared constants go stale the moment a feature with a longer
window is added — the defect T23 removed for the single-name registry. The same
probe answer is available here: build the feature once on a synthetic universe
and see which row it first resolves on.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Rows a cross-sectional feature is allowed to need before `warmup_rows` gives
#: up on it, and the height of the synthetic probe panel.
_PROBE_ROWS = 600

#: Symbols in the probe panel. A residual against a one-name "cross-section" is
#: the return itself, and several features refuse below a floor, so the probe
#: has to be wide enough to be a real cross-section.
_PROBE_SYMBOLS = 12


@dataclass(frozen=True)
class CrossSectionPanel:
    """Wide `(date x symbol)` views of the universe on one aligned index.

    Args:
        columns: Per-name input frames, keyed by the column they came from
            (``"close"``, ``"volume"``, later ``"book_value"``). Already
            lagged by the decorator — a feature body must not shift again.
        benchmark: The market series, when the caller has a real index.
            `None` means the feature should fall back to the cross-section's
            own composite, which is what `market_composite` is for.
    """

    columns: Mapping[str, pd.DataFrame]
    benchmark: Optional[pd.Series] = None

    def get(self, column: str) -> pd.DataFrame:
        """One input frame, or a failure that names what is missing.

        Raises:
            KeyError: If the caller did not supply this column. The feature
                declared it in `inputs`, so this is a builder-side mistake and
                is worth saying so rather than returning an empty frame that
                would score as "no signal".
        """
        if column not in self.columns:
            raise KeyError(
                f"Panel has no '{column}' column. Available: "
                f"{sorted(self.columns)}. A cross-sectional feature declares "
                f"what it reads in `inputs=`, and the builder supplies those."
            )
        return self.columns[column]

    @property
    def symbols(self) -> Tuple[str, ...]:
        for frame in self.columns.values():
            return tuple(frame.columns)
        return ()

    @property
    def index(self) -> pd.Index:
        for frame in self.columns.values():
            return frame.index
        return pd.Index([])

    def __len__(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class CrossSectionalFeature:
    """A registered feature plus the contract it was registered under."""

    name: str
    func: Callable[[CrossSectionPanel], pd.DataFrame]
    inputs: Tuple[str, ...]
    lag: int

    def __call__(self, panel: CrossSectionPanel) -> pd.DataFrame:
        return self.func(panel)


_REGISTRY: Dict[str, CrossSectionalFeature] = {}


def register_cross_sectional_feature(
    name: str,
    *,
    inputs: Sequence[str],
    lag: int = 1,
) -> Callable:
    """Register a `(date x symbol) -> (date x symbol)` feature.

    Args:
        name: Registry name. Shares a namespace with nothing — the single-name
            registry is separate, and a name may not be in both (asserted in
            the tests, because a caller routing by name would otherwise pick
            whichever registry it happened to check first).
        inputs: Per-name columns the feature reads. Declared rather than
            discovered so the builder knows what to pivot, and so a caller who
            cannot supply one fails before any computation.
        lag: Bars to shift each input before the feature sees it. **1 is the
            convention and the default**: a value used to decide date `t` must
            not have read `t`. Pass 0 only for a quantity that is genuinely
            known at the decision (the reference close is the one such case in
            the single-name registry), and expect to justify it — a test
            enumerates every `lag=0` feature.

    Raises:
        ValueError: On a duplicate name, an empty `inputs`, or a negative lag.
            All three are programming errors that would otherwise surface as a
            silently wrong number much later.

    Example:
        >>> @register_cross_sectional_feature("beta_60", inputs=("close",))
        ... def beta_60(panel):
        ...     returns = panel.get("close").pct_change()   # already lagged
        ...     return rolling_beta(returns, window=60)
    """
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError(
            f"Cross-sectional feature '{name}' declares no inputs. It would "
            f"receive an empty panel and could only return a constant."
        )
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag} for '{name}'")
    if name in _REGISTRY:
        raise ValueError(
            f"Cross-sectional feature '{name}' is already registered. Two "
            f"definitions under one name is how the platform ended up "
            f"reporting a metric four different ways."
        )

    def decorator(func: Callable[[CrossSectionPanel], pd.DataFrame]) -> Callable:
        @functools.wraps(func)
        def lagged(panel: CrossSectionPanel) -> pd.DataFrame:
            # The enforcement. The body receives shifted inputs and cannot
            # reach the unshifted ones, so the convention holds by
            # construction rather than by twenty-two individual `.shift(1)`
            # calls that each have to be right.
            if lag:
                shifted = {
                    column: frame.shift(lag)
                    for column, frame in panel.columns.items()
                }
                benchmark = (
                    panel.benchmark.shift(lag)
                    if panel.benchmark is not None
                    else None
                )
                panel = CrossSectionPanel(columns=shifted, benchmark=benchmark)
            return func(panel)

        _REGISTRY[name] = CrossSectionalFeature(
            name=name, func=lagged, inputs=inputs, lag=lag
        )
        # The undecorated function is returned so the module keeps a direct,
        # unlagged handle for its own tests. Callers go through the registry.
        return func

    return decorator


def get_cross_sectional_feature(name: str) -> CrossSectionalFeature:
    """One registered feature.

    Raises:
        KeyError: If the name is not registered, listing what is.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Cross-sectional feature '{name}' not found. Available: "
            f"{sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_cross_sectional_features() -> list[str]:
    """Every registered cross-sectional feature name."""
    return sorted(_REGISTRY)


def is_cross_sectional_feature(name: str) -> bool:
    """Whether `name` names a cross-sectional feature rather than a per-ticker one."""
    return name in _REGISTRY


def required_columns(names: Sequence[str]) -> Tuple[str, ...]:
    """Every per-name input the requested features read, deduplicated.

    What the builder has to pivot. Computed from the registrations rather than
    passed alongside them, so adding a feature that reads `volume` does not
    also require finding every call site.
    """
    columns: list[str] = []
    for name in names:
        for column in get_cross_sectional_feature(name).inputs:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def panel_from_frames(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    benchmark: Optional[pd.Series] = None,
) -> CrossSectionPanel:
    """Pivot per-symbol frames into the wide panel the registry expects.

    Every caller already holds `{symbol: DataFrame}` — the harness builds one,
    the backtest engine builds one, `score_batch` receives one — and each was
    going to pivot it by hand. Doing it once means they agree about alignment:
    the union of every symbol's index, sorted, with missing cells left NaN
    rather than forward-filled. Forward-filling here would manufacture a price
    on a day a stock did not trade, which is exactly the observation a
    liquidity screen exists to catch.

    Args:
        frames_by_symbol: Per-symbol frames, each indexed by date.
        columns: Column names to pivot. Symbols missing a column are dropped
            from that column's frame rather than filled.
        benchmark: Optional market series, passed through to the panel.

    Returns:
        A `CrossSectionPanel` on the union index.
    """
    columns = list(columns)
    usable = {
        symbol: frame
        for symbol, frame in frames_by_symbol.items()
        if frame is not None and not frame.empty
    }

    index = pd.Index([])
    for frame in usable.values():
        index = index.union(frame.index)
    index = index.sort_values()

    pivoted: Dict[str, pd.DataFrame] = {}
    for column in columns:
        series_by_symbol = {
            symbol: frame[column]
            for symbol, frame in usable.items()
            if column in frame.columns
        }
        pivoted[column] = (
            pd.DataFrame(series_by_symbol).reindex(index)
            if series_by_symbol
            else pd.DataFrame(index=index, dtype=float)
        )

    aligned = benchmark.reindex(index) if benchmark is not None else None
    return CrossSectionPanel(columns=pivoted, benchmark=aligned)


def build_cross_section(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    feature_names: Sequence[str],
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """Build the named cross-sectional features from per-symbol frames.

    The counterpart to `pipeline.build_features`, and the entry point callers
    should use: it resolves each name, pivots exactly the columns those
    features declared, and applies each feature's own lag.

    Args:
        frames_by_symbol: Per-symbol frames, each indexed by date and carrying
            at least the columns the requested features declare.
        feature_names: Registered cross-sectional feature names.
        benchmark: Optional market series. Features fall back to the
            cross-section's own composite when this is None.

    Returns:
        `{feature_name: (date x symbol) frame}`.

    Raises:
        KeyError: If a name is not registered.
    """
    names = list(feature_names)
    if not names:
        return {}

    panel = panel_from_frames(
        frames_by_symbol, required_columns(names), benchmark=benchmark
    )
    return {name: get_cross_sectional_feature(name)(panel) for name in names}


def latest_values(frame: pd.DataFrame) -> Dict[str, float]:
    """The last defined value per symbol, as a plain mapping.

    What a ranking strategy actually wants out of a `(date x symbol)` frame.
    Symbols whose value is NaN on the final row are **omitted rather than
    filled**: a strategy that substituted a fallback would rank a name it could
    not measure alongside names it could, and T14 established that mixing two
    measures into one ranking is harder to notice than a thin cross-section.
    """
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {
        str(symbol): float(value)
        for symbol, value in row.items()
        if pd.notna(value) and np.isfinite(value)
    }


def warmup_rows(feature_names: Sequence[str]) -> int:
    """Rows of history before every named cross-sectional feature is defined.

    Measured on a synthetic universe, for the reason T23 established for the
    single-name registry: a declared constant is right until someone registers
    a feature with a longer window, and then it is silently wrong.

    Raises:
        KeyError: If a name is not registered.
    """
    return max((_warmup_for(name) for name in feature_names), default=0)


@lru_cache(maxsize=None)
def _warmup_for(name: str) -> int:
    """First row at which one cross-sectional feature is defined on the probe.

    Defined means *for the median symbol*, not for any symbol: a feature can
    resolve early for one lucky name while the cross-section it is meant to
    rank is still mostly NaN, and ranking a handful of names against each other
    is the thin-cross-section failure `MIN_CROSS_SECTION_NAMES` exists to
    refuse.
    """
    feature = get_cross_sectional_feature(name)
    panel = _probe_panel(feature.inputs)
    built = feature(panel)
    if built.empty:
        return _PROBE_ROWS

    defined = built.notna().sum(axis=1) >= max(2, built.shape[1] // 2)
    if not defined.any():
        return _PROBE_ROWS
    # +1 because a value at position p means p+1 rows were needed for it.
    return int(defined.idxmax()) + 1


@lru_cache(maxsize=None)
def _probe_panel(columns: Tuple[str, ...]) -> CrossSectionPanel:
    """A synthetic universe wide and long enough to warm any feature.

    Geometric random walks with a shared market component, because a feature
    that decomposes return into market and residual would report a degenerate
    answer on independent walks (zero beta everywhere) and a different
    degenerate answer on a constant series (zero variance, never warm).
    """
    rng = np.random.default_rng(0)
    n, k = _PROBE_ROWS, _PROBE_SYMBOLS
    index = pd.RangeIndex(n)

    market = rng.normal(0.0004, 0.009, n)
    frames: Dict[str, pd.DataFrame] = {}
    closes = pd.DataFrame(
        {
            f"P{i}": 100.0
            * np.exp(np.cumsum(market * (0.6 + 0.1 * i) + rng.normal(0, 0.008, n)))
            for i in range(k)
        },
        index=index,
    )

    for column in columns:
        if column == "close":
            frames[column] = closes
        elif column == "volume":
            frames[column] = pd.DataFrame(
                rng.integers(1e5, 1e6, (n, k)).astype(float),
                index=index, columns=closes.columns,
            )
        else:
            # An unrecognized column still gets a plausible positive series, so
            # a newly registered feature warms up rather than reporting the cap
            # merely because this probe has not been taught about its input.
            frames[column] = closes * (1.0 + rng.normal(0, 0.05, (n, k)))

    return CrossSectionPanel(columns=frames, benchmark=None)
