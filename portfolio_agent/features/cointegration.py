"""Cointegrated pairs: the relationship between two tickers, as a per-name score.

`docs/QUANT_RESEARCH.md` §7 recorded this as scoped out, and was precise about
why — an architectural gap rather than a research gap:

> Every strategy in this platform scores *one ticker at a time* against a
> shared context; pairs trading fundamentally needs a *relationship between two
> tickers* … Supporting it properly needs (a) a pair-selection step
> (cointegration screening across O(n^2) candidate pairs) that doesn't fit
> `BaseStrategy`'s per-ticker interface, and (b) either accepting long-only
> spread trades … or a deliberate, explicit exception to the short-selling
> guardrail.

**(a) is what T24's cross-sectional registry is for.** A feature that receives
the whole `(date x symbol)` panel can screen pairs internally and emit a
per-symbol number — how cheap each name is against its own partner. The
per-ticker interface never has to change; it was the *feature* layer that could
not express the question.

**(b) is resolved by taking the long leg only, and saying so.** When the spread
is stretched, this buys the undervalued name and does not short the expensive
one. That is a real and costly concession, not a technicality: the short leg is
what makes textbook pairs trading market-neutral, and without it this is a
relative-value signal carrying full market exposure. Every result it produces
should be read that way, and `PAIRS_NOT_NEUTRAL_NOTE` exists so a report says
it rather than implying otherwise by using the words "pairs trading".

Two traps this module exists to avoid
-------------------------------------
**Look-ahead through pair selection.** Screening for cointegration on the whole
sample and then "trading" the pairs it found is a severe and very easy
mistake — the pairs are chosen *because* their spread mean-reverted over the
period being tested. Selection here runs on a trailing formation window and the
pairs it produces are only ever applied to dates *after* that window closes.

**Multiple testing.** Screening every pair of a 50-name universe is 1,225
hypothesis tests. At p < 0.05 roughly 61 pairs pass by chance alone, so an
uncorrected screen finds "cointegration" in pure noise and finds a lot of it.
The correction is on by default and the count of tests travels into the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cross_section import CrossSectionPanel, register_cross_sectional_feature

logger = logging.getLogger(__name__)

#: Sessions of history a pair screen runs on. One year: long enough for an
#: Engle-Granger test to have power, short enough that the relationship being
#: tested is the current one rather than an average over a decade of it.
DEFAULT_FORMATION_WINDOW = 252

#: Sessions a selected set of pairs is traded before re-screening. One quarter.
#: Re-screening daily would be both expensive (O(n^2) ADF tests per date) and
#: noisy — pair membership would flicker on test-statistic sampling error and
#: the book would churn against the Indian friction stack for no signal.
DEFAULT_REFRESH_EVERY = 63

#: Uncorrected significance level for one Engle-Granger test.
DEFAULT_P_THRESHOLD = 0.05

#: How the O(n^2) screen's significance level is adjusted.
CORRECTIONS = ("bonferroni", "none")

#: Cap on pairs carried into trading, applied after sorting by p-value. A
#: universe where thousands of pairs pass is one where the screen has found
#: structure in noise, and holding all of them would make the book a
#: diversified bet on the screen being wrong.
DEFAULT_MAX_PAIRS = 50

#: Said in every report the pairs strategy appears in.
PAIRS_NOT_NEUTRAL_NOTE = (
    "This is a long-only pairs signal: it buys the undervalued leg and does "
    "not short the expensive one, because the platform does not short. The "
    "short leg is what makes textbook pairs trading market-neutral, so these "
    "results carry full market exposure and are not comparable with published "
    "market-neutral pairs returns."
)


@dataclass(frozen=True)
class Pair:
    """One cointegrated relationship, as measured on a formation window."""

    left: str
    right: str
    hedge_ratio: float
    p_value: float
    spread_mean: float
    spread_std: float

    def spread(self, closes: pd.DataFrame) -> pd.Series:
        """`P_left - beta * P_right`, on whatever dates `closes` covers."""
        return closes[self.left] - self.hedge_ratio * closes[self.right]

    def zscore(self, closes: pd.DataFrame) -> pd.Series:
        """The spread standardized by the *formation window's* moments.

        Deliberately not re-estimated on the trading window. A z-score computed
        against its own window's mean is centred at zero by construction and
        can never say the spread is stretched — which is the one thing this
        number exists to say.
        """
        if self.spread_std <= 0:
            return pd.Series(np.nan, index=closes.index)
        return (self.spread(closes) - self.spread_mean) / self.spread_std


@dataclass(frozen=True)
class PairSelection:
    """What a screen found, and how hard it looked."""

    pairs: List[Pair]
    n_tested: int
    p_threshold: float
    correction: str
    notes: List[str] = field(default_factory=list)

    @property
    def effective_threshold(self) -> float:
        """The per-test level actually applied after correction."""
        if self.correction == "bonferroni" and self.n_tested > 0:
            return self.p_threshold / self.n_tested
        return self.p_threshold

    @property
    def expected_false_positives(self) -> float:
        """How many pairs this many tests would pass by chance alone.

        Reported rather than merely corrected for, because it is the number
        that says whether a screen finding 40 pairs found anything.
        """
        return self.n_tested * self.effective_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pairs": len(self.pairs),
            "n_pairs_tested": self.n_tested,
            "pair_p_threshold": self.p_threshold,
            "pair_effective_threshold": self.effective_threshold,
            "pair_multiple_testing_correction": self.correction,
            "pair_expected_false_positives": self.expected_false_positives,
        }


def engle_granger(left: pd.Series, right: pd.Series) -> Tuple[float, float]:
    """Engle-Granger two-step test for one pair.

    Step one regresses `left` on `right` to get the hedge ratio; step two tests
    the residual for a unit root. `statsmodels.tsa.stattools.coint` does both
    with MacKinnon's critical values, which is the reason not to hand-roll it —
    the ADF regression is easy and its critical-value table is not.

    Args:
        left, right: Aligned price series.

    Returns:
        `(hedge_ratio, p_value)`. The p-value is NaN when the test could not be
        run at all, which is different from a large p-value and is treated
        differently by the screen.
    """
    from statsmodels.tsa.stattools import coint

    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 30 or aligned.iloc[:, 1].std() == 0:
        return float("nan"), float("nan")

    a = aligned.iloc[:, 0].to_numpy(dtype=float)
    b = aligned.iloc[:, 1].to_numpy(dtype=float)

    try:
        _stat, p_value, _critical = coint(a, b)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate input
        return float("nan"), float("nan")

    # The hedge ratio from the same regression the test is built on, without an
    # intercept: the intercept is absorbed into `spread_mean`, so carrying both
    # would double-count the level.
    denominator = float(b @ b)
    if denominator <= 0:
        return float("nan"), float("nan")
    return float(a @ b) / denominator, float(p_value)


def select_pairs(
    closes: pd.DataFrame,
    *,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    correction: str = "bonferroni",
    max_pairs: int = DEFAULT_MAX_PAIRS,
    candidates: Optional[Sequence[Tuple[str, str]]] = None,
) -> PairSelection:
    """Screen a formation window for cointegrated pairs.

    Args:
        closes: `(date x symbol)` prices for the **formation window only**. The
            caller is responsible for that slicing, and `rolling_pair_scores`
            below is what does it correctly.
        p_threshold: Uncorrected significance level.
        correction: `"bonferroni"` (default) or `"none"`. See the module
            docstring — an uncorrected screen of a 50-name universe runs 1,225
            tests and passes about 61 by chance.
        max_pairs: Keep at most this many, lowest p-value first.
        candidates: Restrict the screen to these pairs. The natural use is a
            sector map: cointegration between two banks has an economic story,
            and between a bank and a cement maker it usually does not.

    Returns:
        A `PairSelection`.

    Raises:
        ValueError: On an unknown correction.
    """
    if correction not in CORRECTIONS:
        raise ValueError(
            f"Unknown correction {correction!r}. Available: {list(CORRECTIONS)}"
        )

    usable = closes.dropna(axis=1, how="any")
    symbols = list(usable.columns)
    pairs_to_test = (
        list(combinations(symbols, 2))
        if candidates is None
        else [(a, b) for a, b in candidates if a in symbols and b in symbols]
    )
    n_tested = len(pairs_to_test)

    if n_tested == 0:
        return PairSelection(
            pairs=[], n_tested=0, p_threshold=p_threshold, correction=correction,
            notes=["No candidate pair had complete history over the formation window."],
        )

    threshold = p_threshold / n_tested if correction == "bonferroni" else p_threshold

    found: List[Pair] = []
    for left, right in pairs_to_test:
        hedge_ratio, p_value = engle_granger(usable[left], usable[right])
        if not np.isfinite(p_value) or p_value > threshold:
            continue
        spread = usable[left] - hedge_ratio * usable[right]
        std = float(spread.std())
        if std <= 0:
            continue
        found.append(Pair(
            left=left, right=right, hedge_ratio=float(hedge_ratio),
            p_value=float(p_value), spread_mean=float(spread.mean()),
            spread_std=std,
        ))

    found.sort(key=lambda pair: pair.p_value)
    kept = found[:max_pairs]

    notes = [PAIRS_NOT_NEUTRAL_NOTE]
    if correction == "none":
        notes.append(
            f"No multiple-testing correction over {n_tested} tests: about "
            f"{n_tested * p_threshold:.0f} pairs pass at p<{p_threshold} by "
            f"chance alone, so this selection is not evidence of cointegration."
        )
    if len(found) > max_pairs:
        notes.append(
            f"{len(found)} pairs passed; kept the {max_pairs} with the lowest "
            f"p-values."
        )
    return PairSelection(
        pairs=kept, n_tested=n_tested, p_threshold=p_threshold,
        correction=correction, notes=notes,
    )


def pair_scores(closes: pd.DataFrame, selection: PairSelection) -> pd.DataFrame:
    """Per-symbol relative cheapness, from every pair a symbol belongs to.

    For a pair with spread `P_left - beta * P_right` and z-score `z`:

    - `z` very **negative** means `left` is cheap against `right`,
    - `z` very **positive** means `right` is cheap against `left`.

    So `left` scores `-z` and `right` scores `+z`, and a symbol in several
    pairs takes the **mean** of its scores. The mean rather than the extreme:
    one stretched pair out of five is as likely to be that pair breaking down
    as it is an opportunity, and taking the max would rank a name entirely on
    its single most extreme relationship.

    Returns:
        `(date x symbol)` cheapness, higher meaning more undervalued. Symbols
        in no surviving pair are absent rather than zero — zero would read as
        "fairly valued", which is a claim the screen did not make.
    """
    if not selection.pairs or closes.empty:
        return pd.DataFrame(index=closes.index, dtype=float)

    totals: Dict[str, pd.Series] = {}
    counts: Dict[str, pd.Series] = {}

    for pair in selection.pairs:
        if pair.left not in closes.columns or pair.right not in closes.columns:
            continue
        z = pair.zscore(closes)
        for symbol, score in ((pair.left, -z), (pair.right, z)):
            defined = score.notna().astype(float)
            filled = score.fillna(0.0)
            totals[symbol] = totals.get(symbol, 0.0) + filled
            counts[symbol] = counts.get(symbol, 0.0) + defined

    if not totals:
        return pd.DataFrame(index=closes.index, dtype=float)

    frame = pd.DataFrame(totals) / pd.DataFrame(counts).replace(0.0, np.nan)
    return frame


def rolling_pair_scores(
    closes: pd.DataFrame,
    *,
    formation: int = DEFAULT_FORMATION_WINDOW,
    refresh_every: int = DEFAULT_REFRESH_EVERY,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    correction: str = "bonferroni",
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> pd.DataFrame:
    """Pair scores over a whole panel, with causal pair selection.

    **This is the function that makes the strategy honest.** Screening for
    cointegration on the full sample and then trading the pairs it found is a
    severe look-ahead bias, and an easy one to commit without noticing: the
    pairs are selected *because* their spread mean-reverted over the very
    period being evaluated.

    Here, the panel is walked forward. At each refresh point the screen sees
    only the `formation` sessions **ending at that point**, and the pairs it
    produces are applied only to the `refresh_every` sessions **after** it. A
    date's score therefore depends on no price after that date, which a test
    asserts directly by perturbing the future and checking the past is
    unchanged.

    Args:
        closes: `(date x symbol)` prices over the whole evaluation.
        formation: Sessions each screen runs on.
        refresh_every: Sessions between screens.
        p_threshold / correction / max_pairs: Passed to `select_pairs`.

    Returns:
        `(date x symbol)` relative cheapness, NaN before the first formation
        window closes and for symbols in no surviving pair.
    """
    if refresh_every < 1:
        raise ValueError(f"refresh_every must be at least 1, got {refresh_every}")
    if closes.empty or len(closes) <= formation:
        return pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)

    blocks: List[pd.DataFrame] = []
    for start in range(formation, len(closes), refresh_every):
        window = closes.iloc[start - formation:start]
        applied = closes.iloc[start:start + refresh_every]
        if applied.empty:
            break

        selection = select_pairs(
            window, p_threshold=p_threshold, correction=correction,
            max_pairs=max_pairs,
        )
        if not selection.pairs:
            blocks.append(pd.DataFrame(index=applied.index, dtype=float))
            continue
        blocks.append(pair_scores(applied, selection))

    if not blocks:
        return pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)

    scores = pd.concat(blocks).reindex(index=closes.index, columns=closes.columns)
    return scores


# --------------------------------------------------------------------------
# Registered cross-sectional features
#
# The formation window lives in the name, matching `idiosyncratic_vol_*` and
# `market_beta_*` — and for the same reason T24 gave: a caller asking for a
# 126-session screen and silently receiving the 252-session answer would be
# ranking on a measurement its config says it is not using.
# --------------------------------------------------------------------------

#: Formation windows the pair screen is registered at.
REGISTERED_FORMATION_WINDOWS = (126, 252)


def _register_formation_family() -> None:
    for window in REGISTERED_FORMATION_WINDOWS:

        def make(w: int):
            def feature(panel: CrossSectionPanel) -> pd.DataFrame:
                return rolling_pair_scores(panel.get("close"), formation=w)

            feature.__name__ = f"pair_cheapness_{w}"
            feature.__doc__ = (
                f"Relative cheapness against a cointegrated partner, screened "
                f"on a rolling {w}-session formation window."
            )
            return feature

        register_cross_sectional_feature(
            f"pair_cheapness_{window}", inputs=("close",)
        )(make(window))


def pair_cheapness_feature(window: int) -> str:
    """Registry name for the pair screen at `window` sessions of formation.

    Raises:
        ValueError: If no feature is registered at that window. Loud, because
            a screen run over a different window is a different screen, and
            rounding to a neighbour would rank on pairs the config did not ask
            for.
    """
    window = int(window)
    if window not in REGISTERED_FORMATION_WINDOWS:
        raise ValueError(
            f"No 'pair_cheapness' feature registered at a {window}-session "
            f"formation window. Available: {sorted(REGISTERED_FORMATION_WINDOWS)}. "
            f"Add the window to `REGISTERED_FORMATION_WINDOWS` rather than "
            f"rounding — the window is part of what the screen measures."
        )
    return f"pair_cheapness_{window}"


_register_formation_family()
