"""One answer per risk question, asserted so a second cannot come back.

`src/risk.py` carried two families. `calculate_stop_target` /
`calculate_quantity` are what the strategies call; `calculate_stop_loss`,
`calculate_target_price`, `calculate_position_size`, `calculate_portfolio_risk`
and `check_risk_limits` were an older set with no importer anywhere — not a
strategy, not the orchestrator, not a test.

Dead code is not, on its own, worth a task. This was, because the two families
disagreed:

| question | live | dead |
| --- | --- | --- |
| stop, 5-rupee ATR at 100 | 92.5 | 90.0 |
| stop, no ATR at 100 | 98.0 | 95.0 |
| vol, 20 names @ 20% / rho 0.85 | 18.5% | 4.5% |

The last one is the dangerous one, and its error is structural rather than a
tuning difference: it assumed zero correlation by default, so an Indian
long-only book — where everything loads on the same market — looked diversified
when it was one bet.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# The second family is gone
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "calculate_stop_loss",
        "calculate_target_price",
        "calculate_position_size",
        "calculate_portfolio_risk",
        "check_risk_limits",
    ],
)
def test_the_duplicate_risk_function_is_gone(name):
    import portfolio_agent.src.risk as risk

    assert not hasattr(risk, name)


@pytest.mark.parametrize(
    "name",
    [
        "calculate_stop_loss",
        "calculate_target_price",
        "calculate_position_size",
        "calculate_portfolio_risk",
        "check_risk_limits",
    ],
)
def test_nothing_defines_it_anywhere_else(name):
    """Searched by definition rather than by import.

    The failure mode is someone re-adding a local copy next to a caller, not
    re-adding the module — the same reason T10 searches for `def rsi` rather
    than for `import indicators`.
    """
    pattern = re.compile(rf"^\s*def\s+{name}\s*\(", re.M)
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "portfolio_agent").rglob("*.py")
        if "tests" not in path.parts and pattern.search(path.read_text())
    ]
    assert offenders == [], offenders


def test_the_survivors_are_still_there():
    """The deletion must not have taken the live family with it."""
    from portfolio_agent.src.risk import calculate_quantity, calculate_stop_target

    assert callable(calculate_stop_target)
    assert callable(calculate_quantity)


# --------------------------------------------------------------------------
# The disagreement, pinned to the surviving answer
# --------------------------------------------------------------------------


class TestTheSurvivingAnswers:
    def test_an_atr_stop_is_the_wider_one(self):
        """2.5x the ATR below entry, not a flat 10%.

        Pinned because the deleted twin returned 90.0 here, and a stop is the
        denominator position sizing divides by — so the two produced books of
        different sizes from the same signal.
        """
        from portfolio_agent.src.risk import calculate_stop_target

        stop, target = calculate_stop_target(100.0, atr=5.0)
        assert stop == pytest.approx(92.5)
        assert target > 100.0

    def test_without_an_atr_the_stop_is_tighter_not_looser(self):
        """98.0, where the deleted twin said 95.0.

        A 2.5x difference in risk-per-share, and therefore in quantity.
        """
        from portfolio_agent.src.risk import calculate_stop_target

        stop, _ = calculate_stop_target(100.0, atr=None)
        assert stop == pytest.approx(98.0)

    def test_the_stop_difference_would_have_been_a_2_5x_sizing_difference(self):
        """Stated as the thing that actually goes wrong, not as two constants."""
        from portfolio_agent.src.risk import calculate_stop_target

        live_stop, _ = calculate_stop_target(100.0, atr=None)
        deleted_stop = 95.0
        assert (100.0 - deleted_stop) / (100.0 - live_stop) == pytest.approx(2.5)


class TestBookRiskMeasuresCovariance:
    def test_correlated_names_do_not_diversify_away(self):
        """The structural error the deleted function made.

        Twenty names at 20% volatility with pairwise correlation 0.85 have a
        portfolio volatility of 18.5%, not the 4.5% you get by assuming
        independence. The closed form for an equicorrelated equal-weight book
        is sigma * sqrt((1 + (n-1) rho) / n).
        """
        from portfolio_agent.src.portfolio import portfolio_volatility

        n, sigma, rho = 20, 0.20, 0.85
        weights = np.full(n, 1.0 / n)
        covariance = np.full((n, n), rho * sigma * sigma)
        np.fill_diagonal(covariance, sigma * sigma)

        expected = sigma * math.sqrt((1 + (n - 1) * rho) / n)
        assert portfolio_volatility(weights, covariance) == pytest.approx(
            expected, rel=1e-6
        )
        assert expected == pytest.approx(0.185, abs=0.001)

    def test_the_independence_assumption_understates_it_fourfold(self):
        """Kept measurable rather than described.

        `independent_portfolio_volatility` still exists and is correct *for
        independent positions*; the deleted function used that formula as its
        default for any book at all. `correlation_risk_multiple` is the ratio
        between the two, and it already existed — which is the sharpest version
        of the finding: the module that measures the error was sitting next to
        the module that was making it.
        """
        from portfolio_agent.src.portfolio import (
            correlation_risk_multiple,
            independent_portfolio_volatility,
            portfolio_volatility,
        )

        n, sigma, rho = 20, 0.20, 0.85
        weights = np.full(n, 1.0 / n)
        covariance = np.full((n, n), rho * sigma * sigma)
        np.fill_diagonal(covariance, sigma * sigma)

        correlated = portfolio_volatility(weights, covariance)
        independent = independent_portfolio_volatility(weights, covariance)

        assert independent == pytest.approx(0.0447, abs=0.001)
        assert correlated == pytest.approx(0.1852, abs=0.001)
        assert correlation_risk_multiple(weights, covariance) == pytest.approx(
            correlated / independent
        )
        assert correlated / independent == pytest.approx(4.1, abs=0.1)


# --------------------------------------------------------------------------
# The module says what it is now
# --------------------------------------------------------------------------


def test_the_module_records_why_the_second_family_went():
    """A deletion whose reason is only in a commit message is a deletion that
    gets undone by the next person who wants a `calculate_position_size`."""
    import portfolio_agent.src.risk as risk

    doc = risk.__doc__ or ""
    assert "calculate_stop_loss" in doc
    assert "92.5" in doc and "90.0" in doc
    assert "zero correlation" in doc
    assert "portfolio.py" in doc


def test_book_level_risk_is_not_reachable_from_the_per_trade_module():
    """The boundary the split established, asserted rather than assumed."""
    import portfolio_agent.src.risk as risk

    for name in dir(risk):
        assert "portfolio_risk" not in name
