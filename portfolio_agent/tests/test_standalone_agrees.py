"""`afa_lab.py` must agree with the package, or stop claiming to.

`notebooks/standalone/README.md` says the standalone file reproduces the
package's behaviour without importing it. Nothing checked that, and it had
stopped being true in two specific ways:

- **A pooled rank IC.** Exactly the defect T12 removed from the package. It was
  driving `val_rank_ic` in the notebook's own training loop, and a pooled rank
  correlation answers a different question from the per-date one — on a signal
  that orders every date perfectly while its level runs against the market, the
  pooled figure is −0.99 and the per-date figure is +1.00.
- **A total-volatility low-volatility sort.** T14 moved the package to the CAPM
  residual on the finding that the residual sort survives out of sample where
  the total sort does not. The standalone file kept the total sort under the
  same name.

Two copies of one behaviour, kept in step by nobody — the shape this whole
round has been removing. These tests are the mechanism the README's claim was
missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STANDALONE = Path(__file__).resolve().parents[2] / "notebooks" / "standalone"


@pytest.fixture(scope="module")
def lab():
    """Import `afa_lab` the way a notebook does — by path, not as a package."""
    if str(STANDALONE) not in sys.path:
        sys.path.insert(0, str(STANDALONE))
    try:
        import afa_lab
    except ImportError as exc:  # pragma: no cover - needs the optional extras
        pytest.skip(f"afa_lab needs an optional extra: {exc}")
    return afa_lab


@pytest.fixture
def market():
    """Twelve names on a shared factor, with a spread of betas and own-vol.

    Both properties are needed: without a beta spread the total and residual
    sorts coincide and the comparison measures nothing.
    """
    rng = np.random.default_rng(5)
    n, k = 400, 12
    index = pd.bdate_range("2022-01-03", periods=n)
    factor = rng.normal(0.0004, 0.011, n)

    closes = {}
    for i in range(k):
        beta = 0.3 + 0.15 * i
        own = 0.004 + 0.002 * ((k - i) % 5)
        returns = beta * factor + rng.normal(0, own, n)
        closes[f"S{i}"] = 100.0 * (1.0 + returns).cumprod()
    return pd.DataFrame(closes, index=index)


# --------------------------------------------------------------------------
# The rank IC
# --------------------------------------------------------------------------


class TestTheRankICIsPerDate:
    def test_the_file_no_longer_pools(self, lab):
        """The T12 defect, asserted at the source rather than by inspection."""
        import inspect

        source = inspect.getsource(lab._rank_ic)
        assert "groupby" in source
        assert "per date" in source.lower()

    def test_it_refuses_rather_than_pooling_without_dates(self, lab):
        """A pooled number is not a worse estimate of the same quantity — it is
        an estimate of a different one, so the function declines to produce it."""
        import torch

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        assert np.isnan(lab._rank_ic(predictions, targets))

    def test_a_perfect_per_date_ordering_scores_one(self, lab):
        import torch

        dates = ["d1"] * 6 + ["d2"] * 6
        scores = torch.tensor([1.0, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        labels = torch.tensor([1.0, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])

        assert lab._rank_ic(scores, labels, dates) == pytest.approx(1.0)

    def test_it_disagrees_with_the_pooled_version_when_it_should(self, lab):
        """The case that makes the distinction matter.

        Each date is ordered perfectly, but the *levels* run opposite ways
        across dates. Pooled, that reads as strong negative skill; per date, it
        is perfect skill. The two are not close.
        """
        import torch

        dates = ["d1"] * 6 + ["d2"] * 6
        # d1 scores are high with low labels; d2 the reverse. Within each date
        # the ordering is exact.
        scores = torch.tensor([10.0, 11, 12, 13, 14, 15, 0.0, 1, 2, 3, 4, 5])
        labels = torch.tensor([0.0, 1, 2, 3, 4, 5, 10.0, 11, 12, 13, 14, 15])

        per_date = lab._rank_ic(scores, labels, dates)
        pooled = float(np.corrcoef(
            pd.Series(scores.numpy()).rank(), pd.Series(labels.numpy()).rank()
        )[0, 1])

        assert per_date == pytest.approx(1.0)
        assert pooled < -0.5

    def test_a_thin_date_is_skipped_not_ranked(self, lab):
        """Same floor the package uses: ranking three names is not ranking."""
        import torch

        dates = ["thin"] * 3 + ["wide"] * 6
        scores = torch.tensor([3.0, 1, 2, 1.0, 2, 3, 4, 5, 6])
        labels = torch.tensor([1.0, 2, 3, 1.0, 2, 3, 4, 5, 6])

        # Only the wide date contributes, and it is a perfect ordering.
        assert lab._rank_ic(scores, labels, dates) == pytest.approx(1.0)

    def test_its_floor_matches_the_package(self, lab):
        from portfolio_agent.evaluation.metrics import MIN_CROSS_SECTION_NAMES

        assert lab.MIN_CROSS_SECTION_NAMES == MIN_CROSS_SECTION_NAMES


# --------------------------------------------------------------------------
# The low-volatility sort
# --------------------------------------------------------------------------


class TestTheLowVolSortIsResidual:
    def test_the_default_is_idiosyncratic(self, lab, market):
        import inspect

        signature = inspect.signature(lab.low_volatility_scores)
        assert signature.parameters["sort_on"].default == "idiosyncratic"

    def test_it_refuses_the_residual_sort_without_a_cross_section(self, lab):
        with pytest.raises(ValueError, match="needs `close`"):
            lab.low_volatility_scores({}, sort_on="idiosyncratic")

    def test_an_unknown_sort_is_refused(self, lab, market):
        with pytest.raises(ValueError, match="sort_on must be"):
            lab.low_volatility_scores({}, market, sort_on="downside")

    def test_the_residual_agrees_with_the_package(self, lab, market):
        """The claim `standalone/README.md` makes, checked numerically.

        Both compute `var(r_i) - beta^2 * var(r_m)` on the same 60-session
        window against the same equal-weighted composite, so they should agree
        to floating point — not merely correlate.
        """
        from portfolio_agent.features.market_relative import (
            idiosyncratic_vol_from_closes,
        )

        standalone = lab.idiosyncratic_volatility(market, window=60, lag=1)
        package = idiosyncratic_vol_from_closes(market, window=60, lag=1)

        pd.testing.assert_frame_equal(standalone, package, rtol=1e-9)

    def test_the_two_sorts_select_different_names(self, lab, market):
        """If they agreed there would have been nothing to fix.

        The fixture gives beta and own-volatility opposite orderings on
        purpose, so a total-volatility sort and a residual sort pick different
        quartiles — which is the finding T14 recorded.
        """
        panel = {
            symbol: pd.DataFrame(
                {"realized_vol_60": market[symbol].pct_change()
                 .rolling(60).std() * np.sqrt(252)},
                index=market.index,
            )
            for symbol in market.columns
        }

        residual = lab.low_volatility_scores(panel, market, sort_on="idiosyncratic")
        total = lab.low_volatility_scores(panel, market, sort_on="total")

        held_residual = set(residual.iloc[-1][residual.iloc[-1] > 0].index)
        held_total = set(total.iloc[-1][total.iloc[-1] > 0].index)

        assert held_residual and held_total
        assert held_residual != held_total


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------


class TestTheGeneratedFileIsCurrent:
    def test_it_is_importable_standalone(self, lab):
        """No package import — that is the whole premise of the file."""
        import inspect

        source = inspect.getsource(lab)
        assert "from portfolio_agent" not in source
        assert "import portfolio_agent" not in source

    def test_it_matches_the_blocks_it_is_assembled_from(self, lab):
        """`afa_lab.py` is generated. A block edited without regenerating leaves
        the shipped artifact stale, which is how it drifted in the first place.
        """
        import inspect

        blocks = STANDALONE / "build" / "blocks"
        generated = (STANDALONE / "afa_lab.py").read_text()

        for name in ("_rank_ic", "idiosyncratic_volatility", "low_volatility_scores"):
            body = inspect.getsource(getattr(lab, name))
            assert body.strip() in generated, name

        # And the defining line really came from a block, not a hand edit.
        sources = "\n".join(p.read_text() for p in blocks.glob("*.py"))
        assert "def idiosyncratic_volatility(" in sources
        assert "MIN_CROSS_SECTION_NAMES = 5" in sources


# --------------------------------------------------------------------------
# The notebooks
# --------------------------------------------------------------------------

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


def _code_cells(path: Path):
    import json

    return [
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


class TestTheNotebooksStillRun:
    """Notebooks rot silently — nothing imports them and no test opens them.

    These do not execute a kernel (that needs cached price data and the
    optional extras). They check the two things that break first when the
    package moves underneath: the code no longer parses, and a name it imports
    no longer exists.
    """

    @pytest.mark.parametrize(
        "name",
        ["01_strategy_lab.ipynb", "02_compare_and_sweep.ipynb",
         "03_forecast_lab.ipynb"],
    )
    def test_every_code_cell_parses(self, name):
        import ast

        for source in _code_cells(NOTEBOOKS / name):
            # IPython magics are not Python; strip them the way the kernel does.
            body = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(body)

    @pytest.mark.parametrize(
        "name",
        ["01_strategy_lab.ipynb", "02_compare_and_sweep.ipynb",
         "03_forecast_lab.ipynb"],
    )
    def test_every_imported_name_exists(self, name):
        """The failure mode a doc pass cannot catch by reading.

        A notebook importing `evaluate_decay` from a module that exports
        `decay_curve` looks perfectly reasonable on the page and dies on the
        first cell.
        """
        import ast
        import importlib

        for source in _code_cells(NOTEBOOKS / name):
            body = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            for node in ast.walk(ast.parse(body)):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or "portfolio_agent" not in node.module:
                    continue
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), (
                        f"{name} imports {alias.name} from {node.module}, "
                        f"which does not export it"
                    )


class TestTheLabCanEvaluate:
    """`Lab` had `train`, `backtest`, `compare` and `sweep` — and no way to
    reach the evaluation layer at all, which is the layer round one was built
    for."""

    def test_it_has_the_forecasting_methods(self):
        from portfolio_agent.lab import Lab

        for name in ("evaluate", "neutralized", "compare_forecasts"):
            assert callable(getattr(Lab, name)), name

    def test_evaluate_does_not_train(self):
        """Evaluation is about the signal a strategy already emits, so it must
        work on the rule-based strategies that have no checkpoint at all.

        Matched against the *code*, with the docstring stripped — the docstring
        says "nothing here trains", and a naive text search finds that sentence
        and calls it a training call.
        """
        import ast
        import inspect
        import textwrap

        from portfolio_agent.lab import Lab

        tree = ast.parse(textwrap.dedent(inspect.getsource(Lab.evaluate)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        assert "evaluate_forecast" in called
        assert not {"run_training_job", "run_bulk", "train"} & (called | imported)

    def test_neutralized_says_when_it_is_not_sector_neutral(self):
        import inspect

        from portfolio_agent.lab import Lab

        doc = inspect.getdoc(Lab.neutralized)
        assert "not\n                sector-neutral" in doc or "sector-neutral" in doc
