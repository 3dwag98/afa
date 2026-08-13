"""The deletions, asserted so they stay deleted.

Every item here came back at least once in the history of some codebase because
nothing was watching. A test is cheaper than the argument.

The one that matters most is the duplicate-implementation check. `src/
indicators.py` held a second RSI, ATR, SMA, MACD and Bollinger alongside
`features/technical.py` — and the two families were not merely different code
for the same thing. Everything in `features/technical.py` shifts its inputs by
one bar so a feature cannot read the session it is used to decide; the
`indicators.py` copies did not. Which one produced a published number depended
on which module the caller happened to import.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def tracked_files() -> list:
    result = subprocess.run(
        ["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:  # pragma: no cover - not a git checkout
        pytest.skip("not a git repository")
    return result.stdout.splitlines()


# --------------------------------------------------------------------------
# Nothing generated is tracked
# --------------------------------------------------------------------------


def test_no_bytecode_is_tracked():
    """One of the six was bytecode for `setup_afa.py`, a file that no longer exists.

    Tracked bytecode also produces a phantom diff on every machine that imports
    the package, which trains people to ignore `git status`.
    """
    offenders = [f for f in tracked_files() if f.endswith(".pyc") or "__pycache__" in f]
    assert offenders == []


def test_no_market_data_is_tracked():
    """2,397 parquet files, 112 MB, that could not be refreshed without bloating
    history — so nobody refreshed them, and stale bars look exactly like fresh ones."""
    offenders = [f for f in tracked_files() if f.endswith(".parquet")]
    assert offenders == []


def test_no_model_checkpoint_is_tracked():
    """A checkpoint in the repository implies a canonical model.

    `models/lstm_best.pt` had no record of the universe, config or revision
    that produced it. Run manifests exist so a checkpoint can carry one; a
    binary committed before they existed cannot.
    """
    offenders = [f for f in tracked_files() if f.endswith((".pt", ".pth", ".joblib"))]
    assert offenders == []


def test_obtaining_data_is_documented():
    """Untracking data is only defensible if getting it is one command."""
    doc = REPO / "docs" / "OBTAINING_DATA.md"
    assert doc.exists()
    text = doc.read_text()
    assert "portfolio-agent download-data" in text
    assert "data status" in text
    # The unreclaimed history is noted rather than quietly implied to be fixed.
    assert "history is not rewritten" in text.lower()


# --------------------------------------------------------------------------
# The duplicate indicator module
# --------------------------------------------------------------------------


def test_the_duplicate_indicator_module_is_gone():
    with pytest.raises(ImportError):
        import portfolio_agent.src.indicators  # noqa: F401


def test_there_is_exactly_one_rsi_implementation():
    """Two implementations of one indicator is a correctness hazard.

    Searched by definition rather than by import, because the failure mode is
    someone re-adding a local copy rather than re-adding the module.
    """
    import re

    # Word-bounded, or `diversification_ratio` matches on the "rsi" inside it.
    name_pattern = re.compile(r"^def\s+(\w*\brsi\b\w*|rsi\w*|\w*_rsi\w*)\s*\(", re.I)
    definitions = []
    for path in (REPO / "portfolio_agent").rglob("*.py"):
        if "tests" in path.parts:
            continue
        for line in path.read_text().splitlines():
            match = name_pattern.match(line.strip())
            if match:
                definitions.append(f"{path.relative_to(REPO)}: {match.group(1)}")
    assert len(definitions) == 1, definitions


def test_adx_is_available_from_the_feature_registry():
    from portfolio_agent.features.registry import get_feature, list_features

    assert "adx_14" in list_features()
    assert get_feature("adx_14") is not None


def test_the_registered_adx_is_lag_safe_like_its_neighbours():
    """Every feature in that module shifts; a mixed convention is the hazard.

    The value at row t must not depend on row t's own bar, or a model trained
    on it is reading the session it is being asked to predict.
    """
    from portfolio_agent.features.technical import adx_14

    rng = np.random.default_rng(0)
    index = pd.date_range("2023-01-02", periods=200, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 200))
    frame = pd.DataFrame(
        {"close": close, "high": close * 1.01, "low": close * 0.99}, index=index
    )

    full = adx_14(frame)
    # Rewriting the final bar must not change any earlier value.
    altered = frame.copy()
    altered.iloc[-1] = altered.iloc[-1] * 1.5
    assert adx_14(altered).iloc[:-1].equals(full.iloc[:-1])


def test_the_unshifted_adx_is_still_available_for_the_regime_filter():
    """`regime.py` passes an already-truncated frame; shifting would lag twice."""
    from portfolio_agent.features.technical import calculate_adx

    rng = np.random.default_rng(1)
    index = pd.date_range("2023-01-02", periods=100, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 100))
    frame = pd.DataFrame(
        {"close": close, "high": close * 1.01, "low": close * 0.99}, index=index
    )
    series = calculate_adx(frame, period=14).dropna()
    assert not series.empty
    assert (series >= 0).all() and (series <= 100).all()


def test_the_snapshot_helpers_travelled_with_their_only_caller():
    from portfolio_agent.execution.orchestrator import (  # noqa: F401
        calculate_all_indicators,
        calculate_indicators,
    )


# --------------------------------------------------------------------------
# Synthetic data is opt-in
# --------------------------------------------------------------------------


def test_synthetic_fallback_is_off_by_default():
    """A platform that silently substitutes generated data will eventually
    publish a number describing a random-walk generator."""
    from portfolio_agent.config.schema import DataConfig

    assert DataConfig().allow_synthetic_fallback is False


def test_the_shipped_config_does_not_enable_it():
    import yaml

    for name in ("config.yaml", "portfolio_agent/config/default_config.yaml"):
        document = yaml.safe_load((REPO / name).read_text())
        assert document["data"]["allow_synthetic_fallback"] is False, name


# --------------------------------------------------------------------------
# Unread settings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,field",
    [
        ("DataConfig", "market_data_dir"),
        ("FeaturesConfig", "lookbacks"),
        ("FeaturesConfig", "feature_sets"),
        ("PathsConfig", "log_dir"),
        ("PathsConfig", "output_dir"),
    ],
)
def test_the_unread_settings_are_gone(model, field):
    """Someone will set one of these and expect an effect.

    Only 5 of 108 settings were unread, so deleting beat documenting: a config
    where every key does something is a config people can trust.
    """
    import portfolio_agent.config.schema as schema

    assert field not in getattr(schema, model).model_fields


def test_the_shipped_config_does_not_mention_them():
    for name in ("config.yaml", "portfolio_agent/config/default_config.yaml"):
        text = (REPO / name).read_text()
        for field in ("market_data_dir", "lookbacks", "feature_sets",
                      "log_dir", "output_dir"):
            assert f"{field}:" not in text, f"{field} still in {name}"


def test_the_config_still_loads_and_every_key_is_known():
    """`extra` is not forbidden on these models, so a stale key would be silent."""
    import yaml

    from portfolio_agent.config.loader import load_config
    from portfolio_agent.config.schema import AppConfig

    config = load_config(str(REPO / "config.yaml"))
    assert config is not None

    document = yaml.safe_load((REPO / "config.yaml").read_text())
    for section, values in document.items():
        if section not in AppConfig.model_fields or not isinstance(values, dict):
            continue
        model = AppConfig.model_fields[section].annotation
        known = set(getattr(model, "model_fields", {}))
        if not known:
            continue
        unknown = set(values) - known
        assert not unknown, f"config.yaml [{section}] has unread key(s): {sorted(unknown)}"


# --------------------------------------------------------------------------
# The src symlink
# --------------------------------------------------------------------------


def test_the_src_symlink_is_gone():
    """It preserved exactly the ambiguity that made the package uninstallable.

    With it, `import src.data_store` and `import portfolio_agent.src.data_store`
    load the same file as two distinct modules with two sets of module-level
    state — and only one of them exists after a pip install.
    """
    assert not (REPO / "src").exists()
    assert "src" not in tracked_files()


def test_no_module_imports_the_flat_path():
    """Both the import itself and the string form used by mock.patch."""
    import re

    offenders = []
    for path in (REPO / "portfolio_agent").rglob("*.py"):
        text = path.read_text()
        for pattern in (r"^\s*from src\.", r"^\s*import src\.", r"['\"]src\.[\w.]+['\"]"):
            if re.search(pattern, text, re.M):
                offenders.append(str(path.relative_to(REPO)))
                break
    assert offenders == []


def test_no_module_keeps_a_flat_import_fallback():
    """`try: from .x import y / except ImportError: from x import y` is the shape.

    It exists to let a module be run as a loose script from inside its own
    directory, and the cost of that convenience was a package that could not be
    installed.
    """
    import re

    pattern = re.compile(
        r"try:\n(?:    from \.[\w.]+ import [^\n]*\n)+except ImportError:[^\n]*\n"
        r"(?:    from [\w][\w.]* import [^\n]*\n)+"
    )
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "portfolio_agent").rglob("*.py")
        if "tests" not in path.parts and pattern.search(path.read_text())
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# Explicitly not deleted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "portfolio_agent.src.rl",
        "portfolio_agent.src.markov_regime",
        "portfolio_agent.src.portfolio_optimizer",
        "portfolio_agent.src.volatility_models",
    ],
)
def test_the_unwired_modules_survive(module):
    """Unwired today, which makes them tempting.

    Under the forecasting premise they become feature generators and
    conditioning variables — a regime state is a conditioning variable, a GARCH
    volatility forecast is a feature. Deleting them means rebuilding them in
    three weeks, so they stay.
    """
    import importlib

    try:
        importlib.import_module(module)
    except ImportError as exc:
        # An optional extra being absent is fine; the module being gone is not.
        assert "cvxpy" in str(exc) or "arch" in str(exc) or "torch" in str(exc), exc
