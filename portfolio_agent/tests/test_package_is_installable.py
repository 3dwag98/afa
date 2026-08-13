"""The package must work when imported from outside the repository.

Both failures these cover were found by building a wheel and importing it from
a clean virtualenv, and neither is visible during development — the repo root
has a `src` symlink, so a flat import resolves as long as the process starts
there, and `config.yaml` is always underfoot.

The checks here are static and cheap so they run in the normal suite. A wheel
build is the belt-and-braces version and belongs in CI, but a static assertion
catches the regression at the moment someone writes it rather than at release.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent

# Modules that live inside `portfolio_agent.src`. Imported as bare names or as
# `src.X`, they resolve only when the working directory happens to be the
# repository root.
INTERNAL_MODULES = {
    "risk", "models", "indicators", "monte_carlo", "portfolio", "storage",
    "reporting", "outcomes", "orchestrator", "data_store", "liquidity", "regime",
    "compliance", "execution_sim", "learning", "logging_utils", "sectors",
    "universe", "calibration", "performance_stats", "hf_dataset",
    "backtest_engine", "markov_regime", "risk_analytics", "trigger_engine",
    "volatility_models", "portfolio_optimizer", "backtest_reporting", "rl",
}

FLAT_IMPORT = re.compile(r"^(\s*)from (?:src\.([\w.]+)|([a-z_]+)) import ")


def runtime_modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in str(path) or "/tests/" in str(path):
            continue
        yield path


def first_attempt_flat_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Flat imports of internal modules that are not inside a fallback branch.

    A flat import in an `except ImportError:` branch is fine — it is the
    script-style path, reached only when the canonical one failed. A flat
    import as the *first* attempt is the bug: there is no canonical path to
    fall back from.
    """
    lines = path.read_text().split("\n")
    offenders = []

    for i, line in enumerate(lines):
        match = FLAT_IMPORT.match(line)
        if not match:
            continue
        module = match.group(2) or match.group(3)
        if module is None or module.split(".")[0] not in INTERNAL_MODULES:
            continue

        preceding = "\n".join(lines[max(0, i - 25):i])
        in_fallback = (
            "except ImportError" in preceding
            and preceding.rindex("except ImportError")
            > (preceding.rindex("try:") if "try:" in preceding else -1)
        )
        if not in_fallback:
            offenders.append((i + 1, line.strip()))
    return offenders


@pytest.mark.parametrize("path", list(runtime_modules()), ids=lambda p: p.name)
def test_no_module_imports_internals_flatly_as_its_first_attempt(path):
    """`A1`: this is what makes an installed copy fail to load any strategy.

    `strategies/rule_based.py` tried `from portfolio_agent.src.risk`, then `from risk`, and
    never `from portfolio_agent.src.risk` — which works. Since
    `strategies/__init__.py` imports the registry eagerly, that one chain took
    down the registry, the backtester and the engine together.
    """
    offenders = first_attempt_flat_imports(path)
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} imports an internal module flatly before "
        f"trying the canonical path, which resolves only when the process starts "
        f"at the repository root:\n"
        + "\n".join(f"  line {line}: {source}" for line, source in offenders)
        + "\n\nUse `from portfolio_agent.src.X import ...` as the first attempt; "
          "keep the flat form in an `except ImportError:` branch if script-style "
          "execution still needs it."
    )


def test_the_strategy_registry_imports_without_a_repo_root():
    """The end state `A1` is about: every strategy loads."""
    from portfolio_agent.strategies.registry import get_available_strategies

    strategies = get_available_strategies()
    assert "rule_based" in strategies
    assert "momentum" in strategies
    assert "low_volatility" in strategies


# --------------------------------------------------------------------------
# A2 — configuration
# --------------------------------------------------------------------------


def test_a_default_config_ships_inside_the_package():
    """`A2`: without this, an installed copy silently uses schema defaults.

    It still produces results, charts and a report — all of which look normal
    while describing settings nobody chose. The repository asks for a 4000-name
    universe; the schema default is 10.
    """
    from portfolio_agent.config.loader import PACKAGED_DEFAULT

    assert PACKAGED_DEFAULT.exists(), (
        "portfolio_agent/config/default_config.yaml is missing. It is the only "
        "configuration an installed copy has."
    )


def test_the_packaged_default_is_valid_and_not_the_schema_defaults():
    import yaml

    from portfolio_agent.config.loader import PACKAGED_DEFAULT
    from portfolio_agent.config.schema import AppConfig

    payload = yaml.safe_load(PACKAGED_DEFAULT.read_text())
    config = AppConfig(**payload)

    # If the packaged file were empty or stripped, this would silently equal the
    # schema default and the whole point would be lost.
    assert config.data.universe_size != AppConfig().data.universe_size


def test_resolution_order_prefers_an_explicit_path(tmp_path):
    import yaml

    from portfolio_agent.config.loader import load_config, resolve_config_path

    explicit = tmp_path / "experiment.yaml"
    explicit.write_text(yaml.safe_dump({"data": {"universe_size": 77}}))

    assert resolve_config_path(str(explicit)) == explicit
    assert load_config(str(explicit)).data.universe_size == 77


def test_a_missing_explicit_path_does_not_silently_fall_back(tmp_path):
    """Asking for a file that is not there is a mistake, not a default.

    Falling back to the packaged config here would run the experiment someone
    did not ask for, under a name they chose.
    """
    from portfolio_agent.config.loader import resolve_config_path

    assert resolve_config_path(str(tmp_path / "absent.yaml")) is None


def test_resolution_falls_back_to_the_packaged_default(monkeypatch, tmp_path):
    """From a directory with no config.yaml — i.e. an installed copy."""
    from portfolio_agent.config.loader import PACKAGED_DEFAULT, resolve_config_path

    monkeypatch.chdir(tmp_path)
    # The project-root candidate still resolves inside a checkout, so this
    # asserts the final fallback exists rather than that it is reached here.
    assert PACKAGED_DEFAULT.exists()
    assert resolve_config_path("definitely-not-a-config.yaml") == PACKAGED_DEFAULT


def test_cli_exposes_the_global_options():
    from portfolio_agent.cli import create_parser

    parser = create_parser()
    options = {action.dest for action in parser._actions}

    assert {"config", "json", "verbose", "quiet"} <= options
