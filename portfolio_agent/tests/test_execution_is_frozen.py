"""The research path must not depend on the frozen execution namespace.

Freezing code only helps if the boundary holds. Without a test, the first
person who needs one function from `orchestrator.py` imports it, and within a
few commits the "frozen" package is load-bearing again — at which point it has
all the costs of being maintained and none of the maintenance.

Scoped deliberately. `main.py` is the live entry point and *should* import the
frozen code; that is what it is for. What must stay clean is everything a
forecast run touches.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent

# Every subpackage a forecasting or backtesting run goes through. `src` is
# included with an exception list, since the frozen modules were moved out of
# it and nothing left behind should reach back in.
RESEARCH_PATHS = ("features", "strategies", "training", "models", "agents", "data", "utils", "src")

# The live entry point, which drives the frozen code on purpose.
ALLOWED = {"main.py"}


def _imports(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover - a parse failure is another test's problem
        return set()

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def research_modules():
    for subpackage in RESEARCH_PATHS:
        root = PACKAGE / subpackage
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            yield path


@pytest.mark.parametrize("path", list(research_modules()), ids=lambda p: str(p.name))
def test_research_module_does_not_import_execution(path):
    if path.name in ALLOWED:
        pytest.skip("the live entry point drives the frozen code by design")

    offenders = {
        module for module in _imports(path)
        if module == "portfolio_agent.execution"
        or module.startswith("portfolio_agent.execution.")
    }
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} imports the frozen execution namespace "
        f"({sorted(offenders)}). If the research path genuinely needs this, the "
        f"code should move out of execution/ rather than be reached into — see "
        f"portfolio_agent/execution/README.md."
    )


def test_the_frozen_package_is_importable():
    """Frozen is not broken. Its tests must still be runnable on request."""
    import portfolio_agent.execution  # noqa: F401


def test_the_frozen_package_says_why():
    readme = PACKAGE / "execution" / "README.md"
    assert readme.exists(), "a frozen namespace without a README is just dead code"

    text = readme.read_text()
    # The README's job is to explain the decision, not just list the contents.
    assert "not maintained" in text.lower()
    assert "reviving" in text.lower()


def test_run_agent_is_no_longer_a_command():
    """The live command is gone from the maintained surface."""
    from portfolio_agent.cli import create_parser

    parser = create_parser()
    subparsers = [
        action for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    ]
    assert subparsers, "expected a subcommand parser"
    assert "run-agent" not in subparsers[0].choices
    # The research commands are still there.
    assert "train" in subparsers[0].choices
    assert "backtest" in subparsers[0].choices
