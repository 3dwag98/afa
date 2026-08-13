# T17 — Give `src/` the `execution/` treatment

**Status:** done (the duplicate pairs; the wholesale move is deliberately not
attempted) · **Effort:** ~1 day · **Depends on:** T10, T11 (the method)
**Review reference:** `docs/architecture_review_2.html`, last item of "What I
would do next"

## Goal

Resolve the places where `src/` answers one question twice.

## Why

`src/` is 14,093 lines across 25 modules, 24 of which are imported from
elsewhere in the package. The review's instruction was specific: **the
duplicate pairs first.** A mass mechanical move of 25 modules would be an
unreviewable diff that conflicts with everything and fixes nothing — the
problem is not where the files sit, it is that some of them disagree.

## What the pairs turned out to be

### Two families of risk arithmetic, in the same file

`src/risk.py` carried `calculate_stop_target` / `calculate_quantity` — what the
strategies actually call — alongside `calculate_stop_loss`,
`calculate_target_price`, `calculate_position_size`, `calculate_portfolio_risk`
and `check_risk_limits`. The second family had **no importer anywhere**: not a
strategy, not the orchestrator, not a test.

Dead code alone is not worth a task. This was, because the two disagreed:

| question | live | dead |
| --- | ---: | ---: |
| stop, 5-rupee ATR at 100 | **92.5** | 90.0 |
| stop, no ATR at 100 | **98.0** | 95.0 |
| vol, 20 names @ 20%, rho 0.85 | **18.5%** | 4.5% |

The stop difference is not cosmetic: the stop is the denominator
`calculate_quantity` divides by, so the no-ATR case sized the same trade **2.5x
larger** depending on which function a caller reached for.

`calculate_portfolio_risk` is the serious one, because its error was structural
rather than a tuning difference — **it assumed zero correlation by default**.
On an ordinary Indian long-only book, where everything loads on the same
market, that understates portfolio volatility **4.1x**, in the direction that
makes one bet look like twenty.

The sharpest part: `src/portfolio.py` already exposed
`correlation_risk_multiple`, whose entire purpose is to report that ratio. The
module that measures the error was sitting next to the module making it.

**Resolution:** the second family is deleted, and `src/risk.py`'s docstring now
records what each disagreement was, so the next person who wants a
`calculate_position_size` reads why there isn't one. Book-level risk lives in
`src/portfolio.py`, which measures the covariance instead of assuming it away.

### Flat-layout import fallbacks that T10's test could not see

`try: from .x import y / except ImportError: from x import y` — the shape that
let a module run as a loose script and made the package uninstallable. T10
deleted the ones it found and added a test to keep them gone.

**The test only matched at module scope.** It hard-coded four spaces of
indentation and required the `try` branch to be a *relative* import, so several
survived: two nested inside functions in `monte_carlo.py`, plus
`compliance.py`, `trigger_engine.py`, `hf_dataset.py` and `universe.py`. One in
`strategies/weighting.py` imported the **identical absolute path in both
branches** — a fallback that could never have fallen back.

The test was rewritten to key on the structure that matters — an `ImportError`
handler that re-imports the *same names* — at any indentation. An
optional-dependency guard imports something different or nothing at all, so the
two shapes are now distinguished by what they do rather than by where they sit.

#### Then removing them broke two tests, which was the real finding

`data_store.py`'s fallback was **not** dead. Five test modules did

    sys.path.insert(0, .../portfolio_agent/src)
    from data_store import ...

which loads the same file a second time under a bare top-level name. So
`portfolio_agent.src.data_store` and `data_store` were two distinct modules
with two sets of module state — precisely what removing the `src` symlink in
T10 was meant to end, surviving in the test suite where T10 was not looking.

It was not merely untidy:

- `monkeypatch.setattr("data_store.batch_download_and_cache", ...)` patched the
  copy the code under test was **not** using. The test passed without the patch
  having any effect on anything.
- The flat copy has no parent package, so its relative imports fail — which is
  the entire reason `data_store.py` still carried a `try/except ImportError`.
  Shipped code was carrying a branch whose only remaining purpose was to keep
  five test modules importable.

**Resolution:** the five test modules import through the package, the `sys.path`
hacks are gone, and two new tests assert that neither the hack nor a bare-name
import of any `src/` module returns. With those gone the fallback is genuinely
dead, and stays deleted.

### A name collision between two `DEFAULT_VOL_WINDOW`s

Found while integrating, not present before T14. `strategies/cross_sectional.py`
imports from `features/market_relative.py` and from `src/regime.py`, and both
export a `DEFAULT_VOL_WINDOW`. The second import silently shadowed the first.

Both are 60 today, which is exactly what makes it worth fixing rather than
leaving: they measure different things — one is the CAPM residual-estimation
window, the other the market's own volatility lookback — so the day either
moves, the idiosyncratic sort would start using the regime filter's number with
nothing failing.

Worse, both read the **same param key**. Setting `vol_window` moved the regime
filter's view of market stress *and* the length of the CAPM regression at once.

**Resolution:** the import is aliased to `DEFAULT_IDIOSYNCRATIC_WINDOW`, and the
idiosyncratic estimation window gets its own key, `idiosyncratic_window`. Three
tests pin it, including one asserting the two constants stay distinguishable
while they happen to be equal.

## Acceptance criteria

- [x] Each duplicate pair identified by reading, with its numerical
      disagreement measured rather than asserted.
- [x] The dead risk family deleted; a test searches by *definition* for any
      return, the way T10 does for RSI.
- [x] The surviving answers pinned, so the deletion cannot be undone by
      changing the survivor to match the deleted one.
- [x] The 4.1x correlation understatement demonstrated against the covariance
      implementation that was already there.
- [x] Every remaining flat-import fallback removed, and the test that missed
      them rewritten to catch the shape at any indentation.
- [x] No test module puts `src/` on `sys.path` or imports a `src/` module by
      its bare name, so no file is loaded twice under two identities.
- [x] `src/risk.py` records why the second family went.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/src/risk.py` | Second risk family deleted (−145 lines); docstring records each disagreement |
| `portfolio_agent/src/{monte_carlo,compliance,trigger_engine,hf_dataset,universe,data_store}.py` | Flat-import fallbacks removed |
| `portfolio_agent/strategies/{cross_sectional,india_sac,rule_based,weighting}.py` | Same, plus the `DEFAULT_VOL_WINDOW` alias and `idiosyncratic_window` key |
| `portfolio_agent/tests/{test_data_store,test_execution_sim,test_concurrency_paths,test_volatility_models,test_backtest_reporting}.py` | Import through the package; `sys.path` hacks removed |
| `portfolio_agent/tests/test_one_risk_family.py` | New — 18 tests |
| `portfolio_agent/tests/test_dead_code_is_gone.py` | Fallback test rewritten; 2 tests for the double-load |
| `portfolio_agent/tests/test_idiosyncratic_volatility.py` | 3 tests for the collision |

## Deliberately not done

**The other 20 modules are not moved.** The review said "the duplicate pairs
first", and after resolving them the remaining case for a wholesale
reorganization is weaker than it looked: `src/` holds `data_store`,
`execution_sim`, `performance_stats`, `portfolio`, `regime`, `liquidity` and
the rest, each with one clear owner and no twin. Moving them would rewrite
every import in the package to make the directory listing prettier, and would
conflict with every open branch. If a second pass happens, it should be driven
by a specific coupling problem, not by the line count.

**Two allocators and two regime models were not confirmed.** The review named
three duplicate pairs from a reading of the tree. Risk was real and is fixed.
For allocation and regime, the search found one implementation each with
genuinely different callers — `portfolio.py` sizes a book while
`strategies/weighting.py` blends component scores; `src/regime.py` classifies
the market while `src/markov_regime.py` is an unwired feature generator that
T10 explicitly kept. Neither is a duplicate. Recording that here because "the
review said there were three" is otherwise a loose end someone re-opens.
