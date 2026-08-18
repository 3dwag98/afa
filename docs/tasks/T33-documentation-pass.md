# T33 — Documentation pass

**Status:** done · **Effort:** ~1 day · **Depends on:** T19–T32
**Review reference:** round-three plan, Phase 5

## Goal

Make the documentation describe what shipped. Last, deliberately — a doc pass
run before the code settles is one that has to be run again.

## What was wrong

### `run-agent` in twelve places

T11 froze the execution namespace and removed the command. The docs went on
documenting it across README, STRATEGIES and ARCHITECTURE — including a
**Quickstart line** and **two cron examples** that would fail on a fresh
install. Someone following the README from the top hit a dead command on step
three.

All twelve are gone. One mention survives on purpose, in the scheduling
section, saying it *was* removed — a reader who has seen the old docs needs
that sentence.

### The CLI reference was missing six commands

`evaluate`, `compare`, `list-features`, `data status|validate|build`, and
`report` — the entire round-one surface — were absent. Flags were read off the
parsers rather than recalled.

### The evaluation layer had no user-facing documentation at all

Rank IC, decile spread, decay, cost netting, neutralization, run manifests: in
none of README, ARCHITECTURE, STRATEGIES or CUSTOM_TRAINING. The layer round
one exists to provide was reachable only by reading its source.

The new README section leads with *why the headline claim is deliberately not
an equity curve* — a curve reports the product of "does the signal order the
cross-section" and "does that ordering survive becoming a book", and never
their difference. Round one found a strategy that ranks well and has a
**negative** decile spread; one number cannot say that.

### `CUSTOM_TRAINING.md` never mentioned `gbm` or `rank_ic`

Two of four shipped trainers undocumented — including the one that is *the
baseline to beat* and the one that needs no optional extra at all. Its "four
seams" table was also missing T24's cross-sectional feature registry, so it is
now five.

### No CHANGELOG

Added, and it leads with **numbers that changed**, not features that landed:
the backtest reading a session stale, `rule_based` scoring 0.0 under
`evaluate`, the two paths drawing 6 shared names out of 50, the warm-up that
was never loaded. It carries the retractions — including the two
`QUANT_RESEARCH.md` claims that *stopped work from being attempted* — and the
one behaviour preserved on purpose rather than fixed.

### The module map predated five packages

`evaluation/`, `data_quality/`, `validation/`, `training/` and `provenance/`
were all absent, and it claimed 1,111 tests against 2,400+.

## A defect the pass found

Not a stale sentence — a live bug.

**`compare` accepted `--membership`, `--gross`, `--slippage-bps` and
`--index-name`, and silently dropped all four.** They are declared through the
shared parser, and `cmd_compare` then built its own kwargs without them.

argparse accepted every one. So a user asking for a membership-filtered,
cost-charged comparison received an **unfiltered gross one**, with nothing in
the output indicating that four flags had been ignored.

**A flag that is accepted and dropped is worse than one that does not exist.**
A missing flag fails loudly at the parser; this failed silently in the result —
and it failed in the direction that makes a strategy look better.

Both commands now build kwargs through one `shared_evaluation_kwargs`, and a
test asserts the two produce identical dictionaries from identical flags. Same
fix shape as T25's `scaled_quantity` and T12's single rank IC: two call sites
of one contract, kept in step by nobody.

`--fundamentals` was wired at the same time, which is what T31 deferred here.

## What changed

- `README.md` — CLI reference rewritten from the parsers; new evaluation
  section; module map rebuilt; Quickstart, scheduling and env examples fixed;
  the "researched-but-not-implemented" bullet corrected, since T30–T32
  implemented four of the five families it listed.
- `docs/STRATEGIES.md`, `docs/ARCHITECTURE.md` — `run-agent` removed.
- `docs/CUSTOM_TRAINING.md` — the four shipped trainers, why `rank_ic` exists,
  five seams.
- `docs/CHANGELOG.md` (new).
- `docs/tasks/T21-one-feature-set.md` — one broken relative link.
- `portfolio_agent/cli_forecast.py` — `shared_evaluation_kwargs`,
  `--fundamentals`.
- `portfolio_agent/evaluation/harness.py` — an import left unused by T31.

## Verification

Both checks are mechanical and were run:

```python
# Every relative markdown link resolves.
for md in Path().glob("docs/**/*.md"):
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", md.read_text()):
        assert (md.parent / target).resolve().exists()
```

```bash
# Every strategy the docs name is registered.
python -m pytest portfolio_agent/tests/test_forecast_cli.py -q
```

Zero broken links. The only unregistered strategy names remaining in the docs
are documentation placeholders (`a,b,c`, `my_strategy`, `yours`).

## What this does not do

**Notebooks are T34.** `notebooks/standalone/afa_lab.py` still carries the
pooled rank IC that T12 removed from the package, and still sorts low
volatility on total volatility — while `standalone/README.md` claims
equivalence with the package. That is a code defect in a shipped artifact, not
a documentation gap, which is why it has its own task.

**`--free-float`, `--sector-map` and `--flows` are not wired.** Unlike
`--fundamentals`, these reach `add_exposures` rather than `evaluate_forecast`,
so wiring them means threading a store through the neutralization call — a
change to the evaluation API rather than to the parser.
