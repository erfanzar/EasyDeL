# Scheduler mutation testing

Mutation testing asks whether tests detect deliberately changed behavior. A
passing unmodified suite is necessary but not sufficient: another candidate
winning a tie-break can hide a broken admission guard.

## Reproduce in an isolated snapshot

The optional `[tool.mutmut]` profile in `pyproject.toml` targets
`eyvan/domain/*.py` with an explicit test selection and Hypothesis seed. Read
that configuration for the current selection; it is not the full Python suite.

Use **mutmut 3.7.0** with Python 3.11 and the checkout's development requirements.
Create a new, empty scratch directory outside the main checkout:

```bash
snapshot="$(mktemp -d)"
rsync -a --exclude=__pycache__ eyvan tests pyproject.toml README.md requirements-dev.txt "$snapshot/"
cd "$snapshot"
uvx --python 3.11 --from mutmut==3.7.0 --with-requirements requirements-dev.txt mutmut run --max-children 2
uvx --python 3.11 --from mutmut==3.7.0 mutmut results
```

Keep the main checkout out of the runner's working directory. Do not overlay a
new candidate on an old run or run `mutmut apply` in the main checkout. Record
input identities, configuration, selected tests, seed, tool/interpreter,
unmodified control, raw verdict map and exit status. For an independent repeat,
start from a fresh snapshot and compare the full verdict maps, not only totals.
The optional tool environment is not the package's hash-pinned runtime export.

Use `mutmut show MUTANT_NAME` to inspect a generated change. A targeted probe
can set `MUTANT_UNDER_TEST=MUTANT_NAME` while running the selected test from the
snapshot's generated `mutants/` directory in the same tool environment. Always
run its unmutated control too. Collection/import failures are not semantic
detections.

## Historical aggregate baseline

At source `4c2d8d5`, an expanded 27-file selection and an independent repeat
both detected **1,582 of 1,959 generated mutants**, leaving **377 survivors**
across nine implementation modules. The repeats had matching full verdict maps.
This is a historical baseline, not current-source qualification: the domain
has changed and includes `governance.py` beyond that baseline.

## Interpretation and acceptance criteria

- Exit 0 does **not** mean zero survivors. Retain surviving, timeout, error and
  no-test verdicts separately; none is a passing quality gate by itself.
- Inspect observable resource conservation, budget/quota enforcement, fairness,
  preemption, deterministic placement and useful diagnostics. Expand an existing
  relevant test selection before inventing a production change for a survivor.
- Classify equivalence per mutant with an explicit argument. Diagnostic-only
  changes alter visible output and are not automatically equivalent. Do not
  suppress survivors broadly or assert unused private storage to improve a score.
- Pin seeds for generated tests. Unseeded comparisons may reflect stochastic or
  corpus changes, not a regression test's effect. Re-run after source, selected
  tests, configuration or dependencies change; generated IDs are snapshot-local.
- The pinned generator skips some decorated classes/functions, including many
  dataclass methods and properties. Targeting a module does not cover those
  bodies. Qualify their public contracts separately and keep hand-selected
  sensitivity probes outside the generated-mutant denominator.

## Useful contract targets

Existing tests cover affordable whole-cell alternatives, snapshot identity and
currency validation, empty costing totals, decimal spend aggregation, accelerator
name ordering, finite carbon/freshness inputs, exact decimal affordability,
pricing specificity and resource-vector semantics. These are useful starting
points for a new run, not evidence that all mutations are detected.

For a narrowed experiment, override both `only_mutate` and
`pytest_add_cli_args_test_selection` in the scratch configuration. Preserve the
same tool and control procedure. Do not compare a narrowed denominator with a
full-domain score.

## Still open

Run a fresh aggregate on the intended candidate, cover new domain modules,
verify seeded repeatability, classify meaningful and equivalent survivors, and
account for skipped dataclass/property bodies. A blocking lane needs a reviewed
survivor policy. The historical baseline does not close full-domain qualification.
