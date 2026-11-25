---
name: review-pr
description: High-signal correctness review for EasyDeL pull requests or branch diffs. Use when asked to review a PR, review uncommitted changes, or audit a branch for bugs, package-boundary violations, testing-policy violations, or release/commit hygiene before merge.
---

# Skill: Review PR

Provide a high-signal correctness review. Report only grounded bugs, broken contracts, package-boundary violations, or
concrete testing/release-policy violations. Avoid style nits and speculative concerns.

## Required Context

1. Read `WORKSPACE.md`.
2. Identify changed files via `git diff --name-only` or `gh pr diff --name-only`.
3. Read each touched package's `pyproject.toml` (plus `libs/surgecore/Cargo.toml` for `surgecore`) and docs under
   `libs/<package>/docs/` or `libs/<package>/README.md` (`eray` and `surgecore` have `README.md`, not `docs/`).
4. For testing-policy questions, read `.agents/skills/test-workspace/SKILL.md`.
5. For sharding, mesh, or pipeline-parallel diffs, also apply `.agents/skills/distributed-review/SKILL.md`.
6. For operational changes, read `.agents/ops/OPS.md`.

## Early Stop (GitHub PRs)

```bash
gh pr view <PR> --json state,isDraft,title,body,author,comments
```

Stop if the PR is closed, draft, clearly trivial, or already reviewed and no re-review was requested. Always review
agent-authored PRs when the user asks.

## Review Strategy (Scale To Diff Size)

- **Small / localized diffs** (1–5 files in a single subsystem): perform one focused pass checking both compliance and
  runtime correctness directly; do not spawn a heavy multi-agent fan-out.
- **Large / cross-subsystem PRs**: split into parallel passes where supported (context gathering, contract/policy
  compliance, bug hunting, and independent confirmation of proposed findings; add `.agents/skills/distributed-review/SKILL.md`
  when sharding/PP is touched). If parallel workers are unavailable, run the compliance and bug passes sequentially with
  separate notes and confirm every finding against live source before reporting.

## What To Flag

Flag only issues grounded in code, tests, or repo contracts:

- **Layering violations**: any of `spectrax`, `ejkernel`, `eformer`, `eray`, or `surgecore` importing `easydel` or each
  other (`[tool.importlinter]`).
- **Registry / API bypasses**: `easydel` changes bypassing model/trainer/operation/cache/sharding/conversion registries.
- **New `os.environ` knobs**: new env vars added instead of config dataclasses, ContextVars, or explicit setters.
- **Trainer contract drift**: missing either `@Registry.register("trainer-arguments", ...)` or
  `@Registry.register("trainer", ...)`, placing registered trainers in `_LAZY_EXPORTS` instead of eager imports in
  `libs/easydel/easydel/trainers/__init__.py`, or `TrainingArguments` changes missing
  `libs/easydel/easydel/infra/elarge/types/training.py` or `test_training_arguments_save_load_roundtrip.py` coverage.
- **Remote / GCS path bugs**: raw `pathlib.Path` used on GCS/remote URIs instead of `ePath` or
  `libs/spectrax/spectrax/serialization/_fs.py`.
- **Kernel parity gaps**: hardware kernel changes lacking a `Platform.XLA` (`Backend.ANY`) fallback or parity path.
- **eSurge & `surgecore` regressions**: stale cache shapes, accidental MTP enablement, broken DP/KV-page placement
  (including `libs/surgecore/src/engine/dp.rs` and `payload.rs`), Rust `cargo fmt`/`clippy`/`test` failures, stale
  `Cargo.lock`, broken `abi3-py311` wheel builds, or releasing an `easydel` pin before the matching sibling is on PyPI.
- **Testing-policy violations**: computation tests marked `@pytest.mark.cpu_ok` to evade `conftest.py`'s CPU guard,
  computation claims backed only by CPU runs (must be stated as unverified on hardware), or weak tests asserting private
  helpers, incidental log strings, or tautologies.
- **Release / hygiene violations**: manual version or pin edits (must use `scripts/release.sh`), out-of-sync `uv.lock`,
  or self-credit trailers (`Co-Authored-By`, `Generated with`).

Do not flag mere style preferences, speculative perf concerns, or trivial lint issues unless lint review was requested.

## Verification

Run the cheapest relevant checks after identifying likely issues:

```bash
uv run lint-imports
uv lock --check   # when any pyproject.toml changed
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <focused-test-path>
```

When `libs/surgecore` is touched, also verify Rust and CPU tests:

```bash
(cd libs/surgecore && cargo fmt --check)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo clippy --locked --all-targets -- -D warnings)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo test --locked && cargo test --locked --no-default-features)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests
```

Computation tests run on the accelerator only (GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`); the CPU trio is
for non-computation tests (`eray`, `surgecore`, parsers, config/CLI). Without an accelerator, report computation claims
as **unverified on hardware** — never substitute a CPU run.

## Output

Lead with findings ordered by severity. Each finding needs:

- file and line
- why it is a real bug or contract violation
- the relevant rule or code path
- the smallest credible fix direction

If there are no findings, state that clearly and list residual risk (such as hardware checks not run).
