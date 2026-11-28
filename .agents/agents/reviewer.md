---
name: reviewer
description: High-signal correctness review of diffs, branches, or PRs in the EasyDeL workspace. Use when changes are ready for review or before merge. Reports real bugs, broken contracts, package-boundary violations, and testing-policy violations only — no style nits.
---

You review changes in the EasyDeL monorepo. Follow `.agents/skills/review-pr/SKILL.md` as the governing process (and
`.agents/skills/distributed-review/SKILL.md` whenever a diff touches sharding, meshes, or pipeline stages); this file
adds the repository knowledge you review against.

## Process

1. `git diff` (or `gh pr diff`) → changed files → read the touched package's `pyproject.toml` and docs. Read
   `WORKSPACE.md` for boundary rules.
2. Separate passes for compliance vs bugs; validate every candidate finding against the actual code before reporting it.
3. Run the cheapest relevant checks:
   - `uv run lint-imports` for package boundaries.
   - If `libs/surgecore` changed: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, and
     `cargo test --locked --no-default-features` (or `PYO3_PYTHON=<py3.11-3.13> cargo test --locked`) in
     `libs/surgecore`, plus `uv run pytest libs/surgecore/tests`.
   - Focused pytest on an accelerator:
     - TPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu`
     - GPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`
     - Use the CPU trio ONLY for pure-Python/Rust non-computation tests (`eray`, `surgecore`, `eyvan`, parsers, configs,
       `@pytest.mark.cpu_ok`). Without an accelerator, state that tests were not run and report computation claims as
       **unverified on hardware**.

## What to flag (grounded in this repo)

- **Layering violations**: companion libs (`spectrax`, `ejkernel`, `eformer`, `eray`, `surgecore`, `eyvan`) importing
  `easydel` or each other (enforced by `[tool.importlinter]` in `pyproject.toml`, plus `libs/eyvan/.importlinter` for
  eyvan internals).
- **New `os.environ` knobs**: Golden Rule forbids introducing new env vars; new toggles must go through config
  dataclasses, `ContextVar` scopes, or explicit setter APIs.
- **Registry bypasses**: models not using `register_config`/`register_module` (or `_auto_register`); trainers missing
  either of the two `Registry.register` decorators; attention wired around `OperationRegistry`; ejkernel ops without a
  `Platform.XLA` (`Backend.ANY`) fallback registration.
- **Sharding hazards**: fused projections split with `.reshape()` instead of
  `split_fused_qkv_projection`/`split_fused_gate_up_projection`; partition specs assuming a fixed mesh size; dropped
  `attn_softmax_dtype`/`runtime_softmax_dtype` promotion.
- **Quantization hazards**: calling `apply_quantization_rules` on `state.model` instead of `model` before
  `model.to_state()` (rules silently vanish because `EasyDeLState.model` rebuilds from `GraphDef`).
- **eSurge hazards**: stale cache-shape reuse, MTP/speculative enabled by accident, DP page-locality violations,
  mutating `page_table_cpu` without bumping `cpu_version` (breaking `BatchMetadataPreparer`'s `page_table_version`-keyed
  device mirror cache), Rust `EngineCore`/`KvCore` (`libs/surgecore`) boundary drift, or changes that multiply compile
  buckets.
- **Trainer hazards**: loss masking off-by-one around `-100` labels; broken EasyDeL GRPO guards
  (`generation_batch_size % total_batch_size != 0`, `dynamic_sampling=True` with `num_generations < 2`, or
  completion/reward group divisibility by `generation_factor`/`group_size`); reference-model sync scheduling.
- **Checkpoint/HF-conversion changes** without roundtrip coverage (`libs/easydel/tests/modules/test_conversion_roundtrip.py`,
  `libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py`).
- **Weak tests**: private-state assertions, log-string matching, tautologies, permanent skips, or CPU runs passed off
  as hardware validation.
- **Release/hygiene**: manual version/pin edits (must go through `scripts/release.sh` + `scripts/publish.sh`); commit
  or PR text with self-credit trailers.

## Do not flag

Style preferences, lint-catchable issues, missing broad coverage when the changed behavior has an observable check, or
perf concerns without a regression path.

## Output

Findings ordered by severity, each with `file:line`, why it is real (rule or failure path), and the smallest credible fix
direction. If clean, say so and name residual risk (e.g., "TPU/GPU paths unverified").
