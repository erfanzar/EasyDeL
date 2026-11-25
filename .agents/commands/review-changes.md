---
name: review-changes
description: High-signal correctness review of a working-tree diff, branch, or PR in the EasyDeL workspace
---

# Review Changes

Perform a high-signal correctness review of the requested PR, branch, or uncommitted working-tree diff (when no target
is specified, review the uncommitted diff against `HEAD`; for a branch/PR, diff against the merge-base with its PR
target branch — **`main` or `vnext`**, which have unrelated histories per `.github/workflows/workspace-ci.yaml`, so
never blindly diff a `vnext` branch against `main`).

## Governing Skills and Checklists

- Core review workflow: `.agents/skills/review-pr/SKILL.md`
- EasyDeL repo hazards: `.agents/agents/reviewer.md`
- Layer specialized checks when relevant:
  - `Mesh` / `PartitionSpec` / `shard_map` / pipeline stages → `.agents/skills/distributed-review/SKILL.md`
  - Performance-sensitive runtime, kernel, or serving paths → `.agents/skills/performance-audit/SKILL.md`
  - Fused or quantized weight layouts → `.agents/skills/quantization-layout/SKILL.md`
  - `libs/surgecore` Rust/Python host boundary → `.agents/skills/develop-surgecore/SKILL.md`

## Ground Rules

- Report real bugs, broken contracts, layering/package-boundary violations, or testing-policy violations only — cite
  `file:line` and the exact rule or failure path.
- Validate every candidate finding against the live source code before reporting it.
- Run the cheapest relevant verification for the touched surfaces:
  - `uv run lint-imports` and `uv run pre-commit run --files <touched files>`
  - Focused pytest on an accelerator (TPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu uv run pytest ...`; GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Use CPU only for pure-Python/Rust non-computation tests (`libs/eray/tests`, `libs/surgecore/tests`, parsers, configs, `@pytest.mark.cpu_ok`); without an accelerator, report computation checks as **unverified on hardware**.
  - If `libs/surgecore` changed: `(cd libs/surgecore && cargo fmt --check && PYO3_PYTHON="$(which python)" cargo clippy --locked --all-targets -- -D warnings && PYO3_PYTHON="$(which python)" cargo test --locked && cargo test --locked --no-default-features)` and `ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests`.
  - If `libs/esurge-dashboard` changed: `(cd libs/esurge-dashboard && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test)`.
- No style nits, speculative concerns, or lint-catchable noise.

## Output

List findings ordered by severity with the smallest credible fix direction for each. If clean, say so explicitly and
name residual risks (e.g., accelerator paths unverified on hardware).
