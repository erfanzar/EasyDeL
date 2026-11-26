---
name: run-research
description: Base workflow for EasyDeL research, optimization, debugging, or multi-step implementation tasks that require hypotheses, experiments, evidence, and a clean handoff. Use when work spans multiple files, affects performance/numerics/runtime behavior, or needs repeated verify-and-adjust loops.
---

# Skill: Run Research

Base workflow for non-trivial EasyDeL tasks. Load this first, then layer any domain specialization (e.g.
`.agents/skills/add-ejkernel-kernel/SKILL.md`, `.agents/skills/debug-training-oom/SKILL.md`,
`.agents/skills/develop-surgecore/SKILL.md`). Keep shared cadence here and package-specific commands in the
specialization or `.agents/ops/OPS.md`.

## Load Order

1. Read `WORKSPACE.md`.
2. Read each touched package's `pyproject.toml` (plus `libs/surgecore/Cargo.toml` when touching `surgecore`) and its
   documentation under `libs/<package>/docs/` or `libs/<package>/README.md` (`eray` and `surgecore` use `README.md`
   instead of a `docs/` tree).
3. If the task is operational or hardware/failure-prone, read `.agents/ops/OPS.md`.
4. For test selection and execution commands, load `.agents/skills/test-workspace/SKILL.md`.

## Grounding And Policy Rules

- **Verify before citing**: never cite a script, flag, entry point, env var, symbol, or path until you have opened it
  or found it with `rg` (or `git grep` on minimal containers where `rg` is absent).
- **No new `os.environ` knobs**: wire new toggles through config dataclasses, constructor arguments, ContextVars, or
  explicit setters — never invent new env vars (even temporary ones left in diffs).
- **Layering & registries**: only `easydel` may import `spectrax`, `ejkernel`, `eformer`, `eray`, `surgecore`, or `eyvan`
  (`uv run lint-imports`). Never bypass model/trainer/operation/kernel registries.

## Workflow

1. Inventory the surface:
   - `git status --short` and `git diff --stat`
   - `rg` (or `git grep`) for the target symbol, flag, error text, or entry point
2. State the active hypothesis and the exact evidence that would falsify it.
3. Make the smallest code or doc change that tests the hypothesis.
4. Run focused verification via `.agents/skills/test-workspace/SKILL.md`.
5. Keep only changes supported by verification.
6. Record durable multi-pass notes in `.agents/projects/<topic>.md` (goal/stop condition, baseline command + result,
   hypotheses tested, per-attempt command/output summary, negative results, next action). Operational recovery belongs
   in `.agents/ops/OPS.md`.

Do not create GitHub issues, W&B runs, tags, or release artifacts unless requested or required by the task.

## Environment Defaults

Full per-package commands live in `.agents/skills/test-workspace/SKILL.md`. Core rules:

- **JAX computation tests & benchmarks** run on an accelerator (TPU:
  `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu`, GPU:
  `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). TPU hosts hold a single `libtpu` lock: run one accelerator process
  at a time.
- **CPU trio** is ONLY for non-computation tests (`eray`, `surgecore`, `eyvan`, parsers, configs, `@pytest.mark.cpu_ok`).
  A CPU run of a computation test is never validation; with no accelerator available, report the change as **unverified
  on hardware**.
- **`libs/surgecore` changes** require Rust crate checks (`cargo fmt`, `cargo clippy`, `cargo test`) plus CPU pytest
  (see `.agents/skills/test-workspace/SKILL.md` and `.agents/skills/develop-surgecore/SKILL.md`).

## Reporting

Report:

- Files changed.
- Exact verification commands and outcomes.
- Remaining risk, especially skipped hardware or end-to-end checks.
- For performance work: direct baseline vs candidate numbers with hardware, shape/dtype, compile-including timing when
  relevant, and steady-state timing.

Never include self-credit trailers (`Co-Authored-By`, `Generated with`) in commits, PRs, tags, or release notes.
