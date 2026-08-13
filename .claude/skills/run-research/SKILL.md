---
name: run-research
description: Base workflow for EasyDeL research, optimization, debugging, or multi-step implementation tasks that require hypotheses, experiments, evidence, and a clean handoff. Use when work spans multiple files, affects performance/numerics/runtime behavior, or needs repeated verify-and-adjust loops.
---

# Skill: Run Research

Use this as the base skill for non-trivial EasyDeL work. Specialized skills add domain rules; this file owns the shared
cadence.

## How To Apply This Skill

1. Load this file first.
2. Load any task-specific specialization, such as
   `.claude/skills/add-ejkernel-kernel/SKILL.md`.
3. Keep lifecycle rules here. Keep package-specific commands, symptoms, and examples in the specialization or in
   `.claude/ops/OPS.md`.

## Specialization Policy

A good EasyDeL skill should do: route the agent into the real repo, not into generic advice. When creating or editing a
skill:

- Include frontmatter with `name` and a concrete `description`.
- Add `allowed-tools` only when the task needs a constrained tool surface, as in `.claude/skills/review-pr/SKILL.md`.
- Declare specialization explicitly when the skill builds on this one.
- Cite only verified scripts, flags, functions, env vars, docs, and paths.
- Point symptoms to `.claude/ops/OPS.md` when they are operational runbooks.
- Prefer a short "first reads" list over paraphrasing existing package docs.
- Include exact verification commands only after checking they exist.
- Omit any target you did not verify.

## Load Order

1. Read `WORKSPACE.md`.
2. Read the touched package's `pyproject.toml` and relevant docs under
   `libs/<package>/docs/`.
3. If the task is operational or failure-prone, read `.claude/ops/OPS.md`.
4. If the task needs test selection, load
   `.claude/skills/test-workspace/SKILL.md`.

## Grounding Rule

Do not cite a script, flag, entry point, env var, function, or file path until you have opened it or found it with `rg`.
If a useful target should exist but does not, create the routing doc or helper first instead of inventing behavior
inside the skill output.

## Workflow

1. Inventory the real surface:
    - `git status --short`
    - `git diff --stat`
    - `rg` for the relevant symbol, flag, error text, or entry point.
2. State the current hypothesis and the exact evidence that would falsify it.
3. Make the smallest code or doc change that tests the hypothesis.
4. Run focused verification first. Use
   `.claude/skills/test-workspace/SKILL.md` for test selection.
5. Keep only changes supported by the verification result.
6. Record durable design or research notes in `.claude/projects/` only when the task is too large to finish in the
   current pass. Operational recovery steps belong in `.claude/ops/OPS.md`.

## Project Notes

For long-running work, create a note under `.claude/projects/<topic>.md` with:

- goal and stop condition
- baseline command and result
- hypotheses tested
- exact command/output summary for each meaningful attempt
- negative results worth preserving
- next action

Do not create GitHub issues, W&B runs, tags, or release artifacts unless the user asks for that workflow or the task
already depends on it.

## Environment Defaults

CPU JAX checks use:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

TPU uses a single-process libtpu lock. If the machine is busy, run only unrelated host-side probes with
`JAX_PLATFORMS=cpu`. CPU checks do not validate TPU kernels, Mosaic lowering, eSurge runtime behavior, or performance
claims.

## Reporting

Report:

- Files changed.
- Verification commands and outcomes.
- Remaining risk, especially skipped hardware or end-to-end checks.
- For performance work, direct baseline vs candidate numbers with hardware, shape/dtype, compile-including timing when
  relevant, and steady-state timing.

Do not include self-credit trailers in commits, PRs, tags, or release notes.
