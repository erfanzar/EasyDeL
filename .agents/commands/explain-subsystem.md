---
name: explain-subsystem
description: Explain how an EasyDeL workspace subsystem works, grounded in live source code
---

# Explain a Subsystem

Explain the requested subsystem of the EasyDeL monorepo (e.g., eSurge scheduling, fused projection layouts, SpectraX
MPMD runtime, or ejkernel dispatch), grounded in the current tree.

## Ground Rules

1. Start from `WORKSPACE.md`, `AGENTS.md`, and `.agents/repo-map.yaml` so all packages (`easydel`, `spectrax`,
   `ejkernel`, `eformer`, `eray`, `surgecore`, `eyvan`, and standalone `libs/esurge-dashboard`) are covered from the
   start. Then read the live source code before explaining — reflect the code as it is today, not assumptions from
   similar systems.
2. Structure the explanation as:
   - **(a) Ownership and layering**: what the subsystem owns and where it sits in the workspace layering contract.
   - **(b) Main flow**: step-by-step execution with concrete class/function names and repo-root-relative file paths.
   - **(c) Extension points**: registries, mixins, or config hooks, plus the matching skill in `.agents/skills/README.md`.
   - **(d) Gotchas**: verified hazards that bite newcomers (only ones confirmed in code).
3. Prefer one concrete worked example over exhaustive enumeration — for example, trace one request through eSurge
   (`libs/easydel/easydel/inference/esurge/engine/core_driver.py` → `libs/surgecore/` Rust `EngineCore`/`KvCore`), one
   weight tensor through HF conversion, or one microbatch through a 1F1B pipeline schedule.
4. If the subsystem already has dedicated package docs (`libs/*/docs/**` or `libs/*/README.md` — note that `libs/eray`,
   `libs/surgecore`, and `libs/esurge-dashboard` document themselves via `README.md`) or a skill, cite them rather than
   paraphrasing them wholesale.
5. Flag any contradiction found between live code and existing docs explicitly — report it as a finding rather than
   silently reconciling it.
