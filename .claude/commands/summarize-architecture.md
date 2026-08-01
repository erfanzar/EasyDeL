---
description: Concise architecture summary of the EasyDeL monorepo for onboarding or context refresh
---

Summarize the architecture of this monorepo for someone joining the
project.

Source it from WORKSPACE.md (authoritative), CLAUDE.md, and
`.claude/repo-map.yaml` — spot-check anything that looks stale against the
tree and note discrepancies.

Cover, in order:
1. The five packages and the one-line job of each, plus the layering
   contract and why it exists.
2. The three main flows end-to-end in a paragraph each: model
   forward/loading, training, and eSurge serving — with the key class
   names.
3. How pipeline parallelism differs here from stock JAX (spectrax true-MPMD
   per-rank executables vs the SPMD path).
4. The extension-point philosophy: registries everywhere (models, trainers,
   operations, kernels, parsers, optimizers) — adding ≠ modifying.
5. The development loop: uv sync, the CPU test env trio, lint-imports,
   pre-commit, and the release/mirror flow in two sentences.

Keep it under a page. Link file paths for every load-bearing claim.
