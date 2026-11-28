---
name: plan-feature
description: Design an implementation plan for an EasyDeL feature before any code is written
---

# Plan a Feature

Design a concrete implementation plan for the requested feature in the EasyDeL monorepo before writing any code.

Act as the `architect` agent (`.agents/agents/architect.md`). Read `WORKSPACE.md`, `AGENTS.md`, and
`.agents/repo-map.yaml`, then inspect every live code surface the feature touches — ground every file path and symbol
with `git grep` / `rg` or by opening the file (never guess paths; `rg` may not be installed in every container).

## Planning Constraints

- **Package ownership & layering**: place general mechanisms in the appropriate Python companion library (`spectrax`,
  `ejkernel`, `eformer`, `eray`, `eyvan`), eSurge host-path/UI Rust logic in `libs/surgecore` or
  `libs/esurge-dashboard`, and LLM composition in `easydel`. Only `easydel` may import sibling workspace packages.
- **Extension points & skills**: land on existing registries, mixins, or config surfaces; name the governing skill from
  `.agents/skills/README.md` for each step. Whenever adding a hardware kernel consumed by EasyDeL, pair
  `add-ejkernel-kernel` (mandatory `Platform.XLA`, `Backend.ANY` fallback) with `port-ejkernel-to-easydel-operation`
  (Golden Rule 7).
- **Config and contract hygiene**:
  - **No new `os.environ` knobs** — wire settings through config dataclasses, ContextVars, or explicit setters.
  - Any field added to `TrainingArguments` must include JSON roundtrip coverage in
    `libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py`.

## Deliverable

Follow the output checklist in `.agents/agents/architect.md`, covering:

1. One-sentence goal, chosen extension point(s), and package ownership decision.
2. Ordered, independently verifiable steps (files, registrations, matching skill, and required test per step).
3. Cross-package impact and domain agents to consult before coding (`sharding-expert`, `quantization-expert`,
   `inference-expert`, `kernel-expert`, `jax-expert`, `model-expert`, `trainer-expert`, `rlhf-expert`, `data-expert`,
   `tpu-expert`, `gpu-expert`).
4. Verification ladder: exact accelerator pytest targets (TPU/GPU; CPU trio only for non-computation tests; Rust
   `cargo fmt/clippy/test` for `libs/surgecore` or `libs/esurge-dashboard`), plus hardware validation that must be
   scheduled or reported as **unverified on hardware**.
5. Risks: sharding, recompilation, checkpoint compatibility, and HF conversion.

Do not write implementation code. If the request is ambiguous, state the chosen interpretation and why.
