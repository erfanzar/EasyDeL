---
name: summarize-architecture
description: Concise architecture summary of the EasyDeL monorepo for onboarding or context refresh
---

# Summarize Monorepo Architecture

Produce a concise, under-one-page architecture summary of the EasyDeL monorepo for onboarding or context refresh.

Source it from `WORKSPACE.md` (authoritative), `AGENTS.md`, and `.agents/repo-map.yaml` — spot-check counts, paths, and
symbols against the live tree and flag any doc discrepancies you discover.

## Required Sections (in Order)

1. **Workspace packages and layering**: the seven uv-workspace packages (`easydel`, `spectrax`, `ejkernel`, `eformer`,
   `eray`, `surgecore`, `eyvan`) plus the standalone `libs/esurge-dashboard` Rust crate, a one-line job for each, and the
   layering contract (only `easydel` imports sibling packages; companion libs never import `easydel` or each other) and
   why it exists.
2. **Core end-to-end flows** (one short paragraph each with concrete class/module names):
   - **Model loading & forward**: `AutoEasyDeLModelForCausalLM`, HF conversion + fused layouts, `UnifiedAttention` →
     `FlexibleAttentionModule` → `OperationRegistry.create` → `ejkernel`.
   - **Data pipeline**: `libs/easydel/easydel/data/` (`Pipeline` fluent builder, transforms DSL, Ray distributed
     preprocessing).
   - **Training**: `BaseTrainer`, `EasyDeLState`, eFormer `OptimizerFactory`, SpectraX/TensorStore checkpointing.
   - **eSurge serving**: `eSurge` engine, Python↔Rust boundary (`engine/core_driver.py` ↔ `libs/surgecore`
     `EngineCore`/`KvCore`), compiled bucket runners, OpenAI API server + tool/reasoning parsers, and
     `libs/esurge-dashboard`.
3. **Pipeline parallelism**: how SpectraX true-MPMD per-rank executables and schedules (`SpxMesh`, `sxstage_iter`,
   `pipeline_step`) differ from the SPMD path and stock JAX.
4. **Extension-point philosophy**: registries everywhere (models, trainers, operations, kernels, tool/reasoning
   parsers, optimizers) — adding new implementations without modifying shared code.
5. **Development and release loop**: `uv sync` (requires a stable Rust/`cargo` toolchain to compile `libs/surgecore`),
   Rust crate checks (`cargo fmt`/`clippy`/`test` for `libs/surgecore` and `libs/esurge-dashboard`), the accelerator
   test env (TPU/GPU; CPU trio only for non-computation tests), `uv run lint-imports`, `uv run pre-commit`, and the
   two-step `scripts/release.sh` / `scripts/publish.sh` + subtree-mirror flow in two sentences.

Link repo-root-relative file paths for every load-bearing claim.
