---
name: implement-feature
description: Implement a planned or described feature in the EasyDeL workspace end to end with tests
---

# Implement a Feature

Implement the requested feature or approved plan in the EasyDeL monorepo following the workspace workflow and invariants.

## Process

1. **Load skills**: read `.agents/skills/run-research/SKILL.md`, then load the matching domain skill(s) from
   `.agents/skills/README.md` (e.g., `add-easydel-model`, `add-easydel-trainer`, `add-easydel-layer`,
   `add-ejkernel-kernel` paired with `port-ejkernel-to-easydel-operation`, `quantization-layout`, `develop-surgecore`,
   `build-dataset-pipeline`, `tool-reasoning-parser`, `spectrax-modules` / `spectrax-sharding` /
   `spectrax-pipeline-runtime` / `spectrax-serialization` / `spectrax-quantization`, `add-eformer-optimizer`,
   `eray-orchestration`). Follow every required registration, export, test, and doc surface in that skill.
2. **Plan and consult before editing**:
   - If no plan exists yet and the change spans subsystems or packages, run `.agents/commands/plan-feature.md` first.
   - Consult the owning domain agent before editing when touching their invariants (`sharding-expert`,
     `quantization-expert`, `inference-expert`, `kernel-expert`, `jax-expert`, `tpu-expert`, `gpu-expert`,
     `model-expert`, `trainer-expert`, `rlhf-expert`, `data-expert`).
3. **Implement cleanly**:
   - Make the smallest complete change matching surrounding style (`import typing as tp`, `from jax import numpy as jnp`,
     `import spectrax as spx`, ruff line length 121, Google docstrings).
   - Respect layering (`easydel` only imports sibling packages; foundation libs and `surgecore` never import `easydel`
     or each other).
   - Never bypass registries, never hand-edit generated `docs/api_docs/**`, and never hand-edit version pins (use
     `scripts/release.sh`).
   - Respect repo-wide runtime contracts: **no new `os.environ` knobs** (wire settings through config dataclasses,
     ContextVars, or explicit setters), use eFormer `ePath` for GCS-capable I/O, and preserve float32
     `attn_softmax_dtype` accumulation.
4. **Write tests and verify**:
   - Add non-tautological tests following `.agents/skills/generate-tests/SKILL.md` and
     `.agents/skills/test-workspace/SKILL.md`.
   - Run JAX computation tests on an accelerator (TPU shown; for GPU use
     `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Use CPU only for pure-Python/Rust non-computation tests
     (`libs/eray/tests`, `libs/surgecore/tests`, parsers, configs, `@pytest.mark.cpu_ok`):

     ```bash
     env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
       uv run pytest <focused targets>
     uv run lint-imports
     uv run pre-commit run --files <touched files>
     ```

   - When touching `libs/surgecore` (Rust PyO3 crate):

     ```bash
     (cd libs/surgecore && cargo fmt --check && \
       PYO3_PYTHON="$(which python)" cargo clippy --locked --all-targets -- -D warnings && \
       PYO3_PYTHON="$(which python)" cargo test --locked && \
       cargo test --locked --no-default-features)
     ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests
     ```

   - When touching `libs/esurge-dashboard` (Rust Axum binary):

     ```bash
     (cd libs/esurge-dashboard && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test)
     ```

5. **Review and report**:
   - Run a correctness pass using `.agents/commands/review-changes.md` (or the `reviewer` agent).
   - Report files changed, exact verification commands with outcomes, and remaining risk. Without an accelerator,
     report computation tests as **unverified on hardware** — never as CPU-validated. Do not claim training or serving
     readiness from constructor-only tests, and do not commit unless requested (and never with self-credit trailers).
