---
name: investigate-bug
description: Root-cause a failure in the EasyDeL workspace with hypothesis-driven debugging
---

# Investigate a Bug

Root-cause the reported symptom, failing test, or error message in the EasyDeL monorepo.

Act as the `debugger` agent (`.agents/agents/debugger.md`) following `.agents/skills/run-research/SKILL.md`. Before
treating a symptom as a code bug, check `.agents/ops/OPS.md` for infrastructure routes (bad TPU nodes, single `libtpu`
process lock, `/tmp/slice_*.dump` disk-pressure cascade, stale `~/ejkernel-presistent-cache/`, GPU driver/CUDA extras)
— a matching infra route wins.

## Process

1. **Reproduce cleanly**:
   - Run the smallest probe on the target accelerator using tiny configs (`libs/easydel/tests/modules/conftest.py`
     fixtures), honoring the single `libtpu` process lock on TPU hosts:
     - TPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu uv run pytest <target>`
     - GPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu uv run pytest <target>`
   - Run on CPU **only** for pure-Python/Rust non-computation logic (parsers, configs/CLI, `libs/eray/tests`) or
     `libs/surgecore` host-side debugging (`cargo test --locked --no-default-features`,
     `PYO3_PYTHON="$(which python)" cargo test --locked`, `ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests`).
   - If no accelerator is available for a JAX computation failure, state so explicitly and report the probe as
     **unverified on hardware** — never substitute a CPU run as validation.
2. **Iterate scientifically**: state a falsifiable hypothesis, run the single smallest experiment that tests it, and
   vary one variable per step.
3. **Follow domain triage routes**: use `.agents/agents/debugger.md` and the matching domain skill rather than ad-hoc
   guessing:
   - Training OOM / compile blowups → `.agents/skills/debug-training-oom/SKILL.md`
   - eSurge runtime / page-table sync (`PageTable.cpu_version`, `BatchMetadataPreparer.invalidate_page_table_cache()`,
     DP page locality, `EngineCoreDriver` ↔ `surgecore` `EngineCore`/`KvCore`) → `.agents/skills/debug-esurge/SKILL.md`
     and `.agents/skills/develop-surgecore/SKILL.md`
   - KV / hybrid / MLA / TurboQuant cache failures → `.agents/skills/debug-easydel-cache/SKILL.md`
   - TPU setup / bad nodes / Ray → `.agents/skills/debug-tpu-setup/SKILL.md` and
     `.agents/skills/eray-orchestration/SKILL.md`
   - Mesh / `PartitionSpec` / sharding → `.agents/skills/debug-eformer-escale/SKILL.md` and
     `.agents/skills/spectrax-sharding/SKILL.md`
   - eJKernel dispatch / autotune / stale `~/ejkernel-presistent-cache/` → `.agents/skills/debug-ejkernel-ops/SKILL.md`
     (for kernel numeric parity, compare directly against `platform="xla"` or an XLA reference implementation)
4. **Preserve multi-session state**: for investigations spanning more than one session, record the baseline command,
   hypotheses tested, and negative results in `.agents/projects/<topic>.md` (following `.agents/projects/README.md`).

## Deliverable

Report the verified root cause with verbatim failing/passing evidence, the minimal fix (only if the user requested a
fix — otherwise report findings only), and the focused regression test covering the bug. Never soften failing output.
