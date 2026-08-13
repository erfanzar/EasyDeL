---
name: debug-ejkernel-ops
description: Debug or extend the ejkernel execution framework under libs/ejkernel/ejkernel/ops. Use for Kernel base class, Executor, ConfigSelectorChain, autotuning, profiling, device fingerprinting, config caches, or invocation recording.
---

# Skill: Debug ejKernel Ops Framework

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the failure is in the execution path of
`libs/ejkernel/ejkernel/ops`: autotune, config caching, executor dispatch, profiling, or device fingerprinting.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/ops/__init__.py`
- `libs/ejkernel/ejkernel/ops/core/kernel.py`
- `libs/ejkernel/ejkernel/ops/execution/executor.py`
- `libs/ejkernel/ejkernel/ops/config/selection.py`
- `libs/ejkernel/ejkernel/ops/utils/fingerprint.py`
- `libs/ejkernel/ejkernel/ops/execution/tuning.py`

## Typical Tasks

1. Add a new `Kernel` subclass with `run`, `heuristic_cfg`, and optionally
   `candidate_cfgs` / `fwd_with_residuals` / `vjp`.
2. Introduce a platform-specific dispatch path (e.g., `run_tpu`,
   `run_shard_map_gpu`) or a `create_shard_map_wrapper`.
3. Tune or extend config selection: policies, persistent cache, overlays, or autotune candidate generation.
4. Debug invocation recording, cache hits/misses, or profiler-based autotuning regressions.

## Routing

- Adding a new kernel: load `.claude/skills/add-ejkernel-kernel/SKILL.md`.
- Backend-specific kernel performance: load the matching backend skill (`optimize-triton-gpu`, `optimize-pallas-tpu`,
  `optimize-cuda-gpu`,
  `optimize-tilelang-gpu`).
- EasyDeL side wiring: load
  `.claude/skills/port-ejkernel-to-easydel-operation/SKILL.md`.
- Quantized operations: load `.claude/skills/ejkernel-quantization/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/ejkernel/test/ops/
```

Autotune and profiling paths often need real hardware; document which behaviors were validated on CPU and which require
a GPU/TPU run.
