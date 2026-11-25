---
name: debug-ejkernel-ops
description: Debug or extend the ejkernel execution framework under libs/ejkernel/ejkernel/ops. Use for Kernel base class, Executor, ConfigSelectorChain, autotuning, profiling, device fingerprinting, config caches, invocation recording, or the shipped ops/tuned table.
---

# Skill: Debug eJKernel Ops Framework

Specialization of `.agents/skills/run-research/SKILL.md` for diagnosing or extending `libs/ejkernel/ejkernel/ops/` —
`Kernel` dispatch, `Executor`, `ConfigSelectorChain`, `ConfigCache`, `PersistentCache`, `Tuner`, device fingerprinting,
invocation recording, offline autotuning, and `ejkernel.ops.tuned`.

## First Reads

Read the authoritative source files before editing:

- `WORKSPACE.md` and `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/ops/__init__.py`
- `libs/ejkernel/ejkernel/ops/core/kernel.py` (`Kernel`, `Invocation`, `_get_platform_method`, `_has_custom_vjp`)
- `libs/ejkernel/ejkernel/ops/execution/executor.py` (`Executor`, `_choose_heuristics_only`, `_prefer_cuda_cfg`)
- `libs/ejkernel/ejkernel/ops/config/selection.py` (`ConfigSelectorChain`, `AutotunePolicy`, `Tuner`,
  `policy_override`, `forward_autotune_only`, `log_autotune_progress`, `set_autotune_progress`)
- `libs/ejkernel/ejkernel/ops/config/cache.py` (`ConfigCache`, `overlay_cache`)
- `libs/ejkernel/ejkernel/ops/config/persistent.py` (`PersistentCache`, `PROVENANCE_AUTOTUNE`, `TRUSTED_PROVENANCE`,
  `CFG_CLASS_KEY`)
- `libs/ejkernel/ejkernel/ops/utils/fingerprint.py` (`device_fingerprint`, `get_device_platform`, `sharding_fingerprint`)
- `libs/ejkernel/ejkernel/ops/execution/tuning.py` (`autotune`, `autotune_recorded`, `AutotuningResult`)
- `libs/ejkernel/ejkernel/ops/execution/offline.py` (`autotune_lowered`)
- `libs/ejkernel/ejkernel/ops/execution/profiler.py` (`Profiler`)
- `libs/ejkernel/ejkernel/ops/execution/batch.py` (`vmap_with_config`, `pmap_with_config`)
- `libs/ejkernel/ejkernel/ops/registry.py` (`record_invocation`, `get_invocations`)
- `libs/ejkernel/ejkernel/ops/tuned/{__init__,_lookup,_store,__main__}.py`

## Architecture & Selection Invariants (Key Debugging Traps)

1. **7-tier `ConfigSelectorChain` order (`config/selection.py`):**
   1. Explicit `override_cfg`
   2. `overlay_cache` temporary context override
   3. In-memory `ConfigCache`
   4. Disk `PersistentCache`
   5. `Tuner` autotuning (`candidate_cfgs`)
   6. Kernel heuristics (`heuristic_cfg`)
   7. Error if no config is available
2. **`Executor._choose_heuristics_only` fast path (`execution/executor.py`):** when
   `policy.cache_miss_fallback == "heuristics"`, `Executor` bypasses `ConfigSelectorChain.choose()` completely
   (including any `on_event` hooks!) and resolves Override → Overlay → Memory → Persistent → `heuristic_cfg`. Results
   are written to memory cache only — never persisted to disk — so heuristic fallbacks never poison the shared
   `PersistentCache`.
3. **`Executor._prefer_cuda_cfg` silent NVIDIA upgrade (`execution/executor.py`):** on NVIDIA GPUs, `Executor` silently
   rewrites `cfg.platform` from `"auto"` or `"triton"` to `"cute"` (preferred) or `"cuda"` whenever a CuTe/CUDA
   implementation exists, unless `override_cfg` is supplied or an explicit non-`"auto"` `platform=` kwarg was passed.
   When debugging Triton on NVIDIA hardware, pass explicit `platform="triton"` or an explicit `cfg`.
4. **`PersistentCache` provenance & tuple preservation (`config/persistent.py`):**
   - Default directory is `~/ejkernel-presistent-cache/<op>.json` (exact spelling; override with
     `EJKERNEL_PERSISTENT_CACHE_DIR`).
   - On read, **only** entries whose provenance is in `TRUSTED_PROVENANCE` (`PROVENANCE_AUTOTUNE`) are returned. Legacy
     or manual entries without provenance are ignored as cache misses.
   - `CFG_CLASS_KEY` (`__ejk_cfg_class__`) records the `ejkernel` dataclass/Pydantic class and converts JSON lists back
     to tuples on load (critical for tuple fields such as per-mode block sizes).
5. **`method="shard_map"` bypasses `custom_vjp` (`execution/executor.py`):** when `method="shard_map"`, `Executor`
   dispatches to `kernel.create_shard_map_wrapper(...)` instead of wrapping `fwd_with_residuals`/`vjp`. Kernels using
   `shard_map` must handle custom VJP inside their shard-map wrapper.
6. **4-tier `_get_platform_method` lookup & MPS caveat (`core/kernel.py`, `utils/fingerprint.py`):** resolves methods in
   order `{method}_{context}_{platform}` → `{method}_{context}` → `{method}_{platform}` → `{method}` (e.g.,
   `run_shard_map_gpu` → `run_shard_map` → `run_gpu` → `run`). Note `get_device_platform()` only maps `"gpu"`, `"tpu"`,
   and `"cpu"` and returns `"unknown"` on Apple Silicon MPS (`Backend.MPS`), so MPS falls through to generic methods
   unless handled via `jax.default_backend()`.

## Shipped Tuned Table (`ejkernel.ops.tuned`)

`libs/ejkernel/ejkernel/ops/tuned/tuned_kernels.db` stores measured `(platform, config)` winners and runner-up timings:

- Runtime lookup: `tuned_choice(kernel, dtypes=..., shape=...)` (returns `TunedEntry | None`; `None` preserves existing
  heuristics). Override store programmatically via `set_tuned_store(...)` or via `EJKERNEL_TUNED_DB=/path/to/test.db`.
- CLI inspection, sweeps, and merging:
  ```bash
  uv run python -m ejkernel.ops.tuned stats
  uv run python -m ejkernel.ops.tuned dump --kernel <op>
  uv run python -m ejkernel.ops.tuned sweep --kernel <op> --out /tmp/<op>.db
  uv run python -m ejkernel.ops.tuned merge --into libs/ejkernel/ejkernel/ops/tuned/tuned_kernels.db /tmp/<op>.db
  ```

## Controls: Env Vars & Context Managers

Follow existing env conventions (do not invent new `os.environ` knobs — use config dataclasses, ContextVars, or setters):

- **Env vars:** `EJKERNEL_AUTOTUNE_POLICY` (`heuristics` | `autotune`), `EJKERNEL_PERSISTENT_CACHE_DIR`,
  `EJKERNEL_TUNED_DB`, `EJKERNEL_OPS_RECORD=1` (populates invocation registry for `autotune_recorded` /
  `autotune_lowered`), `EJKERNEL_OPS_STAMP=none|hash|json` (stamps HLO/xprof scopes), `EJKERNEL_AUTOTUNE_PROGRESS=1`,
  `EJKERNEL_LOG_AUTOTUNE=1`.
- **Context managers & setters:** `policy_override(selector, ...)`, `overlay_cache(...)`, `forward_autotune_only()`,
  `log_autotune_progress()`, `set_autotune_progress(bool)`, `set_tuned_store(...)`.

## Routing

- Adding a new kernel or wrapper: `.agents/skills/add-ejkernel-kernel/SKILL.md`
- Kernel-level optimization: `.agents/skills/optimize-ejkernel-kernel/SKILL.md` + backend skill (`optimize-pallas-tpu`,
  `optimize-triton-gpu`, `optimize-cuda-gpu`, `optimize-cute-gpu`, `optimize-tilelang-gpu`)
- Wiring an ejkernel op into EasyDeL: `.agents/skills/port-ejkernel-to-easydel-operation/SKILL.md`
- Quantization / `quantized_matmul`: `.agents/skills/ejkernel-quantization/SKILL.md`

## Verification

Run on accelerator hardware (TPU or GPU); a CPU run of computation tests is not validation. Without an accelerator,
report the change as **unverified on hardware**.

```bash
# TPU
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest \
    libs/ejkernel/test/ops/ \
    libs/ejkernel/test/test_platform_specific_ops.py \
    libs/ejkernel/test/modules/operations/test_autotune_candidate_surface.py

# GPU
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest \
    libs/ejkernel/test/ops/ \
    libs/ejkernel/test/test_platform_specific_ops.py \
    libs/ejkernel/test/modules/operations/test_autotune_candidate_surface.py
```
