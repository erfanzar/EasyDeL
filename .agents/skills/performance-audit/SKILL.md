---
name: performance-audit
description: Post-implementation performance audit for EasyDeL workspace changes. Use after a feature or fix lands to inspect compile cost, HBM memory, communication volume, kernel selection, and throughput/FLOPs/MFU impact before claiming a change is performance-neutral or an improvement.
---

# Skill: Performance Audit

Specialization of `.agents/skills/run-research/SKILL.md` for whole-change performance review. Delegate A/B benchmark
execution to `.agents/skills/benchmark-changes/SKILL.md`, kernel-level tuning to
`.agents/skills/optimize-ejkernel-kernel/SKILL.md`, OOM attribution to `.agents/skills/debug-training-oom/SKILL.md`, and
sharding/collective audits to `.agents/skills/distributed-review/SKILL.md`.

## First Reads

- `WORKSPACE.md`
- The diff under audit (`git diff --stat`, then the hot paths)
- `.agents/ops/OPS.md` when TPU locks or serving throughput are in scope

## Audit Dimensions

Work through all five dimensions; never skip one silently — write `"not applicable because <reason>"` when a dimension
is genuinely untouched.

1. **Compile cost**
   - Inspect new or changed `jit`/`ejit` signatures (`libs/easydel/easydel/utils/compiling_utils.py` re-exports
     `libs/ejkernel/ejkernel/callib/_ejit.py`; compile-time constants live in `libs/easydel/easydel/utils/jit_context.py`).
     Watch for unhashable Python objects, changing shapes, or dynamic static args that trigger recompiles.
   - **eSurge buckets:** verify whether `(num_tokens, padded_num_reqs)` bucket cardinality changed and check
     `libs/easydel/tests/inference/esurge/runners/test_compile_buckets.py` and
     `libs/easydel/tests/inference/esurge/runners/test_no_recompile_after_warmup.py`.
2. **Memory (device HBM)**
   - Confirm `auto_remat` save/exclude policies still match `checkpoint_name` tags (`libs/easydel/easydel/infra/utils.py`).
   - Confirm loss chunking (`LossConfig.chunk_vocab_size`, `chunk_token_size`) and blockwise FFN paths are unaffected.
   - Monitor **JAX device HBM** via `SMPMemoryMonitor` in `libs/easydel/easydel/utils/analyze_memory.py` (polls
     `device.memory_stats()` — `bytes_limit`, `bytes_in_use`, `peak_bytes_in_use` on `jax.local_devices()`; this is a
     device HBM monitor, not a host RAM probe), paired with `compiled.memory_analysis()` and `jax.profiler`
     `memory_viewer`. For OOM diagnosis route to `.agents/skills/debug-training-oom/SKILL.md`.
3. **Communication**
   - Check RowParallel all-reduces, MoE all-to-alls, and PP send/recv for unintended implicit all-gathers or reshardings
     introduced by sharding or layout changes (see `.agents/skills/distributed-review/SKILL.md`).
   - A static sharding/jaxpr inspection on a fake 8-device CPU mesh is only a coarse structural sanity check — CPU XLA
     lowering does **not** reflect TPU/GPU custom calls or backend collective fusions. Verify real collective volume in
     target-hardware HLO/xprof traces.
4. **Kernel selection**
   - Verify which backend actually ran (a silent XLA fallback looks like a severe regression). Keep EasyDeL and ejkernel
     controls distinct:
     - **EasyDeL ops:** `FORCE_NATIVE_RUNTIME=1` (`libs/easydel/easydel/operations/_base_operation.py`) forces
       `BaseOperation.__call__` to `forward_native` (and ejkernel adapters should forward `platform="xla"`). Note
       `FORCE_NATIVE_RUNTIME` is **never read inside `libs/ejkernel/`**.
     - **ejkernel ops:** inspect/control dispatch via `detect_platform()` (`libs/ejkernel/ejkernel/modules/base.py`),
       `kernel_registry.list_implementations()`, explicit `platform=` (`xla`/`pallas`/`cute`/`cuda`/`triton`/`tilelang`),
       `EJKERNEL_AUTOTUNE_POLICY`, `EJKERNEL_OPS_STAMP=hash|json`, `~/ejkernel-presistent-cache/`
       (`EJKERNEL_PERSISTENT_CACHE_DIR`), and `EJKERNEL_TUNED_DB`. Remember NVIDIA GPU auto-dispatch
       (`Executor._prefer_cuda_cfg`) silently upgrades `"auto"`/`"triton"` to `"cute"`/`"cuda"` unless explicitly
       overridden.
5. **Throughput / FLOPs / MFU**
   - Run baseline-vs-candidate measurements following `.agents/skills/benchmark-changes/SKILL.md`:
     - **eSurge serving:** build a focused harness with `libs/easydel/easydel/inference/esurge/benchmarking.py` and diff
       per-bucket `(total_tokens, num_scheduled_reqs)` timings from `RunnerPerfTracker.history` / `phase_history`
       (`libs/easydel/easydel/inference/esurge/runners/perf.py`). For speculative decoding use
       `libs/easydel/tests/inference/esurge/bench_specdecode.py`. Use `EASYDEL_SYNC_INPUTS_FOR_TIMING=1` only when
       isolating input-prep overhead. State sharding dims in `pp,dp,fsdp,ep,tp,sp` order.
     - **Kernels:** `libs/ejkernel/benchmarks/benchmark_suite.py`.
     - **Training / Pipeline:** `libs/spectrax/benchmarks/` (`bench_mpmd_jit_vs_spmd.py`, `train_llm.py`,
       `train_pipeline.py`, `llama3_8b.py`) or focused EasyDeL trainer runs with `TrainingArguments.profiler_path` +
       `profiler_stop_step`.
   - **FLOPs & MFU (do NOT use `spectrax/inspect/counting.py` — it counts parameters/bytes only):**
     - XLA cost-model FLOPs/bytes: `spectrax.inspect.tabulate.hlo_cost(module, *args)`
       (`libs/spectrax/spectrax/inspect/tabulate.py`). Note XLA `cost_analysis()` omits FLOPs inside opaque Pallas
       custom calls — run it on the XLA path when needed.
     - Analytical model FLOPs/MFU: `EasyDeLBaseModule.flops_per_token()` (`libs/easydel/easydel/infra/base_module.py`)
       or `easydel.infra.utils.flops_per_token(FlopCalcConfig(...))` (`libs/easydel/easydel/infra/utils.py`). Always
       state the assumed hardware peak FLOP/s alongside MFU.

## Reporting

Report baseline vs candidate on identical accelerator hardware with shapes, dtypes, sharding dims (`pp,dp,fsdp,ep,tp,sp`),
compile time separated from steady state, and exact commands. Never average away a per-bucket regression, and for any
regressed dimension name the bottleneck and smallest credible fix direction.

**Hardware-only rule:** never substitute CPU timings or label them "CPU-only, indicative" to substantiate TPU/GPU
performance claims. Without target accelerator hardware, report the hardware dimensions as **unverified on hardware**.
