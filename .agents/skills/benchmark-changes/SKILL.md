---
name: benchmark-changes
description: Benchmark modified EasyDeL workspace components against a baseline — ejkernel microbenchmarks, eSurge serving and speculative decoding, SpectraX pipeline/sharded LLM training, and EasyDeL trainer step timing. Use when a change needs honest before/after accelerator numbers.
---

# Skill: Benchmark Changes

Specialization of `.agents/skills/run-research/SKILL.md` for producing rigorous A/B hardware measurements of a change.
For deep kernel optimization load `.agents/skills/optimize-ejkernel-kernel/SKILL.md`; for broad 5-dimension change
audits load `.agents/skills/performance-audit/SKILL.md`.

## Verified In-Repo Harness Map

Every path below exists in this repository (removed legacy root benchmark scripts are gone — write a small throwaway
harness using the shared helpers when a surface has no standing script):

| Surface | Verified In-Repo Path & Usage |
|---|---|
| **ejkernel multi-op / multi-platform matrix** | `libs/ejkernel/benchmarks/benchmark_suite.py` (driven by `EJKERNEL_BENCH_OPS`, `EJKERNEL_BENCH_PLATFORMS`, `EJKERNEL_BENCH_CONFIG_LIMIT`, `EJKERNEL_BENCH_WARMUP`, `EJKERNEL_BENCH_ITERS`, `EJKERNEL_BENCH_OUTPUT_DIR`, `EJKERNEL_BENCH_IGNORE_PLATFORMS`; specs in `libs/ejkernel/benchmarks/_op_benchmark_registry.py`; single-op shims `libs/ejkernel/benchmarks/benchmark_<op>.py`). Note: `libs/ejkernel/benchmarks/baselines/` are historical artifacts, not an active perf gate. |
| **ejkernel fused losses** | `libs/ejkernel/scripts/benchmark_fused_losses.py` (plus `bench_final.py`, `bench_kl_target.py`) |
| **eSurge speculative decoding (Qwen3.5 MTP)** | `libs/easydel/tests/inference/esurge/bench_specdecode.py` and `libs/easydel/tests/inference/esurge/sweep_specdecode_k.py`. Configured via `EASYDEL_BENCH_*` env vars (not CLI flags: `EASYDEL_BENCH_CKPT`, `EASYDEL_BENCH_MODES`, `EASYDEL_BENCH_TRIALS`, `EASYDEL_BENCH_WARMUPS`, `EASYDEL_BENCH_SHARDING_AXIS_DIMS`, `EASYDEL_BENCH_PROFILE_JSON`, `EASYDEL_BENCH_XPROF_DIR`). Note: `bench_specdecode.py` hardcodes `JAX_PLATFORMS=tpu` / `JAX_PLATFORM_NAME=tpu` at lines 39–40. |
| **eSurge serving (general / non-spec)** | Build a focused throwaway harness using `libs/easydel/easydel/inference/esurge/benchmarking.py` (`make_prompts`, `make_real_prompts`, `all_finished`, `zero_schedule_needs_update`) and collect per-step timings from `RunnerPerfTracker` (`history: deque[RunnerPerfSample]`, `phase_history`) in `libs/easydel/easydel/inference/esurge/runners/perf.py`. |
| **SpectraX / pipeline / sharded LLM training** | `libs/spectrax/benchmarks/` (`bench.py`, `bench_mpmd_jit_vs_spmd.py`, `bench_mpmd_jit_llama8b.py`, `train_llm.py`, `train_pipeline.py`, `llama3_8b.py`, `llama3_3b_scan.py`) |
| **EasyDeL trainer step timing** | Focused `python -m easydel.scripts.elarge --config <yaml>` or short trainer loop (`.agents/skills/train-elarge/SKILL.md`) with `TrainingArguments.profiler_path` + `profiler_stop_step` for steady-state traces. |

## A/B Protocol

1. **Baseline first:** run on the pre-change commit (worktree or stash) with frozen shapes, dtypes, and sharding
   (`pp,dp,fsdp,ep,tp,sp` order); save raw JSON/Markdown output.
2. **Hardware isolation:**
   - **TPU:** `libtpu` holds a single-process lock — verify no concurrent TPU job (`.agents/ops/OPS.md`), and run with
     `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu`.
   - **GPU:** lock clocks where permitted (`nvidia-smi -lgc <min>,<max>`), and run with
     `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`.
3. **Control all caches symmetrically** (either cleared on both sides or warmed identically on both sides — never mixed):
   - **ejkernel selection & cache:** pin `EJKERNEL_AUTOTUNE_POLICY=heuristics` (or `autotune` explicitly), control
     `~/ejkernel-presistent-cache/` (`EJKERNEL_PERSISTENT_CACHE_DIR`), and keep `EJKERNEL_TUNED_DB` identical.
   - **ejit compiled executables (`libs/ejkernel/ejkernel/callib/_ejit.py`):** control `EASYDEL_CACHE_COMPILES`
     (`ECACHE_COMPILES`), `EASYDEL_RECOMPILE_FORCE` (`RECOMPILE_FORCE`), and `EJKERNEL_COMPILE_CACHE_DIR`.
   - **Triton compile cache (GPU):** control `EJKERNEL_TRITON_CACHE_COMPILES` / `EJKERNEL_TRITON_CACHE_DIR`.
4. **Warmup & compile separation:** exclude first-call compile from steady-state timings; measure and report cold-start
   compile separately whenever compile cost is part of the claim.
5. **Compare buckets and distributions, not single averages:**
   - **eSurge:** group `RunnerPerfTracker.history` (`RunnerPerfSample.total_tokens`, `num_scheduled_reqs`, `total_time`,
     `agg_tps`) and `phase_history` by `(total_tokens, num_scheduled_reqs)` bucket instead of averaging regressions away
     in aggregate tok/s. Set `EASYDEL_SYNC_INPUTS_FOR_TIMING=1` (`runners/execution_manager.py`) only when attributing
     input-prep latency (it inserts a device round-trip sync).
   - **Kernels & training:** report per-shape rows across ≥2 trials (or `EJKERNEL_BENCH_WARMUP=5`, `EJKERNEL_BENCH_ITERS=30`)
     and flag variance if it approaches the observed delta.

## Reporting & Accelerator Policy

Report hardware model, device count, shapes, dtypes, sharding dims (`pp,dp,fsdp,ep,tp,sp`), cache policy, exact baseline
and candidate commands, raw artifact paths, per-shape/per-bucket comparison tables, and an explicit verdict
(`improved` / `neutral` / `regressed` with magnitude).

**Hardware-only rule:** computation benchmarks MUST run on target TPU or GPU hardware. Never report CPU timings as
"indicative-only" performance evidence; if no accelerator is available, report the benchmark as **unverified on
hardware**.
