---
name: performance-audit
description: Post-implementation performance audit for EasyDeL workspace changes. Use after a feature or fix lands to inspect compile cost, memory, communication volume, kernel selection, and throughput impact, and to identify bottlenecks before claiming the change is performance-neutral or an improvement.
---

# Skill: Performance Audit

This is a specialization of `.claude/skills/run-research/SKILL.md`. For
kernel-level optimization use `.claude/skills/optimize-ejkernel-kernel/SKILL.md`
instead; this skill audits whole changes.

## First Reads

- `WORKSPACE.md`
- the diff under audit (`git diff --stat`, then the hot files)
- `.claude/ops/OPS.md` when serving throughput is in scope

## Audit Dimensions

Work through each; skip none silently — state "not applicable because X".

1. **Compile cost**
   - New or changed jit signatures? Python objects/static args in traced
     signatures cause recompiles (`easydel/utils/compiling_utils.py` ejit
     caching, `easydel/utils/jit_context.py` for compile-time constants).
   - eSurge: did the change alter compile-bucket cardinality
     ((num_tokens, padded_num_reqs) pairs)? Check
     `libs/easydel/tests/inference/esurge/runners/test_compile_buckets.py`
     still reflects intent.
2. **Memory**
   - Remat policy still effective: `auto_remat` save/exclude names match
     `checkpoint_name` tags (`easydel/infra/utils.py`).
   - Loss chunking (`LossConfig.chunk_vocab_size`/`chunk_token_size`) and
     blockwise FFN unaffected.
   - Host memory probes: `easydel/utils/analyze_memory.py`.
3. **Communication**
   - RowParallel all-reduces, MoE all-to-alls, PP send/recv: did a sharding
     or layout change introduce an implicit all-gather? Inspect jaxpr/HLO of
     the touched function on the fake 8-device CPU mesh.
4. **Kernel selection**
   - Confirm which backend actually runs (a silent XLA fallback via
     priority dispatch looks like a regression). `FORCE_NATIVE_RUNTIME=1`
     for A/B; autotune cache under `~/ejkernel-presistent-cache/` (override: `EJKERNEL_PERSISTENT_CACHE_DIR`) pinned or cleared.
5. **Throughput / FLOPs / MFU**
   - Serving: `python scripts/bench_esurge.py --json-out ...`; compare
     `profile_by_total_tokens` buckets against a baseline run of the
     pre-change commit. Sharding dims stated in `pp,dp,fsdp,ep,tp,sp` order.
   - Training: `scripts/bench_llama_8b_sharded.py`,
     `scripts/sft_pp_tp_comparison.py`, or a focused trainer-step timing;
     FLOP counting via spectrax `inspect/counting.py`; report MFU with the
     assumed peak.

## Reporting

Baseline vs candidate on the same hardware, shapes/dtypes stated, compile
time separated from steady state, and the command lines included. CPU
timings never substantiate TPU/GPU claims — mark them "CPU-only, indicative".
If a dimension regressed, name the bottleneck and the smallest credible fix
direction; do not average a regression away across buckets.
