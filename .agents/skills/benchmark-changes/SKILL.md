---
name: benchmark-changes
description: Benchmark modified EasyDeL components against a baseline — eSurge serving throughput, trainer step time, kernel microbenchmarks, packed-prefill and sharding comparisons. Use when a change needs before/after numbers or when asked to benchmark something in this repo.
---

# Skill: Benchmark Changes

Specialization of `.agents/skills/run-research/SKILL.md`. For deep kernel
tuning load `.agents/skills/optimize-ejkernel-kernel/SKILL.md`; this skill
produces honest A/B numbers for a change.

## Harness Map (all verified in-repo)

| surface | command |
| ------- | ------- |
| eSurge serving | `python scripts/bench_esurge.py --num-prompts N --prompt-len L --output-len M --warmups 1 --trials T --json-out out.json` (`--sharding-axis-dims` in `pp,dp,fsdp,ep,tp,sp` order; builds a no-MTP workload) |
| eSurge penalized sampling | `scripts/bench_esurge_penalized_sampler.py` |
| sharded training step | `scripts/bench_llama_8b_sharded.py` |
| PP vs TP SFT comparison | `scripts/sft_pp_tp_comparison.py` |
| packed prefill (qwen3-next) | `scripts/bench_qwen3_next_packed_prefill.py` |
| scan-trace overhead | `scripts/bench_llama_scan_trace.py` |
| GDR kernel probe | `scripts/gdr_synthetic_probe.py` |
| ejkernel ops | `libs/ejkernel/benchmarks/benchmark_<op>.py` (registry: `_op_benchmark_registry.py`; baselines in `benchmarks/baselines/`) |
| speculative decoding | `libs/easydel/tests/inference/esurge/bench_specdecode.py` |

## Protocol

1. **Baseline first**: run the harness on the pre-change commit (stash or
   worktree) with fixed shapes/dtypes/sharding; save JSON/output.
2. **Candidate**: identical command, identical hardware, same process
   hygiene (no concurrent TPU users — libtpu is single-process; check
   `.agents/ops/OPS.md`).
3. **Control caches**: ejkernel autotune cache (`~/ejkernel-presistent-cache/` (override: `EJKERNEL_PERSISTENT_CACHE_DIR`)) and
   ejit compilation caches either cleared on both sides or warmed on both
   sides — never mixed.
4. **Warmup discipline**: compile excluded from steady-state numbers;
   report cold-start separately when compile cost is part of the claim.
5. **Compare distributions, not single numbers**: for eSurge diff the
   `profile_by_total_tokens` buckets; for kernels report per-shape rows.
   ≥2 trials; call out variance if it approaches the delta.

## Reporting

Hardware, shapes, dtypes, sharding dims, exact commands, baseline vs
candidate tables, and an explicit verdict per surface (improved / neutral /
regressed with magnitude). CPU results are labeled indicative-only.
`EASURGE_SYNC_INPUTS_FOR_TIMING=1` only when measuring input-prep accuracy —
it costs a device round trip. Do not benchmark speculative decoding with
`bench_esurge.py`; use `bench_specdecode.py`.
