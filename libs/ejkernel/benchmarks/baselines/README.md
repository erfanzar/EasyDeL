# Benchmark Baselines

This directory is the default home for performance baseline JSON artifacts.

Notes:

- `benchmark_outputs/` is gitignored and intended for ephemeral local runs.
- Baselines under `benchmarks/baselines/` are intended to be long-lived and can be committed when appropriate.

## Historical baseline artifacts

The checked-in `qmm_llm_tpu_xla_strict.json`, `quantize_tpu.json`, and
`dequantize_tpu.json` are historical measurement artifacts. The perf-gate
scripts that originally produced them are not included in this repository;
there is currently no in-repo command to refresh or compare these baselines.
Do not interpret their presence as an active CI performance gate.
