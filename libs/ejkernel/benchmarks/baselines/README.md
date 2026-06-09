# Benchmark Baselines

This directory is the default home for performance baseline JSON artifacts.

Notes:

- `benchmark_outputs/` is gitignored and intended for ephemeral local runs.
- Baselines under `benchmarks/baselines/` are intended to be long-lived and can be committed when appropriate.

## QMM LLM Gate

Default baseline path (used by `benchmarks/qmm_perf_gate.py`):

- `benchmarks/baselines/qmm_llm_<backend>_xla_strict.json` where `<backend>` is
  `gpu`, `tpu`, `mps`, or `cpu` (and `cuda` is normalized to `gpu`).

Examples:

- GPU: `benchmarks/baselines/qmm_llm_gpu_xla_strict.json`
- TPU: `benchmarks/baselines/qmm_llm_tpu_xla_strict.json`

Generate or refresh the baseline:

```bash
.venv/bin/python benchmarks/qmm_perf_gate.py --write-new-baseline
```

## Quant/Dequant Microbench Gate

Default baseline paths (used by `benchmarks/quant_perf_gate.py`):

- `benchmarks/baselines/quantize_<backend>.json`
- `benchmarks/baselines/dequantize_<backend>.json`

Examples:

- GPU: `benchmarks/baselines/quantize_gpu.json`, `benchmarks/baselines/dequantize_gpu.json`
- TPU: `benchmarks/baselines/quantize_tpu.json`, `benchmarks/baselines/dequantize_tpu.json`

Generate or refresh the baselines:

```bash
.venv/bin/python benchmarks/quant_perf_gate.py --write-new-baseline
```
