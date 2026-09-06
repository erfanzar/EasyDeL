---
name: ejkernel-quantization
description: Work on ejkernel weight-compression formats and the quantized_matmul operation under libs/ejkernel/ejkernel/quantization and modules/operations/quantized_matmul. Use for TurboQuant, affine/NF4/MXFP/NVFP4 packing, grouping, bitpacking, or runtime config tuning.
---

# Skill: Work On ejKernel Quantization

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work involves
`libs/ejkernel/ejkernel/quantization` or the `quantized_matmul` operation.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/quantization/__init__.py`
- `libs/ejkernel/ejkernel/quantization/quantized_array.py`
- `libs/ejkernel/ejkernel/quantization/_quants/quantizations.py`
- `libs/ejkernel/ejkernel/quantization/_utils/bitpack.py`
- `libs/ejkernel/ejkernel/quantization/_utils/qparams.py`
- `libs/ejkernel/ejkernel/modules/operations/quantized_matmul.py`
- `libs/ejkernel/ejkernel/modules/operations/configs.py`

## Typical Tasks

1. Add or extend a quantization mode / bit-width in
   `_quants/quantizations.py` and update `resolve_qparams`.
2. Wire a new fused backend / kernel family into `QuantizedMatmul` and update
   `modules/operations/configs.py`.
3. Fix layout / axis / transpose contract mismatches using
   `validate_packed_quantized_matmul_layout`.
4. Tune `QuantRuntimeConfig` defaults or quant/dequant runtime autotuning for a backend / shape.

## Routing

- EasyDeL quantized linear / fused projection layout: load
  `.xerxes/skills/quantization-layout/SKILL.md`.
- New backend kernel implementation: load
  `.xerxes/skills/add-ejkernel-kernel/SKILL.md`.
- Performance tuning: load `.xerxes/skills/optimize-ejkernel-kernel/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/ejkernel/test/modules/operations/test_quantized_matmul.py
```

Also run `libs/ejkernel/test/quantization/` unit tests if the quantization container or bitpacking changed.
