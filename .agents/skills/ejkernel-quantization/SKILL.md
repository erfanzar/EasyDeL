---
name: ejkernel-quantization
description: Work on ejkernel weight-compression formats, TurboQuant KV-cache compression, and the quantized_matmul operation under libs/ejkernel/ejkernel/quantization and modules/operations/quantized_matmul. Use for TurboQuant, affine/NF4/MXFP/NVFP packing, grouping, bitpacking, layout validation, or runtime config tuning.
---

# Skill: Work On eJKernel Quantization

Specialization of `.agents/skills/run-research/SKILL.md` for `libs/ejkernel/ejkernel/quantization/` (weight compression,
runtime packing/dequantization, TurboQuant KV-cache compression) and `quantized_matmul` / `ragged_page_attention_v{2,3}_turboquant`.

## First Reads

Read these authoritative modules before editing:

- `WORKSPACE.md` and `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/quantization/__init__.py`
- `libs/ejkernel/ejkernel/quantization/quantized_array.py` (`QuantizedArray`, `quantize_array`, `prepack_quantized_array`)
- `libs/ejkernel/ejkernel/quantization/runtime.py` (`QuantRuntimeConfig`, `resolve_runtime_config`)
- `libs/ejkernel/ejkernel/quantization/_quants/quantizations.py` (`quantize`, `dequantize`, `prepack_quantized_weights`,
  `clear_runtime_autotune_cache`, `runtime_autotune_cache_sizes`)
- `libs/ejkernel/ejkernel/quantization/_utils/{qparams,bitpack,fp_tables,grouping}.py`
- `libs/ejkernel/ejkernel/quantization/turboquant/{__init__,codebook,matrices,ops,packing}.py`
- `libs/ejkernel/ejkernel/modules/operations/quantized_matmul.py` and `configs.py` (`QuantizedMatmulConfig`)
- `libs/ejkernel/ejkernel/modules/operations/ragged_page_attention_v{2,3}_turboquant.py`

## Mode, Bit-Width, Group-Size & Zeros Contracts

Enforced by `resolve_qparams()` in `libs/ejkernel/ejkernel/quantization/_utils/qparams.py`:

| mode | bits | allowed `group_size` | `scales` dtype | `zeros` |
|---|---|---|---|---|
| `affine` | `1..8` (default `4`) | `{16, 32, 64, 128, 256, 512, 1024}` (default `64`) | float | **required** (same shape as `scales`) |
| `nf4` | `4` (fixed) | `{16, 32, 64, 128, 256, 512, 1024}` (default `64`) | float | must be `None` |
| `mxfp4` / `mxfp8` | `4` / `8` (fixed) | `32` (required) | `uint8` | must be `None` |
| `nvfp4` / `nvfp8` | `4` / `8` (fixed) | `16` (required) | `uint8` | must be `None` |

- Ambiguous shorthands `"mxfp"` and `"nvfp"` are explicitly rejected (`ValueError`).
- For `affine`, `quantize` stores zero-points using `(q - zero) * scale`, and backend wrappers convert `zeros` to
  additive per-group offsets internally before launching low-level Triton/CUDA/CuTe/Pallas kernels.

## Packed Layout Contract (`validate_packed_quantized_matmul_layout`)

Canonical runtime contract checked before fused dispatch (`_utils/qparams.py`):

- **`axis="row"` (`transpose=False`):** `w_q` is `uint32` with shape `(K, ceil(N * bits / 32))`; `scales` (and `zeros`
  for `affine`) have shape `(K, N // group_size)`.
- **`axis="col"` (`transpose=True`):** `w_q` is `uint32` with shape `(N, ceil(K * bits / 32))`; `scales` (and `zeros`
  for `affine`) have shape `(N, K // group_size)`.
- Use `prepack_quantized_weights` / `prepack_quantized_array` to produce matmul-ready layouts, and call
  `validate_packed_quantized_matmul_layout` to catch swapped `axis`/`transpose` or dtype mismatches before dispatch.

## `quantized_matmul` Dispatch & Runtime Controls

`quantized_matmul` ships 6 kernel backends under `libs/ejkernel/ejkernel/kernels/`: `_xla`, `_pallas/tpu`, `_triton`,
`_cuda` (with `libs/ejkernel/csrc/quantized_matmul/`), `_cute`, and `_tilelang`.

- **TPU strategy controls (`modules/operations/quantized_matmul.py`):**
  - `EJKERNEL_QMM_TPU_DEFAULT_STRATEGY`: `"predecode_once"` (default) or `"packed"`.
  - `tpu_path`: `"packed" | "hybrid" | "predecode"` (normalized on `QuantizedMatmulConfig` / call kwargs).
  - `EJKERNEL_QMM_STRICT_FUSED` (`strict_fuse=True`): forbids silent dense fallback; incompatible with
    `allow_dense_fallback=True` or `fuse=False`.
  - `_QMM_TPU_BESTCFG_POLICY` (`use_best_config=True`, `EJKERNEL_QMM_TPU_BESTCFG_SMALL_M` default `1024`) selects
    policy-pinned TPU configs.
- **Apple MPS caveat (`quantized_matmul.py`):** `fuse=True` on `Backend.MPS` (`jax.default_backend() == "mps"`) warns
  and falls back to reference `dequantize + matmul` for stability (or raises `ValueError` when `strict_fuse=True`).
- **Quant/dequant runtime autotuning (`runtime.py`, `_quants/quantizations.py`):**
  - `QuantRuntimeConfig` controls fast-path flags (`enable_u4_u8_fastpath`, `enable_threshold_codebook`,
    `prefer_compute_dtype`, `dequant_unpack_policy`, `minifloat_decode_policy`) without changing numerical semantics;
    `QuantRuntimeConfig.fastest_for_backend()` returns TPU (`bf16`) vs GPU/MPS/CPU (`fp16`) throughput defaults.
  - Controlled by `EJKERNEL_QRUNTIME_AUTOTUNE` (defaults to `"1"`), plus
    `EJKERNEL_QRUNTIME_AUTOTUNE_{WARMUP,ITERS,MIN_GAIN,MAX_CANDIDATES,CONFIRM_WARMUP,CONFIRM_ITERS}`. Inspect or reset
    with `runtime_autotune_cache_sizes()` and `clear_runtime_autotune_cache()`.

## TurboQuant KV-Cache Compression

`libs/ejkernel/ejkernel/quantization/turboquant/` implements two-stage KV-cache compression:

1. **Stage 1 (MSE-optimal):** random orthogonal rotation (`generate_rotation_matrix`) + Lloyd-Max scalar quantization
   (`solve_lloyd_max` / `LloydMaxCodebook`), bit-packed with `pack_4bit` / `unpack_4bit`.
2. **Stage 2 (QJL residual correction on keys):** random projection (`generate_projection_matrix`) + 1-bit sign packing
   (`pack_signs` / `unpack_signs`) for unbiased inner products.

Core ops (`turboquant/ops.py`): `turboquant_compress_keys`, `turboquant_compress_values`,
`turboquant_dequantize_values`, `turboquant_asymmetric_scores`. Wired into
`libs/ejkernel/ejkernel/modules/operations/ragged_page_attention_v{2,3}_turboquant.py` and consumed in EasyDeL by
`TurboQuantRaggedPagesCacheView` (`libs/easydel/easydel/caching/turboquant_ragged_page/cache.py`) and
`RaggedPageAttnV2`/`V3` (`libs/easydel/easydel/operations/kernels/ragged_page_attention.py`).

## Routing

- EasyDeL quantized linears & TP-portable fused layouts: `.agents/skills/quantization-layout/SKILL.md`
- SpectraX quantization layers/QAT: `.agents/skills/spectrax-quantization/SKILL.md`
- EasyDeL KV/TurboQuant cache integration: `.agents/skills/debug-easydel-cache/SKILL.md`
- New backend kernel or optimization: `.agents/skills/add-ejkernel-kernel/SKILL.md` /
  `.agents/skills/optimize-ejkernel-kernel/SKILL.md`

## Verification

All quantization, `quantized_matmul`, and TurboQuant tests execute JAX numerics and MUST run on TPU or GPU; a CPU run
is not validation. Without an accelerator, report the change as **unverified on hardware**.

```bash
# TPU: container/runtime controls, QMM ops, strict-fuse parity, TurboQuant, and TPU Pallas/XLA kernels
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest \
    libs/ejkernel/test/modules/operations/test_quantized_runtime_controls.py \
    libs/ejkernel/test/modules/operations/test_quantized_matmul.py \
    libs/ejkernel/test/modules/operations/test_quantized_matmul_pallas.py \
    libs/ejkernel/test/modules/operations/test_quantized_matmul_migration.py \
    libs/ejkernel/test/modules/operations/test_qmm_strict_fuse_tpu_parity.py \
    libs/ejkernel/test/modules/operations/test_ragged_page_attention_v2_turboquant.py \
    libs/ejkernel/test/modules/operations/test_ragged_page_attention_v3_turboquant.py \
    libs/ejkernel/test/kernels/_xla/test_channelwise_quantized_matmul.py \
    libs/ejkernel/test/kernels/_xla/test_integer_quantization_lowering.py \
    libs/ejkernel/test/kernels/_xla/test_quantized_matmul_fallback.py \
    libs/ejkernel/test/kernels/_xla/test_ragged_page_attention_v2_turboquant.py \
    libs/ejkernel/test/kernels/_xla/test_ragged_page_attention_v3_turboquant.py \
    libs/ejkernel/test/kernels/_pallas/tpu/test_quantized_matmul.py

# GPU: runtime controls, QMM ops, CuTe/strict-fuse parity, and GPU backend kernels
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest \
    libs/ejkernel/test/modules/operations/test_quantized_runtime_controls.py \
    libs/ejkernel/test/modules/operations/test_quantized_matmul.py \
    libs/ejkernel/test/modules/operations/test_quantized_matmul_cute.py \
    libs/ejkernel/test/modules/operations/test_qmm_strict_fuse_gpu_parity.py \
    libs/ejkernel/test/kernels/_triton/test_quantized_matmul.py \
    libs/ejkernel/test/kernels/_cuda/test_quantized_matmul.py \
    libs/ejkernel/test/kernels/_cute/test_quantized_matmul.py \
    libs/ejkernel/test/kernels/_tilelang/test_quantized_matmul.py
```
