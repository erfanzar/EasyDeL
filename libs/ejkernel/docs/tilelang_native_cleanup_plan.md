<!--
Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# TileLang Native Cleanup Plan

This document is the working plan for making the TileLang backend honest and complete.
The goal is not "registered on `Platform.TILELANG`"; the goal is native TileLang kernels
that do the algorithm's real work inside `@T.prim_func` CUDA kernels.

## Non-Negotiable Rules

1. TileLang implementations must not silently ignore result-changing arguments.
   If an argument changes semantics and is unsupported, raise `EjkernelRuntimeError`.
2. No `del <arg>` for public kernel arguments. Either honor the argument or raise.
3. `jnp` is allowed only for cheap shape/layout plumbing:
   reshape, transpose, dtype validation, small dummy buffers for FFI signatures.
4. `jnp` is not allowed for real algorithm work:
   matmul, reduction, scan, softmax, mask construction, gather/scatter, page lookup,
   dequantization, dropout/RNG, top-k, prefix sums, gate math, scale/bias application,
   or dense fallback computation.
5. If the XLA algorithm is differentiable/training-facing, the TileLang backend needs
   a native backward path through `jax.custom_vjp` and TileLang backward kernels.
   Inference-only decode/page-attention kernels can remain backward-free.
6. "Feature gated" means an explicit error, not a dense fallback, not a dropped arg,
   and not a JAX implementation hidden around a TileLang core.
7. A kernel is not marked complete until:
   native forward exists, required native backward exists, every argument is honored
   or explicitly rejected, parity tests pass against XLA, and ruff passes.

## Audit Flags

- `D`: deletes, ignores, or no-ops a result-changing argument.
- `J`: performs real algorithm work in JAX instead of TileLang.
- `B`: missing required native backward, or backward is only partial.
- `Partial`: unsupported features raise cleanly, but the implemented subset is smaller
  than the XLA surface.

## Current Audit Table

| Algorithm | Flags | Status / Required Fix |
|---|---:|---|
| `flash_attention` | clean | Feature construction, dropout, GQA, masks, sinks, fwd/bwd are native after latest cleanup. Keep parity coverage broad. |
| `scaled_dot_product_attention` | clean | Thin wrapper over native FlashAttention; ragged packed paths raise. |
| `ring_attention` | clean | Single-device wrapper over native FlashAttention; multidevice/mask callback raises. |
| `deepseek_attn` | clean | Indexer, top-k bias, KV projection, RoPE score-tail packing, value-dim padding/cropping and attention fwd/bwd are native; top-k path is stop-gradient by design. |
| `attention` | clean | Output uses native FlashAttention; returned dense weights and gradients through weights to Q/K use native TileLang kernels. Score metadata follows the FlashAttention contract and is nondifferentiable. |
| `blocksparse_attention` | `Partial` | Current explicit-mask path uses native FA. It is not a selected sparse-block kernel. Implement real block-sparse traversal or keep opaque sparse layouts rejected. |
| `page_attention` | clean | Native forward now performs page-table lookup, context masking, GQA head mapping, block-first/head-first cache handling, sliding window and soft-cap inside TileLang. Inference-only, so no backward required. |
| `prefill_page_attention` | clean | Native forward now performs page-index lookup, causal/context/window masks, soft-cap and GQA inside TileLang. Inference-only, so no backward required. |
| `ragged_page_attention_v2` | clean | Native forward now performs interleaved K/V page lookup, query-start mapping, context/causal/window masks, soft-cap, sinks and GQA inside TileLang. Inference-only, so no backward required. |
| `ragged_page_attention_v3` | Clean | Native forward updates packed KV cache, reads page tables, handles live K/V, causal/window masks, soft-cap, q/k/v scales and sink logits inside TileLang. Inference-only, so no backward required. |
| `ragged_page_attention_v2_turboquant` | Clean | Native forward now performs packed index/sign unpack, codebook lookup, QJL correction, page lookup, causal/window masks, soft-cap and sink logits inside TileLang. Inference-only, so no backward required. |
| `ragged_page_attention_v3_turboquant` | Clean | Native update kernel compresses raw K/V into packed TurboQuant pages, then native v2 TurboQuant attention reads the updated pages. Inference-only, so no backward required. |
| `chunked_prefill_paged_decode` | Clean | Native fused cache-update + paged causal attention kernel covers page lookup, GQA, ALiBi, softmax sink, sliding window, softcap, and cache outputs. Inference-only, so no backward required. |
| `unified_attention` | Clean | Native forward now performs page-table traversal, ragged query ranges, GQA, ALiBi, query-query bias, sinks, sliding window and soft-cap inside TileLang. Inference-only, so no backward required. |
| `multi_latent_ragged_page_attention` | Clean | Native forward updates packed latent KV cache, reads flat page tables, combines no-PE/PE logits, applies q/k/v scales, sliding window and soft-cap inside TileLang. Inference-only, so no backward required. |
| `multi_latent_ragged_page_attention_v2` | Clean | Shares the native v1 MLA kernel and normalizes v2 tuple block hints before compilation. Inference-only, so no backward required. |
| `flash_mla` | `Partial` | Native projection GEMMs plus native pack/pad/crop kernels support no-RoPE, `b_k`, `b_q/b_k`, GQA, FA score features and `v_head_dim != qk_head_dim` fwd/bwd. Custom non-float32 `softmax_dtype` remains unsupported. |
| `native_sparse_attention` | `Partial` | Precomputed `block_indices` path now calls the native selected-block sparse kernel and supports native `g_slc` gating. Add native compressed-block attention, top-k block selection, `g_cmp` and `cu_seqlens`. |
| `apply_native_sparse_attention` | `Partial` | Selected-block sparse forward/backward are native for per-token and per-query-block metadata. Add `token_indices` and `cu_seqlens` support. |
| `rwkv4` | Clean | Forward, backward, decay transform and default state initialization are native TileLang. |
| `rwkv6` | Clean | Padded and packed `cu_seqlens` forward/backward are native, including reverse, scale, default state initialization and initial-state gradients. |
| `rwkv7` | Clean | Padded and packed `cu_seqlens` forward/backward are native, including reverse, scale, default state initialization and initial-state gradients. |
| `rwkv7_mul` | Clean | Padded and packed `cu_seqlens` forward/backward are native, including the `kk * a`, `-kk` transform, default state initialization and initial-state gradients. |
| `ragged_gated_delta_rule` | Clean | Decode and ragged prefill forward/backward are native, including beta, decay, q/k l2norm, state-slot passthrough gradients and packed request ranges. `chunk_size` affects only schedule in the XLA path, not numerics. |
| `gated_delta_rule` | Clean | Native recurrent beta/decay/l2norm forward and backward. `use_chunked`/`chunk_size` affect only schedule, not numerics, so the TileLang path validates them and computes the exact recurrence. |
| `kernel_delta_attention` | Clean | Shares the native GDR recurrence with KDA scaling, beta/decay/l2norm and backward. |
| `kda` | Clean | Alias shares the native KDA path. |
| `gla` | Clean | Key-side gate decay, GQA head mapping, per-head/per-sequence gamma decay, reverse traversal, default state initialization, packed `cu_seqlens` and backward are native through the recurrent gated kernel. |
| `lightning_attn` | Clean | Layer-dependent per-head decay, GQA head mapping, reverse traversal, default state initialization, packed `cu_seqlens` and backward are native through the recurrent gamma kernel. |
| `recurrent` | Clean | Ungated, key-side gated, split-gate, GQA and gamma-decayed padded/packed forward/backward are native, including reverse traversal, default state initialization and initial-state gradients. |
| `state_space_v1`, `ssm1`, `mamba1` | `Partial` | Core scan/bwd, reductions, default-state init, initial-state gradient and silu output gating are native. Arbitrary `act_fn` callables are rejected; add native variants only for concrete activations. |
| `state_space_v2`, `ssm2`, `mamba2` | `Partial` | Grouped B/C, grouped gradients, default-state init, initial-state gradient, silu gate and gated RMSNorm with custom `rmsnorm_eps` are native. Custom precision remains unsupported. |
| `mean_pooling` | Clean | Padded and packed `cu_seqlens` fwd/bwd are native TileLang kernels. Packed forward intentionally matches the XLA reference's clamped dynamic-slice behavior. |
| `quantized_matmul` | `Partial` | Packed affine weights are native for 1/2/4/8-bit row and col layouts, including in-kernel unpack/dequant, zeros/scales, fwd, `dx`, `dscales` and `dzeros`. Legacy symmetric int8 remains supported. Add NF4/MXFP/NVFP modes and split-k, or keep them explicitly rejected. |
| `grouped_matmul` | `Partial` | Forward/backward for lhs/rhs/existing output are native, including group starts, `group_offset` and transpose handling. Add LUT tiling and custom precision support. |
| `grouped_matmulv2` | `Partial` | Same native forward/backward as grouped matmul v1, including `group_offset`. Add LUT tiling and custom precision support. |
| `grouped_matmulv3` | `Partial` | Forward/backward for lhs/rhs/existing output/rhs scale/rhs bias are native, including group starts, `group_offset` and transpose handling. Add LUT tiling and custom precision support. |
| `all_gather_matmul` | `Partial` | Single-device matmul forward/backward are native. Real collectives now raise explicit errors instead of using JAX collectives; add native collective integration for `tp_size > 1`. |
| `reduce_scatter_matmul` | `Partial` | Single-device matmul forward/backward are native. Real collectives now raise explicit errors instead of using JAX collectives; add native collective integration for `tp_size > 1`. |
| `decode_attention` | Clean | Native forward uses `req_to_tokens`, `seq_lens`, `page_size`, GQA and soft-cap inside TileLang and emits LSE inside the kernel. Inference-only, so no backward required. |
| `ragged_decode_attention` | Clean | Native forward handles sequence ranges, GQA, sliding windows, soft-cap and sink logits inside TileLang. Inference-only, so no backward required. |

## Performance Benchmark Status

CUDA, JAX and TileLang are working on the local H100 device. The reusable suite runner
is `benchmarks/benchmark_tilelang_suite.py`; it writes JSON and Markdown reports under
`benchmark_results/`.

The first smoke report was
`benchmark_results/tilelang_suite_20260521T010158Z.md`, run with one config per op,
one warmup and five timed iterations. The latest smoke report after the performance
pass is `benchmark_results/tilelang_suite_20260521T014816Z.md`.

Fixes completed during the performance pass:

- `ragged_page_attention_v3`: fixed the illegal shared-memory launch, then added whole-tile
  skipping for padded page-table capacity and causal/window bounds. The first included run
  was about `0.057x` of XLA; the targeted report
  `benchmark_results/tilelang_suite_20260521T011555Z.md` moved it to `1.204x`, and the
  latest full smoke reports `1.256x`.
- `native_sparse_attention`: replaced serial per-score dot products with parallel
  reductions and added a block-size-specific schedule. Targeted 20-iteration report
  `benchmark_results/tilelang_suite_20260521T013152Z.md` measured `1.391x`.
- `apply_native_sparse_attention`: added the small-block schedule and 32-thread launch.
  Targeted 20-iteration report `benchmark_results/tilelang_suite_20260521T013152Z.md`
  measured `1.278x`.
- `decode_attention`: added small-page `block_k` selection and whole-tile skipping. It moved
  from a clear loss to roughly tied on the first shape; the 20-iteration targeted report
  `benchmark_results/tilelang_suite_20260521T013523Z.md` measured `0.994x` by mean with
  TileLang faster by median.
- `mean_pooling`: moved the final output cast into the TileLang forward kernel so the FFI
  output dtype matches XLA. This reduced overhead but the tiny first shape remains close
  rather than clearly won.

Remaining measured performance targets:

- `grouped_matmul` and `grouped_matmulv2`: small first-shape latency still favors XLA.
  Quick tile-size experiments did not help and were reverted. A real fix needs a grouped
  schedule that avoids computing rows outside each group without falling off efficient MMA
  layouts.
- `prefill_page_attention`, `ragged_decode_attention`, `ring_attention` and
  `gated_delta_rule`: first-shape results are close/noisy or launch-overhead dominated.
  Run multi-shape sweeps before changing kernel structure.
- `lightning_attention`: TileLang runs, but the XLA benchmark path reports `inf` in the
  suite, so it is not currently comparable in the summary table.

## Priority Plan

### Phase 0: Enforce Honesty

- Replace every `del <public_arg>` in `_tilelang` with either real handling or
  an explicit `EjkernelRuntimeError`.
- Add a static test that fails on `del <arg>` in TileLang public interfaces.
- Add a static test that fails on banned JAX operations in `_tilelang` implementation
  files, with an allowlist for reshape, transpose, dtype checks and tiny FFI buffers.
- Update misleading docstrings that claim native support where the implementation is
  a wrapper, subset, dense fallback or JAX orchestration.

### Phase 1: Paged / Ragged Attention Family

- Build a shared native TileLang paged-attention core:
  page-table lookup, context length masking, query-start mapping, causal/window masks,
  soft-cap, sinks and optional split-K combine all inside TileLang.
- Use that core to replace:
  `page_attention`, `prefill_page_attention`, `ragged_page_attention_v2`,
  `ragged_page_attention_v3`, `chunked_prefill_paged_decode`,
  `unified_attention`, and `ragged_decode_attention`.
- Add native dequant/page-gather variants for:
  `ragged_page_attention_v2_turboquant` and `ragged_page_attention_v3_turboquant`.
- Add parity tests for contiguous, non-contiguous page tables, variable context lengths,
  sliding windows, soft caps and softmax aux where the XLA reference supports them.

### Phase 2: MLA / Native Sparse Attention

- Keep the shared MLA pack/pad/crop path covered for `flash_mla` and DeepSeek;
  add custom non-float32 `softmax_dtype` only if it becomes a real requirement.
- Implement native latent paged attention kernels for
  `multi_latent_ragged_page_attention` and `_v2`.
- Implement real native sparse attention selected-block/compressed-block traversal.
- Delete dense-fallback behavior for NSA once selected-block kernels exist.

### Phase 3: Recurrent / RWKV / Delta Kernels

- Keep native backward coverage for recurrent, RWKV and delta kernels on every
  supported differentiable option.

### Phase 4: State Space Kernels

- Add native activation variants beyond silu only when the callable has a concrete
  TileLang implementation.
- Decide whether custom precision is a real SSM2 requirement or should remain an
  explicit unsupported option.

### Phase 5: Matmul / Quant / Collectives

- Implement full native grouped matmul variants with group offsets, rhs scale/bias,
  existing output, preferred dtype and native backward.
- Expand `quantized_matmul` beyond packed affine into NF4/MXFP/NVFP modes and
  split-k, or keep unsupported modes explicitly rejected before any work is done.
- Decide the TileLang backend contract for collectives:
  either native collective integration, explicit single-device-only errors, or a
  separately documented JAX-collective wrapper that is not advertised as fully native.

## Test Plan

For each algorithm moved to complete:

- Add forward parity against XLA with fp16 tolerance.
- Add backward parity for every differentiable input unless the algorithm is explicitly
  inference-only.
- Add feature parity tests for every supported argument.
- Add unsupported-argument tests that assert `EjkernelRuntimeError`, not silent deletion.
- Run:

```bash
EJKERNEL_TILELANG_AUTOTUNE=0 .venv/bin/python -m pytest test/test_tilelang_parity.py -q
.venv/bin/python -m ruff check <changed files>
```

## Immediate Next Work

1. Design the grouped matmul small-group schedule so it avoids rows outside the active
   group while preserving efficient MMA layouts.
2. Run multi-shape sweeps on the small regressions before rewriting them; separate launch
   overhead from real algorithmic loss.
3. Add native compressed-block attention, top-k selection and gating for
   `native_sparse_attention`.
4. Add LUT/custom-precision support for the grouped matmul variants, or keep
   unsupported modes explicitly rejected.
5. Expand `quantized_matmul` to NF4/MXFP/NVFP, and continue narrowing or
   implementing attention-weights VJP and the real multi-device collectives.
