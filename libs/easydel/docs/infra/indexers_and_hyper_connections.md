# Modular indexers and hyper-connections

Sparse selection and residual-stream mixing are reusable layers, rather than
numerical implementations embedded in model attention/decoder classes. Model
adapters translate family configurations and retain checkpoint-native parameter
paths. They do not wrap a second parameter-owning module below an extra prefix.

## Indexer contract

`easydel.layers.indexer` exposes `BaseIndexer`, `SelectionSpec`, and
`IndexerSelection`.

- `SelectionSpec` describes the static candidate and output units, and whether
  selection is per group. A compressed-entry offset is **not** a token offset.
- `BaseIndexer.select_candidates(scores, k, valid)` ranks arbitrary leading axes,
  applies validity before ranking, and pads invalid choices with `-1`. Positive
  infinity is supported for local blocks promoted inside a top-k budget.
- `IndexerSelection` is an array-only JAX pytree with `indices` and optional
  `score_proxy`. `to_mask(size)` scatters valid offsets without materializing a
  selection-by-context one-hot tensor. Negative/out-of-range IDs select nothing.
  `to_bias(size)` produces additive bias; an optional full-domain score proxy
  contributes a zero-valued straight-through gradient only at selected entries.
- Cache ownership stays explicit. A strategy returns logical offsets; the cache
  adapter maps them to physical pages and owns request boundaries and packed
  state updates.

The common contract does not force different algorithms to share incorrect math:

| Family | Shared strategy | Ranked candidates → attention offsets |
| --- | --- | --- |
| Qwen4-Exp | `BlockTopKIndexer`: mean-pool keys, then score | blocks → tokens, with visible tail |
| GLM-5-Next | `SparseIndexer` with pool config: learned gated pooling | pools → tokens, with configured tail |
| MiniMax-M3-VL | `BlockMaxIndexer`: max-pool token scores per head/group | blocks → tokens; local blocks consume budget |
| GLM-MoE-DSA | `TokenIndexer`: dedicated operation dispatch | tokens → tokens |
| DeepSeek-V4 | `CompressedIndexer` with explicit compressor/cache adapter | compressed entries → compressed entries |

Each strategy owns its reusable numerical path. Family adapters provide norms,
rotary conventions, projection/compressor configuration, and cache integration
where these differ. The legacy `easydel.layers.sparse_attention.BlockTopKIndexer`
import is retained as a re-export, not a duplicate implementation.

For example, attention can consume an already-computed selection as follows:

```python
from easydel.layers.indexer import BaseIndexer, IndexerSelection

selection = BaseIndexer.select_candidates(scores, k=16, valid=visible)
mask = selection.to_mask(context_capacity)
# Insert the attention-head axis for ungrouped [B,Q,KV] selections.
attention_bias = selection.to_bias(context_capacity)[:, None, :, :]
```

Use the capacity of the selection's **output domain**. Do not expand DeepSeek
compressed-entry selections as if they were pooled token blocks. Grouped results
already have a group/head dimension and must not be collapsed across it.

Existing output tuples are preserved where callers also need cache state.
`IndexerOutput.selection` exposes the common view without changing its tuple
arity. Legacy `indices_to_bool_mask` clipping behavior remains unchanged; the new
selection contract deliberately rejects out-of-range offsets instead.

### Exact top-k on TPU

On TPU, `jax.lax.top_k` lowers to a full sort of every score row. Every strategy
therefore ranks through `top_k_indices` in `_selection.py`, and the GLM-MoE-DSA
indexer operation uses the same approach. These call ejkernel's registered `topk`
operation. Its Pallas kernel finds the exact `k`-th key by in-VMEM bisection,
compacts the survivors, and sorts only those. The indices are **bit-identical**
to `jax.lax.top_k`, including ties (lowest index first), signed zeros,
infinities and NaN payloads, because the kernel uses the same IEEE totalOrder.
The operation keeps XLA where XLA measured faster: `k < 32`, fewer than 768
candidates, or fewer than 256 rows. There are no parameters, so checkpoint
leaves are unchanged.

A Pallas call is not SPMD-partitionable, so rows are split explicitly. They
follow the activation `[batch, query]` layout resolved from the indexer's
`mesh_source` (model adapters pass their config). Candidates stay whole, so no
score rows are gathered. Without a mesh source, multi-device selection keeps
`jax.lax.top_k`.

The kernel's output merge writes each compacted group into the `k`-wide result.
For `k` up to 1024 it sweeps the whole result with masked stores; above that it
visits only the 128-lane blocks the group lands in, which is 8-13% faster at
k=2048 and identical at k<=512.

Measured on 4x TPU v5 (median):

| workload | `lax.top_k` | threshold kernel |
| --- | --- | --- |
| `[8192, 16384]`, k=2048 | 38.03 ms | 9.74 ms |
| `[4096, 32768]`, k=2048 | 45.54 ms | 8.37 ms |
| `[4096, 65536]`, k=2048 | 108.89 ms | 19.36 ms |
| `[8192, 8192]`, k=512 | 15.19 ms | 3.74 ms |
| `[8192, 16384]`, k=64 | 25.11 ms | 6.09 ms |

Model effect depends on how much selection the layer does. DeepSeek-V4's
compressed indexer (k=512 over 2048 entries, fsdp=4) is 0.8% faster forward and
0.4% faster fwd+bwd. GLM-5-Next's pooled selection (k=128 over 512 pools) stays
on XLA, where it is faster. All results were bit-identical.

Against `lax.top_k` in GLM-MoE-DSA, compiled temporary memory falls on the
fsdp, tp and sp meshes but rises from 4873 to 5921 MiB on the dp=4 mesh. That
is XLA scheduling around the custom call, not a new buffer: the sort outputs it
replaces were larger.

### Scatter-free selection masks

Sparse attention consumes a selection as a dense `[batch, query, kv]` boolean
mask. Building it from the selected indices is the expensive part on TPU: a
`one_hot(...).any(-2)` materializes a `[batch, query, k, kv]` intermediate
(76 ms at 8192 queries, k=2048), a scatter is no better (87 ms), and even the
element gather that fetches `k` values per row costs as much as a scatter.

`topk_selection_mask(scores, values, indices)` in `easydel.layers.indexer`
derives the mask by comparison instead. A score is selected when its IEEE
totalOrder key is above the `k`-th value's key, or equal to it at a position no
later than the last selected tie. Because top-k breaks ties by lowest index,
this equals the one-hot of the indices exactly, including NaN payloads and
signed zeros. The values come from the top-k call itself
(`top_k_values_indices`), never from a gather.

`select_candidates` attaches this mask to the returned `IndexerSelection` as
`mask`, and `IndexerSelection.to_mask` / `to_bias` return it when its width
matches the requested domain. Otherwise they scatter the indices as before, so
callers that build their own selection are unaffected. Block and pool
strategies expand the candidate-level mask to tokens by repeating each block
over its members and shifting by the first visible key; the incomplete tail is
a range compare. Masks are saved with the `indexer_topk` checkpoint name.

Measured on 4x TPU v5, 2 layers, 8192 tokens, batch 4, bf16 (fwd / fwd+bwd):

| model | mesh | before | after |
| --- | --- | --- | --- |
| GLM-MoE-DSA | dp=4 | 183.1 / 197.6 ms | 32.1 / 46.5 ms |
| GLM-MoE-DSA | fsdp=4 | 184.0 / 198.6 ms | 33.0 / 47.5 ms |
| GLM-MoE-DSA | tp=4 | 186.8 / 193.6 ms | 35.8 / 42.5 ms |
| GLM-MoE-DSA | sp=4 | 70.9 / 101.2 ms | 32.6 / 62.4 ms |
| GLM-5-Next | dp=4 | 284.6 / 321.5 ms | 44.0 / 80.9 ms |
| GLM-5-Next | fsdp=4 | 298.5 / 335.6 ms | 58.4 / 95.6 ms |
| GLM-5-Next | tp=4 | 1190.0 / 1217.1 ms | 255.2 / 282.5 ms |
| GLM-5-Next | sp=4 | 862.5 / 979.9 ms | 59.3 / 173.6 ms |
| DeepSeek-V4 | dp=4 | 146.1 / 363.4 ms | 51.4 / 269.1 ms |
| DeepSeek-V4 | fsdp=4 | 155.9 / 387.0 ms | 61.1 / 292.2 ms |
| DeepSeek-V4 | tp=4 | 654.4 / 763.5 ms | 85.5 / 187.8 ms |
| DeepSeek-V4 | sp=4 | 309.5 / 520.2 ms | 75.4 / 286.1 ms |

Losses are bit-identical everywhere, and gradients are bit-identical for
GLM-MoE-DSA and GLM-5-Next on every mesh and for DeepSeek-V4 on dp and fsdp.
On DeepSeek-V4 tp and sp, 3 of 73 gradient leaves (indexer projections, which
train through the straight-through score proxy) differ by rel-L2 at most
4.4e-5: XLA fuses that backward differently. The same model repeated with the
old code is bit-identical run to run. Compiled temporary memory moves by at
most 2% (it falls on tp and sp). Collectives are unchanged except on sp.

Qwen4-Exp's `BlockTopKIndexer` also spread its block scores to tokens (the
training score proxy) with an element gather over `[query, kv]`, whose reverse
is a scatter-add. It now uses the same repeat-and-shift as its mask. One
indexer, 8 heads, budget 2048, batch 1, bf16, fwd / fwd+bwd:

| tokens | before | after |
| --- | --- | --- |
| 4096 | 157.3 / 179.4 ms | 1.08 / 1.83 ms |
| 8192 | 2019 / 1880 ms | 4.08 / 7.02 ms |

The mask is identical. Gradients differ at rel-L2 up to 4e-4 in bf16 (2.6e-5 in
float32), because each block now sums its tokens' cotangents with a reduction
instead of a scatter-add. The spread's own VJP matches an exact float64
reference.

`BlockTopKIndexer.project` splits the fused query/key projection on the head
axis. Slicing the lane axis before the float32 query norm made XLA:TPU abort
compilation (no error message, jax 0.11.2 and libtpu 0.0.48) for bf16 inputs at
4096 tokens or more. That is reproducible in plain JAX, so it is a compiler bug
and not an EasyDeL one; the head-axis split computes the same values.

## Manifold hyper-connections (mHC)

`easydel.layers.residual` provides:

- `ManifoldHyperConnectionConfig`: hidden size, stream count, normalization and
  Sinkhorn epsilon, iteration count, and initializer scale.
- `ManifoldHyperConnection`: checkpoint-native `fn`, `base`, and `scale`
  parameters; fp32 RMS normalization, read/write gates, and Sinkhorn projection.
- `manifold_residual_write`: residual update with the **transposed** mixer and
  family-compatible low-precision arithmetic.
- `ManifoldHyperHead`: DeepSeek's learned sigmoid-weighted final collapse.
- `MeanHyperHead` / `mean_hyper_head`: GLM's parameter-free mean collapse.

```python
from easydel.layers.residual import (
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    manifold_residual_write,
)

connection = ManifoldHyperConnection(
    ManifoldHyperConnectionConfig(hidden_size=hidden_size, hc_mult=4),
    rngs=rngs,
    mesh_source=model_config,
)
post, comb, collapsed = connection(streams)  # streams: [B,S,hc,D]
sublayer_output = sublayer(collapsed)
streams = manifold_residual_write(streams, sublayer_output, post, comb)
```

A mesh source is resolved dynamically during forward, so stage-local model mesh
resolution is not frozen at construction. `HyperStreamSharding` describes the
four logical axes `[BATCH, QUERY_LENGTH, EMPTY, EMBED]`: streams stay local and
hidden features, not the small stream count, receive the hidden-axis partition.
GLM-5 uses this layout at stream initialization and decoder outputs.

When the mesh source provides `runtime_sharding_resolver`, the shared projection
flattens only **local** stream features and correspondingly sliced weights. It
reduces per-token RMS statistics and `(hc + 2) * hc` projection logits across
feature partitions (25 fp32 values per token for `hc=4`, counting the RMS statistic).
It does not need to gather the full `[B,S,hc,D]` activation. The checkpoint's
`fn` remains a two-dimensional `(hc * (hc + 2), hc * D)` parameter.

The multi-device TPU Sinkhorn path retains token partitions inside its explicit
same-mesh `shard_map`; only the small matrix dimensions are replicated. Bare
meshes without a resolver retain the unpartitioned fallback. CPU tests cannot
validate this TPU lowering path. As with other changes to distributed reduction
order, this layout is not a promise of bitwise-equal bf16 results.

### Opt-in fused coefficients

`ManifoldHyperConnectionConfig(use_fused_coefficients=True)` routes the gates,
softmax, and Sinkhorn projection through ejkernel's registered
`mhc_coefficients` operation. It is off by default; GLM-5-Next and DeepSeek-V4
expose it as `hc_use_fused_coefficients`. Parameters and checkpoint leaves are
unchanged, so the flag can be toggled on an existing checkpoint.

On TPU with four streams and at most 20 Sinkhorn iterations, the operation runs
a lane-packed Pallas kernel. Its backward pass saves only the small projection
logits and recomputes the normalization tape in VMEM. Other stream counts,
iteration counts above 20, and empty inputs fall back to the XLA
implementation.

**Differentiation limit:** the Pallas path supports **first-order reverse-mode
AD only** (`jax.grad`/`jax.vjp`). Leave the flag off where forward-mode
(`jax.jvp`) or higher-order derivatives are needed. Calling the operation
directly with `platform="xla"` supports both.

The multi-device path uses the same token-local `shard_map` as the unfused
Sinkhorn. The kernel declares that its outputs vary over exactly its inputs'
manual axes, so replicated (feature-parallel) axes get no extra backward
all-reduce. Compiled collective counts match the unfused path (the
(1, 2, 2) backward has two fewer all-gathers).

Measured on 4x TPU v5 with identical parameter state (median steady state;
standalone layer `[4, 2048, 4, 4096]`, default precision):

| mesh (dp, sp, tp) | dtype | forward ms (off → on) | fwd+bwd ms (off → on) |
| --- | --- | --- | --- |
| (1, 1, 4) | f32 | 0.873 → 0.364 | 1.909 → 1.286 |
| (1, 1, 4) | bf16 | 0.828 → 0.372 | 1.697 → 1.070 |
| (2, 1, 2) | f32 | 0.575 → 0.367 | 1.564 → 1.215 |
| (2, 1, 2) | bf16 | 0.530 → 0.363 | 1.344 → 0.996 |
| (1, 2, 2) | f32 | 0.575 → 0.382 | 1.566 → 1.215 |
| (1, 2, 2) | bf16 | 0.530 → 0.376 | 1.344 → 0.996 |

In a two-layer GLM-5-Next (hidden 2048, 4x1024 tokens), whole-model time
improves by 0.25–3.5% forward and 0.4–1.7% fwd+bwd. Backward temporary
memory falls by 1–5% on the (1,1,4) and (2,1,2) meshes but rises by about 1% on
(1,2,2). f32 forward-only temporary memory also rises (up to +21%) from the
padded, packed kernel buffers. At a toy size (hidden 256, 256 tokens) the
layer is dispatch-bound and model timings move by ±3% either way. Without
discrete selections (no MoE routing, no DSA indexer), f32 `HIGHEST` model
outputs match the unfused path to about 4e-7 relative L2 (gradients to about
2e-6). With MoE top-k or indexer top-k enabled, those ulp-level differences can
flip near-tied selections. Against an f32 `HIGHEST` reference, bf16 errors for
fused and unfused are indistinguishable.

`GatedResidual` remains a separate residual family: Qwen's element-wise gated
read/write is not a Sinkhorn stream mixer. The two must not share parameterization
or silently substitute for one another.

## Matmul precision on TPU

Activation dtype and contraction precision are separate settings. Float32
operands and outputs do not guarantee full-fp32 multiplication on TPU.
`precision=None` preserves JAX's ambient policy; use the layer's
`precision=jax.lax.Precision.HIGHEST` argument when full-fp32 reference parity
is required. This is an explicit accuracy/performance choice, not a global
precision override. Near-tied indexer candidates can change ordering between
precision policies.

`BlockTopKIndexer` applies its precision setting to projection and both dense
and paged scoring. `TokenIndexer` forwards its setting through the registered
DSA operation to both query-key scoring and head-weight reduction. The
compressed indexer and manifold layers likewise honor their precision arguments.
Strict NumPy-reference tests request matching precision locally; checkpoint and
default-policy tests retain the default behavior.

## Scope and capability boundaries

This extraction does not add missing serving/training capabilities. MiniMax's
cached sparse decoding and Qwen's packed-document selection/cached training
remain unsupported where their existing checks reject them. GLM-5's ragged MLA
path still cannot consume the dense per-query top-k mask, and existing shared
indexer schedule behavior is unchanged. The subsequent mHC layout optimization
changes GLM-5's rank-four stream constraints, not attention or expert placement.
