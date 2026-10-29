# dsa_sparse_attention: pool-bitmap block-sparse DSA attention (Pallas TPU)

Attention of absorbed-MLA queries `[B, H, S, D]` over a shared `K == V` latent `[B, S, D]`, restricted to a DSA
indexer selection given as a per-query pool mask `[B, S, P]` (pool `p` = keys `p*kpool .. p*kpool + kpool - 1`,
anchored at key 0) plus up to 8 always-selected tail keys per query. No `O(S^2)` tensor is built anywhere.

## Layout

| piece | file |
| --- | --- |
| block tables, modes, DMA index clamp | `kernels/_xla/dsa_sparse_attention/_tables.py` |
| dense XLA reference (registered XLA / ANY, priority 0) | `kernels/_xla/dsa_sparse_attention/_xla_impl.py`, `_interface.py` |
| sparse forward | `_pallas_impl_fwd.py` |
| two-pass backward (dQ + pool cotangent, dKV) | `_pallas_impl_bwd.py` |
| fused single-pass backward | `_pallas_impl_bwd_fused.py` |
| `custom_vjp`, padding, registration (PALLAS / TPU, priority 10) | `_interface.py` |
| public operation + config | `modules/operations/dsa_sparse_attention.py`, `DsaSparseAttentionConfig` |

## Tables (`build_tables`)

* `bits_t [B, W_pad, S]` int32: bit `p % 32` of word `p // 32` set iff pool `p` is selected (words on sublanes,
  queries on lanes: a kernel reads one `[1, block_q]` row per word, no mask transpose).
* `tails_t [B, 8, S]` int32: tail keys, `-1` = none.
* `count [B, nQB]`, `kv_list [B, nQB, nKB]`: per q-block, the active kv-blocks (some query selected a key in it or
  holds a tail key in it, and the block is not entirely above the diagonal), ascending, padded with `nKB`.
  Scalar-prefetched into SMEM. `build_q_tables` transposes them for the kv-block-major dKV pass.

Visit `mode` (tables only; no kernel code depends on it): `"sparse"` = the active pairs; `"dense"` = every causal
pair (constant work, bitwise `"sparse"`: inactive blocks have only closed keys); `"budget:B"` = the `B` kv-blocks
with the most selected keys per q-block, the rest dropped (bounded work, a numerics change).

## Kernels

All three kernels use a fully static pipeline, `grid = (B, nQB, nKB)` (fused: `(G, B, nQB/G, nKB)`). The active
kv-block id is folded into the latent BlockSpec index map from the scalar-prefetched list, so steps past `count`
re-target the resident block (no DMA) and `pl.when(i < count)` skips their compute. Tiles are transposed (keys on
sublanes, queries on lanes), so the in-block mask is iota shifts of the bitmap row and the softmax statistics are
column reductions.

**Clamp (must keep):** a q-block with `count == 0` (a fully padded row, the padded tail of a packed row) holds only
the `nKB` pad sentinel. `kv_block_index` / `q_block_index` clip the entry into `[0, n)`. Without the clip the
BlockSpec DMAs one block past the buffer, which real TPUs report as a DMA bounds-check fault. Interpret mode clamps
silently, so `check_tables_interpret` asserts the ranges via a host callback in interpret mode only.

Rows with no open key emit output 0 and finite statistics (`m = 0`, `log l = 0`), so the recompute backward gets
`P = 0` there.

Backward options:

* `"fused"` (default): one pass. `P^T` / `dS^T` are recomputed once per (pair, head), and
  `dkv += [P^T | scale dS^T] @ [dO; q]` is one matmul (5 matmuls per pair-head instead of 3 + 4). The per-core partial
  `dkv_partial[g]` lives in HBM, with explicit DMA read-modify-write per pair. The cores split only the leading
  `G` axis. `G = 2` when `nQB` is even, else 1 (`fused_groups` forces it).
* `"two_pass"`: a q-block-major dQ + pool-cotangent pass (output aliased to a zero buffer, so never-visited pairs
  read 0) and a kv-block-major dKV pass over the transposed lists.

Scoped VMEM: forward 48 MiB, backward 63 MiB. A v5p refuses a full 64 MiB request (the max valid value is
67043328 bytes).

## Per-pair cost options (explicit arguments / `DsaSparseAttentionConfig` fields)

| option | effect | default |
| --- | --- | --- |
| `block_q` x `block_kv` | tiles; 256x512 is bitwise 256x256 and ~8% faster on v5p | 256 x 512 |
| `backward` | `"fused"` / `"two_pass"` | `"fused"` |
| `fwd_nt` | forward `K q^T` as NT dots, no q^T scratch (bit-identical) | off |
| `fwd_lean` | `-1e30` running-max seed drops the post-exp select; slab-built masks (bitwise) | off |
| `bwd_lean` | two-pass: `m + log l` folded, dKV scale applied once (fp32 reassociation) | off |
| `pool_sum` | pool-cotangent reduce: `"reshape"` or 0/1 fp32 `"matmul"` | `"reshape"` |
| `stale_tau` | speculative stale-max forward (exact LSE split kept) | `None` |
| `fused_groups` | fused backward core groups | `None` (auto) |

## Measured (one v5p chip, bf16, H = 16, D = 512)

Numbers taken over from the source development (S = 32k, kpool 4, 2048 keys per query):

* sparse forward 30.9 ms vs 63.7 ms for dense CWA with causal skip and 76.9 ms dense, at block density 0.40;
  forward + backward 114.4 ms (two-pass sparse backward) vs 276.1 ms dense.
* the forward is linear in block density d: 14.1 ms at d = 0.19 up to 76.1 ms at d = 1.0.
* fused backward: 133.7 ms vs 209.1 ms two-pass at d = 1 (-36%), 60.9 vs 78.6 ms at d = 0.4 (-22%). It is the
  default.
* `stale_tau` forward: 1.7x slower (117.5 vs 69.7 ms at d = 1), so it is never the default.

Measured in this tree (`benchmarks/benchmark_dsa_sparse_attention.py`, S = 8192, H = 16, R = 512, kpool 4,
topk 2048, window 0.4, block density 0.570, 256 x 512, fused backward) against `sparse_selected_attention`
(decompressed Dqk = Dv = 256, pre-built dense mask): forward 2.39 vs 2.92 ms, forward + backward 8.21 vs 9.91 ms.

## Limits

* At most 8 tail keys per query (`tails_t` has 8 rows). A selection with `kpool > 9` and a full trailing-pool tail
  needs a wider tail layout.
* Pools are anchored at key 0. A selection that groups pools over compacted valid-token ranks (or per document)
  must be mapped to key-0-anchored pools before calling.
* The sequence axis must stay unsharded (the tables span the whole sequence); shard batch and heads.
