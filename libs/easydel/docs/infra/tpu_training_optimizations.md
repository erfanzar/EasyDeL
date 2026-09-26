# TPU training optimizations: MoE dispatch, selection, remat

These results were measured locally on 4x TPU v5 (jax 0.11.2, libtpu 0.0.48).
Every accepted change was compared with an independent reference and with
the previous code on identical parameters. Rejected candidates are listed
with the measurement that ruled them out. Top-k selection for indexers is
described in [Modular indexers and hyper-connections](indexers_and_hyper_connections.md).

## MoE token permutation

`permute` gathers the replicated tokens into expert order, and `unpermute`
gathers the expert outputs back. JAX transposes a gather into a scatter-add,
and on TPU those row scatters cost as much as the rest of the dispatch
(`unpermute` fwd+bwd: 4.41 ms, of which the scatter was 2.15 ms, at 65536 rows
x 2048 bf16).

`sort_activations_custom` now computes the permutation as the linear solve
`jax.lax.custom_linear_solve(y -> y[inverse], x)`. Because a permutation
matrix is orthogonal, its transpose solve is the inverse gather, so reverse
mode has no scatter. Unlike `custom_vjp`, which cannot be forward-differentiated,
or `linear_call`, which has no batching rule, this keeps JVP, `jacfwd`, `vmap`
and higher derivatives. Outputs and gradients are **bit-identical** to the
previous implementation.

| Qwen3-MoE (2 layers, H=1024, E=64, k=8), 8x2048 tokens, bf16 | fwd+bwd before | after |
| --- | --- | --- |
| dp=4 | 28.81 ms | 26.85 ms (-6.8%) |
| fsdp=4 (folded expert mesh) | 72.37 ms | 60.78 ms (-16.0%) |
| tp=4 | 52.37 ms | 46.74 ms (-10.8%) |

The forward pass, compiled memory and collective counts are unchanged.

### Folded expert mesh

When fsdp or sp is bound into expert parallelism (`fsdp_is_ep_bound` and
`sp_is_ep_bound`, both on by default), the fused MoE runs on a folded
`(dp, ep, tp)` mesh whose expert axis spans `ep * fsdp * sp` devices. The
per-shard expert count now follows that mesh. Previously it followed the
physical `ep` axis, so every such layout, including the default fsdp-bound
training mesh, failed with
`expected rhs group dimension size to be 8, got 2`.

### ejkernel grouped matmul default on TPU

The ejkernel `grouped_matmul` default (a 128x128x128 hint) ran 6-12x slower
than XLA's own `ragged_dot` tiler, from decode (m=512) to training
(m=65536) row counts. For example, at m=65536, k=2048, n=1536 the forward
took 27.7 ms against 2.28 ms. The TPU default is now the unhinted XLA path.
EasyDeL's MoE layers already requested it explicitly.

### Rejected

- **Gather fused into a Pallas grouped matmul.** A per-row DMA gather, which is
  the core of such a kernel, is issue-bound at about 60 ns per row: 3.96 ms against
  XLA's 0.18 ms for 65536 rows of 4 KB. Unaligned single-row bf16 copies into
  VMEM do not compile at all, and the Pallas GMM itself is 1.1-2.3x slower
  than XLA `ragged_dot` here.
- **Gathering `x[p // k]` instead of materializing the repeat.** On a single
  block this was 16% faster forward. At model level it gave only 0.3-2.3% and
  raised tp=4 compiled memory by 15% (2893 to 3325 MiB).
- `unique_indices` gather hints (no effect), a counting sort (17x slower than
  `argsort`), and an integer-scatter inverse permutation (5.7x slower than
  `argsort`).

## Selective rematerialization

Under `nothing_saveable`, the backward pass replays each decoder layer. That
replay includes the indexer's scoring and top-k, and the mHC projection and
Sinkhorn, even though their results are small. Three checkpoint names now mark
those results, so a policy can keep them while recomputing everything else:

- `indexer_topk`: selected indices, `[B, S, k]` int32 (sparse indexer,
  compressed indexer, GLM-MoE-DSA indexer operation);
- `mhc_logits`: hyper-connection projection logits, `[B, S, (hc + 2) * hc]`;
- `mhc_coefficients`: read/write gates and mixer.

```python
config.gradient_checkpointing = "save_only_these_names"
config.gradient_checkpointing_targets = ["indexer_topk", "mhc_logits", "mhc_coefficients"]
```

This is a configuration choice; the meaning of `nothing_saveable` is unchanged.
Measured on the real trainer step (`trainers.trainer._fn.training_step` with
AdamW), 4 layers, fsdp=4, batch 4 x 8192 tokens, bf16:

| family | `nothing_saveable` | retain the three names | compiled temp |
| --- | --- | --- | --- |
| GLM-MoE-DSA | 779.2 ms | 750.1 ms (-3.7%) | 3919 -> 4152 MiB |
| GLM-5-Next | 2895.4 ms | 2757.8 ms (-4.8%) | 6668 -> 6581 MiB |
| DeepSeek-V4 | 1058.5 ms | 944.1 ms (-10.8%) | 8260 -> 6923 MiB |

The GLM-MoE-DSA memory increase is the retained indices themselves: `[1, 8192,
2048]` int32 is 64 MiB per sparse layer. First-order gradients were compared with
a no-remat reference in f32 at `Precision.HIGHEST`. Both policies are equally
close: max relative L2 5.1e-4 for both on GLM-MoE-DSA with dense MLPs, 6.4e-4
for both on DeepSeek-V4, and identical results on GLM-5-Next. With MoE routing
enabled, the changed fusion order can flip a near-tied router choice, exactly as
switching between other remat policies does. In bf16, `nothing_saveable` itself
differs from no remat by up to 0.9 relative L2 for the same reason.
