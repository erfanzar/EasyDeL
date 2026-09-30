# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block-sparse DSA attention for training (no KV cache).

The dense training path of the DSA models (glm5_next, glm_moe_dsa) combines the
indexer's top-k selection with the padding/segment mask into a ``[B, S, S]`` boolean
mask and runs full masked attention, materializing ``[B, H, S, S]`` float32 scores.
This helper feeds the same per-head queries/keys/values and the same combined mask to
ejkernel's ``sparse_selected_attention``: a tiled flash kernel with an online softmax
that skips tiles holding no selected pair, with a recomputing backward. The result is
identical by construction; memory drops from ``O(H * S^2)`` floats to the ``S^2`` byte
mask plus ``O(S * H * D)``.

``config.dsa_sparse_training``: ``True`` / ``False`` force the sparse / dense path; unset
(``None``, the default) picks sparse on TPU for sequences of at least
``AUTO_SPARSE_MIN_SEQ`` tokens, where it is faster than dense and uses ~10x less temp
memory, and dense below that. The dense masked path is the test reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from eformer.loggings import get_logger
from ejkernel.kernels import Backend, Platform, kernel_registry  # pyright: ignore[reportMissingTypeStubs]
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float
from spectrax import common_types

from easydel.infra.sequence_packing import pairwise_attention_mask_from_mask_info

logger = get_logger(__name__)


def sparse_selected_training_attention(
    *,
    query: Float[Array, "batch seq heads qk_dim"],
    key: Float[Array, "batch seq kv_heads qk_dim"],
    value: Float[Array, "batch seq kv_heads v_dim"],
    selection_mask: Bool[Array, "batch seq seq"],
    mask_info,
    softmax_scale: float,
    causal: bool,
    dtype: jnp.dtype | None = None,
    block_q: int = 512,
    block_k: int = 512,
    metadata=None,
) -> Float[Array, "batch seq heads v_dim"]:
    """Top-k selected attention over decompressed heads, differentiable, tile-skipping.

    Args:
        query: Per-head queries (BTHD).
        key: Per-head keys (BTHD).
        value: Per-head values (BTHD).
        selection_mask: Indexer top-k selection as a dense boolean ``[B, S, S]`` mask.
        mask_info: Mask metadata; its padding / packed-segment mask is AND-ed in.
        softmax_scale: Same logit scale as the dense path.
        causal: Apply ``j <= i`` in the kernel (as the dense path's ``causal`` flag).
        dtype: Compute dtype for q/k/v (the attention ``attn_dtype``); ``None`` keeps them.
        block_q: Query tile size (XLA fallback).
        block_k: Key tile size (XLA fallback).
        metadata: The attention op's :class:`OperationMetadata`; supplies the mesh and the
            batch/head partition specs for running the Pallas kernel under ``shard_map``.

    Returns:
        ``[B, S, H, v_dim]`` attention output.
    """
    q_len, kv_len = query.shape[1], key.shape[1]
    mask = selection_mask.astype(bool)
    pairwise = pairwise_attention_mask_from_mask_info(mask_info, q_len, kv_len)
    if pairwise is not None:
        pairwise = pairwise.astype(bool)
        if pairwise.ndim == 4:
            pairwise = pairwise[:, 0]
        mask = mask & pairwise
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)
    kw = dict(softmax_scale=float(softmax_scale), causal=bool(causal))
    xla = kernel_registry.get("sparse_selected_attention", platform=Platform.XLA, backend=Backend.ANY)
    if jax.default_backend() != "tpu":
        return xla(query, key, value, mask, block_q=int(block_q), block_k=int(block_k), **kw)
    pallas = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)

    def run_xla(reason: str):
        logger.warning_once(f"sparse_selected_training_attention: {reason}; using the XLA implementation.")
        return xla(query, key, value, mask, block_q=int(block_q), block_k=int(block_k), **kw)

    # Inside an outer shard_map with manual axes (e.g. an SPMD pipeline body) a nested shard_map
    # over the full mesh is invalid: run the kernel directly if every remaining auto axis has
    # size 1, else keep the XLA path.
    ambient = jax.sharding.get_abstract_mesh()
    if not ambient.empty and ambient.manual_axes:
        auto = [n for n in ambient.axis_names if n not in ambient.manual_axes and ambient.shape[n] > 1]
        if auto:
            return run_xla(f"ambient manual axes {tuple(ambient.manual_axes)} with sharded auto axes {auto}")
        return pallas(query, key, value, mask, **kw)
    # Mosaic kernels are not SPMD-partitionable: run directly on one device, under shard_map
    # over the (batch, head) axes on a multi-device mesh, and fall back to XLA otherwise.
    mesh = None if metadata is None else metadata.mesh
    if mesh is None:
        if jax.device_count() == 1:
            return pallas(query, key, value, mask, **kw)
        return run_xla("no mesh on a multi-device host")
    if mesh.size == 1:
        return pallas(query, key, value, mask, **kw)
    specs = _bthd_specs(metadata)
    if specs is None:
        return run_xla("attention sharding is not a plain (batch, head) layout")
    q_spec, kv_spec, batch_axes = specs
    if not (
        _divides(mesh, q_spec[0], query.shape[0])
        and _divides(mesh, q_spec[2], query.shape[2])
        and _divides(mesh, kv_spec[2], key.shape[2])
    ):
        # A non-dividing batch/head shard would error inside shard_map; keep the SPMD XLA path.
        return run_xla(
            f"batch {query.shape[0]} / heads {query.shape[2]} / kv heads {key.shape[2]} "
            "are not divisible by their mesh axes"
        )
    return jax.shard_map(
        lambda q, k, v, m: pallas(q, k, v, m, **kw),
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec, PartitionSpec(batch_axes, None, None)),
        out_specs=q_spec,
        check_vma=False,
    )(query, key, value, mask)


def _axes_size(mesh, axes) -> int:
    """Number of devices along a PartitionSpec entry (None, an axis name, or a tuple of names)."""
    if axes is None:
        return 1
    names = axes if isinstance(axes, tuple) else (axes,)
    size = 1
    for name in names:
        size *= int(mesh.shape[name])
    return size


def _divides(mesh, axes, dim: int) -> bool:
    return int(dim) % _axes_size(mesh, axes) == 0


def _bthd_specs(metadata):
    """``(q_spec, kv_spec, batch_axes)`` when only batch/head axes are sharded, else ``None``."""
    try:
        rules = metadata.get_shardings(common_types.MODE_TRAIN, layout="bthd", qkv_mni_sharding=True)
    except Exception as err:
        logger.warning_once(f"sparse_selected_training_attention: could not resolve attention shardings ({err!r}).")
        return None
    mesh = metadata.mesh

    def eff(axes):
        # Axes spanning a single device (e.g. ``sp`` of size 1) are not really sharded.
        return None if axes is None or _axes_size(mesh, axes) == 1 else axes

    q = tuple(eff(a) for a in tuple(rules.query) + (None,) * (4 - len(tuple(rules.query))))
    k = tuple(eff(a) for a in tuple(rules.key) + (None,) * (4 - len(tuple(rules.key))))
    # q and kv must agree on batch and head axes: a mismatch would hand each shard local q heads
    # with a different kv-head slice, and the GQA map ``hi // g`` would pick the wrong kv head.
    if q[1] is not None or q[3] is not None or k[1] is not None or k[3] is not None or q[0] != k[0] or q[2] != k[2]:
        return None
    return PartitionSpec(q[0], None, q[2], None), PartitionSpec(k[0], None, k[2], None), q[0]


# Auto mode threshold. One glm_moe_dsa layer fwd+bwd on TPU v5 with 512x512 tiles: sparse
# 8.05 / 21.3 / 66.4 / 224.5 ms vs dense 7.35 / 22.2 / 74.1 / OOM ms at 4k / 8k / 16k / 32k.
AUTO_SPARSE_MIN_SEQ = 8192


def use_sparse_training_attention(
    layer, topk_indices, cache_view, mask_info, output_attentions: bool, sliding_window, seq_len: int | None = None
) -> bool:
    """Whether to run the sparse op (it computes exactly what the dense top-k-mask path would).

    ``config.dsa_sparse_training``: ``True`` / ``False`` force it on / off; unset (``None``)
    enables it on TPU when ``seq_len >= AUTO_SPARSE_MIN_SEQ``, where it is faster than the
    dense path and uses ~10x less temp memory.

    ``concatenate``'s ``init_attention_bias`` is a lazy closure derived from ``mask_info``
    alone; the sparse helper receives the same combined mask, so it needs no check here.
    """
    eligible = (
        topk_indices is not None
        and cache_view is None
        and mask_info is not None
        and not output_attentions
        and sliding_window is None
        and layer._softmax_aux() is None
    )
    if not eligible:
        return False
    flag = getattr(layer.config, "dsa_sparse_training", None)
    if flag is not None:
        return bool(flag)
    return jax.default_backend() == "tpu" and seq_len is not None and int(seq_len) >= AUTO_SPARSE_MIN_SEQ
