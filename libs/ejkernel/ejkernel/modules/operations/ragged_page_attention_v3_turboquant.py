# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Ragged Page Attention v3 with TurboQuant KV cache compression.

This module provides the high-level operation for **fused update + attention**
over a TurboQuant-compressed paged KV cache.  It is the "v3" variant -- new
K/V tokens are compressed and written into the page cache, then attention is
computed against the updated cache, all in a single call.

TurboQuant (ICLR 2026, arXiv:2504.19874) compresses the KV cache in two
stages:

1. **MSE stage** -- each token's key/value vector is normalised, randomly
   rotated by a Haar-distributed orthogonal matrix *Pi*, and then each
   coordinate is independently quantised with a Lloyd-Max scalar quantiser
   optimal for N(0, 1/d).  The packed 4-bit codebook indices and the
   original vector norm are stored.

2. **QJL correction (keys only)** -- the MSE reconstruction residual is
   projected through a random Gaussian matrix *S* and the resulting sign
   vector is stored as packed bits.  At attention time the unbiased
   inner-product estimator combines the codebook MSE term with a QJL
   sign-bit correction::

       <q, k> ~ <q_rot, cb[k_idx]> * ||k||
              + sqrt(pi/2) / qjl_dim * <q_proj, signs> * ||r_k||

Differences from v2
~~~~~~~~~~~~~~~~~~~
* **v2** (``ragged_page_attention_v2_turboquant``) is read-only: pages must
  be pre-populated before calling the kernel.
* **v3** (this module) accepts raw ``keys`` and ``values`` tensors, compresses
  them on-the-fly, writes the compressed representation into the page
  tensors, and then computes attention -- making it suitable for single-call
  prefill + decode serving.

The module exposes two APIs:

* :class:`RaggedPageAttentionv3TurboQuant` -- ``Kernel`` subclass for the
  executor / autotuner framework (class-based).
* :func:`ragged_page_attention_v3_turboquant` -- functional entry point that
  wraps the kernel in an :class:`Executor` with config caching and
  autotuning.

Both APIs support distributed execution via ``shard_map`` when ``mesh``,
``in_specs``, and ``out_specs`` are provided.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int32, UInt8

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedPageAttentionv3TurboQuantConfig


class RaggedPageAttentionv3TurboQuant(
    Kernel[RaggedPageAttentionv3TurboQuantConfig, tuple[Array, Array, Array, Array, Array, Array]]
):
    """Fused update + attention ragged page attention over TurboQuant-compressed KV pages.

    This kernel performs two phases per sequence in a single ``fori_loop``:

    **Phase 1 -- KV compression and page update:**
    For each block of new tokens, the raw key and value vectors are
    compressed via TurboQuant (normalise -> rotate -> Lloyd-Max quantise
    -> pack; additionally for keys: compute residual -> QJL project -> pack
    signs) and the resulting packed tensors are written into the page cache
    at the correct positions.

    **Phase 2 -- Asymmetric attention:**
    Identical to v2: queries are pre-rotated and pre-projected once per
    query block, then a tiled KV-block loop unpacks compressed data,
    computes asymmetric logits (MSE + QJL correction), dequantises values,
    and accumulates via online softmax.

    The six-element return tuple contains the attention output followed by
    the five *updated* page tensors (keys indices, keys signs, keys norms,
    values indices, values norms), allowing the caller to persist the
    modified cache for subsequent steps.

    Attributes:
        version: Kernel version string (``"3"``).

    See Also:
        :func:`ragged_page_attention_v3_turboquant` -- functional API that
        wraps this kernel in an ``Executor`` with autotuning.
    """

    version = "3"

    def __init__(self):
        """Initialize the RaggedPageAttentionv3TurboQuant kernel module."""
        super().__init__(op_id="ragged_page_attention_v3_turboquant")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
        key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
        value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        rotation_matrix: Float[Array, "head_dim head_dim"],
        qjl_projection: Float[Array, "qjl_dim head_dim"],
        key_codebook: Float[Array, "key_levels"],
        value_codebook: Float[Array, "value_levels"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        bits: int = 4,
        qjl_dim: int = 128,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv3TurboQuantConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed TurboQuant RPA v3 execution."""
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_rpa_v3_turboquant(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            keys: Float[Array, "total_tokens num_kv_heads head_dim"],
            values: Float[Array, "total_tokens num_kv_heads head_dim"],
            key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
            key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
            value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
            kv_lens: Int32[Array, "max_num_seqs"],
            block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
            query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
            distribution: Int32[Array, "3"],
            rotation_matrix: Float[Array, "head_dim head_dim"],
            qjl_projection: Float[Array, "qjl_dim head_dim"],
            key_codebook: Float[Array, "key_levels"],
            value_codebook: Float[Array, "value_levels"],
            softmax_aux: Float[Array, "num_q_heads"] | None,
        ) -> tuple[
            Float[Array, "total_tokens num_q_heads head_dim"],
            UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
            Float[Array, "num_pages page_size num_kv_heads two"],
            UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            Float[Array, "num_pages page_size num_kv_heads"],
        ]:
            return self.run(
                queries=queries,
                keys=keys,
                values=values,
                key_indices_pages=key_indices_pages,
                key_signs_pages=key_signs_pages,
                key_norms_pages=key_norms_pages,
                value_indices_pages=value_indices_pages,
                value_norms_pages=value_norms_pages,
                kv_lens=kv_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                distribution=distribution,
                rotation_matrix=rotation_matrix,
                qjl_projection=qjl_projection,
                key_codebook=key_codebook,
                value_codebook=value_codebook,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                bits=bits,
                qjl_dim=qjl_dim,
                vmem_limit_bytes=vmem_limit_bytes,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            queries,
            keys,
            values,
            key_indices_pages,
            key_signs_pages,
            key_norms_pages,
            value_indices_pages,
            value_norms_pages,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
            rotation_matrix,
            qjl_projection,
            key_codebook,
            value_codebook,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_rpa_v3_turboquant,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def get_impl(self, cfg: RaggedPageAttentionv3TurboQuantConfig):
        """Resolve the registered backend implementation for TurboQuant RPA v3."""
        platform = detect_platform("ragged_page_attention_v3_turboquant", cfg.platform)
        return kernel_registry.get("ragged_page_attention_v3_turboquant", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
        key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
        value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        rotation_matrix: Float[Array, "head_dim head_dim"],
        qjl_projection: Float[Array, "qjl_dim head_dim"],
        key_codebook: Float[Array, "key_levels"],
        value_codebook: Float[Array, "value_levels"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        bits: int = 4,
        qjl_dim: int = 128,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv3TurboQuantConfig,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads head_dim"],
        UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
        Float[Array, "num_pages page_size num_kv_heads two"],
        UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        Float[Array, "num_pages page_size num_kv_heads"],
    ]:
        """Execute TurboQuant-compressed ragged page attention v3.

        Compresses new K/V tokens, writes them into the page cache, then
        computes attention against the full (updated) compressed cache.

        Args:
            queries: Packed query tokens.
                Shape: ``[total_tokens, num_q_heads, head_dim]``
            keys: New key tokens to compress and write into the cache.
                Shape: ``[total_tokens, num_kv_heads, head_dim]``
            values: New value tokens to compress and write into the cache.
                Shape: ``[total_tokens, num_kv_heads, head_dim]``
            key_indices_pages: Existing packed 4-bit key codebook indices.
                Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
                dtype ``uint8``.  New token indices are written into this
                tensor at the positions specified by ``block_tables``; the
                updated tensor is returned as element 1 of the output tuple.
            key_signs_pages: Existing packed QJL key sign bits.
                Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
                dtype ``uint8``.
            key_norms_pages: Existing key norms (col 0 = original, col 1 =
                residual).
                Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
            value_indices_pages: Existing packed 4-bit value codebook indices.
                Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
                dtype ``uint8``.
            value_norms_pages: Existing value norms.
                Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
            kv_lens: Context length (including new tokens) per sequence.
                Shape: ``[max_num_seqs]``, dtype ``int32``.
            block_tables: Flattened page-table mapping.
                Shape: ``[max_num_seqs * pages_per_seq]``, dtype ``int32``.
            query_start_loc: Cumulative query token counts.
                Shape: ``[max_num_seqs + 1]``, dtype ``int32``.
            distribution: Three-element vector ``[num_prefill, num_decode,
                num_total]`` describing the workload mix.
                Shape: ``[3]``, dtype ``int32``.
            rotation_matrix: Haar-distributed orthogonal matrix *Pi*.
                Shape: ``[head_dim, head_dim]``
            qjl_projection: QJL projection matrix *S*.
                Shape: ``[qjl_dim, head_dim]``
            key_codebook: Lloyd-Max centroids for keys.
                Shape: ``[2^bits]``
            value_codebook: Lloyd-Max centroids for values.
                Shape: ``[2^bits]``
            softmax_aux: Optional attention-sink logits.
                Shape: ``[num_q_heads]``
            softmax_scale: QK^T scaling (default ``1 / sqrt(head_dim)``).
            sliding_window: Optional sliding-window size.
            logits_soft_cap: Optional logit soft-capping value.
            bits: Quantisation bits per coordinate (default 4).
            qjl_dim: QJL projection dimensionality (default 128).
            vmem_limit_bytes: Optional TPU VMEM budget hint.
            platform: Override platform selection.
            cfg: Explicit configuration object.

        Returns:
            Six-element tuple:

            * ``output`` -- attention result,
              shape ``[total_tokens, num_q_heads, head_dim]``
            * ``key_indices_pages`` -- updated packed key indices
            * ``key_signs_pages`` -- updated packed key signs
            * ``key_norms_pages`` -- updated key norms
            * ``value_indices_pages`` -- updated packed value indices
            * ``value_norms_pages`` -- updated value norms
        """
        if platform is not None:
            cfg = RaggedPageAttentionv3TurboQuantConfig(
                chunk_prefill_size=cfg.chunk_prefill_size,
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            keys=keys,
            values=values,
            key_indices_pages=key_indices_pages,
            key_signs_pages=key_signs_pages,
            key_norms_pages=key_norms_pages,
            value_indices_pages=value_indices_pages,
            value_norms_pages=value_norms_pages,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            rotation_matrix=rotation_matrix,
            qjl_projection=qjl_projection,
            key_codebook=key_codebook,
            value_codebook=value_codebook,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            bits=bits,
            qjl_dim=qjl_dim,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
        )

    def heuristic_cfg(
        self,
        inv: Invocation[RaggedPageAttentionv3TurboQuantConfig, tuple[Array, Array, Array, Array, Array, Array]],
    ) -> RaggedPageAttentionv3TurboQuantConfig:
        """Return a conservative default config for first-run execution.

        Uses ``None`` for block-size parameters so the backend can select
        safe defaults automatically.

        Args:
            inv: Invocation object (unused; retained for interface consistency).

        Returns:
            :class:`RaggedPageAttentionv3TurboQuantConfig` with ``platform="auto"``
            and all block-size fields set to ``None``.
        """
        return RaggedPageAttentionv3TurboQuantConfig(
            chunk_prefill_size=None,
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(
        self,
        inv: Invocation[RaggedPageAttentionv3TurboQuantConfig, tuple[Array, Array, Array, Array, Array, Array]],
    ):
        """Generate a small set of block-size candidates for autotuning.

        Returns six configs with ``chunk_prefill_size=None`` covering the
        default ``(None, None)`` and five common
        ``(num_kv_pages_per_block, num_queries_per_block)`` pairs.

        Args:
            inv: Invocation object (unused; retained for interface consistency).

        Returns:
            List of :class:`RaggedPageAttentionv3TurboQuantConfig` objects.
        """
        return [
            RaggedPageAttentionv3TurboQuantConfig(
                chunk_prefill_size=None,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
                num_warps=4,
                num_stages=1,
                platform="auto",
                backend="any",
            )
            for num_kv_pages_per_block, num_queries_per_block in (
                (None, None),
                (1, 8),
                (2, 16),
                (4, 16),
                (2, 32),
                (4, 32),
            )
        ]

    def candidate_cfgs_gpu(
        self,
        inv: Invocation[RaggedPageAttentionv3TurboQuantConfig, tuple[Array, Array, Array, Array, Array, Array]],
    ):
        """Generate GPU candidates for TileLang and XLA TurboQuant RPA v3."""
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        pairs = ((None, None), (1, 8), (2, 16), (4, 16), (2, 32), (4, 32))
        candidates: list[RaggedPageAttentionv3TurboQuantConfig] = []
        if "tilelang" in platforms:
            candidates.extend(
                RaggedPageAttentionv3TurboQuantConfig(
                    chunk_prefill_size=None,
                    num_kv_pages_per_block=kv_block,
                    num_queries_per_block=q_block,
                    num_warps=4,
                    num_stages=1,
                    platform="tilelang",
                    backend="gpu",
                )
                for kv_block, q_block in pairs
            )
        if "xla" in platforms:
            candidates.extend(
                RaggedPageAttentionv3TurboQuantConfig(
                    chunk_prefill_size=None,
                    num_kv_pages_per_block=kv_block,
                    num_queries_per_block=q_block,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
                for kv_block, q_block in pairs
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(
        self,
        inv: Invocation[RaggedPageAttentionv3TurboQuantConfig, tuple[Array, Array, Array, Array, Array, Array]],
    ):
        """Generate TPU candidates for the XLA TurboQuant RPA v3 path."""
        return [
            RaggedPageAttentionv3TurboQuantConfig(
                chunk_prefill_size=cfg.chunk_prefill_size,
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform="xla",
                backend="any",
            )
            for cfg in self.candidate_cfgs(inv)
        ]


_ragged_page_attention_v3_tq_executor: Executor[
    RaggedPageAttentionv3TurboQuantConfig,
    tuple[Array, Array, Array, Array, Array, Array],
] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=40),
        persistent=PersistentCache("ragged_page_attention_v3_turboquant"),
    )
)


def ragged_page_attention_v3_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedPageAttentionv3TurboQuantConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    Float[Array, "num_pages page_size num_kv_heads two"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    Float[Array, "num_pages page_size num_kv_heads"],
]:
    """Execute TurboQuant ragged page attention v3 with autotuned dispatch.

    Functional entry point for fused compress-and-attend TurboQuant paged
    attention.  Wraps :class:`RaggedPageAttentionv3TurboQuant` in an
    :class:`~ejkernel.ops.Executor` that caches compiled configurations
    and optionally autotunes block sizes on first invocation.

    This is the preferred API for most users.  The class-based API via
    :meth:`RaggedPageAttentionv3TurboQuant.run` is available for advanced
    use cases that require manual configuration control.

    Args:
        queries: Packed query tokens.
            Shape: ``[total_tokens, num_q_heads, head_dim]``
        keys: New key tokens to compress and cache.
            Shape: ``[total_tokens, num_kv_heads, head_dim]``
        values: New value tokens to compress and cache.
            Shape: ``[total_tokens, num_kv_heads, head_dim]``
        key_indices_pages: Existing packed 4-bit key codebook indices.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        key_signs_pages: Existing packed QJL key sign bits.
            Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
            dtype ``uint8``.
        key_norms_pages: Existing key norms (col 0 = original norm,
            col 1 = MSE residual norm).
            Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
        value_indices_pages: Existing packed 4-bit value codebook indices.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        value_norms_pages: Existing value norms.
            Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
        kv_lens: Context lengths per sequence (including new tokens).
            Shape: ``[max_num_seqs]``, dtype ``int32``.
        block_tables: Flattened page-table mapping.
            Shape: ``[max_num_seqs * pages_per_seq]``, dtype ``int32``.
        query_start_loc: Cumulative query token counts.
            Shape: ``[max_num_seqs + 1]``, dtype ``int32``.
        distribution: Workload descriptor ``[num_prefill, num_decode,
            num_total]``.
            Shape: ``[3]``, dtype ``int32``.
        rotation_matrix: Haar-distributed orthogonal matrix *Pi*.
            Shape: ``[head_dim, head_dim]``
        qjl_projection: QJL projection matrix *S*.
            Shape: ``[qjl_dim, head_dim]``
        key_codebook: Lloyd-Max centroids for keys.
            Shape: ``[2^bits]``
        value_codebook: Lloyd-Max centroids for values.
            Shape: ``[2^bits]``
        softmax_aux: Optional attention-sink logits.
            Shape: ``[num_q_heads]``
        softmax_scale: QK^T scaling (default ``1 / sqrt(head_dim)``).
        sliding_window: Optional sliding-window attention size.
        logits_soft_cap: Optional logit soft-capping value.
        bits: Quantisation bits per coordinate (default 4).
        qjl_dim: QJL projection dimensionality (default 128).
        vmem_limit_bytes: Optional TPU VMEM budget hint.
        platform: Override platform selection.
        cfg: Explicit :class:`RaggedPageAttentionv3TurboQuantConfig`.
        mesh: JAX ``Mesh`` for distributed execution via ``shard_map``.
        in_specs: Input ``PartitionSpec`` tuple for ``shard_map``.
        out_specs: Output ``PartitionSpec`` tuple for ``shard_map``.

    Returns:
        Six-element tuple:

        * ``output`` -- attention result,
          shape ``[total_tokens, num_q_heads, head_dim]``
        * ``key_indices_pages`` -- updated packed key indices
        * ``key_signs_pages`` -- updated packed key signs
        * ``key_norms_pages`` -- updated key norms
        * ``value_indices_pages`` -- updated packed value indices
        * ``value_norms_pages`` -- updated value norms

    Example:
        >>> from ejkernel.modules.operations import ragged_page_attention_v3_turboquant
        >>> output, ki, ks, kn, vi, vn = ragged_page_attention_v3_turboquant(
        ...     queries, keys, values,
        ...     key_indices_pages, key_signs_pages, key_norms_pages,
        ...     value_indices_pages, value_norms_pages,
        ...     kv_lens, block_tables, query_start_loc, distribution,
        ...     rotation_matrix, qjl_projection, key_codebook, value_codebook,
        ... )
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _ragged_page_attention_v3_tq_executor(
        RaggedPageAttentionv3TurboQuant(),
        queries=queries,
        keys=keys,
        values=values,
        key_indices_pages=key_indices_pages,
        key_signs_pages=key_signs_pages,
        key_norms_pages=key_norms_pages,
        value_indices_pages=value_indices_pages,
        value_norms_pages=value_norms_pages,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        rotation_matrix=rotation_matrix,
        qjl_projection=qjl_projection,
        key_codebook=key_codebook,
        value_codebook=value_codebook,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        bits=bits,
        qjl_dim=qjl_dim,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


__all__ = (
    "RaggedPageAttentionv3TurboQuant",
    "ragged_page_attention_v3_turboquant",
)
