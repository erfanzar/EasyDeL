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

"""Ragged Page Attention v2 with TurboQuant KV cache compression.

This module provides the high-level operation for read-only paged attention
over a pre-populated TurboQuant-compressed KV cache.  It is the "v2" variant
-- the cache is already filled; no new K/V tokens are written.

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

Compressed page layout
~~~~~~~~~~~~~~~~~~~~~~
Each KV page stores five tensors per token:

* ``key_indices_pages``  -- uint8, shape ``[num_pages, page_size, num_kv_heads, head_dim//2]``
  (two 4-bit indices per byte)
* ``key_signs_pages``    -- uint8, shape ``[num_pages, page_size, num_kv_heads, qjl_dim//8]``
  (eight sign bits per byte)
* ``key_norms_pages``    -- bf16, shape ``[num_pages, page_size, num_kv_heads, 2]``
  (column 0 = original norm, column 1 = residual norm)
* ``value_indices_pages`` -- uint8, shape ``[num_pages, page_size, num_kv_heads, head_dim//2]``
* ``value_norms_pages``  -- bf16, shape ``[num_pages, page_size, num_kv_heads]``

The module exposes two APIs:

* :class:`RaggedPageAttentionv2TurboQuant` -- ``Kernel`` subclass for the
  executor / autotuner framework (class-based).
* :func:`ragged_page_attention_v2_turboquant` -- functional entry point that
  wraps the kernel in an :class:`Executor` with config caching and
  autotuning.

Both APIs support distributed execution via ``shard_map`` when ``mesh``,
``in_specs``, and ``out_specs`` are provided.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int32, UInt8

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
from .configs import RaggedPageAttentionv2TurboQuantConfig


class RaggedPageAttentionv2TurboQuant(Kernel[RaggedPageAttentionv2TurboQuantConfig, Array]):
    """Read-only ragged page attention over TurboQuant-compressed KV pages.

    This kernel computes multi-query / grouped-query attention where the
    KV cache has already been compressed into the TurboQuant packed format.
    No cache writes are performed -- use ``RaggedPageAttentionv3TurboQuant``
    for the fused compress-and-attend variant.

    The computation is structured as a three-level loop:

    1. **Sequence loop** -- iterate over ragged sequences identified by
       ``query_start_loc`` and ``context_lens``.
    2. **Query-block loop** -- tile queries into blocks of
       ``num_queries_per_block`` tokens.  For each block, pre-rotate
       (``Q @ Pi^T``) and pre-project (``Q @ S^T``) the queries once.
    3. **KV-block loop** -- tile compressed KV pages into blocks of
       ``num_kv_pages_per_block`` pages.  For each block, unpack indices
       and signs, compute asymmetric logits (MSE + QJL), dequantise values,
       and accumulate via online softmax.

    Attributes:
        version: Kernel version string (``"2"``).

    See Also:
        :func:`ragged_page_attention_v2_turboquant` -- functional API that
        wraps this kernel in an ``Executor`` with autotuning.
    """

    version = "2"

    def __init__(self):
        """Initialize the RaggedPageAttentionv2TurboQuant kernel module."""
        super().__init__(op_id="ragged_page_attention_v2_turboquant")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
        key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
        value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
        context_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs pages_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_one"],
        num_seqs: Int32[Array, "1"] | int,
        rotation_matrix: Float[Array, "head_dim head_dim"],
        qjl_projection: Float[Array, "qjl_dim head_dim"],
        key_codebook: Float[Array, "key_levels"],
        value_codebook: Float[Array, "value_levels"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        sliding_window: int | None = None,
        mask_value: float | None = None,
        bits: int = 4,
        qjl_dim: int = 128,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv2TurboQuantConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed TurboQuant RPA v2 execution."""
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_rpa_v2_turboquant(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
            key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
            value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
            value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
            context_lens: Int32[Array, "num_seqs"],
            block_tables: Int32[Array, "num_seqs pages_per_seq"],
            query_start_loc: Int32[Array, "num_seqs_plus_one"],
            num_seqs: Int32[Array, "1"] | int,
            rotation_matrix: Float[Array, "head_dim head_dim"],
            qjl_projection: Float[Array, "qjl_dim head_dim"],
            key_codebook: Float[Array, "key_levels"],
            value_codebook: Float[Array, "value_levels"],
            softmax_aux: Float[Array, "num_q_heads"] | None,
        ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
            return self.run(
                queries=queries,
                key_indices_pages=key_indices_pages,
                key_signs_pages=key_signs_pages,
                key_norms_pages=key_norms_pages,
                value_indices_pages=value_indices_pages,
                value_norms_pages=value_norms_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                rotation_matrix=rotation_matrix,
                qjl_projection=qjl_projection,
                key_codebook=key_codebook,
                value_codebook=value_codebook,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                compute_dtype=compute_dtype,
                sliding_window=sliding_window,
                mask_value=mask_value,
                bits=bits,
                qjl_dim=qjl_dim,
                vmem_limit_bytes=vmem_limit_bytes,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            queries,
            key_indices_pages,
            key_signs_pages,
            key_norms_pages,
            value_indices_pages,
            value_norms_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs,
            rotation_matrix,
            qjl_projection,
            key_codebook,
            value_codebook,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_rpa_v2_turboquant,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def get_impl(self, cfg: RaggedPageAttentionv2TurboQuantConfig):
        """Resolve the registered backend implementation for TurboQuant RPA v2."""
        platform = detect_platform("ragged_page_attention_v2_turboquant", cfg.platform)
        return kernel_registry.get("ragged_page_attention_v2_turboquant", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
        key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
        value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
        value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
        context_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs pages_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_one"],
        num_seqs: Int32[Array, "1"] | int,
        rotation_matrix: Float[Array, "head_dim head_dim"],
        qjl_projection: Float[Array, "qjl_dim head_dim"],
        key_codebook: Float[Array, "key_levels"],
        value_codebook: Float[Array, "value_levels"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        sliding_window: int | None = None,
        mask_value: float | None = None,
        bits: int = 4,
        qjl_dim: int = 128,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv2TurboQuantConfig,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute TurboQuant-compressed ragged page attention v2.

        Resolves the backend implementation from the kernel registry and
        dispatches all positional and keyword arguments to it.

        Args:
            queries: Packed query tokens.
                Shape: ``[total_tokens, num_q_heads, head_dim]``
            key_indices_pages: Packed 4-bit Lloyd-Max codebook indices for
                keys.
                Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
                dtype ``uint8``.
            key_signs_pages: Packed 1-bit QJL residual signs for keys.
                Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
                dtype ``uint8``.
            key_norms_pages: Per-token key norms (column 0 = original vector
                norm, column 1 = MSE residual norm after Lloyd-Max).
                Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
            value_indices_pages: Packed 4-bit codebook indices for values.
                Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
                dtype ``uint8``.
            value_norms_pages: Per-token original value norms.
                Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
            context_lens: Number of valid KV tokens per sequence.
                Shape: ``[num_seqs]``, dtype ``int32``.
            block_tables: Page-table mapping sequence positions to physical
                page indices.
                Shape: ``[num_seqs, pages_per_seq]``, dtype ``int32``.
            query_start_loc: Cumulative query token counts.
                Shape: ``[num_seqs + 1]``, dtype ``int32``.
            num_seqs: Total number of sequences (scalar or ``[1]``).
            rotation_matrix: Haar-distributed orthogonal matrix *Pi* shared
                across all heads.
                Shape: ``[head_dim, head_dim]``, dtype ``float32``.
            qjl_projection: Random Gaussian projection matrix *S* for QJL.
                Shape: ``[qjl_dim, head_dim]``, dtype ``float32``.
            key_codebook: Lloyd-Max centroids for key quantisation.
                Shape: ``[2^bits]``, dtype ``float32``.
            value_codebook: Lloyd-Max centroids for value quantisation.
                Shape: ``[2^bits]``, dtype ``float32``.
            softmax_aux: Optional attention-sink logits added before the
                first real KV position to stabilise softmax.
                Shape: ``[num_q_heads]``, dtype ``float32``.
            softmax_scale: Multiplicative scale applied to QK^T logits.
                Defaults to ``1 / sqrt(head_dim)`` when ``None``.
            logits_soft_cap: When set, logits are capped via
                ``cap * tanh(logits / cap)`` to prevent overflow.
            compute_dtype: Dtype for intermediate computation (default
                ``bfloat16``).
            sliding_window: When set, only the most recent
                ``sliding_window`` KV tokens are attended to.
            mask_value: Value used for masked (invalid) logit positions.
            bits: Number of quantisation bits per coordinate (default 4).
            qjl_dim: Dimensionality of the QJL projection (default 128).
            vmem_limit_bytes: Optional hint for TPU VMEM budget.
            platform: Override platform selection (``"xla"``, ``"pallas"``,
                ``"auto"``, etc.).
            cfg: Explicit configuration; when ``None``, the executor
                selects one via heuristics or autotuning.

        Returns:
            Attention output tensor.
                Shape: ``[total_tokens, num_q_heads, head_dim]``
        """
        if platform is not None:
            cfg = RaggedPageAttentionv2TurboQuantConfig(
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
            key_indices_pages=key_indices_pages,
            key_signs_pages=key_signs_pages,
            key_norms_pages=key_norms_pages,
            value_indices_pages=value_indices_pages,
            value_norms_pages=value_norms_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            rotation_matrix=rotation_matrix,
            qjl_projection=qjl_projection,
            key_codebook=key_codebook,
            value_codebook=value_codebook,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            compute_dtype=compute_dtype,
            sliding_window=sliding_window,
            mask_value=mask_value,
            bits=bits,
            qjl_dim=qjl_dim,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def heuristic_cfg(
        self,
        inv: Invocation[RaggedPageAttentionv2TurboQuantConfig, Array],
    ) -> RaggedPageAttentionv2TurboQuantConfig:
        """Return a conservative default config for first-run execution.

        Uses ``None`` for both block-size parameters so the backend can
        select safe defaults automatically.

        Args:
            inv: Invocation object (unused; retained for interface consistency).

        Returns:
            :class:`RaggedPageAttentionv2TurboQuantConfig` with ``platform="auto"``
            and both block sizes set to ``None``.
        """
        return RaggedPageAttentionv2TurboQuantConfig(
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv2TurboQuantConfig, Array]):
        """Generate a small set of block-size candidates for autotuning.

        Returns three configs covering the default (``None, None``) and two
        common ``(num_kv_pages_per_block, num_queries_per_block)`` pairs.

        Args:
            inv: Invocation object (unused; retained for interface consistency).

        Returns:
            List of three :class:`RaggedPageAttentionv2TurboQuantConfig` objects.
        """
        return [
            RaggedPageAttentionv2TurboQuantConfig(
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
            )
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv2TurboQuantConfig, Array]):
        """Generate GPU candidates for TileLang and XLA TurboQuant RPA v2.

        TurboQuant RPA v2 tiles along KV-pages and queries. Pairs sweep:

        * ``(None, None)`` — kernel-default baseline.
        * Small KV-per-block (1, 2) with various queries-per-block — best
          for short contexts.
        * Larger KV-per-block (4, 8) — amortise per-block fixed cost on
          long contexts.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        pairs = (
            (None, None),
            (1, 8),
            (1, 16),
            (2, 16),
            (2, 32),
            (4, 16),
            (4, 32),
            (8, 16),
        )
        candidates: list[RaggedPageAttentionv2TurboQuantConfig] = []
        if "tilelang" in platforms:
            for kv_block, q_block in pairs:
                for stages in (1, 2):
                    candidates.append(
                        RaggedPageAttentionv2TurboQuantConfig(
                            num_kv_pages_per_block=kv_block,
                            num_queries_per_block=q_block,
                            num_warps=4,
                            num_stages=stages,
                            platform="tilelang",
                            backend="gpu",
                        )
                    )
        if "xla" in platforms:
            candidates.extend(
                RaggedPageAttentionv2TurboQuantConfig(
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

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv2TurboQuantConfig, Array]):
        """Generate TPU candidates for the XLA TurboQuant RPA v2 path."""
        return [
            RaggedPageAttentionv2TurboQuantConfig(
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform="xla",
                backend="any",
            )
            for cfg in self.candidate_cfgs(inv)
        ]


_ragged_page_attention_v2_tq_executor: Executor[RaggedPageAttentionv2TurboQuantConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=40),
        persistent=PersistentCache("ragged_page_attention_v2_turboquant"),
    )
)


def ragged_page_attention_v2_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    context_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_one"],
    num_seqs: Int32[Array, "1"] | int,
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    sliding_window: int | None = None,
    mask_value: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedPageAttentionv2TurboQuantConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Execute TurboQuant ragged page attention v2 with autotuned dispatch.

    Functional entry point for read-only TurboQuant-compressed paged
    attention.  Wraps :class:`RaggedPageAttentionv2TurboQuant` in an
    :class:`~ejkernel.ops.Executor` that caches compiled configurations
    and optionally autotunes block sizes on first invocation.

    This is the preferred API for most users.  The class-based API via
    :meth:`RaggedPageAttentionv2TurboQuant.run` is available for advanced
    use cases that require manual configuration control.

    Args:
        queries: Packed query tokens.
            Shape: ``[total_tokens, num_q_heads, head_dim]``
        key_indices_pages: Packed 4-bit Lloyd-Max codebook indices for keys.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        key_signs_pages: Packed 1-bit QJL residual signs for keys.
            Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
            dtype ``uint8``.
        key_norms_pages: Per-token key norms.  Column 0 is the original
            vector norm; column 1 is the MSE residual norm after Lloyd-Max
            quantisation.
            Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
        value_indices_pages: Packed 4-bit codebook indices for values.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        value_norms_pages: Per-token original value norms.
            Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
        context_lens: Number of valid KV tokens per sequence.
            Shape: ``[num_seqs]``, dtype ``int32``.
        block_tables: Page-table mapping sequence positions to physical
            page indices.
            Shape: ``[num_seqs, pages_per_seq]``, dtype ``int32``.
        query_start_loc: Cumulative query token counts.
            Shape: ``[num_seqs + 1]``, dtype ``int32``.
        num_seqs: Total number of sequences (scalar or ``[1]``).
        rotation_matrix: Haar-distributed orthogonal matrix *Pi*.
            Shape: ``[head_dim, head_dim]``
        qjl_projection: Random Gaussian projection matrix *S*.
            Shape: ``[qjl_dim, head_dim]``
        key_codebook: Lloyd-Max centroids for key quantisation.
            Shape: ``[2^bits]``
        value_codebook: Lloyd-Max centroids for value quantisation.
            Shape: ``[2^bits]``
        softmax_aux: Optional attention-sink logits.
            Shape: ``[num_q_heads]``
        softmax_scale: QK^T scaling factor (default ``1 / sqrt(head_dim)``).
        logits_soft_cap: Optional logit soft-capping value.
        compute_dtype: Intermediate compute dtype (default ``bfloat16``).
        sliding_window: Optional sliding-window attention size.
        mask_value: Value for masked logit positions.
        bits: Quantisation bits per coordinate (default 4).
        qjl_dim: QJL projection dimensionality (default 128).
        vmem_limit_bytes: Optional TPU VMEM budget hint.
        platform: Override platform selection.
        cfg: Explicit :class:`RaggedPageAttentionv2TurboQuantConfig`.
        mesh: JAX ``Mesh`` for distributed execution via ``shard_map``.
        in_specs: Input ``PartitionSpec`` tuple for ``shard_map``.
        out_specs: Output ``PartitionSpec`` for ``shard_map``.

    Returns:
        Attention output tensor.
            Shape: ``[total_tokens, num_q_heads, head_dim]``

    Example:
        >>> from ejkernel.modules.operations import ragged_page_attention_v2_turboquant
        >>> output = ragged_page_attention_v2_turboquant(
        ...     queries, key_indices_pages, key_signs_pages, key_norms_pages,
        ...     value_indices_pages, value_norms_pages, context_lens,
        ...     block_tables, query_start_loc, num_seqs,
        ...     rotation_matrix, qjl_projection, key_codebook, value_codebook,
        ... )
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _ragged_page_attention_v2_tq_executor(
        RaggedPageAttentionv2TurboQuant(),
        queries=queries,
        key_indices_pages=key_indices_pages,
        key_signs_pages=key_signs_pages,
        key_norms_pages=key_norms_pages,
        value_indices_pages=value_indices_pages,
        value_norms_pages=value_norms_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        rotation_matrix=rotation_matrix,
        qjl_projection=qjl_projection,
        key_codebook=key_codebook,
        value_codebook=value_codebook,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        compute_dtype=compute_dtype,
        sliding_window=sliding_window,
        mask_value=mask_value,
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
    "RaggedPageAttentionv2TurboQuant",
    "ragged_page_attention_v2_turboquant",
)
