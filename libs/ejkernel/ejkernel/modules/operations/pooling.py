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


"""Pooling operation modules with automatic optimization.

This module implements efficient pooling operations for sequence data, optimized
for JAX execution. Mean pooling is particularly useful for:
    - Sentence embeddings in NLP (pooling token representations)
    - Sequence classification (reducing sequence to fixed-size representation)
    - Feature aggregation across time steps
    - Dimensionality reduction in transformer outputs

Key Features:
    - Efficient mean computation over sequence dimension
    - Support for variable-length sequences via cu_seqlens
    - Configurable chunk size for memory-efficient processing
    - Automatic platform selection (Triton/Pallas/XLA/CUDA)
    - Proper handling of padding in variable-length scenarios

Mathematical Formulation:
    For fixed-length sequences:
        output[b, :] = mean(x[b, :seq_len, :], axis=0)

    For variable-length sequences with cu_seqlens:
        output[i, :] = mean(x[cu_seqlens[i]:cu_seqlens[i+1], :], axis=0)

Use Cases:
    - Sentence-BERT style embeddings (pooling token representations)
    - Text classification (reducing sequence to fixed-size representation)
    - Document embedding for retrieval systems
    - Feature aggregation in encoder-only models

Example:
    >>> from ejkernel.modules.operations import mean_pooling
    >>> import jax.numpy as jnp
    >>>
    >>> # Basic mean pooling
    >>> x = jnp.ones((2, 128, 768))  # batch=2, seq_len=128, hidden=768
    >>> pooled = mean_pooling(x)  # [2, 768]
    >>>
    >>> # With variable-length sequences
    >>> cu_seqlens = jnp.array([0, 50, 128])  # two sequences of lengths 50 and 78
    >>> pooled = mean_pooling(x, cu_seqlens=cu_seqlens)

References:
    - Sentence-BERT: https://arxiv.org/abs/1908.10084
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
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
from .configs import MeanPoolingConfig


class MeanPooling(Kernel[MeanPoolingConfig, Array]):
    """Mean Pooling with custom optimization logic.

    Computes the mean of sequence elements along the sequence dimension, with
    support for variable-length sequences and chunked processing for memory efficiency.

    Features:
        - Efficient mean computation over sequence dimension
        - Support for variable-length sequences via cu_seqlens
        - Configurable chunk size for memory-efficient processing
        - Automatic platform selection (Triton/Pallas/XLA/CUDA)
        - Proper handling of padding in variable-length scenarios

    This is commonly used to convert variable-length token sequences into
    fixed-size representations for classification or embedding tasks.
    """

    def __init__(self):
        """Initialize Mean Pooling module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="mean_pooling")

    def get_impl(self, cfg: MeanPoolingConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for mean pooling

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("mean_pooling", cfg.platform)
        return kernel_registry.get("mean_pooling", platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "... hidden_dim"],
        chunk_size: int = 32,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: MeanPoolingConfig,
    ) -> Float[Array, "batch hidden_dim"]:
        """Execute mean pooling over sequence dimension.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            chunk_size: Size of chunks for processing (default: 32)
            cu_seqlens: Optional cumulative sequence lengths [num_seqs + 1] for
                variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Pooled output [batch, hidden_dim]

        Note:
            When cu_seqlens is provided, padding tokens are excluded from the mean
            computation, ensuring accurate pooling for variable-length sequences.
        """

        block_size = getattr(cfg, "block_size", chunk_size)
        block_dim = getattr(cfg, "block_dim", 64)
        num_warps = getattr(cfg, "num_warps", 4)
        num_stages = getattr(cfg, "num_stages", 1)
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        effective_chunk_size = chunk_size if chunk_size != 32 else block_size

        if platform is not None:
            cfg = MeanPoolingConfig(
                block_size=block_size,
                block_dim=block_dim,
                num_warps=num_warps,
                num_stages=num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
        resolved_platform = detect_platform("mean_pooling", cfg.platform)
        impl = kernel_registry.get("mean_pooling", platform=resolved_platform, backend=cfg.backend)
        kwargs = dict(x=x, chunk_size=effective_chunk_size, cu_seqlens=cu_seqlens)
        if resolved_platform in (Platform.TRITON, Platform.TILELANG, Platform.XLA):
            kwargs.update(
                block_dim=cfg.block_dim,
                block_size=cfg.block_size,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        return impl(**kwargs)

    def heuristic_cfg(self, inv: Invocation[MeanPoolingConfig, Array]) -> MeanPoolingConfig:
        """Provide default configuration with block sizes.

        Selects default block size and warp configuration based on typical
        sequence pooling workloads. These defaults work well for most cases
        but can be overridden via autotuning.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block_size=64, num_warps=4, num_stages=1
        """
        return MeanPoolingConfig(
            block_size=64,
            block_dim=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[MeanPoolingConfig, Array]):
        """Generate candidate configurations for autotuning.

        Mean pooling has tunable block_size for chunked processing. Generates
        configurations with varying block sizes to find optimal performance
        for the specific hardware and input dimensions.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            List of candidate configurations with different block sizes (32, 64, 128)
            and corresponding warp/stage configurations

        Note:
            Smaller block sizes (32) reduce memory usage but may have lower throughput.
            Larger block sizes (128) improve throughput for large sequences.
        """
        block_configs = [
            (32, 32, 2, 1),
            (64, 64, 4, 1),
            (128, 128, 8, 2),
        ]

        candidates = []
        for block_size, block_dim, num_warps, num_stages in block_configs:
            candidates.append(
                MeanPoolingConfig(
                    block_size=block_size,
                    block_dim=block_dim,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[MeanPoolingConfig, Array]):
        """Generate GPU candidates for mean pooling across registered GPU backends.

        Mean pooling is bandwidth-bound: each token is read once and
        accumulated. The two tiles are:

        * ``block_size`` — sequence-axis tile (how many tokens per CTA).
        * ``block_dim`` — hidden-axis tile (how many features per CTA).

        On H100:

        * Small ``block_size`` (32, 64) is wasteful for long sequences.
        * ``block_size=256, block_dim=128`` covers most cases.
        * For wide hidden_dim (>=1024) try ``block_dim=256``.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        triton_configs = [
            (32, 32, 2, 1),
            (64, 64, 4, 1),
            (128, 64, 4, 2),
            (128, 128, 8, 2),
            (256, 64, 4, 2),
            (256, 128, 8, 2),
            (256, 256, 8, 2),
            (512, 128, 8, 2),
        ]
        tilelang_configs = [
            (128, 64, 4, 1),
            (128, 128, 4, 2),
            (256, 64, 4, 1),
            (256, 128, 4, 2),
            (256, 256, 8, 2),
            (512, 128, 8, 2),
        ]
        candidates: list[MeanPoolingConfig] = []
        if "triton" in platforms:
            for block_size, block_dim, num_warps, num_stages in triton_configs:
                candidates.append(
                    MeanPoolingConfig(
                        block_size=block_size,
                        block_dim=block_dim,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="triton",
                        backend="gpu",
                    )
                )
        if "tilelang" in platforms:
            for block_size, block_dim, num_warps, num_stages in tilelang_configs:
                candidates.append(
                    MeanPoolingConfig(
                        block_size=block_size,
                        block_dim=block_dim,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(
                MeanPoolingConfig(
                    block_size=128,
                    block_dim=128,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[MeanPoolingConfig, Array]):
        """Generate TPU candidates for the XLA mean-pooling path."""
        return [
            MeanPoolingConfig(
                block_size=64,
                block_dim=64,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]


_mean_pooling_executor: Executor[MeanPoolingConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("pooling"),
    )
)


def mean_pooling(
    x: Float[Array, "... hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    chunk_size: int = 32,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: MeanPoolingConfig | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Execute mean pooling with automatic optimization.

    Efficiently computes the mean of sequence elements along the sequence dimension,
    optimized for variable-length sequences and chunked processing.

    Note:
        The Triton ``mean_pooling`` kernel is shared with the Native Sparse Attention
        block-compression path and operates on 4-D inputs. When ``platform`` is
        ``None`` or ``"auto"`` this function always routes through the XLA backend,
        which handles arbitrary-rank inputs correctly.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim].
        cu_seqlens: Cumulative sequence lengths for variable-length batches
            (shape ``[num_seqs + 1]``). When ``None``, the full ``seq_len``
            dimension is averaged for every batch element.
        chunk_size: Number of tokens processed per kernel tile (default: 32).
            Smaller values reduce per-tile memory; larger values improve
            throughput for long sequences.
        platform: Specific platform to use (``"triton"``, ``"pallas"``,
            ``"cuda"``, ``"xla"``, or ``"auto"``). When ``None``/``"auto"``,
            the XLA backend is selected (see Note above).
        cfg: Optional :class:`~.configs.MeanPoolingConfig` override. When
            provided, its ``block_size``, ``num_warps``, and ``num_stages``
            fields are used instead of the autotuned values.

    Returns:
        Mean-pooled output of shape [batch, hidden_dim] (or [num_seqs, hidden_dim]
        when ``cu_seqlens`` is supplied and ``batch`` is interpreted as the token
        dimension).

    Example:
        >>> pooled = mean_pooling(x)
        >>>
        >>> pooled = mean_pooling(x, chunk_size=64)
        >>>
        >>> pooled = mean_pooling(x, cu_seqlens=cu_seqs)
        >>>
        >>> out = mean_pooling(x, platform="xla")
    """
    return _mean_pooling_executor(
        MeanPooling(),
        x=x,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        platform=platform,
        _cfg=cfg,
    )
