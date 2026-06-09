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

"""Gated Delta Rule (GDR) operation module with automatic optimization.

This module provides the GatedDeltaRule operation, a linear attention
mechanism that combines gated delta rule updates with learnable decay.
GDR achieves O(N) complexity while supporting efficient incremental inference.

Key characteristics of GDR:
    - Memory matrix: [num_heads, head_dim, value_dim] per batch
    - Gated delta updates: Selective memory modification
    - Learnable decay: Controls memory retention over time
    - Beta parameter: Per-token gating for updates

The core recurrence:
    h_t = exp(decay_t) * h_{t-1} + k_t (x) (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

Algorithm modes:
    - Chunked (default): Parallel within chunks using Neumann series
    - Recurrent: Pure sequential scan for memory efficiency
    - Single-step: Optimized path for autoregressive generation

Features:
    - Automatic platform selection (XLA primary)
    - Configuration caching for consistent performance
    - L2 normalization option for stability
    - State passthrough for incremental inference
    - Custom VJP with analytical backward for numerical stability

Example:
    >>> from ejkernel.modules.operations import gated_delta_rule
    >>>
    >>> # Training forward pass
    >>> output = gated_delta_rule(
    ...     query, key, value, beta, decay,
    ...     use_chunked=True,
    ... )
    >>>
    >>> # Inference with state
    >>> output, state = gated_delta_rule(
    ...     query, key, value, beta, decay,
    ...     return_state=True,
    ... )
    >>> # Next token
    >>> output_new, state_new = gated_delta_rule(
    ...     q_new, k_new, v_new, beta_new, decay_new,
    ...     initial_state=state, return_state=True,
    ... )

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

from __future__ import annotations

import os
import typing
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

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
from .configs import GatedDeltaRuleConfig


class GatedDeltaRule(Kernel[GatedDeltaRuleConfig, Array]):
    """Gated Delta Rule (GDR) operation.

    A linear attention mechanism using gated delta rule updates to maintain
    an associative memory matrix. Supports both training (chunked) and
    inference (single-step) modes with O(N) complexity.

    The operation maintains a [head_dim, value_dim] memory matrix per head
    that stores key-value associations. The gated delta update mechanism
    ensures only novel information is added, improving memory efficiency.

    Attributes:
        op_id: Operation identifier for registry lookup ("gated_delta_rule")
    """

    version = "2"

    def __init__(self):
        """Initialize GatedDeltaRule operation.

        Sets up the kernel with operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="gated_delta_rule")

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_heads v_head_dim"],
        beta: Float[Array, "batch seq_len num_heads"],
        decay: Float[Array, "batch seq_len num_heads"] | None = None,
        initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
        *,
        use_qk_l2norm: bool = True,
        use_chunked: bool = True,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GatedDeltaRuleConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None,
        check_vma: bool = False,
        seg_ids: Int[Array, "batch seq_len"] | None = None,
        **_,
    ):
        """Create a shard_map wrapper for GDR execution.

        The wrapped function keeps the public BTHD layout and delegates to
        ``self.run`` inside each shard.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_gdr(
            query: Float[Array, "batch seq_len num_heads qk_head_dim"],
            key: Float[Array, "batch seq_len num_heads qk_head_dim"],
            value: Float[Array, "batch seq_len num_heads v_head_dim"],
            beta: Float[Array, "batch seq_len num_heads"],
            decay: Float[Array, "batch seq_len num_heads"] | None,
            initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None,
            seg_ids: Int[Array, "batch seq_len"] | None = None,
        ):
            return self.run(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                seg_ids=seg_ids,
                initial_state=initial_state,
                use_qk_l2norm=use_qk_l2norm,
                use_chunked=use_chunked,
                return_state=return_state,
                platform=platform,
                cfg=cfg or GatedDeltaRuleConfig(chunk_size=64, platform="auto", backend="any"),
            )

        call_args = (
            query,
            key,
            value,
            beta,
            decay,
            initial_state,
        )

        if seg_ids is not None:
            call_args = (*call_args, seg_ids)

        assert len(in_specs) == len(call_args), (
            f"in_specs length {len(in_specs)} != call_args length {len(call_args)} "
            "(when seg_ids is provided, in_specs must include its sharding as the last entry)"
        )

        shard_map_fn = shard_map(
            _wrapped_gdr,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def get_impl(self, cfg: GatedDeltaRuleConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend.

        Returns:
            Callable kernel implementation for GDR.

        Raises:
            ValueError: If no matching implementation is found.
        """
        platform = detect_platform("gated_delta_rule", cfg.platform)
        return kernel_registry.get("gated_delta_rule", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_heads v_head_dim"],
        beta: Float[Array, "batch seq_len num_heads"],
        decay: Float[Array, "batch seq_len num_heads"] | None = None,
        initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
        seg_ids: Int[Array, "batch seq_len"] | None = None,
        *,
        use_qk_l2norm: bool = True,
        use_chunked: bool = True,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GatedDeltaRuleConfig,
        **_,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "batch num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute Gated Delta Rule operation.

        ``cfg.chunk_size`` takes precedence so the autotuner can sweep
        over chunk sizes transparently. ``use_qk_l2norm`` is read from
        the function call directly.

        Args:
            query: Query tensor for attention
                Shape: [batch, seq_len, num_heads, qk_head_dim]
            key: Key tensor for memory addressing
                Shape: [batch, seq_len, num_heads, qk_head_dim]
            value: Value tensor to store in memory
                Shape: [batch, seq_len, num_heads, v_head_dim]
            beta: Per-token gating for delta updates
                Shape: [batch, seq_len, num_heads]
            decay: Per-token decay for memory retention (should be <= 0)
                Shape: [batch, seq_len, num_heads]
            initial_state: Optional initial memory state
                Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            use_qk_l2norm: Apply L2 normalization to queries and keys
            use_chunked: Use chunked (True) or recurrent (False) algorithm
            return_state: If True, return (output, state) tuple
            platform: Override platform selection
            cfg: Kernel configuration object

        Returns:
            If return_state is False:
                output: Attention output [batch, seq_len, num_heads, v_head_dim]
            If return_state is True:
                Tuple of (output, final_state) where final_state has shape
                [batch, num_heads, qk_head_dim, v_head_dim]
        """
        if platform is not None:
            cfg = GatedDeltaRuleConfig(
                chunk_size=cfg.chunk_size,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        if not use_chunked:
            cfg.backend = Backend.ANY
            cfg.platform = "xla"

        impl = self.get_impl(cfg)
        out, final_state = impl(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            seg_ids=seg_ids,
            chunk_size=int(cfg.chunk_size),
            initial_state=initial_state,
            use_qk_l2norm=bool(use_qk_l2norm),
            use_chunked=use_chunked,
        )

        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[GatedDeltaRuleConfig, Array]) -> GatedDeltaRuleConfig:
        """Provide default configuration derived from invocation kwargs.

        Extracts ``chunk_size`` from the caller's keyword arguments so
        that the heuristic matches the user's intent.

        Args:
            inv: Invocation object with call metadata.

        Returns:
            Configuration matching the caller's chunk_size.
        """
        return GatedDeltaRuleConfig(chunk_size=256, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[GatedDeltaRuleConfig, Array]):
        """Generate candidate configurations for chunk-size autotuning.

        Produces configs with ``[chunk_size/2, chunk_size, chunk_size*2]``
        (or an explicit list from ``autotune_chunk_candidates`` kwarg /
        the ``EASYDEL_GDR_AUTOTUNE_CHUNK_CANDIDATES`` env var).

        Args:
            inv: Invocation object with call metadata.

        Returns:
            List of GatedDeltaRuleConfig candidates to benchmark.
        """

        cands = [128, 256]
        return [GatedDeltaRuleConfig(chunk_size=c, platform="auto", backend="any") for c in cands]

    def candidate_cfgs_gpu(self, inv: Invocation[GatedDeltaRuleConfig, Array]):
        """Generate GPU candidates for GDR across TileLang and XLA.

        ``chunk_size`` is the dominant tuning knob: smaller chunks raise
        inter-chunk state-propagation cost; larger chunks raise intra-chunk
        attention-matrix memory. Sweep {32, 64, 128, 256, 512} — the
        tilelang chunked path additionally tries a wider set on H100.
        """
        requested = inv.kwargs.get("platform", None)
        use_chunked = bool(inv.kwargs.get("use_chunked", True))
        chunk_choices = (32, 64, 128, 256, 512)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[GatedDeltaRuleConfig] = []
        if "tilelang" in platforms and use_chunked:
            for c in (64, 128, 256, 512):
                candidates.append(GatedDeltaRuleConfig(chunk_size=c, platform="tilelang", backend="gpu"))
        if "xla" in platforms:
            candidates.extend(GatedDeltaRuleConfig(chunk_size=c, platform="xla", backend="any") for c in chunk_choices)
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[GatedDeltaRuleConfig, Array]):
        """Generate TPU candidates for the Pallas and XLA GDR paths."""
        cands = [128, 256]
        return [
            GatedDeltaRuleConfig(chunk_size=c, platform=platform, backend=backend)
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for c in cands
        ]

    heuristic_cfg_shard_map = heuristic_cfg
    candidate_cfgs_shard_map = candidate_cfgs
    candidate_cfgs_shard_map_xla = candidate_cfgs
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[GatedDeltaRuleConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=10, iters=40),
        persistent=PersistentCache("gated_delta_rule"),
    )
)


def gated_delta_rule(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    /,
    *,
    autotune_chunk_candidates: tuple[int, ...] | list[int] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: GatedDeltaRuleConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "batch num_heads qk_head_dim v_head_dim"],
    ]
):
    """Execute Gated Delta Rule (GDR) with automatic optimization.

    GDR is a linear attention variant that combines gated delta rule updates
    with learnable decay for efficient sequence processing. It is used in
    hybrid transformer architectures like Qwen3Next.

    The core recurrence:
        h_t = exp(decay_t) * h_{t-1} + k_t (x) (beta_t * (v_t - h_{t-1} @ k_t))
        o_t = h_t @ q_t

    Args:
        query: Query tensor for attention
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        key: Key tensor for memory addressing
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        value: Value tensor to store in memory
            Shape: [batch, seq_len, num_heads, v_head_dim]
        beta: Per-token gating for delta updates (typically in [0, 1])
            Shape: [batch, seq_len, num_heads]
        decay: Per-token decay for memory retention (values should be <= 0).
            Shape: [batch, seq_len, num_heads].
            If None, defaults to zeros (no decay, full memory retention).
        initial_state: Optional initial memory state for incremental inference
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
        autotune_chunk_candidates: Optional list/tuple of chunk sizes to
            benchmark during autotuning, overriding ``candidate_cfgs``.
            When None, ``GatedDeltaRule.candidate_cfgs`` provides ``[128, 256]``
            unless the ``EASYDEL_GDR_AUTOTUNE_CHUNK_CANDIDATES`` env var is set.
        use_qk_l2norm: If True, apply L2 normalisation to queries and keys
            before attention (improves numerical stability; default: True).
        use_chunked: If True, use the chunked intra-chunk parallel algorithm
            (faster for training); if False, use the pure recurrent scan
            (more memory efficient but sequential; forces XLA platform).
        return_state: If True, return (output, final_state) tuple instead
            of just output.
        platform: Force a specific platform (``"triton"``, ``"pallas"``,
            ``"cuda"``, ``"xla"``, or ``"auto"``).  When ``use_chunked=False``,
            this is always overridden to ``"xla"``.
        cfg: Optional ``GatedDeltaRuleConfig`` override; ``chunk_size`` inside
            the config is the primary autotuned parameter.
        mesh: JAX device mesh for ``shard_map`` distributed execution.
        in_specs: Input partition specs for ``shard_map``; length must equal
            the number of positional args (query, key, value, beta, decay,
            initial_state).
        out_specs: Output partition spec for ``shard_map``; use a tuple when
            ``return_state=True``.

    Returns:
        If return_state is False:
            output: Attention output [batch, seq_len, num_heads, v_head_dim]
        If return_state is True:
            Tuple of:
                - output: Attention output [batch, seq_len, num_heads, v_head_dim]
                - final_state: Final memory state
                    [batch, num_heads, qk_head_dim, v_head_dim]

    Note:
        When ``seq_len == 1`` **and** ``initial_state is not None`` the function
        bypasses the executor entirely and calls the optimised single-step XLA
        kernel ``_single_step_gdr_fwd`` directly.  This fast path does not go
        through autotuning or config caching.

    Example:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Training forward pass
        >>> batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
        >>> key = random.PRNGKey(0)
        >>> q = random.normal(random.fold_in(key, 0), (batch, seq_len, num_heads, head_dim))
        >>> k = random.normal(random.fold_in(key, 1), (batch, seq_len, num_heads, head_dim))
        >>> v = random.normal(random.fold_in(key, 2), (batch, seq_len, num_heads, head_dim))
        >>> beta = jax.nn.sigmoid(random.normal(random.fold_in(key, 3), (batch, seq_len, num_heads)))
        >>> decay = random.normal(random.fold_in(key, 4), (batch, seq_len, num_heads)) * -0.01
        >>>
        >>> output = gated_delta_rule(q, k, v, beta, decay)
        >>> output.shape
        (2, 64, 8, 32)
        >>>
        >>> # Inference with state
        >>> output, state = gated_delta_rule(q, k, v, beta, decay, return_state=True)
        >>>
        >>> # Next token generation
        >>> q_new = random.normal(random.fold_in(key, 5), (batch, 1, num_heads, head_dim))
        >>> k_new = random.normal(random.fold_in(key, 6), (batch, 1, num_heads, head_dim))
        >>> v_new = random.normal(random.fold_in(key, 7), (batch, 1, num_heads, head_dim))
        >>> beta_new = jax.nn.sigmoid(random.normal(random.fold_in(key, 8), (batch, 1, num_heads)))
        >>> decay_new = random.normal(random.fold_in(key, 9), (batch, 1, num_heads)) * -0.01
        >>>
        >>> output_new, state_new = gated_delta_rule(
        ...     q_new, k_new, v_new, beta_new, decay_new,
        ...     initial_state=state, return_state=True,
        ... )
    """
    if query.shape[1] == 1 and initial_state is not None:
        from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import (
            _single_step_gdr_fwd,
        )

        q = query.transpose(0, 2, 1, 3)
        k = key.transpose(0, 2, 1, 3)
        v = value.transpose(0, 2, 1, 3)
        b = beta.transpose(0, 2, 1)
        d = decay.transpose(0, 2, 1) if decay is not None else None
        out, final_state = _single_step_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            recurrent_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )
        out = out.transpose(0, 2, 1, 3)
        if return_state:
            return out, final_state
        return out

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _executor(
        GatedDeltaRule(),
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        seg_ids=seg_ids,
        initial_state=initial_state,
        autotune_chunk_candidates=autotune_chunk_candidates,
        use_qk_l2norm=use_qk_l2norm,
        use_chunked=use_chunked,
        return_state=return_state,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
