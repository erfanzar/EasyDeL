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

"""Ragged Gated Delta Rule operation module with automatic platform selection.

Provides the ``RaggedGatedDeltaRule`` operation class and the public
``ragged_gated_delta_rule`` function for processing variable-length
sequences packed into a flat token stream. This is the primary interface
for continuous-batching inference engines that need to process multiple
requests with different sequence lengths in a single kernel call.

The operation supports two execution paths:
    - **Decode** (all sequences length 1): Parallel per-token updates.
      On TPU, uses a Pallas kernel for up to 3.7x speedup over XLA.
    - **Prefill** (variable-length sequences): Chunked computation with
      triangular-solve inversion and inter-chunk state propagation.

Example:
    >>> from ejkernel.modules.operations import ragged_gated_delta_rule
    >>>
    >>> output, updated_state = ragged_gated_delta_rule(
    ...     query, key, value, beta, decay,
    ...     recurrent_state=state_pool,
    ...     query_start_loc=jnp.array([0, 1, 33, 34]),
    ...     state_indices=jnp.array([0, 1, 2]),
    ... )
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
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
from .configs import RaggedGatedDeltaRuleConfig


class RaggedGatedDeltaRule(Kernel[RaggedGatedDeltaRuleConfig, Array]):
    """Ragged Gated Delta Rule operation for packed continuous-batching inference.

    Processes flat token streams where variable-length sequences are
    identified by cumulative offsets (``query_start_loc``). Each active
    request owns a slot in a global recurrent state pool, indexed by
    ``state_indices``.

    Unlike the standard ``GatedDeltaRule`` operation which expects
    ``[batch, seq_len, num_heads, dim]`` inputs, this operation accepts
    flat ``[num_tokens, num_heads, dim]`` tensors — the natural format
    for continuous-batching engines.

    The operation automatically selects between:
        - **XLA**: Pure JAX implementation (CPU/GPU/TPU).
        - **Pallas TPU**: Optimized Pallas kernel for decode (TPU only,
          up to 3.7x faster than XLA for decode workloads).

    Attributes:
        op_id: Operation identifier (``"ragged_gated_delta_rule"``).
        version: Schema version for configuration compatibility.
    """

    version = "1"

    def __init__(self):
        """Initialize the ragged GDR operation with registry identifier."""
        super().__init__(op_id="ragged_gated_delta_rule")

    def get_impl(self, cfg: RaggedGatedDeltaRuleConfig):
        """Retrieve the kernel implementation from the registry.

        Resolves the platform (auto-detecting TPU for Pallas, falling
        back to XLA) and looks up the registered implementation.

        Args:
            cfg: Configuration specifying platform and backend preferences.

        Returns:
            A callable kernel implementation.

        Raises:
            ValueError: If no matching implementation is found.
        """
        platform = detect_platform("ragged_gated_delta_rule", cfg.platform)
        return kernel_registry.get("ragged_gated_delta_rule", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_tokens num_heads qk_head_dim"],
        key: Float[Array, "num_tokens num_heads qk_head_dim"],
        value: Float[Array, "num_tokens num_heads v_head_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads"] | None,
        recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
        query_start_loc: Int[Array, "num_requests_plus_1"],
        state_indices: Int[Array, "num_requests"],
        *,
        use_qk_l2norm: bool = True,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedGatedDeltaRuleConfig,
        **_,
    ) -> tuple[
        Float[Array, "num_tokens num_heads v_head_dim"],
        Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
    ]:
        """Execute ragged GDR.

        Args:
            query: Flat queries (num_tokens, num_heads, qk_head_dim).
            key: Flat keys (num_tokens, num_heads, qk_head_dim).
            value: Flat values (num_tokens, num_heads, v_head_dim).
            beta: Gating (num_tokens, num_heads).
            decay: Log-space decay (num_tokens, num_heads) or None.
            recurrent_state: Global state pool (num_slots, H, d_k, d_v).
            query_start_loc: Cumulative offsets (num_requests + 1,).
            state_indices: Request-to-slot mapping (num_requests,).
            use_qk_l2norm: L2-normalize queries and keys.
            platform: Override platform.
            cfg: Configuration.

        Returns:
            (output, updated_recurrent_state)
        """
        if platform is not None:
            cfg = RaggedGatedDeltaRuleConfig(
                chunk_size=cfg.chunk_size,
                platform=platform,
                backend=cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            chunk_size=int(cfg.chunk_size),
            use_qk_l2norm=bool(use_qk_l2norm),
        )

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "num_tokens num_heads qk_head_dim"],
        key: Float[Array, "num_tokens num_heads qk_head_dim"],
        value: Float[Array, "num_tokens num_heads v_head_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads"] | None,
        recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
        query_start_loc: Int[Array, "num_requests_plus_1"],
        state_indices: Int[Array, "num_requests"],
        *,
        use_qk_l2norm: bool = True,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedGatedDeltaRuleConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed ragged GDR.

        Wraps the ragged GDR call in ``jax.shard_map`` so the Pallas kernel
        runs independently on each device shard. Required for TPU Pallas
        kernels which cannot be automatically partitioned.

        Args:
            query: Flat queries (num_tokens, H, d_k).
            key: Flat keys (num_tokens, H, d_k).
            value: Flat values (num_tokens, H, d_v).
            beta: Gating (num_tokens, H).
            decay: Log-space decay (num_tokens, H) or None.
            recurrent_state: State pool (num_slots, H, d_k, d_v).
            query_start_loc: Cumulative offsets (num_requests + 1,).
            state_indices: Request-to-slot mapping (num_requests,).
            use_qk_l2norm: L2-normalize queries and keys.
            platform: Override platform.
            cfg: Configuration.
            mesh: JAX device mesh for sharding.
            in_specs: PartitionSpec per input tensor.
            out_specs: PartitionSpec per output tensor.
            check_vma: Check virtual memory alignment.

        Returns:
            (shard_map_fn, call_args) ready to execute.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        _platform = platform
        _cfg = cfg
        _use_qk_l2norm = use_qk_l2norm

        def _wrapped(q, k, v, b, d, s, qsl, si):
            return self.run(
                query=q,
                key=k,
                value=v,
                beta=b,
                decay=d,
                recurrent_state=s,
                query_start_loc=qsl,
                state_indices=si,
                use_qk_l2norm=_use_qk_l2norm,
                platform=_platform,
                cfg=_cfg or RaggedGatedDeltaRuleConfig(),
            )

        import jax.numpy as jnp

        if decay is None:
            decay = jnp.zeros_like(beta)

        call_args = (query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        shard_map_fn = shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def heuristic_cfg(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array]) -> RaggedGatedDeltaRuleConfig:
        """Return the default configuration for ragged GDR.

        Uses ``chunk_size=64`` which provides a good balance between
        intra-chunk parallelism and inter-chunk state propagation
        overhead for typical Qwen3Next workloads.

        Args:
            inv: Invocation metadata (unused, present for API compat).

        Returns:
            Default configuration with chunk_size=64.
        """
        return RaggedGatedDeltaRuleConfig(chunk_size=64, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array]):
        """Generate candidate configurations for autotuning.

        Produces configs with chunk sizes [32, 64, 128] for the
        autotuner to evaluate.

        Args:
            inv: Invocation metadata (unused, present for API compat).

        Returns:
            List of candidate configurations.
        """
        return [RaggedGatedDeltaRuleConfig(chunk_size=c, platform="auto", backend="any") for c in [32, 64, 128]]

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array]):
        """Generate GPU candidates for ragged GDR across TileLang and XLA.

        Ragged variant operates on packed variable-length sequences;
        smaller chunks (32, 64) work well for short sequences in the pack,
        larger chunks (128, 256) for longer ones.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        chunk_choices = (32, 64, 128, 256)
        candidates: list[RaggedGatedDeltaRuleConfig] = []
        if "tilelang" in platforms:
            for c in (32, 64, 128):
                candidates.append(RaggedGatedDeltaRuleConfig(chunk_size=c, platform="tilelang", backend="gpu"))
        if "xla" in platforms:
            candidates.extend(
                RaggedGatedDeltaRuleConfig(chunk_size=c, platform="xla", backend="any") for c in chunk_choices
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array]):
        """Generate TPU candidates for ragged GDR across Pallas and XLA."""
        return [
            RaggedGatedDeltaRuleConfig(chunk_size=c, platform=platform, backend=backend)
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for c in [32, 64, 128]
        ]

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[RaggedGatedDeltaRuleConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=10),
        persistent=PersistentCache("ragged_gated_delta_rule"),
    )
)


def ragged_gated_delta_rule(
    query: Float[Array, "num_tokens num_heads qk_head_dim"],
    key: Float[Array, "num_tokens num_heads qk_head_dim"],
    value: Float[Array, "num_tokens num_heads v_head_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads"] | None = None,
    recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"] | None = None,
    query_start_loc: Int[Array, "num_requests_plus_1"] | None = None,
    state_indices: Int[Array, "num_requests"] | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedGatedDeltaRuleConfig | None = None,
) -> tuple[
    Float[Array, "num_tokens num_heads v_head_dim"],
    Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
]:
    """Ragged Gated Delta Rule for packed continuous-batching inference.

    Processes variable-length sequences packed into a flat token stream.
    Each request is identified by a ``[query_start_loc[i], query_start_loc[i+1])``
    slice of the token dimension, and reads/writes recurrent state from/to slot
    ``state_indices[i]`` in the global ``recurrent_state`` pool.

    Execution paths:
        - **Decode** (all sequences have length 1): Each token performs a
          single-step state update. On TPU, routes to the Pallas kernel for
          up to 3.7x speedup over XLA.
        - **Prefill** (at least one sequence has length > 1): Chunked
          parallel forward pass with triangular-solve inversion and
          inter-chunk state propagation.

    The gated delta rule recurrence per token t:
        h_t = decay_t * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
        o_t = h_t @ q_t

    Args:
        query: Flat queries, shape ``(num_tokens, num_heads, qk_head_dim)``.
        key: Flat keys, shape ``(num_tokens, num_heads, qk_head_dim)``.
        value: Flat values, shape ``(num_tokens, num_heads, v_head_dim)``.
        beta: Per-token gating coefficient, shape ``(num_tokens, num_heads)``.
            Typically in ``[0, 1]``; controls how much of the residual is
            written into the state.
        decay: Per-token log-space decay, shape ``(num_tokens, num_heads)``,
            or ``None`` (interpreted as zeros, i.e., no decay). Should be
            ``<= 0`` for stable memory retention.
        recurrent_state: Global recurrent state pool,
            shape ``(num_slots, num_heads, qk_head_dim, v_head_dim)``.
            Slot assignment is specified via ``state_indices``.
        query_start_loc: Cumulative token offsets marking sequence boundaries,
            shape ``(num_requests + 1,)``. ``query_start_loc[0]`` must be 0
            and ``query_start_loc[-1]`` must equal ``num_tokens``.
        state_indices: Maps each request index to its recurrent state slot,
            shape ``(num_requests,)``.
        chunk_size: Chunk size used for the prefill parallel scan (default: 64).
            Ignored for decode-only batches. Larger values increase intra-chunk
            parallelism at the cost of higher memory for the chunk attention matrix.
            Typical values: 32, 64, 128.
        use_qk_l2norm: If ``True``, L2-normalise queries and keys before the
            delta update (default: ``True``). Improves numerical stability.
        platform: Override automatic platform selection. Useful to force XLA
            for debugging or to force Pallas for profiling.
        cfg: Optional :class:`~.configs.RaggedGatedDeltaRuleConfig` override.
            When provided, its ``chunk_size`` takes precedence over the
            ``chunk_size`` argument.

    Returns:
        Tuple of:
            - ``output``: Attention output, same flat layout as the inputs,
              shape ``(num_tokens, num_heads, v_head_dim)``.
            - ``updated_recurrent_state``: State pool with each request's
              final state written back, shape identical to ``recurrent_state``.

    Note:
        Autotuning is disabled for this operation (``allow_autotune=False``).
        The heuristic ``chunk_size=64`` is used unless a ``cfg`` is explicitly
        supplied.
    """
    if cfg is None:
        cfg = RaggedGatedDeltaRuleConfig(
            chunk_size=chunk_size,
            platform=platform or "auto",
            backend="any",
        )

    return _executor(
        RaggedGatedDeltaRule(),
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        use_qk_l2norm=use_qk_l2norm,
        platform=platform,
        _cfg=cfg,
    )
