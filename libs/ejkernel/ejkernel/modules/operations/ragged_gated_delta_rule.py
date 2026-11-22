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

from typing import Literal

import jax
import jax.numpy as jnp
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
        """Create a ``shard_map`` wrapper for distributed ragged GDR.

        Wraps the ragged GDR call in ``jax.shard_map`` so the kernel runs
        independently on each device shard. Required for TPU Pallas kernels which
        cannot be automatically partitioned.

        Two strategies are produced depending on the resolved platform:

        - **TPU Pallas split** (selected when the platform resolves to
          ``"pallas"``/``"auto"``/``None`` and the default JAX backend is TPU):
          separate sharded decode and prefill functions are built, and the
          returned callable (``_dispatch``) chooses between them at run time with
          ``jax.lax.cond`` based on whether every request has length <= 1. Here
          ``decay`` defaults to zeros when None.
        - **Generic path** (all other cases): a single ``self.run`` call is
          wrapped with ``jax.shard_map``.

        In both cases the positional argument tuple is
        ``(query, key, value, beta, decay, recurrent_state, query_start_loc,
        state_indices)`` and ``in_specs`` must have the same length.

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
            check_vma: Forwarded to ``jax.shard_map`` (varying-manual-axis check).

        Returns:
            A tuple ``(fn, call_args)`` where ``fn`` is the executable callable
            (the ``_dispatch`` chooser for the TPU Pallas split, or the wrapped
            ``self.run`` for the generic path) and ``call_args`` is the positional
            argument tuple to invoke it with.

        Raises:
            AssertionError: If ``mesh``, ``in_specs`` or ``out_specs`` is None, or
                if ``len(in_specs)`` does not equal the number of call arguments.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        selected_platform = platform or (cfg.platform if cfg is not None else "auto")
        selected_platform = getattr(selected_platform, "value", selected_platform)
        use_tpu_pallas_split = selected_platform in ("pallas", "auto", None) and jax.default_backend() == "tpu"

        if decay is None:
            decay = jnp.zeros_like(beta)

        call_args = (query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        if use_tpu_pallas_split:
            from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule._interface import _decode_path
            from ejkernel.kernels._xla.ragged_gated_delta_rule._xla_impl_fwd import _ragged_gdr_chunked_prefill

            chunk_size = int((cfg or RaggedGatedDeltaRuleConfig()).chunk_size)

            @jax.named_scope("ragged_gdr_decode_shard_map")
            def _decode_shard(q, k, v, b, d, s, si):
                """Run the per-shard decode (length-1) path via the TPU Pallas kernel.

                Captures ``use_qk_l2norm`` from the enclosing scope. Note that the
                ``query_start_loc`` argument is intentionally omitted: the decode
                path treats every token as a single-step update.

                Args:
                    q: Per-shard flat queries, shape ``(num_tokens, H, d_k)``.
                    k: Per-shard flat keys, shape ``(num_tokens, H, d_k)``.
                    v: Per-shard flat values, shape ``(num_tokens, H, d_v)``.
                    b: Per-shard gate, shape ``(num_tokens, H)``.
                    d: Per-shard log-space decay, shape ``(num_tokens, H)``.
                    s: Per-shard recurrent state pool, shape ``(num_slots, H, d_k, d_v)``.
                    si: Per-token state-slot indices (padded/truncated to
                        ``num_tokens`` by the dispatcher).

                Returns:
                    A tuple ``(output, updated_recurrent_state)``.
                """
                return _decode_path(q, k, v, b, d, s, si, use_qk_l2norm)

            decode_shard_fn = shard_map(
                _decode_shard,
                mesh=mesh,
                in_specs=(
                    in_specs[0],
                    in_specs[1],
                    in_specs[2],
                    in_specs[3],
                    in_specs[4],
                    in_specs[5],
                    in_specs[7],
                ),
                out_specs=out_specs,
                check_vma=check_vma,
            )

            @jax.named_scope("ragged_gdr_prefill_shard_map")
            def _prefill_shard(q, k, v, b, d, s, qsl, si):
                """Run the per-shard prefill (variable-length) path via the XLA chunked kernel.

                Captures ``chunk_size`` and ``use_qk_l2norm`` from the enclosing
                scope. ``_ragged_gdr_chunked_prefill`` returns ``(new_state, out)``;
                this wrapper reorders them to the public ``(out, new_state)``
                convention.

                Args:
                    q: Per-shard flat queries, shape ``(num_tokens, H, d_k)``.
                    k: Per-shard flat keys, shape ``(num_tokens, H, d_k)``.
                    v: Per-shard flat values, shape ``(num_tokens, H, d_v)``.
                    b: Per-shard gate, shape ``(num_tokens, H)``.
                    d: Per-shard log-space decay, shape ``(num_tokens, H)``.
                    s: Per-shard recurrent state pool, shape ``(num_slots, H, d_k, d_v)``.
                    qsl: Cumulative token offsets, shape ``(num_requests + 1,)``.
                    si: Request-to-slot mapping, shape ``(num_requests,)``.

                Returns:
                    A tuple ``(output, updated_recurrent_state)``.
                """
                new_state, out = _ragged_gdr_chunked_prefill(
                    q,
                    k,
                    v,
                    b,
                    d,
                    s,
                    qsl,
                    si,
                    chunk_size,
                    use_qk_l2norm,
                )
                return out, new_state

            prefill_shard_fn = shard_map(
                _prefill_shard,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
            )

            def _dispatch(q, k, v, b, d, s, qsl, si):
                """Select decode vs prefill at trace time and run the matching sharded path.

                Computes per-request sequence lengths from ``qsl`` and uses
                ``jax.lax.cond`` to choose between the decode and prefill sharded
                functions. For the decode branch the state-index array is
                padded with zeros (or truncated) so its length matches
                ``num_tokens``, since the decode kernel expects one slot index per
                token.

                Args:
                    q: Flat queries, shape ``(num_tokens, H, d_k)``.
                    k: Flat keys, shape ``(num_tokens, H, d_k)``.
                    v: Flat values, shape ``(num_tokens, H, d_v)``.
                    b: Gate, shape ``(num_tokens, H)``.
                    d: Log-space decay, shape ``(num_tokens, H)``.
                    s: Recurrent state pool, shape ``(num_slots, H, d_k, d_v)``.
                    qsl: Cumulative token offsets, shape ``(num_requests + 1,)``.
                    si: Request-to-slot mapping, shape ``(num_requests,)``.

                Returns:
                    A tuple ``(output, updated_recurrent_state)`` from whichever
                    branch is taken (decode when every request has length <= 1,
                    otherwise prefill).
                """
                seq_lengths = qsl[1:] - qsl[:-1]
                is_all_decode = jnp.all(seq_lengths <= 1)

                num_tokens = q.shape[0]
                num_indices = si.shape[0]
                if num_tokens > num_indices:
                    decode_state_indices = jnp.pad(si, (0, num_tokens - num_indices))
                elif num_tokens < num_indices:
                    decode_state_indices = si[:num_tokens]
                else:
                    decode_state_indices = si

                def _run_decode(_):
                    """``lax.cond`` true-branch: invoke the sharded decode function.

                    Args:
                        _: Unused ``operand`` placeholder required by ``lax.cond``.

                    Returns:
                        ``(output, updated_recurrent_state)`` from the decode path.
                    """
                    return decode_shard_fn(q, k, v, b, d, s, decode_state_indices)

                def _run_prefill(_):
                    """``lax.cond`` false-branch: invoke the sharded prefill function.

                    Args:
                        _: Unused ``operand`` placeholder required by ``lax.cond``.

                    Returns:
                        ``(output, updated_recurrent_state)`` from the prefill path.
                    """
                    return prefill_shard_fn(q, k, v, b, d, s, qsl, si)

                return jax.lax.cond(is_all_decode, _run_decode, _run_prefill, operand=None)

            return _dispatch, call_args

        _platform = platform
        _cfg = cfg
        _use_qk_l2norm = use_qk_l2norm

        def _wrapped(q, k, v, b, d, s, qsl, si):
            """Run ragged GDR on a single shard via ``self.run`` (non-TPU-Pallas path).

            Captures ``_platform``, ``_cfg`` and ``_use_qk_l2norm`` from the
            enclosing scope. Used when the TPU Pallas decode/prefill split is not
            selected, so a single registry implementation handles the shard.

            Args:
                q: Per-shard flat queries, shape ``(num_tokens, H, d_k)``.
                k: Per-shard flat keys, shape ``(num_tokens, H, d_k)``.
                v: Per-shard flat values, shape ``(num_tokens, H, d_v)``.
                b: Per-shard gate, shape ``(num_tokens, H)``.
                d: Per-shard log-space decay, shape ``(num_tokens, H)``.
                s: Per-shard recurrent state pool, shape ``(num_slots, H, d_k, d_v)``.
                qsl: Cumulative token offsets, shape ``(num_requests + 1,)``.
                si: Request-to-slot mapping, shape ``(num_requests,)``.

            Returns:
                A tuple ``(output, updated_recurrent_state)``.
            """
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
        """Generate GPU autotuning candidates for ragged GDR across TileLang and XLA.

        The ragged variant operates on packed variable-length sequences; smaller
        chunks (32, 64) work well for short sequences in the pack and larger
        chunks (128, 256) for longer ones. The TileLang path sweeps
        ``{32, 64, 128}`` and the XLA path sweeps ``{32, 64, 128, 256}``.

        The set of platforms considered is ``("tilelang", "xla")`` when the
        invocation's ``platform`` kwarg is ``None`` or ``"auto"``, otherwise just
        the explicitly requested platform.

        Args:
            inv: Invocation object whose ``kwargs`` are read for ``"platform"``
                (default ``None``).

        Returns:
            A list of ``RaggedGatedDeltaRuleConfig`` candidates. Falls back to
            ``self.candidate_cfgs(inv)`` if no GPU-specific candidate matches the
            requested platform.
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
        """Generate TPU autotuning candidates for ragged GDR across Pallas and XLA.

        Produces the cross product of chunk sizes ``{32, 64, 128}`` with the two
        TPU execution paths: ``(platform="pallas", backend="tpu")`` and
        ``(platform="xla", backend="any")`` — six candidates total.

        Args:
            inv: Invocation metadata (unused, present for API compat).

        Returns:
            A six-element list of ``RaggedGatedDeltaRuleConfig`` candidates.
        """
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
            cache_miss_fallback="heuristics",
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
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None,
    check_vma: bool = False,
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

    The gated delta rule recurrence per token t (decay is in log space, so the
    state multiplier is ``exp(decay_t)``):
        h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - exp(decay_t) * h_{t-1} @ k_t))
        o_t = exp(decay_t) * (h_{t-1} @ q_t) + (q_t · k_t) * v_new_t

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
        mesh: Optional JAX mesh for ``shard_map`` distributed execution.
        in_specs: Input partition specs for ``shard_map``. When provided with
            ``mesh`` and ``out_specs``, must cover ``query``, ``key``,
            ``value``, ``beta``, ``decay``, ``recurrent_state``,
            ``query_start_loc``, and ``state_indices``.
        out_specs: Output partition specs for ``shard_map`` as
            ``(output_spec, recurrent_state_spec)``.
        check_vma: Passed through to ``jax.shard_map``.

    Returns:
        Tuple of:
            - ``output``: Attention output, same flat layout as the inputs,
              shape ``(num_tokens, num_heads, v_head_dim)``.
            - ``updated_recurrent_state``: State pool with each request's
              final state written back, shape identical to ``recurrent_state``.

    Note:
        The module executor enables autotuning (``allow_autotune=True``) with a
        light tuner (3 warmup / 10 timed iterations) and a persistent cache, but
        backward validation is disabled. Cache misses use the heuristic config
        unless the caller provides an explicit ``cfg`` override.
    """
    if cfg is None:
        cfg = RaggedGatedDeltaRuleConfig(
            chunk_size=chunk_size,
            platform=platform or "auto",
            backend="any",
        )

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

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
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )
