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

"""Public grouped GDR single-step decode operation.

This module owns the eJKernel operation wrapper for the grouped-head
variant of the Gated Delta Rule decode step. The XLA backend carries the
canonical JAX reference; the TPU Pallas backend provides a VMEM-friendly
alternative suitable for tensor-parallel head sharding.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.modules.base import mesh_to_jax_mesh
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
from .configs import GatedDeltaRuleGroupedDecodeConfig


class GatedDeltaRuleGroupedDecode(Kernel[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):
    """Executor-driven wrapper for grouped GDR single-step decode.

    Implements the :class:`~ejkernel.ops.Kernel` protocol for the grouped-head
    Gated Delta Rule decode step, where multiple value heads (``expand_ratio``)
    share each key head. ``run`` dispatches to the registered backend and falls
    back to XLA when the Pallas-required ``decay`` is missing;
    ``create_shard_map_wrapper`` builds the tensor-parallel head-sharded path.

    Attributes:
        version: Cache-busting version tag for the persistent config cache.
    """

    version = "1"

    def __init__(self):
        """Initialize the grouped GDR decode kernel.

        Registers the operation under ``op_id="gated_delta_rule_grouped_decode"``
        for registry and persistent-cache lookups.
        """
        super().__init__(op_id="gated_delta_rule_grouped_decode")

    def get_impl(self, cfg: GatedDeltaRuleGroupedDecodeConfig):
        """Resolve the backend implementation for a config.

        Args:
            cfg: Operation config carrying the requested ``platform`` and
                ``backend``.

        Returns:
            The registered callable implementing
            ``gated_delta_rule_grouped_decode`` for the detected platform and
            the config's backend.
        """
        platform = detect_platform("gated_delta_rule_grouped_decode", cfg.platform)
        return kernel_registry.get("gated_delta_rule_grouped_decode", platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
        *,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GatedDeltaRuleGroupedDecodeConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: tuple[PartitionSpec | None, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a ``shard_map`` wrapper for TP head-sharded grouped decode.

        Builds a :func:`jax.shard_map`-wrapped closure over :meth:`run` so the
        executor can run the grouped decode with head/value axes partitioned
        across the mesh.

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor, shape
                ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients, shape
                ``[batch, num_k_heads, expand_ratio]``.
            decay: Optional log-space decay, shape
                ``[batch, num_k_heads, expand_ratio]``; ``None`` forces the XLA
                fallback inside :meth:`run`.
            recurrent_state: Current recurrent state, shape
                ``[batch, num_v_heads, head_dim, value_dim]``.
            platform: Optional implementation-family override forwarded to
                :meth:`run`. Defaults to ``None``.
            cfg: Optional operation config; when ``None`` a default config is
                built from ``platform`` (or ``"auto"``). Defaults to ``None``.
            mesh: Mesh used by :func:`jax.shard_map`. Must not be ``None``.
            in_specs: Partition specs for ``(query, key, value, beta, decay,
                recurrent_state)``; length must match the call args. Must not be
                ``None``.
            out_specs: Partition specs for the ``(output, new_state)`` outputs.
                Must not be ``None``.
            check_vma: Forwarded to :func:`jax.shard_map`. Defaults to ``False``.

        Returns:
            Tuple ``(shard_map_fn, call_args)`` where ``shard_map_fn`` is the
            sharded callable and ``call_args`` is the positional argument tuple
            to invoke it with.

        Raises:
            AssertionError: If ``mesh``, ``in_specs``, or ``out_specs`` is
                ``None``, or if ``len(in_specs)`` does not match the number of
                call arguments.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        run_cfg = cfg or GatedDeltaRuleGroupedDecodeConfig(platform=platform or "auto")
        call_args = (query, key, value, beta, decay, recurrent_state)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        def _wrapped(q, k, v, b, d, s):
            """Run grouped GDR decode on a per-shard slice of the inputs.

            Args:
                q: Per-shard query slice.
                k: Per-shard key slice.
                v: Per-shard value slice.
                b: Per-shard beta slice.
                d: Per-shard decay slice (or ``None``).
                s: Per-shard recurrent-state slice.

            Returns:
                Tuple ``(output, new_state)`` for this shard.
            """
            return self.run(
                q,
                k,
                v,
                b,
                d,
                s,
                platform=platform,
                cfg=run_cfg,
            )

        shard_map_fn = jax.shard_map(
            _wrapped,
            mesh=mesh_to_jax_mesh(mesh),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def run(
        self,
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
        *,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GatedDeltaRuleGroupedDecodeConfig,
        **_,
    ) -> tuple[
        Float[Array, "batch num_v_heads value_dim"],
        Float[Array, "batch num_v_heads head_dim value_dim"],
    ]:
        """Run the selected grouped GDR decode backend implementation.

        Applies any ``platform`` override, then handles the optional ``decay``:
        the Pallas kernel requires an explicit decay, so ``decay=None`` with a
        Pallas config falls back to the XLA reference, while a non-Pallas backend
        with ``decay=None`` is given a zero decay (an effective no-decay step).

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor, shape
                ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients, shape
                ``[batch, num_k_heads, expand_ratio]``.
            decay: Optional log-space decay, shape
                ``[batch, num_k_heads, expand_ratio]``; ``None`` triggers the
                fallback behavior described above.
            recurrent_state: Current recurrent state, shape
                ``[batch, num_v_heads, head_dim, value_dim]``.
            platform: Optional implementation-family override. When given, a new
                config is built with this platform and ``backend=ANY`` for
                ``"xla"`` (else the existing backend). Defaults to ``None``.
            cfg: Operation config selected by the executor.
            **_: Ignored extra keyword arguments forwarded by the executor.

        Returns:
            Tuple ``(output, new_state)`` of shapes
            ``[batch, num_v_heads, value_dim]`` and
            ``[batch, num_v_heads, head_dim, value_dim]``.
        """
        if platform is not None:
            cfg = GatedDeltaRuleGroupedDecodeConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        # Pallas kernel requires explicit decay; fall back to XLA reference if None.
        if decay is None and cfg.platform == "pallas":
            cfg = GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any")

        impl = self.get_impl(cfg)
        if decay is None and cfg.platform != "pallas":
            decay = jnp.zeros_like(beta)

        return impl(query, key, value, beta, decay, recurrent_state)

    def heuristic_cfg(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):
        """Return the default XLA config for an invocation.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A :class:`GatedDeltaRuleGroupedDecodeConfig` on ``platform="xla"``
            with ``backend="any"``.
        """
        return GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):
        """Return the default candidate config list (heuristic only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):
        """Return GPU autotuning candidates (heuristic XLA only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):
        """Return TPU autotuning candidates (XLA reference plus Pallas).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A two-element list: the XLA reference config followed by the
            Pallas/TPU config.
        """
        return [
            GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any"),
            GatedDeltaRuleGroupedDecodeConfig(platform="pallas", backend="tpu"),
        ]

    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs
    heuristic_cfg_shard_map = heuristic_cfg
    candidate_cfgs_shard_map = candidate_cfgs
    candidate_cfgs_shard_map_xla = candidate_cfgs
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("gated_delta_rule_grouped_decode"),
    )
)


def gated_delta_rule_grouped_decode(
    query: Float[Array, "batch num_k_heads head_dim"],
    key: Float[Array, "batch num_k_heads head_dim"],
    value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
    beta: Float[Array, "batch num_k_heads expand_ratio"],
    decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
    recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    *,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: GatedDeltaRuleGroupedDecodeConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec | None, ...] | None = None,
    check_vma: bool = False,
) -> tuple[
    Float[Array, "batch num_v_heads value_dim"],
    Float[Array, "batch num_v_heads head_dim value_dim"],
]:
    """Execute grouped GDR decode through the eJKernel operation stack.

    Args:
        query: Query tensor ``[batch, num_k_heads, head_dim]``.
        key: Key tensor ``[batch, num_k_heads, head_dim]``.
        value: Value tensor ``[batch, num_k_heads, expand_ratio, value_dim]``.
        beta: Gating coefficients ``[batch, num_k_heads, expand_ratio]``.
        decay: Optional log-space decay ``[batch, num_k_heads, expand_ratio]``.
        recurrent_state: Current recurrent state
            ``[batch, num_v_heads, head_dim, value_dim]``.
        platform: Optional implementation-family override.
        cfg: Optional operation config.
        mesh: Optional mesh for ``shard_map`` execution.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition specs for ``shard_map``.
        check_vma: Forwarded to ``jax.shard_map``.

    Returns:
        ``(output, new_state)``.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    if cfg is None:
        cfg = GatedDeltaRuleGroupedDecodeConfig(platform=platform or "auto")

    return _executor(
        GatedDeltaRuleGroupedDecode(),
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )


gated_delta_rule_grouped_decode_op = gated_delta_rule_grouped_decode
