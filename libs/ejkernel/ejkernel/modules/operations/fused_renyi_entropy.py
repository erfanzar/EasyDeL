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

"""Fused per-row Renyi entropy with platform & sharding auto-dispatch.

``H_alpha(softmax(z / T))`` per row, computed from two logsumexps
(``log sum p^alpha = LSE(alpha * z~) - alpha * LSE(z~)``) without materializing
the softmax. ``alpha == 1.0`` selects the exact Shannon branch; ``alpha = 2`` is
collision entropy; ``alpha -> inf`` approaches min-entropy ``-log(max_v p_v)``.

Differentiable in the logits (``stop_gradient`` outside for detached
diagnostics). When ``mesh`` / ``in_specs`` / ``out_specs`` are provided the call
is wrapped in :func:`jax.shard_map` with vocab parallelism inferred from the last
entry of the logits' partition spec, the same contract as
:func:`~ejkernel.modules.operations.fused_kl_divergence`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

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
from ._row_loss_shard_map import build_row_loss_shard_map, resolve_shard_map_method
from .configs import FusedRenyiEntropyConfig

PlatformName = Literal["xla", "auto"] | str


class RenyiEntropyOutput(NamedTuple):
    """Per-call metrics returned by :func:`fused_renyi_entropy`.

    Attributes:
        entropy: Differentiable scalar (``reduction in {"mean", "sum"}``) or
            ``logits.shape[:-1]`` array (``reduction == "none"``).
        weight_sum: Sum of the per-row weights (denominator of the ``"mean"``
            reduction), detached. ``entropy * weight_sum`` reconstructs the
            weighted SUM for ``"mean"``.
    """

    entropy: Array
    weight_sum: Array


class FusedRenyiEntropy(Kernel[FusedRenyiEntropyConfig, Array]):
    """Fused per-row Renyi entropy with platform + sharding auto-dispatch."""

    def __init__(self):
        """Create the operation object bound to the registry op id."""
        super().__init__(op_id="fused_renyi_entropy")

    def get_impl(self, cfg: FusedRenyiEntropyConfig):
        """Resolve the concrete backend implementation for ``cfg``."""
        platform = detect_platform("fused_renyi_entropy", cfg.platform)
        return kernel_registry.get("fused_renyi_entropy", platform=platform, backend=cfg.backend)

    def run(
        self,
        logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        alpha: float = 1.0,
        reduction: str = "mean",
        temperature: float = 1.0,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedRenyiEntropyConfig,
    ) -> Float[Array, "..."]:
        """Run the registered backend; ``attention_mask`` is multiplied into ``weights``."""
        if attention_mask is not None:
            mask_f32 = attention_mask.astype(jnp.float32)
            weights = mask_f32 if weights is None else weights.astype(jnp.float32) * mask_f32
        if platform is not None:
            cfg = FusedRenyiEntropyConfig(platform=platform, backend="any" if platform == "xla" else cfg.backend)
        return self.get_impl(cfg)(
            logits,
            weights,
            alpha=alpha,
            reduction=reduction,
            temperature=temperature,
            vocab_parallel_axis=vocab_parallel_axis,
        )

    def create_shard_map_wrapper(
        self,
        logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        alpha: float = 1.0,
        reduction: str = "mean",
        temperature: float = 1.0,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedRenyiEntropyConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = True,
    ):
        """Wrap the call in ``shard_map``; the last axis of ``in_specs[0]`` is the vocab mesh axis."""
        return build_row_loss_shard_map(
            self.run,
            (logits,),
            weights,
            attention_mask,
            reduction=reduction,
            vocab_parallel_axis=vocab_parallel_axis,
            run_kwargs={"alpha": alpha, "temperature": temperature, "platform": platform, "cfg": cfg},
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

    def heuristic_cfg(self, inv: Invocation[FusedRenyiEntropyConfig, Array]) -> FusedRenyiEntropyConfig:
        """Build the fallback config (XLA is the only backend)."""
        del inv
        return FusedRenyiEntropyConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[FusedRenyiEntropyConfig, Array]):
        """Single-candidate autotune space: the XLA implementation."""
        return [self.heuristic_cfg(inv)]

    candidate_cfgs_shard_map = candidate_cfgs


# Single candidate: always take the heuristic config, never autotune.
_executor: Executor[FusedRenyiEntropyConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("fused_renyi_entropy", cfg_type=FusedRenyiEntropyConfig),
    )
)


def fused_renyi_entropy(
    logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    /,
    *,
    attention_mask: Array | None = None,
    alpha: float = 1.0,
    reduction: str = "mean",
    temperature: float = 1.0,
    vocab_parallel_axis: str | None = None,
    platform: PlatformName | None = None,
    cfg: FusedRenyiEntropyConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    check_vma: bool = False,
) -> RenyiEntropyOutput:
    """Fused per-row Renyi entropy ``H_alpha(softmax(logits / T))``.

    Args:
        logits: ``(..., V)`` logits (differentiable).
        weights: Optional ``(...,)`` per-row weights / mask.
        attention_mask: Optional extra mask folded into ``weights``.
        alpha: Renyi order ``> 0``; ``1.0`` is the exact Shannon branch.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        temperature: Softmax temperature. There is NO ``T**2`` rescaling (this is
            not a KL-family term); callers compose their own scaling.
        vocab_parallel_axis: Mesh axis for per-shard inputs inside ``shard_map``.
        platform: Backend override (``"xla"``, ``"auto"``).
        cfg: Optional :class:`FusedRenyiEntropyConfig` override.
        mesh: With ``in_specs`` and ``out_specs``, wraps the call in ``jax.shard_map``.
        in_specs: Specs for ``(logits, [weights], [attention_mask])``.
        out_specs: Output spec.
        check_vma: Forwarded to ``shard_map``; forced on when a vocab axis is in use.

    Returns:
        :class:`RenyiEntropyOutput` ``(entropy, weight_sum)``.

    Raises:
        ValueError: On an invalid ``alpha``, ``reduction``, ``temperature`` or
            ``weights`` shape.
    """
    if not float(alpha) > 0.0:
        raise ValueError(f"alpha must be positive; got {alpha}")
    method, check_vma = resolve_shard_map_method(vocab_parallel_axis, mesh, in_specs, out_specs, check_vma)

    entropy = _executor(
        FusedRenyiEntropy(),
        logits=logits,
        weights=weights,
        attention_mask=attention_mask,
        alpha=alpha,
        reduction=reduction,
        temperature=temperature,
        vocab_parallel_axis=vocab_parallel_axis,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )

    flat_w = jnp.ones(logits.shape[:-1], dtype=jnp.float32) if weights is None else weights.astype(jnp.float32)
    if attention_mask is not None:
        flat_w = flat_w * attention_mask.astype(jnp.float32)
    return RenyiEntropyOutput(entropy=entropy, weight_sum=jax.lax.stop_gradient(flat_w.sum()))
