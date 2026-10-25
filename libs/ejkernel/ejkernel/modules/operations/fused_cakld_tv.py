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

"""Fused dynamic-CAKLD + TV distillation loss with platform & sharding dispatch.

Computes, per row, the confidence-gated blend::

    gamma * KL(q || p) + (1 - gamma) * KL(p || q)

with ``p = softmax(teacher / T)``, ``q = softmax(student / T)`` and the per-row
teacher confidence ``gamma = max_v p_v`` (detached), optionally mixed with a
total-variation term::

    loss = ((1 - rho) * blend + rho * 2 * TV(p, q)) * T**2

Peaked teacher tokens get mode-seeking reverse KL and flat tokens get
mass-covering forward KL. One read of the two logit tensors replaces the composed
pipeline of four (gamma pass, reverse KL, forward KL, TV). The teacher is
detached on every path.

When ``mesh`` / ``in_specs`` / ``out_specs`` are provided the call is wrapped in
:func:`jax.shard_map`: vocab parallelism is inferred from the last entry of the
logits' partition spec and the loss/count ``psum`` is inserted over the leading
sharded axes, the same contract as ``fused_kl_divergence``.
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
from .configs import FusedCakldTvConfig
from .fused_renyi_entropy import fused_renyi_entropy

PlatformName = Literal["xla", "auto"] | str


class CakldTvOutput(NamedTuple):
    """Per-call metrics returned by :func:`fused_cakld_tv`.

    Attributes:
        loss: Differentiable scalar (``reduction in {"mean", "sum"}``) or
            ``logits.shape[:-1]`` array (``reduction == "none"``).
        weight_sum: Sum of the per-row weights (denominator of the ``"mean"``
            reduction), detached. For ``"mean"``, ``loss * weight_sum``
            reconstructs the weighted SUM exactly.
        teacher_entropy: Teacher entropy ``H(p)`` (temperature-softened), reduced
            and ``T**2``-scaled like ``loss``; ``None`` unless
            ``return_teacher_entropy=True``. Detached.
    """

    loss: Array
    weight_sum: Array
    teacher_entropy: Array | None = None


class FusedCakldTv(Kernel[FusedCakldTvConfig, Array]):
    """Fused dynamic-CAKLD + TV blend with platform + sharding auto-dispatch."""

    def __init__(self):
        """Create the operation object bound to the registry op id."""
        super().__init__(op_id="fused_cakld_tv")

    def get_impl(self, cfg: FusedCakldTvConfig):
        """Resolve the concrete backend implementation for ``cfg``."""
        platform = detect_platform("fused_cakld_tv", cfg.platform)
        return kernel_registry.get("fused_cakld_tv", platform=platform, backend=cfg.backend)

    def run(
        self,
        student_logits: Float[Array, "... vocab_size"],
        teacher_logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        reduction: str = "mean",
        temperature: float = 1.0,
        tv_weight: float = 0.0,
        entropy_weight: float = 0.0,
        entropy_alpha: float = 1.0,
        entropy_gate: float | None = None,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCakldTvConfig,
    ) -> Float[Array, "..."]:
        """Run the registered backend; ``attention_mask`` is multiplied into ``weights``."""
        if attention_mask is not None:
            mask_f32 = attention_mask.astype(jnp.float32)
            weights = mask_f32 if weights is None else weights.astype(jnp.float32) * mask_f32
        if platform is not None:
            cfg = FusedCakldTvConfig(platform=platform, backend="any" if platform == "xla" else cfg.backend)
        return self.get_impl(cfg)(
            student_logits,
            teacher_logits,
            weights,
            reduction=reduction,
            temperature=temperature,
            tv_weight=tv_weight,
            entropy_weight=entropy_weight,
            entropy_alpha=entropy_alpha,
            entropy_gate=entropy_gate,
            vocab_parallel_axis=vocab_parallel_axis,
        )

    def create_shard_map_wrapper(
        self,
        student_logits: Float[Array, "... vocab_size"],
        teacher_logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        reduction: str = "mean",
        temperature: float = 1.0,
        tv_weight: float = 0.0,
        entropy_weight: float = 0.0,
        entropy_alpha: float = 1.0,
        entropy_gate: float | None = None,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCakldTvConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = True,
    ):
        """Wrap the call in ``shard_map``; the last axis of ``in_specs[0]`` is the vocab mesh axis."""
        return build_row_loss_shard_map(
            self.run,
            (student_logits, teacher_logits),
            weights,
            attention_mask,
            reduction=reduction,
            vocab_parallel_axis=vocab_parallel_axis,
            run_kwargs={
                "temperature": temperature,
                "tv_weight": tv_weight,
                "entropy_weight": entropy_weight,
                "entropy_alpha": entropy_alpha,
                "entropy_gate": entropy_gate,
                "platform": platform,
                "cfg": cfg,
            },
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

    def heuristic_cfg(self, inv: Invocation[FusedCakldTvConfig, Array]) -> FusedCakldTvConfig:
        """Build the fallback config (XLA is the only backend)."""
        del inv
        return FusedCakldTvConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[FusedCakldTvConfig, Array]):
        """Single-candidate autotune space: the XLA implementation."""
        return [self.heuristic_cfg(inv)]

    candidate_cfgs_shard_map = candidate_cfgs


# Single candidate: always take the heuristic config, never autotune.
_executor: Executor[FusedCakldTvConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("fused_cakld_tv", cfg_type=FusedCakldTvConfig),
    )
)


def fused_cakld_tv(
    student_logits: Float[Array, "... vocab_size"],
    teacher_logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    /,
    *,
    attention_mask: Array | None = None,
    reduction: str = "mean",
    temperature: float = 1.0,
    tv_weight: float = 0.0,
    entropy_weight: float = 0.0,
    entropy_alpha: float = 1.0,
    entropy_gate: float | None = None,
    vocab_parallel_axis: str | None = None,
    return_teacher_entropy: bool = False,
    platform: PlatformName | None = None,
    cfg: FusedCakldTvConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    check_vma: bool = False,
) -> CakldTvOutput:
    """Fused per-token dynamic-CAKLD (+ optional TV mix) distillation loss.

    Per row: ``((1-rho) * (gamma*KL(q||p) + (1-gamma)*KL(p||q)) + rho*2*TV(p,q)) * T**2``
    with ``gamma = max_v softmax(teacher/T)_v`` computed inside the kernel
    (detached). ``tv_weight=0.0`` is exactly per-token dynamic CAKLD.

    Args:
        student_logits: ``(..., V)`` student logits (differentiable).
        teacher_logits: ``(..., V)`` teacher logits (detached target).
        weights: Optional ``(...,)`` per-row weights (``completion_mask`` for
            assistant-only loss).
        attention_mask: Optional extra mask folded into ``weights``.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        temperature: Softmax temperature ``T``; the loss is scaled by ``T**2``.
        tv_weight: TV mixing weight ``rho`` in ``[0, 1]``.
        entropy_weight: Coefficient ``>= 0`` of an optional gated Renyi
            student-entropy penalty ``entropy_weight * 1[gamma > entropy_gate] *
            H_alpha(q)`` added per row before the ``T**2`` scaling.
        entropy_alpha: Renyi order ``> 0`` of the penalty (``1.0`` is Shannon).
        entropy_gate: Teacher-confidence threshold in ``(0, 1)``; ``None``
            applies the penalty on every row.
        vocab_parallel_axis: Mesh axis for per-shard inputs inside ``shard_map``.
        return_teacher_entropy: Also return the detached teacher entropy (same
            reduction and ``T**2`` scaling as ``loss``).
        platform: Backend override (``"xla"``, ``"auto"``).
        cfg: Optional :class:`FusedCakldTvConfig` override.
        mesh: With ``in_specs`` and ``out_specs``, wraps the call in ``jax.shard_map``.
        in_specs: Specs for ``(student, teacher, [weights], [attention_mask])``.
        out_specs: Output spec.
        check_vma: Forwarded to ``shard_map``; forced on when a vocab axis is in use.

    Returns:
        :class:`CakldTvOutput` ``(loss, weight_sum, teacher_entropy)``.

    Raises:
        ValueError: On an out-of-range ``tv_weight`` / ``entropy_*``, invalid
            ``reduction`` / ``temperature``, or mismatched shapes.

    Example:
        >>> out = fused_cakld_tv(student, teacher, mask, temperature=1.0, tv_weight=0.25)
        >>> weighted_sum = out.loss * out.weight_sum  # == sum_i w_i * loss_i
    """
    if not 0.0 <= float(tv_weight) <= 1.0:
        raise ValueError(f"tv_weight must be within [0, 1]; got {tv_weight}")
    if float(entropy_weight) < 0.0:
        raise ValueError(f"entropy_weight must be non-negative; got {entropy_weight}")
    if not float(entropy_alpha) > 0.0:
        raise ValueError(f"entropy_alpha must be positive; got {entropy_alpha}")
    if entropy_gate is not None and not 0.0 < float(entropy_gate) < 1.0:
        raise ValueError(f"entropy_gate must be within (0, 1) when set; got {entropy_gate}")
    method, check_vma = resolve_shard_map_method(vocab_parallel_axis, mesh, in_specs, out_specs, check_vma)

    loss = _executor(
        FusedCakldTv(),
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        weights=weights,
        attention_mask=attention_mask,
        reduction=reduction,
        temperature=temperature,
        tv_weight=tv_weight,
        entropy_weight=entropy_weight,
        entropy_alpha=entropy_alpha,
        entropy_gate=entropy_gate,
        vocab_parallel_axis=vocab_parallel_axis,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )

    flat_w = jnp.ones(student_logits.shape[:-1], dtype=jnp.float32) if weights is None else weights.astype(jnp.float32)
    if attention_mask is not None:
        flat_w = flat_w * attention_mask.astype(jnp.float32)
    weight_sum = jax.lax.stop_gradient(flat_w.sum())

    teacher_entropy = None
    if return_teacher_entropy:
        # Shannon entropy of the temperature-softened teacher through the Renyi op (alpha=1); it is
        # detached and scaled by T**2 so it reads like `loss`. Global arrays: XLA inserts any collectives.
        ent = fused_renyi_entropy(
            jax.lax.stop_gradient(teacher_logits),
            weights,
            attention_mask=attention_mask,
            alpha=1.0,
            reduction=reduction,
            temperature=temperature,
            platform=platform,
        ).entropy
        teacher_entropy = jax.lax.stop_gradient(ent * (float(temperature) ** 2))

    return CakldTvOutput(loss=loss, weight_sum=weight_sum, teacher_entropy=teacher_entropy)
