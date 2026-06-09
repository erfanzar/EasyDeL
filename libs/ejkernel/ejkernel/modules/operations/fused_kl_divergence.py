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

"""Fused forward-KL divergence operation with automatic platform & sharding dispatch.

Computes ``KL(softmax(teacher) || softmax(student))`` per row, together
with the analytic gradient w.r.t. the student logits
(``softmax(student) - softmax(teacher)``). The teacher is treated as
detached (``jax.grad`` w.r.t. ``teacher_logits`` returns zeros).
Common use case: knowledge distillation.

Backends fuse both softmaxes' running state into per-row registers and
never allocate a full ``[..., V]`` softmax tensor in HBM. The operation
routes to whichever registered backend the platform dispatcher selects
(``tilelang`` on NVIDIA GPUs, ``xla`` as the portable fallback on
TPU/CPU/AMD); additional backends can be registered without touching
this file.

When ``mesh`` / ``in_specs`` / ``out_specs`` are provided the call is
wrapped in :func:`jax.shard_map` automatically: vocab parallelism is
auto-detected from the last entry of the logits' partition spec, and
the loss/count ``psum`` reduction is auto-inserted over the leading
sharded axes.
"""

from __future__ import annotations

import os
from functools import reduce
from operator import mul
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

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
from .configs import FusedKLDivergenceConfig

PlatformName = Literal["tilelang", "xla", "triton", "pallas", "cuda", "cute", "auto"] | str


class KLDivergenceOutput(NamedTuple):
    """Per-call KL-divergence metrics returned by :func:`fused_kl_divergence`.

    Attributes:
        loss: Differentiable scalar (``reduction in {"mean", "sum"}``) or
            ``logits.shape[:-1]`` array (``reduction == "none"``). This
            is the quantity to backprop through.
        weight_sum: Sum of the per-row weights (denominator of
            ``"mean"`` reduction), detached from the gradient flow.
        teacher_entropy: Teacher distribution entropy
            ``H(p_t) = -sum_v p_t(v) log p_t(v)`` (temperature-softened),
            reduced and ``T²``-scaled exactly like ``loss``. ``None`` unless
            ``return_teacher_entropy=True``. Detached (teacher-only, no
            gradient). For forward KL this gives the standard distillation
            decomposition: the teacher cross-entropy is ``loss + teacher_entropy``.
    """

    loss: Array
    weight_sum: Array
    teacher_entropy: Array | None = None


def _flatten_axes(spec_entry) -> list[str]:
    """Return the mesh-axis names referenced by a single PartitionSpec entry."""
    if spec_entry is None:
        return []
    if isinstance(spec_entry, str):
        return [spec_entry]
    if isinstance(spec_entry, tuple):
        out: list[str] = []
        for ax in spec_entry:
            out.extend(_flatten_axes(ax))
        return out
    return []


def _infer_vocab_axis(logits_spec: PartitionSpec | None) -> str | None:
    """Pull the vocab-axis mesh name out of the logits partition spec."""
    if logits_spec is None or len(logits_spec) == 0:
        return None
    last = logits_spec[-1]
    if isinstance(last, str):
        return last
    if isinstance(last, tuple) and len(last) > 0:
        # Vocab sharded over >1 mesh axis. The collectives below take a single axis name, so silently
        # falling back to the dense per-shard KL would use a wrong (partial-vocab) normalizer -- fail loud.
        raise NotImplementedError(
            f"Vocab parallelism over multiple mesh axes (logits_spec[-1]={last!r}) is not supported; "
            "shard the vocab dimension over a single mesh axis."
        )
    return None


def _infer_leading_axes(leading_spec: PartitionSpec | None) -> tuple[str, ...]:
    """Return the flat list of mesh axes sharding the leading (batch/seq) dims."""
    if leading_spec is None:
        return ()
    out: list[str] = []
    for entry in leading_spec:
        out.extend(_flatten_axes(entry))
    return tuple(out)


class FusedKLDivergence(Kernel[FusedKLDivergenceConfig, Array]):
    """Fused forward KL between two logit tensors with platform + sharding auto-dispatch."""

    def __init__(self):
        """Create the operation object bound to the registry op id."""
        super().__init__(op_id="fused_kl_divergence")

    def get_impl(self, cfg: FusedKLDivergenceConfig):
        """Resolve the concrete backend implementation for ``cfg``."""
        platform = detect_platform("fused_kl_divergence", cfg.platform)
        return kernel_registry.get("fused_kl_divergence", platform=platform, backend=cfg.backend)

    def run(
        self,
        student_logits: Float[Array, "... vocab_size"],
        teacher_logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        reduction: str = "mean",
        direction: str = "forward",
        temperature: float = 1.0,
        beta: float = 0.5,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedKLDivergenceConfig,
    ) -> Float[Array, "..."]:
        """Run the registered KL backend with optional mask folding.

        ``attention_mask`` is multiplied into ``weights`` before dispatch
        so masked rows have zero loss and zero gradient. ``direction`` selects
        forward KL, reverse KL, or the XLA-only JSD path exposed by the public
        wrapper.
        """
        if attention_mask is not None:
            mask_f32 = attention_mask.astype(jnp.float32)
            if weights is None:
                weights = mask_f32
            else:
                weights = weights.astype(jnp.float32) * mask_f32
        n_rows = 1
        for dim in student_logits.shape[:-1]:
            n_rows *= int(dim)
        cfg_block_v = int(getattr(cfg, "block_v", self._heuristic_block_v(int(student_logits.shape[-1]))))
        cfg_block_m = int(getattr(cfg, "block_m", self._heuristic_block_m(n_rows)))
        cfg_num_warps = int(getattr(cfg, "num_warps", 4))
        cfg_num_stages = int(getattr(cfg, "num_stages", 2))
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = FusedKLDivergenceConfig(
                block_v=cfg_block_v,
                block_m=cfg_block_m,
                num_warps=cfg_num_warps,
                num_stages=cfg_num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_block_v = cfg.block_v
            cfg_block_m = cfg.block_m
        resolved = detect_platform("fused_kl_divergence", cfg.platform)
        impl = kernel_registry.get("fused_kl_divergence", platform=resolved, backend=cfg.backend)
        return impl(
            student_logits,
            teacher_logits,
            weights,
            reduction=reduction,
            direction=direction,
            temperature=temperature,
            beta=beta,
            vocab_parallel_axis=vocab_parallel_axis,
            block_v=cfg_block_v,
            block_m=cfg_block_m,
        )

    def create_shard_map_wrapper(
        self,
        student_logits: Float[Array, "... vocab_size"],
        teacher_logits: Float[Array, "... vocab_size"],
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        reduction: str = "mean",
        direction: str = "forward",
        temperature: float = 1.0,
        beta: float = 0.5,
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedKLDivergenceConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = True,
    ):
        """Wrap the KL call in ``shard_map`` with automatic collective insertion.

        Behaviour, deduced from ``in_specs``:

        * The last axis of ``in_specs[0]`` (the student spec) is treated as
          the vocab-parallel mesh axis. ``in_specs[1]`` (teacher) must
          match — both shards' softmaxes are merged via ``pmax`` / ``psum``
          inside the kernel.
        * For ``reduction in ("sum", "mean")`` the leading-axis mesh names
          (``in_specs[0][:-1]``) form the reduction axes for the per-row
          count / sum ``psum``.
        * ``check_vma`` is forwarded to ``shard_map``. TPU Pallas kernels
          need the public default (``False``) because nested ``pallas_call``
          outputs do not currently annotate ``manual_axis_type``.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if vocab_parallel_axis is None:
            vocab_parallel_axis = _infer_vocab_axis(in_specs[0])

        student_spec = in_specs[0]
        leading_spec = PartitionSpec(*student_spec[:-1]) if student_spec is not None else None
        leading_axes = _infer_leading_axes(leading_spec)

        if weights is None:
            call_args: tuple = (student_logits, teacher_logits)
            actual_in_specs = in_specs[:2]
        else:
            call_args = (student_logits, teacher_logits, weights)
            actual_in_specs = in_specs[:3]
        if attention_mask is not None:
            call_args = (*call_args, attention_mask)
            actual_in_specs = in_specs[: len(call_args)]
        if len(actual_in_specs) != len(call_args):
            raise ValueError(f"in_specs length {len(actual_in_specs)} != call_args length {len(call_args)}")

        _run = self.run
        _reduction = reduction
        _direction = direction
        _temperature = temperature
        _beta = beta
        _vocab_axis = vocab_parallel_axis
        _platform = platform
        _cfg = cfg
        _leading = leading_axes
        _has_weights = weights is not None
        _has_attention_mask = attention_mask is not None
        _inner_red = "none" if reduction == "none" else "sum"

        def _per_device(*args):
            """Run one shard-local KL call and merge scalar reductions globally."""
            ms = None
            if _has_attention_mask:
                *args, ms = args
            if _has_weights:
                ss, tt, ws = args
            else:
                ss, tt = args
                ws = None
            local_out = _run(
                ss,
                tt,
                ws,
                reduction=_inner_red,
                direction=_direction,
                temperature=_temperature,
                beta=_beta,
                attention_mask=ms,
                vocab_parallel_axis=_vocab_axis,
                platform=_platform,
                cfg=_cfg,
            )
            if _reduction == "none":
                return local_out
            if ws is None:
                local_cnt = reduce(mul, ss.shape[:-1], 1)
                cnt_weights = jnp.ones((local_cnt,), dtype=jnp.float32)
            else:
                cnt_weights = ws.astype(jnp.float32).reshape(-1)
            if ms is not None:
                cnt_weights = cnt_weights * ms.astype(jnp.float32).reshape(-1)
            cnt_local = cnt_weights.sum()
            loss_sum = jax.lax.psum(local_out, _leading) if _leading else local_out
            cnt = jax.lax.psum(cnt_local, _leading) if _leading else cnt_local
            if _reduction == "sum":
                return loss_sum
            return loss_sum / jnp.maximum(cnt, 1e-8)

        return (
            shard_map(
                _per_device,
                mesh=mesh,
                in_specs=actual_in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
            ),
            call_args,
        )

    @staticmethod
    def _shape_from_inv(inv: Invocation[FusedKLDivergenceConfig, Array]) -> tuple[int, int]:
        """Extract ``(num_rows, vocab_size)`` from the invocation's student logits."""
        logits = inv.kwargs.get("student_logits")
        if logits is None and inv.args:
            logits = inv.args[0]
        shape = getattr(logits, "shape", None)
        if shape is None or len(shape) < 2:
            return (0, 0)
        v = int(shape[-1])
        n = 1
        for d in shape[:-1]:
            n *= int(d)
        return (n, v)

    @staticmethod
    def _heuristic_block_v(v: int) -> int:
        """Operation-side cold-start ``block_v``. The autotuner sweeps
        larger sizes via :meth:`candidate_cfgs_gpu`."""
        if v == 0 or v <= 1024:
            return 256
        if v <= 16384:
            return 512
        if v <= 65536:
            return 2048
        return 4096

    @staticmethod
    def _heuristic_block_m(n: int) -> int:
        """Pick the row-block size used before autotuning has a cache hit."""
        return 1 if n < 1024 else 4

    def heuristic_cfg(self, inv: Invocation[FusedKLDivergenceConfig, Array]) -> FusedKLDivergenceConfig:
        """Build the non-autotuned fallback config for this invocation."""
        n, v = self._shape_from_inv(inv)
        return FusedKLDivergenceConfig(
            block_v=self._heuristic_block_v(v),
            block_m=self._heuristic_block_m(n),
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FusedKLDivergenceConfig, Array]):
        """Return autotune candidates for the default GPU-tuning path."""
        return self.candidate_cfgs_gpu(inv)

    def candidate_cfgs_gpu(self, inv: Invocation[FusedKLDivergenceConfig, Array]):
        """GPU candidates for fused KL.

        KL holds three streaming SMEM tiles (teacher, student, accumulator),
        so the practical ``block_v`` envelope is tighter than CE's. Sweep:

        * ``block_v`` ∈ {256, 512, 1024, 2048, 4096, 8192}, pruned by vocab.
        * ``block_m`` ∈ {1, 2, 4, 8} — bigger for small-V / large-N.
        * ``num_warps`` ∈ {4, 8}; 8 helps when ``V >= 32K`` (memory-bound).
        * ``num_stages`` ∈ {2, 3}; 3 helps memory-bound large-V.
        * SMEM filter: ``3 * BV * BM * 4B`` <= 192 KB (H100 envelope).
        """
        n, v = self._shape_from_inv(inv)
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[FusedKLDivergenceConfig] = []
        if "tilelang" in platforms:
            block_v_choices: list[int] = []
            for bv in (256, 512, 1024, 2048, 4096, 8192):
                if v == 0 or bv <= max(v, 1):
                    block_v_choices.append(bv)
            if not block_v_choices:
                block_v_choices = [self._heuristic_block_v(v)]
            if n < 1024:
                block_m_choices = [1, 2]
            elif n < 8192:
                block_m_choices = [1, 2, 4]
            else:
                block_m_choices = [1, 4, 8]
            warp_choices = (4, 8) if v >= 32768 else (4,)
            stage_choices = (2, 3) if v >= 16384 else (2,)
            for bv in block_v_choices:
                for bm in block_m_choices:
                    if bv * bm * 4 * 3 > 192 * 1024:
                        continue
                    for warps in warp_choices:
                        for stages in stage_choices:
                            candidates.append(
                                FusedKLDivergenceConfig(
                                    block_v=bv,
                                    block_m=bm,
                                    num_warps=warps,
                                    num_stages=stages,
                                    platform="tilelang",
                                    backend="gpu",
                                )
                            )
        if "xla" in platforms:
            candidates.append(
                FusedKLDivergenceConfig(
                    block_v=0,
                    block_m=0,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[FusedKLDivergenceConfig, Array]):
        """Return TPU autotune candidates: XLA baseline + Pallas streaming variants.

        XLA is the bandwidth-floor baseline and is usually fastest on TPU; the Pallas
        candidates let the autotuner pick a streaming kernel where it wins (very sparse
        masks, other TPU generations). ``direction="jsd"`` falls back to XLA inside the
        Pallas path regardless. Only ``block_v`` is swept (Pallas fixes ``block_m``);
        values are pruned to the vocab size.
        """
        _, v = self._shape_from_inv(inv)
        candidates = [
            FusedKLDivergenceConfig(
                block_v=0,
                block_m=0,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]
        for bv in (4096, 8192):
            if v == 0 or bv <= max(v, 1):
                candidates.append(
                    FusedKLDivergenceConfig(
                        block_v=bv, block_m=256, num_warps=4, num_stages=2, platform="pallas", backend="any"
                    )
                )
        return candidates

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[FusedKLDivergenceConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("fused_kl_divergence"),
    )
)


def fused_kl_divergence(
    student_logits: Float[Array, "... vocab_size"],
    teacher_logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    /,
    *,
    attention_mask: Array | None = None,
    reduction: str = "mean",
    direction: str = "forward",
    temperature: float = 1.0,
    beta: float = 0.5,
    vocab_parallel_axis: str | None = None,
    return_teacher_entropy: bool = False,
    platform: PlatformName | None = None,
    cfg: FusedKLDivergenceConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    check_vma: bool = False,
) -> KLDivergenceOutput:
    """Fused KL divergence (forward / reverse / JSD) with temperature softening.

    Three directions selected via ``direction``:

      * ``"forward"`` (default): ``KL(softmax(t/T) ‖ softmax(s/T))``.
        EasyDeL ``distillation_loss`` gradient-equivalent.
      * ``"reverse"``: ``KL(softmax(s/T) ‖ softmax(t/T))``. GKD ``β=0``.
      * ``"jsd"``: ``β·KL(p_t‖m) + (1-β)·KL(p_s‖m)`` with mixture
        ``m = β·p_t + (1-β)·p_s``. GKD intermediate ``β``.

    Loss is multiplied by ``T²`` (Hinton distillation convention).

    Args:
        student_logits: ``(..., V)`` student logits (differentiable).
        teacher_logits: ``(..., V)`` teacher logits (detached target).
        weights: Optional ``(..., )`` per-token weights. Use
            ``weights=completion_mask`` for **assistant-only loss** in
            chat distillation (1.0 on assistant tokens, 0.0 on prompt /
            padding — the kernel correctly excludes those rows from
            the loss and gradient).
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        direction: ``"forward"`` / ``"reverse"`` / ``"jsd"``.
        temperature: Softmax temperature ``T``. The loss is scaled by
            ``T²`` so gradient magnitudes are comparable across
            temperatures.
        beta: JSD interpolation factor in ``(0, 1)``; ignored unless
            ``direction="jsd"``.
        vocab_parallel_axis: TP mesh axis (forward / reverse / JSD all supported; any
            temperature supported).
        return_teacher_entropy: When ``True``, also return the teacher
            entropy ``H(p_t)`` on the output (same reduction + ``T²`` scaling
            as ``loss``, detached). Lets distillation callers recover the
            cross-entropy decomposition (teacher cross-entropy =
            ``loss + teacher_entropy`` for forward KL) without a second kernel.
        platform: Backend override (``"tilelang"``, ``"xla"``, …).
        cfg: Optional :class:`FusedKLDivergenceConfig` override.
        mesh / in_specs / out_specs: When all three are provided the
            call is wrapped in ``jax.shard_map`` automatically.

    Returns:
        :class:`KLDivergenceOutput` NamedTuple ``(loss, weight_sum)``.

    Example (basic forward KL — EasyDeL distillation gradient):
        >>> out = fused_kl_divergence(student, teacher, temperature=4.0)
        >>> grads = jax.grad(
        ...     lambda s: fused_kl_divergence(s, teacher, temperature=4.0).loss
        ... )(student)

    Example (assistant-only / completion-only loss, like GKD's
    ``completion_mask``):
        >>> # completion_mask: 1.0 on assistant tokens, 0.0 on prompt+padding
        >>> out = fused_kl_divergence(
        ...     student, teacher, completion_mask, temperature=2.0
        ... )
        >>> # out.loss is mean KL over assistant tokens only;
        >>> # out.weight_sum == completion_mask.sum() (active token count)

    Example (reverse KL):
        >>> out = fused_kl_divergence(student, teacher, direction="reverse")

    Example (generalised JSD, β=0.5):
        >>> out = fused_kl_divergence(
        ...     student, teacher, direction="jsd", beta=0.5, temperature=2.0
        ... )

    Example (3D mesh, ``dp × sp × tp`` — forward KL, ``T=1``):
        >>> from jax.experimental.mesh_utils import create_device_mesh
        >>> from jax.sharding import Mesh, PartitionSpec as P
        >>>
        >>> mesh = Mesh(create_device_mesh((2, 2, 2)), ("dp", "sp", "tp"))
        >>> in_specs = (
        ...     P("dp", "sp", "tp"),  # student — vocab on tp
        ...     P("dp", "sp", "tp"),  # teacher — same sharding
        ... )
        >>> out = fused_kl_divergence(
        ...     student, teacher,
        ...     mesh=mesh,
        ...     in_specs=in_specs,
        ...     out_specs=P(),
        ... )

        With per-token ``weights`` (and the same assistant-mask
        semantics), add a third spec:

        >>> in_specs = (
        ...     P("dp", "sp", "tp"),
        ...     P("dp", "sp", "tp"),
        ...     P("dp", "sp"),       # completion_mask — no vocab dim
        ... )
        >>> out = fused_kl_divergence(
        ...     student, teacher, completion_mask,
        ...     mesh=mesh, in_specs=in_specs, out_specs=P(),
        ... )
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
        # The vocab-parallel custom backward needs VMA (varying-manual-axis)
        # propagation through shard_map to be differentiated correctly. Without
        # check_vma the per-shard normalization term is dropped from the VJP and
        # the student-logit gradient is wrong (~7% rel error, value exact). Force
        # it on whenever vocab-parallel reduction is requested -- explicitly or
        # inferred from in_specs (same inference run() uses) -- so the op API
        # produces exact gradients without the caller knowing this quirk.
        if (vocab_parallel_axis if vocab_parallel_axis is not None else _infer_vocab_axis(in_specs[0])) is not None:
            check_vma = True

    loss = _executor(
        FusedKLDivergence(),
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        weights=weights,
        attention_mask=attention_mask,
        reduction=reduction,
        direction=direction,
        temperature=temperature,
        beta=beta,
        vocab_parallel_axis=vocab_parallel_axis,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )

    if weights is not None:
        flat_w = weights.astype(jnp.float32)
    else:
        flat_w = jnp.ones(student_logits.shape[:-1], dtype=jnp.float32)
    if attention_mask is not None:
        flat_w = flat_w * attention_mask.astype(jnp.float32)
    weight_sum = jax.lax.stop_gradient(flat_w.sum())

    teacher_entropy = None
    if return_teacher_entropy:
        # H(p_t) = -sum_v p_t log p_t over the temperature-softened teacher distribution.
        # Teacher-only and detached (no gradient); reduced and T**2-scaled exactly like `loss`.
        t_scaled = jax.lax.stop_gradient(teacher_logits.astype(jnp.float32) / temperature)
        log_p_t = jax.nn.log_softmax(t_scaled, axis=-1)
        ent = -jnp.sum(jnp.exp(log_p_t) * log_p_t, axis=-1)  # (...,)
        t_sq = float(temperature) ** 2
        if reduction == "none":
            teacher_entropy = ent * t_sq
        elif reduction == "sum":
            teacher_entropy = jnp.sum(ent * flat_w) * t_sq
        else:  # "mean"
            teacher_entropy = (jnp.sum(ent * flat_w) / jnp.maximum(flat_w.sum(), 1e-8)) * t_sq
        teacher_entropy = jax.lax.stop_gradient(teacher_entropy)

    return KLDivergenceOutput(loss=loss, weight_sum=weight_sum, teacher_entropy=teacher_entropy)
