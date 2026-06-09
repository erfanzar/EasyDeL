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

"""Fused cross-entropy operation with automatic platform & sharding dispatch.

Computes per-row cross-entropy ``-log p(target | logits)`` (or its
soft-target generalisation) together with the analytic gradient
``softmax - target_dist``, fused into a single sweep of the vocabulary
so the ``[..., V]`` log-softmax / softmax tensor is never materialised
in HBM. Supports ``label_smoothing``, z-loss regularisation, and dense
soft targets.

The operation routes to whichever registered backend the platform
dispatcher selects (e.g. ``tilelang`` on NVIDIA GPUs, ``xla`` as the
portable fallback on TPU/CPU/AMD); additional backends can be
registered without touching this file.

When ``mesh`` / ``in_specs`` / ``out_specs`` are provided the call is
wrapped in :func:`jax.shard_map` automatically: vocab parallelism is
auto-detected from the last entry of the logits' partition spec, and
the loss/count ``psum`` reduction is auto-inserted over the leading
sharded axes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
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
from .configs import FusedCrossEntropyConfig

PlatformName = Literal["tilelang", "xla", "triton", "pallas", "cuda", "cute", "auto"] | str
ChunkStrategy = Literal["vocab", "token", "block"]


class CrossEntropyOutput(NamedTuple):
    """Per-call cross-entropy metrics returned by :func:`fused_cross_entropy`.

    Attributes:
        loss: Differentiable scalar (``reduction in {"mean", "sum"}``) or
            ``logits.shape[:-1]`` array (``reduction == "none"``). This is
            the quantity to backprop through; the other fields are
            ``stop_gradient`` metrics.
        z_loss: ``z_loss · mean(lse²)`` (or ``sum`` / per-row, matching
            ``reduction``). Zero when the ``z_loss`` coefficient is 0.
        weight_sum: Sum of the per-token weights actually used by the
            kernel (the denominator of ``"mean"`` reduction).
        accuracy: weight-weighted fraction of correct argmax predictions,
            ``sum((argmax(logits) == targets) * w) / sum(w)`` (a plain
            correct/active count for binary masks). Sparse mode only;
            ``None`` in dense mode where there is no single integer target
            per row.
    """

    loss: Array
    z_loss: Array
    weight_sum: Array
    accuracy: Array | None


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
    """Pull the vocab-axis mesh name out of the logits partition spec.

    Returns the last-axis sharding when it's a single string; ``None``
    otherwise (replicated, or sharded over a tuple of axes which the
    caller must handle explicitly).
    """
    if logits_spec is None or len(logits_spec) == 0:
        return None
    last = logits_spec[-1]
    if isinstance(last, str):
        return last
    return None


def _infer_leading_axes(leading_spec: PartitionSpec | None) -> tuple[str, ...]:
    """Return the flat list of mesh axes sharding the leading (batch/seq) dims."""
    if leading_spec is None:
        return ()
    out: list[str] = []
    for entry in leading_spec:
        out.extend(_flatten_axes(entry))
    return tuple(out)


class FusedCrossEntropy(Kernel[FusedCrossEntropyConfig, Array]):
    """Fused cross-entropy with platform + sharding auto-dispatch.

    Computes per-token cross-entropy ``-log p(target | logits)`` together with
    the analytic gradient ``softmax - onehot`` in a single fused kernel.
    The full ``[..., V]`` log-softmax tensor is never materialised in HBM.
    """

    def __init__(self):
        """Create the operation object bound to the registry op id."""
        super().__init__(op_id="fused_cross_entropy")

    def get_impl(self, cfg: FusedCrossEntropyConfig):
        """Resolve the concrete backend implementation for ``cfg``."""
        platform = detect_platform("fused_cross_entropy", cfg.platform)
        return kernel_registry.get("fused_cross_entropy", platform=platform, backend=cfg.backend)

    def run(
        self,
        logits: Float[Array, "... vocab_size"],
        targets: Int[Array, "..."] | None = None,
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss: float = 0.0,
        soft_targets: Float[Array, "... vocab_size"] | None = None,
        reduction: str = "mean",
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCrossEntropyConfig,
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Run the registered backend, returning ``(loss, per_row_correct)``.

        ``per_row_correct`` is a 0/1 float array (with sentinel ``-1`` in
        the dense / TP modes where argmax isn't computed) — the public
        wrapper rolls it into :attr:`CrossEntropyOutput.accuracy`.

        When ``attention_mask`` is supplied it is multiplied into
        ``weights`` before dispatch (a position with mask=0 has its loss
        and gradient zeroed out and triggers the kernel's per-block
        sparse early-exit — saving the full ``O(V)`` softmax pass for
        inactive rows). Combining order:
        ``effective_weights = (weights or 1.0) * attention_mask``.
        """
        if attention_mask is not None:
            mask_f32 = attention_mask.astype(jnp.float32)
            if weights is None:
                weights = mask_f32
            else:
                weights = weights.astype(jnp.float32) * mask_f32
        n_rows = 1
        for dim in logits.shape[:-1]:
            n_rows *= int(dim)
        cfg_block_v = int(getattr(cfg, "block_v", self._heuristic_block_v(int(logits.shape[-1]))))
        cfg_block_m = int(getattr(cfg, "block_m", self._heuristic_block_m(n_rows)))
        cfg_num_warps = int(getattr(cfg, "num_warps", 4))
        cfg_num_stages = int(getattr(cfg, "num_stages", 2))
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = FusedCrossEntropyConfig(
                block_v=cfg_block_v,
                block_m=cfg_block_m,
                num_warps=cfg_num_warps,
                num_stages=cfg_num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_block_v = cfg.block_v
            cfg_block_m = cfg.block_m
        resolved = detect_platform("fused_cross_entropy", cfg.platform)
        impl = kernel_registry.get("fused_cross_entropy", platform=resolved, backend=cfg.backend)
        return impl(
            logits,
            targets,
            weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            soft_targets=soft_targets,
            reduction=reduction,
            vocab_parallel_axis=vocab_parallel_axis,
            block_v=cfg_block_v,
            block_m=cfg_block_m,
        )

    def create_shard_map_wrapper(
        self,
        logits: Float[Array, "... vocab_size"],
        targets: Int[Array, "..."] | None = None,
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss: float = 0.0,
        soft_targets: Float[Array, "... vocab_size"] | None = None,
        reduction: str = "mean",
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCrossEntropyConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = True,
    ):
        """Wrap the loss call in ``shard_map`` with automatic collective insertion.

        Behaviour, deduced from ``in_specs``:

        * The last axis of ``in_specs[0]`` (the logits spec) is treated as
          the vocab-parallel mesh axis. If it's a single string and
          ``vocab_parallel_axis`` was not user-overridden, that mesh axis
          is passed through to ``run`` so the per-shard kernel emits the
          ``pmax`` / ``psum`` collectives needed to merge per-shard
          softmax stats.
        * For ``reduction in ("sum", "mean")``, all mesh axes sharding the
          leading (batch/seq) dimensions of ``targets`` are collected and
          a ``psum`` over them is inserted inside the wrapper so the
          returned scalar is the *global* (mesh-wide) loss.
        * ``check_vma`` is forwarded to ``shard_map``. TPU Pallas kernels
          need the public default (``False``) because nested ``pallas_call``
          outputs do not currently annotate ``manual_axis_type``.

        Returns ``(shard_map_fn, call_args)`` per the Executor contract.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if vocab_parallel_axis is None:
            vocab_parallel_axis = _infer_vocab_axis(in_specs[0])

        # The logits spec carries the vocab sharding; the per-token spec is just its leading (batch/seq)
        # dims with the vocab axis dropped. Deriving it here (rather than indexing in_specs[1:]) keeps the
        # wrapper correct for every operand layout -- sparse targets, dense soft_targets, optional
        # weights / attention_mask -- and gives the per-row second output its proper sharding.
        logit_spec = in_specs[0]
        token_spec = PartitionSpec(*logit_spec[:-1])
        leading_axes = _infer_leading_axes(token_spec)

        _is_soft = soft_targets is not None
        _has_targets = (not _is_soft) and targets is not None
        _has_weights = weights is not None
        _has_attention_mask = attention_mask is not None

        # Assemble the sharded operands + specs in a fixed order. ``targets`` (sparse) and
        # ``soft_targets`` (dense) are mutually exclusive; soft_targets is vocab-sharded like logits.
        call_args: tuple = (logits,)
        actual_in_specs: tuple = (logit_spec,)
        if _is_soft:
            call_args = (*call_args, soft_targets)
            actual_in_specs = (*actual_in_specs, logit_spec)
        elif _has_targets:
            call_args = (*call_args, targets)
            actual_in_specs = (*actual_in_specs, token_spec)
        if _has_weights:
            call_args = (*call_args, weights)
            actual_in_specs = (*actual_in_specs, token_spec)
        if _has_attention_mask:
            call_args = (*call_args, attention_mask)
            actual_in_specs = (*actual_in_specs, token_spec)

        _run = self.run
        _ignore_index = ignore_index
        _label_smoothing = label_smoothing
        _z_loss = z_loss
        _reduction = reduction
        _vocab_axis = vocab_parallel_axis
        _platform = platform
        _cfg = cfg
        _leading = leading_axes
        _inner_red = "none" if reduction == "none" else "sum"

        def _per_device(*args):
            """Run one shard-local loss and merge scalar reductions globally."""
            args = list(args)
            xs = args.pop(0)
            soft = args.pop(0) if _is_soft else None
            ts = args.pop(0) if _has_targets else None
            ws = args.pop(0) if _has_weights else None
            ms = args.pop(0) if _has_attention_mask else None
            local_loss, local_correct = _run(
                xs,
                ts,
                ws,
                ignore_index=_ignore_index,
                label_smoothing=_label_smoothing,
                z_loss=_z_loss,
                soft_targets=soft,
                attention_mask=ms,
                reduction=_inner_red,
                vocab_parallel_axis=_vocab_axis,
                platform=_platform,
                cfg=_cfg,
            )
            if _reduction == "none":
                return local_loss, local_correct
            if ws is not None:
                cnt_weights = ws.astype(jnp.float32)
            elif ts is not None:
                cnt_weights = (ts != _ignore_index).astype(jnp.float32)
            else:
                cnt_weights = jnp.ones(xs.shape[:-1], dtype=jnp.float32)
            if ms is not None:
                cnt_weights = cnt_weights * ms.astype(jnp.float32)
            cnt_local = cnt_weights.sum()
            loss_sum = jax.lax.psum(local_loss, _leading) if _leading else local_loss
            cnt = jax.lax.psum(cnt_local, _leading) if _leading else cnt_local
            if _reduction == "sum":
                final_loss = loss_sum
            else:
                final_loss = loss_sum / jnp.maximum(cnt, 1e-8)
            return final_loss, local_correct

        wrapped_out_specs = (out_specs, token_spec)

        return (
            shard_map(
                _per_device,
                mesh=mesh,
                in_specs=actual_in_specs,
                out_specs=wrapped_out_specs,
                check_vma=check_vma,
            ),
            call_args,
        )

    @staticmethod
    def _shape_from_inv(inv: Invocation[FusedCrossEntropyConfig, Array]) -> tuple[int, int]:
        """Extract ``(num_rows, vocab_size)`` from the invocation's logits arg."""
        logits = inv.kwargs.get("logits")
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
        """Operation-side ``block_v`` heuristic (mirrors the kernel-side fallback).

        Lives here so the autotuner / heuristic_cfg controls block sizes
        without crossing the operation/kernel boundary.
        """
        if v == 0 or v <= 1024:
            return 256
        if v <= 16384:
            return 512
        if v <= 65536:
            return 1024
        return 2048

    @staticmethod
    def _heuristic_block_m(n: int) -> int:
        """Pick the row-block size used before autotuning has a cache hit."""
        return 1 if n < 1024 else 4

    def heuristic_cfg(self, inv: Invocation[FusedCrossEntropyConfig, Array]) -> FusedCrossEntropyConfig:
        """Build the non-autotuned fallback config for this invocation."""
        n, v = self._shape_from_inv(inv)
        return FusedCrossEntropyConfig(
            block_v=self._heuristic_block_v(v),
            block_m=self._heuristic_block_m(n),
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        """Return autotune candidates for the default GPU-tuning path."""
        return self.candidate_cfgs_gpu(inv)

    def candidate_cfgs_gpu(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        """GPU candidates: enumerate (block_v, block_m, num_warps) for
        tilelang + one XLA baseline.

        The autotuner picks the fastest of these; ``heuristic_cfg`` is
        the cold-start default before autotune results are cached.

        Tuning notes for H100:

        * ``block_v`` ∈ {256, 512, 1024, 2048, 4096, 8192}, pruned by
          actual vocab. Bigger ``block_v`` amortises chunk-loop overhead
          at the cost of SMEM/registers per CTA.
        * ``block_m`` ∈ {1, 2, 4, 8} — ``1`` maximises occupancy for
          wide-vocab; bigger values amortise fixed cost on small-V/big-N.
        * ``num_warps`` ∈ {4, 8} — 8 helps when ``V >= 32K`` (memory-bound).
        * ``num_stages`` ∈ {2, 3} — 3 helps memory-bound large-V.
        * SMEM filter: rough fp32 estimate ``block_v * block_m * 12B``
          must fit in 192KB (H100 envelope; tighter than the 228KB
          hardware limit to leave buffer for the pipeline).
        """
        n, v = self._shape_from_inv(inv)
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[FusedCrossEntropyConfig] = []
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
                                FusedCrossEntropyConfig(
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
                FusedCrossEntropyConfig(
                    block_v=0,
                    block_m=0,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        """Return TPU autotune candidates: XLA baseline + Pallas streaming variants.

        XLA is the bandwidth-floor baseline and is usually fastest on TPU for dense
        CE; the Pallas candidates are included so the autotuner can empirically pick
        a streaming kernel where it wins (e.g. very sparse / completion-only masks,
        or other TPU generations). The Pallas CE path fixes ``block_m`` internally,
        so only ``block_v`` is swept; values are pruned to the vocab size.
        """
        _, v = self._shape_from_inv(inv)
        candidates = [
            FusedCrossEntropyConfig(
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
                    FusedCrossEntropyConfig(
                        block_v=bv,
                        block_m=256,
                        num_warps=4,
                        num_stages=2,
                        platform="pallas",
                        backend="any",
                    )
                )
        return candidates

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[FusedCrossEntropyConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("fused_cross_entropy"),
    )
)


def _combine_weights(targets, weights, attention_mask, ignore_index, compute_dtype):
    """Build effective per-token float weights: ``(weights or valid) * mask``.

    ``valid`` is the ``targets != ignore_index`` indicator when no explicit
    ``weights`` are supplied; ``attention_mask`` (sparse padding/completion
    mask) is multiplied in afterwards.
    """
    if weights is None:
        eff = (targets != ignore_index).astype(compute_dtype)
    else:
        eff = weights.astype(compute_dtype)
    if attention_mask is not None:
        eff = eff * attention_mask.astype(compute_dtype)
    return eff


def _chunked_cross_entropy_dispatch(
    *,
    logits,
    targets,
    weights,
    attention_mask,
    ignore_index,
    label_smoothing,
    z_loss,
    reduction,
    chunk_size,
    chunk_strategy,
    compute_dtype,
    checkpoint=True,
):
    """Route a chunked-logits CE call to the matching XLA streaming kernel."""
    from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_chunked import (
        blockwise_cross_entropy,
        chunked_token_cross_entropy,
        chunked_vocab_cross_entropy,
    )

    if logits is None or targets is None:
        raise ValueError("chunked cross-entropy requires `logits` and `targets`.")
    cdtype = jnp.dtype(compute_dtype) if compute_dtype is not None else logits.dtype
    eff_weights = _combine_weights(targets, weights, attention_mask, ignore_index, cdtype)

    common = dict(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        reduction=reduction,
        compute_dtype=cdtype,
    )
    if chunk_strategy == "vocab":
        # vocab chunking is a two-pass logsumexp (no per-block checkpoint knob).
        loss, zl, wsum, acc = chunked_vocab_cross_entropy(
            logits, targets, eff_weights, chunk_size=int(chunk_size), **common
        )
    elif chunk_strategy == "block":
        loss, zl, wsum, acc = blockwise_cross_entropy(
            logits, targets, eff_weights, block_size=int(chunk_size), checkpoint=checkpoint, **common
        )
    elif chunk_strategy == "token":
        loss, zl, wsum, acc = chunked_token_cross_entropy(
            logits, targets, eff_weights, token_chunk_size=int(chunk_size), **common
        )
    else:
        raise ValueError(f"chunk_strategy must be vocab/token/block, got {chunk_strategy!r}")

    return CrossEntropyOutput(
        loss=loss,
        z_loss=jax.lax.stop_gradient(zl),
        weight_sum=jax.lax.stop_gradient(wsum),
        accuracy=jax.lax.stop_gradient(acc),
    )


def _fused_linear_cross_entropy_dispatch(
    *,
    hidden,
    targets,
    weights,
    lm_head_weight,
    lm_head_bias,
    lm_head_fn,
    logit_softcap,
    attention_mask,
    ignore_index,
    label_smoothing,
    z_loss,
    reduction,
    token_chunk_size,
    compute_dtype,
    checkpoint=True,
    sparse_skip=False,
):
    """Route an FLCE call to the token-chunked XLA fused-linear kernel."""
    from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_linear import fused_linear_cross_entropy

    if targets is None:
        raise ValueError("FLCE mode requires integer `targets`.")
    cdtype = jnp.dtype(compute_dtype) if compute_dtype is not None else hidden.dtype
    eff_weights = _combine_weights(targets, weights, attention_mask, ignore_index, cdtype)

    loss, zl, wsum, acc = fused_linear_cross_entropy(
        hidden,
        targets,
        eff_weights,
        lm_head_weight=lm_head_weight,
        lm_head_bias=lm_head_bias,
        lm_head_fn=lm_head_fn,
        logit_softcap=logit_softcap,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        reduction=reduction,
        token_chunk_size=int(token_chunk_size),
        compute_dtype=cdtype,
        checkpoint=checkpoint,
        sparse_skip=sparse_skip,
    )
    return CrossEntropyOutput(
        loss=loss,
        z_loss=jax.lax.stop_gradient(zl),
        weight_sum=jax.lax.stop_gradient(wsum),
        accuracy=jax.lax.stop_gradient(acc),
    )


def _fused_linear_cross_entropy_vp_dispatch(
    *,
    hidden,
    targets,
    weights,
    lm_head_weight,
    lm_head_bias,
    lm_head_fn,
    logit_softcap,
    attention_mask,
    ignore_index,
    reduction,
    token_chunk_size,
    compute_dtype,
    checkpoint,
    vocab_parallel_axis,
    mesh,
    in_specs,
    out_specs,
    sparse_skip=False,
):
    """Vocab-parallel fused-linear cross-entropy (FLCE) wrapped in ``shard_map``.

    A closure ``lm_head_fn`` cannot be vocab-sharded (``shard_map`` manual mode materializes a closed-over
    weight in full on every device), so vocab-parallel FLCE requires the raw ``lm_head_weight`` ``[H, V]``
    sharded ``[H, V/tp]`` and passed as a shard_map argument. Each token chunk projects to ``[chunk,
    V/tp]`` and the per-chunk CE ``psum``s the softmax normalizer over the vocab axis -- so the full
    ``[chunk, V]`` logits are never formed. The per-shard token sums are then ``psum``-reduced over the
    batch/seq mesh axes; gradients flow to ``hidden`` (psum-replicated over the vocab axis) and the local
    weight slab. ``in_specs`` is ``(hidden_spec, lm_head_weight_spec, ...)``.
    """
    from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_linear import fused_linear_cross_entropy

    if targets is None:
        raise ValueError("FLCE mode requires integer `targets`.")
    if lm_head_weight is None:
        raise ValueError(
            "vocab-parallel FLCE requires `lm_head_weight` (a closure `lm_head_fn` cannot be sharded by "
            "shard_map; pass the raw [H, V] weight sharded on the vocab axis instead)."
        )
    if lm_head_fn is not None:
        raise ValueError("vocab-parallel FLCE: pass `lm_head_weight`, not `lm_head_fn`.")

    cdtype = jnp.dtype(compute_dtype) if compute_dtype is not None else hidden.dtype
    eff_weights = _combine_weights(targets, weights, attention_mask, ignore_index, cdtype)

    hidden_spec = in_specs[0]
    weight_spec = in_specs[1]
    token_spec = PartitionSpec(*hidden_spec[:-1])
    leading_axes = _infer_leading_axes(token_spec)

    _has_bias = lm_head_bias is not None
    call_args: tuple = (hidden, lm_head_weight, targets, eff_weights)
    actual_in_specs: tuple = (hidden_spec, weight_spec, token_spec, token_spec)
    if _has_bias:
        call_args = (*call_args, lm_head_bias)
        actual_in_specs = (*actual_in_specs, PartitionSpec(weight_spec[-1]))

    _tcs = int(token_chunk_size)
    _is_mean = reduction == "mean"

    def _per_device(*args):
        args = list(args)
        h = args.pop(0)
        w_head = args.pop(0)
        t = args.pop(0)
        w = args.pop(0)
        b = args.pop(0) if _has_bias else None
        loss_sum, _z, w_sum, acc = fused_linear_cross_entropy(
            h,
            t,
            w,
            lm_head_weight=w_head,
            lm_head_bias=b,
            lm_head_fn=None,
            logit_softcap=logit_softcap,
            ignore_index=ignore_index,
            reduction="sum",
            token_chunk_size=_tcs,
            compute_dtype=cdtype,
            checkpoint=checkpoint,
            vocab_parallel_axis=vocab_parallel_axis,
            sparse_skip=sparse_skip,
            # Make the per-chunk sparse-skip predicate uniform across the token (batch/seq) shards inside
            # shard_map, so divergent control flow never deadlocks the inner vocab psum (no-op when
            # sparse_skip is off).
            sparse_reduce_axes=tuple(leading_axes),
        )
        correct_local = acc * w_sum  # recover the per-shard correct count from the ratio
        loss_g = jax.lax.psum(loss_sum, leading_axes) if leading_axes else loss_sum
        w_g = jax.lax.psum(w_sum, leading_axes) if leading_axes else w_sum
        correct_g = jax.lax.psum(correct_local, leading_axes) if leading_axes else correct_local
        final_loss = loss_g / jnp.maximum(w_g, jnp.asarray(1e-8, dtype=loss_g.dtype)) if _is_mean else loss_g
        return final_loss, w_g, correct_g

    loss, w_sum, correct = shard_map(
        _per_device,
        mesh=mesh,
        in_specs=actual_in_specs,
        out_specs=(out_specs, PartitionSpec(), PartitionSpec()),
        check_vma=True,
    )(*call_args)
    return CrossEntropyOutput(
        loss=loss,
        z_loss=jax.lax.stop_gradient(jnp.zeros((), dtype=loss.dtype)),
        weight_sum=jax.lax.stop_gradient(w_sum),
        accuracy=jax.lax.stop_gradient(correct / jnp.maximum(w_sum, jnp.asarray(1e-8, dtype=w_sum.dtype))),
    )


def fused_cross_entropy(
    logits: Float[Array, "... vocab_size"] | None = None,
    targets: Int[Array, "..."] | None = None,
    weights: Float[Array, "..."] | None = None,
    *,
    hidden: Float[Array, "... hidden_size"] | None = None,
    lm_head_weight: Float[Array, "hidden_size vocab_size"] | None = None,
    lm_head_bias: Float[Array, "vocab_size"] | None = None,
    lm_head_fn: "Callable[[Array], Array] | None" = None,
    logit_softcap: float | None = None,
    chunk_size: int = 0,
    chunk_strategy: ChunkStrategy = "vocab",
    attention_mask: Array | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets: Float[Array, "... vocab_size"] | None = None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
    compute_dtype: jnp.dtype | None = None,
    checkpoint: bool = True,
    sparse_skip: bool = False,
    platform: PlatformName | None = None,
    cfg: FusedCrossEntropyConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    check_vma: bool = False,
) -> CrossEntropyOutput:
    """Fused cross-entropy with automatic platform + sharding dispatch.

    Input modes (selected by which arrays are passed):
      * **Sparse** (default): integer ``targets`` of shape
        ``logits.shape[:-1]``. Optional ``label_smoothing`` and ``z_loss``
        regularisation fold into the kernel at build time (no runtime
        cost when both are 0).
      * **Dense**: pass ``soft_targets`` (full probability distribution
        over the vocab). ``targets`` is ignored; ``label_smoothing``
        must be applied externally before the call.
      * **Chunked logits**: pass ``logits`` with ``chunk_size > 0`` to stream
        the vocab/token axis in slices (``chunk_strategy`` ∈
        ``{"vocab", "token", "block"}``), bounding the transient softmax
        working set. XLA path; analytic-gradient parity with the dense path.
      * **Fused linear (FLCE)**: pass ``hidden`` plus either ``lm_head_weight``
        (+ optional ``lm_head_bias``) or ``lm_head_fn``. Projects the LM head
        in ``chunk_size`` token chunks and computes CE per chunk, so the full
        ``[..., V]`` logits are **never** materialised (only ``[..., chunk, V]``
        at a time). Backward recomputes per chunk; gradients flow to ``hidden``
        and the LM-head weights. XLA path; ``reduction`` ∈ ``{"sum", "mean"}``.

    Args:
        logits: ``(..., V)`` predicted logits.
        targets: Integer token ids with shape ``logits.shape[:-1]`` (or
            ``hidden.shape[:-1]`` in FLCE mode).
        weights: Optional per-token weights of shape ``targets.shape``.
        hidden: ``(..., T, H)`` hidden states for the FLCE path.
        lm_head_weight: ``(H, V)`` LM-head weight (FLCE raw-matmul mode).
        lm_head_bias: Optional ``(V,)`` LM-head bias (FLCE raw-matmul mode).
        lm_head_fn: Callable ``(..., H) -> (..., V)`` projecting the head
            (FLCE custom-head mode; mutually exclusive with ``lm_head_weight``).
        logit_softcap: Optional ``cap·tanh(logits/cap)`` soft-cap applied per
            FLCE chunk before the loss.
        chunk_size: Chunk length (>0 enables chunked/FLCE streaming).
        chunk_strategy: Which axis to stream in chunked-logits mode.
        compute_dtype: CE math dtype for the chunked / FLCE paths
            (defaults to the input dtype — no forced fp32).
        checkpoint: For the FLCE and ``"block"`` chunked paths, whether to wrap
            each chunk/block body in :func:`jax.checkpoint` so the backward
            recomputes its logits instead of storing them (default ``True`` —
            the memory-bounded behaviour). ``False`` keeps residuals live
            (faster, more memory). No effect on the dense / vocab / token paths.
        ignore_index: Sparse-mode sentinel for ignored positions.
        label_smoothing: ``α ∈ [0, 1)`` — smoothed target distribution
            ``p[target] = 1 - α``, ``p[v ≠ target] = α / (V - 1)``.
        z_loss: Coefficient for ``z_loss · lse²`` regularisation
            (Mesh-TF / PaLM-style logit magnitude penalty).
        soft_targets: ``(..., V)`` dense probability targets. Switches
            to the dense kernel path.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        vocab_parallel_axis: Mesh axis name along which ``V`` is sharded.
            Usually inferred automatically from ``in_specs``.
        platform: Backend override (``"tilelang"``, ``"xla"``, …).
            Routes through ``kernel_registry``; any registered backend
            name is accepted.
        cfg: Optional :class:`FusedCrossEntropyConfig` override.
        mesh / in_specs / out_specs: When all three are provided the call
            is wrapped in ``jax.shard_map`` automatically.

    Returns:
        :class:`CrossEntropyOutput` NamedTuple with ``(loss, z_loss,
        weight_sum, accuracy)``. ``.loss`` is the differentiable scalar
        (or per-token array for ``reduction="none"``); the other fields
        are detached metrics. For ``jax.grad`` / ``jax.value_and_grad``,
        either index in (``.loss``) or wrap with
        ``lambda *a: fused_cross_entropy(*a).loss``.

    Example (sparse, single-device):
        >>> out = fused_cross_entropy(logits, targets)
        >>> out.loss        # scalar
        >>> out.accuracy    # scalar in [0, 1]

    Example (with label smoothing + z-loss for EasyDeL training):
        >>> out = fused_cross_entropy(
        ...     logits, targets, weights,
        ...     label_smoothing=0.1, z_loss=1e-4,
        ... )
        >>> out.loss, out.z_loss

    Example (distillation — dense soft targets from teacher):
        >>> teacher_probs = jax.nn.softmax(teacher_logits / T, axis=-1)
        >>> out = fused_cross_entropy(
        ...     student_logits, soft_targets=teacher_probs, weights=mask,
        ... )
        >>> out.accuracy is None  # dense mode

    Example (gradient through ``.loss``):
        >>> grads = jax.grad(lambda x: fused_cross_entropy(x, targets).loss)(logits)

    Example (3D mesh, ``dp × sp × tp``):
        Build the mesh once and pass it explicitly along with
        per-input partition specs. The wrapper auto-detects the
        vocab-parallel axis from the **last** entry of ``in_specs[0]``
        (here, ``"tp"``) and inserts ``psum`` over the leading sharded
        axes (``"dp"``, ``"sp"``) so the scalar ``.loss`` is the
        mesh-wide mean.

        >>> from jax.experimental.mesh_utils import create_device_mesh
        >>> from jax.sharding import Mesh, PartitionSpec as P
        >>>
        >>> mesh = Mesh(create_device_mesh((2, 2, 2)), ("dp", "sp", "tp"))
        >>> in_specs = (
        ...     P("dp", "sp", "tp"),  # logits — vocab on tp, batch+seq on dp+sp
        ...     P("dp", "sp"),        # targets — only batch+seq sharded
        ... )
        >>> out = fused_cross_entropy(
        ...     logits, targets,
        ...     mesh=mesh,
        ...     in_specs=in_specs,
        ...     out_specs=P(),       # scalar loss replicated across the mesh
        ... )
        >>> out.loss     # global mean cross-entropy
        >>> out.accuracy # global accuracy

        With per-token ``weights``, extend ``in_specs`` to three entries:

        >>> in_specs = (
        ...     P("dp", "sp", "tp"),
        ...     P("dp", "sp"),
        ...     P("dp", "sp"),       # weights — same sharding as targets
        ... )
        >>> out = fused_cross_entropy(
        ...     logits, targets, weights,
        ...     mesh=mesh, in_specs=in_specs, out_specs=P(),
        ... )
    """
    if hidden is not None:
        # Vocab-parallel FLCE: when a mesh + specs are supplied and the vocab axis is given (explicitly or
        # inferred from the lm_head_weight spec's last dim), run the token-chunked projection + CE inside a
        # shard_map with a column-parallel ``[H, V/tp]`` weight so the full ``[chunk, V]`` is never formed.
        flce_vp_axis = vocab_parallel_axis
        if flce_vp_axis is None and lm_head_weight is not None and in_specs is not None and len(in_specs) > 1:
            flce_vp_axis = _infer_vocab_axis(in_specs[1])
        if flce_vp_axis is not None and mesh is not None and in_specs is not None and out_specs is not None:
            return _fused_linear_cross_entropy_vp_dispatch(
                hidden=hidden,
                targets=targets,
                weights=weights,
                lm_head_weight=lm_head_weight,
                lm_head_bias=lm_head_bias,
                lm_head_fn=lm_head_fn,
                logit_softcap=logit_softcap,
                attention_mask=attention_mask,
                ignore_index=ignore_index,
                reduction=reduction,
                token_chunk_size=chunk_size,
                compute_dtype=compute_dtype,
                checkpoint=checkpoint,
                vocab_parallel_axis=flce_vp_axis,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                sparse_skip=sparse_skip,
            )
        return _fused_linear_cross_entropy_dispatch(
            hidden=hidden,
            targets=targets,
            weights=weights,
            lm_head_weight=lm_head_weight,
            lm_head_bias=lm_head_bias,
            lm_head_fn=lm_head_fn,
            logit_softcap=logit_softcap,
            attention_mask=attention_mask,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            reduction=reduction,
            token_chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            checkpoint=checkpoint,
            sparse_skip=sparse_skip,
        )

    if chunk_size and soft_targets is None and vocab_parallel_axis is None and mesh is None:
        return _chunked_cross_entropy_dispatch(
            logits=logits,
            targets=targets,
            weights=weights,
            attention_mask=attention_mask,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            reduction=reduction,
            chunk_size=chunk_size,
            chunk_strategy=chunk_strategy,
            compute_dtype=compute_dtype,
            checkpoint=checkpoint,
        )

    if logits is None:
        raise ValueError("`logits` is required unless `hidden` (FLCE mode) is provided.")

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
        # The vocab-parallel custom backward needs VMA (varying-manual-axis)
        # propagation through shard_map to be differentiated correctly. Without
        # check_vma the per-shard normalization term is dropped from the VJP and
        # the logit gradient is wrong (value exact, grad ~off). Force it on
        # whenever vocab-parallel reduction is requested -- explicitly or inferred
        # from in_specs (same inference run() uses) -- so the op API produces
        # exact gradients without the caller knowing this quirk.
        if (vocab_parallel_axis if vocab_parallel_axis is not None else _infer_vocab_axis(in_specs[0])) is not None:
            check_vma = True

    loss, per_row_correct = _executor(
        FusedCrossEntropy(),
        logits=logits,
        targets=targets,
        weights=weights,
        attention_mask=attention_mask,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        soft_targets=soft_targets,
        reduction=reduction,
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
        flat_w = weights.reshape(-1).astype(jnp.float32)
    elif soft_targets is None and targets is not None:
        flat_w = (targets.reshape(-1) != ignore_index).astype(jnp.float32)
    else:
        flat_w = jnp.ones(logits.shape[:-1], dtype=jnp.float32).reshape(-1)
    if attention_mask is not None:
        flat_w = flat_w * attention_mask.reshape(-1).astype(jnp.float32)
    weight_sum = jax.lax.stop_gradient(flat_w.sum())

    if z_loss > 0.0:
        lse = jax.scipy.special.logsumexp(logits, axis=-1)
        per_row_zterm = z_loss * lse * lse
        if reduction == "none":
            z_loss_metric = per_row_zterm.astype(jnp.float32)
        elif reduction == "sum":
            z_loss_metric = jnp.sum(per_row_zterm * flat_w.reshape(logits.shape[:-1]))
        else:
            z_loss_metric = jnp.sum(per_row_zterm * flat_w.reshape(logits.shape[:-1])) / jnp.maximum(weight_sum, 1e-8)
        z_loss_metric = jax.lax.stop_gradient(z_loss_metric)
    else:
        z_loss_metric = jnp.zeros((), dtype=jnp.float32)

    inferred_vocab_parallel_axis = vocab_parallel_axis
    if inferred_vocab_parallel_axis is None and in_specs is not None and len(in_specs) > 0:
        inferred_vocab_parallel_axis = _infer_vocab_axis(in_specs[0])
    accuracy: Array | None
    is_per_row_sentinel = soft_targets is not None or targets is None or inferred_vocab_parallel_axis is not None
    if is_per_row_sentinel:
        accuracy = None
    else:
        per_row_correct = jax.lax.stop_gradient(per_row_correct)
        if reduction == "none":
            accuracy = per_row_correct
        else:
            # Weight-weighted accuracy: matches the per-token weighting of the loss
            # (sum(correct * w) / sum(w)). Identical to a plain correct/active count for
            # binary masks, and consistent with importance weights for non-binary weights.
            # per_row_correct is per-row; flatten to match the flat per-token weights.
            num_correct = jnp.sum(per_row_correct.reshape(-1) * flat_w)
            num_active = jnp.maximum(flat_w.sum(), 1e-8)
            accuracy = num_correct / num_active

    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss_metric,
        weight_sum=weight_sum,
        accuracy=accuracy,
    )
