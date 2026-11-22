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

"""Public fused-MLP operation: one surface for inference and training.

``fused_mlp`` computes ``down(act(gate(x)) * up(x))`` with:

* **formats** — bf16 dense; channelwise int8; channelwise int4 (optionally
  bit-packed for the TPU decode kernel);
* **layouts** — gate/up separate, fused-concat, or TP-interleaved (normalized
  by :func:`split_gate_up` before any kernel runs);
* **activations** — the shared table (silu, gelu, gelu_tanh, relu, sigmoid);
* **forward** — dispatched through :class:`FusedMlp` (a ``Kernel`` over the
  registry): op id ``"fused_mlp"`` (XLA composition, any format) or
  ``"fused_mlp_w4a4"`` (single-dispatch Pallas decode kernel on TPU, with a
  tile-exact XLA reference registered as the fallback);
* **backward** — format-dependent, chosen by measurement:
  **dense** weights run the plain composition under JAX autodiff (XLA's
  saved-residual schedule measured strictly better than a recompute
  ``custom_vjp``: 0.78x); **integer-code** weights use a ``custom_vjp`` —
  naive autodiff through the quantized forward produces garbage gradients
  (cosine 0.03 vs ground truth, measured) — that recomputes pre-activations
  through the fast channelwise forward and, under ``quantize_activations``,
  dynamically quantizes the cotangents per token so the transposed products
  run on the int8 MXU. Integer codes are frozen (``float0`` cotangents) with
  correct ``dx`` (cosine 0.999 vs dequantized ground truth) — the
  LoRA / frozen-backbone training contract.

Gradient math (concat convention, ``a = x Wg``, ``b = x Wu``,
``h = act(a) * b``, ``y = h Wd``):

    dh  = gy Wd^T          dWd = h^T gy
    da  = dh * b * act'(a) db  = dh * act(a)
    dx  = da Wg^T + db Wu^T
    dWg = x^T da           dWu = x^T db

For channelwise-quantized weights every ``W`` above is ``codes * scale``; the
per-output-channel scale folds into the *cotangent* side of each transposed
product, so the transposed dots run on the raw integer codes with the same
fused-upcast behavior as the forward.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import typing as tp

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array

from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import pack_int4_adjacent
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.fused_mlp import resolve_mlp_combine, split_gate_up
from ejkernel.kernels._xla.quantized_matmul import channelwise_quantized_matmul
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
from .configs import FusedMlpConfig

__all__ = [
    "FusedMlp",
    "channelwise_quantized_matmul",
    "fused_mlp",
    "pack_int4_adjacent",
    "resolve_mlp_combine",
    "split_gate_up",
]


class FusedMlp(Kernel[FusedMlpConfig, Array]):
    """Fused MLP block dispatched through the kernel registry.

    Two variants share the class, mirroring :class:`GroupedMatmul`'s
    version pattern:

    * ``variant="reference"`` → op id ``"fused_mlp"`` — the XLA composition
      for dense / channelwise-integer formats, differentiable.
    * ``variant="w4a4"`` → op id ``"fused_mlp_w4a4"`` — bit-packed int4
      weights and int4 activations; the single-dispatch Pallas kernel on TPU
      (``Platform.PALLAS``/``Backend.TPU``), the tile-exact XLA reference
      anywhere else. Inference-only.
    """

    def __init__(self, variant: tp.Literal["reference", "w4a4", "bf16"] = "reference"):
        """Select the op id for registry lookup.

        Args:
            variant: ``"reference"``, ``"w4a4"`` or ``"bf16"``.

        Raises:
            ValueError: On an unknown variant.
        """
        match variant:
            case "reference":
                op_id = "fused_mlp"
            case "w4a4":
                op_id = "fused_mlp_w4a4"
            case "bf16":
                op_id = "fused_mlp_bf16"
            case _:
                raise ValueError(f"Unsupported fused_mlp variant: {variant}")
        super().__init__(op_id=op_id)

    def get_impl(self, cfg: FusedMlpConfig):
        """Resolve the implementation from the kernel registry.

        Args:
            cfg: Configuration carrying platform/backend selection.

        Returns:
            The registered callable for ``(op_id, platform, backend)``.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(self, *args, cfg: FusedMlpConfig, **kwargs) -> Array:
        """Execute the selected implementation.

        Args:
            *args: Positional arguments for the implementation.
            cfg: Resolved configuration. ``tile_i`` is forwarded to the W4A4
                variant (semantic there: hidden re-quantization block width);
                ``prefill_threshold`` to the reference variant.
            **kwargs: Keyword arguments for the implementation.

        Returns:
            The MLP output ``[m, k]``.
        """
        impl = self.get_impl(cfg)
        if self.op_id == "fused_mlp_w4a4":
            kwargs.setdefault("tile_i", cfg.tile_i)
        elif self.op_id == "fused_mlp_bf16":
            kwargs.setdefault("tile_i", cfg.tile_i)
            kwargs.setdefault("tile_m", cfg.tile_m)
        else:
            kwargs.setdefault("prefill_threshold", cfg.prefill_threshold)
        return impl(*args, **kwargs)

    def heuristic_cfg(self, inv: Invocation[FusedMlpConfig, Array]) -> FusedMlpConfig:
        """Default configuration (defaults are the measured TPU v5 optima).

        Args:
            inv: Invocation context (unused — the defaults are shape-stable).

        Returns:
            The default :class:`FusedMlpConfig`.
        """
        del inv
        return FusedMlpConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[FusedMlpConfig, Array]):
        """Autotune candidates: ``tile_i`` variants for the W4A4 kernel.

        Args:
            inv: Invocation context.

        Returns:
            Iterable of candidate configurations.
        """
        if self.op_id == "fused_mlp_w4a4":
            return [FusedMlpConfig(platform="auto", backend="any", tile_i=tile) for tile in (512, 1024, 2048)]
        if self.op_id == "fused_mlp_bf16":
            return [
                FusedMlpConfig(platform="auto", backend="any", tile_i=ti, tile_m=tm)
                for ti in (512, 1024)
                for tm in (256, 512)
            ]
        return [self.heuristic_cfg(inv)]


_fused_mlp_executor: Executor[FusedMlpConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            # The W4A4 variant takes int4 primals; gradient-based backward
            # validation cannot differentiate integer inputs (measured crash
            # on grouped-matmul autotune with integer rhs), so it stays off.
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("fused-mlp"),
    )
)


def _forward_reference(
    x,
    weights,
    scales,
    *,
    activation,
    act_params=(),
    quantize_activations=False,
    activation_bits=8,
    prefill_threshold=256,
    platform=None,
):
    """Forward through the registry-dispatched XLA composition."""
    w_gate, w_up, w_down = weights
    gate_scale, up_scale, down_scale = scales
    return _fused_mlp_executor(
        FusedMlp("reference"),
        x,
        w_gate,
        w_up,
        w_down,
        gate_scale,
        up_scale,
        down_scale,
        activation=activation,
        act_params=act_params,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        _cfg=FusedMlpConfig(platform=platform or "auto", backend="any", prefill_threshold=prefill_threshold),
    )


def _spec_axes(entry) -> tuple[str, ...]:
    """Flatten one PartitionSpec entry to a tuple of axis names."""
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    return tuple(a for a in entry if a is not None)


def _axes_size(mesh, axes: tuple[str, ...]) -> int:
    size = 1
    for a in axes:
        size *= mesh.shape[a]
    return size


def _largest_tile(dim: int, cap: int, multiple: int) -> int | None:
    """Largest divisor of ``dim`` that is a multiple of ``multiple``, <= cap."""
    best = None
    for cand in range(multiple, min(dim, cap) + 1, multiple):
        if dim % cand == 0:
            best = cand
    return best


@dataclasses.dataclass(frozen=True)
class _DenseShardedPlan:
    """Static routing decision for the dense bf16 Pallas path.

    Hashable so it can ride ``custom_vjp`` nondiff argnums; carries the mesh,
    the operand specs exactly as given, the tp axes to ``psum`` over, and the
    resolved tiling.
    """

    mesh: tp.Any
    x_spec: tp.Any
    wf_spec: tp.Any
    wd_spec: tp.Any
    rows_axes: tuple[str, ...]
    tp_axes: tuple[str, ...]
    activation: str
    act_params: tuple[float, ...]
    tile_m: int
    tile_i: int
    layout: str
    segments: int
    out_spec: tp.Any = None


def resolve_dense_sharded_plan(
    x2d: jax.Array,
    gate_up: jax.Array,
    w_down: jax.Array,
    *,
    mesh,
    in_specs,
    activation: str,
    act_params: tuple[float, ...] = (),
    layout: str = "concat",
    segments: int = 1,
    min_rows_per_shard: int = 512,
    out_spec=None,
) -> _DenseShardedPlan | None:
    """Decide whether the given shardings can run the fused Pallas kernel.

    The decision policy is measurement-driven ("never slower than XLA"):

    * rows (dp/fsdp/sp on the token axis) — shard-local, supported;
    * tp on the intermediate axis — kernel-local columns + ``psum``, the
      same collective XLA's TP schedule pays, supported;
    * K-sharded weights (fsdp storage sharding) — **rejected**: execution
      needs a gather and XLA's GSPMD collective-matmul overlaps it in a way
      a shard_map'd kernel cannot (measured: every materialized-gather
      integration lands ~4% behind XLA on the 4B model);
    * small per-shard row counts — rejected: below prefill scale XLA's
      composition sits at the DMA roofline and the extra dispatch loses.

    Args:
        x2d: Token-major activations ``[m, k]``.
        gate_up: Fused weight ``[k, 2i]``.
        w_down: Down projection ``[i, k]``.
        mesh: The device mesh the specs refer to.
        in_specs: ``(x_spec, gate_up_spec, down_spec)`` PartitionSpecs.
        activation: Gate-branch activation (validated by the kernel).
        min_rows_per_shard: Reject when the local row count is below this.

    Returns:
        A plan, or ``None`` when the XLA composition should run instead.
    """
    if mesh is None or in_specs is None or jax.default_backend() != "tpu":
        return None
    if not jnp.issubdtype(gate_up.dtype, jnp.floating):
        return None
    try:
        resolve_mlp_combine(activation, tuple(act_params))
    except ValueError:
        return None
    x_spec, wf_spec, wd_spec = in_specs

    def live_axes(spec, idx):
        # Mesh axes of size 1 appear in EasyDeL's specs but shard nothing;
        # treating them as sharding would reject every real layout.
        names = _spec_axes(spec[idx] if len(spec) > idx else None)
        return tuple(a for a in names if mesh.shape[a] > 1)

    rows_axes = live_axes(x_spec, 0)
    x_k_axes = live_axes(x_spec, 1)
    wf_k_axes = live_axes(wf_spec, 0)
    wf_i_axes = live_axes(wf_spec, 1)
    wd_i_axes = live_axes(wd_spec, 0)
    wd_k_axes = live_axes(wd_spec, 1)

    if x_k_axes or wf_k_axes or wd_k_axes:
        # Contraction-dim (K) sharding: storage-style fsdp — XLA composition
        # keeps GSPMD's overlapped collective matmul.
        return None
    if set(wf_i_axes) != set(wd_i_axes):
        return None
    tp_axes = tuple(wf_i_axes)
    if tp_axes:
        # tp engagement measured a TIE forward and a loss on the train step
        # at 4B shapes (psum-dominated; backward re-fusion churn) — the
        # never-slower rule routes tp to the XLA composition. The sharded
        # machinery below still supports tp (covered by tests) for
        # re-evaluation at larger per-shard shapes.
        return None

    # Layout gate: the kernel reads its LOCAL shard as a concat [gate | up]
    # pair. A concat GLOBAL layout sharded on the intermediate axis puts all
    # gate columns on the low ranks and all up columns on the high ranks —
    # locally garbage. Only two combinations are locally-concat:
    tp_size = _axes_size(mesh, tp_axes)
    if layout == "concat":
        if tp_axes or segments != 1:
            return None
    elif layout == "interleaved":
        if tp_size != segments:
            return None
    else:
        return None

    m_local = x2d.shape[0] // max(1, _axes_size(mesh, rows_axes))
    i_local = (gate_up.shape[1] // 2) // max(1, _axes_size(mesh, tp_axes))
    if m_local < min_rows_per_shard:
        return None
    tile_m = _largest_tile(m_local, 512, 8)
    tile_i = _largest_tile(i_local, 1152, 128)
    if tile_m is None or tile_i is None:
        return None
    return _DenseShardedPlan(
        mesh=mesh,
        x_spec=x_spec,
        wf_spec=wf_spec,
        wd_spec=wd_spec,
        rows_axes=tuple(rows_axes),
        tp_axes=tp_axes,
        activation=activation,
        act_params=tuple(act_params),
        tile_m=tile_m,
        tile_i=tile_i,
        layout=layout,
        segments=segments,
        out_spec=out_spec,
    )


def _dense_sharded_call(plan: _DenseShardedPlan, x2d, wf, wd, *, save_preacts: bool):
    """shard_map the registry-dispatched bf16 kernel per the plan."""
    jmesh = getattr(plan.mesh, "jax_mesh", plan.mesh)
    rows = plan.x_spec[0] if len(plan.x_spec) > 0 else None
    tp_entry = plan.wf_spec[1] if len(plan.wf_spec) > 1 else None
    # The output leaves EXACTLY as sharded as the activations arrived
    # (rows stay sharded; hidden unsharded — guaranteed by the resolver,
    # which rejects hidden-sharded x). Never auto-replicated. Callers with a
    # different downstream layout (e.g. reduce-scattered sequence-parallel)
    # override via ``out_spec``.
    out_spec = plan.out_spec if plan.out_spec is not None else PartitionSpec(rows, None)
    preact_spec = PartitionSpec(rows, tp_entry)
    cfg = FusedMlpConfig(platform="auto", backend="any", tile_m=plan.tile_m, tile_i=plan.tile_i)

    def local(x, wf, wd):
        result = _fused_mlp_executor(
            FusedMlp("bf16"),
            x,
            wf,
            wd,
            activation=plan.activation,
            act_params=plan.act_params,
            save_preacts=save_preacts,
            _cfg=cfg,
        )
        if save_preacts:
            out, g16, u16 = result
        else:
            out, g16, u16 = result, None, None
        if plan.tp_axes:
            out = jax.lax.psum(out, plan.tp_axes)
        return (out, g16, u16) if save_preacts else out

    return jax.shard_map(
        local,
        mesh=jmesh,
        in_specs=(plan.x_spec, plan.wf_spec, plan.wd_spec),
        out_specs=(out_spec, preact_spec, preact_spec) if save_preacts else out_spec,
        check_vma=False,
    )(x2d, wf, wd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _dense_pallas_sharded(plan: _DenseShardedPlan, x2d, wf, wd):
    """Differentiable sharded dense fused MLP; see :func:`fused_mlp`."""
    return _dense_sharded_call(plan, x2d, wf, wd, save_preacts=False)


def _dense_sharded_fwd(plan, x2d, wf, wd):
    """Forward rule: the ``save_preacts`` kernel supplies bf16 residuals, so
    the backward is recompute-free — structurally XLA's saved-residual
    schedule (a recompute backward measured ~2% slower at 4B model level).

    Under tp the TRAINING forward runs the XLA composition instead: the tp
    regime is psum-dominated (fusion measured a tie forward and -1.3% on the
    full train step from backward re-fusion churn), so the never-slower rule
    routes differentiation to the composition there. The Pallas kernel
    remains the inference primal for tp.
    """
    if plan.tp_axes:
        combine = resolve_mlp_combine(plan.activation, plan.act_params)
        # Fused dot then ACTIVATION split — the exact schedule the model's
        # XLA path runs; splitting the weight instead materializes ~2x
        # weight bytes per layer (measured +22ms at 4B model level).
        gu = x2d @ wf.astype(x2d.dtype)
        g, u = split_gate_up(gu, layout=plan.layout, segments=plan.segments)
        out = combine(g.astype(jnp.float32), u.astype(jnp.float32)).astype(x2d.dtype) @ wd.astype(x2d.dtype)
        return out, (x2d, wf, wd, g.astype(jnp.bfloat16), u.astype(jnp.bfloat16))
    out, g16, u16 = _dense_sharded_call(plan, x2d, wf, wd, save_preacts=True)
    return out, (x2d, wf, wd, g16, u16)


def _fuse_gate_up(d_gate: jax.Array, d_up: jax.Array, *, layout: str, segments: int) -> jax.Array:
    """Inverse of :func:`split_gate_up` for gradient re-fusion."""
    if layout == "concat":
        return jnp.concatenate([d_gate, d_up], axis=-1)
    i_dim = d_gate.shape[-1]
    seg = i_dim // segments
    dg = d_gate.reshape(*d_gate.shape[:-1], segments, 1, seg)
    du = d_up.reshape(*d_up.shape[:-1], segments, 1, seg)
    return jnp.concatenate([dg, du], axis=-2).reshape(*d_gate.shape[:-1], 2 * i_dim)


def _dense_sharded_bwd(plan, res, gy):
    """Backward rule: plain XLA math on the GLOBAL (sharded) arrays.

    Every dot below is regular HLO, so GSPMD inserts and overlaps whatever
    collectives the given shardings require — the transposed products under
    tp contract over the sharded intermediate axis and psum automatically.
    The saved pre-activations are CANONICAL ``[m, i]`` (rank chunks assemble
    in canonical column order), so the weight gradient is computed in
    canonical gate/up terms and re-fused into the declared storage layout.
    """
    x, wf, wd, g16, u16 = res
    combine = resolve_mlp_combine(plan.activation, plan.act_params)
    w_gate, w_up = split_gate_up(wf, layout=plan.layout, segments=plan.segments)
    a, b = g16.astype(jnp.float32), u16.astype(jnp.float32)
    h, combine_vjp = jax.vjp(combine, a, b)
    h16 = h.astype(x.dtype)
    gy16 = gy.astype(x.dtype)
    dh = gy16 @ wd.T.astype(x.dtype)
    dWd = h16.T @ gy16
    da, db = combine_vjp(dh.astype(jnp.float32))
    da16, db16 = da.astype(x.dtype), db.astype(x.dtype)
    dx = da16 @ w_gate.T.astype(x.dtype) + db16 @ w_up.T.astype(x.dtype)
    dWf = _fuse_gate_up(x.T @ da16, x.T @ db16, layout=plan.layout, segments=plan.segments)
    return dx, dWf.astype(wf.dtype), dWd.astype(wd.dtype)


_dense_pallas_sharded.defvjp(_dense_sharded_fwd, _dense_sharded_bwd)


def _zero_cotangent(primal: jax.Array) -> jax.Array:
    """Cotangent for a frozen parameter, honoring JAX dtype rules.

    Integer primals take ``float0`` symbolic zeros; float primals (the scale
    arrays, treated as frozen calibration constants) take ordinary zeros.

    Args:
        primal: The primal value the cotangent corresponds to.

    Returns:
        The appropriate zero cotangent.
    """
    import numpy as np

    if jnp.issubdtype(primal.dtype, jnp.integer):
        return np.zeros(primal.shape, dtype=jax.dtypes.float0)
    return jnp.zeros_like(primal)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _fused_mlp_core(
    x: jax.Array,
    weights: tuple[jax.Array, jax.Array, jax.Array],
    scales: tuple[jax.Array | None, jax.Array | None, jax.Array | None],
    activation: str,
    act_params: tuple[float, ...],
    quantize_activations: bool,
    activation_bits: int,
    prefill_threshold: int,
    platform: str | None,
) -> jax.Array:
    """Differentiable core; see :func:`fused_mlp` for the public contract."""
    return _forward_reference(
        x,
        weights,
        scales,
        activation=activation,
        act_params=act_params,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
        platform=platform,
    )


def _core_fwd(
    x, weights, scales, activation, act_params, quantize_activations, activation_bits, prefill_threshold, platform
):
    """Forward rule: save only primals — the backward recomputes the rest."""
    out = _forward_reference(
        x,
        weights,
        scales,
        activation=activation,
        act_params=act_params,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
        platform=platform,
    )
    return out, (x, weights, scales)


def _core_bwd(activation, act_params, quantize_activations, activation_bits, prefill_threshold, platform, residuals, gy):
    """Backward rule: recompute hidden state, differentiate through dequant.

    Args:
        activation: Static activation name.
        quantize_activations: When set, cotangents are dynamically quantized
            per token and the transposed products run on the int8 MXU — and
            the pre-activation recompute below runs the same int-dot regime
            the forward did.
        prefill_threshold: Token count where the forward's integer-dot path
            engaged; the recompute must use the same threshold.
        platform: Unused in backward (the backward is always the XLA
            composition).
        residuals: ``(x, weights, scales)`` from the forward.
        gy: Output cotangent ``[m, k]``.

    Returns:
        Cotangents for ``(x, weights, scales)``.
    """
    del platform
    x, weights, scales = residuals
    w_gate, w_up, w_down = weights
    gate_scale, up_scale, down_scale = scales
    combine = resolve_mlp_combine(activation, act_params)

    # Recompute the pre-activation products through the same fast forward
    # composition (fused-upcast / int8-dot), in bf16 — an f32 recompute
    # measured slower AND heavier than XLA's own residual saving. The
    # quantization knobs MUST be forwarded: with ``quantize_activations`` at
    # prefill sizes the forward produced int8-dot pre-activations, and
    # linearizing ``combine`` at the fused-upcast recompute instead would
    # evaluate the adjoint at the wrong point (~7e-3 relative drift).
    a = channelwise_quantized_matmul(
        x,
        w_gate,
        gate_scale,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
    ).astype(jnp.float32)
    b = channelwise_quantized_matmul(
        x,
        w_up,
        up_scale,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
    ).astype(jnp.float32)
    _, combine_vjp = jax.vjp(combine, a, b)

    gy32 = gy.astype(jnp.float32)

    def transpose_dot(cotangent, w_codes, w_scale):
        """``cotangent @ dequant(w)^T`` — int8 MXU when activations quantize.

        The per-output-channel weight scale multiplies the cotangent BEFORE
        any quantization (it rides the contraction), so the transposed dot
        runs on raw codes. With ``quantize_activations`` the cotangent is
        dynamically quantized per token and the product runs at the int8 MXU
        rate — the backward twin of the forward's int-dot path.
        """
        scaled = cotangent * w_scale.reshape(1, w_codes.shape[-1])
        if quantize_activations:
            c_abs = jnp.max(jnp.abs(scaled), axis=1, keepdims=True)
            c_scale = c_abs / 127.0
            c_q = jnp.clip(jnp.round(scaled / jnp.where(c_scale == 0, 1, c_scale)), -127, 127).astype(jnp.int8)
            w_dot = w_codes if w_codes.dtype == jnp.int8 else w_codes.astype(jnp.int8)
            out = jax.lax.dot_general(
                c_q, w_dot, dimension_numbers=(((1,), (1,)), ((), ())), preferred_element_type=jnp.int32
            )
            return out.astype(jnp.float32) * c_scale
        out = jax.lax.dot_general(
            scaled.astype(jnp.bfloat16),
            w_codes.astype(jnp.bfloat16),
            dimension_numbers=(((1,), (1,)), ((), ())),
        )
        return out.astype(jnp.float32)

    dh = transpose_dot(gy32, w_down, down_scale)
    da, db = combine_vjp(dh)
    dx = transpose_dot(da, w_gate, gate_scale) + transpose_dot(db, w_up, up_scale)
    dx = dx.astype(x.dtype)

    d_weights = tuple(_zero_cotangent(w) for w in weights)
    d_scales = tuple(None if s is None else _zero_cotangent(s) for s in scales)
    return dx, d_weights, d_scales


_fused_mlp_core.defvjp(_core_fwd, _core_bwd)


def fused_mlp(
    x: jax.Array,
    w_gate: jax.Array | None = None,
    w_up: jax.Array | None = None,
    w_down: jax.Array | None = None,
    *,
    gate_up: jax.Array | None = None,
    gate_up_layout: tp.Literal["concat", "interleaved"] = "concat",
    gate_up_segments: int = 1,
    gate_scale: jax.Array | None = None,
    up_scale: jax.Array | None = None,
    down_scale: jax.Array | None = None,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    quantize_activations: bool = False,
    activation_bits: int = 8,
    prefill_threshold: int = 256,
    packed_weights: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    packed_tile_i: int = 512,
    decode_threshold: int = 64,
    platform: tp.Literal["auto", "xla", "pallas"] | None = None,
    mesh: tp.Any = None,
    in_specs: tuple[tp.Any, tp.Any, tp.Any] | None = None,
    out_spec: tp.Any = None,
) -> jax.Array:
    """Fused ``down(act(gate(x)) * up(x))`` for inference and training.

    Two execution regimes behind one call:

    * **Differentiable path** (default): the XLA composition with a
      recompute-based backward. Dense weights train normally; integer-code
      weights are frozen (symbolic-zero gradients) with exact ``dx`` — the
      LoRA / frozen-backbone contract.
    * **Packed decode path**: when ``packed_weights`` is given, the input is
      float, ``m < decode_threshold``, and the platform is TPU, the forward
      runs the single-dispatch Pallas W4A4 kernel (per-token input quant,
      per-(token, tile) hidden quant). Inference-only: it is wrapped in
      ``stop_gradient`` semantics by construction (no vjp), so training calls
      should not pass ``packed_weights``.

    Args:
        x: Input ``[m, k]``, floating dtype.
        w_gate: Gate weight ``[k, i]`` (dense or integer codes). Mutually
            exclusive with ``gate_up``.
        w_up: Up weight ``[k, i]``.
        w_down: Down weight ``[i, k]``.
        gate_up: Fused gate/up weight ``[k, 2i]`` instead of separate arrays.
        gate_up_layout: Fused layout — ``"concat"`` or ``"interleaved"``.
        gate_up_segments: TP segment count for the interleaved layout.
        gate_scale: Channel scale for integer gate weights.
        up_scale: Channel scale for integer up weights. When ``gate_up`` is
            given with one fused scale, pass it as ``gate_scale`` and leave
            this ``None`` — it is split with the weight.
        down_scale: Channel scale for integer down weights.
        activation: Combine name (see
            :func:`ejkernel.kernels._xla.fused_mlp.resolve_mlp_combine` —
            unary activations plus parameterized two-branch combines:
            clamped_swiglu, swiglu_oai, scaled_silu, situ).
        act_params: Static float parameters for parameterized combines.
        quantize_activations: Integer formats: use the int-MXU dot at prefill
            sizes (per-token dynamic activation quantization).
        prefill_threshold: Token count where the integer dot engages.
        packed_weights: ``(gate_packed, up_packed, down_packed)`` uint8 arrays
            in ``pack_int4_adjacent`` layout, enabling the fused decode
            kernel. Requires all three channel scales.
        packed_tile_i: I-tile width for the packed kernel.
        decode_threshold: Max ``m`` for which the packed kernel is used.
        platform: Force an implementation family. ``"xla"`` is the vanilla-XLA
            opt-out: it disables the packed Pallas kernel entirely (packed
            weights are ignored and the integer codes run the channelwise XLA
            composition — the production XLA path, not the unpack-heavy W4A4
            reference). ``None``/``"auto"`` (default) lets capability probes
            decide; ``"pallas"`` states intent but still yields to the
            capability gates (TPU + int4 MXU).
        mesh: Device mesh the ``in_specs`` refer to. Together with
            ``in_specs`` this enables the sharding-aware dense bf16 Pallas
            path: rows (dp/fsdp/sp) run shard-local, tp column shards run
            the kernel locally followed by the same ``psum`` XLA's TP
            schedule pays, and unsupported layouts (K-sharded fsdp weights,
            sub-prefill row counts) fall back to the XLA composition so the
            op is never slower than GSPMD.
        in_specs: ``(x_spec, gate_up_spec, down_spec)`` PartitionSpecs
            describing how the arrays are ACTUALLY sharded. The op consumes
            the given shardings — it never re-replicates weights.
        out_spec: Optional PartitionSpec for the output. Default: the output
            keeps the input's rows sharding with the hidden axis unsharded
            (mirroring the activations as they arrived — never
            auto-replicated). Override for a different downstream layout.

    Returns:
        ``[m, k]`` in ``x``'s dtype.

    Raises:
        ValueError: On inconsistent weight arguments.
    """
    fused_for_pallas = None
    if gate_up is not None:
        if w_gate is not None or w_up is not None:
            raise ValueError("Pass either gate_up or (w_gate, w_up), not both.")
        if gate_up_layout == "concat" and gate_up_segments == 1:
            fused_for_pallas = gate_up
        elif gate_up_layout == "interleaved" and mesh is not None and in_specs is not None:
            # Each tp rank's shard of the interleaved layout is a plain
            # concat pair, so the kernel's column views are locally correct
            # exactly when the intermediate axis is tp-sharded by the same
            # degree the layout was built for.
            tp_axes = _spec_axes(in_specs[1][1] if len(in_specs[1]) > 1 else None)
            if tp_axes and _axes_size(mesh, tp_axes) == gate_up_segments:
                fused_for_pallas = gate_up
        w_gate, w_up = split_gate_up(gate_up, layout=gate_up_layout, segments=gate_up_segments)
        if gate_scale is not None and up_scale is None:
            gate_scale, up_scale = split_gate_up(
                gate_scale.reshape(1, -1), layout=gate_up_layout, segments=gate_up_segments
            )
    if w_gate is None or w_up is None or w_down is None:
        raise ValueError("fused_mlp requires gate, up and down weights (separate or fused).")

    def _int4_mxu_available() -> bool:
        """Capability probe — no TPU generation is hardcoded anywhere."""
        from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import supports_int4_mxu

        return supports_int4_mxu()

    if packed_weights is not None and platform in (None, "auto", "pallas"):
        # Validate eagerly on the host: a None scale would otherwise surface
        # as an opaque AttributeError inside the packed kernel (TPU-only), or
        # silently change which path runs.
        missing = [
            name
            for name, value in (
                ("gate_scale", gate_scale),
                ("up_scale", up_scale),
                ("down_scale", down_scale),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "fused_mlp: packed_weights requires all three channel scales "
                f"(gate_scale, up_scale, down_scale); missing {missing}."
            )
    use_packed = (
        packed_weights is not None
        and platform in (None, "auto", "pallas")
        and x.shape[0] < decode_threshold
        and jax.default_backend() == "tpu"
        and jnp.issubdtype(x.dtype, jnp.floating)
        and _int4_mxu_available()
    )
    if use_packed:
        gate_packed, up_packed, down_packed = packed_weights
        x_abs = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)
        x_scale = x_abs / 7.0
        x4 = jnp.clip(jnp.round(x.astype(jnp.float32) / jnp.where(x_scale == 0, 1, x_scale)), -7, 7).astype(jnp.int4)
        out = _fused_mlp_executor(
            FusedMlp("w4a4"),
            x4,
            gate_packed,
            up_packed,
            down_packed,
            gate_scale,
            up_scale,
            down_scale,
            x_scale,
            activation=activation,
            act_params=tuple(act_params),
            _cfg=FusedMlpConfig(platform=platform or "auto", backend="any", tile_i=packed_tile_i),
        )
        return out.astype(x.dtype)

    if not jnp.issubdtype(w_gate.dtype, jnp.integer):
        if fused_for_pallas is not None and platform in (None, "auto", "pallas"):
            plan = resolve_dense_sharded_plan(
                x,
                fused_for_pallas,
                w_down,
                mesh=mesh,
                in_specs=in_specs,
                activation=activation,
                act_params=tuple(act_params),
                layout=gate_up_layout,
                segments=gate_up_segments,
                out_spec=out_spec,
            )
            if plan is not None:
                return _dense_pallas_sharded(plan, x, fused_for_pallas, w_down)
        # Dense fallback: the plain composition in the input dtype, under JAX
        # autodiff. Two measured lessons: XLA's saved-residual backward beats
        # a recompute custom_vjp (0.78x), and an f32 elementwise upcast costs
        # 3-6% vs computing the activation in bf16 as XLA's own schedule does
        # — so for unary activations this path is bit-identical to the naive
        # composition (and, under GSPMD with sharded operands, IS the
        # collective-matmul path). Parameterized combines run in the same
        # dtype discipline.
        combine = resolve_mlp_combine(activation, tuple(act_params))
        return combine(x @ w_gate, x @ w_up) @ w_down

    return _fused_mlp_core(
        x,
        (w_gate, w_up, w_down),
        (gate_scale, up_scale, down_scale),
        activation,
        tuple(act_params),
        quantize_activations,
        activation_bits,
        prefill_threshold,
        platform,
    )
