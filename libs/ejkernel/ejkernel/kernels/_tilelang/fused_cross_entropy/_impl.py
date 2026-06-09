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

"""JAX glue around the tile-lang fused cross-entropy prim_funcs.

The kernel set is exposed through ``fused_cross_entropy_tilelang``:

* ``vocab_parallel_axis is None``: single-stage fused kernel that returns
  per-row loss + dlogits in one sweep. Backward is a broadcast multiply.
* ``vocab_parallel_axis="<name>"``: two-stage kernel pair driven by
  ``pmax`` / ``psum`` collectives over the TP mesh axis. Must be called
  inside ``shard_map``.

In both cases the forward writes ``dlogits`` already scaled by the
per-row weight, so the backward is purely a broadcast multiply against
the upstream cotangent.
"""

from __future__ import annotations

import threading
from functools import partial

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_ce_bwd_dense_prim_func,
    make_ce_bwd_prim_func,
    make_ce_dlogits_prim_func,
    make_ce_fwd_dense_prim_func,
    make_ce_fwd_only_prim_func,
    make_ce_partial_stats_prim_func,
    make_fused_ce_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_FWD_ONLY_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_STATS_CACHE: dict[tuple, callable] = {}
_DLOGITS_CACHE: dict[tuple, callable] = {}
_FWD_DENSE_CACHE: dict[tuple, callable] = {}
_BWD_DENSE_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


_DEFAULT_BLOCK_V: int = 1024
_DEFAULT_BLOCK_M: int = 1
_DEFAULT_THREADS: int = 128


def _get_fwd(num_rows: int, vocab_size: int, dtype, ignore_index: int):
    """Retrieve (compiling on first call) the legacy fused-CE forward FFI callable.

    Kept for completeness but no longer used by the default custom-VJP path,
    which calls the split ``fwd_only`` + ``bwd`` pair below.
    """
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_size, bv, str(jnp.dtype(dtype)), int(ignore_index))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fused_ce_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            ignore_index=int(ignore_index),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_fwd_only(
    num_rows: int,
    vocab_size: int,
    dtype,
    ignore_index: int,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    *,
    block_v: int,
    block_m: int,
    threads: int = _DEFAULT_THREADS,
):
    """Lean forward kernel: returns ``(loss[N], lse[N], correct[N])``.

    ``block_v`` / ``block_m`` are **required** — the operation layer
    (``FusedCrossEntropyConfig.block_v`` / ``.block_m``) supplies them
    via ``heuristic_cfg`` or the autotuner. The kernel does not pick
    from shape. The kernel cache key includes both so the autotuner
    can sweep multiple block sizes independently.
    """
    bv = int(block_v)
    bm = int(block_m)
    if num_rows % bm != 0:
        bm = 1
    threads = int(threads)
    key = (
        num_rows,
        vocab_size,
        bv,
        bm,
        threads,
        str(jnp.dtype(dtype)),
        int(ignore_index),
        float(label_smoothing),
        float(z_loss),
    )
    with _LOCK:
        cached = _FWD_ONLY_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_fwd_only_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            ignore_index=int(ignore_index),
            label_smoothing=float(label_smoothing),
            z_loss=float(z_loss),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_ONLY_CACHE[key] = ffi
        return ffi


def _get_bwd(
    num_rows: int,
    vocab_size: int,
    dtype,
    ignore_index: int,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    *,
    block_v: int,
    block_m: int,
    threads: int = _DEFAULT_THREADS,
):
    """Backward kernel: reads ``(logits, lse, targets, weights, dy)``; writes ``dlogits``.

    ``block_v`` / ``block_m`` are caller-supplied (the operation layer's
    autotune); same ``label_smoothing`` / ``z_loss`` build-time
    constants as the forward kernel so the smoothed-target / z-loss
    gradient term collapses to the canonical ``softmax - onehot`` when
    both are 0.
    """
    bv = int(block_v)
    bm = int(block_m)
    if num_rows % bm != 0:
        bm = 1
    threads = int(threads)
    key = (
        num_rows,
        vocab_size,
        bv,
        bm,
        threads,
        str(jnp.dtype(dtype)),
        int(ignore_index),
        float(label_smoothing),
        float(z_loss),
    )
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_bwd_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            ignore_index=int(ignore_index),
            label_smoothing=float(label_smoothing),
            z_loss=float(z_loss),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_fwd_dense(num_rows: int, vocab_size: int, dtype, z_loss: float = 0.0):
    """Dense-target forward kernel: reads ``(logits, soft_targets, weights)`` → ``(loss, lse)``."""
    bv = _DEFAULT_BLOCK_V
    bm = _DEFAULT_BLOCK_M
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = (num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)), float(z_loss))
    with _LOCK:
        cached = _FWD_DENSE_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_fwd_dense_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            z_loss=float(z_loss),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_DENSE_CACHE[key] = ffi
        return ffi


def _get_bwd_dense(num_rows: int, vocab_size: int, dtype, z_loss: float = 0.0):
    """Dense-target backward kernel: ``(logits, soft_targets, lse, weights, dy)`` → ``dlogits``."""
    bv = _DEFAULT_BLOCK_V
    bm = _DEFAULT_BLOCK_M
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = (num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)), float(z_loss))
    with _LOCK:
        cached = _BWD_DENSE_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_bwd_dense_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            z_loss=float(z_loss),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_DENSE_CACHE[key] = ffi
        return ffi


def _get_partial_stats(num_rows: int, vocab_local: int, dtype):
    """Compile/cache the per-shard CE stats kernel (vocab-parallel mode)."""
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_local, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _STATS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_partial_stats_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_local,
            block_v=bv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _STATS_CACHE[key] = ffi
        return ffi


def _get_dlogits(num_rows: int, vocab_local: int, dtype):
    """Compile/cache the per-shard CE dlogits writer (vocab-parallel mode)."""
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_local, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _DLOGITS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ce_dlogits_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_local,
            block_v=bv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows, vocab_local), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _DLOGITS_CACHE[key] = ffi
        return ffi


def _flatten_logits(logits: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten ``logits`` to ``(N, V)`` and remember the original leading shape."""
    if logits.ndim < 2:
        raise ValueError(f"fused_cross_entropy expects rank>=2 logits; got shape {logits.shape}")
    leading = logits.shape[:-1]
    return logits.reshape(-1, logits.shape[-1]), leading


def _per_token_weights(
    targets: jax.Array,
    weights: jax.Array | None,
    ignore_index: int,
) -> jax.Array:
    """Build the per-token float32 weight vector consumed by the kernel."""
    if weights is None:
        mask = (targets != ignore_index).astype(jnp.float32)
        return mask
    return weights.astype(jnp.float32)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _fused_ce_core(
    logits_2d: jax.Array,
    targets_1d: jax.Array,
    weights_1d: jax.Array,
    ignore_index: int,
    label_smoothing: float,
    z_loss: float,
    block_v: int,
    block_m: int,
) -> tuple[jax.Array, jax.Array]:
    """Per-row fused cross-entropy on a flattened ``(N, V)`` logits matrix.

    Returns ``(per_row_loss, correct)`` from one kernel launch. All
    static args (``label_smoothing``, ``z_loss``, ``block_v``,
    ``block_m``) are folded into the prim_func at build time; the
    kernel cache keys on them so flipping ``block_v`` between calls
    triggers a recompile (intentional — that's how the autotuner
    sweeps).
    """
    n, v = logits_2d.shape
    fwd_ffi = _get_fwd_only(
        n,
        v,
        logits_2d.dtype,
        ignore_index,
        label_smoothing,
        z_loss,
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, _lse, correct = fwd_ffi(logits_2d, targets_1d.astype(jnp.int32), weights_1d)
    return loss, correct


def _ce_fwd(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m):
    n, v = logits_2d.shape
    fwd_ffi = _get_fwd_only(
        n,
        v,
        logits_2d.dtype,
        ignore_index,
        label_smoothing,
        z_loss,
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, lse, correct = fwd_ffi(logits_2d, targets_1d.astype(jnp.int32), weights_1d)
    return (loss, correct), (logits_2d, lse, targets_1d, weights_1d)


def _ce_bwd(ignore_index, label_smoothing, z_loss, block_v, block_m, residual, dy):
    logits_2d, lse, targets_1d, weights_1d = residual
    dy_loss, _dy_correct = dy
    n, v = logits_2d.shape
    bwd_ffi = _get_bwd(
        n,
        v,
        logits_2d.dtype,
        ignore_index,
        label_smoothing,
        z_loss,
        block_v=int(block_v),
        block_m=int(block_m),
    )
    dlogits = bwd_ffi(
        logits_2d,
        lse.astype(jnp.float32),
        targets_1d.astype(jnp.int32),
        weights_1d,
        dy_loss.astype(jnp.float32),
    )
    return (dlogits, None, None)


_fused_ce_core.defvjp(_ce_fwd, _ce_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_ce_core_dense(
    logits_2d: jax.Array,
    soft_targets_2d: jax.Array,
    weights_1d: jax.Array,
    z_loss: float,
) -> jax.Array:
    """Per-row fused cross-entropy with **dense** soft targets.

    SoftTargets are cast to the logits dtype before the kernel call —
    the kernel uses a homogeneous SMEM dtype to dodge a TileLang
    ThreadSync planner crash on mixed-dtype SMEM tiles.

    ``z_loss`` is a static Python float (folded into the kernel at
    build time).
    """
    n, v = logits_2d.shape
    fwd_ffi = _get_fwd_dense(n, v, logits_2d.dtype, z_loss)
    loss, _lse = fwd_ffi(logits_2d, soft_targets_2d.astype(logits_2d.dtype), weights_1d)
    return loss


def _ce_dense_fwd(logits_2d, soft_targets_2d, weights_1d, z_loss):
    n, v = logits_2d.shape
    fwd_ffi = _get_fwd_dense(n, v, logits_2d.dtype, z_loss)
    soft_cast = soft_targets_2d.astype(logits_2d.dtype)
    loss, lse = fwd_ffi(logits_2d, soft_cast, weights_1d)
    return loss, (logits_2d, soft_cast, lse, weights_1d)


def _ce_dense_bwd(z_loss, residual, dy):
    logits_2d, soft_targets_2d, lse, weights_1d = residual
    n, v = logits_2d.shape
    bwd_ffi = _get_bwd_dense(n, v, logits_2d.dtype, z_loss)
    dlogits = bwd_ffi(
        logits_2d,
        soft_targets_2d,
        lse.astype(jnp.float32),
        weights_1d,
        dy.astype(jnp.float32),
    )
    log_softmax = logits_2d.astype(jnp.float32) - lse[:, None]
    factor = (weights_1d * dy).astype(jnp.float32)
    dsoft = (-factor[:, None] * log_softmax).astype(soft_targets_2d.dtype)
    return (dlogits, dsoft, None)


_fused_ce_core_dense.defvjp(_ce_dense_fwd, _ce_dense_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _fused_ce_core_tp(
    logits_2d: jax.Array,
    targets_1d: jax.Array,
    weights_1d: jax.Array,
    vocab_start_arr: jax.Array,
    vocab_axis: str,
) -> jax.Array:
    """Vocab-parallel CE on a per-shard ``(N, V_local)`` matrix."""
    n, v_local = logits_2d.shape
    stats_ffi = _get_partial_stats(n, v_local, logits_2d.dtype)
    local_max, local_se, local_tl = stats_ffi(logits_2d, targets_1d.astype(jnp.int32), vocab_start_arr)
    global_max = jax.lax.pmax(local_max, vocab_axis)
    scaled_local_se = local_se * jnp.exp(local_max - global_max)
    global_se = jax.lax.psum(scaled_local_se, vocab_axis)
    global_target_logit = jax.lax.psum(local_tl, vocab_axis)
    lse = jnp.log(global_se) + global_max
    per_row = (lse - global_target_logit) * weights_1d
    return per_row.astype(jnp.float32)


def _ce_tp_fwd(logits_2d, targets_1d, weights_1d, vocab_start_arr, vocab_axis):
    """VJP primal for vocab-parallel CE. Returns (per_row, residual)."""
    n, v_local = logits_2d.shape
    stats_ffi = _get_partial_stats(n, v_local, logits_2d.dtype)
    local_max, local_se, local_tl = stats_ffi(logits_2d, targets_1d.astype(jnp.int32), vocab_start_arr)
    global_max = jax.lax.pmax(local_max, vocab_axis)
    scaled_local_se = local_se * jnp.exp(local_max - global_max)
    global_se = jax.lax.psum(scaled_local_se, vocab_axis)
    global_target_logit = jax.lax.psum(local_tl, vocab_axis)
    lse = jnp.log(global_se) + global_max
    per_row = (lse - global_target_logit) * weights_1d
    residual = (logits_2d, global_max, global_se, targets_1d, vocab_start_arr, weights_1d)
    return per_row.astype(jnp.float32), residual


def _ce_tp_bwd(vocab_axis, residual, dy):
    """Backward for vocab-parallel CE. Local kernel only — no collectives."""
    del vocab_axis
    logits_2d, global_max, global_se, targets_1d, vocab_start_arr, weights_1d = residual
    n, v_local = logits_2d.shape
    dlogits_ffi = _get_dlogits(n, v_local, logits_2d.dtype)
    scaled_weights = (weights_1d * dy).astype(jnp.float32)
    dlogits = dlogits_ffi(
        logits_2d,
        global_max.astype(jnp.float32),
        global_se.astype(jnp.float32),
        targets_1d.astype(jnp.int32),
        vocab_start_arr,
        scaled_weights,
    )
    return (dlogits, None, None, None)


_fused_ce_core_tp.defvjp(_ce_tp_fwd, _ce_tp_bwd)


def fused_cross_entropy_tilelang(
    logits: jax.Array,
    targets: jax.Array,
    weights: jax.Array | None = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets: jax.Array | None = None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
) -> jax.Array:
    """Tile-lang fused cross-entropy with analytic backward.

    Two target modes:
      * **Sparse** (default): ``targets`` are integer token ids of shape
        ``logits.shape[:-1]``. Supports ``label_smoothing`` and ``z_loss``
        regularisation, both folded into the kernel at build time so the
        no-op path stays free.
      * **Dense**: pass ``soft_targets`` of shape ``logits.shape`` (full
        per-row probability distribution). ``targets`` is ignored in this
        mode. Use this for distillation, mixup, or arbitrary soft labels.
        ``label_smoothing`` cannot also be supplied (it would double-apply).

    Args:
        logits: ``(..., V)`` predicted logits.
        targets: ``(...,)`` int token ids (sparse mode). Ignored if
            ``soft_targets`` is provided.
        weights: Optional ``(...,)`` per-token weights (e.g. a loss mask).
            When ``None`` in sparse mode a 0/1 mask is built from
            ``targets != ignore_index``. Required to be non-None in dense
            mode (pass ``jnp.ones(...)`` if you have no mask).
        ignore_index: Target value that disables a position (sparse only).
        label_smoothing: Smoothing coefficient ``α ∈ [0, 1)``. The target
            distribution is ``p[target] = 1 - α``, ``p[v ≠ target] =
            α / (V - 1)``. Static (kernel rebuilds when changed).
        z_loss: Coefficient for ``z_loss · lse²`` regularisation
            (Mesh-TF / PaLM-style auxiliary loss). Static.
        soft_targets: Optional ``(..., V)`` dense probability targets.
            Switches to the dense kernel pair.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``. ``"mean"`` divides
            by ``sum(weights)`` (the number of active tokens).
        vocab_parallel_axis: Mesh axis along which ``V`` is sharded
            (sparse mode only — dense doesn't have a TP variant yet).

    Returns:
        Scalar for ``"mean"`` / ``"sum"``; otherwise ``logits.shape[:-1]``.

    Raises:
        RuntimeError: If tilelang / jax_tvm_ffi are unavailable.
        ValueError: For invalid argument combinations.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang fused_cross_entropy requires both `tilelang` and `jax_tvm_ffi`.")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1); got {label_smoothing}")
    if z_loss < 0.0:
        raise ValueError(f"z_loss must be non-negative; got {z_loss}")

    flat_logits, leading = _flatten_logits(logits)

    if soft_targets is not None:
        if label_smoothing > 0.0:
            raise ValueError(
                "`label_smoothing` cannot combine with `soft_targets` — apply smoothing to the target "
                "distribution before passing it in."
            )
        if vocab_parallel_axis is not None:
            raise ValueError("Dense (soft-target) CE does not support vocab-parallel mode yet.")
        if soft_targets.shape != logits.shape:
            raise ValueError(f"soft_targets.shape={soft_targets.shape} must equal logits.shape={logits.shape}")
        flat_soft = soft_targets.reshape(-1, soft_targets.shape[-1])
        if weights is None:
            flat_weights = jnp.ones(flat_logits.shape[0], dtype=jnp.float32)
        else:
            flat_weights = weights.reshape(-1).astype(jnp.float32)
        per_row_loss = _fused_ce_core_dense(flat_logits, flat_soft, flat_weights, float(z_loss))
        per_row_correct = jnp.full(flat_logits.shape[0], -1.0, dtype=jnp.float32)
    else:
        flat_targets = targets.reshape(-1)
        flat_weights = _per_token_weights(flat_targets, None if weights is None else weights.reshape(-1), ignore_index)
        if vocab_parallel_axis is None:
            per_row_loss, per_row_correct = _fused_ce_core(
                flat_logits,
                flat_targets,
                flat_weights,
                ignore_index,
                float(label_smoothing),
                float(z_loss),
                int(block_v),
                int(block_m),
            )
        else:
            if label_smoothing > 0.0 or z_loss > 0.0:
                raise NotImplementedError(
                    "label_smoothing / z_loss are not yet wired through the vocab-parallel CE kernels."
                )
            v_local = flat_logits.shape[-1]
            tp_idx = jax.lax.axis_index(vocab_parallel_axis).astype(jnp.int32)
            vocab_start_arr = (tp_idx * jnp.asarray(v_local, dtype=jnp.int32))[None]
            per_row_loss = _fused_ce_core_tp(
                flat_logits, flat_targets, flat_weights, vocab_start_arr, vocab_parallel_axis
            )
            per_row_correct = jnp.full(flat_logits.shape[0], -1.0, dtype=jnp.float32)

    if reduction == "none":
        return per_row_loss.reshape(leading), per_row_correct.reshape(leading)
    total = jnp.sum(per_row_loss)
    if reduction == "sum":
        loss_out = total
    else:
        denom = jnp.maximum(jnp.sum(flat_weights), 1e-8)
        loss_out = total / denom
    return loss_out, per_row_correct.reshape(leading)
