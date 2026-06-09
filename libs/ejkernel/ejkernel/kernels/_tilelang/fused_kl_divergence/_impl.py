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

"""JAX glue around the tile-lang fused forward-KL prim_funcs.

The kernel set computes ``KL(softmax(teacher) || softmax(student))``
row-wise together with the gradient w.r.t. the student logits. Two flows:

* ``vocab_parallel_axis is None``: single-stage fused kernel.
* ``vocab_parallel_axis="<name>"``: three-stage flow with collectives —
  ``partial_stats`` → ``pmax/psum`` to get global ``lse_t``/``lse_s`` →
  ``local_loss`` → ``psum`` to get global per-row KL. Backward calls
  ``dstudent`` with the cached ``lse_*``, no collectives.

Backward through the teacher is intentionally not supported — for
distillation the teacher is detached.
"""

from __future__ import annotations

import threading
from functools import partial

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_fused_kl_prim_func,
    make_kl_dstudent_only_prim_func,
    make_kl_dstudent_prim_func,
    make_kl_dstudent_reverse_prim_func,
    make_kl_fwd_only_prim_func,
    make_kl_jsd_fwd_prim_func,
    make_kl_local_loss_prim_func,
    make_kl_partial_stats_prim_func,
    make_kl_single_pass_prim_func,
    make_kl_two_lse_prim_func,
    make_kl_unified_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_FWD_ONLY_CACHE: dict[tuple, callable] = {}
_BWD_DSTUDENT_CACHE: dict[tuple, callable] = {}
_BWD_REVERSE_CACHE: dict[tuple, callable] = {}
_JSD_FWD_CACHE: dict[tuple, callable] = {}
_STATS_CACHE: dict[tuple, callable] = {}
_LOCAL_LOSS_CACHE: dict[tuple, callable] = {}
_DSTUDENT_CACHE: dict[tuple, callable] = {}
_UNIFIED_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


_DEFAULT_BLOCK_V: int = 1024
_DEFAULT_BLOCK_M: int = 1
_DEFAULT_THREADS: int = 128


def _get_two_lse(num_rows: int, vocab_size: int, dtype):
    """Single-pass two-stream online-softmax kernel: returns (lse_t, lse_s)."""
    bv = _DEFAULT_BLOCK_V
    bm = _DEFAULT_BLOCK_M
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = ("two_lse", num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_ONLY_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_two_lse_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
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
        _FWD_ONLY_CACHE[key] = ffi
        return ffi


def _get_kl_single_pass(num_rows: int, vocab_size: int, dtype):
    """Single-pass KL forward — one sweep over T+S, online softmax for
    both streams + incremental KL accumulator with max-correction.

    Cuts forward HBM from 3·N·V (fwd_only) → 2·N·V — the absolute
    minimum to compute lse_t, lse_s and per-row KL while reading each
    logit only once.
    """
    bv = _DEFAULT_BLOCK_V
    bm = _DEFAULT_BLOCK_M
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = ("kl_single_pass", num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_ONLY_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_single_pass_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
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


def _get_kl_unified(num_rows: int, vocab_size: int, dtype):
    """Unified KL fwd+bwd kernel: returns (loss[N], dstudent_unscaled[N, V]).

    Lets the JAX backward shrink to a single broadcast multiply
    ``dy[:, None] * dstudent_unscaled``.
    """
    bv = _DEFAULT_BLOCK_V
    bm = _DEFAULT_BLOCK_M
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = ("kl_unified", num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _UNIFIED_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_unified_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _UNIFIED_CACHE[key] = ffi
        return ffi


def _get_kl_fwd_only(
    num_rows: int,
    vocab_size: int,
    dtype,
    direction: str = "forward",
    temperature: float = 1.0,
    *,
    block_v: int,
    block_m: int,
):
    """Lean fused KL forward (forward or reverse direction).

    ``block_v`` / ``block_m`` are caller-supplied (the operation layer
    ``FusedKLDivergenceConfig`` provides them via ``heuristic_cfg`` /
    autotune). Returns ``(loss[N], lse_t[N], lse_s[N], acc[N])``.
    """
    bv = int(block_v)
    bm = int(block_m)
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = (
        "kl_fwd_only",
        num_rows,
        vocab_size,
        bv,
        bm,
        threads,
        str(jnp.dtype(dtype)),
        direction,
        float(temperature),
    )
    with _LOCK:
        cached = _FWD_ONLY_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_fwd_only_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            direction=direction,
            temperature=float(temperature),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
                jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_ONLY_CACHE[key] = ffi
        return ffi


def _get_kl_jsd_fwd(
    num_rows: int,
    vocab_size: int,
    dtype,
    beta: float,
    temperature: float = 1.0,
    *,
    block_v: int,
    block_m: int,
):
    """JSD-mixture forward kernel. Returns ``(loss[N], lse_t[N], lse_s[N])``.

    ``loss`` is already scaled by ``weight``; the residuals power a
    pure-JAX autodiff backward over recomputed log-probs. ``block_v`` /
    ``block_m`` are caller-supplied — the operation layer
    (``FusedKLDivergenceConfig``) sets them.
    """
    bv = int(block_v)
    bm = int(block_m)
    if num_rows % bm != 0:
        bm = 1
    threads = _DEFAULT_THREADS
    key = (
        "kl_jsd_fwd",
        num_rows,
        vocab_size,
        bv,
        bm,
        threads,
        str(jnp.dtype(dtype)),
        float(beta),
        float(temperature),
    )
    with _LOCK:
        cached = _JSD_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_jsd_fwd_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            beta=float(beta),
            temperature=float(temperature),
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
        _JSD_FWD_CACHE[key] = ffi
        return ffi


def _get_fwd_stats(num_rows: int, vocab_size: int, dtype):
    """Stats kernel: writes ``(max_t, sumexp_t, max_s, sumexp_s)``.

    Compiles in seconds even at V=128K (each pass is a small static IR),
    unlike the all-in-one ``fwd_only`` kernel which blows TileLang's
    compile-time RAM budget at large V.
    """
    bv = _DEFAULT_BLOCK_V
    threads = _DEFAULT_THREADS
    key = ("stats", num_rows, vocab_size, bv, threads, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_ONLY_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_partial_stats_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_size,
            block_v=bv,
            dtype=dtype,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=tuple(jax.ShapeDtypeStruct((num_rows,), jnp.float32) for _ in range(4)),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_ONLY_CACHE[key] = ffi
        return ffi


def _get_fwd_loss(num_rows: int, vocab_size: int, dtype):
    """Local-loss kernel: writes ``loss[N]`` given ``lse_t``/``lse_s``."""
    bv = _DEFAULT_BLOCK_V
    threads = _DEFAULT_THREADS
    key = ("loss", num_rows, vocab_size, bv, threads, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _LOCAL_LOSS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_local_loss_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_size,
            block_v=bv,
            dtype=dtype,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _LOCAL_LOSS_CACHE[key] = ffi
        return ffi


def _get_bwd_dstudent(
    num_rows: int,
    vocab_size: int,
    dtype,
    temperature: float = 1.0,
    *,
    block_v: int,
):
    """Forward-KL backward kernel (``dstudent``).

    ``block_m`` is pinned to 1 (the 2D grid trips a ThreadSync bug);
    ``block_v`` is caller-supplied.
    """
    bv = int(block_v)
    bm = 1
    threads = _DEFAULT_THREADS
    key = (num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)), float(temperature))
    with _LOCK:
        cached = _BWD_DSTUDENT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_dstudent_only_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            temperature=float(temperature),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_DSTUDENT_CACHE[key] = ffi
        return ffi


def _get_bwd_reverse(
    num_rows: int,
    vocab_size: int,
    dtype,
    temperature: float = 1.0,
    *,
    block_v: int,
):
    """Reverse-KL backward kernel.

    Reads ``(student, teacher, lse_s, acc, weights, dy)`` and writes
    ``dstudent = factor · p_s · [(s - t)/T - acc]``. ``block_m`` pinned
    to 1; ``block_v`` is caller-supplied.
    """
    bv = int(block_v)
    bm = 1
    threads = _DEFAULT_THREADS
    key = (num_rows, vocab_size, bv, bm, threads, str(jnp.dtype(dtype)), float(temperature))
    with _LOCK:
        cached = _BWD_REVERSE_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_dstudent_reverse_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
            temperature=float(temperature),
            block_m=bm,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows, vocab_size), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_REVERSE_CACHE[key] = ffi
        return ffi


def _get_fwd(num_rows: int, vocab_size: int, dtype):
    """Retrieve (compiling on first call) the LEGACY fused-KL forward FFI callable.

    Kept for the vocab-parallel two-stage path; the default custom-VJP now
    uses the split ``fwd_only`` + ``bwd_dstudent`` pair above.
    """
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_size, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fused_kl_prim_func(
            num_rows=num_rows,
            vocab_size=vocab_size,
            block_v=bv,
            dtype=dtype,
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


def _get_partial_stats(num_rows: int, vocab_local: int, dtype):
    """Compile/cache the per-shard KL stats kernel."""
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_local, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _STATS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_partial_stats_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_local,
            block_v=bv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=tuple(jax.ShapeDtypeStruct((num_rows,), jnp.float32) for _ in range(4)),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _STATS_CACHE[key] = ffi
        return ffi


def _get_local_loss(num_rows: int, vocab_local: int, dtype):
    """Compile/cache the per-shard KL local-loss kernel."""
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_local, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _LOCAL_LOSS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_local_loss_prim_func(
            num_rows=num_rows,
            vocab_local=vocab_local,
            block_v=bv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_rows,), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _LOCAL_LOSS_CACHE[key] = ffi
        return ffi


def _get_dstudent(num_rows: int, vocab_local: int, dtype):
    """Compile/cache the per-shard KL dstudent writer."""
    bv = _DEFAULT_BLOCK_V
    key = (num_rows, vocab_local, bv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _DSTUDENT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_kl_dstudent_prim_func(
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
        _DSTUDENT_CACHE[key] = ffi
        return ffi


def _flatten_logits(logits: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten to ``(N, V)`` and remember the original leading shape."""
    if logits.ndim < 2:
        raise ValueError(f"fused_kl_divergence expects rank>=2 logits; got shape {logits.shape}")
    return logits.reshape(-1, logits.shape[-1]), logits.shape[:-1]


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _fused_kl_core_forward(student_2d, teacher_2d, weights_1d, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_fwd_only(
        n,
        v,
        student_2d.dtype,
        "forward",
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, _lse_t, _lse_s, _acc = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss


def _kl_forward_fwd(student_2d, teacher_2d, weights_1d, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_fwd_only(
        n,
        v,
        student_2d.dtype,
        "forward",
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, lse_t, lse_s, _acc = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss, (student_2d, teacher_2d, lse_t, lse_s, weights_1d)


def _kl_forward_bwd(temperature, block_v, block_m, residual, dy):
    student_2d, teacher_2d, lse_t, lse_s, weights_1d = residual
    n, v = student_2d.shape
    bwd_ffi = _get_bwd_dstudent(n, v, student_2d.dtype, float(temperature), block_v=int(block_v))
    dstudent = bwd_ffi(
        student_2d,
        teacher_2d,
        lse_t.astype(jnp.float32),
        lse_s.astype(jnp.float32),
        weights_1d.astype(jnp.float32),
        dy.astype(jnp.float32),
    )
    return (dstudent, jnp.zeros_like(teacher_2d), None)


_fused_kl_core_forward.defvjp(_kl_forward_fwd, _kl_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _fused_kl_core_reverse(student_2d, teacher_2d, weights_1d, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_fwd_only(
        n,
        v,
        student_2d.dtype,
        "reverse",
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, _lse_t, _lse_s, _acc = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss


def _kl_reverse_fwd(student_2d, teacher_2d, weights_1d, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_fwd_only(
        n,
        v,
        student_2d.dtype,
        "reverse",
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, _lse_t, lse_s, acc = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss, (student_2d, teacher_2d, lse_s, acc, weights_1d)


def _kl_reverse_bwd(temperature, block_v, block_m, residual, dy):
    student_2d, teacher_2d, lse_s, acc, weights_1d = residual
    n, v = student_2d.shape
    bwd_ffi = _get_bwd_reverse(n, v, student_2d.dtype, float(temperature), block_v=int(block_v))
    dstudent = bwd_ffi(
        student_2d,
        teacher_2d,
        lse_s.astype(jnp.float32),
        acc.astype(jnp.float32),
        weights_1d.astype(jnp.float32),
        dy.astype(jnp.float32),
    )
    return (dstudent, jnp.zeros_like(teacher_2d), None)


_fused_kl_core_reverse.defvjp(_kl_reverse_fwd, _kl_reverse_bwd)


def _jsd_recompute_loss(student_2d, teacher_2d, lse_t, lse_s, weights_1d, beta, temperature):
    """Pure-JAX recomputation of the per-row JSD value from cached lse residuals.

    Used as the **forward** for the JSD custom_vjp so JAX can autodiff
    it. We pass the lse residuals (computed cheaply in the kernel) so
    this recompute path doesn't need to redo the online softmax —
    just one pass over (student, teacher) to materialise log_m + the
    per-token KL contributions.
    """
    inv_T = 1.0 / float(temperature)
    s_scaled = student_2d.astype(jnp.float32) * inv_T
    t_scaled = teacher_2d.astype(jnp.float32) * inv_T
    log_p_s = s_scaled - lse_s[:, None]
    log_p_t = t_scaled - lse_t[:, None]
    log_beta = jnp.log(jnp.asarray(beta, dtype=jnp.float32))
    log_one_minus_beta = jnp.log1p(-jnp.asarray(beta, dtype=jnp.float32))
    log_m = jax.scipy.special.logsumexp(
        jnp.stack([log_p_t + log_one_minus_beta, log_p_s + log_beta]),
        axis=0,
    )
    p_s = jnp.exp(log_p_s)
    p_t = jnp.exp(log_p_t)
    per_token = beta * p_t * (log_p_t - log_m) + (1.0 - beta) * p_s * (log_p_s - log_m)
    per_row = jnp.sum(per_token, axis=-1) * weights_1d.astype(jnp.float32)
    return per_row


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _fused_kl_core_jsd(student_2d, teacher_2d, weights_1d, beta, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_jsd_fwd(
        n,
        v,
        student_2d.dtype,
        float(beta),
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, _lse_t, _lse_s = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss


def _kl_jsd_fwd(student_2d, teacher_2d, weights_1d, beta, temperature, block_v, block_m):
    n, v = student_2d.shape
    fwd_ffi = _get_kl_jsd_fwd(
        n,
        v,
        student_2d.dtype,
        float(beta),
        float(temperature),
        block_v=int(block_v),
        block_m=int(block_m),
    )
    loss, lse_t, lse_s = fwd_ffi(student_2d, teacher_2d, weights_1d)
    return loss, (student_2d, teacher_2d, lse_t, lse_s, weights_1d)


def _kl_jsd_bwd(beta, temperature, block_v, block_m, residual, dy):
    del block_v, block_m

    """JSD bwd via JAX autodiff over the recompute path.

    The analytic JSD gradient threads through the log-mixture (which
    in turn depends on the student via ``log p_s``), so a hand-written
    TileLang bwd would need two extra reductions per row. We recompute
    the per-row JSD in pure JAX from the cached ``lse_t``/``lse_s``
    residuals (single pass over the logits) and let JAX vjp produce
    the gradient. Slower than the forward/reverse bwds but still
    HBM-bound — ~2x the bandwidth of the forward bwd.
    """
    student_2d, teacher_2d, lse_t, lse_s, weights_1d = residual

    def recompute(s, t):
        return _jsd_recompute_loss(s, t, lse_t, lse_s, weights_1d, float(beta), float(temperature))

    _per_row, vjp_fn = jax.vjp(recompute, student_2d, teacher_2d)
    dstudent, _dteacher = vjp_fn(dy.astype(jnp.float32))
    return (dstudent.astype(student_2d.dtype), jnp.zeros_like(teacher_2d), None)


_fused_kl_core_jsd.defvjp(_kl_jsd_fwd, _kl_jsd_bwd)


_kl_fwd_compute = None  # type: ignore[assignment]  # no longer used
_fused_kl_core = _fused_kl_core_forward
_kl_fwd = _kl_forward_fwd
_kl_bwd = _kl_forward_bwd


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_kl_core_tp(student_2d, teacher_2d, weights_1d, vocab_axis):
    """Vocab-parallel forward KL using stats + local-loss kernels + collectives."""
    n, v_local = student_2d.shape
    stats_ffi = _get_partial_stats(n, v_local, student_2d.dtype)
    local_max_t, local_se_t, local_max_s, local_se_s = stats_ffi(student_2d, teacher_2d)

    global_max_t = jax.lax.pmax(local_max_t, vocab_axis)
    global_se_t = jax.lax.psum(local_se_t * jnp.exp(local_max_t - global_max_t), vocab_axis)
    lse_t = jnp.log(global_se_t) + global_max_t

    global_max_s = jax.lax.pmax(local_max_s, vocab_axis)
    global_se_s = jax.lax.psum(local_se_s * jnp.exp(local_max_s - global_max_s), vocab_axis)
    lse_s = jnp.log(global_se_s) + global_max_s

    local_loss_ffi = _get_local_loss(n, v_local, student_2d.dtype)
    local_part = local_loss_ffi(student_2d, teacher_2d, lse_t, lse_s)
    per_row = jax.lax.psum(local_part, vocab_axis) * weights_1d
    return per_row.astype(jnp.float32)


def _kl_tp_fwd(student_2d, teacher_2d, weights_1d, vocab_axis):
    """VJP primal: stash lse_t / lse_s for the backward dstudent kernel."""
    n, v_local = student_2d.shape
    stats_ffi = _get_partial_stats(n, v_local, student_2d.dtype)
    local_max_t, local_se_t, local_max_s, local_se_s = stats_ffi(student_2d, teacher_2d)

    global_max_t = jax.lax.pmax(local_max_t, vocab_axis)
    global_se_t = jax.lax.psum(local_se_t * jnp.exp(local_max_t - global_max_t), vocab_axis)
    lse_t = jnp.log(global_se_t) + global_max_t

    global_max_s = jax.lax.pmax(local_max_s, vocab_axis)
    global_se_s = jax.lax.psum(local_se_s * jnp.exp(local_max_s - global_max_s), vocab_axis)
    lse_s = jnp.log(global_se_s) + global_max_s

    local_loss_ffi = _get_local_loss(n, v_local, student_2d.dtype)
    local_part = local_loss_ffi(student_2d, teacher_2d, lse_t, lse_s)
    per_row = jax.lax.psum(local_part, vocab_axis) * weights_1d
    residual = (student_2d, teacher_2d, lse_t, lse_s, weights_1d)
    return per_row.astype(jnp.float32), residual


def _kl_tp_bwd(vocab_axis, residual, dy):
    """Backward; uses cached global lse_t/lse_s and a single local kernel."""
    del vocab_axis
    student_2d, teacher_2d, lse_t, lse_s, weights_1d = residual
    n, v_local = student_2d.shape
    scaled_weights = (weights_1d * dy).astype(jnp.float32)
    dstudent_ffi = _get_dstudent(n, v_local, student_2d.dtype)
    dstudent = dstudent_ffi(
        student_2d,
        teacher_2d,
        lse_t.astype(jnp.float32),
        lse_s.astype(jnp.float32),
        scaled_weights,
    )
    dteacher = jnp.zeros_like(teacher_2d)
    return (dstudent, dteacher, None)


_fused_kl_core_tp.defvjp(_kl_tp_fwd, _kl_tp_bwd)


def fused_kl_divergence_tilelang(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    weights: jax.Array | None = None,
    *,
    reduction: str = "mean",
    direction: str = "forward",
    temperature: float = 1.0,
    beta: float = 0.5,
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
) -> jax.Array:
    """Tile-lang fused KL between two logit tensors.

    Supports three directions (selected via ``direction``):

      * ``"forward"`` (default): ``KL(softmax(t/T) ‖ softmax(s/T))`` —
        teacher-to-student forward KL. EasyDeL's ``distillation_loss``
        gradient-equivalent.
      * ``"reverse"``: ``KL(softmax(s/T) ‖ softmax(t/T))`` — student
        first. Matches GKD's ``β=0`` limit.
      * ``"jsd"``: generalised Jensen-Shannon
        ``β·KL(p_t‖m) + (1-β)·KL(p_s‖m)``,
        ``m = β·p_t + (1-β)·p_s`` — matches GKD's intermediate ``β``.

    The kernel computes the KL value in **T-scaled logit space** (the
    softmaxes are taken over ``logits / T``). The wrapper multiplies
    the final loss by ``T²`` to match the EasyDeL / Hinton
    distillation convention. The bwd kernel folds the ``1/T`` chain
    rule into the gradient automatically, so ``jax.grad`` returns the
    correct gradient for the ``T²``-scaled loss.

    Args:
        student_logits: ``(..., V)`` student logits.
        teacher_logits: ``(..., V)`` teacher logits (detached — gradient
            flowing back is zero).
        weights: Optional per-token weights with shape ``logits.shape[:-1]``.
            Use ``weights=completion_mask`` for assistant-only loss in
            instruction-tuning / chat distillation.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        direction: ``"forward"`` / ``"reverse"`` / ``"jsd"``.
        temperature: Softmax temperature ``T`` (default 1.0).
        beta: JSD interpolation factor for ``direction="jsd"``; ignored
            otherwise.
        vocab_parallel_axis: Mesh axis name for TP (forward KL only).
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang fused_kl_divergence requires both `tilelang` and `jax_tvm_ffi`.")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if direction not in ("forward", "reverse", "jsd"):
        raise ValueError(f"Invalid direction '{direction}'; expected one of forward/reverse/jsd.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    if direction == "jsd" and not 0.0 < beta < 1.0:
        raise ValueError(f"JSD requires beta in (0, 1); got {beta}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"fused_kl_divergence: shape mismatch student={student_logits.shape} vs teacher={teacher_logits.shape}"
        )
    if student_logits.dtype != teacher_logits.dtype:
        teacher_logits = teacher_logits.astype(student_logits.dtype)

    flat_student, leading = _flatten_logits(student_logits)
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    if weights is None:
        flat_weights = jnp.ones(flat_student.shape[0], dtype=jnp.float32)
    else:
        if weights.shape != leading:
            raise ValueError(f"weights.shape={weights.shape} must equal logits.shape[:-1]={leading}")
        flat_weights = weights.reshape(-1).astype(jnp.float32)

    if vocab_parallel_axis is not None:
        if direction != "forward" or temperature != 1.0:
            raise NotImplementedError(
                "vocab-parallel mode currently only supports direction='forward' with "
                "temperature=1.0; the TP kernels need the same temperature/direction "
                "plumbing as the single-shard ones."
            )
        per_row = _fused_kl_core_tp(flat_student, flat_teacher, flat_weights, vocab_parallel_axis)
    elif direction == "forward":
        per_row = _fused_kl_core_forward(
            flat_student,
            flat_teacher,
            flat_weights,
            float(temperature),
            int(block_v),
            int(block_m),
        )
    elif direction == "reverse":
        per_row = _fused_kl_core_reverse(
            flat_student,
            flat_teacher,
            flat_weights,
            float(temperature),
            int(block_v),
            int(block_m),
        )
    else:
        per_row = _fused_kl_core_jsd(
            flat_student,
            flat_teacher,
            flat_weights,
            float(beta),
            float(temperature),
            int(block_v),
            int(block_m),
        )

    if temperature != 1.0:
        per_row = per_row * (float(temperature) ** 2)

    if reduction == "none":
        return per_row.reshape(leading)
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total
    denom = jnp.maximum(jnp.sum(flat_weights), 1e-8)
    return total / denom
