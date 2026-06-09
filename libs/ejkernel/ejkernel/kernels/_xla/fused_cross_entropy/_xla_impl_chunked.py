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

"""Memory-efficient chunked cross-entropy over already-materialised logits.

These are pure-JAX, XLA-friendly variants of sparse cross-entropy that trade a
small amount of recompute for a bounded peak working set. They never build the
``[..., V]`` softmax / one-hot tensor as a single live buffer:

* :func:`chunked_vocab_cross_entropy` — streams the log-sum-exp over the vocab
  axis in ``vocab_chunk_size`` slices (two-pass max / sum-of-exp). Equivalent to
  a plain dense CE but caps the intermediate ``exp`` to ``[..., vocab_chunk]``.
* :func:`blockwise_cross_entropy` — single-pass online log-sum-exp over vocab
  blocks with a per-block :func:`jax.checkpoint`, so the backward recomputes
  each ``[N, block]`` slab instead of storing it. Lowest peak memory for very
  large ``V`` on the logits path.
* :func:`chunked_token_cross_entropy` — streams over the token/row axis in
  ``token_chunk_size`` slices, capping the live softmax to ``[token_chunk, V]``.

All three share the label-smoothing / z-loss / weighted-accuracy semantics of
the dense :mod:`_xla_impl_fwd` path and return
``(total_loss, total_z_loss, weight_sum, accuracy)``. They are the ejkernel home
for the chunked CE variants that previously lived in EasyDeL's ``loss_utils``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _logsumexp_chunked(x: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """``logsumexp`` over the last axis computed in ``chunk_size`` vocab slices.

    Two-pass (running max, then running sum of shifted exponentials) so the
    intermediate ``exp`` buffer is ``[..., chunk_size]`` rather than ``[..., V]``.
    Mathematically identical to :func:`jax.scipy.special.logsumexp`.
    """
    V: int = x.shape[-1]
    n_full = V // chunk_size
    tail = V - n_full * chunk_size

    def max_body(i, m):
        start = i * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, chunk_size, axis=-1)
        return jnp.maximum(m, jnp.max(chunk, axis=-1))

    m = jnp.full(x.shape[:-1], -jnp.inf, dtype=x.dtype)
    m = lax.fori_loop(0, n_full, max_body, m)
    if tail:
        start = n_full * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, tail, axis=-1)
        m = jnp.maximum(m, jnp.max(chunk, axis=-1))

    def sum_body(i, s):
        start = i * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, chunk_size, axis=-1)
        return s + jnp.sum(jnp.exp(chunk - m[..., None]), axis=-1)

    s = jnp.zeros_like(m)
    s = lax.fori_loop(0, n_full, sum_body, s)
    if tail:
        start = n_full * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, tail, axis=-1)
        s = s + jnp.sum(jnp.exp(chunk - m[..., None]), axis=-1)

    return jnp.log(s) + m


def _label_smoothing_params(
    vocab_size: int,
    label_smoothing: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(confidence, low_confidence, normalizing_constant)`` for smoothing.

    Decomposes the smoothed target ``q(k) = (1-eps) delta(k,y) + eps/V`` into the
    three scalars needed to evaluate cross-entropy against it without building the
    ``[..., V]`` soft-target tensor. ``normalizing_constant`` is ``H(q)`` so a
    perfect model reaches zero loss (dense one-hot parity).
    """
    confidence = jnp.asarray(1.0 - label_smoothing, dtype=dtype)
    low_confidence = (jnp.asarray(1.0, dtype=dtype) - confidence) / jnp.asarray(vocab_size - 1, dtype=dtype)
    normalizing_constant = -(
        confidence * jnp.log(confidence)
        + jnp.asarray(vocab_size - 1, dtype=dtype) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    return confidence, low_confidence, normalizing_constant


def _apply_sparse_label_smoothing(
    log_z: jax.Array,
    target_logit: jax.Array,
    sum_logits: jax.Array,
    *,
    vocab_size: int,
    label_smoothing: float,
    dtype: jnp.dtype,
) -> jax.Array:
    """Label-smoothed sparse cross-entropy from streamed sufficient statistics.

    Equivalent to ``(1-eps) NLL + eps (log_z - mean_logits)`` but uses the
    dense-parity decomposition from :func:`_label_smoothing_params`:
    ``log_z - [(conf - low) * target_logit + low * sum_logits] - norm_const``.
    """
    confidence, low_confidence, normalizing_constant = _label_smoothing_params(
        vocab_size=vocab_size,
        label_smoothing=label_smoothing,
        dtype=dtype,
    )
    target_mass_logits = (confidence - low_confidence) * target_logit + low_confidence * sum_logits
    return log_z - target_mass_logits - normalizing_constant


def chunked_vocab_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "mean",
    chunk_size: int = 8192,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sparse cross-entropy with vocabulary-dimension chunking.

    Computes the log-sum-exp with :func:`_logsumexp_chunked` (peak ``exp`` buffer
    ``[..., chunk_size]``) then gathers the target logit. Returns
    ``(total_loss, total_z_loss, weight_sum, accuracy)``; ``reduction`` is one of
    ``"none"`` / ``"sum"`` / ``"mean"`` and ``total_z_loss`` is the weighted
    z-loss sum (per-token for ``"none"``).
    """
    logits = logits.astype(compute_dtype)
    valid = targets != ignore_index
    safe_targets = jnp.where(valid, targets, 0)

    lse = _logsumexp_chunked(logits, chunk_size)
    logit_y = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    nll = lse - logit_y

    if label_smoothing > 0.0:
        nll = _apply_sparse_label_smoothing(
            lse,
            logit_y,
            jnp.sum(logits, axis=-1),
            vocab_size=logits.shape[-1],
            label_smoothing=label_smoothing,
            dtype=compute_dtype,
        )

    z_term = (z_loss * jnp.square(lse)) if z_loss > 0.0 else jnp.zeros_like(lse)
    nll = nll + z_term

    w = valid.astype(compute_dtype) if weights is None else valid.astype(compute_dtype) * weights.astype(compute_dtype)
    weight_sum = jnp.sum(w)
    correct = (jnp.argmax(logits, axis=-1) == targets).astype(compute_dtype) * w
    accuracy = jnp.sum(correct) / jnp.maximum(weight_sum, 1e-8)

    if reduction == "none":
        return (nll * w).astype(compute_dtype), (z_term * w).astype(compute_dtype), weight_sum, accuracy

    total_loss = jnp.sum(nll * w)
    total_z_loss = jnp.sum(z_term * w)
    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(weight_sum, 1e-8)
    return total_loss, total_z_loss, weight_sum, accuracy


def blockwise_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "sum",
    block_size: int = 8192,
    compute_dtype: jnp.dtype = jnp.float32,
    checkpoint: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-pass online-softmax blockwise sparse cross-entropy.

    Streams the vocab axis in ``block_size`` blocks accumulating a running
    log-sum-exp, target logit, logit sum (for smoothing) and streamed argmax (for
    accuracy). When ``checkpoint`` is ``True`` (default) each block body is wrapped
    with :func:`jax.checkpoint` so the backward recomputes the ``[N, block]`` slab
    rather than storing it — lowest peak memory of the logits-path variants; set
    ``checkpoint=False`` to keep the block residuals live (faster, no recompute).
    Returns ``(total_loss, total_z_loss, weight_sum, accuracy)``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if logits.ndim == 3:
        B, T, V = logits.shape
        L = B * T
        logits2d = logits.reshape(L, V)
        y = targets.reshape(L)
        w = None if weights is None else weights.reshape(L).astype(compute_dtype)
    elif logits.ndim == 2:
        L, V = logits.shape
        logits2d = logits
        y = targets
        w = None if weights is None else weights.astype(compute_dtype)
    else:
        raise ValueError(f"logits must be [B, T, V] or [N, V], got {logits.shape}")

    logits2d = logits2d.astype(compute_dtype)
    valid = y != ignore_index
    y_safe = jnp.where(valid, y, 0)
    w = valid.astype(compute_dtype) if w is None else valid.astype(compute_dtype) * w

    neg_inf = jnp.array(-jnp.inf, dtype=compute_dtype)
    m = jnp.full((L,), neg_inf)
    log_z = jnp.full((L,), neg_inf)
    o = jnp.zeros((L,), dtype=compute_dtype)
    sum_logits = jnp.zeros((L,), dtype=compute_dtype)
    best_logit = jnp.full((L,), neg_inf)
    best_id = jnp.zeros((L,), dtype=jnp.int32)

    n_full = V // block_size
    tail = V - n_full * block_size

    def process_block(start, size, m, log_z, o, sum_logits, best_logit, best_id):
        chunk = lax.dynamic_slice_in_dim(logits2d, start, size, axis=1)
        chunk_max = jnp.max(chunk, axis=1)
        new_m = jnp.maximum(m, chunk_max)
        log_z = new_m + jnp.log(jnp.exp(log_z - new_m) + jnp.sum(jnp.exp(chunk - new_m[:, None]), axis=1))
        m = new_m

        in_block = (y_safe >= start) & (y_safe < start + size)
        idx = jnp.where(in_block, (y_safe - start).astype(jnp.int32), 0)
        logit_y_b = jnp.take_along_axis(chunk, idx[:, None], axis=1)[:, 0]
        o = o + jnp.where(in_block, logit_y_b, 0.0)

        sum_logits = sum_logits + jnp.sum(chunk, axis=1)

        block_best = jnp.argmax(chunk, axis=1)
        block_best_id = start + block_best.astype(jnp.int32)
        update = chunk_max > best_logit
        best_logit = jnp.where(update, chunk_max, best_logit)
        best_id = jnp.where(update, block_best_id, best_id)
        return m, log_z, o, sum_logits, best_logit, best_id

    if checkpoint:
        process_block = jax.checkpoint(process_block, prevent_cse=False, static_argnums=(1,))

    def full_body(i, carry):
        start = i * block_size
        return process_block(start, block_size, *carry)

    carry = (m, log_z, o, sum_logits, best_logit, best_id)
    if n_full > 0:
        carry = lax.fori_loop(0, n_full, full_body, carry)
    if tail:
        start = n_full * block_size
        carry = process_block(start, tail, *carry)
    m, log_z, o, sum_logits, best_logit, best_id = carry

    nll = log_z - o
    if label_smoothing and label_smoothing != 0.0:
        nll = _apply_sparse_label_smoothing(
            log_z,
            o,
            sum_logits,
            vocab_size=V,
            label_smoothing=label_smoothing,
            dtype=compute_dtype,
        )

    zterm = (z_loss * (log_z**2)) if (z_loss and z_loss != 0.0) else jnp.zeros_like(log_z)
    per_tok = nll + zterm
    weight_sum = jnp.sum(w)
    acc = jnp.sum((best_id == y_safe).astype(compute_dtype) * w) / jnp.maximum(
        weight_sum, jnp.asarray(1e-8, dtype=compute_dtype)
    )

    if reduction == "none":
        return (per_tok * w).astype(compute_dtype), (zterm * w).astype(compute_dtype), weight_sum, acc
    total_loss = jnp.sum(per_tok * w)
    total_z_loss = jnp.sum(zterm * w)
    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(weight_sum, jnp.asarray(1e-8, dtype=compute_dtype))
    return total_loss, total_z_loss, weight_sum, acc


def chunked_token_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "sum",
    token_chunk_size: int = 8192,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sparse cross-entropy with token/row-dimension chunking.

    Streams over the flattened token axis in ``token_chunk_size`` slices so the
    live softmax is ``[token_chunk, V]`` instead of ``[N, V]``; complementary to
    vocab chunking when sequence length is the memory bottleneck. Returns
    ``(total_loss, total_z_loss, weight_sum, accuracy)`` with ``reduction`` in
    ``{"sum", "mean"}``.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"chunked_token_cross_entropy supports reduction sum/mean, got {reduction!r}")
    logits = logits.astype(compute_dtype)
    V = logits.shape[-1]
    logits2d = logits.reshape(-1, V)
    targets1d = targets.reshape(-1)
    weights1d = None if weights is None else weights.reshape(-1).astype(compute_dtype)
    N: int = logits2d.shape[0]
    token_chunk_size = max(1, min(int(token_chunk_size), N))
    n_full = N // token_chunk_size
    tail = N - n_full * token_chunk_size

    def _chunk(chunk_logits, chunk_targets, chunk_weights):
        lse = jax.scipy.special.logsumexp(chunk_logits, axis=-1)
        valid = chunk_targets != ignore_index
        safe = jnp.where(valid, chunk_targets, 0)
        logit_y = jnp.take_along_axis(chunk_logits, safe[:, None], axis=-1)[:, 0]
        nll = lse - logit_y
        if label_smoothing > 0.0:
            nll = _apply_sparse_label_smoothing(
                lse,
                logit_y,
                jnp.sum(chunk_logits, axis=-1),
                vocab_size=V,
                label_smoothing=label_smoothing,
                dtype=compute_dtype,
            )
        zterm = (z_loss * jnp.square(lse)) if z_loss > 0.0 else jnp.zeros_like(lse)
        nll = nll + zterm
        w = valid.astype(compute_dtype) if chunk_weights is None else valid.astype(compute_dtype) * chunk_weights
        loss_sum = jnp.sum(nll * w)
        w_sum = jnp.sum(w)
        z_sum = jnp.sum(zterm * w)
        acc = jnp.sum((jnp.argmax(chunk_logits, axis=-1) == chunk_targets).astype(compute_dtype) * w)
        return loss_sum, w_sum, acc, z_sum

    def body(i, carry):
        tot, wsum, acc_sum, zsum = carry
        start = i * token_chunk_size
        cl = lax.dynamic_slice_in_dim(logits2d, start, token_chunk_size, axis=0)
        ct = lax.dynamic_slice_in_dim(targets1d, start, token_chunk_size, axis=0)
        cw = None if weights1d is None else lax.dynamic_slice_in_dim(weights1d, start, token_chunk_size, axis=0)
        loss_sum, w_sum, acc, z_sum = _chunk(cl, ct, cw)
        return (tot + loss_sum, wsum + w_sum, acc_sum + acc, zsum + z_sum)

    init = (
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
    )
    carry = lax.fori_loop(0, n_full, body, init)

    if tail:
        start = n_full * token_chunk_size
        cl = lax.dynamic_slice_in_dim(logits2d, start, tail, axis=0)
        ct = lax.dynamic_slice_in_dim(targets1d, start, tail, axis=0)
        cw = None if weights1d is None else lax.dynamic_slice_in_dim(weights1d, start, tail, axis=0)
        loss_sum, w_sum, acc, z_sum = _chunk(cl, ct, cw)
        carry = (carry[0] + loss_sum, carry[1] + w_sum, carry[2] + acc, carry[3] + z_sum)

    total_loss, total_wsum, acc_sum, total_z_loss = carry
    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(total_wsum, 1e-8)
    accuracy = acc_sum / jnp.maximum(total_wsum, 1e-8)
    return total_loss, total_z_loss, total_wsum, accuracy


__all__ = (
    "_apply_sparse_label_smoothing",
    "_label_smoothing_params",
    "_logsumexp_chunked",
    "blockwise_cross_entropy",
    "chunked_token_cross_entropy",
    "chunked_vocab_cross_entropy",
)
