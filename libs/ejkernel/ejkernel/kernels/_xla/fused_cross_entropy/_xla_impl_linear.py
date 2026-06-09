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

"""Fused linear cross-entropy (FLCE) — the LM-head-chunked CE path.

Projects hidden states through the LM head in token-dimension chunks and computes
cross-entropy per chunk, so the full ``[..., V]`` logit tensor is **never**
materialised — only ``[..., chunk, V]`` lives at a time. Each chunk body is
wrapped with :func:`jax.checkpoint`, so the backward recomputes its logits
instead of storing them; gradients flow to the hidden states and (through the
matmul / the ``lm_head_fn`` closure) to the LM-head weights.

This is the ejkernel home for EasyDeL's ``causal_lm_loss_chunked_lm_head`` inner
loop. The training-loop orchestration (token shifting, normalizing factor, batch
plumbing) stays in the caller; this kernel owns the chunked projection + CE math
and its memory-bounded gradient.

Two projection modes (mutually exclusive):

* ``lm_head_weight`` (+ optional ``lm_head_bias``): the projection is
  ``hidden @ W (+ b)``, differentiable w.r.t. both ``hidden`` and the weights.
* ``lm_head_fn``: an arbitrary callable ``[..., H] -> [..., V]`` (e.g. a model
  head carrying bias / tied embeddings / soft-capping). Must be trace-safe to
  call inside ``fori_loop`` / ``jax.checkpoint`` (no ``nn.remat`` at call time).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

from ._xla_impl_fwd import _fused_ce_core, _fused_ce_core_tp, _label_smoothing_correction


def _default_token_chunk_size(seq_len: int, vocab_size: int | None, dtype_bytes: int) -> int:
    """Pick a token chunk that keeps the transient ``[chunk, V]`` logits ~<=1 GiB."""
    if vocab_size is None or vocab_size <= 0:
        chunk = 1024
    else:
        target_bytes = 1 * 1024 * 1024 * 1024
        raw = max(1, target_bytes // max(1, vocab_size * max(1, dtype_bytes)))
        chunk = 1 << max(0, int(raw).bit_length() - 1)
    return max(1, min(chunk, seq_len))


def fused_linear_cross_entropy(
    hidden: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    *,
    lm_head_weight: jnp.ndarray | None = None,
    lm_head_bias: jnp.ndarray | None = None,
    lm_head_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    logit_softcap: float | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "mean",
    token_chunk_size: int = 0,
    compute_dtype: jnp.dtype | None = None,
    checkpoint: bool = True,
    vocab_parallel_axis: str | None = None,
    sparse_skip: bool = False,
    sparse_reduce_axes: tuple[str, ...] = (),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Token-chunked fused linear cross-entropy.

    Args:
        hidden: Hidden states ``[..., T, H]`` — chunked along the token axis
            (``axis=-2``). Any number of earlier leading (e.g. batch) dims.
        targets: Token ids ``[..., T]`` (``hidden.shape[:-1]``).
        weights: Optional per-token weights ``[..., T]``.
        lm_head_weight: ``[H, V]`` projection weight (raw-matmul mode).
        lm_head_bias: Optional ``[V]`` bias for the raw-matmul mode.
        lm_head_fn: Callable projecting ``[..., H] -> [..., V]`` (custom-head
            mode). Mutually exclusive with ``lm_head_weight``.
        logit_softcap: If set, applies ``cap * tanh(logits / cap)`` to each
            chunk's logits before the loss (Gemma-2 style).
        ignore_index: Sparse-mode ignore sentinel.
        label_smoothing: ``alpha in [0, 1)`` (dense one-hot parity).
        z_loss: Coefficient for ``z_loss * logsumexp(logits)**2``.
        reduction: ``"sum"`` or ``"mean"`` (``"none"`` unsupported under
            token chunking).
        token_chunk_size: Tokens per chunk; ``0`` picks a ~1 GiB-budget default.
        compute_dtype: CE math dtype; defaults to ``hidden.dtype`` (no forced
            fp32). The projection runs in its native dtype.
        checkpoint: When ``True`` (default) each chunk body is wrapped in
            :func:`jax.checkpoint` so the backward recomputes that chunk's
            logits instead of storing them — the memory-bounded behaviour that
            makes FLCE worthwhile. Set ``False`` to keep the per-chunk logits
            live for the backward (faster, no recompute) when the ``[chunk, V]``
            residuals comfortably fit in memory.

    Returns:
        ``(total_loss, total_z_loss, weight_sum, accuracy)``. ``total_loss`` /
        ``total_z_loss`` follow ``reduction`` (the global normalizing factor is
        the caller's responsibility); ``accuracy`` is weight-weighted.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"fused_linear_cross_entropy supports reduction sum/mean, got {reduction!r}")
    if (lm_head_weight is None) == (lm_head_fn is None):
        raise ValueError("provide exactly one of `lm_head_weight` or `lm_head_fn`.")
    if vocab_parallel_axis is not None and (label_smoothing > 0.0 or z_loss > 0.0):
        raise NotImplementedError("label_smoothing / z_loss are not yet wired through the vocab-parallel FLCE path.")
    if hidden.ndim < 2:
        raise ValueError(f"hidden must be at least rank-2 ([..., T, H]); got shape {hidden.shape}.")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError(f"targets.shape={targets.shape} must equal hidden.shape[:-1]={hidden.shape[:-1]}.")

    compute_dtype = jnp.dtype(compute_dtype) if compute_dtype is not None else hidden.dtype
    seq_len = int(hidden.shape[-2])

    vocab_size = int(lm_head_weight.shape[-1]) if lm_head_weight is not None else None
    if not token_chunk_size:
        token_chunk_size = _default_token_chunk_size(seq_len, vocab_size, jnp.dtype(compute_dtype).itemsize)
    token_chunk_size = max(1, min(int(token_chunk_size), seq_len))

    def _project(chunk_hidden):
        if lm_head_fn is not None:
            logits = lm_head_fn(chunk_hidden)
        else:
            logits = jnp.matmul(chunk_hidden, lm_head_weight)
            if lm_head_bias is not None:
                logits = logits + lm_head_bias
        if logit_softcap is not None:
            cap = jnp.asarray(logit_softcap, dtype=logits.dtype)
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def _chunk_loss(chunk_hidden, chunk_targets, chunk_weights):
        logits = _project(chunk_hidden).astype(compute_dtype)
        valid = chunk_targets != ignore_index
        safe = jnp.where(valid, chunk_targets, 0).astype(jnp.int32)
        w = valid.astype(compute_dtype) if chunk_weights is None else valid.astype(compute_dtype) * chunk_weights
        if vocab_parallel_axis is not None:
            # Vocab-parallel FLCE: each ``_project`` produced the per-shard ``[chunk, V_local]`` slice
            # (column-parallel LM head); the TP core merges the softmax normalizer + gathers the (possibly
            # cross-shard) target logit via psum, so the per-row loss is globally correct without ever
            # forming the full ``[chunk, V]``. Backward is the local ``(softmax - onehot)`` slab.
            per_row = _fused_ce_core_tp(logits, safe, w, ignore_index, vocab_parallel_axis)
            z_row = jnp.zeros_like(per_row)
            # Cross-shard argmax for accuracy (a pure metric -- fully detached so the ``pmax`` never enters
            # autodiff): the shard owning the global max contributes its global token id; ``psum`` reduces
            # to that id on every shard.
            det_logits = jax.lax.stop_gradient(logits)
            tp_idx = jax.lax.axis_index(vocab_parallel_axis)
            local_best_val = jnp.max(det_logits, axis=-1)
            local_best_id = tp_idx * det_logits.shape[-1] + jnp.argmax(det_logits, axis=-1).astype(jnp.int32)
            global_best_val = jax.lax.pmax(local_best_val, vocab_parallel_axis)
            is_winner = local_best_val >= global_best_val
            # Deterministic tie-break: when >=2 shards hold the global max, ``psum`` of ids would *sum* them
            # into a bogus id. Take the smallest winning id via ``pmin`` (losers carry a large sentinel).
            cand_id = jnp.where(is_winner, local_best_id, jnp.int32(2**30))
            global_best_id = jax.lax.pmin(cand_id, vocab_parallel_axis)
            correct = jnp.sum((global_best_id == chunk_targets).astype(compute_dtype) * jax.lax.stop_gradient(w))
        else:
            # Base CE via the analytic custom-VJP core (``softmax - onehot``). Under the
            # outer ``jax.checkpoint`` this hand-written backward schedules markedly
            # better than autodiff-through-logsumexp + take_along_axis (~5% faster
            # fwd+bwd on TPU), and matches the dense path's gradient exactly. ``per_row``
            # is ``(lse - target_logit) * w``.
            per_row = _fused_ce_core(logits, safe, w, ignore_index)
            z_row = jnp.zeros_like(per_row)
            if label_smoothing > 0.0 or z_loss > 0.0:
                lse = jax.scipy.special.logsumexp(logits, axis=-1)
                if label_smoothing > 0.0:
                    per_row = per_row + _label_smoothing_correction(logits, lse, safe, w, label_smoothing) * w
                if z_loss > 0.0:
                    z_row = z_loss * lse * lse * w
                    per_row = per_row + z_row
            correct = jnp.sum((jnp.argmax(logits, axis=-1) == chunk_targets).astype(compute_dtype) * w)
        # _fused_ce_core returns fp32 per-row; cast the sums back to compute_dtype
        # so the fori_loop carry types match and the output contract is preserved.
        loss_sum = jnp.sum(per_row).astype(compute_dtype)
        z_sum = jnp.sum(z_row).astype(compute_dtype)
        w_sum = jnp.sum(w)
        return loss_sum, z_sum, w_sum, correct

    # Pad the token axis up to a whole number of chunks (ignored labels => zero weight).
    pad_len = (-seq_len) % token_chunk_size
    if pad_len:
        h_pad = [(0, 0)] * hidden.ndim
        h_pad[-2] = (0, pad_len)
        hidden = jnp.pad(hidden, h_pad)
        t_pad = [(0, 0)] * targets.ndim
        t_pad[-1] = (0, pad_len)
        targets = jnp.pad(targets, t_pad, constant_values=ignore_index)
        if weights is not None:
            weights = jnp.pad(weights, t_pad)
    num_chunks = (seq_len + pad_len) // token_chunk_size

    # Sparse skip: only the leading chunks that contain a real (unmasked) token do work; the trailing
    # fully-masked chunks (the common case -- a short prompt right-padded to a long ``max_length``) are
    # skipped. ``fori_loop`` must keep a STATIC trip count for reverse-mode AD, so the loop still spans all
    # ``num_chunks`` and each chunk is gated by ``i < sparse_upper`` -- but ``sparse_upper`` is computed ONCE
    # (not a per-chunk predicate) and is UNIFORM across mesh shards, so the gate never diverges: every shard
    # runs the same branch each iteration and the inner vocab ``psum`` stays in lock-step (a per-shard /
    # per-chunk predicate would deadlock shard_map). Uniformity: under SPMD the reduction already
    # all-reduces to a replicated scalar; under shard_map we ``pmax`` over the token (batch/seq) mesh axes.
    # Trailing-only -- interior / left padding simply isn't skipped (still correct, just not faster).
    sparse_upper = None
    if sparse_skip:
        active = targets != ignore_index
        if weights is not None:
            active = active & (weights > 0)
        active_chunks = jnp.any(
            active.reshape(*active.shape[:-1], num_chunks, token_chunk_size),
            axis=tuple(a for a in range(active.ndim + 1) if a != active.ndim - 1),
        )  # [num_chunks]
        idx = jnp.arange(num_chunks, dtype=jnp.int32)
        sparse_upper = jnp.max(jnp.where(active_chunks, idx + 1, 0)).astype(jnp.int32)
        if sparse_reduce_axes:
            sparse_upper = jax.lax.pmax(sparse_upper, sparse_reduce_axes)

    def _chunk_step(ch, ct, cw, i):
        if sparse_skip:

            def _skip():
                zr = jnp.zeros((), compute_dtype) * jnp.sum(ch.astype(compute_dtype))
                return zr, zr, zr, zr

            return lax.cond(i < sparse_upper, lambda: _chunk_loss(ch, ct, cw), _skip)
        return _chunk_loss(ch, ct, cw)

    # Checkpoint the whole per-chunk step (the ``cond`` included, not just the inner projection+CE): the
    # backward then recomputes a skipped chunk as the trivial ``_skip`` branch instead of re-running the
    # LM-head projection, so the sparse skip carries through to the gradient pass too.
    if checkpoint:
        _chunk_step = jax.checkpoint(_chunk_step, prevent_cse=False)

    def _accumulate(i, carry):
        start = i * token_chunk_size
        ch = lax.dynamic_slice_in_dim(hidden, start, token_chunk_size, axis=-2)
        ct = lax.dynamic_slice_in_dim(targets, start, token_chunk_size, axis=-1)
        cw = None if weights is None else lax.dynamic_slice_in_dim(weights, start, token_chunk_size, axis=-1)
        loss_sum, z_sum, w_sum, correct = _chunk_step(ch, ct, cw, i)
        return (carry[0] + loss_sum, carry[1] + z_sum, carry[2] + w_sum, carry[3] + correct)

    zero = jnp.array(0.0, dtype=compute_dtype)
    if vocab_parallel_axis is not None:
        # Seed the accumulators with the varying-manual-axis (VMA) type of the (possibly batch/seq-sharded)
        # inputs so the fori_loop carry-in matches the carry-out under shard_map(check_vma=True) -- the
        # per-chunk loss varies over the batch/seq mesh axes, the bare scalar zero does not. The
        # ``0 * sum(hidden)`` term contributes neither value nor gradient; it only carries the VMA tag.
        zero = zero + jnp.zeros((), compute_dtype) * jnp.sum(hidden.astype(compute_dtype))
    total_loss, total_z_loss, weight_sum, correct_sum = lax.fori_loop(
        0, num_chunks, _accumulate, (zero, zero, zero, zero)
    )

    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(weight_sum, jnp.asarray(1e-8, dtype=compute_dtype))
    accuracy = correct_sum / jnp.maximum(weight_sum, jnp.asarray(1e-8, dtype=compute_dtype))
    return total_loss, total_z_loss, weight_sum, accuracy


__all__ = ("fused_linear_cross_entropy",)
