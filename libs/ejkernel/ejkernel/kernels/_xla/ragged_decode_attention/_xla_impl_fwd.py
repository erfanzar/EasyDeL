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

"""Ragged Decode Attention forward pass using XLA/JAX.

This module provides the forward implementations for decode-phase attention with
variable-length (ragged) sequences.  Each sequence in the batch can have a
different KV range described by ``sequence_start`` / ``sequence_end`` integer
arrays.

Key Components:
    - ``create_attention_mask``: Build a boolean causal/validity mask per sequence.
    - ``apply_logits_soft_cap``: Apply tanh soft capping to attention logits.
    - ``apply_attention_sinks_block``: Add per-block sink biases to scores.
    - ``flash_attention_block``: One iteration of the online-softmax flash loop.
    - ``ragged_flash_attention_xla``: Multi-head flash attention for a ragged batch.
    - ``ragged_decode_mqa_xla``: MQA/GQA decode wrapper (public entry point).
    - ``inner_decode_xla``: JIT-compiled dispatcher; handles both single-token
      (``q_len=1``) and multi-query (``q_len>1``) inputs.

Memory Layout:
    - ``query``: ``[batch, q_len, num_q_heads, head_dim]`` (or 3-D for single step)
    - ``key``/``value``: ``[batch, kv_len, num_kv_heads, head_dim]``
    - ``sequence_start``/``sequence_end``: ``[batch]`` int32 boundary arrays.

Algorithm (``ragged_flash_attention_xla``):
    1. Build a boolean mask from ``sequence_start``/``sequence_end`` and an
       optional sliding window.
    2. Pad ``key``/``value``/``mask`` to a multiple of ``block_size``.
    3. Run a ``lax.scan`` over KV blocks, accumulating output with the
       online-softmax (running max + normaliser) algorithm.
    4. If ``softmax_aux`` is provided, seed the running max and normaliser with
       the sink logits before entering the KV scan.

Note:
    Ragged batching (distinct ``sequence_start``/``sequence_end`` per element)
    avoids padding waste when sequences have varying KV lengths.  All
    computation is XLA-compatible and runs on CPU/GPU/TPU.
"""

import chex
import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams


def create_attention_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    sliding_window: tuple[int, int] | None = None,
) -> Float[Array, "batch q_len 1 kv_len"]:
    """Create a boolean attention mask for ragged sequences with an optional sliding window.

    Position ``kv_pos`` is attended to by query at position ``q_pos`` iff:

    * ``kv_pos >= sequence_start[b]`` and ``kv_pos < sequence_end[b]``, AND
    * (if ``sliding_window`` is set) ``q_pos - window_left <= kv_pos <= q_pos + window_right``.

    For single-token decode (``q_len=1``), the query position is inferred as
    ``sequence_end[b] - 1``.  For multi-token queries each query token is placed
    at ``sequence_start[b] + q_token_idx``.

    Args:
        batch_size: Number of sequences in the batch.
        q_len: Number of query tokens.
        kv_len: Number of key/value tokens (static axis size of the KV cache).
        sequence_start: First valid KV position per sequence, shape ``[batch]``.
        sequence_end: One-past-last valid KV position per sequence, shape ``[batch]``.
        sliding_window: Optional ``(left, right)`` window sizes.  When provided,
            only keys within ``[q_pos - left, q_pos + right]`` are attended to.

    Returns:
        Boolean mask of shape ``[batch, q_len, 1, kv_len]``.
        ``True`` means the corresponding key/value position is *attended to*.
    """

    kv_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :]
    start = jnp.asarray(sequence_start, jnp.int32)[:, None, None]
    end = jnp.asarray(sequence_end, jnp.int32)[:, None, None]

    kv_valid = (kv_positions >= start) & (kv_positions < end)
    kv_valid = jnp.broadcast_to(kv_valid, (batch_size, q_len, kv_len))

    mask = kv_valid

    if sliding_window is not None:
        window_left, window_right = sliding_window
        if q_len == 1:
            q_pos = (end - 1).astype(jnp.int32)
        else:
            q_pos = (jnp.arange(q_len, dtype=jnp.int32)[None, :, None] + start).astype(jnp.int32)

        window_mask = (kv_positions >= (q_pos - window_left)) & (kv_positions <= (q_pos + window_right))
        mask = mask & window_mask

    return mask[:, :, None, :]


def apply_logits_soft_cap(scores: Float[Array, "... seq_len"], logits_soft_cap: float) -> Float[Array, "... seq_len"]:
    """Apply tanh soft capping to attention logits.

    Computes ``logits_soft_cap * tanh(scores / logits_soft_cap)``, which smoothly
    limits the magnitude of logits to roughly ``[-logits_soft_cap, +logits_soft_cap]``
    while remaining differentiable everywhere.

    Args:
        scores: Attention logits, arbitrary shape ``[..., seq_len]``.
        logits_soft_cap: The capping radius (must be > 0).

    Returns:
        Soft-capped scores, same shape and dtype as ``scores``.
    """
    return jnp.tanh(scores / logits_soft_cap) * logits_soft_cap


def apply_attention_sinks_block(
    scores: Float[Array, "batch q_len heads block_size"],
    sink_scores: Float[Array, "heads num_sinks"] | None = None,
    num_sinks: int = 0,
    block_offset: int = 0,
) -> Float[Array, "batch q_len heads block_size"]:
    """Add attention-sink bias values to the scores for a single KV block.

    For each KV position ``p = block_offset + i`` that falls within
    ``[0, num_sinks)``, the corresponding score is increased by
    ``sink_scores[h, p]`` (or ``sink_scores[p]`` if 1-D).  Positions
    at or beyond ``num_sinks`` receive no bias.

    Note:
        This function is used by the non-scan attention path.  The main
        ``ragged_flash_attention_xla`` uses a different sink-seeding strategy
        (initialising the running max/normaliser from ``softmax_aux`` before
        the KV scan) and does not call this function.

    Args:
        scores: Attention logits for this block, shape ``[B, Q, H, block_size]``.
        sink_scores: Learned sink biases, shape ``[H, num_sinks]`` or
            ``[num_sinks]``.  If ``None`` or ``num_sinks == 0``, scores are
            returned unchanged.
        num_sinks: Number of leading KV positions that are sink positions.
        block_offset: Token offset of the first element of this block within
            the full KV sequence.

    Returns:
        Scores with sink biases applied, same shape as input ``scores``.
    """
    if num_sinks == 0 or sink_scores is None:
        return scores

    _batch_size, _q_len, heads, block_size_val = scores.shape

    if sink_scores.ndim == 1:
        sink_scores = jnp.broadcast_to(sink_scores[None, :], (heads, num_sinks))

    block_positions = jnp.arange(block_size_val) + block_offset

    is_sink_position = block_positions < num_sinks
    sink_indices = jnp.minimum(block_positions, num_sinks - 1)

    block_sink_biases = sink_scores[:, sink_indices]

    block_sink_biases = jnp.where(is_sink_position[None, :], block_sink_biases, 0.0)

    block_sink_biases = block_sink_biases[None, None, :, :]

    return scores + block_sink_biases


def flash_attention_block(
    carry: tuple[Array, Array, Array],
    block_inputs: tuple[Array, Array, Array, Array],
    softmax_scale: float,
    logits_soft_cap: float | None = None,
) -> tuple[tuple[Array, Array, Array], None]:
    """Process one KV block with the online-softmax flash attention algorithm.

    Performs a single iteration of the blockwise online-softmax accumulation:

    * Computes scaled dot-product scores ``q * scale @ k_block.T``.
    * Applies optional tanh soft capping.
    * Masks invalid positions with ``finfo.min`` (avoids ``-inf`` NaNs in
      all-masked blocks).
    * Updates the running maximum ``m_new = max(m_prev, max(scores))``.
    * Rescales the previous accumulator and normaliser by ``exp(m_prev - m_new)``.
    * Accumulates ``exp(scores - m_new) @ v_block`` into the output.

    GQA support: if ``kv_heads < q_heads``, ``k_block`` and ``v_block`` are
    broadcast-repeated along the head axis (``repeat_factor = q_heads // kv_heads``).

    Args:
        carry: Running online-softmax state ``(output, max_logits, normalizer)``
            where shapes are ``[B, Q, H, D]``, ``[B, Q, H, 1]``, ``[B, Q, H, 1]``.
        block_inputs: Tuple of
            ``(queries, keys_block, values_block, mask_block)`` with shapes
            ``[B, Q, H_q, D]``, ``[B, block, H_kv, D]``, ``[B, block, H_kv, D]``,
            ``[B, Q, 1, block]`` (mask is ``True`` for valid positions).
        softmax_scale: Multiplicative scale for QK^T logits.
        logits_soft_cap: Optional tanh soft-capping radius.

    Returns:
        Updated ``(output, max_logits, normalizer)`` carry and ``None`` as the
        per-step output (no per-block outputs are needed).
    """
    o_prev, m_prev, l_prev = carry
    q, k_block, v_block, mask_block = block_inputs

    _batch_size, _q_len, q_heads, _head_dim = q.shape
    _, _block_size, kv_heads, _ = k_block.shape

    if kv_heads < q_heads:
        assert q_heads % kv_heads == 0, f"Query heads {q_heads} must be divisible by KV heads {kv_heads}"
        repeat_factor = q_heads // kv_heads
        k_block = jnp.repeat(k_block, repeat_factor, axis=2)
        v_block = jnp.repeat(v_block, repeat_factor, axis=2)

    scores = jnp.einsum("...qhd,...khd->...qhk", q * softmax_scale, k_block)

    if logits_soft_cap is not None:
        scores = apply_logits_soft_cap(scores, logits_soft_cap)

    mask_expanded = jnp.broadcast_to(mask_block, scores.shape)

    min_score = jnp.finfo(scores.dtype).min
    scores = jnp.where(mask_expanded, scores, min_score)

    m_curr = jnp.max(scores, axis=-1, keepdims=True)
    m_new = jnp.maximum(m_prev, m_curr)

    exp_scores = jnp.exp(scores - m_new)
    exp_scores = jnp.where(mask_expanded, exp_scores, 0.0)

    l_curr = jnp.sum(exp_scores, axis=-1, keepdims=True)
    correction_prev = jnp.exp(m_prev - m_new)
    l_new = correction_prev * l_prev + l_curr

    l_new_safe = jnp.where(l_new == 0, 1.0, l_new)

    o_curr_times_l_curr = jnp.einsum("...qhk,...khd->...qhd", exp_scores, v_block)
    o_new = (correction_prev * l_prev * o_prev + o_curr_times_l_curr) / l_new_safe

    o_new = o_new.astype(o_prev.dtype)
    m_new = m_new.astype(m_prev.dtype)
    l_new = l_new.astype(l_prev.dtype)

    return (o_new, m_new, l_new), None


def ragged_flash_attention_xla(
    query: Float[Array, "batch q_len num_heads head_dim"],
    key: Float[Array, "batch kv_len num_heads head_dim"],
    value: Float[Array, "batch kv_len num_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch q_len num_heads head_dim"]:
    """XLA-compatible ragged flash attention over arbitrary-length query and KV sequences.

    Implements the online-softmax (flash attention) algorithm over blocked KV
    tiles for a ragged batch where each sequence has a distinct valid KV range.
    Supports multi-query (Q > 1), sliding-window local attention, tanh logit
    soft-capping, and attention-sink initialisation.

    The attention mask is built from ``sequence_start``/``sequence_end`` per
    sequence, with an optional per-sequence sliding window applied on top.

    If ``softmax_aux`` is provided, the running maximum and normaliser of the
    online-softmax are seeded from the sink logits before the KV scan begins.
    This ensures sink tokens always receive some probability mass.

    Args:
        query: Query tensor, shape ``[B, Q, H_q, D]``.
        key: Key tensor, shape ``[B, K, H_kv, D]``.  GQA is supported:
            ``H_kv`` can be less than ``H_q`` as long as ``H_q % H_kv == 0``.
        value: Value tensor, shape ``[B, K, H_kv, D]``.
        sequence_start: First valid KV position per sequence, shape ``[B]``.
        sequence_end: One-past-last valid KV position per sequence, shape ``[B]``.
        softmax_scale: Scale for QK^T logits.  Defaults to ``1/sqrt(D)``.
        block_size: Number of KV tokens per scan step.  Default 256.
        sliding_window: Optional ``(left, right)`` local-attention window sizes.
        logits_soft_cap: Optional tanh capping radius for logits.
        softmax_aux: Optional sink logits.  Accepted shapes:

            * 1-D ``[num_sinks]``: broadcast over all heads.
            * 2-D ``[1, num_sinks]``: broadcast over all heads.
            * 2-D ``[H_q, num_sinks]``: per-query-head sink values.

            These values are used only to seed the online-softmax state; they
            do not produce corresponding output values.

    Returns:
        Attention output, shape ``[B, Q, H_q, D]``, same dtype as ``query``.
    """
    batch_size, q_len, num_heads, head_dim = query.shape
    _, kv_len, kv_heads, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim)

    sink_logits = None
    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim == 1:
            sink_logits = aux.reshape(1, 1, 1, -1)
        elif aux.ndim == 2:
            if aux.shape[0] == 1:
                sink_logits = aux.reshape(1, 1, 1, -1)
            elif aux.shape[0] == num_heads:
                sink_logits = aux.reshape(1, 1, num_heads, -1)
            else:
                raise ValueError(f"softmax_aux first dim must be 1 or num_heads ({num_heads}); got {aux.shape[0]}")
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")
        sink_logits = jnp.broadcast_to(sink_logits, (batch_size, q_len, num_heads, sink_logits.shape[-1]))

    mask = create_attention_mask(batch_size, q_len, kv_len, sequence_start, sequence_end, sliding_window=sliding_window)

    num_blocks = (kv_len + block_size - 1) // block_size

    output_init = jnp.zeros_like(query, dtype=query.dtype)
    if sink_logits is None:
        max_logits_init = jnp.full((batch_size, q_len, num_heads, 1), -jnp.inf, dtype=jnp.float32)
        normalizer_init = jnp.zeros((batch_size, q_len, num_heads, 1), dtype=jnp.float32)
    else:
        max_logits_init = jnp.max(sink_logits, axis=-1, keepdims=True)
        normalizer_init = jnp.sum(jnp.exp(sink_logits - max_logits_init), axis=-1, keepdims=True)

    pad_len = num_blocks * block_size - kv_len
    if pad_len > 0:
        key = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")
        value = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")

        if mask.ndim == 4:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, pad_len)), mode="constant")
        elif mask.ndim == 5:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_len)), mode="constant")

    key_blocks = key.reshape(batch_size, num_blocks, block_size, kv_heads, head_dim)
    value_blocks = value.reshape(batch_size, num_blocks, block_size, kv_heads, head_dim)

    if mask.ndim == 4:
        mask_blocks = mask.reshape(batch_size, q_len, 1, num_blocks, block_size)
    else:
        mask_blocks = mask.reshape(batch_size, q_len, mask.shape[2], num_blocks, block_size)
    mask_blocks = jnp.transpose(mask_blocks, (0, 3, 1, 2, 4))

    def scan_fn(carry, inputs):
        """Process one KV block in the flash attention scan.

        Loads the key, value, and mask blocks for the current index
        and delegates to the flash attention block computation.

        Args:
            carry: Online softmax state (output, max_logits, normalizer).
            inputs: Tuple containing the block index.

        Returns:
            Updated carry and None (no per-step output).
        """
        (block_idx,) = inputs
        k_block = key_blocks[:, block_idx]
        v_block = value_blocks[:, block_idx]
        m_block = mask_blocks[:, block_idx]

        return flash_attention_block(
            carry,
            (query, k_block, v_block, m_block),
            softmax_scale,
            logits_soft_cap=logits_soft_cap,
        )

    (output, _, _), _ = lax.scan(scan_fn, (output_init, max_logits_init, normalizer_init), (jnp.arange(num_blocks),))

    return output


def ragged_decode_mqa_xla(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Single-token MQA/GQA decode attention for a ragged batch (XLA).

    Splits the ``num_q_heads`` query heads into ``num_kv_heads`` groups of
    size ``group_size = num_q_heads // num_kv_heads``, then runs
    ``ragged_flash_attention_xla`` over each KV-head group via ``jax.vmap``.
    The block size is taken from ``fwd_params.kv_blocksize`` (or defaults to
    256) and is clamped to ``[1, kv_len]``.

    Args:
        query: Single decode-step query, shape ``[B, H_q, D]``.
        key: Full KV cache keys, shape ``[B, S, H_kv, D]``.
        value: Full KV cache values, shape ``[B, S, H_kv, D]``.
        sequence_start: First valid KV position per sequence, shape ``[B]``.
        sequence_end: One-past-last valid KV position, shape ``[B]``.
        softmax_scale: Scale for QK^T logits.  Defaults to ``1/sqrt(D)``.
        fwd_params: Optional ``FwdParams``; only ``kv_blocksize`` is used.
        sliding_window: Optional ``(left, right)`` local-attention window.
        logits_soft_cap: Optional tanh capping radius for logits.
        softmax_aux: Optional sink logits.  Accepted shapes:

            * 1-D ``[num_sinks]``: broadcast over all heads.
            * 2-D ``[H_kv, num_sinks]``: per-KV-head sinks.
            * 2-D ``[H_q, num_sinks]``: per-query-head sinks; reshaped to
              ``[H_kv, group_size, num_sinks]`` for per-group dispatch.

    Returns:
        Attention output, shape ``[B, H_q, D]``.
    """
    batch_size, num_heads_q, head_dim = query.shape
    _, kv_len, num_heads_kv, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim)

    if fwd_params is None:
        fwd_params = FwdParams()
    block_size = 256 if fwd_params.kv_blocksize is None else int(fwd_params.kv_blocksize)
    block_size = max(1, min(block_size, kv_len))

    group_size = num_heads_q // num_heads_kv
    query = query.reshape(batch_size, num_heads_kv, group_size, head_dim)

    query = jnp.transpose(query, (1, 0, 2, 3))
    key = jnp.transpose(key, (2, 0, 1, 3))
    value = jnp.transpose(value, (2, 0, 1, 3))

    aux = softmax_aux
    if aux is not None and aux.ndim == 2:
        if aux.shape[0] == num_heads_kv:
            aux = aux
        elif aux.shape[0] == num_heads_q:
            aux = aux.reshape(num_heads_kv, group_size, aux.shape[1])
        else:
            raise ValueError(
                "softmax_aux must have shape (num_sinks,), (num_kv_heads, num_sinks) or (num_q_heads, num_sinks); "
                f"got shape {aux.shape} for num_q_heads={num_heads_q}, num_kv_heads={num_heads_kv}."
            )

    def process_kv_head(q_group, k_head, v_head, aux_i):
        """Compute attention for one KV head group against its query heads.

        Reshapes the single KV head into a broadcastable form and runs
        flash attention with the query group for this KV head.

        Args:
            q_group: Query group for this KV head [batch, group_size, head_dim].
            k_head: Key for this KV head [batch, kv_len, head_dim].
            v_head: Value for this KV head [batch, kv_len, head_dim].
            aux_i: Attention sink auxiliary logits for this head or None.

        Returns:
            Attention output for this head group [batch, group_size, head_dim].
        """
        k_head = k_head[:, :, None, :]
        v_head = v_head[:, :, None, :]
        q_group = q_group[:, None, :, :]

        output = ragged_flash_attention_xla(
            q_group,
            k_head,
            v_head,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=block_size,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=aux_i,
        )

        return output[:, 0, :, :]

    if aux is None or aux.ndim == 1:
        outputs = jax.vmap(process_kv_head, in_axes=(0, 0, 0, None))(query, key, value, aux)
    else:
        outputs = jax.vmap(process_kv_head, in_axes=(0, 0, 0, 0))(query, key, value, aux)

    outputs = jnp.transpose(outputs, (1, 0, 2, 3))
    return outputs.reshape(batch_size, num_heads_q, head_dim)


@ejit(static_argnames=["block_size", "softmax_scale", "logits_soft_cap", "sliding_window"])
def inner_decode_xla(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> chex.Array:
    """JIT-compiled dispatch for ragged MQA/GQA flash attention.

    Handles both 3-D (single decode step, ``query.shape=[B, H_q, D]``) and
    4-D (multi-token, ``query.shape=[B, q_len, H_q, D]``) query layouts:

    * 3-D input: expand to ``[B, 1, H_q, D]``, then collapse back after.
    * ``q_len == 1``: use the efficient single-step ``ragged_decode_mqa_xla``.
    * ``q_len > 1``: broadcast KV heads if needed (GQA), then call
      ``ragged_flash_attention_xla`` directly.

    Static args (cached across calls): ``block_size``, ``softmax_scale``,
    ``logits_soft_cap``, ``sliding_window``.

    Args:
        query: Query tensor.  Either ``[B, H_q, D]`` or ``[B, q_len, H_q, D]``.
        key: Key cache, shape ``[B, S, H_kv, D]``.
        value: Value cache, shape ``[B, S, H_kv, D]``.
        sequence_start: First valid KV position per sequence, shape ``[B]``.
        sequence_end: One-past-last valid KV position, shape ``[B]``.
        softmax_scale: Scale for QK^T logits.  Defaults to ``1/sqrt(D)`` when
            ``None``.  Treated as a static argument for JIT specialisation.
        block_size: KV block size for flash iteration.  Static for JIT.
        sliding_window: Optional ``(left, right)`` local-attention window.
            Static for JIT.
        logits_soft_cap: Optional tanh capping radius.  Static for JIT.
        softmax_aux: Optional attention-sink logits.  Shape ``[H, num_sinks]``
            or ``[num_sinks]``.

    Returns:
        Output tensor matching the input query shape (3-D or 4-D).
    """
    batch_size = query.shape[0]
    num_heads_q = query.shape[-2]
    head_dim = query.shape[-1]

    out_shape = (batch_size, 1, num_heads_q, head_dim)
    if query.ndim == 3:
        query = jnp.expand_dims(query, 1)
        out_shape = (batch_size, num_heads_q, head_dim)

    if query.shape[1] == 1:
        query = query[:, 0]
        output = ragged_decode_mqa_xla(
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=block_size,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )
    else:
        _, _seq_len_q, _, _ = query.shape
        _, _seq_len_kv, num_heads_kv, _ = key.shape

        if num_heads_kv != num_heads_q:
            repeat_factor = num_heads_q // num_heads_kv
            key = jnp.repeat(key, repeat_factor, axis=2)
            value = jnp.repeat(value, repeat_factor, axis=2)

        output = ragged_flash_attention_xla(
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=block_size,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )

    return jnp.reshape(output, out_shape)
