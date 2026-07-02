# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared internals for DeepSpec-style drafter modules.

This file is intentionally private to the model zoo.  It holds the repeated
mechanics used by EAGLE3, DSpark, and DFlash: target-layer feature extraction,
dense JAX masks equivalent to DeepSpec's block masks, small Qwen-style
attention/MLP blocks, Markov logit heads, and confidence prediction.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.utils import ACT2FN
from easydel.layers import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear


@auto_pytree
class DraftModelOutput:
    """Output schema shared by the drafter families.

    Sequence-mode drafters fill ``hidden_states`` plus ``draft_logits`` /
    ``logits`` shaped ``[batch, seq, vocab]``.  DSpark/DFlash block-mode
    forwards fill ``block_logits``, ``target_ids``, ``eval_mask``,
    ``block_keep_mask``, and ``anchor_positions`` using DeepSpec's
    ``[batch, anchors, block]`` layout.
    """

    hidden_states: Float[Array, "batch seq hidden"] | None = None
    draft_logits: Float[Array, "batch seq vocab"] | Float[Array, "batch anchors block vocab"] | None = None
    logits: Float[Array, "batch seq vocab"] | None = None
    target_logits: Float[Array, "batch seq vocab"] | None = None
    confidence_logits: Float[Array, "batch anchors block"] | Float[Array, "batch seq"] | None = None
    target_ids: Int[Array, "batch anchors block"] | None = None
    eval_mask: Bool[Array, "batch anchors block"] | None = None
    block_keep_mask: Bool[Array, "batch anchors"] | None = None
    anchor_positions: Int[Array, "batch anchors"] | None = None
    block_logits: Float[Array, "batch anchors block vocab"] | None = None
    aligned_target_logits: Float[Array, "batch anchors block vocab"] | None = None


def validate_target_layer_ids(
    layer_ids: tp.Sequence[int],
    *,
    num_target_layers: int | None = None,
    allow_embedding: bool,
    expected_count: int | None = None,
) -> tuple[int, ...]:
    """Validate DeepSpec target-layer selectors.

    DeepSpec uses Hugging Face hidden-state indexing where decoder layer
    ``n`` is read from ``hidden_states[n + 1]``.  DSpark/DFlash additionally
    allow ``-1`` to mean the embedding output at ``hidden_states[0]``; EAGLE3
    does not.

    Args:
        layer_ids: Sequence of target layer indices.
        num_target_layers: Maximum number of target layers (for range checking). Defaults to ``None``.
        allow_embedding: Whether ``-1`` (embedding layer) is permitted.
        expected_count: Exact number of layers required. Defaults to ``None``.

    Returns:
        Validated layer IDs as a tuple of ints.

    Raises:
        ValueError: If ``layer_ids`` is empty, has wrong count, contains invalid IDs, is out of range,
            or is not strictly increasing.
    """
    values = tuple(int(layer_id) for layer_id in layer_ids)
    if not values:
        raise ValueError("`target_layer_ids` must not be empty.")
    if expected_count is not None and len(values) != int(expected_count):
        raise ValueError(f"`target_layer_ids` must contain exactly {expected_count} layers.")
    previous = None
    for layer_id in values:
        if layer_id == -1 and not allow_embedding:
            raise ValueError("`-1` target layer is only valid for DSpark/DFlash.")
        if layer_id < -1 or (layer_id == -1 and not allow_embedding):
            raise ValueError(f"Invalid target layer id: {layer_id}.")
        if num_target_layers is not None and layer_id >= int(num_target_layers):
            raise ValueError(f"target_layer_id {layer_id} is out of range for {num_target_layers} target layers.")
        if previous is not None and layer_id <= previous:
            raise ValueError("`target_layer_ids` must be strictly increasing.")
        previous = layer_id
    return values


def extract_context_feature(
    hidden_states: tp.Any,
    layer_ids: tp.Sequence[int],
    *,
    target_hidden_size: int | None = None,
    allow_embedding: bool = True,
) -> jax.Array:
    """Extract and concatenate target hidden states.

    Tuple/list inputs follow Hugging Face output-hidden-state indexing:
    ``-1`` maps to embedding output at index 0, and layer ``n`` maps to index
    ``n + 1``.  Array inputs may either be pre-concatenated
    ``[batch, seq, layers * hidden]`` features or layer-stacked target states
    shaped ``[layers, batch, seq, hidden]`` / ``[batch, seq, layers, hidden]``.

    Args:
        hidden_states: Hidden states as a tuple/list, a 3-D array, or a 4-D array.
        layer_ids: Target layer indices to extract (validated internally).
        target_hidden_size: Hidden size per layer (used for 3-D array validation). Defaults to ``None``.
        allow_embedding: Whether ``-1`` (embedding layer) is permitted. Defaults to ``True``.

    Returns:
        Concatenated hidden features array.

    Raises:
        ValueError: If the input shape is unsupported or does not match ``target_hidden_size``.
    """
    layer_ids = validate_target_layer_ids(layer_ids, allow_embedding=allow_embedding)
    if isinstance(hidden_states, list | tuple):
        selected = [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids]
        return jnp.concatenate(selected, axis=-1)

    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim == 3:
        expected = None if target_hidden_size is None else len(layer_ids) * int(target_hidden_size)
        if expected is None or int(hidden_states.shape[-1]) == expected or len(layer_ids) == 1:
            return hidden_states
        raise ValueError("3D target hidden states must be a single layer or pre-concatenated target features.")
    if hidden_states.ndim != 4:
        raise ValueError(f"Unsupported target hidden-state shape: {hidden_states.shape}.")

    required_layers = max((0 if layer_id == -1 else layer_id + 1) for layer_id in layer_ids) + 1
    layer_first = int(hidden_states.shape[0]) >= required_layers and (
        int(hidden_states.shape[0]) == required_layers or int(hidden_states.shape[2]) < required_layers
    )
    sequence_first = int(hidden_states.shape[2]) >= required_layers
    if layer_first:
        selected = [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids]
    elif sequence_first:
        selected = [hidden_states[:, :, 0 if layer_id == -1 else layer_id + 1, :] for layer_id in layer_ids]
    else:
        raise ValueError(f"Target hidden states do not contain requested layers {layer_ids}: {hidden_states.shape}.")
    return jnp.concatenate(selected, axis=-1)


def make_position_ids(input_ids: jax.Array | None = None, inputs_embeds: jax.Array | None = None) -> jax.Array:
    """Create dense absolute position IDs for a token or embedding sequence.

    Generates a ``[batch, seq]`` array where each row is ``[0, 1, ..., seq_len-1]``.

    Args:
        input_ids: Token IDs shaped ``[batch, seq]``. Either this or ``inputs_embeds`` must be
            provided.
        inputs_embeds: Embedding vectors shaped ``[batch, seq, ...]``. Either this or ``input_ids``
            must be provided.

    Returns:
        Position indices shaped ``[batch, seq]`` with dtype ``int32``.

    Raises:
        ValueError: If both ``input_ids`` and ``inputs_embeds`` are ``None``.
    """
    reference = input_ids if input_ids is not None else inputs_embeds
    if reference is None:
        raise ValueError("`input_ids` or `inputs_embeds` must be provided.")
    batch_size, seq_len = reference.shape[:2]
    return jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))


def causal_attention_mask(batch_size: int, q_len: int, kv_len: int, past_seen_tokens: int = 0) -> jax.Array:
    """Build a boolean causal mask for dense attention logits.

    The mask is ``True`` where a query position may attend to a key position
    (i.e. key position <= query position).

    Args:
        batch_size: Number of sequences in the batch.
        q_len: Number of query positions.
        kv_len: Number of key/value positions.
        past_seen_tokens: Number of tokens already seen in the past (KV cache offset). Defaults to
            ``0``.

    Returns:
        Boolean mask shaped ``[batch_size, q_len, kv_len]``.
    """
    q_pos = jnp.arange(past_seen_tokens, past_seen_tokens + q_len)[:, None]
    kv_pos = jnp.arange(kv_len)[None, :]
    return jnp.broadcast_to(kv_pos <= q_pos, (batch_size, q_len, kv_len))


def expand_attention_mask(mask: jax.Array | None, *, q_len: int, kv_len: int) -> jax.Array | None:
    """Broadcast a padding/attention mask to ``[batch, q_len, kv_len]``.

    Accepts 2-D, 3-D, or 4-D masks and normalizes them to a 3-D boolean mask.

    Args:
        mask: Input mask of shape ``[batch, kv_len]``, ``[batch, q_len, kv_len]``, or
            ``[batch, 1, q_len, kv_len]``.
        q_len: Target query length.
        kv_len: Target key/value length.

    Returns:
        Boolean mask of shape ``[batch, q_len, kv_len]``, or ``None`` if input was ``None``.
    """
    if mask is None:
        return None
    if mask.ndim == 2:
        return jnp.broadcast_to(mask[:, None, :kv_len].astype(bool), (mask.shape[0], q_len, kv_len))
    if mask.ndim == 3:
        return mask.astype(bool)
    if mask.ndim == 4:
        return jnp.squeeze(mask, axis=1).astype(bool)
    return mask.astype(bool)


def gather_sequence(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather ``values[B, S, ...]`` at per-batch sequence ``indices[B, ...]``.

    Flattens the batch and index dimensions, uses ``take_along_axis`` on axis ``1``,
    then restores the original index shape.

    Args:
        values: Array of shape ``[batch, seq, ...]`` to gather from.
        indices: Integer indices of shape ``[batch, ...]`` (any rank >= 1).

    Returns:
        Gathered values of shape ``indices.shape + values.shape[2:]``.
    """
    batch_size = values.shape[0]
    suffix = values.shape[2:]
    flat_indices = indices.reshape(batch_size, -1)
    gather_idx = flat_indices.reshape((batch_size, flat_indices.shape[1]) + (1,) * len(suffix))
    gathered = jnp.take_along_axis(values, gather_idx, axis=1)
    return gathered.reshape(indices.shape + suffix)


def sample_anchor_positions(
    loss_mask: jax.Array,
    *,
    num_anchors: int,
) -> tuple[jax.Array, jax.Array]:
    """Select DSpark/DFlash anchor positions from supervised token spans.

    DeepSpec samples anchors randomly during training.  This EasyDeL helper uses
    a deterministic hash order so tests and jit traces are stable while anchors
    still cover the full valid span instead of always taking the earliest tokens.

    Args:
        loss_mask: Float supervision mask shaped ``[batch, seq]`` (values > 0.5 are supervised).
        num_anchors: Number of anchor positions to sample per batch.

    Returns:
        Tuple of ``(anchors, keep)`` where ``anchors`` is an int32 array of shape
        ``[batch, num_anchors]`` and ``keep`` is a boolean array of the same shape
        indicating which anchors are valid.
    """
    batch_size, seq_len = loss_mask.shape
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        anchors = jnp.zeros((batch_size, int(num_anchors)), dtype=jnp.int32)
        return anchors, jnp.zeros_like(anchors, dtype=bool)

    valid = (loss_mask[:, :num_candidates] > 0.5) & (loss_mask[:, 1 : num_candidates + 1] > 0.5)
    candidate_ids = jnp.broadcast_to(jnp.arange(num_candidates, dtype=jnp.int32)[None, :], (batch_size, num_candidates))
    batch_ids = jnp.arange(batch_size, dtype=jnp.uint32)[:, None]
    candidate_hash = candidate_ids.astype(jnp.uint32)
    scores = (candidate_hash ^ jnp.uint32(0x9E3779B9)) * jnp.uint32(0x85EBCA6B)
    scores = (scores ^ (scores >> jnp.uint32(13))) * jnp.uint32(0xC2B2AE35)
    scores = scores ^ (scores >> jnp.uint32(16)) ^ (batch_ids * jnp.uint32(0x27D4EB2D))
    invalid_score = jnp.array(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32)
    sorted_order = jnp.argsort(jnp.where(valid, scores, invalid_score), axis=1)
    sorted_candidates = jnp.take_along_axis(candidate_ids, sorted_order, axis=1)
    sorted_keep = jnp.take_along_axis(valid, sorted_order, axis=1)
    if num_candidates < int(num_anchors):
        pad = jnp.full((batch_size, int(num_anchors) - num_candidates), seq_len + 1, dtype=jnp.int32)
        keep_pad = jnp.zeros((batch_size, int(num_anchors) - num_candidates), dtype=bool)
        sorted_candidates = jnp.concatenate((sorted_candidates, pad), axis=1)
        sorted_keep = jnp.concatenate((sorted_keep, keep_pad), axis=1)
    anchors = sorted_candidates[:, : int(num_anchors)]
    keep = sorted_keep[:, : int(num_anchors)]
    anchor_sentinel = jnp.full_like(anchors, seq_len + 1)
    anchors = jnp.sort(jnp.where(keep, anchors, anchor_sentinel), axis=1)
    keep = anchors < seq_len
    return jnp.where(keep, anchors, 0), keep


def build_eval_mask(
    *,
    input_ids: jax.Array,
    loss_mask: jax.Array,
    label_indices: jax.Array,
    safe_label_indices: jax.Array,
    block_keep_mask: jax.Array,
) -> jax.Array:
    """Build the contiguous block supervision mask used by DSpark losses.

    A position is valid when the corresponding label token is inside the
    sequence, its loss mask is active, and the block itself is kept.
    Cumulative product enforces contiguous supervision from the start of each
    block.

    Args:
        input_ids: Original token IDs shaped ``[batch, seq]``.
        loss_mask: Float supervision mask shaped ``[batch, seq]`` (values > 0.5 are active).
        label_indices: Indices into ``input_ids`` for each block token, shaped ``[batch, anchors,
            block]``.
        safe_label_indices: Clipped version of ``label_indices`` safe for ``take_along_axis``.
        block_keep_mask: Boolean mask of shape ``[batch, anchors]`` indicating active blocks.

    Returns:
        Boolean evaluation mask of shape ``[batch, anchors, block]``.
    """
    batch_size, seq_len = input_ids.shape
    num_blocks = label_indices.shape[1]
    target_loss = jnp.take_along_axis(
        jnp.broadcast_to(loss_mask[:, None, :], (batch_size, num_blocks, seq_len)),
        safe_label_indices,
        axis=2,
    )
    eval_mask = (label_indices < seq_len) & (target_loss > 0.5) & block_keep_mask[:, :, None]
    return jnp.cumprod(eval_mask.astype(jnp.int32), axis=-1).astype(bool)


def create_noise_ids(
    input_ids: jax.Array,
    anchor_positions: jax.Array,
    block_keep_mask: jax.Array,
    *,
    mask_token_id: int,
    block_size: int,
) -> jax.Array:
    """Create teacher-forced draft-block token IDs.

    Each block starts from the anchor token and fills the remaining positions
    with the configured mask token, matching DSpark/DFlash's noise-embedding
    input to the block drafter.

    Args:
        input_ids: Original token IDs shaped ``[batch, seq]``.
        anchor_positions: Anchor positions shaped ``[batch, num_blocks]``.
        block_keep_mask: Boolean mask of shape ``[batch, num_blocks]`` indicating active blocks.
        mask_token_id: Token ID used for masked (non-anchor) positions.
        block_size: Number of tokens per block.

    Returns:
        Noise token IDs shaped ``[batch, num_blocks * block_size]``.
    """
    batch_size, num_blocks = anchor_positions.shape
    noise_ids = jnp.full((batch_size, num_blocks, int(block_size)), int(mask_token_id), dtype=jnp.int32)
    anchor_token_ids = jnp.take_along_axis(input_ids, anchor_positions, axis=1)
    noise_ids = noise_ids.at[:, :, 0].set(jnp.where(block_keep_mask, anchor_token_ids, int(mask_token_id)))
    return noise_ids.reshape(batch_size, num_blocks * int(block_size))


def create_dspark_attention_mask(
    *,
    anchor_positions: jax.Array,
    block_keep_mask: jax.Array,
    seq_len: int,
    block_size: int,
) -> jax.Array:
    """Create DSpark's block attention mask as a dense JAX boolean array.

    Query positions inside a draft block can attend to target context strictly
    before that block's anchor and to draft positions from the same block.  They
    cannot attend across sampled blocks.

    Args:
        anchor_positions: Anchor positions shaped ``[batch, num_blocks]``.
        block_keep_mask: Boolean mask of shape ``[batch, num_blocks]`` indicating active blocks.
        seq_len: Length of the target context sequence.
        block_size: Number of tokens per block.

    Returns:
        Boolean attention mask of shape ``[batch, num_blocks * block_size, seq_len + num_blocks *
        block_size]``.
    """
    batch_size, num_blocks = anchor_positions.shape
    q_idx = jnp.arange(num_blocks * int(block_size))
    q_block = q_idx // int(block_size)
    anchor_for_q = jnp.take_along_axis(anchor_positions, jnp.broadcast_to(q_block[None, :], (batch_size, q_idx.size)), 1)

    context_idx = jnp.arange(seq_len)
    context_allowed = context_idx[None, None, :] < anchor_for_q[:, :, None]

    draft_idx = jnp.arange(num_blocks * int(block_size))
    draft_block = draft_idx // int(block_size)
    draft_allowed = jnp.broadcast_to(
        q_block[None, :, None] == draft_block[None, None, :],
        (batch_size, q_idx.size, draft_idx.size),
    )

    keep = jnp.take_along_axis(block_keep_mask, jnp.broadcast_to(q_block[None, :], (batch_size, q_idx.size)), 1)
    return jnp.concatenate((context_allowed, draft_allowed), axis=-1) & keep[:, :, None]


class DraftMLP(spx.Module):
    """Qwen-style SwiGLU feed-forward block for draft layers."""

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize a Qwen-style SwiGLU draft MLP.

        Args:
            config: Model configuration object. Must expose ``hidden_act``, ``hidden_size``,
                ``intermediate_size``, and ``initializer_range``.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def forward(self, hidden_states: jax.Array) -> jax.Array:
        """Apply gated feed-forward projection and return hidden-size states.

        Args:
            hidden_states: Input hidden states of shape ``[batch, seq, hidden_size]``.

        Returns:
            Output hidden states of shape ``[batch, seq, hidden_size]``.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DraftAttention(spx.Module):
    """Dense RoPE GQA attention used by EAGLE3 and DSpark layers.

    The input widths are parameterized because EAGLE3 attends over
    ``[input_embedding, target_feature]`` concatenations, while DSpark/DFlash
    attends from draft-token queries into projected target context plus draft
    tokens.
    """

    def __init__(
        self,
        config: tp.Any,
        *,
        q_input_size: int,
        kv_input_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ):
        """Initialize dense RoPE GQA draft attention.

        Args:
            config: Model configuration object. Must expose ``num_attention_heads``,
                ``num_key_value_heads``, ``head_dim``, ``attention_bias``, ``hidden_size``,
                ``rms_norm_eps``, ``initializer_range``, and ``get_basic_rope``.
            q_input_size: Size of the query input projection.
            kv_input_size: Size of the key/value input projection.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        self.config = config
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim**-0.5
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = ColumnParallelLinear(
            q_input_size,
            q_out,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.k_proj = ColumnParallelLinear(
            kv_input_size,
            kv_out,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.v_proj = ColumnParallelLinear(
            kv_input_size,
            kv_out,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.o_proj = RowParallelLinear(
            q_out,
            config.hidden_size,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.rotary = config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def _apply_rotary(
        self,
        query: jax.Array,
        key: jax.Array,
        q_position_ids: jax.Array | None,
        kv_position_ids: jax.Array | None,
        frequencies: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply RoPE to query/key tensors, allowing different Q and KV positions.

        Args:
            query: Query tensor of shape ``[batch, q_len, num_heads, head_dim]``.
            key: Key tensor of shape ``[batch, kv_len, num_kv_heads, head_dim]``.
            q_position_ids: Position IDs for queries, or ``None`` to skip RoPE.
            kv_position_ids: Position IDs for keys, or ``None`` to skip RoPE.
            frequencies: Pre-computed rotary frequencies, or ``None``.

        Returns:
            Tuple of ``(rotated_query, rotated_key)`` with the same shapes as the inputs.
        """
        if q_position_ids is None or kv_position_ids is None:
            return query, key
        if query.shape[1] == key.shape[1] and q_position_ids.shape == kv_position_ids.shape:
            return self.rotary(positions=q_position_ids, query=query, key=key, frequencies=frequencies)
        query, _ = self.rotary(
            positions=q_position_ids,
            query=query,
            key=jnp.zeros_like(query),
            frequencies=frequencies,
        )
        key, _ = self.rotary(
            positions=kv_position_ids,
            query=key,
            key=key,
            frequencies=frequencies,
        )
        return query, key

    def forward(
        self,
        query_states: jax.Array,
        key_value_states: jax.Array,
        attention_mask: jax.Array | None = None,
        q_position_ids: jax.Array | None = None,
        kv_position_ids: jax.Array | None = None,
        frequencies: jax.Array | None = None,
    ) -> jax.Array:
        """Project Q/K/V, apply RoPE and dense masking, then return attended states.

        Args:
            query_states: Query input of shape ``[batch, q_len, q_input_size]``.
            key_value_states: Key/value input of shape ``[batch, kv_len, kv_input_size]``.
            attention_mask: Optional attention mask (broadcasted to ``[batch, q_len, kv_len]``).
            q_position_ids: Optional position IDs for queries.
            kv_position_ids: Optional position IDs for keys/values.
            frequencies: Optional pre-computed rotary frequencies.

        Returns:
            Attended output hidden states of shape ``[batch, q_len, hidden_size]``.
        """
        batch_size, q_len = query_states.shape[:2]
        kv_len = key_value_states.shape[1]
        query = self.q_proj(query_states).reshape(batch_size, q_len, self.num_heads, self.head_dim)
        key = self.k_proj(key_value_states).reshape(batch_size, kv_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(key_value_states).reshape(batch_size, kv_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self._apply_rotary(query, key, q_position_ids, kv_position_ids, frequencies)
        if self.num_key_value_groups > 1:
            key = jnp.repeat(key, self.num_key_value_groups, axis=2)
            value = jnp.repeat(value, self.num_key_value_groups, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query.astype(jnp.float32), key.astype(jnp.float32)) * self.scaling
        if attention_mask is not None:
            allowed = expand_attention_mask(attention_mask, q_len=q_len, kv_len=kv_len)
            if allowed is not None:
                scores = jnp.where(allowed[:, None, :, :], scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
        output = jnp.einsum("bhqk,bkhd->bqhd", probs, value).reshape(batch_size, q_len, -1)
        return self.o_proj(output)


class VanillaMarkovHead(spx.Module):
    """Memoryless low-rank Markov bias over the previous draft token."""

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize a memoryless Markov bias head.

        Args:
            config: Model configuration object. Must expose ``markov_rank``, ``vocab_size``,
                ``hidden_size``, and ``initializer_range``.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        self.markov_rank = int(config.markov_rank)
        self.prev_embed = Embed(
            config.vocab_size,
            self.markov_rank,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.markov_w2 = ColumnParallelLinear(
            self.markov_rank,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def get_prev_embeddings(self, token_ids: jax.Array) -> jax.Array:
        """Embed previous-token IDs into the Markov latent space.

        Args:
            token_ids: Integer token IDs of any shape.

        Returns:
            Embedded vectors of shape ``token_ids.shape + (markov_rank,)``.
        """
        return self.prev_embed(token_ids.astype(jnp.int32))

    def project_bias(self, latent_states: jax.Array) -> jax.Array:
        """Project Markov latent states to vocabulary-sized logit bias.

        Args:
            latent_states: Latent vectors of shape ``[..., markov_rank]``.

        Returns:
            Logit bias of shape ``[..., vocab_size]``.
        """
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: jax.Array, hidden_states: jax.Array | None = None) -> jax.Array:
        """Compute a token-only logit bias for one step or teacher-forced block.

        Args:
            token_ids: Integer token IDs of any shape.
            hidden_states: Unused in the vanilla head (ignored).

        Returns:
            Logit bias of shape ``token_ids.shape + (vocab_size,)``.
        """
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_logits(self, base_logits: jax.Array, *, token_ids: jax.Array, hidden_states: jax.Array | None = None):
        """Add the Markov bias to base drafter logits.

        Args:
            base_logits: Base logit tensor of shape ``[..., vocab_size]``.
            token_ids: Integer token IDs used to compute the Markov bias.
            hidden_states: Unused in the vanilla head (ignored).

        Returns:
            Adjusted logits of the same shape as ``base_logits``.
        """
        return base_logits + self.compute_step_bias(token_ids, hidden_states)


class GatedMarkovHead(VanillaMarkovHead):
    """Markov head that gates previous-token features by drafter hidden states."""

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize a hidden-state-gated Markov bias head.

        Args:
            config: Model configuration object. Must expose ``markov_rank``, ``vocab_size``,
                ``hidden_size``, and ``initializer_range``.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        super().__init__(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size + self.markov_rank,
            self.markov_rank,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def compute_step_bias(self, token_ids: jax.Array, hidden_states: jax.Array | None = None) -> jax.Array:
        """Compute hidden-state-conditioned Markov bias.

        Args:
            token_ids: Integer token IDs of any shape.
            hidden_states: Drafter hidden states of shape ``[..., hidden_size]``.

        Returns:
            Gated logit bias of shape ``[..., vocab_size]``.

        Raises:
            ValueError: If ``hidden_states`` is ``None``.
        """
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = jax.nn.sigmoid(self.gate_proj(jnp.concatenate((hidden_states, prev_embeddings), axis=-1)))
        return self.project_bias(gate * prev_embeddings)


class RNNMarkovHead(VanillaMarkovHead):
    """Recurrent Markov head that carries a low-rank state through each block."""

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize a recurrent GRU-like Markov bias head.

        Args:
            config: Model configuration object. Must expose ``markov_rank``, ``vocab_size``,
                ``hidden_size``, and ``initializer_range``.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        super().__init__(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.joint_proj = ColumnParallelLinear(
            2 * self.markov_rank + config.hidden_size,
            3 * self.markov_rank,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def _rnn_step(self, state: jax.Array, prev_embeddings: jax.Array, hidden_states: jax.Array):
        """Run one GRU-like update and emit a vocabulary bias.

        Args:
            state: Recurrent state of shape ``[..., markov_rank]``.
            prev_embeddings: Previous-token embeddings of shape ``[..., markov_rank]``.
            hidden_states: Drafter hidden states of shape ``[..., hidden_size]``.

        Returns:
            Tuple of ``(new_state, bias)`` where ``new_state`` has shape ``[..., markov_rank]`` and
            ``bias`` has shape ``[..., vocab_size]``.
        """
        gate_raw, candidate_raw, output_raw = jnp.split(
            self.joint_proj(jnp.concatenate((state, prev_embeddings, hidden_states), axis=-1)),
            3,
            axis=-1,
        )
        gate = jax.nn.sigmoid(gate_raw)
        candidate = jnp.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        return new_state, self.project_bias(jnp.tanh(output_raw))

    def compute_step_bias(self, token_ids: jax.Array, hidden_states: jax.Array | None = None) -> jax.Array:
        """Compute one-step recurrent bias with a zero initial state.

        Args:
            token_ids: Integer token IDs of any shape.
            hidden_states: Drafter hidden states of shape ``[..., hidden_size]``.

        Returns:
            Recurrent logit bias of shape ``[..., vocab_size]``.

        Raises:
            ValueError: If ``hidden_states`` is ``None``.
        """
        if hidden_states is None:
            raise ValueError("RNNMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = jnp.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_logits(self, base_logits: jax.Array, *, token_ids: jax.Array, hidden_states: jax.Array | None = None):
        """Apply recurrent Markov bias across a teacher-forced draft block.

        Args:
            base_logits: Base logits of shape ``[batch, anchors, block, vocab_size]`` or any shape
                compatible with ``compute_step_bias``.
            token_ids: Token IDs of shape ``[batch, anchors, block]``.
            hidden_states: Drafter hidden states of shape ``[batch, anchors, block, hidden_size]``.

        Returns:
            Corrected logits with the same shape as ``base_logits``.

        Raises:
            ValueError: If ``hidden_states`` is ``None``.
        """
        if base_logits.ndim != 4:
            return super().apply_logits(base_logits, token_ids=token_ids, hidden_states=hidden_states)
        if hidden_states is None:
            raise ValueError("RNNMarkovHead requires hidden_states.")
        state = jnp.zeros((*base_logits.shape[:-2], self.markov_rank), dtype=hidden_states.dtype)
        corrected = []
        for step in range(base_logits.shape[-2]):
            prev_embeddings = self.get_prev_embeddings(token_ids[..., step])
            state, bias = self._rnn_step(state, prev_embeddings, hidden_states[..., step, :])
            corrected.append(base_logits[..., step, :] + bias)
        return jnp.stack(corrected, axis=-2)


def build_markov_head(
    config: tp.Any,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
    precision: jax.lax.PrecisionLike,
    rngs: spx.Rngs,
) -> VanillaMarkovHead | None:
    """Instantiate the configured DSpark Markov head, or return ``None``.

    Args:
        config: Model configuration object. Must expose ``markov_rank`` and ``markov_head_type``.
        dtype: Computation dtype for the Markov head.
        param_dtype: Parameter dtype for the Markov head.
        precision: JAX precision setting for matmuls.
        rngs: spectrax RNG key manager.

    Returns:
        A ``VanillaMarkovHead`` subclass instance, or ``None`` if ``markov_rank <= 0``.

    Raises:
        ValueError: If ``markov_head_type`` is not one of ``"vanilla"``, ``"gated"``, or ``"rnn"``.
    """
    if int(config.markov_rank) <= 0:
        return None
    if config.markov_head_type == "vanilla":
        return VanillaMarkovHead(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
    if config.markov_head_type == "gated":
        return GatedMarkovHead(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
    if config.markov_head_type == "rnn":
        return RNNMarkovHead(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
    raise ValueError(f"Unsupported markov_head_type: {config.markov_head_type!r}")


class AcceptRatePredictor(spx.Module):
    """Linear confidence head for per-position draft acceptance logits."""

    def __init__(
        self,
        input_dim: int,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the linear confidence head.

        Args:
            input_dim: Size of the input feature vector.
            config: Model configuration object. Must expose ``initializer_range``.
            dtype: Computation dtype for the layer. Defaults to ``bfloat16``.
            param_dtype: Parameter dtype for the layer. Defaults to ``bfloat16``.
            precision: JAX precision setting for matmuls. Defaults to ``None``.
            rngs: spectrax RNG key manager.
        """
        self.proj = ColumnParallelLinear(
            input_dim,
            1,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def forward(self, features: jax.Array) -> jax.Array:
        """Return scalar confidence logits for each draft position.

        Args:
            features: Input features of shape ``[batch, seq, input_dim]`` or ``[..., input_dim]``.

        Returns:
            Scalar confidence logits of shape ``[batch, seq]`` or ``[...,]`` (last axis squeezed).
        """
        return jnp.squeeze(self.proj(features), axis=-1).astype(jnp.float32)


__all__ = (
    "AcceptRatePredictor",
    "DraftAttention",
    "DraftMLP",
    "DraftModelOutput",
    "build_eval_mask",
    "build_markov_head",
    "causal_attention_mask",
    "create_dspark_attention_mask",
    "create_noise_ids",
    "expand_attention_mask",
    "extract_context_feature",
    "gather_sequence",
    "make_position_ids",
    "sample_anchor_positions",
    "validate_target_layer_ids",
)
