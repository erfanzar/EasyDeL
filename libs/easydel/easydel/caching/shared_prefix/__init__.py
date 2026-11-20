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

"""Training-time cache views for a prefix-shared group forward.

A group forward scores ``G`` completions per prompt. Instead of running every
prompt ``G`` times, a *capture* pass runs the prompts once (batch ``B``) and
records each attention layer's post-RoPE key/value states; a *fork* pass then
runs only the completions (batch ``B * G``) and attends to
``concat(prefix[r // G], completion[r])``.

Both views are plain pytrees of arrays with no preallocated storage, so the
gather from the captured prefix is differentiable: gradients of the fork pass
flow back into the prompt activations of the capture pass.

With a dense attention operation (``vanilla``/``sdpa``) the fork pass never materializes that
concatenation: :meth:`PrefixForkView.grouped_fork_attention` lets the queries of all ``G`` completions of
a prompt attend that prompt's ``B``-row prefix once and merges it with each completion's causal
self-attention in one exact softmax. Other operations receive the concatenated key/value states.

Visibility in the concatenated fork pass is expressed with segment ids (``-1`` marks
padding) and causality with absolute column positions
(``q_positions = P + arange(C)``, ``kv_positions = arange(P + C)``), which the
dense attention operations consume directly instead of a dense mask.

Gated-delta linear-attention layers carry state instead of key/value history;
their capture/fork views live in :mod:`.recurrent`.
"""

from __future__ import annotations

import jax
from eformer.pytree import auto_pytree, field
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp

from .recurrent import (
    PREFIX_SHARED_RECURRENT_MODEL_TYPES,
    PrefixRecurrentCaptureView,
    PrefixRecurrentForkView,
    is_shared_prefix_recurrent_view,
)

__all__ = (
    "PREFIX_SHARED_ATTENTION_MECHANISMS",
    "PREFIX_SHARED_GROUPED_ATTENTION_MECHANISMS",
    "PREFIX_SHARED_RECURRENT_MODEL_TYPES",
    "PrefixCaptureView",
    "PrefixForkView",
    "PrefixRecurrentCaptureView",
    "PrefixRecurrentForkView",
    "is_shared_prefix_recurrent_view",
    "is_shared_prefix_view",
)

PREFIX_SHARED_ATTENTION_MECHANISMS = frozenset({"vanilla", "sdpa", "flash_attn2", "blocksparse"})
"""Attention operations verified on TPU (output and q/k/v gradients) for the prefix-shared fork representation."""

PREFIX_SHARED_GROUPED_ATTENTION_MECHANISMS = frozenset({"vanilla", "sdpa"})
"""Dense operations whose fork attention is computed by :meth:`PrefixForkView.grouped_fork_attention` instead."""

GROUPED_FORK_QUERY_BLOCK = 256
"""Completion query block of :meth:`PrefixForkView.grouped_fork_attention` (causal block skipping)."""


@auto_pytree
class PrefixCaptureView:
    """Records one layer's key/value states while passing them through unchanged.

    Attributes:
        key: Captured key states ``[batch, prefix_len, kv_heads, head_dim]``; ``None`` before capture.
        value: Captured value states ``[batch, prefix_len, kv_heads, head_dim]``; ``None`` before capture.
    """

    key: jax.Array | None = None
    value: jax.Array | None = None

    def concatenate_to_cache(
        self, *, key: jax.Array, value: jax.Array, mask_info: MaskInfo | None
    ) -> tuple[jax.Array, jax.Array, MaskInfo | None, PrefixCaptureView]:
        """Return the inputs unchanged plus a view holding them."""
        return key, value, mask_info, PrefixCaptureView(key=key, value=value)


@auto_pytree
class PrefixForkView:
    """Prepends a captured, group-shared prefix to each completion row's key/value states.

    Row ``r`` of the fork batch reads prefix row ``r // num_generations``.

    Attributes:
        prefix_key: Captured prefix keys ``[prompt_batch, prefix_len, kv_heads, head_dim]``.
        prefix_value: Captured prefix values ``[prompt_batch, prefix_len, kv_heads, head_dim]``.
        prefix_segment_ids: ``[prompt_batch, prefix_len]`` int32; ``0`` for prompt tokens, ``-1`` for padding.
        num_generations: Completions per prompt (``G``).
        deferred_attention: Set by the attention layer when its operation computes the fork attention with
            :meth:`grouped_fork_attention`: key/value states then stay completion-only.
    """

    prefix_key: jax.Array
    prefix_value: jax.Array
    prefix_segment_ids: jax.Array
    num_generations: int = field(pytree_node=False, default=1)
    deferred_attention: bool = field(pytree_node=False, default=False)

    def grouped_fork_attention(
        self,
        *,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask_info: MaskInfo | None,
        softmax_scale: float | None,
        runtime_dtype: jnp.dtype,
        softmax_dtype: jnp.dtype | None,
        precision: jax.lax.PrecisionLike = None,
    ) -> jax.Array:
        """Fork attention without a per-completion copy of the prefix.

        The queries of all ``G`` completions of one prompt attend that prompt's prefix in one matmul
        (only prefix padding is masked: every prefix position precedes every completion position), and
        each completion attends causally to its own keys. Both parts share one row maximum and one
        softmax denominator, which is the exact log-sum-exp merge of the two partial softmaxes; prefix
        key/value gradients accumulate over the group inside the matmul.

        Args:
            query: Completion queries ``[prompt_batch * G, completion_len, q_heads, head_dim]``.
            key: Completion keys ``[prompt_batch * G, completion_len, kv_heads, head_dim]``.
            value: Completion values ``[prompt_batch * G, completion_len, kv_heads, v_head_dim]``.
            mask_info: Mask whose query segment ids mark valid completion tokens (``-1`` = padding).
            softmax_scale: Logit scale; ``1 / sqrt(head_dim)`` when ``None``.
            runtime_dtype: Dtype of the value products and of the result.
            softmax_dtype: Logit/softmax dtype; promoted to at least float32.
            precision: Matmul precision; ``None`` inherits the ambient setting.

        Returns:
            ``[prompt_batch * G, completion_len, q_heads, v_head_dim]`` attention outputs in ``runtime_dtype``;
            rows of padded completion queries are zero.

        Raises:
            ValueError: When the batch is not ``prompt_batch * G``, the query heads are not a multiple of the
                key heads, or ``mask_info`` carries no query segment ids.
        """
        fork_batch, completion_len, num_q_heads, head_dim = query.shape
        prompt_batch = self.prefix_segment_ids.shape[0]
        groups = self.num_generations
        num_kv_heads, v_head_dim = key.shape[2], value.shape[-1]
        if fork_batch != prompt_batch * groups:
            raise ValueError(
                f"Fork batch {fork_batch} must equal prompt batch {prompt_batch} x num_generations {groups}."
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"q heads ({num_q_heads}) must be a multiple of kv heads ({num_kv_heads}).")
        query_valid = _completion_segment_ids(mask_info) >= 0
        reps = num_q_heads // num_kv_heads
        compute_dtype = jnp.promote_types(softmax_dtype or runtime_dtype, jnp.float32)
        scale = jnp.asarray(softmax_scale if softmax_scale is not None else head_dim**-0.5, compute_dtype)
        # Query head ``h * reps + r`` reads KV head ``h``, as in the GQA head repeat of the dense operations.
        q = query.astype(compute_dtype).reshape(prompt_batch, groups, completion_len, num_kv_heads, reps, head_dim)
        own_key = key.astype(compute_dtype).reshape(prompt_batch, groups, completion_len, num_kv_heads, head_dim)
        own_value = value.astype(runtime_dtype).reshape(prompt_batch, groups, completion_len, num_kv_heads, v_head_dim)
        prefix_key = self.prefix_key.astype(compute_dtype)
        prefix_value = self.prefix_value.astype(runtime_dtype)
        prefix_valid = (jnp.asarray(self.prefix_segment_ids) >= 0)[:, None, None, None, None, :]
        own_key_valid = query_valid.reshape(prompt_batch, groups, completion_len)[:, None, None, :, None, :]
        blocks = []
        # Causal query blocks: block ``[start, stop)`` sees own keys ``[0, stop)`` only, so the blocks above
        # the diagonal are never computed.
        for start in range(0, completion_len, GROUPED_FORK_QUERY_BLOCK):
            stop = min(start + GROUPED_FORK_QUERY_BLOCK, completion_len)
            q_block = q[:, :, start:stop]
            prefix_logits = scale * jnp.einsum(
                "bgchrd,bphd->bhrgcp", q_block, prefix_key, preferred_element_type=compute_dtype, precision=precision
            )
            own_logits = scale * jnp.einsum(
                "bgchrd,bgkhd->bhrgck",
                q_block,
                own_key[:, :, :stop],
                preferred_element_type=compute_dtype,
                precision=precision,
            )
            causal = jnp.arange(start, stop)[:, None] >= jnp.arange(stop)[None, :]
            prefix_logits = jnp.where(prefix_valid, prefix_logits, -jnp.inf)
            own_logits = jnp.where(causal & own_key_valid[..., :stop], own_logits, -jnp.inf)
            row_max = jnp.maximum(
                jnp.max(prefix_logits, axis=-1, keepdims=True), jnp.max(own_logits, axis=-1, keepdims=True)
            )
            row_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(row_max), row_max, 0.0))
            prefix_exp = jnp.exp(prefix_logits - row_max)
            own_exp = jnp.exp(own_logits - row_max)
            denominator = jnp.sum(prefix_exp, axis=-1, keepdims=True) + jnp.sum(own_exp, axis=-1, keepdims=True)
            inverse = 1.0 / jnp.where(denominator > 0, denominator, 1.0)
            blocks.append(
                jnp.einsum(
                    "bhrgcp,bphd->bgchrd",
                    (prefix_exp * inverse).astype(runtime_dtype),
                    prefix_value,
                    preferred_element_type=runtime_dtype,
                    precision=precision,
                )
                + jnp.einsum(
                    "bhrgck,bgkhd->bgchrd",
                    (own_exp * inverse).astype(runtime_dtype),
                    own_value[:, :, :stop],
                    preferred_element_type=runtime_dtype,
                    precision=precision,
                )
            )
        outputs = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=2)
        outputs = outputs.reshape(fork_batch, completion_len, num_q_heads, v_head_dim)
        return jnp.where(query_valid[:, :, None, None], outputs, 0).astype(runtime_dtype)

    def concatenate_to_cache(
        self, *, key: jax.Array, value: jax.Array, mask_info: MaskInfo | None
    ) -> tuple[jax.Array, jax.Array, MaskInfo, PrefixForkView]:
        """Concatenate the gathered prefix before the completion states and build the fork mask.

        Args:
            key: Completion keys ``[prompt_batch * G, completion_len, kv_heads, head_dim]``.
            value: Completion values, same leading shape as ``key``.
            mask_info: Mask whose query segment ids mark valid completion tokens (``-1`` = padding).

        Returns:
            ``(key, value, mask_info, self)`` where key/value have length ``prefix_len + completion_len``
            and ``mask_info`` carries query/key segment ids plus absolute query/key positions.
        """
        fork_batch, completion_len = key.shape[0], key.shape[1]
        prompt_batch, prefix_len = self.prefix_segment_ids.shape
        if fork_batch != prompt_batch * self.num_generations:
            raise ValueError(
                f"Fork batch {fork_batch} must equal prompt batch {prompt_batch} x num_generations "
                f"{self.num_generations}."
            )
        q_segment_ids = _completion_segment_ids(mask_info)
        rows = jnp.arange(fork_batch, dtype=jnp.int32) // self.num_generations
        full_key = jnp.concatenate([jnp.take(self.prefix_key, rows, axis=0).astype(key.dtype), key], axis=1)
        full_value = jnp.concatenate([jnp.take(self.prefix_value, rows, axis=0).astype(value.dtype), value], axis=1)
        prefix_segment_ids = jnp.take(jnp.asarray(self.prefix_segment_ids, jnp.int32), rows, axis=0)
        kv_segment_ids = jnp.concatenate([prefix_segment_ids, q_segment_ids], axis=1)
        q_positions = jnp.broadcast_to(prefix_len + jnp.arange(completion_len, dtype=jnp.int32), q_segment_ids.shape)
        kv_positions = jnp.broadcast_to(jnp.arange(prefix_len + completion_len, dtype=jnp.int32), kv_segment_ids.shape)
        fork_mask = MaskInfo.from_segments(
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )
        return full_key, full_value, fork_mask, self


def _completion_segment_ids(mask_info: MaskInfo | None) -> jax.Array:
    """``[fork_batch, completion_len]`` int32 completion segment ids: ``0`` valid, ``-1`` padding."""
    if mask_info is None or mask_info.q_segment_ids is None:
        raise ValueError("PrefixForkView requires a MaskInfo carrying completion query segment ids.")
    q_segment_ids = jnp.asarray(mask_info.q_segment_ids, jnp.int32)
    if q_segment_ids.ndim == 3:
        q_segment_ids = q_segment_ids[:, 0, :]
    return jnp.where(q_segment_ids >= 0, 0, -1).astype(jnp.int32)


def is_shared_prefix_view(view: object) -> bool:
    """Whether ``view`` is a prefix-capture or prefix-fork view."""
    return isinstance(view, (PrefixCaptureView, PrefixForkView))
