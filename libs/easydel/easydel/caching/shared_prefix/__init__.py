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

Visibility in the fork pass is expressed with segment ids (``-1`` marks
padding) and causality with absolute column positions
(``q_positions = P + arange(C)``, ``kv_positions = arange(P + C)``), which the
dense attention operations consume directly instead of a dense mask.
"""

from __future__ import annotations

import jax
from eformer.pytree import auto_pytree, field
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp

__all__ = ("PREFIX_SHARED_ATTENTION_MECHANISMS", "PrefixCaptureView", "PrefixForkView", "is_shared_prefix_view")

PREFIX_SHARED_ATTENTION_MECHANISMS = frozenset({"vanilla", "sdpa"})
"""Attention operations verified on TPU (output and q/k/v gradients) for the prefix-shared fork representation."""


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
    """

    prefix_key: jax.Array
    prefix_value: jax.Array
    prefix_segment_ids: jax.Array
    num_generations: int = field(pytree_node=False, default=1)

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
        if mask_info is None or mask_info.q_segment_ids is None:
            raise ValueError("PrefixForkView requires a MaskInfo carrying completion query segment ids.")
        rows = jnp.arange(fork_batch, dtype=jnp.int32) // self.num_generations
        full_key = jnp.concatenate([jnp.take(self.prefix_key, rows, axis=0).astype(key.dtype), key], axis=1)
        full_value = jnp.concatenate([jnp.take(self.prefix_value, rows, axis=0).astype(value.dtype), value], axis=1)
        q_segment_ids = jnp.asarray(mask_info.q_segment_ids, jnp.int32)
        if q_segment_ids.ndim == 3:
            q_segment_ids = q_segment_ids[:, 0, :]
        q_segment_ids = jnp.where(q_segment_ids >= 0, 0, -1).astype(jnp.int32)
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


def is_shared_prefix_view(view: object) -> bool:
    """Whether ``view`` is a prefix-capture or prefix-fork view."""
    return isinstance(view, (PrefixCaptureView, PrefixForkView))
