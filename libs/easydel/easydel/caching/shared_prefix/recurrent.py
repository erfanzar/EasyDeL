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

"""Prefix-shared views for gated-delta linear-attention layers.

The capture pass runs the left-padded prompts once (batch ``B``) and records each
linear-attention layer's end-of-prompt state: the causal-conv window (the trailing
``d_conv`` projected inputs) and the delta-rule recurrent state. The fork pass runs
the completions (batch ``B * G``) starting from the state of prompt ``r // G``.

Padding rows are zeroed before the projections, so left padding contributes nothing
to either state, and the prompt's last column is always valid, so the captured state
is the state after the last prompt token. Both views are pytrees of plain arrays and
the fork view gathers with ``jnp.take``, so the fork pass's gradients sum back over the
``G`` rows of each group into the capture pass.
"""

from __future__ import annotations

import dataclasses

import jax
from eformer.pytree import auto_pytree
from jax import numpy as jnp

__all__ = (
    "PREFIX_SHARED_RECURRENT_MODEL_TYPES",
    "PrefixRecurrentCaptureView",
    "PrefixRecurrentForkView",
    "is_shared_prefix_recurrent_view",
)

PREFIX_SHARED_RECURRENT_MODEL_TYPES = frozenset({"qwen3_next"})
"""Model types whose ``linear_attention`` (gated-delta) layers were verified on TPU with the recurrent views."""


@auto_pytree
class PrefixRecurrentCaptureView:
    """Records one linear-attention layer's state at the end of the prompt.

    The layer reads ``conv_state``/``recurrent_state`` (both ``None``, so it starts from zero state) and
    stores its end-of-prompt state through ``replace``, which returns a populated view.

    Attributes:
        conv_state: ``[batch, conv_dim, d_conv]`` trailing projected inputs; ``None`` before capture.
        recurrent_state: ``[batch, value_heads, key_head_dim, value_head_dim]`` delta-rule state;
            ``None`` before capture.
    """

    conv_state: jax.Array | None = None
    recurrent_state: jax.Array | None = None


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PrefixRecurrentForkView:
    """Starts every completion row from the captured state of its prompt.

    Row ``r`` of the fork batch reads prompt row ``r // num_generations``.

    Attributes:
        prefix_conv_state: Captured conv windows ``[prompt_batch, conv_dim, d_conv]``.
        prefix_recurrent_state: Captured recurrent states ``[prompt_batch, value_heads, key_dim, value_dim]``.
        num_generations: Completions per prompt (``G``).
    """

    prefix_conv_state: jax.Array
    prefix_recurrent_state: jax.Array
    num_generations: int = 1

    def tree_flatten(self):
        """Flatten into the two state arrays, with ``num_generations`` as static auxiliary data."""
        return (self.prefix_conv_state, self.prefix_recurrent_state), self.num_generations

    @classmethod
    def tree_unflatten(cls, num_generations, children):
        """Rebuild the view from :meth:`tree_flatten` output."""
        return cls(*children, num_generations=num_generations)

    def _gather(self, state: jax.Array) -> jax.Array:
        rows = jnp.arange(state.shape[0] * self.num_generations, dtype=jnp.int32) // self.num_generations
        return jnp.take(state, rows, axis=0)

    @property
    def conv_state(self) -> jax.Array:
        """``[prompt_batch * G, conv_dim, d_conv]`` conv window of each row's prompt."""
        return self._gather(self.prefix_conv_state)

    @property
    def recurrent_state(self) -> jax.Array:
        """``[prompt_batch * G, value_heads, key_dim, value_dim]`` recurrent state of each row's prompt."""
        return self._gather(self.prefix_recurrent_state)

    def replace(self, **kwargs) -> PrefixRecurrentForkView:
        """Return ``self`` unchanged: the fork pass only scores, so end-of-completion states are dropped.

        Args:
            **kwargs: Updated ``conv_state``/``recurrent_state`` from the layer; ignored.

        Returns:
            This view.
        """
        del kwargs
        return self


def is_shared_prefix_recurrent_view(view: object) -> bool:
    """Whether ``view`` is a prefix-capture or prefix-fork view of a linear-attention layer."""
    return isinstance(view, (PrefixRecurrentCaptureView, PrefixRecurrentForkView))
