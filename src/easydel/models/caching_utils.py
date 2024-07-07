from dataclasses import dataclass
from typing import Optional

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class KVCache:
    key_states: Optional[jax.Array] = None
    value_states: Optional[jax.Array] = None
    target_length: Optional[jax.Array] = None
    started_index: Optional[jax.Array] = None
    moved_idx: jax.Array = 0

    def tree_unflatten(self):
        return (
            self.key_states,
            self.value_states,
            self.target_length,
            self.started_index,
            self.moved_idx,
        ), {}

    def tree_flatten(cls, aux, children):
        return cls(*children)

    def update(self, key_states, value_states):
        assert self.key_states is not None
        assert self.value_states is not None
        started_index = self.started_index
        if started_index is None:
            started_index = value_states.shape[1]
            self.started_index = started_index
        self.value_states = jax.lax.dynamic_update_slice(
            self.value_states,
            value_states,
            (0, started_index + self.moved_idx, 0, 0),
        )

        self.key_states = jax.lax.dynamic_update_slice(
            self.key_states,
            key_states,
            (0, started_index + self.moved_idx, 0, 0),
        )

        self.moved_idx += value_states.shape[1]

    def get(self, attention_mask: Optional[jax.Array]):
        assert self.key_states is not None
        assert self.value_states is not None
        max_length = self.value_states.shape[1]
        if attention_mask is not None:
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < self.moved_idx - 1 + attention_mask.shape[2],
                (self.key_states.shape[1], 1, attention_mask.shape[2], max_length),
            )
            attention_mask = jnp.logical_and(attention_mask, pad_mask)
        return (self.key_states, self.value_states, attention_mask)
