import copy
import math

import flax
import jax

from jax import numpy as jnp
from jax import lax, random
from flax import linen as nn

import functools
import collections

from einops import rearrange, repeat, reduce
from typing import Union, Optional, OrderedDict, NamedTuple, Callable, Any

from .utils import log, log_prob, shift, masked_mean, AVAILABLE_MODELS_FOR_RLHF


class PPOActionCriticReturn(NamedTuple):
    actions: Union[Any, jnp.ndarray]
    sequence: Union[Any, jnp.ndarray]
    attention_mask: Union[Any, jnp.ndarray]
    prompt_mask: Union[Any, jnp.ndarray]
    action_logit: Union[Any, jnp.ndarray]
    values: Union[Any, jnp.ndarray]


class ActorCritic(nn.Module):
    model: AVAILABLE_MODELS_FOR_RLHF
    critic_model: Optional[AVAILABLE_MODELS_FOR_RLHF]
    pooled_values: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[None, lax.Precision]] = lax.Precision('fastest')

    def setup(self) -> None:
        if self.critic_model is None:
            self.critic_model = self.model
        self.head = nn.Sequential(
            nn.Dense(
                1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                init_kernel=nn.initializers.orthogonal(math.sqrt(2)),
                init_bias=nn.initializers.zeros
            ),
            lambda x: rearrange(x, '... 1 -> ...')
        )

    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: Optional[Union[jnp.ndarray, None]] = None,
                 return_values: Optional[bool] = False,
                 **extra_model_inputs
                 ):
        _ = extra_model_inputs.pop('return_dict', None)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **extra_model_inputs
        ).logits
        if not return_values:
            return logits, None
        critic_embeddings = self.critic_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if self.pooled_values:
            critic_embeddings = shift(critic_embeddings, shift=1, axis=-2)
            critic_embeddings = masked_mean(critic_embeddings, attention_mask, axis=1)

        values = self.value_head(critic_embeddings)

        return logits, values


@functools.partial(jax.jit, static_argnums=0)
def policy_action(
        apply_fn: Callable[..., Any],
        params: flax.core.frozen_dict.FrozenDict,
        state: jnp.ndarray,
):
    out = apply_fn({'params': params}, state)
    return out
