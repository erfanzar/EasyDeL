import copy
import math

import einops
import flax
import jax
import transformers

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

    def generate(
            self,
            params: Union[flax.core.FrozenDict, dict],
            input_ids,
            attention_mask,
            max_sequence_length: int,
            eos_token_id: int = None,
            return_values: bool = False,
            **kwargs
    ):
        b, s = input_ids.shape
        input_ids = self.model.generate(
            params=params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=transformers.GenerationConfig(
                max_new_tokens=max_sequence_length
            ),

        ).sequences
        prompt_mask = einops.repeat(jnp.arange(input_ids.shape[-1]) < s, 's -> b s', b=b)
        action_mask = ~prompt_mask

        if eos_token_id is not None:
            attention_mask = (jnp.cumsum(input_ids == eos_token_id, axis=-1) == 0)
            # mask = F.pad(mask, (1, -1), value=True)  # include eos token
            action_mask &= attention_mask
            attention_mask = action_mask & action_mask

        action_logits, value = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_values=return_values,

        )
        return PPOActionCriticReturn(
            input_ids[:, s:],
            input_ids,
            attention_mask,
            prompt_mask,
            action_logits,
            value
        )


@functools.partial(jax.jit, static_argnums=0)
def policy_action(
        apply_fn: Callable[..., Any],
        params: flax.core.frozen_dict.FrozenDict,
        state: jnp.ndarray,
):
    out = apply_fn({'params': params}, state)
    return out
