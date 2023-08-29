import typing

import flax.core
import jax
from jax import lax
from flax import struct
from pathlib import Path
import tqdm
from jax import numpy as jnp
from flax import linen as nn
from EasyDel import FlaxMptForCausalLM, FlaxLlamaForCausalLM, MptConfig, LlamaConfig

AVAILABLE_MODELS_FOR_RLHF = typing.Union[
    FlaxLlamaForCausalLM, FlaxMptForCausalLM
]
AVAILABLE_MODELS_CONFIG_FOR_RLHF = typing.Union[
    LlamaConfig, MptConfig
]


class RewardModel(nn.Module):
    model: AVAILABLE_MODELS_FOR_RLHF
    params: flax.core.FrozenDict
    config: AVAILABLE_MODELS_CONFIG_FOR_RLHF
    dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, lax.Precision]] = lax.Precision('fastest')

    def setup(self) -> None:
        self.prompt_embedding = self.param(
            'prompt_embedding',
            nn.initializers.zeros,
            (1, 1, self.config.hidden_size),
            self.dtype
        )
        self.response_embedding = self.param(
            'response_embedding',
            nn.initializers.zeros,
            (1, 1, self.config.hidden_size),
            self.dtype
        )

    def __call__(self, *args, **kwargs):
        ...
