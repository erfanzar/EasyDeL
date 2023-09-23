import math
import typing

import einops

import jax
from jax import lax

from jax import numpy as jnp
from flax import linen as nn
from .utils import AVAILABLE_MODELS_FOR_RLHF, AVAILABLE_MODELS_CONFIG_FOR_RLHF


class RewardModel(nn.Module):
    model: AVAILABLE_MODELS_FOR_RLHF
    config: AVAILABLE_MODELS_CONFIG_FOR_RLHF
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, lax.Precision]] = lax.Precision('fastest')
    num_outputs: int = 0
    use_output_bias: bool = False
    use_extra_embeddings: bool = True

    def setup(self) -> None:
        if self.use_extra_embeddings:
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
        self.binned_output = self.num_outputs > 1
        self.wo = nn.Dense(
            1 if self.binned_output else self.num_outputs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.use_output_bias,
            init_kernel=nn.initializers.orthogonal(math.sqrt(2)),
            init_bias=nn.initializers.zeros
        )

    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: typing.Optional[typing.Union[jnp.ndarray, None]] = None,
                 prompt_mask: typing.Optional[typing.Union[jnp.ndarray, None]] = None,
                 prompt_length: typing.Optional[typing.Union[jnp.ndarray, None]] = None,
                 sample: bool = False,
                 sample_temperature=1.,
                 **extra_model_inputs
                 ):
        batch, seq = input_ids.shape
        assert not (
                prompt_mask is not None and prompt_length is not None), ('Prompt Mask and Prompt Length can not '
                                                                         'be None')

        if prompt_length is not None:
            prompt_mask = einops.repeat(jnp.arange(seq), 'n -> b n', b=batch) > einops.rearrange(
                prompt_mask, 'b -> b 1'
            )
        extra_embeddings = None
        if self.use_extra_embeddings:
            if prompt_mask is not None:
                extra_embeddings = jax.lax.select(
                    einops.rearrange(prompt_mask, 'b n -> b n 1'),
                    self.prompt_embedding,
                    self.response_embedding
                )
        _ = extra_model_inputs.pop('return_dict', None)
        prediction = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            extra_embedding=extra_embeddings,
            **extra_model_inputs
        )

        if attention_mask is None:
            pool = jnp.mean(prediction, axis=1, keepdims=False)

        else:
            if prediction.ndim == 3:
                attention_mask = einops.rearrange(attention_mask, 'b n -> b n 1')

            masked_seq = jax.lax.select(
                attention_mask,
                prediction,
                0.

            )
            numer = jnp.sum(masked_seq, axis=1, keepdims=False)
            denom = jnp.sum(attention_mask, axis=1, keepdim=False)

            pool = numer / jnp.clip(denom, a_min=1e-3)
            pool = jax.lax.select(
                denom == 0,
                0.,
                pool
            )

        pred = self.wo(pool)
        if not self.binned_output:
            pred = einops.rearrange(pred, '... 1 -> ...')
        if sample and self.binned_output:
            pred = ((pred / max(sample_temperature, 1e-10)) + -jnp.log(-jnp.log(jnp.zeros_like(pred)))).argmax(-1)

        return pred
