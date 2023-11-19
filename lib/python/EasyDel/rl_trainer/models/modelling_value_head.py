import typing

import jax.lax
from flax import linen as nn
from overrides import overrides
from transformers import FlaxPreTrainedModel
from EasyDel.modules import FlaxLlamaForCausalLM
from typing import Type

from transformers.modeling_flax_outputs import FlaxCausalLMOutput

from .modelling_base import FlaxPreTrainedModelWrapper
from jax import numpy as jnp
import chex


class ValueHead(nn.Module):
    config: typing.Any
    summary_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    def setup(self):
        config = self.config

        self.dropout = nn.Dropout(self.summary_dropout_prob)

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(self, hidden_states: chex.Array, deterministic: bool = True):
        output = self.dropout(hidden_states, deterministic=deterministic)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class FlaxAutoModelForCausalLMWithValueHead(FlaxPreTrainedModelWrapper):
    pretrained_model: Type[FlaxPreTrainedModel] = FlaxLlamaForCausalLM
    transformers_parent_class: Type[FlaxPreTrainedModel] = FlaxPreTrainedModel
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def setup(self):
        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config)

    def __call__(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_model_output: FlaxCausalLMOutput = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits

        value = self.v_head(last_hidden_state).squeeze(-1)

        return lm_logits, value

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)

        return self.pretrained_model.push_to_hub(*args, **kwargs)
