import typing

import jax.lax
from flax import linen as nn
from transformers import FlaxPreTrainedModel
from EasyDel.modules import FlaxLlamaForCausalLM
from typing import Type

from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from ...modules.auto_models import AutoEasyDelModelForCausalLM
from .modelling_base import FlaxPreTrainedModelWrapper
from jax import numpy as jnp
import chex
import flax
from flax.traverse_util import unflatten_dict, flatten_dict


class ValueHead(nn.Module):
    config: typing.Any
    summary_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[jax.lax.Precision] = jax.lax.Precision(
        "fastest")

    def setup(self):
        """
        The setup function is called by the model's constructor.
        It initializes all the layers in your model, and assigns them to member variables.
        The setup function should be used for any initialization that needs to happen before running forward().
        This includes things like loading weights from a file, or setting up an optimizer.

        :param self: Represent the instance of the class
        :return: A tuple of the following:

        """
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
        """
        The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, e.g., x(arg).
        The __call__ method enables instances of a class to be called like standard Python functions.

        :param self: Represent the instance of the class
        :param hidden_states: chex.Array: Pass the hidden states of the previous layer
        :param deterministic: bool: Determine whether to use dropout
        :return: A tensor of shape (batch_size, num_classes)

        """
        output = self.dropout(hidden_states, deterministic=deterministic)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class FlaxAutoModelForCausalLMWithValueHead(FlaxPreTrainedModelWrapper,flax.linen.Module):
    pretrained_model: Type[FlaxPreTrainedModel] = FlaxLlamaForCausalLM
    transformers_parent_class: Type[FlaxPreTrainedModel] = AutoEasyDelModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )
    summary_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[jax.lax.Precision] = jax.lax.Precision(
        "fastest")

    def setup(self):
        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError(
                "The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(
            self.pretrained_model.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

    def __call__(

            self,
            module_params: dict,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
    ):
        """
        The function takes in module parameters, input IDs, past key values, and attention mask, and
        returns language model logits and a value.

        :param module_params: The `module_params` parameter is a dictionary that contains the parameters
        for the module. It is used to pass any additional parameters that are specific to the module
        being called
        :type module_params: dict
        :param input_ids: The `input_ids` parameter is used to specify the input sequence of token IDs
        for the model. These token IDs represent the input text that the model will process
        :param past_key_values: The `past_key_values` parameter is used for autoregressive decoding. It
        allows you to pass the previous key-value pairs from the attention mechanism to the model, which
        can be used to generate the next token in the sequence. This is useful when generating text one
        token at a time
        :param attention_mask: The `attention_mask` parameter is used to mask certain tokens in the
        input sequence. It is a binary tensor of shape `(batch_size, sequence_length)` where each
        element is either 0 or 1. A value of 0 indicates that the corresponding token should be masked,
        while a value of
        :return: two values: `lm_logits` and `value`.
        """

        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_model_output: FlaxCausalLMOutput = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            params=module_params,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits

        value = self.v_head(last_hidden_state)

        return lm_logits, value

    def generate(self, *args, **kwargs):
        raise NotImplementedError()

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError()

    def post_init(self, params: dict, config) -> dict:
        self.setup()
        has_v_head = True in set(
            [key for key, _ in flatten_dict(params).items()]
        )
        if not has_v_head:
            vh_params = self.v_head.init(
                jnp.ones((1, config.vocab_size))
            )
            print(
                vh_params
            )
            
        return params
