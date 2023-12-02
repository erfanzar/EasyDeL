import typing

import jax.lax
from flax import linen as nn
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
        """
        The setup function is called by the model's constructor.
        It initializes all of the layers in your model, and assigns them to member variables.
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
        """
        The __call__ function is the main function of a Flax model.
        It takes in input_ids, attention_mask, and past_key_values as arguments.
        The output is a tuple containing lm logits and value.

        :param self: Represent the instance of the class
        :param input_ids: Pass the input to the model
        :param past_key_values: Pass the past key values to the model
        :param attention_mask: Mask out the padding tokens
        :param **kwargs: Pass in the past_key_values parameter
        :param : Pass the past key values to the model
        :return: The logits and the value
        
        """
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
        """
        The push_to_hub function is used to push the model to a remote location.

        :param self: Represent the instance of the class
        :param *args: Send a non-keyworded variable length argument list to the function
        :param **kwargs: Pass keyworded, variable-length argument list to a function
        :return: The pretrained model
        
        """
        setattr(self.pretrained_model, "v_head", self.v_head)

        return self.pretrained_model.push_to_hub(*args, **kwargs)
