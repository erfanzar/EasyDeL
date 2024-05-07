import functools
import math
import typing

import fjformer
import flax
from flax.struct import dataclass
from jax import numpy as jnp, lax
from jax.sharding import PartitionSpec
import jax
from flax import linen as nn
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.core import freeze, unfreeze, FrozenDict
from typing import Union, Optional, Tuple
from flax.linen import partitioning as nn_partitioning, combine_masks
from transformers.modeling_flax_outputs import FlaxMaskedLMOutput
from fjformer.func import auxiliary_load_balancing_loss_func
from ..attention_module import AttentionModule
from fjformer.linen import Linear
from ..flax_modelling_utils import (
    ACT2FN,
    with_sharding_constraint,
    repeat_kv_bnsh,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    get_gradient_checkpoint_policy,
    block_wise_ffn
)
import chex
from .deepseek_configuration import DeepseekV2Config
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel

re_mat = nn_partitioning.remat


@flax.struct.dataclass
class MoeModelOutput:
    last_hidden_state: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    router_logits: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
    aux_loss: Optional[chex.Array] = None
    router_logits: Optional[Tuple[chex.Array]] = None


class DeepseekV2RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = fjformer.linen.linen.control_quantization(self.weight, self.dtype)
        return output * weight


def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
    )


def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1., mscale=1.):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min) / (max - min)
    return jnp.clip(linear_func, 0, 1)


def init_deepseek_rotary_embedding(
        dim,
        max_position_embeddings=2048,
        base=10000,
        method: typing.Literal["linear", "yarn", "dynamic", None] = None,
        kwargs: typing.Optional[dict] = None,
):
    if method is None:
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "linear":
        assert kwargs is not None
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get("scaling_factor")
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "dynamic":
        assert kwargs is not None
        targeted_len = kwargs.get("targeted_len", max_position_embeddings)
        if targeted_len > max_position_embeddings:
            base = base * (
                    (kwargs.get("scaling_factor") * targeted_len / max_position_embeddings)
                    - (kwargs.get("scaling_factor") - 1)
            ) ** (dim / (dim - 2))
            inv_freq = 1.0 / (
                    base ** (jnp.arange(0, dim, 2).astype("float32") / dim)
            )

        else:
            inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get("scaling_factor")

        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "yarn":

        scaling_factor = kwargs.get("scaling_factor", 1.0)
        original_max_position_embeddings = kwargs.get("original_max_position_embeddings", 4096)
        beta_fast = kwargs.get("beta_fast", 32)
        beta_slow = kwargs.get("beta_slow", 1)
        mscale = kwargs.get("mscale", 1)
        mscale_all_dim = kwargs.get("mscale_all_dim", 0)
        freq_extra = 1.0 / (
                base
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        freq_inter = 1.0 / (
                scaling_factor
                * base
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).astype("float32")
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)

        freqs = jnp.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
        )

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return (jnp.sin(emb) * _mscale).astype("float32"), (jnp.cos(emb) * _mscale).astype("float32")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = jnp.expand_dims(cos[position_ids], unsqueeze_dim)
    sin = jnp.expand_dims(sin[position_ids], unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxDeepseekV2MLP(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None

    def setup(self) -> None:
        dense = functools.partial(
            Linear,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.gate_proj = dense(self.intermediate_size or self.config.intermediate_size)
        self.up_proj = dense(self.intermediate_size or self.config.intermediate_size)
        self.down_proj = dense(self.hidden_size or self.config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
            self,
            x: chex.Array,
            e: bool = False  # Ignored
    ):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV2PreTrainedModel(EasyDelFlaxPretrainedModel):
    config_class: DeepseekV2Config = DeepseekV2Config
    module_class: nn.Module = None
    base_model_prefix = "model"

    # main_input_name = "input_ids"

    def __init__(
            self,
            config: DeepseekV2Config,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            input_shape: Tuple[int, int] = (1, 1),
            seed: int = 0,
            _do_init: bool = False,
            **kwargs
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs
        )

        super().__init__(
            dtype=dtype, _do_init=_do_init,
            module=module, config=config, input_shape=input_shape,
            seed=seed,
        )

    def init_weights(
            self,
            rng: jax.random.PRNGKey,
            input_shape: Tuple,
            params: FrozenDict = None
    ) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.
        It takes in a rng, which is a random number generator key that can be used to generate random numbers.
        The input_shape parameter specifies the shape of the inputs that will be fed into this model.
        The params parameter allows you to pass in pre-trained weights for your model, if you have them available.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Initialize the input_ids, attention_mask and position_ids
        :param params: flax.core.FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters
        """

        self.config.initialization_of_moe = True
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1], dtype="i4"),
            input_shape,
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(
                input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=False
            )
        random_params = module_init_outputs["params"]

        self.config.initialization_of_moe = False
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        """
        The __call__ function is the main function of a JAX module.
        It takes as input:
        - The parameters of the model (self.params)
        - The inputs to the model (input_ids, attention_mask, position_ids)
        - Whether we are training (train=True/False) and whether we want to return all hidden states and
        attentions weights at each layer in addition to just the last layer output (output_hidden_states=True/False).

        :param self: Represent the instance of the class
        :param input_ids: Pass the input sequence to the model
        :param attention_mask: Mask out the padding tokens
        :param position_ids: Specify the position of each token in the sequence
        :param params: dict: Pass in the parameters of the model
        :param past_key_values: dict: Pass the past key values to the model
        :param dropout_rng: jax.random.PRNGKey: Pass in a random number generator key to the model
        :param train: bool: Determine whether to use dropout or not
        :param output_attentions: Optional[bool]: Determine whether to return the attention weights
        :param output_hidden_states: Optional[bool]: Determine whether to return the hidden states of all layers
        :param return_dict: Optional[bool]: Return a dictionary of the outputs
        :param add_params_field: bool: Add a params field to the inputs dictionary
        :return: A tuple of (last_hidden_state, past_key_values)

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[
                                            None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rng_s = {}
        if dropout_rng is not None:
            rng_s["dropout"] = dropout_rng

        inputs = {
            "params": params or self.params} if add_params_field else params or self.params

        if self.config.bits is not None:
            rng_s['params'] = jax.random.key(0)
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),  # input_ids: chex.Array
            # attention_mask: Optional[chex.Array] = None
            jnp.array(attention_mask, dtype="i4"),
            # position_ids: Optional[chex.Array] = None
            jnp.array(position_ids, dtype="i4"),
            None,  # inputs_embeds: Optional[chex.Array] = None
            output_attentions,  # output_attentions: Optional[bool] = None
            # output_hidden_states: Optional[bool] = None
            output_hidden_states,
            # output_router_logits: Optional[bool] = None
            output_router_logits,
            False,  # init_cache: bool = False
            not train,  # deterministic: bool = True
            return_dict,  # return_dict: bool = True
            rngs=rng_s,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + \
                      (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxDeepseekV2ForCausalLMModule(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.model = FlaxDeepseekV2Module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = Linear(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            return_dict: bool = True,
    ) -> MoeCausalLMOutput | Tuple:

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            init_cache=init_cache,
            deterministic=deterministic,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        batch_size, seq_length, hd = logits.shape
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=tuple(  # type:ignore
                    [logit.reshape(batch_size * seq_length, -1) for logit in outputs.router_logits]  # type:ignore
                ),
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef
        if not return_dict:
            outputs = (logits,) + tuple(
                v
                for v in [
                    aux_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits
                ]
                if v is not None
            )
            return outputs

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class FlaxDeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    module_class = FlaxDeepseekV2ForCausalLMModule

    def set_input_embeddings(self, value):
        self.module.model.embed_tokens = value

    def get_input_embeddings(self):
        return self.module.model.embed_tokens

    def set_decoder(self, decoder):
        self.module.model = decoder

    def get_decoder(self):
        return self.module.model

    def get_output_embeddings(self):
        return self.module.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        """
        The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

        :param self: Access variables that belong to the class
        :param input_ids: Pass in the input tokens
        :param max_length: Set the length of the sequence to be generated
        :param attention_mask: Optional[chex.Array]: Mask the attention weights
        :return: A dictionary of the past_key_values, attention_mask and position ids

        """
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones(
            (batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[
                                            None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
