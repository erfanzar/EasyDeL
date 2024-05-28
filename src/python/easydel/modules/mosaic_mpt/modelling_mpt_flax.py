import math
from flax.core import FrozenDict
from typing import Optional, Union, Tuple
from flax.linen import combine_masks
from jax import numpy as jnp, lax
import jax
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
import flax
from einops import rearrange
from flax.linen.partitioning import remat
from ..flax_modelling_utils import (
    get_gradient_checkpoint_policy,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
)
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
import chex
from fjformer.linen import Dense
from fjformer import linen as nn
from .mosaic_configuration import MptConfig
from ..attention_module import AttentionModule


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param("kernel", nn.initializers.ones, (self.dim,), self.param_dtype, )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.bfloat16))
        output = self._norm(x).astype(self.dtype)
        weight = nn.linen.control_quantization(self.weight, self.dtype)
        return output * weight


class FlaxMptMLP(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.up_proj = Dense(
            self.config.expansion_ratio * self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.down_proj = Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.hidden_dropout = nn.Dropout(self.config.attn_config.attn_pdrop)

    def __call__(
            self, hidden_states: chex.Array, residual: chex.Array, deterministic: bool = False
    ):
        return self.hidden_dropout(
            self.down_proj(
                jax.nn.gelu(
                    self.up_proj(
                        hidden_states
                    ), approximate=False
                )
            ),
            deterministic=deterministic
        ) + residual


class FlaxMptAttention(BaseJAXAttentionModule):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:

        self.Wqkv = Dense(
            self.config.hidden_size * 3,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.use_bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision)
        self.out_proj = Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.dropout = nn.Dropout(self.config.attn_config.attn_pdrop)

        self.hidden_size = self.config.hidden_size
        self.n_heads = self.config.n_heads
        self.max_seq_length = self.config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = self.config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attention_performer = AttentionModule(
            attention_dropout=self.config.attn_config.attn_pdrop,
            num_attention_heads=self.config.num_attention_heads,
            head_dims=self.head_dim,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            mesh=self.config.jax_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            base_module_class=self.config
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_bias: chex.Array,
            causal_mask: chex.Array,
            init_cache: bool = False,
            deterministic: bool = False
    ):

        """The __call__ function is the main function of a JAX module.
        It takes in inputs and returns outputs, just like any other Python function.
        The difference is that __call__ can also take in state (e.g., parameters) from the module itself,
        and it can update that state as part of its computation.

        Args:
            self: Access variables that belong to the class
            hidden_states: chex.Array: Pass the input to the attention
                layer
            attention_mask: chex.Array: Mask out certain positions in
                the sequence
            position_bias: chex.Array: Add a bias to the attention
                scores
            causal_mask: chex.Array: Mask out certain positions in the
                sequence
            init_cache: bool: Initialize the cache
            deterministic: bool: deterministic to activate dropouts and
                detect training process

        Returns:
            The output of the attention layer
        """
        inp_shape = hidden_states.shape
        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

        query_states = rearrange(query_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        key_states = rearrange(key_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        value_states = rearrange(value_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(
            attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states,
                value_states,
                query_states,
                attention_mask
            )
        if position_bias is not None:
            key_length = key_states.shape[1]

            position_bias_query_index = max(0, position_bias.shape[2] - query_length)
            position_bias_key_index = max(0, position_bias.shape[3] - key_length)

            position_bias = position_bias[:, :, position_bias_query_index:, position_bias_key_index:]
        attention_mask = attention_mask.repeat(position_bias.shape[1], 1)
        attention_bias = lax.select(
            attention_mask.astype("bool"),
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype) + position_bias.astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )

        attention = self.attention_performer.__call__(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            deterministic=deterministic,
            segment_ids=None,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            bias=attention_bias,
            causal=False,
        )

        attn_output = self.out_proj(attention.attention_outputs.reshape(inp_shape))

        return attn_output, attention.attention_weights


class FlaxMptBlock(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        attn_block = FlaxMptAttention
        mlp_block = FlaxMptMLP
        if self.config.gradient_checkpointing != "":
            mlp_block = remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(2,)
            )
            attn_block = remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(3, 4, 5)
            )

        self.norm_1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.use_norm_bias
        )
        self.attn = attn_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.norm_2 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.use_norm_bias
        )
        self.ffn = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.dropout_rate = self.config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_bias: chex.Array,
            causal_mask: chex.Array,
            init_cache: bool = False,
            deterministic: bool = False,
            output_attentions: bool = False,
    ):
        attn_outputs, attn_weights = self.attn(
            self.norm_1(hidden_states),
            attention_mask,
            position_bias,
            causal_mask,
            init_cache,
            deterministic
        )
        hidden_states = self.resid_attn_dropout(attn_outputs, deterministic=deterministic) + hidden_states
        output = self.ffn(self.norm_2(hidden_states), hidden_states)
        outputs = (output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # hidden_states, attentions


class FlaxMptDecoratorCollection(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = FlaxMptBlock
        self.blocks = [
            block(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(
                self.config.n_layers
            )
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_bias: chex.Array,
            causal_mask: chex.Array,
            init_cache: bool = False,
            deterministic: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = True
    ):

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for block in self.blocks:
            output = block(
                hidden_states=hidden_states,
                deterministic=deterministic,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                output_attentions=output_attentions,
                init_cache=init_cache,
                position_bias=position_bias,
            )
            hidden_states = output[0]
            if output_attentions:
                all_attentions += (output[-1],)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states, all_attentions


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    alibi = jnp.arange(1 - sequence_length, 1, dtype="i4").reshape(1, 1, 1, sequence_length)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / jnp.pow(2, base)
    slopes = slopes.reshape(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = jnp.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], axis=1)[:, :num_heads, ...]

    alibi = alibi * slopes
    return alibi


class FlaxMptModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model)

        self.blocks = FlaxMptDecoratorCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_f = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            epsilon=self.config.layer_norm_epsilon,
            use_bias=self.config.use_norm_bias,
        )
        self.alibi = build_mpt_alibi_tensor(
            sequence_length=self.config.max_seq_len,
            num_heads=self.config.n_heads,
        )
        self.causal_mask = jnp.tril(
            jnp.ones(
                (self.config.max_seq_len, self.config.max_seq_len), dtype="bool"
            )
        ).reshape(1, 1, self.config.max_seq_len, self.config.max_seq_len)

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            input_embeds: Optional[chex.Array] = None,
            extra_embedding: Optional[chex.Array] = None,
            init_cache: bool = False,
            deterministic: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = True,
            return_dict: bool = True,
    ):
        if input_embeds is None:
            input_embeds = self.wte(input_ids)
        hidden_states = input_embeds + extra_embedding if extra_embedding is not None else input_embeds

        hidden_states, all_hidden_states, all_attentions = self.blocks(
            position_bias=self.alibi,
            causal_mask=self.causal_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            hidden_states=hidden_states,
        )
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions
            )

        return (
            hidden_states,
            all_hidden_states,
            all_attentions
        )


class FlaxMptPretrainedModel(EasyDeLFlaxPretrainedModel):
    module_class: nn.Module = None
    config_class: MptConfig = MptConfig

    def __init__(
            self,
            config,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: lax.PrecisionLike = None,
            _do_init: bool = False,
            input_shape: Tuple = (1, 16),
            **kwargs
    ):
        module = self.module_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision
        )
        super().__init__(_do_init=_do_init, config=config, input_shape=input_shape, module=module, **kwargs)

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.ones(input_shape, dtype="i4")
        if params is None:
            return self.module.init(
                rngs=rng,
                input_ids=input_ids,
                attention_mask=jnp.ones(input_shape, dtype="i4"),
                init_cache=False
            )["params"]
        else:
            return params

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            input_embeds: Optional[chex.Array] = None,
            extra_embedding: Optional[chex.Array] = None,
            init_cache: bool = False,
            deterministic: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = True,
            return_dict: bool = True,
            params: dict = None,
            add_params_field: bool = False,
            past_key_values: Optional[Tuple[Tuple[chex.Array]]] = None,
            **kwargs
    ):

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        params = {"params": params or self.params} if add_params_field else params or self.params
        input_ids = jnp.asarray(input_ids, dtype="i4")
        mutable = False
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype="i4")
        if past_key_values is not None:
            params["cache"] = past_key_values
            mutable = ["cache"]
        rngs = {}
        if self.config.bits is not None:
            rngs["params"] = jax.random.key(0)
        predict = self.module.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            extra_embedding=extra_embedding,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mutable=mutable,
            rngs=rngs
        )
        if past_key_values is not None and return_dict:
            predict, past_key_values = predict
            predict["past_key_values"] = flax.core.unfreeze(past_key_values["cache"])
            return predict
        elif past_key_values is not None and not return_dict:
            predict, past_key_values = predict
            predict = predict[:1] + (flax.core.unfreeze(past_key_values["cache"]),) + predict[1:]
        return predict


class FlaxMptModel(FlaxMptPretrainedModel):
    module_class = FlaxMptModule

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class FlaxMptForCausalLMModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.transformer = FlaxMptModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.lm_head = Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            input_embeds: Optional[chex.Array] = None,
            extra_embedding: Optional[chex.Array] = None,
            init_cache: bool = False,
            deterministic: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = True,
            return_dict: bool = True,
    ):
        predict: FlaxBaseModelOutput = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            extra_embedding=extra_embedding,
            output_hidden_states=output_hidden_states,
            init_cache=init_cache,
            output_attentions=output_attentions,
            deterministic=deterministic,
            input_embeds=input_embeds
        )
        last_hidden_state = predict.last_hidden_state

        if self.config.use_lm_head:
            shared_kernel = self.model.variables["params"]["wte"]["embedding"]
            shared_kernel = nn.linen.control_quantization(shared_kernel, self.param_dtype).T
            logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, last_hidden_state)
        else:
            logits = self.lm_head(last_hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=logits,
                hidden_states=predict.hidden_states
            )
        return logits, predict.hidden_states if output_hidden_states else (logits,)


class FlaxMptForCausalLM(FlaxMptPretrainedModel):
    module_class = FlaxMptForCausalLMModule

    def get_input_embeddings(self):
        return self.module.transformer.wte

    def get_decoder(self):
        return self.module.transformer

    def set_input_embeddings(self, value):
        self.module.transformer.wte = value

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.module.lm_head

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs
