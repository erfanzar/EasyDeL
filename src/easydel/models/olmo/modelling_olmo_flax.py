import functools
import math
from typing import Optional, Tuple, Union

import chex
import flax
import jax
import jax.numpy as jnp
from fjformer import linen as nn
from fjformer.linen import Dense
from flax.core.frozen_dict import freeze, unfreeze
from flax.linen import combine_masks
from flax.linen.partitioning import remat
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.common import LayerNormRaw

# easydel.modules
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    get_gradient_checkpoint_policy,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.olmo.olmo_configuration import OlmoConfig

re_mat = remat


class FlaxOlmoRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freqs_cis, position_ids):
        sin, cos = freqs_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxOlmoMLP(nn.Module):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        dense = functools.partial(
            Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
        )
        self.gate_proj = dense(self.config.intermediate_size)
        self.up_proj = dense(self.config.intermediate_size)
        self.down_proj = dense(self.config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x: chex.Array, e: bool = False):  # Ignored
        x = control_mlp_sharding(x, self.config.partition_axis)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FlaxOlmoAttention(BaseAttentionModule):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.k_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.v_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.o_proj = Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

        self.rotary = FlaxOlmoRotaryEmbedding(self.dtype)
        self.attention_module = FlexibleAttentionModule(
            attention_dropout=self.config.attention_dropout,
            num_attention_heads=self.config.num_attention_heads,
            head_dims=self.head_dim,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.config.attn_dtype,
            mesh=self.config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            base_config=self.config,
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(
        self, batch_size, sequence_length, query, key, value, freqs_cis, position_ids
    ):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            batch_size: Reshape the query, key and value tensors
            sequence_length: Reshape the query, key and value tensors
            query: Calculate the attention weights
            key: Calculate the attention
            value: Compute the attention weights
            freqs_cis: Calculate the frequency of each word in the
                vocabulary
            position_ids: Identify the position of each token in the
                sequence

        Returns:
            A tuple of 3 tensors: query, key and value
        """

        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary(
            position_ids=position_ids, query=query, key=key, freqs_cis=freqs_cis
        )
        return self._transpose_sequence_head(query, key, value)

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        """The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        Args:
            self: Access variables that belong to the class
            hidden_states: chex.Array: Pass the hidden states of the
                previous layer
            freqs_cis: Tuple[chex.Array, chex.Array],: Pass in the
                frequency coefficients for each position
            attention_mask: chex.Array: Mask out certain tokens in the
                input sequence
            position_ids: chex.Array: Determine the position of each
                token in a sequence
            causal_mask: chex.Array: Mask out the future tokens in the
                decoder
            deterministic: bool: Determine whether to use dropout or not
            init_cache: bool: Initialize the cache
            output_attentions: bool: Determine whether to return the
                attention weights or not
            fcm_mask: Mask out the attention weights between the input
                and output tokens
        :param : Determine if the attention is causal or not

        Returns:
            A tuple of two arrays
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        if self.config.clip_qkv is not None:
            query_states, key_states, value_states = map(
                lambda x: jnp.clip(
                    x, min=-self.config.clip_qkv, max=self.config.clip_qkv
                ),
                [query_states, key_states, value_states],
            )

        query_states = query_states.reshape(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        key_states, value_states = self.repeat_key_value(
            key_states, value_states, self.num_key_value_groups
        )
        # if self.config.use_sharding_constraint:
        #     query_states = with_sharding_constraint(
        #         query_states, PartitionSpec(("dp", "fsdp"), "sp" if query_states.shape[1] != 1 else None, "tp", None)
        #     )
        #     key_states = with_sharding_constraint(
        #         key_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
        #     )
        #     value_states = with_sharding_constraint(
        #         value_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
        #     )
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_module.__call__(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            segment_ids=segment_ids,
            causal_mask=causal_mask,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.o_proj(attn_output)

        outputs = (
            (attn_output, attentions.attention_weights)
            if output_attentions
            else (attn_output,)
        )
        return outputs


class FlaxOlmoDecoderLayer(nn.Module):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        attn_block = FlaxOlmoAttention
        mlp_block = FlaxOlmoMLP

        if self.config.gradient_checkpointing != "":
            attn_block = re_mat(
                attn_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1, 3, 4, 6, 7, 8),
            )
            mlp_block = re_mat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1,),
            )
        self.self_attn = attn_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.mlp = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.input_layernorm = LayerNormRaw(
            eps=1e-5,
        )
        self.post_attention_layernorm = LayerNormRaw(
            eps=1e-5,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        causal_mask: chex.Array,
        position_ids: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = True,
    ):
        """The __call__ function is the main function of a TransformerEncoderLayer.
        It takes in the following arguments:
            hidden_states (chex.Array): The input to the encoder layer, which is also its output after being processed
            by all sublayers.
            freqs_cis (chex.Array): A tensor containing frequency-domain representations of each token's context vector,
            used for computing self-attention weights and biases in a more efficient manner than using position
            embeddings or sinusoidal positional encoding vectors would allow for [2].

        Args:
            self: Represent the instance of the class
            hidden_states: chex.Array: Represent the input to the
                encoder layer
            freqs_cis: Tuple[chex.Array, chex.Array],: Pass the frequency
                information to the attention layer
            attention_mask: chex.Array: Mask out the attention weights
                for certain positions
            causal_mask: chex.Array: Mask the future tokens
            position_ids: chex.Array: Indicate the position of each
                token in the sequence
            deterministic: bool: Determine whether to use dropout or not
            init_cache: bool: Initialize the cache for the self-
                attention layer
            output_attentions: bool: Determine whether to return the
                attention weights or not

        Returns:
            A tuple of hidden_states and attention_output
        """

        # hidden_states: chex.Array,
        # freqs_cis: Tuple[chex.Array, chex.Array],
        # attention_mask: chex.Array,
        # position_ids: chex.Array,
        # causal_mask: chex.Array,
        # segment_ids: Optional[chex.Array] = None,
        # deterministic: bool = True,
        # init_cache: bool = False,
        # output_attentions: bool = False,
        # fcm_mask = None,
        residual = hidden_states
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states),
            freqs_cis,
            attention_mask,
            position_ids,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
            None,
        )

        hidden_states = attention_output[0] + residual
        ffd_inp = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                ffd_inp,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.mlp(
                ffd_inp,
                deterministic,
            )

        hidden_states = hidden_states + feed_forward_hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_output[1],)
        return outputs


class FlaxOlmoPretrainedModel(BaseNNXModule):
    config_class = OlmoConfig
    base_model_prefix = "mistral"
    module_class: nn.Module = None

    def __init__(
        self,
        config: OlmoConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self,
        rng: jax.random.PRNGKey,
        input_shape: Tuple,
        params: flax.core.FrozenDict = None,
    ) -> flax.core.FrozenDict:
        """The init_weights function is used to initialize the weights of a model.
        It takes in an rng, which is a random number generator key that can be used to generate random numbers.
        The input_shape parameter specifies the shape of the inputs that will be fed into this model.
        The params parameter allows you to pass in pre-trained weights for your model, if you have them available.

        Args:
            self: Access variables that belong to the class
            rng: jax.random.PRNGKey: Initialize the weights of the model
            input_shape: Tuple: Initialize the input_ids, attention_mask
                and position_ids
            params: flax.core.FrozenDict: Pass in the parameters of a
                pre-trained model

        Returns:
            A frozendict of parameters
        """
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rng_s = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rng_s,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rng_s, input_ids, attention_mask, position_ids, return_dict=False
            )

        random_params = module_init_outputs["params"]

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
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return init_variables["cache"]

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_params_field: bool = False,
        **kwargs,
    ):
        """The __call__ function is the main function of a JAX module.
        It takes as input:
        - The parameters of the model (self.params)
        - The inputs to the model (input_ids, attention_mask, position_ids)
        - Whether we are training (train=True/False) and whether we want to return all hidden states and
        attentions weights at each layer in addition to just the last layer output (output_hidden_states=True/False).

        Args:
            self: Represent the instance of the class
            input_ids: Pass the input sequence to the model
            attention_mask: Mask out the padding tokens
            position_ids: Specify the position of each token in the
                sequence
            params: dict: Pass in the parameters of the model
            past_key_values: dict: Pass the past key values to the model
            dropout_rng: jax.random.PRNGKey: Pass in a random number
                generator key to the model
            train: bool: Determine whether to use dropout or not
            output_attentions: Optional[bool]: Determine whether to
                return the attention weights
            output_hidden_states: Optional[bool]: Determine whether to
                return the hidden states of all layers
            return_dict: Optional[bool]: Return a dictionary of the
                outputs
            add_params_field: bool: Add a params field to the inputs
                dictionary

        Returns:
            A tuple of (last_hidden_state, past_key_values)
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rng_s = {}
        if dropout_rng is not None:
            rng_s["dropout"] = dropout_rng

        inputs = (
            {"params": params or self.params}
            if add_params_field
            else params or self.params
        )

        if self.config.bits is not None:
            rng_s["params"] = jax.random.key(0)
        if past_key_values is not None:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            None,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rng_s,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxOlmoDecoratorCollection(nn.Module):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxOlmoDecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i),
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        causal_mask: chex.Array,
        position_ids: chex.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # hidden_states: chex.Array,
            # freqs_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: chex.Array,
            # causal_mask: chex.Array,
            # position_ids: chex.Array,
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # init_cache: bool = False,
            # output_attentions: bool = True

            output = layer(
                hidden_states,
                freqs_cis,
                attention_mask,
                causal_mask,
                position_ids,
                None,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = output[0]

            if output_attentions:
                output_attentions += (output[1],)

        return hidden_states, all_hidden_states, all_attentions


class FlaxOlmoModule(nn.Module):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=nnx.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.layers = FlaxOlmoDecoratorCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = LayerNormRaw(
            eps=1e-5,
        )

        initial_rope_kwargs = dict(rope_type="none")
        if self.config.rope_scaling is not None:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor, rope_type=scaling_type
            )
        self.freqs_cis = precompute_freqs_cis(
            max_position_embeddings=(
                getattr(
                    self.config,
                    "freq_max_position_embeddings",
                    self.config.max_position_embeddings,
                )
            ),
            dim=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rope_theta,
            **initial_rope_kwargs,
        )
        self.causal_mask = flax.linen.make_causal_mask(
            jnp.ones(
                (
                    1,
                    getattr(
                        self.config,
                        "causal_mask_max_position_embeddings",
                        self.config.max_position_embeddings,
                    ),
                ),
                dtype="bool",
            ),
            dtype="bool",
        )

    def __call__(
        self,
        input_ids: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        input_embeds: chex.Array = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[jax.Array, ...], FlaxBaseModelOutput]:
        """The __call__ function is the main function of a Flax model.
        It takes in input_ids, attention_mask, and position_ids as inputs to the model.
        The output is a tuple containing: last hidden state (hidden states), all hidden states (if output_hidden_states=True), attentions (if output attentions=True).

        Args:
            self: Represent the instance of the class
            input_ids: chex.Array: Pass in the input ids
            attention_mask: chex.Array: Mask out the attention weights
                for certain tokens
            position_ids: chex.Array: Determine the position of each
                token in a sequence
            deterministic: bool: Determine whether to use dropout or not
            input_embeds: chex.Array: Pass in the embedding of the
                input_ids
            init_cache: bool: Initialize the cache for the decoder
            output_attentions: bool: Determine whether to return the
                attention weights or not
            output_hidden_states: bool: Return all hidden states or just
                the last one
            return_dict: bool: Return a dictionary of the outputs or not
        :param : Determine whether the model is in training mode or not

        Returns:
            A tuple of the hidden states, all hidden states, and
            attentions
        """
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids.astype("i4"))
        if attention_mask.ndim == 2:
            b, s = attention_mask.shape
            attention_mask = attention_mask.reshape(b, 1, 1, s)

        outputs = self.layers(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freqs_cis=self.freqs_cis,
            init_cache=init_cache,
            output_attentions=output_attentions,
            deterministic=deterministic,
            causal_mask=self.causal_mask,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(value for value in outputs if value is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxOlmoModel(FlaxOlmoPretrainedModel):
    module_class = FlaxOlmoModule

    def set_input_embeddings(self, value):
        self.module.embed_tokens = value

    def get_input_embeddings(self):
        return self.module.embed_tokens


class FlaxOlmoForCausalLMModule(nn.Module):
    config: OlmoConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.model: FlaxOlmoModule = FlaxOlmoModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.lm_head = Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        deterministic: bool = True,
        input_embeds: chex.Array = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """The __call__ function is the main function of a Flax module. It defines how the model will be called,
        and what it returns. In this case, we are calling our Transformer model with input_ids and attention_mask
        as inputs (these are defined in __init__). We also have some optional arguments that can be passed to
        the call function: deterministic (whether to use dropout), input_embeds (if you want to pass your own embeddings),
        output_attentions and output_hidden states which return additional outputs from the transformer layers if set True. Finally,

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass in the input tokens
            attention_mask: chex.Array: Mask out the padding tokens
            position_ids: chex.Array: Specify the position of each token
                in the sequence
            deterministic: bool: Determine whether to use dropout in the
                model
            input_embeds: chex.Array: Pass in the embeddings of the
                input tokens
            init_cache: bool: Initialize the cache for the decoder
            output_attentions: bool: Return the attention weights
            output_hidden_states: bool: Return the hidden states of all
                layers
            return_dict: bool: Return a dictionary of the outputs or
                just the logits
        :param : Determine whether to return the logits or not

        Returns:
            A tuple of (lm_logits, hidden_states, attentions)
        """
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            input_embeds=input_embeds,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["embed_tokens"][
                "embedding"
            ]
            shared_kernel = nn.control_quantization(shared_kernel, self.param_dtype).T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        # lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxOlmoForCausalLM(FlaxOlmoPretrainedModel):
    module_class = FlaxOlmoForCausalLMModule

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

    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: Optional[chex.Array] = None
    ):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
