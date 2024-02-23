import math
from typing import Optional, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput

from .llama_configuration import LlamaConfig
from ..easy_attention import EasyAttention
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
# EasyDel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits, BaseJAXAttentionModule, block_wise_ffn
)


class FlaxLlamaEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


def repeat_kv(x: chex.Array, n_rep: int) -> chex.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jnp.newaxis, :, :]
    x = jnp.repeat(x, n_rep, axis=2)

    return x.reshape(bs, s,
                     n_kv_heads * n_rep,
                     head_dim)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

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
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxLlamaAttention(BaseJAXAttentionModule):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.k_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.v_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.o_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.rotary = FlaxLlamaEmbedding(self.dtype)
        self.attention_performer = EasyAttention(
            attn_type="normal",
            block_k_major=self.config.block_k_major,
            block_b=self.config.block_b,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            block_q_major_dkv=self.config.block_q_major_dkv,
            block_k_major_dkv=self.config.block_k_major_dkv,
            block_k_major_dq=self.config.block_k_major_dq,
            block_k_dkv=self.config.block_k_dkv,
            block_q_dkv=self.config.block_q_dkv,
            block_q_dq=self.config.block_q_dq,
            block_k_dq=self.config.block_k_dq,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.head_dim,
            attention_partition_spec=self.config.attention_partition_spec,
            use_shard_map=self.config.use_shard_map,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
        )
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @staticmethod
    def _transpose_sequence_head(query, key, value):
        """
        The _transpose_sequence_head function transposes the query, key and value matrices.

        :param query: Get the attention weights for each of the heads
        :param key: Determine the number of heads
        :param value: Store the values of the input
        :return: The transpose of the query, key and value matrices

        """
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def apply_rotary(self, batch_size, sequence_length, query, key, value, freq_cis, position_ids):
        """
        The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freq_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        :param self: Access variables that belong to the class
        :param batch_size: Reshape the query, key and value tensors
        :param sequence_length: Reshape the query, key and value tensors
        :param query: Calculate the attention weights
        :param key: Calculate the attention
        :param value: Compute the attention weights
        :param freq_cis: Calculate the frequency of each word in the vocabulary
        :param position_ids: Identify the position of each token in the sequence
        :return: A tuple of 3 tensors: query, key and value

        """
        query = query.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )

        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary(
            position_ids=position_ids, query=query, key=key, freq_cis=freq_cis
        )
        key = repeat_kv_bnsh(key, self.num_key_value_groups)
        value = repeat_kv_bnsh(value, self.num_key_value_groups)
        return self._transpose_sequence_head(query, key, value)

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
    ):
        """

        The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        :param self: Access variables that belong to the class
        :param hidden_states: chex.Array: Pass the hidden states of the previous layer
        :param freq_cis: chex.Array: Pass in the frequency coefficients for each position
        :param attention_mask: chex.Array: Mask out certain tokens in the input sequence
        :param position_ids: chex.Array: Determine the position of each token in a sequence
        :param causal_mask: chex.Array: Mask out the future tokens in the decoder
        :param deterministic: bool: Determine whether to use dropout or not
        :param init_cache: bool: Initialize the cache
        :param output_attentions: bool: Determine whether to return the attention weights or not
        :param fcm_mask: Mask out the attention weights between the input and output tokens
        :param : Determine if the attention is causal or not
        :return: A tuple of two arrays

        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_state, key_state, value_state = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(
            hidden_states)

        if self.config.use_pjit_attention_force:
            query_state = with_sharding_constraint(
                query_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))
            key_state = with_sharding_constraint(
                key_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))
            value_state = with_sharding_constraint(
                value_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))

        query_state = query_state.reshape(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_state = key_state.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value_state = value_state.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query_state, key_state, value_state = self.apply_rotary(
            query=query_state,
            key=key_state,
            value=value_state,
            position_ids=position_ids,
            freq_cis=freq_cis,
            batch_size=batch_size,
            sequence_length=sequence_length
        )

        assert_msg = (
            "num_attention_heads repeat wont work likely\n"
            f"INFO :\n\trepeat_kv_bnsh Used with num_key_value_groups = {self.num_key_value_groups}\n\t"
            f"NH : {self.config.num_attention_heads} KVH : {self.config.num_attention_heads}"
        )

        assert query_state.shape[-2] == self.config.num_attention_heads, assert_msg
        assert key_state.shape[-2] == self.config.num_attention_heads, assert_msg
        assert value_state.shape[-2] == self.config.num_attention_heads, assert_msg

        query_length, key_length = query_state.shape[1], key_state.shape[1]

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
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_state, value_state, attention_mask = self._concatenate_to_cache(
                key_state,
                value_state,
                query_state,
                attention_mask
            )

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )

        query_length, key_length = query_state.shape[1], key_state.shape[1]

        attentions = self.attention_performer.__call__(
            query_states=query_state,
            key_states=key_state,
            value_states=value_state,
            bias=attention_bias,
            causal=False,
            use_pjit_attention_force=self.config.use_pjit_attention_force,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
        )
        attentions.attention_outputs = attentions.attention_outputs

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.o_proj(attn_output)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attentions.attention_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.gate_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        :param self: Represent the instance of the class
        :param x: jnp.ndarray: Pass in the input to the layer
        :param deterministic: bool: Determine whether to use dropout
        :return: A tensor that is the result of applying a dropout function to x

        """
        x = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class FlaxLlamaBlock(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        attn_block = FlaxLlamaAttention
        if self.config.gradient_checkpointing != "":
            attn_block = nn_partitioning.remat(
                FlaxLlamaAttention, static_argnums=(5, 6, 7),
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing)
            )

        self.self_attn = attn_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        mlp_block = FlaxLlamaMLP

        if self.config.gradient_checkpointing != "":
            mlp_block = nn_partitioning.remat(
                FlaxLlamaMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                )
            )

        self.mlp = mlp_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.input_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,

        )

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask: Optional[jnp.ndarray] = None,
    ):
        """
        The __call__ function is the main function of a TransformerEncoderLayer.
        It takes in hidden states, frequency-domain inputs, and masks as input. It then
        applies self-attention to the hidden states using those inputs and returns an
        output tensor with shape (batch_size, sequence_length, model_dim).

        :param self: Refer to the class instance itself
        :param hidden_states: chex.Array: Pass in the hidden state of the previous layer
        :param freq_cis: chex.Array: Pass in the frequency information
        :param attention_mask: chex.Array: Mask out the attention weights for padding tokens
        :param position_ids: chex.Array: Determine the position of each token in the sequence
        :param causal_mask: chex.Array: Mask the attention weights
        :param deterministic: bool: Control whether the dropout is applied or not
        :param init_cache: bool: Initialize the cache in the attention layer
        :param output_attentions: bool: Return the attention weights
        :param fcm_mask: Optional[jnp.ndarray]: Mask the self-attention
        :param : Control the dropout in the self attention layer
        :return: A tuple of two items

        """
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            freq_cis,
            attention_mask,
            position_ids,
            causal_mask,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.mlp(
                feed_forward_input,
                deterministic,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxLlamaPreTrainedModel(EasyDelFlaxPretrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
            self,
            config: LlamaConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what happens when it's created.
        The __init__ function can take arguments, but self is always required (it refers to the instance of the object).


        :param self: Refer to the object itself
        :param config: LlamaConfig: Pass the configuration to the module
        :param input_shape: Tuple: Specify the shape of the input to the model
        :param seed: int: Set the seed for random number generation
        :param dtype: jnp.dtype: Specify the data type of the input
        :param _do_init: bool: Control whether the module is initialized or not
        :param kwargs: Pass in any additional parameters that the module_class might need
        :param : Specify the number of layers in the network
        :return: The super() of the class

        """
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape,
                         seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Specify the shape of the input tensor
        :param params: FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters

        """
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_shape)
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
                rngs, input_ids, attention_mask, position_ids, return_dict=False)

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
        """
        The init_cache function is used to initialize the cache for a given batch size and sequence length.
        The cache is a dictionary that contains all the intermediate states from each layer in the model.
        This allows us to run inference on multiple batches without having to re-run forward passes through every layer in
        the model, which would be very slow.

        :param self: Access the module
        :param batch_size: Define the batch size of the input tensors
        :param max_length: Set the length of the input sequence
        :return: A dictionary with the following keys:

        """
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
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        """
        The __call__ function is the main function of a JAX module.
        It takes in inputs and returns outputs, but it also has some other important features:
        - It can take in mutable state (e.g., past_key_values) that will be updated during the call and returned at the end.
        - It can take in random number generators (rngs) that are used to generate random numbers for dropout or sampling operations.

        :param self: Represent the instance of the class
        :param input_ids: chex.Array: Pass in the input tokens
        :param attention_mask: chex.Array: Mask out certain tokens in the input
        :param position_ids: chex.Array: Create the positional embeddings
        :param params: dict: Pass in the parameters of the model
        :param past_key_values: dict: Pass in the past key values from a previous call to __call__
        :param dropout_rng: jax.random.PRNGKey: Make sure that the dropout is applied in a random way
        :param train: bool: Determine whether to use dropout or not
        :param output_attentions: Optional[bool]: Determine whether to return the attention weights
        :param output_hidden_states: Optional[bool]: Return the hidden states of all layers
        :param return_dict: Optional[bool]: Determine whether to return a dictionary or not
        :param extra_embedding: Optional[Union[jnp.ndarray,None]]: Pass in the embedding for the input_ids
        :param add_params_field: bool: Add the params field to the inputs dictionary
        :return: A tuple of the following:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        assert sequence_length <= self.config.max_position_embeddings, "Maximum Position Embedding Reached !"

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[
                                            None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if self.config.bits is not None:
            rngs['params'] = jax.random.key(0)

        inputs = {
            "params": params or self.params
        } if add_params_field else params or self.params

        if past_key_values:
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
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            extra_embedding,
            rngs=rngs,
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


class FlaxLlamaBlockCollection(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.blocks = [
            FlaxLlamaBlock(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        """
        The __call__ function is the main function of a JAX nn.Module.
        It defines how the module behaves when called as a function, and it's what you'll use to call your model
         in training loops or inference scripts.
        The __call__ method should take all inputs that are necessary for computing outputs from the module,
        and return all outputs that are computed by this module.

        :param self: Represent the instance of the class
        :param hidden_states: chex.Array: Pass the input tensor to the encoder
        :param freq_cis: chex.Array: Pass in the frequency of each token
        :param attention_mask: chex.Array: Mask out certain tokens in the input sequence
        :param position_ids: chex.Array: Specify the position of each token in a sequence
        :param causal_mask: chex.Array: Mask the attention weights
        :param deterministic: bool: Determine whether the model is in training or evaluation mode
        :param init_cache: bool: Initialize the cache for each layer
        :param output_attentions: bool: Determine whether to output the attention weights
        :param output_hidden_states: bool: Determine whether to return the hidden states of each layer
        :param return_dict: bool: Return a dictionary of the outputs
        :param : Determine whether to use the forgetful causal mask
        :return: A tuple of 3 values

        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng('fcm'), shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio
            )
            fcm_mask = jax.random.uniform(
                self.make_rng('fcm'),
                shape=(batch_size, 1, seq_length, seq_length)
            ) > fcm_ratio
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype('bool')
        else:
            fcm_mask = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                freq_cis=freq_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                fcm_mask=fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.layers = FlaxLlamaBlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                                               precision=self.precision)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype,
                            param_dtype=self.param_dtype)
        config = self.config
        self.causal_mask = nn.make_causal_mask(
            jnp.ones(
                (1, getattr(self.config, "c_max_position_embeddings", self.config.max_position_embeddings)),
                dtype="bool"
            ), dtype="bool"
        )

        initial_rope_kwargs = dict(
            rope_type="none"
        )
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor,
                rope_type=scaling_type
            )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=getattr(
                self.config,
                "freq_max_position_embeddings",
                self.config.max_position_embeddings
            ),
            dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            **initial_rope_kwargs
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            inputs_embeds: chex.Array = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        """
        The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
        and returns the output of the model. The __call__ function also has optional arguments that can be used to control
        the behavior of the model (e.g., deterministic=True). These optional arguments are passed as keyword arguments when
        calling a Flax model.

        :param self: Represent the instance of the class
        :param input_ids: chex.Array: Pass in the input token ids
        :param attention_mask: chex.Array: Mask out the padding tokens
        :param position_ids: chex.Array: Indicate the position of each token in a sequence
        :param deterministic: bool: Control whether dropout is applied or not
        :param inputs_embeds: chex.Array: Pass in the embeddings of the input tokens
        :param init_cache: bool: Initialize the cache
        :param output_attentions: bool: Determine whether to return the attentions or not
        :param output_hidden_states: bool: Determine whether to return hidden states
        :param return_dict: bool: Return a dictionary of the output or not
        :param extra_embedding: Optional[Union[jnp.ndarray: Pass in the embedding of the
        :param None]]: Pass in the extra embedding
        :return: A tuple of:

        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        batch_size, sequence_length, _ = inputs_embeds.shape

        assert sequence_length <= self.config.max_position_embeddings, "Maximum Position Embedding Reached !"
        inputs_embeds = inputs_embeds + \
                        extra_embedding if extra_embedding is not None else inputs_embeds
        hidden_states = self.dropout(
            inputs_embeds, deterministic=deterministic)

        outputs = self.layers(
            hidden_states=hidden_states,
            freq_cis=self.freq_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=self.causal_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaModule

    def set_input_embeddings(self, value):
        self.module.embed_tokens = value

    def get_input_embeddings(self):
        return self.module.embed_tokens


class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.model = FlaxLlamaModule(self.config,
                                     dtype=self.dtype,
                                     param_dtype=self.param_dtype,
                                     precision=self.precision,
                                     )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        """
        The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        :param self: Refer to the object itself
        :param input_ids: chex.Array: Pass the input token ids to the model
        :param attention_mask: chex.Array: Mask out the padding tokens
        :param position_ids: chex.Array: Specify the position of each token in the input sequence
        :param deterministic: bool: Control whether the model is trained or not
        :param init_cache: bool: Initialize the cache for the decoder
        :param output_attentions: bool: Return the attention weights
        :param output_hidden_states: bool: Determine whether to return the hidden states
        :param return_dict: bool: Return a dictionary of the outputs or not
        :param extra_embedding: Optional[Union[jnp.ndarray: Pass in the embedding of the word that we want to predict
        :param None]]: Pass in the extra embedding
        :return: The logits and the hidden states

        """
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForCausalLMModule

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


class FlaxLlamaForSequenceClassificationModule(nn.Module):
    num_classes: int
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        """
        The setup function is called once at the beginning of training.
        It initializes the model and optimizer, and sets up any other state that needs to be initialized.

        :param self: Access variables that belong to the class
        :return: A tuple of the model and the classifier
        """
        self.model = FlaxLlamaModule(self.config, dtype=self.dtype)
        self.classifier = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        """
        The __call__ function is the main function of a Flax module.
        It takes in all the inputs to the model and returns all outputs from it.
        The __call__ function can be called directly on an instance of a class, or by using parentheses after an instance:
            &gt;&gt;&gt; my_model = MyModel()  # instantiate your model class
            &gt;&gt;&gt; output = my_model(input)  # call your model with input data as arguments to __call__

        :param self: Refer to the class instance
        :param input_ids: chex.Array: Pass the input to the model
        :param attention_mask: chex.Array: Specify which tokens are masked
        :param position_ids: chex.Array: Specify the position of each token in the sequence
        :param deterministic: bool: Control whether the model is run in deterministic or stochastic mode
        :param init_cache: bool: Initialize the cache for the transformer
        :param output_attentions: bool: Return the attention weights
        :param output_hidden_states: bool: Return the hidden states of all layers
        :param return_dict: bool: Return a dictionary of outputs
        :param extra_embedding: Optional[Union[jnp.ndarray: Pass in the embedding of a new word
        :param None]]: Pass the extra embedding to the model
        :return: A tuple of logits and hidden_states

        """
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding
        )

        hidden_states = outputs[0]
        prediction = self.classifier(hidden_states)
        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=prediction,
                hidden_states=hidden_states
            )
        else:
            return prediction,


class FlaxLlamaForSequenceClassification(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForSequenceClassificationModule
