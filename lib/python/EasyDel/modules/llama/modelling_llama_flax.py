from functools import partial
from typing import Dict, Optional, Tuple, Union, Sequence

import fjformer.attention
from einops import einops
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning, dot_product_attention_weights
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput
# EasyDel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    JaxBaseClassModel,
    smart_flash_attention, get_dot_general_by_bits
)
import chex
from fjformer.bits import config as q_config, q_flax


class LlamaConfig(JaxBaseClassModel):
    model_type = "llama"

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            number_rep_kv: int = 1,
            num_key_value_heads: Optional[int] = None,
            max_position_embeddings: int = 2048,
            rms_norm_eps: float = 1e-6,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            bos_token_id: int = 0,
            eos_token_id: int = 1,
            resid_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            attention_dropout: float = 0.0,
            rope_theta: float = 10000.,
            attention_bias: bool = False,
            tie_word_embeddings: bool = False,
            gradient_checkpointing: str = "nothing_saveable",
            fcm_min_ratio: float = -1,
            fcm_max_ratio: float = -1,
            use_pjit_attention_force: bool = False,
            rope_scaling: Dict[str, Union[str, float]] = None,
            use_flash_attention: bool = False,
            use_sacn_mlp: bool = False,
            flash_attn_query_chunk_size: int = 1024,
            flash_attn_key_chunk_size: int = 1024,
            scan_mlp_chunk_size: int = 1024,
            bits: Optional[int] = None,
            hidden_act: str = 'silu',
            pretraining_tp: int = 1,
            scan_layers: bool = True,
            use_shard_map: bool = True,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object, which are sometimes called fields or properties.
        The __init__ function can accept arguments, but self must be the first one.

        :param self: Refer to the object itself
        :param vocab_size: int: Set the size of the vocabulary
        :param hidden_size: int: Set the size of the hidden layers in each transformer block
        :param intermediate_size: int: Set the size of the intermediate layer
        :param num_hidden_layers: int: Determine the number of layers in the transformer
        :param num_attention_heads: int: Determine the number of attention heads
        :param number_rep_kv: int: Set the number of times to repeat the key and value vectors
        :param num_key_value_heads: Optional[int]: Define the number of key-value heads
        :param max_position_embeddings: int: Set the maximum length of a sequence
        :param rms_norm_eps: float: Prevent division by zero in the rms normalization
        :param initializer_range: float: Initialize the weights of the model
        :param use_cache: bool: Determine whether the attention layer should use a cache for faster computation
        :param bos_token_id: int: Set the beginning of sequence token
        :param eos_token_id: int: Specify the end of sentence token
        :param resid_pdrop: float: Set the dropout rate for residual connections
        :param embd_pdrop: float: Dropout the embedding layer
        :param attention_dropout: float: Dropout the attention weights
        :param tie_word_embeddings: bool: Tie the word embeddings and output layer weights
        :param gradient_checkpointing: str: Specify how to checkpoint the gradients
        :param fcm_min_ratio: float: Set the minimum ratio of the number of elements in a tensor to be processed by flash
        :param fcm_max_ratio: float: Determine the maximum ratio of
        :param use_pjit_attention_force: bool: Determine whether to use the pytorch jit compiler
        :param rope_scaling: Dict[str: Define the scaling of the rope
        :param Union[str: Specify the type of the parameter
        :param float]]: Specify the type of the parameter
        :param use_shard_map: bool: when ever to use shard_map for attention
        :param use_flash_attention: bool: Determine whether to use the flash attention or not
        :param use_sacn_mlp: bool: Determine whether to use scan_mlp or not
        :param flash_attn_query_chunk_size: int: Specify the chunk size of the query tensor
        :param flash_attn_key_chunk_size: int: Determine the chunk size of the key tensor
        :param scan_mlp_chunk_size: int: Specify the chunk size of the scan_mlp
        :param bits: Optional[int]: Specify the number of bits used to quantize the weights
        :param rope_theta: float : rope_theta for compute rope
        :param attention_bias: bool : whenever to use attention bias or no
        :param hidden_act: str : hidden_act for mlp
        :param axis_dims: Sequence[int]: Specify the dimensions of each axis
        :param axis_names: Sequence[str]: Specify the names of the axes in a tensor
        :param scan_layers: bool: Determine whether to use the scan_layers or not
        :param **kwargs: Pass a variable number of keyword arguments to a function
        :param : Define the number of layers in the model
        :return: Nothing

        """
        num_key_value_heads = num_key_value_heads or number_rep_kv * num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size

        self.number_rep_kv = number_rep_kv
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.fcm_min_ratio = fcm_min_ratio
        self.hidden_act = hidden_act
        self.fcm_max_ratio = fcm_max_ratio
        self.rope_scaling = rope_scaling
        self.use_flash_attention = use_flash_attention
        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        self.use_sacn_mlp = use_shard_map
        self.scan_layers = scan_layers
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = True):
        """
        The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
            1) A regex string that matches the name of one or more parameters in the model.
            2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

        :param fully_fsdp: bool: Determine whether to partition the model fully or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PS("tp", ("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PS("tp", ("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PS(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PS("tp", ("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PS(("fsdp", "sp"), "tp")),

            ("input_layernorm/kernel", PS(None)),
            ("post_attention_layernorm/kernel", PS(None)),

            ("model/norm/kernel", PS(None)),
            ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
            ('.*', PS(None)),
        ) if not fully_fsdp else (

            ("model/embed_tokens/embedding", PS(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS(("fsdp", "sp"))),
            ("self_attn/o_proj/kernel", PS(("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PS(("fsdp", "sp"))),
            ("mlp/down_proj/kernel", PS(("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PS(("fsdp", "sp"))),

            ("input_layernorm/kernel", PS(None)),
            ("post_attention_layernorm/kernel", PS(None)),

            ("model/norm/kernel", PS(None)),
            ("lm_head/kernel", PS(("fsdp", "sp"))),
            ('.*', PS(("fsdp", "sp"))),
        )

    def add_jax_args(self,
                     resid_pdrop: float = 0.0,
                     embd_pdrop: float = 0.0,
                     attention_dropout: float = 0.0,
                     tie_word_embeddings: bool = False,
                     gradient_checkpointing: str = 'nothing_saveable',
                     fcm_min_ratio: float = 0.0,
                     fcm_max_ratio: float = 0.0,
                     use_pjit_attention_force: bool = False,
                     use_flash_attention: bool = False,
                     use_sacn_mlp: bool = False,
                     flash_attn_query_chunk_size: int = 1024,
                     flash_attn_key_chunk_size: int = 1024,
                     scan_mlp_chunk_size: int = 1024,
                     number_rep_kv: int = 1,
                     bits: Optional[int] = None,
                     rope_theta: float = 10000.,
                     attention_bias: bool = False,
                     hidden_act: str = 'silu',
                     scan_layers: bool = True,
                     **kwargs,
                     ):
        """
        The add_jax_args function adds the following arguments to the Transformer class:

        :param self: Refer to the current object
        :param resid_pdrop: float: Set the dropout rate for residual connections
        :param embd_pdrop: float: Set the probability of dropping an embedding
        :param attention_dropout: float: Set the probability of dropping out the attention layer
        :param tie_word_embeddings: bool: Tie the word embeddings to the decoder
        :param gradient_checkpointing: str: Control the amount of memory used by jax
        :param fcm_min_ratio: float: Control the minimum ratio of the number of chunks to be used in flash-based computation
        :param fcm_max_ratio: float: Set the maximum ratio of the number of input tokens to output tokens
        :param use_pjit_attention_force: bool: Determine if the attention force is used
        :param use_flash_attention: bool: Determine whether to use the flash attention or not
        :param use_sacn_mlp: bool: Determine whether to use the scan_mlp function or not
        :param flash_attn_query_chunk_size: int: Determine the size of the chunks that will be used to compute
        :param flash_attn_key_chunk_size: int: Set the size of the key chunk
        :param scan_mlp_chunk_size: int: Set the chunk size for scan_mlp
        :param number_rep_kv: int: Determine how many times the key and value vectors are repeated
        :param bits: Optional[int]: Determine the number of bits used in the quantization
        :param rope_theta: float : rope_theta for compute rope
        :param attention_bias: bool : whenever to use attention bias or no
        :param hidden_act: str : hidden_act for mlp
        :param scan_layers: bool: Determine whether to use scan layers or not
        :return: The following:

        """
        self.scan_layers = scan_layers
        self.use_flash_attention = use_flash_attention
        self.embd_pdrop = embd_pdrop
        self.number_rep_kv = number_rep_kv
        self.resid_pdrop = resid_pdrop
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.use_pjit_attention_force = use_pjit_attention_force

        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'


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


class FlaxLlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.number_of_reps = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.number_of_reps == 1:
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

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        The _concatenate_to_cache function is used to concatenate the key and value vectors
        of a query with those of previous queries. This allows for the attention mechanism to
        look at all previous queries when computing its output. The function takes in three
        arguments: key, value, and query. It also uses two variables that are stored in the cache:
        cached_key and cached_value.

        :param self: Access the variables stored in the cache
        :param key: Store the keys of the encoder-decoder attention
        :param value: Initialize the cached_value variable
        :param query: Determine the number of cache vectors to update
        :param attention_mask: Mask out the padded vectors in the cache
        :return: The key, value and attention_mask

        """
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(
                cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors

            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    @staticmethod
    def _t(query, key, value):
        """
        The _t function transposes the query, key and value matrices.

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
        query = query.reshape(batch_size, sequence_length,
                              self.config.num_attention_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length,
                          self.config.num_key_value_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length,
                              self.config.num_key_value_heads, self.head_dim)

        query, key, value = self._t(query, key, value)
        query, key = self.rotary(
            position_ids=position_ids, query=query, key=key, freq_cis=freq_cis)
        key = repeat_kv_bnsh(key, self.number_of_reps)
        value = repeat_kv_bnsh(value, self.number_of_reps)
        return self._t(query, key, value)

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
                query_state, PS(("dp", "fsdp"), "sp", "tp"))
            key_state = with_sharding_constraint(
                key_state, PS(("dp", "fsdp"), "sp", "tp"))
            value_state = with_sharding_constraint(
                value_state, PS(("dp", "fsdp"), "sp", "tp"))

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
            f"INFO :\n\trepeat_kv_bnsh Used with number_of_reps = {self.number_of_reps}\n\t"
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
                causal_mask, (0, 0, mask_shift, 0), (1, 1,
                                                     query_length, max_decoder_length)
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

        if self.config.use_flash_attention and not (self.has_variable("cache", "cached_key") or init_cache):
            if attention_mask.shape[1] != self.config.num_attention_heads:
                attention_mask = attention_mask.repeat(
                    self.config.num_attention_heads, 1, )
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(
                    self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            rtp_axis = (0, 2, 1, 3)
            attn_output = smart_flash_attention(
                q=jnp.transpose(query_state, rtp_axis),
                k=jnp.transpose(key_state, rtp_axis),
                v=jnp.transpose(value_state, rtp_axis),
                q_ps=self.config.q_ps,
                k_ps=self.config.k_ps,
                v_ps=self.config.v_ps,
                b_ps=self.config.b_ps,
                a_ps=self.config.a_ps,
                bias=attention_bias,
                block_q=self.config.flash_attn_query_chunk_size,
                block_k=self.config.flash_attn_key_chunk_size,
                block_b=1,
                num_attention_heads=self.config.num_attention_heads,
                precision=self.precision,
                dtype=self.dtype,
                causal=False,
                mesh=self.config.jax_mesh(),
                dropout_rng=dropout_rng,
                deterministic=deterministic,
                q_seq_len=sequence_length,
                kv_seq_len=key_length,
                attn_pdrop=self.config.attention_dropout,
                head_dims=self.head_dim,
                force_float32_tpu=True
            )
            attn_output = jnp.transpose(attn_output, rtp_axis)
        else:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(
                    self.dtype).min).astype(self.dtype),
            )
            if self.config.use_shard_map:
                attn_weights = shard_map(
                    partial(
                        dot_product_attention_weights,
                        dtype=jnp.promote_types(self.dtype, jnp.float32),
                        deterministic=deterministic,
                        dropout_rate=self.config.attention_dropout,
                        precision=self.precision,
                    ),
                    mesh=self.config.jax_mesh(),
                    in_specs=(
                        self.config.q_ps,
                        self.config.k_ps,
                        self.config.b_ps
                    ),
                    out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
                    check_rep=False
                )(
                    query_state, key_state, attention_bias
                )
            else:
                attn_weights = dot_product_attention_weights(
                    query=query_state,
                    key=key_state,
                    bias=attention_bias,
                    dtype=jnp.promote_types(self.dtype, jnp.float32),
                    deterministic=deterministic,
                    dropout_rate=self.config.attention_dropout,
                    precision=self.precision,
                )

            if self.config.use_pjit_attention_force:
                attn_weights = with_sharding_constraint(
                    attn_weights, PS(("dp", "fsdp"), "sp", "tp", None))

            attn_output = jnp.einsum(
                "...hqk,...khd->...qhd",
                attn_weights,
                value_state,
                precision=self.precision
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        attn_output = self.resid_dropout(
            attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (
            attn_output,)

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
        if self.config.gradient_checkpointing != '':
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

        if self.config.gradient_checkpointing != '':
            mlp_block = nn_partitioning.remat(
                FlaxLlamaMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing)
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

        if self.config.use_sacn_mlp:
            feed_forward_input = einops.rearrange(
                feed_forward_input,
                '... (b s) d -> ... b s d',
                b=self.config.scan_mlp_chunk_size
            )

            def mlp_forward(mlp, carry, x):
                return None, mlp(x, deterministic)

            scan_axis = feed_forward_input.ndim - 3

            _, feed_forward_hidden_states = nn.scan(
                mlp_forward,
                variable_broadcast="params",
                split_rngs={"params": False, "dropout": True},
                in_axes=scan_axis,
                out_axes=scan_axis,
            )(self.mlp, None, feed_forward_input)
            feed_forward_hidden_states = einops.rearrange(
                feed_forward_hidden_states,
                '... b s d -> ... (b s) d'
            )
        else:
            feed_forward_hidden_states = self.mlp(
                feed_forward_input,
                deterministic,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
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
        :param **kwargs: Pass in any additional parameters that the module_class might need
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
            add_params_field: bool = False
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

        assert sequence_length <= self.config.max_position_embeddings, (f'Position out of range '
                                                                        f'(Model Support '
                                                                        f'{self.config.max_position_embeddings} got'
                                                                        f' {sequence_length})')

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
            "params": params or self.params} if add_params_field else params or self.params

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
        It defines how the module behaves when called as a function, and it's what you'll use to call your model in training loops or inference scripts.
        The __call__ method should take all inputs that are necessary for computing outputs from the module, and return all outputs that are computed by this module.

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
        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_position_embeddings)))
        
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
            max_position_embeddings=config.max_position_embeddings,
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
        assert sequence_length <= self.config.max_position_embeddings, (f'Position out of range '
                                                                        f'(Model Support '
                                                                        f'{self.config.max_position_embeddings} got'
                                                                        f' {sequence_length})')
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
        :param output_hidden_states: bool: Determine whether or not to return the hidden states
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
