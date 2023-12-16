import functools
import typing
from typing import Sequence

import fjformer.attention
import flax.core
from jax import numpy as jnp, Array, lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS
import jax
from flax import linen as nn
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.core import freeze, unfreeze
from typing import Union, Optional, Tuple
from transformers import PretrainedConfig, FlaxPreTrainedModel
from flax.linen import partitioning as nn_partitioning, combine_masks, dot_product_attention_weights
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

from ..flax_modelling_utils import (
    ACT2FN,
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    JaxBaseClassModel,
    get_flash_attention,
    smart_flash_attention, get_dot_general_by_bits
)
import chex
from fjformer.bits import config as q_config, q_flax


class MixtralConfig(JaxBaseClassModel):
    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=4096 * 32,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=1e6,
            sliding_window=4096,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            num_local_experts=8,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            use_flash_attention: bool = False,
            use_sacn_mlp: bool = False,
            flash_attn_query_chunk_size: int = 1024,
            flash_attn_key_chunk_size: int = 1024,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            attn_pdrop: float = 0.0,
            c_max_position_embeddings: int = 4096,
            freq_max_position_embeddings: int = 4096,
            bits: Optional[int] = None,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It allows the class to initialize the attributes of a class.
        The self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.

        :param self: Represent the instance of the class
        :param vocab_size: Define the size of the vocabulary
        :param hidden_size: Determine the size of the embedding layers
        :param intermediate_size: Define the size of the intermediate layer in each transformer block
        :param num_hidden_layers: Determine the number of layers in the encoder and decoder
        :param num_attention_heads: Determine the number of attention heads in each layer
        :param num_key_value_heads: Specify the number of heads for key and value
        :param hidden_act: Specify the activation function used in the hidden layers
        :param max_position_embeddings: Set the maximum length of the sequence
        :param initializer_range: Initialize the weights of the model
        :param rms_norm_eps: Avoid division by zero in the rms normalization
        :param use_cache: Determine whether to use the cache in the decoder
        :param pad_token_id: Specify the token id of the padding token
        :param bos_token_id: Specify the beginning of sentence token id
        :param eos_token_id: Specify the end of sentence token
        :param tie_word_embeddings: Tie the word embeddings and the output layer
        :param rope_theta: Control the number of tokens in a rope
        :param sliding_window: Control the number of tokens that are processed in parallel
        :param gradient_checkpointing: str: Specify whether to use gradient checkpointing
        :param use_pjit_attention_force: bool: Force the use of pjit attention
        :param use_flash_attention: bool: Enable the flash attention mechanism
        :param use_sacn_mlp: bool: Determine whether or not to use the scan_mlp function
        :param flash_attn_query_chunk_size: int: Determine the number of rows in each chunk
        :param flash_attn_key_chunk_size: int: Control the size of chunks that are used for the key matrix in flash attention
        :param scan_mlp_chunk_size: int: Specify the chunk size of the scan mlp
        :param number_rep_kv: int: Specify the number of times to repeat the key and value vectors
        :param attn_pdrop: float: Set the dropout rate for the attention layer
        :param c_max_position_embeddings: int: Set the maximum number of tokens in a sequence
        :param freq_max_position_embeddings: int: Set the maximum number of frequency bins that can be used in the model
        :param bits: Optional[int]: Specify the number of bits used for quantization
        :param axis_dims: Sequence[int]: Specify the dimension of each axis
        :param axis_names: Sequence[str]: Specify the names of each axis in the tensor
        :param &quot;mp&quot;): Define the maximum position embeddings
        :param **kwargs: Pass a variable number of keyword arguments to a function
        :param : Define the number of layers in the model
        :return: An instance of the class

        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.bits = bits
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.use_flash_attention = use_flash_attention
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attn_pdrop = attn_pdrop
        self.c_max_position_embeddings = c_max_position_embeddings
        self.freq_max_position_embeddings = freq_max_position_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
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
          2) A PartitionScheme object that defines how those parameters should be partitioned.

        :param fully_fsdp: bool: Determine whether to use the fully_fsdp partitioning scheme or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PS("sp", "fsdp")),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS("fsdp", "sp")),
            ("self_attn/o_proj/kernel", PS("sp", "fsdp")),

            ("mlp/gate_proj/kernel", PS("fsdp", "sp")),
            ("mlp/down_proj/kernel", PS("sp", "fsdp")),
            ("mlp/up_proj/kernel", PS("fsdp", "sp")),

            ("input_layernorm/kernel", PS(None)),
            ("post_attention_layernorm/kernel", PS(None)),

            ("model/norm/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "sp")),
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
                     gradient_checkpointing: str = 'nothing_saveable',
                     use_pjit_attention_force: bool = False,
                     use_flash_attention: bool = False,
                     use_sacn_mlp: bool = False,
                     flash_attn_query_chunk_size: int = 1024,
                     flash_attn_key_chunk_size: int = 1024,
                     scan_mlp_chunk_size: int = 1024,
                     number_rep_kv: int = 1,
                     attn_pdrop: float = 0.0,
                     c_max_position_embeddings: int = 4096,
                     freq_max_position_embeddings: int = None,
                     bits: Optional[int] = None,
                     **kwargs,
                     ):
        """
        The add_jax_args function adds the following arguments to the model:

        :param self: Bind the attributes and methods of a class to an instance of that class
        :param gradient_checkpointing: str: Determine whether to use gradient checkpointing
        :param use_pjit_attention_force: bool: Determine whether to use the pjit_attention_force function
        :param use_flash_attention: bool: Determine if the flash attention module is used or not
        :param use_sacn_mlp: bool: Determine whether to use the scan_mlp function or not
        :param flash_attn_query_chunk_size: int: Specify the number of tokens that will be processed at a time
        :param flash_attn_key_chunk_size: int: Chunk the keys for flash attention
        :param scan_mlp_chunk_size: int: Chunk the input to the mlp
        :param number_rep_kv: int: Control the number of times that the key and value vectors are repeated
        :param attn_pdrop: float: Set the dropout rate for the attention layer
        :param c_max_position_embeddings: int: Set the maximum number of positional embeddings for the causal axis
        :param freq_max_position_embeddings: int: Set the maximum length of the frequency axis
        :param bits: Optional[int]: Specify the number of bits to use for quantization
        :return: A tuple of the following:

        """
        self.use_flash_attention = use_flash_attention
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attn_pdrop = attn_pdrop
        self.c_max_position_embeddings = c_max_position_embeddings
        self.freq_max_position_embeddings = freq_max_position_embeddings
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'


re_mat = nn_partitioning.remat


def _make_sliding_window_causal_mask(
        input_ids_shape,
        dtype: jnp.dtype,
        past_key_values_length: int = 0,
        sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = jnp.full(
        (tgt_len, tgt_len),
        fill_value=1,
    )
    mask = jnp.tril(tensor, 0)
    mask = jnp.triu(mask, -sliding_window)
    mask = jnp.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].repeat(bsz, 0)


class MixtralRMSNorm(nn.Module):
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
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxMixtralRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxMixtralAttention(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        dense = functools.partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.q_proj = dense(self.num_heads * self.head_dim)
        self.k_proj = dense(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(self.hidden_size)
        self.rotary = FlaxMixtralRotaryEmbedding(self.dtype)

    @nn.compact
    def concatenate_to_cache_(self, query: chex.Array, key: chex.Array, value: chex.Array, attention_mask: chex.Array):
        is_cache_available = self.has_variable('cache', 'key')
        key_cache = self.variable('cache', 'key', jnp.zeros, key.shape, key.dtype)
        value_cache = self.variable('cache', 'value', jnp.zeros, key.shape, value.dtype)
        index_cache = self.variable('cache', 'index', lambda: jnp.array(0, dtype=jnp.int32))
        if is_cache_available:
            *bd, ml, nh, dph = key_cache.value.shape
            indices = (0,) * len(bd) + (index_cache.value, 0, 0)
            key = jax.lax.dynamic_update_slice(key_cache.value, key, indices)
            value = jax.lax.dynamic_update_slice(value_cache.value, value, indices)
            key_cache.value = key
            value_cache.value = value
            num_updated_cache_vector = query.shape[1]
            index_cache.value = index_cache.value + num_updated_cache_vector
            pad_mask = jnp.broadcast_to(
                jnp.arange(ml) < index_cache.value,
                tuple(bd) + (1, num_updated_cache_vector, ml)
            )
            attention_mask = nn.combine_masks(pad_mask, attention_mask)
        return query, key, value, attention_mask

    @staticmethod
    def _t(query, key, value):
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def t_rotary(self, batch_size, sequence_length, query, key, value, freq_cis, position_ids):
        query = query.reshape(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query, key, value = self._t(query, key, value)
        query, key = self.rotary(position_ids=position_ids, query=query, key=key, freq_cis=freq_cis)
        key = repeat_kv_bnsh(key, self.num_key_value_groups)
        value = repeat_kv_bnsh(value, self.num_key_value_groups)
        return self._t(query, key, value)

    def __call__(
            self,
            hidden_state: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            causal_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = True
    ):
        """
        The __call__ function is the main function of a JAX module.
        It defines how the module behaves when called as a function, and it's what you'll use to call your model in practice.
        The __call__ method takes an input tensor (x) and returns an output tensor (y).
        In this case, we're defining our model to be a simple linear layer with no activation: y = x @ w + b.

        :param self: Refer to the object itself
        :param hidden_state: chex.Array: Pass in the hidden state of the model
        :param freq_cis: chex.Array: Create the t_rotary variable
        :param attention_mask: chex.Array: Mask the attention weights
        :param causal_mask: chex.Array: Mask the attention weights
        :param position_ids: chex.Array: Specify the position of each token in a sequence
        :param deterministic: bool: Determine whether to use dropout or not
        :param init_cache: bool: Initialize the cache
        :param output_attentions: bool: Determine whether to return the attention weights
        :return: A tuple of (out, attn_output)

        """
        batch_size, sequence_length = hidden_state.shape[:2]
        query, key, value = self.q_proj(hidden_state), self.k_proj(hidden_state), self.v_proj(hidden_state)

        if self.config.use_pjit_attention_force:
            query = with_sharding_constraint(query, PS("fsdp", "sp", None))
            key = with_sharding_constraint(key, PS("fsdp", "sp", None))
            value = with_sharding_constraint(value, PS("fsdp", "sp", None))
        query, key, value = self.t_rotary(
            batch_size=batch_size,
            sequence_length=sequence_length,
            query=query,
            key=key,
            value=value,
            freq_cis=freq_cis,
            position_ids=position_ids
        )
        if self.has_variable('cache', 'key') or init_cache:
            query, key, value, attention_mask = self.concatenate_to_cache_(query, key, value, attention_mask)

        q_l, k_l = query.shape[1], key.shape[1]
        if self.has_variable('cache', 'key'):
            mask_shift: int = self.variables['cache']['index']
            dl = self.variables['cache']['key'].shape[1]
            causal_mask = jax.lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1, q_l, dl)
            )
        else:
            causal_mask = causal_mask[:, :, :q_l, :k_l]
        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask.ndim == 2:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)

        attention_mask = nn.combine_masks(attention_mask, causal_mask)

        if self.config.use_flash_attention and not (self.has_variable("cache", "cached_key") or init_cache):

            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            if attention_mask.shape[1] != self.config.num_attention_heads:
                attention_mask = attention_mask.repeat(self.config.num_attention_heads, 1, )
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            rtp_axis = (0, 2, 1, 3)
            attn_output = smart_flash_attention(
                q=jnp.transpose(query, rtp_axis),
                k=jnp.transpose(key, rtp_axis),
                v=jnp.transpose(value, rtp_axis),
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
                q_seq_len=q_l,
                kv_seq_len=k_l,
                attn_pdrop=self.config.attn_pdrop,
                head_dims=self.head_dim,
                force_float32_tpu=True
            )
            attn_output = jnp.transpose(attn_output, rtp_axis)
        else:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            if self.config.use_shard_map:
                attn_weights = shard_map(
                    functools.partial(
                        dot_product_attention_weights,
                        dtype=jnp.promote_types(self.dtype, jnp.float32),
                        deterministic=deterministic,
                        dropout_rate=self.config.attn_pdrop,
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
                    query, key, attention_bias
                )
            else:
                attn_weights = dot_product_attention_weights(
                    query=query,
                    key=key,
                    bias=attention_bias,
                    dtype=jnp.promote_types(self.dtype, jnp.float32),
                    deterministic=deterministic,
                    dropout_rate=self.config.attn_pdrop,
                    precision=self.precision,
                )

            if self.config.use_pjit_attention_force:
                attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "sp", "tp", None))

            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)

        out = self.o_proj(attn_output.reshape(batch_size, sequence_length, self.hidden_size))
        outputs = (out, attn_weights) if output_attentions else (out,)
        return outputs


class FlaxMixtralBLockSparseTop2MLP(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        dense = functools.partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.w1 = dense(self.config.intermediate_size)
        self.w3 = dense(self.config.intermediate_size)
        self.w2 = dense(self.config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x: chex.Array):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class FlaxMixtralBlocKSparesTop2MLPBlock(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        self.layers = [
            FlaxMixtralBLockSparseTop2MLP(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(self.config.num_local_experts)
        ]

    def __call__(self,
                 expert_mask: chex.Array,
                 hidden_states: chex.Array,
                 routing_weights: chex.Array,
                 batch_size: int,
                 sequence_length: int,
                 hidden_dim: int
                 ) -> chex.Array:
        assert hidden_states.ndim == 2
        final_hidden_states = jnp.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype
        )

        def add_at(x, indices, value):
            # Create a view of the tensor that
            # is the same shape as the tensor being updated, but with zeros in the indices
            view = jnp.zeros_like(x)
            view = view.at[indices].set(value)

            # Add the view to the original tensor
            updated_x = x + view

            return updated_x

        def index_add_inplace(destination, indices, source):
            # Create a custom function to perform in-place addition at specified indices
            def update_element_at_index(arr, index, value):
                return jax.ops.index_update(arr, index, arr[index] + value)

            # Iterate through the indices and perform in-place addition
            for i in range(len(indices)):
                destination = update_element_at_index(destination, indices[i], source[i])

            return destination

        for expert_idx, expert_layer in enumerate(self.layers):

            idx, top_x = jnp.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            final_hidden_states = index_add_inplace(final_hidden_states, top_x,
                                                    current_hidden_states.astype(hidden_states.dtype))
        return final_hidden_states


class FlaxMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """
    config: MixtralConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        self.gate = nn.Dense(
            self.config.num_local_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
        )

    def __call__(self, *args, **kwargs):
        ...


class FlaxMixtralDecoderLayer(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        self.self_attn = FlaxMixtralAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.mlp = FlaxMistralMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.input_layernorm = MixtralRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.post_attention_layernorm = MixtralRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

    def __call__(
            self,
            hidden_state: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            causal_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = True
    ):
        """
        The __call__ function is the main function of a TransformerEncoderLayer.
        It takes in the following arguments:
            hidden_state (chex.Array): The input to the encoder layer, which is also its output after being processed by all sublayers.
            freq_cis (chex.Array): A tensor containing frequency-domain representations of each token's context vector, used for computing self-attention weights and biases in a more efficient manner than using position embeddings or sinusoidal positional encoding vectors would allow for [2]. This tensor has shape `(batch_size, num

        :param self: Represent the instance of the class
        :param hidden_state: chex.Array: Represent the input to the encoder layer
        :param freq_cis: chex.Array: Pass the frequency information to the attention layer
        :param attention_mask: chex.Array: Mask out the attention weights for certain positions
        :param causal_mask: chex.Array: Mask the future tokens
        :param position_ids: chex.Array: Indicate the position of each token in the sequence
        :param deterministic: bool: Determine whether to use dropout or not
        :param init_cache: bool: Initialize the cache for the self-attention layer
        :param output_attentions: bool: Determine whether to return the attention weights or not
        :return: A tuple of hidden_state and attention_output

        """
        residual = hidden_state
        attention_output = self.self_attn(
            hidden_state=self.input_layernorm(hidden_state),
            freq_cis=freq_cis,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions
        )

        hidden_state = attention_output[0] + residual

        hidden_state = self.mlp(self.post_attention_layernorm(hidden_state)) + hidden_state
        outputs = (hidden_state,)
        if output_attentions:
            outputs += attention_output[1]
        return outputs
