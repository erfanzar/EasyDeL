import math
from flax import linen as nn
from flax.core import FrozenDict, unfreeze
from typing import Optional, Dict, Union, Tuple, List, Callable

from flax.linen import combine_masks
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax import numpy as jnp, lax
import jax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
from ..flax_modelling_utils import get_gradient_checkpoint_policy, \
    with_sharding_constraint
import chex
from fjutils.utils import transpose


# transpose = jax.jit(trp)


class FalconConfig(PretrainedConfig):
    model_type = "falcon"
    attribute_map = {
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
    }

    def __init__(
            self,
            vocab_size: int = 65024,
            hidden_size: int = 4544,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 71,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            hidden_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_kv_heads=None,
            alibi: bool = False,
            new_decoder_architecture: bool = False,
            multi_query: bool = True,
            parallel_attn: bool = True,
            bias: bool = False,
            max_position_embeddings: int = 2048,
            rope_theta: float = 10000.0,
            rope_scaling=None,
            bos_token_id: int = 11,
            eos_token_id: int = 11,
            use_pjit_attention_force: bool = False,
            gradient_checkpointing: str = 'nothing_saveable',
            **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.use_pjit_attention_force = use_pjit_attention_force
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
        self.gradient_checkpointing = gradient_checkpointing
        self.parallel_attn = parallel_attn
        self.num_kv_heads = num_kv_heads
        self.new_decoder_architecture = new_decoder_architecture
        self.from_pt = False

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = False):
        return (
            ('word_embeddings/embedding', PartitionSpec('dp', 'fsdp')),
            ('self_attention/query_key_value/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('self_attention/dense/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('mlp/dense_4h_to_h/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('mlp/dense_h_to_4h/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('lm_head/kernel', PartitionSpec('dp', 'fsdp')),
            ('transformer/ln_f/bias', PartitionSpec('fsdp')),
            ('transformer/ln_f/scale', PartitionSpec('fsdp')),
            ('transformer/post_attention_layernorm/scale', PartitionSpec('fsdp')),
            ('transformer/post_attention_layernorm/bias', PartitionSpec('fsdp')),
            ('.*', PartitionSpec('dp'))
        ) if not fully_fsdp else (
            ('word_embeddings/embedding', PartitionSpec('fsdp')),
            ('self_attention/query_key_value/(kernel|bias)', PartitionSpec('fsdp')),
            ('self_attention/dense/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/dense_4h_to_h/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/dense_h_to_4h/(kernel|bias)', PartitionSpec('fsdp')),
            ('lm_head/kernel', PartitionSpec('fsdp')),
            ('transformer/ln_f/bias', PartitionSpec('fsdp')),
            ('transformer/ln_f/scale', PartitionSpec('fsdp')),
            ('transformer/post_attention_layernorm/scale', PartitionSpec('fsdp')),
            ('transformer/post_attention_layernorm/bias', PartitionSpec('fsdp')),
            ('.*', PartitionSpec('fsdp'))
        )

    @staticmethod
    def get_mesh_names():
        return 'dp', 'fsdp', 'mp'

    def add_jax_args(self,
                     vocab_size: int = 65024,
                     hidden_size: int = 4544,
                     num_hidden_layers: int = 32,
                     num_attention_heads: int = 71,
                     layer_norm_epsilon: float = 1e-5,
                     initializer_range: float = 0.02,
                     use_cache: bool = True,
                     hidden_dropout: float = 0.0,
                     attention_dropout: float = 0.0,
                     num_kv_heads=None,
                     alibi: bool = False,
                     new_decoder_architecture: bool = False,
                     multi_query: bool = True,
                     parallel_attn: bool = True,
                     bias: bool = False,
                     max_position_embeddings: int = 2048,
                     rope_theta: float = 10000.0,
                     rope_scaling=None,
                     bos_token_id: int = 11,
                     eos_token_id: int = 11,
                     use_pjit_attention_force: bool = False,
                     gradient_checkpointing: str = 'nothing_saveable',
                     **kwargs,
                     ):
        basics = dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            rope_theta=rope_theta,
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            num_kv_heads=num_kv_heads,
            eos_token_id=eos_token_id,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            multi_query=multi_query,
            alibi=alibi,
            bias=bias,
            parallel_attn=parallel_attn,
            rope_scaling=rope_scaling,
            use_pjit_attention_force=use_pjit_attention_force,
            gradient_checkpointing=gradient_checkpointing,
            new_decoder_architecture=new_decoder_architecture,
            **kwargs
        )
        for key_state, value_state in basics.items():
            if not hasattr(self, key_state):
                setattr(self, key_state, value_state)

        self.from_pt = False
        return self


def built_bloom_alibi(attention_mask, num_attention_heads):
    batch_size, sequence_length = attention_mask.shape
    cp2 = 2 ** math.floor(math.log2(num_attention_heads))
    base = jnp.asarray(
        2 ** (- (2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32
    )
    powers = jnp.arange(1, 1 + cp2, dtype=jnp.float32)
    slops = jnp.power(base, powers)
    if cp2 != num_attention_heads:
        extra_base = jnp.asarray(
            2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32
        )
        num_rem_heads = min(cp2, num_attention_heads - cp2)
        extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
        slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
    arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[:, jnp.newaxis, :]
    alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
    return alibi.reshape(batch_size, num_attention_heads, 1, sequence_length)


def precompute_falcon_freq_cis(max_position_embedding: int, head_dim: int, theta: float = 10000):
    inv_freq_cis = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freq = jnp.einsum("i , j -> i j", jnp.arange(max_position_embedding), inv_freq_cis).astype("float32")

    embed = jnp.concatenate((freq, freq), axis=-1)
    return jnp.sin(embed)[:, :], jnp.cos(embed)[:, :]


def _rotate_half(x):
    return jnp.concatenate((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), axis=-1)


def apply_rotary_pos_embedding(tensor, sin_, cos_):
    return (tensor * cos_) + (_rotate_half(tensor) * sin_)


@nn.compact
def dropout_add(self, x: chex.Array, residual: chex.Array, prob: float, deterministic: bool) -> chex.Array:
    """
    Dropout add function

    Args:
        self (Self)
            self must be passed to the func
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        deterministic (`bool`, *required*):
            training mode
    """
    nn_ = nn.Dropout(prob)
    out = nn_(inputs=x, deterministic=deterministic)
    out = residual + out
    return out


class FlaxFalconRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, :]
        cos = cos[position_ids][:, :]

        _, sequence_length, _ = query.shape

        # query_expansion_factor = int(query.shape[0] / cos.shape[0])
        # key_expansion_factor = int(key.shape[0] / cos.shape[0])

        query_expansion_factor = 1
        key_expansion_factor = 1

        if query_expansion_factor > 1:
            query_cos = jnp.tile(cos, (query_expansion_factor,))
            query_sin = jnp.tile(sin, (query_expansion_factor,))
        else:
            query_cos, query_sin = cos, sin

        if key_expansion_factor > 1:
            if key_expansion_factor != query_expansion_factor:
                key_cos = jnp.tile(cos, (key_expansion_factor,))
                key_sin = jnp.tile(sin, (key_expansion_factor,))
            else:
                key_cos, key_sin = query_cos, query_sin
        else:
            key_cos, key_sin = cos, sin

        query = apply_rotary_pos_embedding(query, query_sin, query_cos)
        key = apply_rotary_pos_embedding(key, key_sin, key_cos)
        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxFalconAttention(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.query_key_value = nn.Dense(
            features=3 * self.config.hidden_size if not self.config.multi_query else (
                    self.config.hidden_size + 2 * head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.inv_norm_factor = 1 / math.sqrt(head_dim)
        self.dense = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.head_dim = head_dim
        self.maybe_rotary = FlaxFalconRotaryEmbedding(self.dtype) if not self.config.alibi else lambda q, k, a, s: (
            q, k)
        assert self.head_dim * self.config.num_attention_heads == self.config.hidden_size
        self.num_kv_heads = self.config.num_kv_heads if (
                self.config.new_decoder_architecture or not self.config.multi_query) else 1

    @nn.compact
    def _concatenate_to_cache(self, key: chex.Array, value: chex.Array, query: chex.Array, attention_mask: chex.Array):
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (int(cur_index), 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
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
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def apply_maybe_rotary(self, batch_size, sequence_length, query, key, value, freq_cis, position_ids):
        query = query.reshape(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query, key, value = self._t(query, key, value)
        query, key = self.rotary(position_ids=position_ids, query=query, key=key, freq_cis=freq_cis)
        return self._t(query, key, value)

    def split_head(self, qkv: chex.Array):
        batch_size, sequence_length, _ = qkv.shape
        if self.config.new_decoder_architecture:
            batch, sequence_length, _ = qkv.shape
            qkv = qkv.reshape(batch, sequence_length, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query_state = qkv[:, :, :, :-2]
            key_state = qkv[:, :, :, [-2]]
            value_state = qkv[:, :, :, [-1]]
            key_state = jnp.broadcast_to(key_state, query_state.shape)
            value_state = jnp.broadcast_to(value_state, query_state.shape)

            query_state, key_state, value_state = [x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],)) for x in
                                                   (query_state, key_state, value_state)]
            if self.config.use_pjit_attention_force:
                query_state = with_sharding_constraint(query_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
                key_state = with_sharding_constraint(key_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
                value_state = with_sharding_constraint(value_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            return query_state, key_state, value_state
        if self.config.multi_query:
            qkv = qkv.reshape(
                batch_size, sequence_length, self.config.num_attention_heads + 2, -1
            )
            query_state, key_state, value_state = qkv[..., :-2, :], qkv[..., [-2], :], qkv[..., [-1], :]

        else:
            query_state, key_state, value_state = jnp.split(qkv, 3, -1)

        if self.config.use_pjit_attention_force:
            query_state = with_sharding_constraint(query_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            key_state = with_sharding_constraint(key_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            value_state = with_sharding_constraint(value_state, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
        return query_state, key_state, value_state

    def _merge_heads(self, x: chex.Array) -> chex.Array:

        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.config.num_attention_heads, seq_length, self.head_dim)

        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.config.num_attention_heads * self.head_dim)

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array = None,
            alibi: chex.Array = None,
            freq_cis: Tuple[chex.Array, chex.Array] = None,
            output_attentions: bool = False,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        num_kv_heads = self.num_kv_heads
        query_layer, key_layer, value_layer = self.split_head(self.query_key_value(hidden_states))
        query_layer = transpose(
            query_layer, 1, 2
        ).reshape(
            batch_size * self.config.num_attention_heads,
            sequence_length,
            self.head_dim
        )
        key_layer = transpose(
            key_layer, 1, 2
        ).reshape(
            batch_size * num_kv_heads,
            sequence_length,
            self.head_dim,
        )
        value_layer = transpose(
            value_layer, 1, 2
        ).reshape(
            batch_size * num_kv_heads,
            sequence_length,
            self.head_dim
        )
        kv_length = key_layer.shape[1]
        if not self.config.alibi:
            query_layer, key_layer = self.maybe_rotary(
                query_layer,
                key_layer,
                freq_cis,
                position_ids
            )

        float_min = jnp.finfo(query_layer.dtype).min
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, float_min).astype(self.dtype),
        )

        query_layer_ = query_layer.reshape(batch_size, self.config.num_attention_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        dtype = jnp.promote_types(key_layer_.dtype, jnp.float32)

        query_layer_, key_layer_, value_layer_, attention_bias = map(lambda x: x.astype(dtype=dtype), (
            query_layer_, key_layer_, value_layer_, attention_bias))

        attention_scores = jax.lax.batch_matmul(query_layer_, transpose(key_layer_, -2, -1))
        if alibi is None:

            attention_scores /= math.sqrt(self.head_dim)

            attention_scores = jax.nn.softmax(
                attention_scores + attention_bias, axis=-1
            )
            attn_output = jax.lax.batch_matmul(attention_scores, value_layer_)
            attn_output = attn_output.reshape(batch_size, self.num_heads, sequence_length, self.head_dim)
            attn_output = transpose(attn_output, 2, 1)
            attn_output = attn_output.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, attention_scores
            else:
                return output_tensor,
        else:

            attention_scores = attention_scores.reshape(batch_size, self.num_heads, sequence_length, kv_length)
            attention_scores = attention_scores + alibi.reshape(batch_size, self.num_heads, 1, -1)
            attention_scores *= self.inv_norm_factor
            attention_scores = jax.nn.softmax(
                attention_scores + attention_bias, axis=-1
            )
            attention_scores = attention_scores.reshape(batch_size, self.num_heads, sequence_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]

            attn_output = jax.lax.batch_matmul(attention_scores, value_layer_)
            attn_output = attn_output.reshape((attn_output.shape[1] * attn_output.shape[0],) + attn_output.shape[2:])
            attn_output = self._merge_heads(attn_output)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, attention_scores
            else:
                return output_tensor,


class FlaxFalconMlp(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.dense_h_to_4h = nn.Dense(
            features=self.config.hidden_size * 4,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.dense_4h_to_h = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )

    def __call__(self, x: chex.Array, deterministic: bool = True):
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))


class FlaxFalconBlock(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config
        self.input_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                            dtype=self.dtype)
        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                                         dtype=self.dtype)
        if config.new_decoder_architecture:
            self.ln_attn = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                        dtype=self.dtype)
            self.ln_mlp = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                       dtype=self.dtype)
        self.mlp = FlaxFalconMlp(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.self_attention = FlaxFalconAttention(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            alibi: chex.Array,
            attention_mask: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            position_ids: chex.Array,
            causal_mask: chex.Array,
            output_attentions: bool = True,
            deterministic: bool = True
    ):
        residual = hidden_states
        mlp_layernorm_out = None
        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            hidden_states=attention_layernorm_out,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            alibi=alibi,
            freq_cis=freq_cis,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    self=self,
                    x=attention_output,
                    residual=residual,
                    prob=self.config.attention_dropout,
                    deterministic=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        mlp_output = self.mlp(mlp_layernorm_out, deterministic=deterministic)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(
            self=self,
            x=mlp_output,
            residual=residual,
            prob=self.config.hidden_dropout,
            deterministic=self.training
        )

        return output, outputs


class FlaxFalconCollection(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = FlaxFalconBlock
        if self.config.gradient_checkpointing != '':
            block = nn.remat(
                block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
        self.layers = [
            block(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(
                self.config.num_hidden_layers
            )
        ]

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 alibi: chex.Array,
                 freq_cis: Tuple[chex.Array, chex.Array],
                 position_ids: chex.Array,
                 causal_mask: chex.Array,
                 output_attentions: bool = True,
                 deterministic: bool = True
                 ):
        for layer in self.layers:
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                alibi=alibi,
                freq_cis=freq_cis,
                position_ids=position_ids,
                causal_mask=causal_mask,
                deterministic=deterministic,
                output_attentions=output_attentions
            )
            hidden_states = out[0]
        return hidden_states


class FlaxFalconModule(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.h = FlaxFalconCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ln_f = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, epsilon=self.config.layer_norm_epsilon)
        self.causal_mask = nn.attention.make_causal_mask(jnp.ones((1, self.config.max_position_embeddings)))
        if not self.config.alibi:
            self.freq_cis: Tuple[chex.Array, chex.Array] = precompute_falcon_freq_cis(
                max_position_embedding=self.config.max_position_embeddings,
                head_dim=self.config.hidden_size // self.config.num_attention_heads
            )
        else:
            self.freq_cis = None

    def __call__(self,
                 input_ids: chex.Array,
                 attention_mask: Optional[chex.Array] = None,
                 position_ids: Optional[chex.Array] = None,
                 output_attentions: bool = True,
                 deterministic: bool = True,
                 use_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = False
                 ):
        batch, sequence_length = input_ids.shape
        if position_ids is None:
            position_ids = jnp.arange(0, sequence_length).reshape(1, -1)
        if attention_mask is None:
            attention_mask = jnp.ones((batch, sequence_length))

        hidden_states = self.word_embeddings(
            inputs=input_ids.astype(jnp.int32)
        )
        alibi = None
        if self.config.alibi:
            alibi = built_bloom_alibi(attention_mask, self.config.num_attention_heads).astype(hidden_states.dtype)

        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :]

        out_layers = self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            freq_cis=self.freq_cis,
            causal_mask=self.causal_mask,
            output_attentions=output_attentions,
            deterministic=deterministic
        )

        out = out_layers[0]
        outputs = out_layers[1:]
        output = self.ln_f(out)

        if return_dict:
            if output_attentions:
                return FlaxBaseModelOutput(
                    last_hidden_state=output,
                    attentions=out_layers[1]
                )
            else:
                return FlaxBaseModelOutput(
                    last_hidden_state=output,
                )
        else:
            return output, outputs


class FlaxFalconPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class = FalconConfig

    def __init__(self, config,
                 _do_init=False,
                 dtype: jnp.dtype = jnp.float32,
                 param_dtype: jnp.dtype = jnp.float32,
                 input_shape: Tuple = (1, 1024),
                 precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision('fastest')
                 ):
        module = self.module_class(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision)
        super().__init__(_do_init=_do_init, module=module, config=config, dtype=dtype, input_shape=input_shape)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> Dict:
        if params is None:
            params = self.module.init(
                rngs=rng,
                input_ids=jnp.ones(input_shape),
                attention_mask=jnp.ones(input_shape)
            )
        return params['params']

    def __call__(self,
                 input_ids: chex.Array,
                 attention_mask: Optional[chex.Array] = None,
                 position_ids: Optional[chex.Array] = None,
                 past_key_values: Optional[nn.Module] = None,
                 output_attentions: bool = True,
                 deterministic: bool = True,
                 use_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = False,
                 params: FrozenDict = None,
                 add_params_field: bool = False,
                 ):
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        inputs = {'params': params or self.params} if add_params_field else params or self.params
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[1])[None, :],
                                            (input_ids.shape[0], input_ids.shape[1]))

        if attention_mask is None:
            attention_mask = jnp.ones((input_ids.shape[0], input_ids.shape[1]))

        # input_ids: chex.Array,
        # attention_mask: Optional[chex.Array] = None
        # position_ids: Optional[chex.Array] = None
        # output_attentions: bool = True
        # deterministic: bool = True
        # use_cache: Optional[bool] = None
        # return_dict: Optional[bool] = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            output_attentions,
            deterministic,
            use_cache,
            return_dict,
            mutable=mutable
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]
        return outputs

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    @staticmethod
    def update_inputs_for_generation(model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


class FlaxFalconModel(FlaxFalconPretrainedModel):
    module_class = FlaxFalconModule


class FlaxFalconForCausalLMModule(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.transformer = FlaxFalconModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False
        )

    def __call__(self,
                 input_ids: chex.Array,
                 attention_mask: Optional[chex.Array] = None,
                 position_ids: Optional[chex.Array] = None,
                 output_attentions: bool = True,
                 deterministic: bool = True,
                 use_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = False
                 ):
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict
        )
        if return_dict:
            hidden_state = transformer_output.last_hidden_state
        else:
            hidden_state = transformer_output[0]
        output = self.lm_head(hidden_state)
        if return_dict:
            if output_attentions:
                return FlaxCausalLMOutput(
                    logits=output,
                    attentions=transformer_output.attentions
                )
            else:
                return FlaxCausalLMOutput(
                    logits=output,
                )
        else:
            return (output, transformer_output[1]) if output_attentions else (output,)


class FlaxFalconForCausalLM(FlaxFalconPretrainedModel):
    module_class = FlaxFalconForCausalLMModule
