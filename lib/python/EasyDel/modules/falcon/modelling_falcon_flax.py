import math
from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Dict, Union, Tuple

from flax.linen import combine_masks
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax import numpy as jnp, lax
import jax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
from einops import rearrange
from ..flax_modelling_utils import get_gradient_checkpoint_policy, \
    with_sharding_constraint
import chex
from fjutils.utils import transpose


class FalconConfig(PretrainedConfig):
    model_type = "falcon"
    attribute_map = {
        "num_hiddenum_hidden_layerss": "num_hidden_layers",
        "num_attentionum_attention_headss": "num_attention_heads",
    }

    def __init__(
            self,
            vocab_size: int = 250880,
            hidden_size: int = 64,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 71,
            n_layers: int = 32,
            n_heads: int = 71,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            apply_residual_connection_post_layernorm: bool = False,
            hidden_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            multi_query: bool = False,
            alibi: bool = False,
            bias: bool = False,
            parallel_attn: bool = False,
            max_seq_len: int = 2048,
            use_pjit_attention_force: bool = False,
            gradient_checkpointing: str = 'nothing_saveable',
            new_decoder_architecture: bool = False,
            num_kv_heads: int = 1,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
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
            ('wte/embedding', PartitionSpec('dp', 'fsdp')),
            ('self_attention/w_qkv/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('self_attention/wo/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('mlp/down/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('mlp/up/(kernel)', PartitionSpec('dp', 'fsdp')),
            ('lm_head/kernel', PartitionSpec('dp', 'fsdp')),
            ('transformer/ln_f/bias', PartitionSpec('fsdp')),
            ('transformer/ln_f/scale', PartitionSpec('fsdp')),
            ('transformer/post_attentionum_hidden_layersnorm/scale', PartitionSpec('fsdp')),
            ('transformer/post_attentionum_hidden_layersnorm/bias', PartitionSpec('fsdp')),
            ('.*', PartitionSpec('dp'))
        ) if not fully_fsdp else (
            ('wte/embedding', PartitionSpec('fsdp')),
            ('self_attention/w_qkv/(kernel|bias)', PartitionSpec('fsdp')),
            ('self_attention/wo/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/down/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/up/(kernel|bias)', PartitionSpec('fsdp')),
            ('lm_head/kernel', PartitionSpec('fsdp')),
            ('transformer/ln_f/bias', PartitionSpec('fsdp')),
            ('transformer/ln_f/scale', PartitionSpec('fsdp')),
            ('transformer/post_attentionum_hidden_layersnorm/scale', PartitionSpec('fsdp')),
            ('transformer/post_attentionum_hidden_layersnorm/bias', PartitionSpec('fsdp')),
            ('.*', PartitionSpec('fsdp'))
        )

    @staticmethod
    def get_mesh_names():
        return 'dp', 'fsdp', 'mp'

    def add_jax_args(self,
                     vocab_size: int = 250880,
                     hidden_size: int = 64,
                     num_hidden_layers: int = 2,
                     num_attention_heads: int = 8,
                     layer_norm_epsilon: float = 1e-5,
                     initializer_range: float = 0.02,
                     use_cache: bool = True,
                     bos_token_id: int = 1,
                     eos_token_id: int = 2,
                     apply_residual_connection_post_layernorm: bool = False,
                     hidden_dropout: float = 0.0,
                     attention_dropout: float = 0.0,
                     multi_query: bool = False,
                     alibi: bool = False,
                     bias: bool = False,
                     parallel_attn: bool = False,
                     max_seq_len: int = 2048,
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
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            apply_residual_connection_post_layernorm=apply_residual_connection_post_layernorm,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            multi_query=multi_query,
            alibi=alibi,
            bias=bias,
            parallel_attn=parallel_attn,
            max_seq_len=max_seq_len,
            use_pjit_attention_force=use_pjit_attention_force,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        for key_state, value_state in basics.items():
            if not hasattr(self, key_state):
                setattr(self, key_state, value_state)

        self.from_pt = False
        return self


def built_bloom_alibi(attention_mask, num_attentionum_attention_headss):
    batch_size, sequence_length = attention_mask.shape
    cp2 = 2 ** math.floor(math.log2(num_attentionum_attention_headss))
    base = jnp.asarray(
        2 ** (- (2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32
    )
    powers = jnp.arange(1, 1 + cp2, dtype=jnp.float32)
    slops = jnp.power(base, powers)
    if cp2 != num_attentionum_attention_headss:
        extra_base = jnp.asarray(
            2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32
        )
        num_rem_heads = min(cp2, num_attentionum_attention_headss - cp2)
        extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
        slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
    arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[:, jnp.newaxis, :]
    alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
    return alibi.reshape(batch_size, num_attentionum_attention_headss, 1, sequence_length)


def precompute_falcon_freq_cis(max_position_embedding: int, head_dim: int, theta: float = 10000):
    inv_freq_cis = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freq = jnp.einsum("i , j -> i j", jnp.arange(max_position_embedding), inv_freq_cis).astype("float32")

    embed = jnp.concatenate((freq, freq), axis=-1)
    return jnp.sin(embed)[:, :], jnp.cos(embed)[:, :]


def _rotate_half(x):
    return jnp.concatenate((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), axis=-1)


def apply_rotary_pos_embedding(tensor, sin_, cos_):
    return (tensor * cos_) + (_rotate_half(tensor) * sin_)


class FlaxFalconRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        _, seq_len, _ = query.shape

        query_expansion_factor = int(query.shape[0] / cos.shape[0])
        if query_expansion_factor > 1:
            query_cos = jnp.tile(cos, (query_expansion_factor,))
            query_sin = jnp.tile(sin, (query_expansion_factor,))
        else:
            query_cos, query_sin = cos, sin

        key_expansion_factor = int(key.shape[0] / cos.shape[0])
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
        self.w_qkv = nn.Dense(
            features=3 * self.config.hidden_size if not self.config.multi_query else (
                    self.config.hidden_size + 2 * head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.factor_scale = 1 / math.sqrt(head_dim)
        self.wo = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.head_dim = head_dim
        self.maybe_rotary = FlaxFalconRotaryEmbedding(self.dtype)
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
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
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
        if self.new_decoder_architecture:
            batch, seq_len, _ = qkv.shape
            qkv = qkv.reshape(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query_state = qkv[:, :, :, :-2]
            key_state = qkv[:, :, :, [-2]]
            value_state = qkv[:, :, :, [-1]]
            key_state = jnp.broadcast_to(key_state, query_state.shape)
            value_state = jnp.broadcast_to(value_state, query_state.shape)

            query_state, key_state, value_state = [x.reshape(x.shape[:-2] + (x.shape[:-2], x.shape[:-1])) for x in
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

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            alibi: chex.Array = None,
            freq_cis: chex.Array = None,
            output_attention_weight: bool = False,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        num_kv_heads = self.config.num_attention_heads if self.config.new_decoder_architecture else self.config.num
        query_layer, key_layer, value_layer = self.split_head(self.w_qkv(hidden_states))
        query_layer = transpose(
            query_layer, 1, 2
        ).reshape(
            batch_size * self.num_heads,
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
        query_state, key_state, value_state = self.apply_maybe_rotary(
            batch_size,
            sequence_length,
            query_state,
            key_state,
            value_state,
            freq_cis,
            position_ids
        )
        attn = jnp.einsum('...qhd,...khd->...hqk', query_state, key_state, precision=self.precision)
        if self.config.use_pjit_attention_force:
            attn = with_sharding_constraint(attn, PartitionSpec(("dp", "fsdp"), "mp", None, None))

        if alibi is not None:
            attn += alibi
        attn = attn * self.factor_scale

        if attention_mask is not None:
            attn += attention_mask

        attn = jax.nn.softmax(attn, axis=-1)
        attn = jnp.einsum('...hqk,...khd->...qhd', attn, value_state, precision=self.precision).reshape(
            (batch_size, sequence_length, -1))
        return self.wo(attn)


class FlaxFalconMlp(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.up = nn.Dense(
            features=self.config.hidden_size * 4,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )
        self.down = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias
        )

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


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
            self.post_attentionum_hidden_layersnorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
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
            freq_cis: chex.Array,
            position_ids: chex.Array
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output_attn = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            alibi=alibi,
            freq_cis=freq_cis,
            position_ids=position_ids
        )
        if not self.config.parallel_attn:
            residual = attn + residual
            hidden_states = self.post_attentionum_hidden_layersnorm(residual)

        mlp_out = self.mlp(hidden_states)
        if self.config.parallel_attn:
            mlp_out += attn
        return mlp_out + residual


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


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
                 alibi: chex.Array,
                 attention_mask: chex.Array,
                 freq_cis: chex.Array,
                 position_ids: chex.Array
                 ):
        for layer in self.layers:
            out = layer(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                alibi=alibi,
                freq_cis=freq_cis,
                position_ids=position_ids
            )
        return hidden_states


class FlaxFalconModule(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(
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
        if self.config.alibi:
            self.freq_cis = precompute_falcon_freq_cis(
                max_position_embedding=self.config.max_length or self.config.max_seq_len,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,

            )
        else:
            self.freq_cis = None

    def __call__(self,
                 input_ids: jnp.int32 = None,
                 attention_mask: Optional[chex.Array] = None,
                 position_ids: Optional[chex.Array] = None,
                 use_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = None,
                 ):
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = jnp.arange(0, seq_len).reshape(1, -1)
        hidden_states = self.wte(
            inputs=input_ids.astype(jnp.int32)
        )
        if attention_mask is None:
            attention_mask = jnp.ones(
                (batch, seq_len)
            )

        alibi = built_bloom_alibi(attention_mask, self.config
                                  .num_attention_heads).astype(hidden_states.dtype) if self.config.alibi else None
        causal_mask = nn.make_causal_mask(
            input_ids,
        )

        mv = jnp.finfo(hidden_states).min
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :]
        *_, dim = attention_mask.shape

        attention_mask += causal_mask[:, :, :dim, :dim]
        attention_mask = jnp.where(
            attention_mask == 2, 0, mv
        )
        out_layers = self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            freq_cis=self.freq_cis
        )
        output = self.ln_f()

        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=output,
            )
        else:
            return output,


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

    def __call__(self, input_ids,
                 attention_mask=None,
                 params: FrozenDict = None,
                 add_params_field: bool = False,
                 return_dict: bool = True):
        params = {'params': params or self.params} if add_params_field else params or self.params
        predict = self.module.apply(
            params,
            input_ids=jnp.asarray(input_ids, dtype=jnp.int32),
            attention_mask=jnp.asarray(attention_mask,
                                       dtype=jnp.int32) if attention_mask is not None else attention_mask,
            return_dict=return_dict
        )
        return predict

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        return {
            "attention_mask": attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
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

    def __call__(self, input_ids, attention_mask, position_ids, return_dict: bool = False):
        output = self.lm_head(
            self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True
            ).last_hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=output
            )
        else:
            return output,


class FlaxFalconForCausalLM(FlaxFalconPretrainedModel):
    module_class = FlaxFalconForCausalLMModule
