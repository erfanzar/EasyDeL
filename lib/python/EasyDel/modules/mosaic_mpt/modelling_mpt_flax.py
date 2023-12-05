import math

import einops
from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Union, Tuple, Sequence
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
import flax
from einops import rearrange
from fjformer.attention import efficient_attention
from ..flax_modelling_utils import (
    get_gradient_checkpoint_policy,
    with_sharding_constraint,
    ACT2FN,
    JaxBaseClassModel,
    smart_flash_attention
)
import chex
from fjformer.bits import config as q_config, q_flax


class MptConfig(PretrainedConfig, JaxBaseClassModel):
    model_type = 'mpt'

    def __init__(self,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 expansion_ratio: int = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50368,
                 resid_prob_drop: float = 0.0,
                 emb_prob_drop: float = 0.0,
                 alibi: bool = True,
                 use_bias: bool = False,
                 learned_pos_emb: bool = True,
                 act_fn: str = 'gelu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = False,
                 verbose: int = 0,
                 embedding_fraction: float = 1.0,
                 use_cache: bool = False,
                 qk_ln: bool = False,
                 use_lm_head: bool = False,
                 use_norm_bias: bool = False,
                 gradient_checkpointing: str = 'nothing_saveable',
                 use_pjit_attention_force: bool = False,
                 use_flash_attention: bool = False,
                 flash_attn_query_chunk_size: int = 1024,
                 flash_attn_key_chunk_size: int = 2048,
                 bits: Optional[int] = None,
                 axis_dims: Sequence[int] = (1, -1, 1, 1),
                 axis_names: Sequence[str] = ("dp", "fsdp",  "mp"),
                 **kwargs
                 ):

        self.d_model = d_model
        self.use_norm_bias = use_norm_bias
        self.use_lm_head = use_lm_head
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_prob_drop = resid_prob_drop
        self.use_bias = use_bias
        self.emb_prob_drop = emb_prob_drop
        self.use_pjit_attention_force = use_pjit_attention_force
        self.gradient_checkpointing = gradient_checkpointing
        self.learned_pos_emb = learned_pos_emb
        self.act_fn = act_fn
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.qk_ln = qk_ln
        self.alibi = alibi
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.use_cache = use_cache
        self.use_flash_attention = use_flash_attention
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.bits = bits

        self.from_pt = False
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(
            axis_dims=axis_dims,
            axis_names=axis_names,
            **kwargs
        )

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = False):
        return (

            ("transformer/wte/embedding", PartitionSpec("dp", "fsdp")),
            ("transformer/wpe/embedding", PartitionSpec("dp", "fsdp")),

            ("attn/w_qkv/kernel", PartitionSpec("fsdp", "dp")),
            ("attn/wo/kernel", PartitionSpec("dp", "fsdp")),
            ("attn/w_qkv/bias", PartitionSpec("fsdp", "dp")),
            ("attn/wo/bias", PartitionSpec("dp", "fsdp")),

            ("ffn/down/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/down/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "dp")),

            ("attention_norm/kernel", PartitionSpec(None)),
            ("norm_f/kernel", PartitionSpec(None)),
            ("norm_f/bias", PartitionSpec(None)),

            ("transformer/norm_f/kernel", PartitionSpec(None)),
            ("transformer/norm_f/bias", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp", "dp")),
            ("lm_head/bias", PartitionSpec("fsdp", "dp")),
            ('.*', PartitionSpec(None)),
        ) if not fully_fsdp else (

            ("transformer/wte/embedding", PartitionSpec("fsdp")),
            ("transformer/wpe/embedding", PartitionSpec("fsdp")),

            ("attn/w_qkv/kernel", PartitionSpec("fsdp")),
            ("attn/wo/kernel", PartitionSpec("fsdp")),
            ("attn/w_qkv/bias", PartitionSpec("fsdp")),
            ("attn/wo/bias", PartitionSpec("fsdp")),

            ("ffn/down/kernel", PartitionSpec("fsdp")),
            ("ffn/up/kernel", PartitionSpec("fsdp")),
            ("ffn/down/kernel", PartitionSpec("fsdp")),
            ("ffn/up/kernel", PartitionSpec("fsdp")),

            ("attention_norm/kernel", PartitionSpec(None)),
            ("norm_f/kernel", PartitionSpec(None)),
            ("norm_f/bias", PartitionSpec(None)),

            ("transformer/norm_f/kernel", PartitionSpec(None)),
            ("transformer/norm_f/bias", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp")),
            ("lm_head/bias", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None)),
        )

    def add_jax_args(self,
                     d_model: int = 2048,
                     n_heads: int = 16,
                     n_layers: int = 24,
                     expansion_ratio: int = 4,
                     max_seq_len: int = 2048,
                     vocab_size: int = 50368,
                     resid_prob_drop: float = 0.0,
                     emb_prob_drop: float = 0.0,
                     alibi: bool = True,
                     use_bias: bool = True,
                     learned_pos_emb: bool = True,
                     act_fn: str = 'gelu',
                     logit_scale: Optional[Union[float, str]] = None,
                     no_bias: bool = False,
                     verbose: int = 0,
                     embedding_fraction: float = 1.0,
                     use_cache: bool = False,
                     qk_ln: bool = True,
                     use_lm_head: bool = False,
                     use_norm_bias: bool = False,
                     gradient_checkpointing: str = 'nothing_saveable',
                     use_pjit_attention_force: bool = False,
                     use_flash_attention: bool = False,
                     flash_attn_query_chunk_size: int = 1024,
                     flash_attn_key_chunk_size: int = 2048,
                     bits: Optional[int] = None,
                     axis_dims: Sequence[int] = (1, -1, 1, 1),
                     axis_names: Sequence[str] = ("dp", "fsdp",  "mp"),
                     q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp", None, "mp"),
                     k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp", None, "mp"),
                     v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp", None, "mp"),
                     b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", None, "fsdp", None),
                     a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp", None, "mp"),
                     backend: Optional[str] = None,
                     **kwargs,
                     ):
        self.axis_names = axis_names
        self.axis_dims = axis_dims
        self.q_ps = q_ps
        self.k_ps = k_ps
        self.v_ps = v_ps
        self.b_ps = b_ps
        self.a_ps = a_ps
        self.backend = backend
        if hasattr(self, 'attn_config'):
            for k, v in self.attn_config.items():
                setattr(self, k, v)
        basics = dict(
            bits=bits,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion_ratio=expansion_ratio,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            resid_prob_drop=resid_prob_drop,
            emb_prob_drop=emb_prob_drop,
            alibi=alibi,
            use_bias=use_bias,
            learned_pos_emb=learned_pos_emb,
            act_fn=act_fn,
            logit_scale=logit_scale,
            no_bias=no_bias,
            verbose=verbose,
            embedding_fraction=embedding_fraction,
            use_cache=use_cache,
            qk_ln=qk_ln,
            use_lm_head=use_lm_head,
            use_norm_bias=use_norm_bias,
            gradient_checkpointing=gradient_checkpointing,
            use_pjit_attention_force=use_pjit_attention_force,
            use_flash_attention=use_flash_attention,
            flash_attn_query_chunk_size=flash_attn_query_chunk_size,
            flash_attn_key_chunk_size=flash_attn_key_chunk_size,
            **kwargs
        )
        for k, v in basics.items():
            if not hasattr(self, k):
                print(f' Key {k} not found in loaded config setting that to default of {v}')
                setattr(self, k, v)

        self.from_pt = False


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
        x = x.astype(jnp.promote_types(self.dtype, jnp.bfloat16))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxMptMLP(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        self.up = nn.Dense(
            self.config.d_model * self.config.expansion_ratio,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=dot_general_cls
        )
        self.down = nn.Dense(
            self.config.d_model,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=dot_general_cls
        )
        self.act = ACT2FN[self.config.act_fn]

    def __call__(self, hidden_states: chex.Array):
        return self.down(self.act(self.up(hidden_states)))


class FlaxMptAttention(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:

        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        self.w_qkv = nn.Dense(
            self.config.d_model * 3,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dot_general=dot_general_cls,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision)
        self.wo = nn.Dense(
            self.config.d_model,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=dot_general_cls
        )
        if self.config.qk_ln:
            self.q_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
            self.k_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.causal_mask = nn.make_causal_mask(jnp.ones((1, self.config.max_seq_len)))

    @nn.compact
    def _concatenate_to_cache(self, key, query, value, attention_mask):
        is_initialized = self.has_variable('cache', 'key')
        cache_key = self.variable('cache', 'key', jnp.zeros, key.shape, key.dtype)
        cache_value = self.variable('cache', 'value', jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable('cache', 'index', lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            *b, s, h, d = cache_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(b) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cache_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cache_value.value, value, indices)
            cache_value.value = value
            cache_key.value = key
            num_updated_vector = query.shape[1]
            cache_index.value = cache_index.value + num_updated_vector
            pad_mask = jnp.broadcast_to(
                jnp.arange(s) < cur_index + num_updated_vector,
                tuple(b) + (1, num_updated_vector, s),
            )

            attention_mask = nn.combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 position_ids: chex.Array,
                 attn_bias: chex.Array = None,
                 init_cache: bool = False
                 ):

        """
        The __call__ function is the main function of a JAX module.
        It takes in inputs and returns outputs, just like any other Python function.
        The difference is that __call__ can also take in state (e.g., parameters) from the module itself,
        and it can update that state as part of its computation.

        :param self: Access variables that belong to the class
        :param hidden_states: chex.Array: Pass the input to the attention layer
        :param attention_mask: chex.Array: Mask out certain positions in the sequence
        :param position_ids: chex.Array: Specify the position of each token in the sequence
        :param attn_bias: chex.Array: Add a bias to the attention scores
        :param init_cache: bool: Initialize the cache
        :return: The output of the attention layer
        
        """
        inp_shape = hidden_states.shape
        b, s, ds = inp_shape
        qkv = self.w_qkv(hidden_states)
        q, k, v = jnp.split(qkv, 3, -1)
        if self.config.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        if self.config.use_pjit_attention_force:
            q = with_sharding_constraint(q, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            k = with_sharding_constraint(k, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            v = with_sharding_constraint(v, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.n_heads)
        attention_mask = attention_mask.reshape(b, 1, 1, -1)
        if self.has_variable('cache', 'key') or init_cache:
            k, v, attention_mask = self._concatenate_to_cache(key=k, value=v, query=q, attention_mask=attention_mask)

        q_l = q.shape[1]
        k_l = k.shape[1]
        dropout_rng = None
        deterministic = False
        if deterministic:
            dropout_rng = self.make_rng("dropout")

        if self.config.use_flash_attention and not (self.has_variable("cache", "cached_key") or init_cache):

            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            if attention_mask.shape[1] != self.config.num_attention_heads:
                attention_mask = attention_mask.repeat(self.config.num_attention_heads, 1, )
            attention_bias = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            rtp_axis = (0, 2, 1, 3)
            attn_output = smart_flash_attention(
                q=jnp.transpose(q, rtp_axis),
                k=jnp.transpose(k, rtp_axis),
                v=jnp.transpose(b, rtp_axis),
                bias=attention_bias + attn_bias,
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
            d = q.shape[-1]
            attn_output = jnp.einsum('...qhd,...khd->...hqk', q, k, precision=self.precision) * jax.lax.rsqrt(
                jnp.asarray(d).astype(v.dtype))
            if self.config.use_pjit_attention_force:
                attn_output = with_sharding_constraint(attn_output, PartitionSpec(('dp', 'fsdp'), 'mp', None, None))
            if attn_bias is not None:
                attn_output += attn_bias[:, :, :, :attn_output.shape[-1]]
            mask = jnp.where(self.causal_mask == 1, 0, jnp.finfo(attn_output).min)
            if attention_mask is not None:
                attention_mask = jnp.where(
                    attention_mask == 1,
                    0,
                    jnp.finfo(attn_output).min
                )
                attn_output += attention_mask
            attn_output += mask[:, :, :attn_output.shape[-2], :attn_output.shape[-1]]
            attn_output = nn.softmax(attn_output, -1)
            attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_output, v)
        return self.wo(attn_output.reshape(inp_shape))


class FlaxMptBlock(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.norm_1 = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.norm_2 = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.attn = FlaxMptAttention(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                                     precision=self.precision)
        self.ffn = FlaxMptMLP(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                              precision=self.precision)

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 position_ids: chex.Array,
                 attn_bias: chex.Array = None,
                 init_cache: bool = False
                 ):
        hidden_states = (
                self.attn(
                    self.norm_1(hidden_states),
                    attn_bias=attn_bias,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    init_cache=init_cache
                ) + hidden_states
        )
        hidden_states = self.ffn(self.norm_2(hidden_states)) + hidden_states
        return hidden_states


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


class FlaxMptCollection(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = FlaxMptBlock

        if self.config.gradient_checkpointing != '':
            block = flax.linen.remat(
                block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(5,)
            )

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

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 position_ids: chex.Array,
                 attn_bias: chex.Array = None,
                 init_cache: bool = False
                 ):
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                position_ids=position_ids,
                init_cache=init_cache
            )
        return hidden_states


def build_alibi(max_length, num_attention_heads, alibi_max: int = 8):
    w_range = jnp.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    # cp2 = jnp.power(2, jnp.ceil(jnp.log2(num_attention_heads)))
    cp2 = 2 ** math.ceil(math.log2(num_attention_heads))
    h_range = jnp.arange(1, 1 + num_attention_heads, ).reshape(1, -1, 1, 1)
    h_range = jnp.matmul(h_range, jnp.asarray(alibi_max / cp2).reshape(1, 1))
    slop = 1 / jnp.power(2, h_range)
    if cp2 != num_attention_heads:
        slop = jnp.concatenate([slop[1::2], slop[::2]], axis=-1)[:num_attention_heads]
    alibi = (w_range * slop).reshape(1, num_attention_heads, 1, max_length)
    return alibi


class FlaxMptModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model)
        if not self.config.alibi:
            self.wpe = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.max_seq_len)
        self.h = FlaxMptCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_f = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.alibi = build_alibi(self.config.max_seq_len, self.config.n_heads)

    def __call__(self,

                 input_ids: chex.Array,
                 attention_mask: chex.Array = None,
                 position_ids: chex.Array = None,
                 init_cache: bool = False,
                 return_dict: bool = True,
                 extra_embedding: Optional[Union[jnp.ndarray, None]] = None
                 ):
        b, s = input_ids.shape
        hidden_states = self.wte(input_ids)
        hidden_states = hidden_states + extra_embedding if extra_embedding is not None else hidden_states

        if self.config.alibi:
            alibi = self.alibi
        else:
            pos_id = self.wpe(jnp.arange(s, dtype='i4').reshape(1, -1))
            hidden_states += pos_id
            alibi = None
        hidden_states = self.norm_f(
            self.h(
                hidden_states,
                attn_bias=alibi,
                attention_mask=attention_mask,
                position_ids=position_ids,
                init_cache=init_cache
            )
        )
        if return_dict:
            return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=None)
        else:
            return (hidden_states,)


class FlaxMptPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class: MptConfig = MptConfig

    def __init__(self,
                 config,
                 dtype: jnp.dtype = jnp.float32,
                 param_dtype: jnp.dtype = jnp.float32,
                 _do_init: bool = False,
                 precision: Optional[Union[jax.lax.Precision, None]] = jax.lax.Precision('fastest'),
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

        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
            init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.ones(input_shape, dtype='i4')
        if params is None:
            return self.module.init(
                rngs=rng,
                input_ids=input_ids,
                attention_mask=jnp.ones(input_shape, dtype='i4'),
                position_ids=jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape),
                init_cache=False
            )['params']
        else:
            return params

    def __call__(self,
                 input_ids,
                 attention_mask=None,
                 past_key_values=None,
                 position_ids=None,
                 init_cache: bool = False,
                 params: dict = None,
                 add_params_field: bool = False,
                 return_dict: bool = True,
                 extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
                 ):
        params = {'params': params or self.params} if add_params_field else params or self.params
        input_ids = jnp.asarray(input_ids, dtype='i4')
        mutable = False
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype='i4')
        if position_ids is None:
            position_ids = jnp.arange(0, attention_mask.shape[-1], 1, dtype='i4').reshape(
                1, -1
            ).repeat(input_ids.shape[0], axis=0)

        if past_key_values is not None:
            params['cache'] = past_key_values
            mutable = ['cache']
        rngs = {'params': jax.random.key(0)}
        predict = self.module.apply(
            params,
            input_ids=input_ids,
            attention_mask=jnp.asarray(attention_mask, dtype='i4'),
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            position_ids=position_ids,
            init_cache=init_cache,
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


class FlaxFlaxMptForCausalLMModule(nn.Module):
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
        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        if self.config.use_lm_head:
            self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(),
                                    use_bias=self.config.use_bias,
                                    dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
                                    dot_general=dot_general_cls)

    def __call__(self,
                 input_ids: chex.Array,
                 attention_mask: chex.Array = None,
                 init_cache: bool = False,
                 position_ids: chex.Array = None,
                 return_dict: bool = True,
                 extra_embedding: Optional[Union[jnp.ndarray, None]] = None,

                 ):
        predict: FlaxBaseModelOutput = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            extra_embedding=extra_embedding,
            position_ids=position_ids,
            init_cache=init_cache
        )
        if self.config.use_lm_head:
            logits = self.lm_head(predict.last_hidden_state)
        else:
            logits = predict.last_hidden_state @ self.transformer.wte.embedding.T
        if return_dict:

            return FlaxCausalLMOutput(
                logits=logits,
                hidden_states=predict.last_hidden_state
            )
        else:
            return (logits,)


class FlaxMptForCausalLM(FlaxMptPretrainedModel):
    module_class = FlaxFlaxMptForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):

        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(
            batch_size, max_length
        )
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
