import math

from flax import linen as nn
from flax.serialization import to_bytes, from_bytes, to_state_dict, from_state_dict
from jax import grad, jit
from flax.core import FrozenDict
from typing import Optional, Dict, Union, Tuple
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax import numpy as jnp
import jax
from jax.interpreters import pxla
from jax.experimental.pjit import pjit, with_sharding_constraint as wsc
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
from jax.random import split, PRNGKey
from functools import partial
import flax
from einops import rearrange

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),

}


def get_names_from_parition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


class MptConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(self, d_model: int = 2048, n_heads: int = 16, n_layers: int = 24, expansion_ratio: int = 4,
                 max_seq_len: int = 2048, vocab_size: int = 50368, resid_prob_drop: float = 0.0,
                 emb_prob_drop: float = 0.0,
                 alibi: bool = True, use_bias: bool = True,
                 learned_pos_emb: bool = True, act_fn: str = 'gelu',
                 logit_scale: Optional[Union[float, str]] = None, no_bias: bool = False, verbose: int = 0,
                 embedding_fraction: float = 1.0, use_cache: bool = False, qk_ln: bool = True,
                 use_lm_head: bool = False,
                 use_norm_bias: bool = False, gradient_checkpointing: str = 'nothing_saveable',
                 use_pjit_attention_force: bool = False,
                 **kwargs):

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
        self.use_pjit_attention_force=use_pjit_attention_force
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
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(**kwargs)

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = False):
        return (

            ("transformer/wte/embedding", PartitionSpec("mp", "fsdp")),
            ("transformer/wpe/embedding", PartitionSpec("mp", "fsdp")),

            ("attn/w_qkv/kernel", PartitionSpec("fsdp", "mp")),
            ("attn/wo/kernel", PartitionSpec("mp", "fsdp")),
            ("attn/w_qkv/bias", PartitionSpec("fsdp", "mp")),
            ("attn/wo/bias", PartitionSpec("mp", "fsdp")),

            ("ffn/down/kernel", PartitionSpec("fsdp", "mp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "mp")),
            ("ffn/down/kernel", PartitionSpec("fsdp", "mp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "mp")),

            ("attention_norm/kernel", PartitionSpec(None)),
            ("norm_f/kernel", PartitionSpec(None)),
            ("norm_f/bias", PartitionSpec(None)),

            ("transformer/norm_f/kernel", PartitionSpec(None)),
            ("transformer/norm_f/bias", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp", "mp")),
            ("lm_head/bias", PartitionSpec("fsdp", "mp")),
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
        self.up = nn.Dense(self.config.d_model * self.config.expansion_ratio, kernel_init=jax.nn.initializers.normal(),
                           use_bias=self.config.use_bias,
                           dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.down = nn.Dense(self.config.d_model, kernel_init=jax.nn.initializers.normal(),
                             use_bias=self.config.use_bias,
                             dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.act = ACT2FN[self.config.act_fn]

    def __call__(self, x: jnp.DeviceArray):
        return self.down(self.act(self.up(x)))


class FlaxMptAttention(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.w_qkv = nn.Dense(self.config.d_model * 3, kernel_init=jax.nn.initializers.normal(),
                              use_bias=self.config.use_bias,
                              dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.wo = nn.Dense(self.config.d_model, kernel_init=jax.nn.initializers.normal(), use_bias=self.config.use_bias,
                           dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        if self.config.qk_ln:
            self.q_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
            self.k_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.causal_mask = nn.make_causal_mask(jnp.ones((1, self.config.max_seq_len)))

    def __call__(self, x, attn_bias=None, attention_mask=None):
        inp_shape = x.shape
        b, s, ds = inp_shape
        qkv = self.w_qkv(x)
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
        d = q.shape[-1]
        atw = jnp.einsum('...qhd,...khd->...hqk', q, k, precision=self.precision) * jax.lax.rsqrt(
            jnp.asarray(d).astype(v.dtype))
        if self.config.use_pjit_attention_force:
            atw = with_sharding_constraint(atw, PartitionSpec(('dp', 'fsdp'), 'mp', None, None))
        if attn_bias is not None:
            atw += attn_bias
        mv = jnp.finfo(atw).min
        mask = jnp.where(self.causal_mask == 1, 0, mv)
        if attention_mask is not None:
            attention_mask = jnp.where(attention_mask.reshape(b, 1, 1, s) == 1, 0, mv)
            atw += attention_mask
        atw += mask[:, :, :s, :s]
        atw = nn.softmax(atw, -1)
        atw = jnp.einsum('...hqk,...khd->...qhd', atw, v)
        return self.wo(atw.reshape(inp_shape))


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

    def __call__(self, x, attn_bias=None, attention_mask=None):
        x = self.attn(self.norm_1(x), attn_bias=attn_bias, attention_mask=attention_mask) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


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
                block, static_argnums=(),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
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

    def __call__(self, x, attn_bias=None, attention_mask=None):
        for block in self.blocks:
            x = block(x=x, attn_bias=attn_bias, attention_mask=attention_mask)
        return x


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

    def __call__(self, input_ids: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None, return_dict: bool = True):
        b, s = input_ids.shape
        hidden_state = self.wte(input_ids)
        if self.config.alibi:
            alibi = build_alibi(s, self.config.n_heads)
        else:
            pos_id = self.wpe(jnp.arange(s, dtype='i4').reshape(1, -1))
            hidden_state += pos_id
            alibi = None
        hidden_state = self.norm_f(self.h(hidden_state, attn_bias=alibi, attention_mask=attention_mask))
        if return_dict:
            return FlaxBaseModelOutput(last_hidden_state=hidden_state, hidden_states=None)
        else:
            return (hidden_state,)


class FlaxMptPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class: MptConfig = MptConfig

    def __init__(self, config, dtype: jnp.dtype = jnp.float32, param_dtype: jnp.dtype = jnp.float32,
                 _do_init: bool = False,
                 input_shape: Tuple = (1, 16), **kwargs):
        module = self.module_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype
        )
        super().__init__(_do_init=_do_init, config=config, input_shape=input_shape, module=module, **kwargs)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:

        if params is None:
            return self.module.init(
                rngs=rng,
                input_ids=jnp.ones(input_shape, dtype='i4'),
                attention_mask=jnp.ones(input_shape, dtype='i4'),

            )['params']
        else:
            return params

    def __call__(self, input_ids, attention_mask=None, params=None, add_params_field: bool = False,
                 return_dict: bool = True):
        params = {'params': params or self.params} if add_params_field else params or self.params
        predict = self.module.apply(
            params,
            input_ids=jnp.asarray(input_ids, dtype='i4'),
            attention_mask=jnp.asarray(attention_mask, dtype='i4') if attention_mask is not None else attention_mask,
            return_dict=return_dict
        )
        return predict

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


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
        if self.config.use_lm_head:
            self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(),
                                    use_bias=self.config.use_bias,
                                    dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)

    def __call__(self, input_ids: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None, return_dict: bool = True):
        predict: FlaxBaseModelOutput = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                                        return_dict=True)
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
