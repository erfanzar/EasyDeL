from flax import linen as nn
from flax.serialization import to_bytes, from_bytes, to_state_dict, from_state_dict
from jax import grad, jit
from flax.core import FrozenDict
from typing import Optional, Dict, Union, Tuple
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax import numpy as jnp
import jax
from jax.experimental.pjit import pjit, PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
from jax.random import split, PRNGKey
from functools import partial
from einops import rearrange

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),

}


class MptConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(self, d_model: int = 2048, n_heads: int = 16, n_layers: int = 24, expansion_ratio: int = 4,
                 max_seq_len: int = 2048, vocab_size: int = 50368, resid_prob_drop: float = 0.0,
                 emb_prob_drop: float = 0.0,
                 alibi: bool = True, use_bias: bool = True,
                 learned_pos_emb: bool = True, act_fn: str = 'gelu',
                 logit_scale: Optional[Union[float, str]] = None, no_bias: bool = False, verbose: int = 0,
                 embedding_fraction: float = 1.0, use_cache: bool = False, qk_ln: bool = True,
                 **kwargs):

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_prob_drop = resid_prob_drop
        self.emb_prob_drop = emb_prob_drop
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
        self._validate_config()

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules():
        return (
            ('mlp.up.kernel', PartitionSpec('fsdp', 'mp')),
            ('mlp.up.bias', PartitionSpec('fsdp', 'mp')),
            ('mlp.down.kernel', PartitionSpec('fsdp', 'mp')),
            ('mlp.down.bias', PartitionSpec('fsdp', 'mp'))
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


class MptMLP(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

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


class MptAttention(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

    def setup(self) -> None:
        self.w_qkv = nn.Dense(self.config.d_model * 3, kernel_init=jax.nn.initializers.normal(),
                              use_bias=self.config.use_bias,
                              dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.wo = nn.Dense(self.config.d_model, kernel_init=jax.nn.initializers.normal(), use_bias=self.config.use_bias,
                           dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        if self.config.qk_ln:
            self.q_ln = nn.LayerNorm()
            self.k_ln = nn.LayerNorm()
        self.causal_mask = nn.make_causal_mask(jnp.ones((1, self.config.max_seq_len)))

    def __call__(self, x, attn_bias=None, attention_mask=None):
        inp_shape = x.shape
        b, s, ds = inp_shape
        qkv = self.w_qkv(x)
        q, k, v = jnp.split(qkv, 3, -1)
        if self.config.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.n_heads)
        d = q.shape[-1]
        atw = jnp.einsum('...qhd,...khd->...hqk', q, k, precision=self.precision) * jax.lax.rsqrt(d)
        if attn_bias is not None:
            atw += attn_bias
        mv = jnp.finfo(atw).min
        mask = jnp.where(self.causal_mask == 1, 0, mv)
        if attention_mask is not None:
            attention_mask = jnp.where(attention_mask.reshape(b, 1, 1, s) == 1, 0, mv)
            atw += attention_mask
        atw += mask
        atw = nn.softmax(atw, -1)
        atw = jnp.einsum('...hqk,...khd->...qhd', atw, v)
        return self.wo(atw.reshape(inp_shape))


class MptBlock(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

    def setup(self) -> None:
        self.norm_1 = nn.LayerNorm()
        self.norm_2 = nn.LayerNorm()
        self.attn = MptAttention(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                                 precision=self.precision)
        self.ffn = MptMLP(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                          precision=self.precision)

    def __call__(self, x, attn_bias=None, attention_mask=None):
        x = self.attn(self.norm_1(x), attn_bias=attn_bias, attention_mask=attention_mask) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class MptCollection(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

    def setup(self) -> None:
        self.blocks = [
            MptBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            for _ in range(
                self.config.n_layers
            )
        ]

    def __call__(self, x, attn_bias=None, attention_mask=None):
        for block in self.blocks:
            x = block(x=x, attn_bias=attn_bias, attention_mask=attention_mask)
        return x


def build_alibi(max_length, num_attention_heads, alibi_max: int = 8):
    w_range = jnp.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    cp2 = jnp.power(2, jnp.ceil(jnp.log2(num_attention_heads)))
    h_range = jnp.arange(1, 1 + num_attention_heads, ).reshape(1, -1, 1, 1)
    h_range = jnp.matmul(h_range, alibi_max / cp2)
    slop = 1 / jnp.power(2, h_range)
    if cp2 != num_attention_heads:
        slop = jnp.concatenate([slop[1::2], slop[::2]], axis=-1)[:num_attention_heads]
    alibi = (w_range * slop).reshape(1, num_attention_heads, 1, max_length)
    return alibi


class MptModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

    def setup(self) -> None:
        self.wte = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model)
        if not self.config.alibi:
            self.wpe = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.max_seq_len)
        self.h = MptCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_f = nn.LayerNorm()

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


class MptPretrainedModel(FlaxPreTrainedModel):
    module: nn.Module = None
    config_class: MptConfig = MptConfig

    def __init__(self, config, _do_init: bool = False):
        super().__init__(_do_init=_do_init, config=config, module=self.module)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        if params is None:
            return self.module.init(
                rng,
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        return predict


class MptModel(MptPretrainedModel):
    module = MptModule


class MptForCausalLMModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None

    def setup(self) -> None:
        self.transformer = MptModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(),
                                use_bias=self.config.use_bias,
                                dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)

    def __call__(self, input_ids: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None, return_dict: bool = True):
        predict: FlaxBaseModelOutput = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                                        return_dict=True)
        logits = self.lm_head(predict.last_hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=logits,
                hidden_states=predict.last_hidden_state
            )
        else:
            return (logits,)


class MptForCausalLM(MptPretrainedModel):
    module = MptForCausalLMModule
