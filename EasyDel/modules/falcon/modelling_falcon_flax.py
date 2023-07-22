import math

from flax import linen as nn
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


class FalconConfig(PretrainedConfig):
    model_type = "falcon"
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
            self,
            vocab_size=250880,
            hidden_size=64,
            n_layer=2,
            n_head=8,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            apply_residual_connection_post_layernorm=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            multi_query=False,
            alibi=False,
            bias=False,
            parallel_attn=False,
            max_seq_len=2048,
            use_pjit_attention_force: bool = False,
            gradient_checkpointing: str = 'nothing_saveable',
            **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
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

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        return not self.alibi

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = False):
        return (
            ('wte/embedding', PartitionSpec('fsdp', 'mp')),
            ('self_attention/w_qkv/(kernel|bias)', PartitionSpec('fsdp', 'mp')),
            ('self_attention/wo/(kernel|bias)', PartitionSpec('fsdp', 'mp')),
            ('mlp/down/(kernel|bias)', PartitionSpec('fsdp', 'mp')),
            ('mlp/up/(kernel|bias)', PartitionSpec('mp', 'fsdp')),
            ('lm_head/kernel', PartitionSpec('fsdp', 'mp')),
            ('transformer/ln_f/bias', PartitionSpec('fsdp', 'mp')),
            ('transformer/ln_f/scale', PartitionSpec('fsdp', 'mp')),
            ('transformer/post_attention_layernorm/scale', PartitionSpec('mp', 'fsdp')),
            ('transformer/post_attention_layernorm/bias', PartitionSpec('mp', 'fsdp')),
            ('.*', PartitionSpec('fsdp', 'mp'))
        ) if not fully_fsdp else (
            ('wte/embedding', PartitionSpec('fsdp')),
            ('self_attention/w_qkv/(kernel|bias)', PartitionSpec('fsdp')),
            ('self_attention/wo/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/down/(kernel|bias)', PartitionSpec('fsdp')),
            ('mlp/up/(kernel|bias)', PartitionSpec('fsdp')),
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


def built_bloom_alibi(attention_mask, num_attention_heads):
    b, s = attention_mask.shape
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
    return alibi.reshape(b, num_attention_heads, 1, s)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                         dtype: jnp.dtype = jnp.bfloat16) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs).astype(dtype)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
        xq: jnp.ndarray,
        xk: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class FlaxFalconAttention(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        head_dim = self.config.hidden_size // self.config.n_head
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
        assert self.head_dim * self.config.n_head == self.config.hidden_size
        if not self.config.alibi:
            self.freq = precompute_freqs_cis(head_dim, self.config.max_seq_len, dtype=self.dtype)

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 alibi: jnp.DeviceArray = None,
                 attention_mask: jnp.DeviceArray = None,
                 ):
        b, s, d = hidden_states.shape

        qkv = self.w_qkv(hidden_states)
        if not self.config.multi_query:
            q, k, v = jnp.split(qkv, 3, -1)
            if self.config.use_pjit_attention_force:
                q = with_sharding_constraint(q, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
                k = with_sharding_constraint(k, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
                v = with_sharding_constraint(v, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.n_head)
            q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.n_head)
            v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.n_head)
        else:
            qkv = qkv.reshape(
                b, s, self.config.n_head + 2, -1
            )
            q, k, v = qkv[..., :-2, :], qkv[..., [-2], :], qkv[..., [-1], :]
            if self.config.use_pjit_attention_force:
                q = with_sharding_constraint(q, PartitionSpec(('dp', 'fsdp'), None, None, 'mp'))
                k = with_sharding_constraint(k, PartitionSpec(('dp', 'fsdp'), None, None, 'mp'))
                v = with_sharding_constraint(v, PartitionSpec(('dp', 'fsdp'), None, None, 'mp'))

        if not self.config.alibi:
            freq = self.freq[:s].reshape(1, s, -1)
            q, k = apply_rotary_emb(q, k, freq, self.dtype)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k, precision=self.precision)
        if self.config.use_pjit_attention_force:
            attn = with_sharding_constraint(attn, PartitionSpec(("dp", "fsdp"), "mp", None, None))

        if alibi is not None:
            attn += alibi
        attn = attn * self.factor_scale

        if attention_mask is not None:
            attn += attention_mask

        attn = jax.nn.softmax(attn, axis=-1)
        attn = jnp.einsum('...hqk,...khd->...qhd', attn, v, precision=self.precision).reshape((b, s, d))
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
            self.post_attention_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
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

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 alibi: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray,
                 ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            alibi=alibi
        )
        if not self.config.parallel_attn:
            residual = attn + residual
            hidden_states = self.post_attention_layernorm(residual)

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
        self.blocks = [
            block(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(
                self.config.n_layer
            )
        ]

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 alibi: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray,

                 ):
        for b in self.blocks:
            hidden_states = b(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                alibi=alibi
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

    def __call__(self,
                 input_ids: jnp.int32 = None,
                 attention_mask: Optional[jnp.DeviceArray] = None,
                 use_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = None,
                 ):
        batch, seq_len = input_ids.shape
        hidden_states = self.wte(
            inputs=input_ids.astype(jnp.int32)
        )
        if attention_mask is None:
            attention_mask = jnp.ones(
                (batch, seq_len)
            )

        alibi = built_bloom_alibi(attention_mask, self.config
                                  .n_head).astype(hidden_states.dtype) if self.config.alibi else None
        causal_mask = nn.make_causal_mask(
            input_ids,
        )

        mv = jnp.finfo(hidden_states).min
        attention_mask = jnp.where(attention_mask == 1, 0, mv) + jnp.where(causal_mask == 1, 0, mv)

        causal_mask += attention_mask
        output = self.ln_f(self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            alibi=alibi
        ))

        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=output,
            )
        else:
            return output,


class FlaxFalconPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class = FalconConfig

    def __init__(self, config, _do_init=False, dtype: jnp.dtype = jnp.float32, param_dtype: jnp.dtype = jnp.float32,
                 input_shape: Tuple = (1, 12)):
        module = self.module_class(config=config, dtype=dtype, param_dtype=param_dtype)
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

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


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

    def __call__(self, input_ids, attention_mask, return_dict: bool = False):
        output = self.lm_head(self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
