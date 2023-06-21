import math

from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Dict, Union, Tuple
from transformers import FlaxPreTrainedModel, PretrainedConfig, GPTNeoXForCausalLM
from jax import numpy as jnp
import jax
from jax.interpreters import pxla
from jax.experimental.pjit import pjit, PartitionSpec, with_sharding_constraint as wsc
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


def with_sharding_constraint(x, partition_specs):
    def names_in_mesh(*names):
        return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)

    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


class GPTNeoXConfig(PretrainedConfig):
    model_type = "gpt_neox"

    def __init__(
            self,
            vocab_size=50432,
            hidden_size=6144,
            num_hidden_layers=44,
            num_attention_heads=64,
            intermediate_size=24576,
            hidden_act="gelu",
            rotary_pct=0.25,
            rotary_emb_base=10000,
            classifier_dropout=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
            gradient_checkpointing='everything_saveable',
            use_parallel_residual=True,
            **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.use_parallel_residual = use_parallel_residual

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


class FlaxGPTNeoXAttention(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.head_size = self.config.hidden_size // self.config.num_attention_heads
        self.freq_cis = precompute_freqs_cis(
            dtype=self.dtype,
            dim=self.head_size,
            end=self.config.max_position_embeddings
        )
        self.w_qkv = nn.Dense(
            3 * self.config.hidden_size
        )
        self.w_o = nn.Dense(
            self.config.hidden_size
        )

        self.factor = jnp.sqrt(jnp.asarray(self.head_size, dtype=jnp.float32))
        self.bias = nn.make_causal_mask(jnp.ones((1, self.config.max_position_embeddings)))

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray = None,
                 ):
        b, s, d = hidden_states.shape
        q, k, v = jnp.split(self.w_qkv(hidden_states), indices_or_sections=3, axis=-1)
        freq = self.freq_cis[:s].reshape(1, s, -1)
        q = with_sharding_constraint(q, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
        k = with_sharding_constraint(k, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
        v = with_sharding_constraint(v, PartitionSpec(('dp', 'fsdp'), None, 'mp'))

        q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)
        bias = jnp.where(self.bias == 1, 0, jnp.finfo(
            hidden_states.dtype
        ).min
                         )
        q, k = apply_rotary_emb(q, k, freqs_cis=freq, dtype=self.dtype)

        attn = jnp.einsum(
            '...qhd,...khd->...hqk', q, k, precision=self.precision
        ) * self.factor
        attn = attn + bias[:, :, :s, :s]
        if attention_mask is not None:
            attn += attention_mask
        attn = jax.nn.softmax(attn, axis=-1)
        attn = with_sharding_constraint(attn, PartitionSpec(('dp', 'fsdp'), 'mp', None, None))
        attn = jnp.einsum('...hqk,..khd->qhd', attn, v, precision=self.precision)
        attn = self.w_o(attn.reshape(b, s, d))


class FlaxGPTNeoXMlp(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        ...

    def __call__(self, x):
        ...


class FlaxGPTNeoXBlock(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        ...

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray,
                 ):
        ...


class FlaxGPTNeoXCollection(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        ...

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray,

                 ):
        ...


class FlaxGPTNeoXModule(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        ...

    def __call__(self,
                 input_ids: jnp.int32 = None,
                 attention_mask: Optional[jnp.DeviceArray] = None,
                 return_dict: Optional[bool] = None,
                 ):
        ...


class FlaxGPTNeoXPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class = GPTNeoXConfig

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


class FlaxGPTNeoXModel(FlaxGPTNeoXPretrainedModel):
    module_class = FlaxGPTNeoXModule


class FlaxGPTNeoXForCausalLMModule(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        ...

    def __call__(self, input_ids, attention_mask, return_dict: bool = False):
        ...


class FlaxGPTNeoXForCausalLM(FlaxGPTNeoXPretrainedModel):
    module_class = FlaxGPTNeoXForCausalLMModule
