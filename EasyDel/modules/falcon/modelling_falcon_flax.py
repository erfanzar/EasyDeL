import math

from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Dict, Union, Tuple
from transformers import FlaxPreTrainedModel, PretrainedConfig
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
            **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
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
        ...

    @staticmethod
    def get_mesh_names():
        return 'dp', 'fsdp', 'mp'


def build_alibi(max_length, num_attention_heads, alibi_max: int = 8):
    w_range = jnp.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    cp2 = 2 ** math.ceil(math.log2(num_attention_heads))
    h_range = jnp.arange(1, 1 + num_attention_heads, ).reshape(1, -1, 1, 1)
    h_range = jnp.matmul(h_range, jnp.asarray(alibi_max / cp2).reshape(1, 1))
    slop = 1 / jnp.power(2, h_range)
    if cp2 != num_attention_heads:
        slop = jnp.concatenate([slop[1::2], slop[::2]], axis=-1)[:num_attention_heads]
    alibi = (w_range * slop).reshape(1, num_attention_heads, 1, max_length)
    return alibi


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
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconMlp(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconBlock(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconCollection(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconModule(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    config_class = FalconConfig

    def __init__(self, config, _do_init=False, dtype: jnp.dtype = jnp.float32, param_dtype: jnp.dtype = jnp.float32,
                 input_shape: Tuple = (1, 26)):
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


class FlaxFalconModel(FlaxFalconPretrainedModel):
    module_class = FlaxFalconModule


class FlaxFalconForCausalLMModule(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class FlaxFalconForCausalLM(FlaxFalconPretrainedModel):
    module_class = FlaxFalconForCausalLMModule
