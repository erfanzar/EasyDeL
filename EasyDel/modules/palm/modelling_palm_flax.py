from typing import Union, Optional, Tuple, Any, Mapping

import jax
import jax.numpy as jnp
import numpy as onp
import transformers.modeling_flax_outputs
from einops import rearrange
import flax.linen as nn
from flax.core import FrozenDict

from jax import numpy as np
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from transformers import PretrainedConfig
from jax.experimental.pjit import with_sharding_constraint as wsc
from jax.sharding import PartitionSpec
from jax.interpreters import pxla


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


class PalmConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size: Optional[int] = 32000,
                 hidden_size: Optional[int] = 4096,
                 dim_head: Optional[int] = None,
                 num_hidden_layers: Optional[int] = 32,
                 num_attention_heads: Optional[int] = 32,
                 up_inner_dim: Optional[int] = 4,
                 eps: Optional[float] = 1e-5,
                 max_length: int = 8196,  # Easydel trained palm with length of 8196
                 bos_token_id: int = 0,
                 eos_token_id: int = 1,
                 gradient_checkpointing='nothing_saveable',
                 use_pjit_attention_force: bool = False,
                 use_tie_word_embedding: bool = True
                 ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        dim_head = dim_head if dim_head is not None else hidden_size // num_attention_heads
        self.dim_head = dim_head
        self.up_inner_dim = up_inner_dim
        self.use_pjit_attention_force = use_pjit_attention_force
        self.gradient_checkpointing = gradient_checkpointing
        self.num_attention_heads = num_attention_heads
        self.use_tie_word_embedding = use_tie_word_embedding
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eps = eps
        self.max_length = max_length

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = False):
        return (
            ('wi/kernel', PartitionSpec('fsdp')),
            ('attn_wo/kernel', PartitionSpec('fsdp', 'mp')),
            ('ff_wo/kernel', PartitionSpec('fsdp', 'mp')),
            ('wte/embedding', PartitionSpec('fsdp', 'mp')),
            ('lm_head/kernel', PartitionSpec('fsdp')),
            ('post_norm/kernel', PartitionSpec('fsdp')),
            ('norm/kernel', PartitionSpec('fsdp', 'mp')),
            ('.*', PartitionSpec(None)),
        ) if not fully_fsdp else (
            ('wi/kernel', PartitionSpec('fsdp')),
            ('attn_wo/kernel', PartitionSpec('fsdp')),
            ('ff_wo/kernel', PartitionSpec('fsdp')),
            ('wte/embedding', PartitionSpec('fsdp')),
            ('lm_head/kernel', PartitionSpec('fsdp')),
            ('post_norm/kernel', PartitionSpec('fsdp')),
            ('norm/kernel', PartitionSpec('fsdp')),
            ('.*', PartitionSpec('fsdp')),
        )


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        return hidden_state * jax.lax.rsqrt(jnp.square(hidden_state).mean(-1, keepdims=True) + self.eps)

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        hidden_state = hidden_state.astype(jnp.promote_types(self.dtype, jnp.bfloat16))
        output = self._norm(hidden_state).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def pre_compute_freq_cis(dim, max_length, theta: int = 10000.0, dtype=jnp.bfloat16):
    freq_cis = 1 / (theta ** (jnp.arange(0, dim, 2).astype(dtype=dtype) / dim))
    length = jnp.arange(max_length)
    cis = jnp.outer(length, freq_cis).astype(dtype)
    sin = jnp.sin(cis)
    cos = jnp.cos(cis)
    freq_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freq_cis)


def apply_rotary_embedding(xq, xk, freq_cis, dtype=jnp.bfloat16):
    reshape_xq = xq.astype(jnp.flaot32).reshape(xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.flaot32).reshape(xk.shape[:-1], -1, 2)

    complex_q = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    complex_k = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freq_cis = freq_cis.reshape(*freq_cis[:2], 1, *freq_cis[2:])
    xq = complex_q * freq_cis
    xk = complex_k * freq_cis
    xq = jnp.stack([jnp.real(xq), jnp.imag(xq)], axis=-1).reshape(xq.shape[:-1], -1)
    xk = jnp.stack([jnp.real(xk), jnp.imag(xk)], axis=-1).reshape(xk.shape[:-1], -1)
    return xq.astype(dtype), xk.astype(dtype)


class ParallelPalmBlock(nn.Module):
    config: PalmConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        attn_inner_dim = self.config.dim_head * self.config.num_attention_heads
        ff_inner_dim = self.config.hidden_size * self.config.up_inner_dim
        self.fused_dims = (attn_inner_dim, self.config.dim_head, self.config.dim_head, ff_inner_dim, ff_inner_dim)

        # INPUT WEIGHTS
        self.wi = self.param(
            'kernel',
            nn.initializers.normal,
            (self.config.hidden_size, sum(self.fused_dims)),
            self.param_dtype,
        )

        # ATTENTION WEIGHT OUTPUT
        self.attn_wo = self.param(
            'kernel',
            nn.initializers.normal,
            (attn_inner_dim, self.config.hidden_size),
            self.param_dtype,
        )

        self.ff_wo = self.param(
            'kernel',
            nn.initializers.normal,
            (attn_inner_dim, self.config.hidden_size),
            self.param_dtype,
        )

        self.norm = RMSNorm(dim=self.config.hidden_size)
        self.post_norm = RMSNorm(dim=self.config.hidden_size)

        self.num_attention_heads: int = self.config.num_attention_heads
        self.scale: float = self.config.dim_head ** -0.5

    def __call__(self, hidden_state, freq_cis, causal_mask):
        split_indices = onp.cumsum(self.fused_dims[:-1])

        hidden_state = self.norm(hidden_state)

        q, k, v, ff, ff_gate = np.split(hidden_state @ self.wi, split_indices, axis=-1)
        q = rearrange(q, 'b s (h d)-> b s h d', h=self.num_attention_heads)
        k = rearrange(k, 'b s (h d)-> b s h d', h=self.num_attention_heads)

        q, k = apply_rotary_embedding(q, k, freq_cis, self.dtype)
        q = rearrange(q, 'b s h d -> b s (h d)')
        k = rearrange(k, 'b s h d -> b s (h d)')
        q = rearrange(q, '... n (h d) -> ... h n d', h=self.num_attention_heads) * self.scale

        sim = jnp.einsum('... h i d, ... j d -> ... h i j', q, k)
        if self.config.use_pjit_attention_force:
            sim = with_sharding_constraint(sim, PartitionSpec(('dp', 'fsdp'), 'mp', None, None))
        mask_value = jnp.finfo(hidden_state).min
        attn = nn.softmax(np.where(causal_mask, sim, mask_value), axis=-1)

        out = jnp.einsum('... h i j, ... j d -> ... h i d', attn, v)
        if self.config.use_pjit_attention_force:
            out = with_sharding_constraint(out, PartitionSpec(('dp', 'fsdp'), 'mp', None, None))
        attn_out = rearrange(out, '... h n hd -> ... n (h hd)') @ self.attn_wo

        ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

        return attn_out + ff_out


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


class ParallelCollection(nn.Module):
    config: PalmConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = ParallelPalmBlock
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
                self.config.num_hidden_layers
            )
        ]

    def __call__(self, hidden_state, freq_cis, causal_mask, output_attention=False):
        saves = []
        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                freq_cis=freq_cis,
                causal_mask=causal_mask
            ) + hidden_state
            if output_attention:
                saves.append(hidden_state)
        return hidden_state, saves


class PalmPretrainedModel(transformers.FlaxPreTrainedModel):
    module_class: nn.Module
    config_class = PalmConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def __init__(self, config: PalmConfig, input_shape=(1, 1), _do_init=False):
        module = self.module_class(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        super().__init__(
            config=config,
            input_shape=input_shape,
            _do_init=_do_init,
            module=module
        )

    def init_weights(self, rng: jax.random.PRNGKey,
                     input_shape: Tuple,
                     params: FrozenDict = None
                     ) -> [Mapping[str, Any], FrozenDict]:

        if params is None:
            return self.module.init(
                rngs=rng,
                input_ids=jnp.ones(input_shape, dtype='i4'),
                attention_mask=jnp.ones(input_shape, dtype='i4'),

            )['params']
        else:
            return params

    def __call__(self, input_ids, attention_mask=None, params=None, add_params_field: bool = False,
                 return_dict: bool = True, output_attention: bool = False):
        params = {'params': params or self.params} if add_params_field else params or self.params
        predict = self.module.apply(
            params,
            input_ids=jnp.asarray(input_ids, dtype='i4'),
            attention_mask=jnp.asarray(attention_mask, dtype='i4') if attention_mask is not None else attention_mask,
            return_dict=return_dict,
            output_attention=output_attention
        )
        return predict


class PalmModule(nn.Module):
    config: PalmConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            embedding_init=jax.nn.initializers.normal
        )
        self.block = ParallelCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.freq_cis = pre_compute_freq_cis(
            self.config.dim_head,
            self.config.max_length,
            dtype=self.dtype
        )

        self.ln_f = RMSNorm(
            dim=self.config.hidden_size,
            dtype=self.dtype,
            precision=self.precision,
            param_dtype=self.param_dtype,
            eps=self.config.eps
        )
        self.causal_mask = nn.make_causal_mask(jnp.ones(
            1, self.config.max_length
        ))

    def make_causal_mask(self, attention_mask=None):
        assert attention_mask is not None
        b, s = attention_mask.shape
        mask = attention_mask + self.causal_mask
        mask = jnp.where(
            mask == 2,
            1, 0
        ).astype(jnp.bool_)
        return mask.reshape(b, 1, 1, s)

    def __call__(self,
                 input_ids: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray = None,
                 return_dict: bool = True,
                 output_attention: bool = False):
        batch, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones(
                (batch, seq_len),
                dtype=jnp.int32
            )

        mask = self.make_causal_mask(
            attention_mask=attention_mask
        )
        hidden_state = self.wte(
            inputs=input_ids
        )
        hidden_state, atn = self.block(
            hidden_state=hidden_state,
            causal_mask=mask,
            output_attention=output_attention,
            freq_cis=self.freq_cis[:seq_len].reshape(1, seq_len, -1)
        )
        hidden_state = self.ln_f(
            hidden_state
        )

        if return_dict:
            return transformers.modeling_flax_outputs.FlaxBaseModelOutput(
                last_hidden_state=hidden_state,
                hidden_states=atn
            )
        else:
            return hidden_state, atn


class PalmModel(PalmPretrainedModel):
    module_class = PalmModule


class FlaxPalmForCausalLMModule(nn.Module):
    config: PalmConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.path_way = PalmModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        if not self.config.use_tie_word_embedding:
            self.lm_head = self.param(
                'kernel',
                jax.nn.initializers.normal,
                (self.config.hidden_size, self.config.vocab_size),
                self.param_dtype
            )

    def __call__(self,
                 input_ids: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray = None,
                 return_dict: bool = True,
                 output_attention: bool = False):
        out = self.path_way(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attention=output_attention
        )
        last_state = out.last_hidden_state
        if not self.config.use_tie_word_embedding:
            last_state = last_state @ self.lm_head
        else:
            last_state = last_state @ self.path_way.wte.embedding.T

        if return_dict:
            return FlaxCausalLMOutput(
                logits=last_state,
                hidden_states=out.hidden_states
            )
        else:
            return last_state, out.hidden_states if output_attention else last_state,


class FlaxPalmForCausalLM(PalmPretrainedModel):
    module_class = FlaxPalmForCausalLMModule
