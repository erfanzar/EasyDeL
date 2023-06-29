from typing import List, Any, Union, Optional

import jax
import jax.numpy as jnp
import numpy as onp
from einops import rearrange
import flax.linen as nn

from jax import numpy as np

from transformers import PretrainedConfig


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
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eps = eps
        self.max_length = max_length


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
    return xq.astype(dtype), xk.dtype(dtype)


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
        mask_value = jnp.finfo(hidden_state).min
        attn = nn.softmax(np.where(causal_mask, sim, mask_value), axis=-1)

        out = jnp.einsum('... h i j, ... j d -> ... h i d', attn, v)

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

    def __call__(self, hidden_state, freq_cis, causal_mask):
        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                freq_cis=freq_cis,
                causal_mask=causal_mask
            )
        return hidden_state


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
        self.causal_mask = nn.make_causal_mask(jnp.ones(
            1, self.config.max_length
        ))

    def __call__(self, input_ids, attention_mask, return_dict: bool = True):
        ...
