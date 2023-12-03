import math
from dataclasses import field, dataclass
from typing import Optional, Tuple, Any, Union, Dict, Sequence

import jax.lax
import transformers
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import unflatten_dict, flatten_dict
from jax import numpy as jnp
from flax import linen as nn
from chex import Array, ArrayDType
from EasyDel.modules.flax_modelling_utils import JaxBaseClassModel, ACT2FN, get_gradient_checkpoint_policy
from transformers import PretrainedConfig
from einops import repeat, rearrange
from transformers.modeling_flax_outputs import FlaxCausalLMOutput


class PhiConfig(PretrainedConfig, JaxBaseClassModel):
    """Phi configuration."""

    model_type = "phi"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size: int = 50304,
            n_positions: int = 2048,
            n_embd: int = 1024,
            n_layer: int = 20,
            n_inner: Optional[int] = None,
            n_head: int = 16,
            n_head_kv: Optional[int] = None,
            rotary_dim: Optional[int] = 32,
            activation_function: Optional[str] = "gelu_new",
            flash_attn: bool = False,
            flash_rotary: bool = False,
            fused_dense: bool = False,
            attn_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            resid_pdrop: float = 0.0,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            tie_word_embeddings: bool = False,
            pad_vocab_size_multiple: int = 64,
            bits: Optional[int] = None,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "mp"),
            gradient_checkpointing: str = 'nothing_saveable',
            **kwargs
    ) -> None:
        self.vocab_size = int(math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.activation_function = activation_function
        self.flash_attn = flash_attn
        self.flash_rotary = flash_rotary
        self.fused_dense = fused_dense
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            axis_dims=axis_dims,
            axis_names=axis_names,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    def add_jax_args(
            self,
            bits: Optional[int] = None,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "mp"),
            gradient_checkpointing: str = 'nothing_saveable',
            **kwargs
    ):
        self.bits = bits
        self.axis_names = axis_names
        self.axis_dims = axis_dims
        self.gradient_checkpointing = gradient_checkpointing

        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)


@dataclass
class InferenceParams:
    max_seq_len: int = field(
        metadata={"help": "Maximum sequence length."}
    )

    max_batch_size: int = field(
        metadata={"help": "Maximum batch size."}
    )

    seq_len_offset: int = field(
        default=0,
        metadata={"help": "Sequence length offset."}
    )

    batch_size_offset: int = field(
        default=0,
        metadata={"help": "Batch size offset."}
    )

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Key value memory dictionary."}
    )

    lengths_per_sample: Union[Array, None] = field(
        default=None,
        metadata={"help": "Lengths per sample."}
    )


class EmbeddingFlax(nn.Module):
    config: PhiConfig

    def setup(self) -> None:
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.embd_pdrop)

    def __call__(self, input_ids: Array) -> Array:
        return self.drop(self.wte(input_ids.reshape(-1, input_ids.shape[-1])))


def _apply_rotary_emb(
        x: Array,
        cos: Array,
        sin: Array,
) -> Array:
    _, seq_len, _, _ = x.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    x1, x2 = x_rot.chunk(2, axis=-1)
    c, s = cos[:seq_len][:, jnp.newaxis, :], sin[:seq_len][:, jnp.newaxis, :]
    x1, x2, c, s = [t.astype(dtype=jnp.float32) for t in [x1, x2, c, s]]

    x_rot = jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)

    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _apply_rotary_emb_kv(
        kv: Array,
        cos: Array,
        sin: Array,
        cos_k: Optional[Array] = None,
        sin_k: Optional[Array] = None,
) -> Array:
    _, seq_len, _, _, _ = kv.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    k_rot = kv[:, :, 0, :, :rotary_dim]
    k_pass = kv[:, :, 0, :, rotary_dim:]

    k1, k2 = k_rot.chunk(2, axis=-1)
    c, s = cos[:seq_len][:, jnp.newaxis, :], sin[:seq_len][:, jnp.newaxis, :]
    k1, k2, c, s = [t.astype(dtype=jnp.float32) for t in [k1, k2, c, s]]

    k_rot = jnp.concatenate([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).astype(kv.dtype)

    return jnp.concatenate(
        [
            jnp.concatenate([k_rot, k_pass], axis=-1)[:, :, jnp.newaxis, :, :],
            kv[:, :, 1:2, :, :],
        ],
        axis=2,
    )


def _apply_rotary_emb_qkv(
        qkv: Array,
        cos: Array,
        sin: Array,
        cos_k: Optional[Array] = None,
        sin_k: Optional[Array] = None,
) -> Array:
    _, seq_len, _, _, _ = qkv.shape
    _, rotary_dim = cos.shape
    rotary_dim *= 2

    q_rot = qkv[:, :, 0, :, :rotary_dim]
    q_pass = qkv[:, :, 0, :, rotary_dim:]

    k_rot = qkv[:, :, 1, :, :rotary_dim]
    k_pass = qkv[:, :, 1, :, rotary_dim:]

    q1, q2 = q_rot.chunk(2, axis=-1)
    k1, k2 = k_rot.chunk(2, axis=-1)
    c, s = cos[:seq_len][:, jnp.newaxis, :], sin[:seq_len][:, jnp.newaxis, :]
    q1, q2, k1, k2, c, s = [t.astype(dtype=jnp.float32) for t in [q1, q2, k1, k2, c, s]]

    q_rot = jnp.concatenate([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).astype(qkv.dtype)
    k_rot = jnp.concatenate([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).astype(qkv.dtype)

    return jnp.concatenate(
        [
            jnp.concatenate([q_rot, q_pass], axis=-1)[:, :, jnp.newaxis, :, :],
            jnp.concatenate([k_rot, k_pass], axis=-1)[:, :, jnp.newaxis, :, :],
            qkv[:, :, 2:3, :, :],
        ],
        axis=2,
    )


class RotaryEmbedding(nn.Module):
    axis: int
    base: int = 10000
    scale_base: Optional[float] = None
    pos_idx_in_fp32: bool = True
    max_position_embeddings: int = 2048
    """Rotary positional embedding (RoPE).
    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://arxiv.org/pdf/2104.09864.pdf.
    """

    def setup(
            self
    ) -> None:

        if self.scale_base is not None:
            raise NotImplementedError

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        scale = (
            (jnp.arange(0, self.axis, 2, dtype=jnp.float32) + 0.4 * self.axis) / (1.4 * self.axis)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._update_cos_sin_cache(self.max_position_embeddings, dtype=jnp.float32)

    def _compute_inv_freq(self) -> Array:
        return 1.0 / (self.base ** (jnp.arange(0, self.axis, 2, dtype=jnp.float32) / self.axis))

    def _update_cos_sin_cache(
            self,
            seq_len: int,
            dtype: Optional[ArrayDType] = None,
    ) -> None:
        self._seq_len_cached = seq_len

        if self.pos_idx_in_fp32:
            t = jnp.arange(seq_len, dtype=jnp.float32)
            if self.inv_freq.dtype != jnp.float32:
                inv_freq = self._compute_inv_freq()
            else:
                inv_freq = self.inv_freq
        else:
            t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
            inv_freq = self.inv_freq

        freqs = jnp.outer(t, inv_freq)
        if self.scale is None:
            self._cos_cached = jnp.cos(freqs).astype(dtype)
            self._sin_cached = jnp.sin(freqs).astype(dtype)
        else:
            power = (
                            jnp.arange(seq_len, dtype=self.scale.dtype) - seq_len // 2
                    ) / self.scale_base
            scale = self.scale ** power[:, jnp.newaxis]

            self._cos_cached = (jnp.cos(freqs) * scale).astype(dtype)
            self._sin_cached = (jnp.sin(freqs) * scale).astype(dtype)
            self._cos_k_cached = (jnp.cos(freqs) / scale).astype(dtype)
            self._sin_k_cached = (jnp.sin(freqs) / scale).astype(dtype)

    def __call__(
            self,
            qkv: Array,
            kv: Optional[Array] = None,
            seq_len_offset: int = 0,
    ) -> Tuple[Array, Array]:
        seq_start = seq_len_offset
        seq_end = seq_start + qkv.shape[1]

        if (
                self._cos_cached.device != qkv.device
                or self._cos_cached.dtype != qkv.dtype
        ):
            self._update_cos_sin_cache(self.max_position_embeddings, dtype=qkv.dtype)

        if kv is None:
            return _apply_rotary_emb_qkv(
                qkv,
                self._cos_cached[seq_start:seq_end],
                self._sin_cached[seq_start:seq_end],
            )
        else:
            q = _apply_rotary_emb(
                qkv,
                self._cos_cached[seq_start:seq_end],
                self._sin_cached[seq_start:seq_end],
            )
            kv = _apply_rotary_emb_kv(
                kv,
                self._cos_cached[seq_start:seq_end],
                self._sin_cached[seq_start:seq_end],
            )

            return q, kv


class MLP(nn.Module):
    config: PhiConfig
    n_inner: Optional[int] = None
    act_fn: Optional[str] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    """Multi-Layer Perceptron.
    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.
    """

    def setup(
            self
    ) -> None:
        act_fn = self.config.activation_function if self.act_fn is None else self.act_fn

        n_inner = getattr(self.config, "n_inner", None) if self.n_inner is None else self.n_inner
        n_inner = n_inner if n_inner is not None else 4 * self.config.n_embd

        self.fc1 = nn.Dense(
            n_inner,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.fc2 = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.act = ACT2FN[act_fn]

    def __call__(self, hidden_states: Array) -> Array:
        return self.fc2(self.act(self.fc1(hidden_states)))


class SelfAttention(nn.Module):
    causal: bool = True
    softmax_scale: Optional[float] = None
    attention_dropout: float = 0.0

    """
    Self-attention layer (compatible with JAX/FLAX).
    """

    def setup(
            self
    ) -> None:
        self.drop = nn.Dropout(self.attention_dropout)

    def __call__(
            self,
            qkv: Array,
            causal: bool = None,
            key_padding_mask: Optional[Array] = None,
            deterministic: bool = True,
            **kwargs,
    ) -> Array:
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(axis=2)

        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or jax.lax.rsqrt(q.shape[-1])

        scores = jnp.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = jax.lax.select(
                key_padding_mask.astype(jnp.bool_), 0.0, -10000.0
            )[:, jnp.newaxis, jnp.newaxis, :]

            scores = scores + padding_mask

        if causal:
            causal_mask = jnp.triu(jnp.full((seq_len, seq_len), -10000.0), 1)
            scores = scores + causal_mask.astype(dtype=scores.dtype)

        attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        attention = self.drop(attention, deterministic=deterministic)
        output = jnp.einsum("bhts,bshd->bthd", attention, v)
        return output


class CrossAttention(nn.Module):
    causal: bool = True
    softmax_scale: Optional[float] = None
    attention_dropout: float = 0.0
    """
    Cross-attention layer (compatible with JAX/FLAX).
    """

    def setup(
            self
    ) -> None:
        self.drop = nn.Dropout(self.attention_dropout)

    def __call__(
            self,
            q: Array,
            kv: Array,
            causal: bool = None,
            key_padding_mask: Optional[Array] = None,
            deterministic: bool = True,
            **kwargs,
    ) -> Array:
        batch_size, seq_len_q = q.shape[0], q.shape[1]
        seq_len_k = kv.shape[1]

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(axis=2)

        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or jax.lax.rsqrt(q.shape[-1])

        scores = jnp.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = jax.lax.select(
                key_padding_mask.astype(jnp.bool_), 0.0, -10000.0
            )[:, jnp.newaxis, jnp.newaxis, :]

            scores = scores + padding_mask

        if causal:
            rows = jnp.arange(seq_len_q, dtype=jnp.int32)[:, jnp.newaxis]
            cols = jnp.arange(seq_len_k, dtype=jnp.int32)
            causal_mask = cols > rows + seq_len_k - seq_len_q

            scores = jax.lax.select(
                causal_mask, scores, -10000.0
            )

        attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        attention = self.drop(attention, deterministic=deterministic)

        output = jnp.einsum("bhts,bshd->bthd", attention, v)

        return output


def _find_mha_dims(
        config: PhiConfig,
        n_head: Optional[int] = None,
        n_head_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
) -> Tuple[Union[int, Any], Union[int, None, Any], Union[int, Any]]:
    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")
    if n_head_kv is None:
        n_head_kv = getattr(config, "n_head_kv", None) or n_head

    return n_head, n_head_kv, head_dim


class MHA(nn.Module):
    config: PhiConfig
    dtype: Optional[jnp.dtype] = jnp.float32
    param_dtype: Optional[jnp.dtype] = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')
    rotary_dim: Optional[int] = None
    rotary_base: float = 10000.0
    rotary_scale_base: Optional[float] = None
    n_head: Optional[int] = None
    n_head_kv: Optional[int] = None
    head_dim: Optional[int] = None
    bias: bool = True
    causal: bool = True
    softmax_scale: Optional[float] = None
    layer_idx: Optional[int] = None
    return_residual: bool = False

    def setup(
            self
    ) -> None:

        self.rotary_dim = self.rotary_dim if self.rotary_dim is not None else getattr(self.config, "rotary_dim", 0)

        if self.rotary_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_dim,
                base=self.rotary_base,
                scale_base=self.rotary_scale_base,
                max_position_embeddings=self.config.n_positions
            )

        # MLP
        self.n_head, self.n_head_kv, self.head_dim = _find_mha_dims(
            self.config, n_head=self.n_head, n_head_kv=self.n_head_kv, head_dim=self.head_dim
        )
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        hidden_size = self.config.n_embd

        self.Wqkv = nn.Dense(
            op_size,
            use_bias=self.bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precition=self.precision,
            kernel_init=nn.initializers.normal(self.config.initializer_range)
        )
        self.out_proj = nn.Dense(
            hidden_size,
            use_bias=self.bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precition=self.precision,
            kernel_init=nn.initializers.normal(self.config.initializer_range)
        )

        self.inner_attn = SelfAttention(
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            attention_dropout=self.config.attn_pdrop,
        )
        self.inner_cross_attn = CrossAttention(
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            attention_dropout=self.config.attn_pdrop,
        )
        self.flash_attn = False

    def _forward_self_attn(
            self,
            x: Array,
            key_padding_mask: Optional[Array],
            deterministic: bool = True
    ) -> Array:
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if self.rotary_dim > 0:
            qkv = self.rotary_emb(qkv)

        return self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            deterministic=deterministic
        )

    def _forward_cross_attn(
            self,
            x: Array,
            key_padding_mask: Optional[Array],
            position_ids: Optional[Array] = None,
            deterministic: bool = True
    ) -> Array:

        # TODO: adding past_key_values
        past_key_values = None

        batch_size = x.shape[0]

        qkv = self.Wqkv(x)

        q = qkv[..., : self.n_head * self.head_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        kv = qkv[..., self.n_head * self.head_dim:]
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        seq_len_offset = position_ids[batch_size - 1, 0] if position_ids is not None else 0
        causal = None if seq_len_offset == 0 else False
        if self.rotary_dim > 0:
            q, kv = self.rotary_emb(q, kv=kv, seq_len_offset=seq_len_offset)

        if past_key_values is not None:
            raise NotImplementedError("TODO ?")

        return self.inner_cross_attn(
            q,
            kv,
            key_padding_mask=key_padding_mask,
            causal=causal,
            deterministic=deterministic
        )

    def __call__(
            self,
            x: Array,
            attention_mask: Array = None,
            deterministic: bool = True,
            **kwargs,
    ) -> Tuple[Array, Array]:

        attention_mask = attention_mask.astype(jnp.bool_)

        # TODO: adding past_key_values
        past_key_values = None
        if self.n_head == self.n_head_kv:
            if past_key_values is None:
                attn_output = self._forward_self_attn(x, attention_mask, deterministic=deterministic)
            else:
                attn_output = self._forward_cross_attn(x, past_key_values, attention_mask, deterministic=deterministic)
        else:
            attn_output = self._forward_cross_attn(x, past_key_values, attention_mask, deterministic=deterministic)

        output = rearrange(attn_output, "... h d -> ... (h d)")
        output = self.out_proj(output)

        return output if not self.return_residual else (output, x)


class ParallelBlock(nn.Module):
    """Parallel block.
    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).
    """
    config: PhiConfig
    block_idx: Optional[int] = None
    dtype: Optional[jnp.dtype] = jnp.float32
    param_dtype: Optional[jnp.dtype] = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    def setup(
            self,
    ) -> None:
        self.ln = nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)

        self.mixer = MHA(
            self.config,
            layer_idx=self.block_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.mlp = MLP(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            hidden_states: Array,
            attention_mask: Optional[Array] = None,
            deterministic: bool = True
    ) -> Array:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic
        )
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class CausalLMHead(nn.Module):
    config: PhiConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')
    """Causal Language Modeling head.
    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.
    """

    def setup(self) -> None:
        self.ln = nn.LayerNorm(
            self.config.n_embd,
            eps=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )
        self.linear = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(self.config.initializer_range)
        )

    def __call__(self, hidden_states: Array) -> Array:
        return self.linear(self.ln(hidden_states)).to(jnp.float32)


class FlaxPhiPreTrainedModel(transformers.FlaxPreTrainedModel):
    """Phi pre-trained model."""
    module_class = None
    config_class = PhiConfig
    base_model_prefix = "transformer"

    def __init__(self,
                 config: PhiConfig,
                 dtype: jnp.dtype = jnp.float32,
                 param_dtype: jnp.dtype = jnp.float32,
                 precision: jax.lax.Precision = jax.lax.Precision('fastest'),
                 input_shape=(1, 1),
                 seed: int = 42,
                 _do_init: bool = False
                 ) -> None:
        module = self.module(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision
        )
        super().__init__(
            config=config,
            module=module,
            input_shape=input_shape,
            _do_init=_do_init,
            seed=seed
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: Array,
            attention_mask: Optional[Union[Array, Array]] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        # TODO: adding past_key_values
        past_key_values = None

        if input_ids.shape[1] > self.config.n_positions:
            input_ids = input_ids[:, -self.config.n_positions:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.n_positions:]

        if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
            past_key_values = InferenceParams(
                max_seq_len=self.config.n_positions,
                max_batch_size=input_ids.shape[0],
                seq_len_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,
            )
        else:
            past_key_values.seq_len_offset = input_ids.shape[1] - 1
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params


class ParallelBlockCollection(nn.Module):
    config: PhiConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        block = ParallelBlock
        if self.config.gradient_checkpointing != '':
            policy = get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            block = nn.remat(
                block,
                policy=policy,
                static_argnums=(-1)
            )
        self.layers = [
            block(
                self.config,
                block_idx=i,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            ) for i in range(self.config.n_layer)
        ]

    def __call__(
            self,
            hidden_states: Array,
            attention_mask: Optional[Array] = None,
            deterministic: bool = True
    ) -> Array:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic
            )
        return hidden_states


class FlaxPhiModule(nn.Module):
    """Phi model."""
    config: PhiConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        self.embd = EmbeddingFlax()
        self.h = ParallelBlockCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __cal__(
            self,
            input_ids: Array,
            attention_mask: Optional[Array] = None,
            deterministic: bool = True
    ) -> Array:
        return self.h(
            self.embd(input_ids),
            attention_mask=attention_mask,
            deterministic=deterministic
        )


class FlaxPhiForCausalLMModule(nn.Module):
    """Phi for Causal Language Modeling."""
    config: PhiConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision('fastest')

    def setup(self) -> None:
        self.transformer = FlaxPhiModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = CausalLMHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids: Array,
            attention_mask: Optional[Array] = None,
            labels: Optional[Array] = None,
            **kwargs,
    ) -> FlaxCausalLMOutput:
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(hidden_states)

        return FlaxCausalLMOutput(logits=lm_logits)


class FlaxPhiForCausalLM(FlaxPhiPreTrainedModel):
    module_class = FlaxPhiForCausalLMModule


class FlaxPhiModel(FlaxPhiPreTrainedModel):
    module_class = FlaxPhiModule
