import math
from typing import Dict, Optional, Tuple, Union

import fjutils.easylm
from einops import einops
from flax.linen import remat
from einops import einsum
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from jax.interpreters import pxla
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput

from jax.experimental.pjit import with_sharding_constraint as wsc
from fjutils import dot_product_attention_multihead
from fjutils.easylm import blockwise_dot_product_attention


def get_names_from_partition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


class LlamaConfig(PretrainedConfig):
    model_type = "Llama.md"

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            number_rep_kv: int = 1,
            num_key_value_heads: Optional[int] = None,
            max_position_embeddings: int = 2048,
            rms_norm_eps: float = 1e-6,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            bos_token_id: int = 0,
            eos_token_id: int = 1,
            resid_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            attn_pdrop: float = 0.0,
            tie_word_embeddings: bool = False,
            gradient_checkpointing: str = 'nothing_saveable',
            fcm_min_ratio: float = -1,
            fcm_max_ratio: float = -1,
            use_pjit_attention_force: bool = True,
            rope_scaling: Dict[str, Union[str, float]] = None,
            use_flash_attention: bool = False,
            use_sacn_mlp: bool = False,
            flash_attn_query_chunk_size: int = 1024,
            flash_attn_key_chunk_size: int = 1024,
            scan_mlp_chunk_size: int = 1024,
            from_pt: bool = False,
            attn_type='llama2',
            **kwargs,
    ):

        num_key_value_heads = num_key_value_heads or number_rep_kv * num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size

        self.number_rep_kv = number_rep_kv
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.rope_scaling = rope_scaling
        self.use_flash_attention = use_flash_attention
        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size,
        self.from_pt = from_pt

        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, fully_fsdp: bool = True):
        if not self.from_pt:
            return (

                ("transformer/wte/embedding", PS("dp", "fsdp")),

                ("attention/(wq|wk|wv)/kernel", PS("fsdp", "dp")),
                ("attention/wo/kernel", PS("dp", "fsdp")),

                ("feed_forward/w1/kernel", PS("fsdp", "dp")),
                ("feed_forward/w2/kernel", PS("dp", "fsdp")),
                ("feed_forward/w3/kernel", PS("fsdp", "dp")),

                ("attention_norm/kernel", PS(None)),
                ("ffn_norm/kernel", PS(None)),

                ("transformer/ln_f/kernel", PS(None)),
                ("lm_head/kernel", PS("fsdp", "dp")),
                ('.*', PS(None)),
            ) if not fully_fsdp else (

                ("transformer/wte/embedding", PS("fsdp")),

                ("attention/(wq|wk|wv)/kernel", PS("fsdp")),
                ("attention/wo/kernel", PS("fsdp")),

                ("feed_forward/w1/kernel", PS("fsdp")),
                ("feed_forward/w2/kernel", PS("fsdp")),
                ("feed_forward/w3/kernel", PS("fsdp")),

                ("attention_norm/kernel", PS(None)),
                ("ffn_norm/kernel", PS(None)),

                ("transformer/ln_f/kernel", PS(None)),
                ("lm_head/kernel", PS("fsdp")),
                ('.*', PS(None)),
            )
        else:
            return (

                ("model/embed_tokens/embedding", PS("dp", "fsdp")),

                ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS("fsdp", "dp")),
                ("self_attn/o_proj/kernel", PS("dp", "fsdp")),

                ("mlp/gate_proj/kernel", PS("fsdp", "dp")),
                ("mlp/down_proj/kernel", PS("dp", "fsdp")),
                ("mlp/up_proj/kernel", PS("fsdp", "dp")),

                ("input_layernorm/kernel", PS(None)),
                ("post_attention_layernorm/kernel", PS(None)),

                ("model/norm/kernel", PS(None)),
                ("lm_head/kernel", PS("fsdp", "dp")),
                ('.*', PS(None)),
            ) if not fully_fsdp else (

                ("model/embed_tokens/embedding", PS("fsdp")),

                ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS("fsdp")),
                ("self_attn/o_proj/kernel", PS("fsdp")),

                ("mlp/gate_proj/kernel", PS("fsdp")),
                ("mlp/down_proj/kernel", PS("fsdp")),
                ("mlp/up_proj/kernel", PS("fsdp")),

                ("input_layernorm/kernel", PS(None)),
                ("post_attention_layernorm/kernel", PS(None)),

                ("model/norm/kernel", PS(None)),
                ("lm_head/kernel", PS("fsdp")),
                ('.*', PS('fsdp')),
            )

    def add_jax_args(self,
                     resid_pdrop: float = 0.0,
                     embd_pdrop: float = 0.0,
                     attn_pdrop: float = 0.0,
                     tie_word_embeddings: bool = False,
                     gradient_checkpointing: str = 'nothing_saveable',
                     fcm_min_ratio: float = 0.0,
                     fcm_max_ratio: float = 0.0,
                     use_pjit_attention_force: bool = True,
                     use_flash_attention: bool = False,
                     use_sacn_mlp: bool = False,
                     flash_attn_query_chunk_size: int = 1024,
                     flash_attn_key_chunk_size: int = 1024,
                     scan_mlp_chunk_size: int = 1024,
                     from_pt: bool = False,
                     number_rep_kv: int = 1,
                     attn_type='llama'
                     ):
        self.from_pt = from_pt
        self.use_flash_attention = use_flash_attention
        self.embd_pdrop = embd_pdrop
        self.attn_type = attn_type
        self.number_rep_kv = number_rep_kv
        self.resid_pdrop = resid_pdrop

        self.attn_pdrop = attn_pdrop
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.use_pjit_attention_force = use_pjit_attention_force

        self.use_sacn_mlp = use_sacn_mlp
        self.flash_attn_query_chunk_size = flash_attn_query_chunk_size
        self.flash_attn_key_chunk_size = flash_attn_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')


remat = nn_partitioning.remat


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return jnp.expand_dims(x[:, :, :, None, :], (bs, s, n_kv_heads, n_rep, head_dim)).reshape(bs, s, n_kv_heads * n_rep,
                                                                                              head_dim)


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
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def precompute_freqs_cis(
        method: str,
        dim: int, end: int, theta: float = 10000.0,
        scaling_factor: float = 8., **kwargs) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)

    if method is not None:
        if method == 'linear':
            t = t / scaling_factor
        elif method == 'dynamic':
            base = theta * (
                    (scaling_factor * end / end) - (scaling_factor - 1)
            ) ** (dim / (dim - 2))
            freqs = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        else:
            raise ValueError(f'unknown {method} method for precompute_freqs_cis')

    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    sin, cos = jnp.sin(freqs).astype(jnp.float32), jnp.cos(freqs).astype(jnp.float32)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
        xq: jnp.ndarray,
        xk: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class FlaxLlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.number_of_reps = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.number_of_reps == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.wq = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='q_proj' if self.config.from_pt else 'wq'
        )
        self.wk = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='k_proj' if self.config.from_pt else 'wk'
        )
        self.wv = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='v_proj' if self.config.from_pt else 'wv'
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='o_proj' if self.config.from_pt else 'wo'
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"),
                                            dtype="bool")

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            if self.config.attn_type == 'llama2':
                ...
            else:
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

    def __call__(
            self,
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
    ):

        if self.config.attn_type == 'llama2':
            bsz, sq_len = hidden_states.shape[:2]
            xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

            if self.config.use_pjit_attention_force:
                xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
                xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
                xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

            xq = xq.reshape(bsz, sq_len, self.config.num_attention_heads, self.head_dim)
            xk = xk.reshape(bsz, sq_len, self.config.num_key_value_heads, self.head_dim)
            xv = xv.reshape(bsz, sq_len, self.config.num_key_value_heads, self.head_dim)

            freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)
            xk = repeat_kv(xk, self.number_of_reps)
            xv = repeat_kv(xv, self.number_of_reps)

            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_wight = nn.dot_product_attention(
                value=xv,
                key=xk,
                query=xq,
                dtype=jnp.promote_types(self.dtype, jnp.float32),
                deterministic=deterministic,
                dropout_rate=self.config.attn_pdrop,
                bias=attention_bias[:, :, :sq_len, :sq_len],
                precision=self.precision,

            )
            out = self.wo(attn_wight.reshape(bsz, sq_len, -1))
            return (out, attn_wight) if output_attentions else (out,)

        else:
            bsz, sq_len = hidden_states.shape[:2]
            xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
            if self.config.use_pjit_attention_force:
                xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
                xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
                xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

            xq = xq.reshape(bsz, sq_len, self.config.num_attention_heads, self.head_dim)
            xk = xk.reshape(bsz, sq_len, self.config.num_key_value_heads, self.head_dim)
            xv = xv.reshape(bsz, sq_len, self.config.num_key_value_heads, self.head_dim)

            freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

            dropout_rng = None
            if not deterministic and self.config.attn_pdrop > 0.0:
                dropout_rng = self.make_rng("dropout")

            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            if self.config.use_flash_attention and not (self.has_variable("cache", "cached_key") or init_cache):
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
                )
                attn_weights = None
                attn_output = blockwise_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    bias=attention_bias,
                    deterministic=deterministic,
                    dropout_rng=dropout_rng,
                    attn_pdrop=self.config.attn_pdrop,
                    causal=True,
                    query_chunk_size=self.config.scan_query_chunk_size,
                    key_chunk_size=self.config.scan_key_chunk_size,
                    dtype=self.dtype,
                    policy=get_gradient_checkpoint_policy('nothing_saveable'),
                    precision=self.precision,
                    float32_logits=True,
                )
                if self.config.use_pjit_attention_force:
                    attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp", None))
                attn_output = self._merge_heads(attn_output)
            else:

                attn_weights = dot_product_attention_weights(
                    query=xq,
                    key=xk,
                    bias=attention_bias,
                    dtype=jnp.promote_types(self.dtype, jnp.float32),
                    deterministic=deterministic,
                    dropout_rate=self.config.attn_pdrop,
                    precision=self.precision,
                )
                if self.config.use_pjit_attention_force:
                    attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))

                attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv)
                attn_output = self._merge_heads(attn_output)

            attn_output = self.wo(attn_output)
            attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs


class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='gate_proj' if self.config.from_pt else 'w1'
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='down_proj' if self.config.from_pt else 'w2'
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            name='up_proj' if self.config.from_pt else 'w3'
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class FlaxLlamaBlock(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        attn_block = FlaxLlamaAttention
        if self.config.gradient_checkpointing != '':
            attn_block = remat(
                FlaxLlamaAttention, static_argnums=(4, 5, 6),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )

        self.attention = attn_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='self_attn' if self.config.from_pt else 'attention'
        )
        mlp_block = FlaxLlamaMLP

        if self.config.gradient_checkpointing != '':
            mlp_block = remat(
                FlaxLlamaMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )

        self.feed_forward = mlp_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='mlp' if self.config.from_pt else 'feed_forward'
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='input_layernorm' if self.config.from_pt else 'attention_norm'
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='post_attention_layernorm' if self.config.from_pt else 'ffn_norm'
        )

    def __call__(
            self,
            hidden_states,
            freqs_cis,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask: Optional[jnp.ndarray] = None,
    ):
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            freqs_cis,
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)

        if self.config.use_sacn_mlp:
            feed_forward_input = einops.rearrange(
                feed_forward_input,
                '... (b s) d -> ... b s d',
                b=self.config.scan_mlp_chunk_size
            )

            def mlp_forward(mlp, carry, x):
                return None, mlp(x, deterministic)

            scan_axis = feed_forward_input.ndim - 3

            _, feed_forward_hidden_states = nn.scan(
                mlp_forward,
                variable_broadcast="params",
                split_rngs={"params": False, "dropout": True},
                in_axes=scan_axis,
                out_axes=scan_axis,
            )(self.feed_forward, None, feed_forward_input)
            feed_forward_hidden_states = einops.rearrange(
                feed_forward_hidden_states,
                '... b s d -> ... (b s) d'
            )
        else:
            feed_forward_hidden_states = self.feed_forward(
                feed_forward_input,
                deterministic,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
            self,
            config: LlamaConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:

        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

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

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params} if add_params_field else params or self.params

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            extra_embedding,
            rngs=rngs,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxLlamaBlockCollection(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        block = FlaxLlamaBlock

        self.blocks = [
            block(self.config, name=str(i), dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states,
            freqs_cis,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng('fcm'), shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio
            )
            fcm_mask = jax.random.uniform(
                self.make_rng('fcm'),
                shape=(batch_size, 1, seq_length, seq_length)
            ) > fcm_ratio
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype('bool')
        else:
            fcm_mask = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                fcm_mask=fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='embed_tokens' if self.config.from_pt else 'wte'
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxLlamaBlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype,
                                          precision=self.precision, name='layers' if self.config.from_pt else 'h')
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype,
                            param_dtype=self.param_dtype, name='norm' if self.config.from_pt else 'ln_f')
        config = self.config
        self.freqs_cis = precompute_freqs_cis(
            method=config.rope_scaling['type'] if config.rope_scaling is not None else None,
            scaling_factor=float(config.rope_scaling['factor'] if config.rope_scaling is not None else 1),
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings * 2
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            deterministic=True,
            input_embeds=None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        if input_embeds is None:
            input_embeds = self.wte(input_ids.astype("i4"))

        input_embeds = input_embeds + extra_embedding if extra_embedding is not None else input_embeds
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states=hidden_states,
            freqs_cis=self.freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaModule


class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.transformer = FlaxLlamaModule(self.config,
                                           dtype=self.dtype,
                                           param_dtype=self.param_dtype,
                                           precision=self.precision,
                                           name='model' if self.config.from_pt else 'transformer')
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
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


class FlaxLlamaForSequenceClassificationModule(nn.Module):
    num_classes: int
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.transformer = FlaxLlamaModule(self.config, dtype=self.dtype)
        self.classifier = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding
        )

        hidden_states = outputs[0]
        prediction = self.classifier(hidden_states)
        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=prediction,
                hidden_states=hidden_states
            )
        else:
            return prediction,


class FlaxLlamaForSequenceClassification(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForSequenceClassificationModule
