import functools
import math
import typing

import fjformer
import flax
from flax.struct import dataclass
from jax import numpy as jnp, lax
import jax
from fjformer import linen as nn
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.core import freeze, unfreeze, FrozenDict
from typing import Union, Optional, Tuple
from flax.linen import partitioning as nn_partitioning, combine_masks
from transformers.modeling_flax_outputs import FlaxMaskedLMOutput, FlaxBaseModelOutput, FlaxCausalLMOutput
from fjformer.func import auxiliary_load_balancing_loss_func
from ..attention_module import AttentionModule
from ..flax_modelling_utils import (
    ACT2FN,
    with_sharding_constraint,
    repeat_kv_bnsh,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    get_gradient_checkpoint_policy,
    block_wise_ffn
)
from jax.sharding import PartitionSpec
import chex
from .deepseek_configuration import DeepseekV2Config
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel

re_mat = nn_partitioning.remat


@flax.struct.dataclass
class MoeModelOutput:
    last_hidden_state: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    router_logits: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
    aux_loss: Optional[chex.Array] = None
    router_logits: Optional[Tuple[chex.Array]] = None


class DeepseekV2RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

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
        weight = fjformer.linen.linen.control_quantization(self.weight, self.dtype)
        return output * weight


def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
    )


def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1., mscale=1.):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min) / (max - min)
    return jnp.clip(linear_func, 0, 1)


def init_deepseek_rotary_embedding(
        dim,
        max_position_embeddings=2048,
        base=10000,
        method: typing.Literal["linear", "yarn", "dynamic", None] = None,
        kwargs: typing.Optional[dict] = None,
):
    if method is None:
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "linear":
        assert kwargs is not None
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get("scaling_factor")
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "dynamic":
        assert kwargs is not None
        targeted_len = kwargs.get("targeted_len", max_position_embeddings)
        if targeted_len > max_position_embeddings:
            base = base * (
                    (kwargs.get("scaling_factor") * targeted_len / max_position_embeddings)
                    - (kwargs.get("scaling_factor") - 1)
            ) ** (dim / (dim - 2))
            inv_freq = 1.0 / (
                    base ** (jnp.arange(0, dim, 2).astype("float32") / dim)
            )

        else:
            inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get("scaling_factor")

        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "yarn":

        scaling_factor = kwargs.get("scaling_factor", 1.0)
        original_max_position_embeddings = kwargs.get("original_max_position_embeddings", 4096)
        beta_fast = kwargs.get("beta_fast", 32)
        beta_slow = kwargs.get("beta_slow", 1)
        mscale = kwargs.get("mscale", 1)
        mscale_all_dim = kwargs.get("mscale_all_dim", 0)
        freq_extra = 1.0 / (
                base
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        freq_inter = 1.0 / (
                scaling_factor
                * base
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).astype("float32")
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)

        freqs = jnp.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
        )

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return (jnp.sin(emb) * _mscale).astype("float32"), (jnp.cos(emb) * _mscale).astype("float32")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = jnp.expand_dims(cos[position_ids], unsqueeze_dim)
    sin = jnp.expand_dims(sin[position_ids], unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxDeepseekV2MLP(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None

    def setup(self) -> None:
        dense = functools.partial(
            nn.Linear,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.gate_proj = dense(self.intermediate_size or self.config.intermediate_size)
        self.up_proj = dense(self.intermediate_size or self.config.intermediate_size)
        self.down_proj = dense(self.hidden_size or self.config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
            self,
            x: chex.Array,
            e: bool = False  # Ignored
    ):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FlaxMoEGate(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = self.param(
            "kernel",
            nn.initializers.kaiming_uniform(dtype=self.param_dtype),
            (self.n_routed_experts, self.gating_dim)
        )

    def __call__(self, hidden_states, deterministic: bool = True):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        logits = jax.lax.batch_matmul(
            hidden_states.astype(jnp.float32),
            self.weight.astype(jnp.float32),
            precision=self.precision
        )
        if self.scoring_func == "softmax":
            scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "gready":
            topk_weight, topk_idx = jax.lax.top_k(
                scores, k=self.top_k
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.reshape(bsz * seq_len, self.n_group, -1).max(axis=-1)  # [n, n_group]

            # Find the indices of the top k scores in each group
            top_k_indices = lax.top_k(group_scores, self.topk_group)[1]  # [n, topk_group]

            # Initialize a mask with zeros
            group_mask = jnp.zeros_like(group_scores)  # [n, n_group]

            # Update the mask: this is a bit tricky in JAX as there is no direct scatter function
            n_indices = jnp.arange(group_mask.shape[0])[:, None]
            group_mask = group_mask.at[n_indices, top_k_indices].set(1)  # [n, n_group]

            # Expand and reshape the group_mask
            score_mask = jnp.repeat(group_mask[:, :, None], self.n_routed_experts // self.n_group, axis=2)
            score_mask = score_mask.reshape(bsz * seq_len, -1)  # [n, e]

            # Apply the mask to scores
            masked_scores = jnp.where(score_mask, scores, 0.0)  # [n, e]

            # Compute the top k scores after masking
            topk_weight, topk_idx = lax.top_k(masked_scores, self.top_k)
        else:
            raise ValueError()
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if not deterministic and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.reshape(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.reshape(bsz, seq_len, -1)
                ce = jnp.zeros(bsz, self.n_routed_experts)
                ce = ce.at[1, topk_idx_for_aux_loss].add(
                    jnp.ones(bsz, seq_len * aux_topk),
                )
                ce = jnp.divide(ce, (seq_len * aux_topk / self.n_routed_experts))
                aux_loss = jnp.mean(jnp.sum((ce * jnp.mean(scores_for_seq_aux, axis=-1)), axis=1)) * self.alpha
            else:
                mask_ce = jax.nn.one_hot(
                    topk_idx_for_aux_loss.reshape(-1), num_classes=self.n_routed_experts
                )
                ce = jnp.mean(mask_ce.astype("float32"), axis=0)
                Pi = jnp.mean(scores_for_aux, axis=0)
                fi = ce * self.n_routed_experts
                aux_loss = jnp.sum(Pi * fi) * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class FlaxDeepseekV2MLPCollection(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.experts = [
            FlaxDeepseekV2MLP(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                intermediate_size=self.config.moe_intermediate_size,
                name=str(i)
            )
            for i in range(self.config.n_routed_experts)
        ]

    def __call__(self, hidden_states, flat_topk_idx):
        y = jnp.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        return y


class FlaxDeepseekV2MoE(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0
        self.experts = FlaxDeepseekV2MLPCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.gate = FlaxMoEGate(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = FlaxDeepseekV2MoE(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                intermediate_size=intermediate_size,
            )

    def __call__(
            self,
            hidden_states: chex.Array,
            e: bool = False  # ignored !
    ):

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.reshape(-1)
        hidden_states = hidden_states.repeat(self.num_experts_per_tok, axis=0)
        y = self.experts(hidden_states=hidden_states, flat_topk_idx=flat_topk_idx)
        y = (y.reshape(*topk_weight.shape, -1) * jnp.expand_dims(topk_weight, -1)).sum(axis=1)
        y = y.reshape(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class FlaxDeepseekV2Attention(BaseJAXAttentionModule):
    config: DeepseekV2Config
    layer_idx: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        dense_class = functools.partial(
            nn.Linear,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.q_a_proj = dense_class(config.q_lora_rank, use_bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.q_b_proj = dense_class(self.num_heads * self.q_head_dim, use_bias=False)

        self.kv_a_proj_with_mqa = dense_class(
            config.kv_lora_rank + config.qk_rope_head_dim,
            use_bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = dense_class(
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            use_bias=False,
        )

        self.o_proj = dense_class(self.hidden_size, use_bias=config.attention_bias)

        softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = self.softmax_scale * mscale * mscale

        self.attention_performer = AttentionModule(
            use_sharding_constraint=self.config.use_sharding_constraint,
            block_k_major=self.config.block_k_major,
            block_b=self.config.block_b,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            block_q_major_dkv=self.config.block_q_major_dkv,
            block_k_major_dkv=self.config.block_k_major_dkv,
            block_k_major_dq=self.config.block_k_major_dq,
            block_k_dkv=self.config.block_k_dkv,
            block_q_dkv=self.config.block_q_dkv,
            block_q_dq=self.config.block_q_dq,
            block_k_dq=self.config.block_k_dq,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.q_head_dim,
            attention_partition_spec=self.config.attention_partition_spec,
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            generation_query_partition_spec=self.config.generation_query_partition_spec,
            generation_bias_partition_spec=self.config.generation_bias_partition_spec,
            generation_attention_partition_spec=self.config.generation_attention_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=softmax_scale,
            axis_name=self.config.attention_axis_name
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
    ):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[:, :, :, :self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim:]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        k_pe = compressed_kv[:, :, :, self.kv_lora_rank:self.kv_lora_rank + self.qk_rope_head_dim]
        compressed_kv = compressed_kv[:, :, :, :self.kv_lora_rank]

        k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        )

        k_nope = kv[:, :, :, :self.qk_nope_head_dim]
        value_states = kv[:, :, :, self.qk_nope_head_dim:self.qk_nope_head_dim + self.v_head_dim]

        sin, cos = freq_cis

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1,
                                                     query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.broadcast_to(
            attention_mask, causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states,
                value_states,
                query_states,
                attention_mask
            )
        # if self.config.use_sharding_constraint:
        #     query_states = with_sharding_constraint(
        #         query_states, PartitionSpec(("dp", "fsdp"), "sp" if query_states.shape[1] != 1 else None, "tp", None)
        #     )
        #     key_states = with_sharding_constraint(
        #         key_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
        #     )
        #     value_states = with_sharding_constraint(
        #         value_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
        #     )
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_performer.__call__(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            segment_ids=segment_ids,
            causal_mask=causal_mask
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output, PartitionSpec(
                    ("dp", "fsdp"),
                    "sp" if attn_output.shape[1] != 1 else None,
                    "tp"
                )
            )
        attn_output = self.o_proj(attn_output)

        outputs = (
            attn_output, attentions.attention_weights
        ) if output_attentions else (
            attn_output,
        )
        return outputs


class FlaxDeepseekV2DecoderLayer(nn.Module):
    config: DeepseekV2Config
    layer_idx: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        config = self.config
        layer_idx = self.layer_idx
        self.hidden_size = config.hidden_size

        attn_block = FlaxDeepseekV2Attention
        mlp_block = FlaxDeepseekV2MLP
        mlp_moe_block = FlaxDeepseekV2MoE

        if self.config.gradient_checkpointing != "":
            # hidden_states: chex.Array,
            # freq_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: chex.Array,
            # position_ids: chex.Array,
            # causal_mask: chex.Array,
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # init_cache: bool = False,
            # output_attentions: bool = False,
            # fcm_mask = None,
            attn_block = re_mat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1, 3, 4, 6, 7, 8)
            )
            mlp_block = re_mat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )

            mlp_moe_block = re_mat(
                mlp_moe_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )

        self.self_attn = attn_block(
            config=config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.mlp = (
            mlp_moe_block(
                config=config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            if (
                    config.n_routed_experts is not None
                    and layer_idx >= config.first_k_dense_replace
                    and layer_idx % config.moe_layer_freq == 0
            )
            else mlp_block(
                config=config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

    def forward(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            causal_mask: chex.Array,
            position_ids: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = True
    ) -> Tuple[
        chex.Array, Optional[chex.Array]
    ]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            freq_cis,
            attention_mask,
            position_ids,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
            None
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.mlp(
                hidden_states,
                deterministic,
            )
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs  # type:ignore


class FlaxDeepseekV2DecoratorCollection(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]
    ] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxDeepseekV2DecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                layer_idx=i,
                name=str(i)
            ) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            causal_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # hidden_states: chex.Array,
            # freq_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: chex.Array,
            # causal_mask: chex.Array,
            # position_ids: chex.Array,
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # init_cache: bool = False,
            # output_attentions: bool = True

            output = layer(
                hidden_states,
                freq_cis,
                attention_mask,
                causal_mask,
                position_ids,
                None,
                deterministic,
                init_cache,
                output_attentions
            )
            hidden_states = output[0]

            if output_attentions:
                output_attentions += (output[1],)

        return hidden_states, all_hidden_states, all_attentions


class FlaxDeepseekV2Module(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.layers = FlaxDeepseekV2DecoratorCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm = DeepseekV2RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        initial_rope_kwargs = {}
        method = None
        if self.config.rope_scaling is not None:
            scaling_type = self.config.rope_scaling["type"]
            method = scaling_type
            if scaling_type != "yarn":
                initial_rope_kwargs = dict(scaling_factor=self.config.rope_scaling["factor"])
            else:
                initial_rope_kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                initial_rope_kwargs["scaling_factor"] = self.config.rope_scaling["factor"]
        self.freq_cis = init_deepseek_rotary_embedding(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            max_position_embeddings=(
                getattr(
                    self.config,
                    "freq_max_position_embeddings",
                    self.config.max_position_embeddings
                )
            ),
            base=self.config.rope_theta,
            method=method,  # type:ignore
            kwargs=initial_rope_kwargs
        )
        self.causal_mask = flax.linen.make_causal_mask(
            jnp.ones(
                (
                    1,
                    getattr(
                        self.config,
                        "c_max_position_embeddings",
                        self.config.max_position_embeddings
                    )
                ),
                dtype="bool"
            ),
            dtype="bool"
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            inputs_embeds: chex.Array = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> typing.Union[Tuple[chex.Array, ...], FlaxBaseModelOutput]:
        """
        The __call__ function is the main function of a Flax model.
        It takes in input_ids, attention_mask, and position_ids as inputs to the model.
        The output is a tuple containing: last hidden state (hidden states), all hidden states (if output_hidden_states=True), attentions (if output attentions=True).


        :param self: Represent the instance of the class
        :param input_ids: chex.Array: Pass in the input ids
        :param attention_mask: chex.Array: Mask out the attention weights for certain tokens
        :param position_ids: chex.Array: Determine the position of each token in a sequence
        :param deterministic: bool: Determine whether to use dropout or not
        :param inputs_embeds: chex.Array: Pass in the embedding of the input_ids
        :param init_cache: bool: Initialize the cache for the decoder
        :param output_attentions: bool: Determine whether to return the attention weights or not
        :param output_hidden_states: bool: Return all hidden states or just the last one
        :param return_dict: bool: Return a dictionary of the outputs or not
        :param : Determine whether the model is in training mode or not
        :return: A tuple of the hidden states, all hidden states, and attentions

        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        if attention_mask.ndim == 2:
            b, s = attention_mask.shape
            attention_mask = attention_mask.reshape(b, 1, 1, s)

        outputs = self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freq_cis=self.freq_cis,
            init_cache=init_cache,
            output_attentions=output_attentions,
            deterministic=deterministic,
            causal_mask=self.causal_mask
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(value for value in outputs if value is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class DeepseekV2PreTrainedModel(EasyDeLFlaxPretrainedModel):
    config_class: DeepseekV2Config = DeepseekV2Config
    module_class: nn.Module = None
    base_model_prefix = "model"

    def __init__(
            self,
            config: DeepseekV2Config,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            input_shape: Tuple[int, int] = (1, 1),
            seed: int = 0,
            _do_init: bool = False,
            **kwargs
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs
        )

        super().__init__(
            dtype=dtype, _do_init=_do_init,
            module=module, config=config, input_shape=input_shape,
            seed=seed,
        )

    def init_weights(
            self,
            rng: jax.random.PRNGKey,
            input_shape: Tuple,
            params: FrozenDict = None
    ) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.
        It takes in a rng, which is a random number generator key that can be used to generate random numbers.
        The input_shape parameter specifies the shape of the inputs that will be fed into this model.
        The params parameter allows you to pass in pre-trained weights for your model, if you have them available.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Initialize the input_ids, attention_mask and position_ids
        :param params: flax.core.FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters
        """

        self.config.initialization_of_moe = True
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1], dtype="i4"),
            input_shape,
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(
                input_shape + (self.config.hidden_size,))
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
            module_init_outputs = self.module.init(
                rngs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=False
            )
        random_params = module_init_outputs["params"]

        self.config.initialization_of_moe = False
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
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        """
        The __call__ function is the main function of a JAX module.
        It takes as input:
        - The parameters of the model (self.params)
        - The inputs to the model (input_ids, attention_mask, position_ids)
        - Whether we are training (train=True/False) and whether we want to return all hidden states and
        attentions weights at each layer in addition to just the last layer output (output_hidden_states=True/False).

        :param self: Represent the instance of the class
        :param input_ids: Pass the input sequence to the model
        :param attention_mask: Mask out the padding tokens
        :param position_ids: Specify the position of each token in the sequence
        :param params: dict: Pass in the parameters of the model
        :param past_key_values: dict: Pass the past key values to the model
        :param dropout_rng: jax.random.PRNGKey: Pass in a random number generator key to the model
        :param train: bool: Determine whether to use dropout or not
        :param output_attentions: Optional[bool]: Determine whether to return the attention weights
        :param output_hidden_states: Optional[bool]: Determine whether to return the hidden states of all layers
        :param return_dict: Optional[bool]: Return a dictionary of the outputs
        :param add_params_field: bool: Add a params field to the inputs dictionary
        :return: A tuple of (last_hidden_state, past_key_values)

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[
                                            None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rng_s = {}
        if dropout_rng is not None:
            rng_s["dropout"] = dropout_rng

        inputs = {
            "params": params or self.params} if add_params_field else params or self.params

        if self.config.bits is not None:
            rng_s['params'] = jax.random.key(0)
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),  # input_ids: chex.Array
            # attention_mask: Optional[chex.Array] = None
            jnp.array(attention_mask, dtype="i4"),
            # position_ids: Optional[chex.Array] = None
            jnp.array(position_ids, dtype="i4"),
            None,  # inputs_embeds: Optional[chex.Array] = None
            output_attentions,  # output_attentions: Optional[bool] = None
            # output_hidden_states: Optional[bool] = None
            output_hidden_states,
            # output_router_logits: Optional[bool] = None
            output_router_logits,
            False,  # init_cache: bool = False
            not train,  # deterministic: bool = True
            return_dict,  # return_dict: bool = True
            rngs=rng_s,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + \
                      (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxDeepseekV2ForCausalLMModule(nn.Module):
    config: DeepseekV2Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.model = FlaxDeepseekV2Module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = nn.Linear(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            inputs_embeds: chex.Array = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        """
            The __call__ function is the main function of a Flax module. It defines how the model will be called,
            and what it returns. In this case, we are calling our Transformer model with input_ids and attention_mask
            as inputs (these are defined in __init__). We also have some optional arguments that can be passed to
            the call function: deterministic (whether to use dropout), inputs_embeds (if you want to pass your own embeddings),
            output_attentions and output_hidden states which return additional outputs from the transformer layers if set True. Finally,

            :param self: Refer to the object itself
            :param input_ids: chex.Array: Pass in the input tokens
            :param attention_mask: chex.Array: Mask out the padding tokens
            :param position_ids: chex.Array: Specify the position of each token in the sequence
            :param deterministic: bool: Determine whether to use dropout in the model
            :param inputs_embeds: chex.Array: Pass in the embeddings of the input tokens
            :param init_cache: bool: Initialize the cache for the decoder
            :param output_attentions: bool: Return the attention weights
            :param output_hidden_states: bool: Return the hidden states of all layers
            :param return_dict: bool: Return a dictionary of the outputs or just the logits
            :param : Determine whether to return the logits or not
            :return: A tuple of (lm_logits, hidden_states, attentions)

        """
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            inputs_embeds=inputs_embeds,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["embed_tokens"]["embedding"]
            shared_kernel = fjformer.linen.linen.control_quantization(shared_kernel, self.param_dtype).T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        # lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxDeepseekV2Model(DeepseekV2PreTrainedModel):
    module_class = FlaxDeepseekV2Module


class FlaxDeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    module_class = FlaxDeepseekV2ForCausalLMModule

    def set_input_embeddings(self, value):
        self.module.model.embed_tokens = value

    def get_input_embeddings(self):
        return self.module.model.embed_tokens

    def set_decoder(self, decoder):
        self.module.model = decoder

    def get_decoder(self):
        return self.module.model

    def get_output_embeddings(self):
        return self.module.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        """
        The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

        :param self: Access variables that belong to the class
        :param input_ids: Pass in the input tokens
        :param max_length: Set the length of the sequence to be generated
        :param attention_mask: Optional[chex.Array]: Mask the attention weights
        :return: A dictionary of the past_key_values, attention_mask and position ids

        """
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones(
            (batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[
                                            None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
