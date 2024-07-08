import functools
import math
import typing
from typing import List, Optional, Tuple, Union

import chex
import flax
import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxMaskedLMOutput,
)

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.common import RMSNorm
from easydel.models.deepseek_v2.deepseek_configuration import (
    DeepseekV2Config as DeepseekV2Config,
)
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    block_wise_ffn,
    control_mlp_sharding,
    with_sharding_constraint,
)
from easydel.models.modelling_utils import BaseNNXModule


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


def yarn_get_mscale(scale=1.0, mscale=1.0):
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
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get(
            "scaling_factor"
        )
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
            inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))

        else:
            inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
        t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get(
            "scaling_factor"
        )

        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)
    elif method == "yarn":
        scaling_factor = kwargs.get("scaling_factor", 1.0)
        original_max_position_embeddings = kwargs.get(
            "original_max_position_embeddings", 4096
        )
        beta_fast = kwargs.get("beta_fast", 32)
        beta_slow = kwargs.get("beta_slow", 1)
        mscale = kwargs.get("mscale", 1)
        mscale_all_dim = kwargs.get("mscale_all_dim", 0)
        freq_extra = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        freq_inter = 1.0 / (
            scaling_factor * base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).astype(
            "float32"
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)

        freqs = jnp.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
        )

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return (jnp.sin(emb) * _mscale).astype("float32"), (
            jnp.cos(emb) * _mscale
        ).astype("float32")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
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


class DeepseekV2MLP(nnx.Module):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        linear = functools.partial(
            nnx.Linear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(),
        )
        self.gate_proj = linear(
            hidden_size or config.hidden_size,
            intermediate_size or config.intermediate_size,
            rngs=rngs,
        )
        self.up_proj = linear(
            hidden_size or config.hidden_size,
            intermediate_size or config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = linear(
            intermediate_size or config.intermediate_size,
            hidden_size or config.hidden_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: chex.Array,
    ):
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MoEGate(nnx.Module):

    def __init__(
        self,
        config: DeepseekV2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
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
        self.kernel = nnx.Param(
            nnx.initializers.kaiming_uniform(
                dtype=self.param_dtype,
            )(
                rngs.params(),
                (
                    self.n_routed_experts,
                    self.gating_dim,
                ),
            ),
        )

    def __call__(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        logits = jax.lax.batch_matmul(
            hidden_states.astype(jnp.float32),
            self.kernel.astype(jnp.float32),
            precision=self.precision,
        )
        if self.scoring_func == "softmax":
            scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "gready":
            topk_kernel, topk_idx = jax.lax.top_k(scores, k=self.top_k)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.reshape(bsz * seq_len, self.n_group, -1).max(
                axis=-1
            )  # [n, n_group]

            # Find the indices of the top k scores in each group
            top_k_indices = lax.top_k(group_scores, self.topk_group)[
                1
            ]  # [n, topk_group]

            # Initialize a mask with zeros
            group_mask = jnp.zeros_like(group_scores)  # [n, n_group]

            # Update the mask: this is a bit tricky in JAX as there is no direct scatter function
            n_indices = jnp.arange(group_mask.shape[0])[:, None]
            group_mask = group_mask.at[n_indices, top_k_indices].set(1)  # [n, n_group]

            # Expand and reshape the group_mask
            score_mask = jnp.repeat(
                group_mask[:, :, None], self.n_routed_experts // self.n_group, axis=2
            )
            score_mask = score_mask.reshape(bsz * seq_len, -1)  # [n, e]

            # Apply the mask to scores
            masked_scores = jnp.where(score_mask, scores, 0.0)  # [n, e]

            # Compute the top k scores after masking
            topk_kernel, topk_idx = lax.top_k(masked_scores, self.top_k)
        else:
            raise ValueError()
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_kernel, axis=-1, keepdims=True) + 1e-20
            topk_kernel = topk_kernel / denominator
        else:
            topk_kernel = topk_kernel * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.alpha > 0.0:
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
                aux_loss = (
                    jnp.mean(
                        jnp.sum((ce * jnp.mean(scores_for_seq_aux, axis=-1)), axis=1)
                    )
                    * self.alpha
                )
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
        return topk_idx, topk_kernel, aux_loss


class DeepseekV2MoE(nnx.Module):

    def __init__(
        self,
        config: DeepseekV2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        config = self.config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0
        self.experts = [
            DeepseekV2MLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=config.moe_intermediate_size,
                rngs=rngs,
                layer_idx=i,
            )
            for i in range(config.n_routed_experts)
        ]
        self.gate = MoEGate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MoE(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=intermediate_size,
            )

    def __call__(
        self,
        hidden_states: chex.Array,
        e: bool = False,  # ignored !
    ):
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_kernel, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.reshape(-1)
        hidden_states = hidden_states.repeat(self.num_experts_per_tok, axis=0)
        # y = self.experts(hidden_states=hidden_states, flat_topk_idx=flat_topk_idx)

        y = jnp.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])

        y = (y.reshape(*topk_kernel.shape, -1) * jnp.expand_dims(topk_kernel, -1)).sum(
            axis=1
        )
        y = y.reshape(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class DeepseekV2Attention(BaseAttentionModule):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
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

        linear_class = functools.partial(
            nnx.Linear,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.q_a_proj = linear_class(
            self.hidden_size,
            config.q_lora_rank,
            use_bias=config.attention_bias,
            rngs=rngs,
        )
        self.q_a_layernorm = RMSNorm(
            config.q_lora_rank,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.q_b_proj = linear_class(
            self.hidden_size,
            self.num_heads * self.q_head_dim,
            use_bias=False,
            rngs=rngs,
        )

        self.kv_a_proj_with_mqa = linear_class(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            use_bias=config.attention_bias,
            rngs=rngs,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = linear_class(
            self.hidden_size,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            use_bias=False,
            rngs=rngs,
        )

        self.o_proj = linear_class(
            self.num_heads * self.q_head_dim,
            self.hidden_size,
            use_bias=config.attention_bias,
            rngs=rngs,
        )

        softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = self.softmax_scale * mscale * mscale

        self.attention_module: FlexibleAttentionModule = FlexibleAttentionModule(
            mesh=config.mesh,
            attn_mechanism=config.attn_mechanism,
            sm_scale=softmax_scale,
            num_attention_heads=config.num_attention_heads,
            head_dims=self.q_head_dim,
            precision=precision,
            base_config=config,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[:, :, :, : self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim :]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        k_pe = compressed_kv[
            :, :, :, self.kv_lora_rank : self.kv_lora_rank + self.qk_rope_head_dim
        ]
        compressed_kv = compressed_kv[:, :, :, : self.kv_lora_rank]

        k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .reshape(
                bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            .transpose(0, 2, 1, 3)
        )

        k_nope = kv[:, :, :, : self.qk_nope_head_dim]
        value_states = kv[
            :, :, :, self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim
        ]

        sin, cos = freqs_cis

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if past_key_values is not None:
            past_key_values.update(key_states=key_states, value_states=value_states)
            key_length, value_states, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        attention_bias = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_length]
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(
                    attention_mask.shape,
                    jnp.finfo(self.dtype).min,
                ).astype(self.dtype),
            )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_module(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            segment_ids=segment_ids,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.o_proj(attn_output)

        return attn_output, attentions.attention_weights


class DeepseekV2DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
        config = self.config
        layer_idx = self.layer_idx
        self.hidden_size = config.hidden_size

        attn_block = DeepseekV2Attention
        mlp_block = DeepseekV2MLP
        mlp_moe_block = DeepseekV2MoE

        # if self.config.gradient_checkpointing != "":

        # attn_block = re_mat(
        #     attn_block,
        #     policy=get_gradient_checkpoint_policy(
        #         self.config.gradient_checkpointing
        #     ),
        #     static_argnums=(
        #         1,
        #         4,
        #     ),
        # )
        # mlp_block = re_mat(
        #     mlp_block,
        #     policy=get_gradient_checkpoint_policy(
        #         self.config.gradient_checkpointing
        #     ),
        #     static_argnums=(),
        # )

        # mlp_moe_block = re_mat(
        #     mlp_moe_block,
        #     policy=get_gradient_checkpoint_policy(
        #         self.config.gradient_checkpointing
        #     ),
        #     static_argnums=(),
        # )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = (
            mlp_moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
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
            )
        else:
            feed_forward_hidden_states = self.mlp(
                hidden_states,
            )
        hidden_states = residual + feed_forward_hidden_states

        return (hidden_states, self_attn_weights)  # type:ignore


class DeepseekV2Model(BaseNNXModule):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            DeepseekV2DecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=i,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self._causal_mask = None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            config = self.config
            initial_rope_kwargs = {}
            method = None
            if config.rope_scaling is not None:
                scaling_type = config.rope_scaling["type"]
                method = scaling_type
                if scaling_type != "yarn":
                    initial_rope_kwargs = dict(
                        scaling_factor=config.rope_scaling["factor"]
                    )
                else:
                    initial_rope_kwargs = {
                        key: config.rope_scaling[key]
                        for key in [
                            "original_max_position_embeddings",
                            "beta_fast",
                            "beta_slow",
                            "mscale",
                            "mscale_all_dim",
                        ]
                        if key in config.rope_scaling
                    }
                    initial_rope_kwargs["scaling_factor"] = config.rope_scaling[
                        "factor"
                    ]
            self._freqs_cis = init_deepseek_rotary_embedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=(
                    getattr(
                        config,
                        "freq_max_position_embeddings",
                        config.max_position_embeddings,
                    )
                ),
                base=config.rope_theta,
                method=method,  # type:ignore
                kwargs=initial_rope_kwargs,
            )
        return self._freqs_cis

    @property
    def causal_mask(self):
        if self._causal_mask is None:
            self._causal_mask = nnx.make_causal_mask(
                jnp.ones(
                    (
                        1,
                        getattr(
                            self.config,
                            "causal_mask_max_position_embeddings",
                            self.config.max_position_embeddings,
                        ),
                    ),
                    dtype=jnp.bool,
                ),
                dtype=jnp.bool,
            )
        return self._causal_mask

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        inputs_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ) -> typing.Union[Tuple[chex.Array, ...], FlaxBaseModelOutput]:
        """
        The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
        and returns the output of the model. These optional arguments are passed as keyword arguments when calling a Flax model.

        Args:
            self: Represent the instance of the class
            input_ids: chex.Array: Pass in the input token ids
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Indicate the position of each token in a sequence
            inputs_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of: predictions
        """

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            ).astype(jnp.int32)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :seq_length, :]
            )

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        inputs_embeds = (
            inputs_embeds + extra_embedding
            if extra_embedding is not None
            else inputs_embeds
        )
        hidden_states = inputs_embeds

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weight = block(
                hidden_states=hidden_states,
                freqs_cis=self.freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values[idx],
            )
            if output_attentions:
                all_attentions += (attn_weight,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, all_attentions, all_attentions)
                if value is not None
            )

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_attentions,
            attentions=all_attentions,
        )


class DeepseekV2ForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config: DeepseekV2Config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
        self.model = DeepseekV2Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=False,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        inputs_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        """
        The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
        and returns the output of the model. These optional arguments are passed as keyword arguments when calling a Flax model.

        Args:
            self: Represent the instance of the class
            input_ids: chex.Array: Pass in the input token ids
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Indicate the position of each token in a sequence
            inputs_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of (lm_logits, hidden_states, attentions)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            extra_embedding=extra_embedding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

    @property
    def can_generate(self):
        return True
