import functools
import math
from typing import List, Optional, Tuple, Union

import chex
import flax.struct
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxMaskedLMOutput

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.dbrx.dbrx_configuration import (
    DbrxAttentionConfig as DbrxAttentionConfig,
)
from easydel.models.dbrx.dbrx_configuration import (
    DbrxConfig as DbrxConfig,
)
from easydel.models.dbrx.dbrx_configuration import (
    DbrxFFNConfig as DbrxFFNConfig,
)

# easydel.modules
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    control_mlp_sharding,
    precompute_freqs_cis,
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


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class DbrxAttention(BaseAttentionModule):
    def __init__(
        self,
        config: DbrxConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.num_attention_heads = self.config.n_heads
        self.num_key_value_heads = self.config.attn_config.kv_n_heads

        self.hidden_size = config.d_model
        self.head_dim = self.config.d_model // self.config.n_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.num_attention_heads == self.config.attn_config.kv_n_heads
        self.Wqkv = nnx.Linear(
            config.d_model,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            self.num_attention_heads * self.head_dim,
            config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.rotary = functools.partial(apply_rope, dtype=dtype)
        self.attention_module: FlexibleAttentionModule = FlexibleAttentionModule(
            mesh=config.mesh,
            attn_mechanism=config.attn_mechanism,
            sm_scale=1 / math.sqrt(self.head_dim),
            num_attention_heads=config.num_attention_heads,
            head_dims=self.head_dim,
            precision=precision,
            base_config=config,
        )
        self.resid_dropout = nnx.Dropout(
            rate=config.resid_pdrop,
            rngs=rngs,
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(self, query, key, freqs_cis, position_ids):
        """
        The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            query: Calculate the attention weights
            key: Calculate the attention
            value: Compute the attention weights
            freqs_cis: Calculate the frequency of each word in the vocabulary
            position_ids: Identify the position of each token in the sequence

        Returns:
            A tuple of 2 tensors: query, key
        """
        query, key = self._transpose_sequence_head(query, key)
        query, key = self.rotary(
            position_ids=position_ids,
            query=query,
            key=key,
            freqs_cis=freqs_cis,
        )
        return self._transpose_sequence_head(query, key)

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        """
        The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        Args:
            self: Access variables that belong to the class
            hidden_states: (chex.Array): Pass the hidden states of the previous layer
            freqs_cis: (Tuple[chex.Array, chex.Array]),: Pass in the frequency coefficients for each position
            attention_mask: (chex.Array): Mask out certain tokens in the input sequence
            past_key_values: (Optional(KVCache)): Past key and values used for generation
            position_ids: (Optional(chex.Array)): Determine the position of each token in a sequence

        Returns:
            A tuple of two arrays HiddenState and attentionWeight
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_states = self.Wqkv(hidden_states)
        if self.config.attn_config.clip_qkv is not None:
            qkv_states = qkv_states.clip(
                min=-self.config.attn_config.clip_qkv,
                max=self.config.attn_config.clip_qkv,
            )

        query_size = self.hidden_size
        key_size = self.num_key_value_heads * self.head_dim

        query_states, key_value_states = jnp.split(qkv_states, [query_size], axis=2)
        key_states, value_states = jnp.split(key_value_states, [key_size], axis=2)
        query_states = query_states.reshape(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states, key_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
        )

        if past_key_values is not None:
            past_key_values.update(key_states=key_states, value_states=value_states)
            key_length, value_states, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        key_states, value_states = self.repeat_key_value(
            key_states,
            value_states,
            self.num_key_value_groups,
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

        attentions = self.attention_module.__call__(
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
        attn_output = self.resid_dropout(self.out_proj(attn_output))
        return attn_output, attentions.attention_weights


class DbrxNormAttentionNorm(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.norm_1 = nnx.LayerNorm(
            num_features=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.attn = DbrxAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.norm_2 = nnx.LayerNorm(
            num_features=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )

        self.dropout = flax.linen.Dropout(config.resid_pdrop)

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            segment_ids=segment_ids,
            past_key_values=past_key_values,
        )

        hidden_states = self.dropout(
            hidden_states,
        )
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states)

        return residual_states, hidden_states, attn_weights


class DbrxExpertGLU(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        shape = (
            self.config.ffn_config.moe_num_experts
            * self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        init_fn = nnx.initializers.normal(dtype=self.dtype)
        self.w1 = nnx.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.v1 = nnx.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.w2 = nnx.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

    def __call__(self, hidden_states: chex.Array, expert_idx: int) -> chex.Array:
        expert_shape = (
            self.config.ffn_config.moe_num_experts,
            self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        expert_w1 = self.w1.value.reshape(expert_shape)[expert_idx]
        expert_v1 = self.v1.value.reshape(expert_shape)[expert_idx]
        expert_w2 = self.w2.value.reshape(expert_shape)[expert_idx]

        x1 = jax.lax.batch_matmul(
            hidden_states,
            jnp.expand_dims(expert_w1.T, 0),
            precision=self.precision,
        )
        x2 = jax.lax.batch_matmul(
            hidden_states,
            jnp.expand_dims(expert_v1.T, 0),
            precision=self.precision,
        )
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = jax.lax.batch_matmul(
            x1,
            jnp.expand_dims(expert_w2, 0),
            precision=self.precision,
        )
        return x1


class DbrxExperts(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.mlp = DbrxExpertGLU(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        weights: chex.Array,
        top_weights: chex.Array,
        top_experts: chex.Array,
    ):
        final_hidden_state = jnp.zeros_like(hidden_states)
        for index in range(self.config.ffn_config.moe_num_experts):
            output_moe_layer = self.mlp(hidden_states, index)
            final_hidden_state += (
                jnp.sum(jnp.multiply(index == top_experts, top_weights), axis=-1)[
                    :, :, None
                ]
                * output_moe_layer
            )
        return final_hidden_state


class DbrxRouter(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.precision = precision
        self.hidden_size = self.config.d_model
        self.moe_num_experts = self.config.ffn_config.moe_num_experts
        self.moe_top_k = self.config.ffn_config.moe_top_k
        self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
        self.moe_normalize_expert_weights = (
            self.config.ffn_config.moe_normalize_expert_weights
        )
        self.uniform_expert_assignment = (
            self.config.ffn_config.uniform_expert_assignment
        )

        self.layer = nnx.Linear(
            self.hidden_size,
            self.moe_num_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )

    def jitter(self, x: chex.Array) -> chex.Array:
        if self.moe_jitter_eps is None:
            raise RuntimeError("The router does not have moe_jitter_eps set.")
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = jax.random.normal(self.rngs.params(), x.shape, dtype=x.dtype)
        return low + noise * (high - low)

    def __call__(
        self, x: chex.Array, deterministic: bool = True
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        if not deterministic and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        weights = self.layer(x.astype(jnp.promote_types(self.dtype, jnp.float32)))
        weights = jax.nn.softmax(
            weights.astype(jnp.promote_types(self.dtype, jnp.float32))
        )
        top_weights, top_experts = jax.lax.top_k(weights, self.moe_top_k)

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / jnp.linalg.norm(
                top_weights,
                ord=int(self.moe_normalize_expert_weights),
                axis=-1,
                keepdims=True,
            )

        if self.uniform_expert_assignment:
            top_experts = jax.lax.stop_gradient(
                (
                    jnp.arange(
                        0,
                        jnp.prod(
                            jnp.asarray(top_experts.shape, dtype=jnp.int32),
                            dtype=jnp.int32,
                        ),
                        dtype=top_experts.dtype,
                    )
                    % self.moe_num_experts
                ).reshape(top_experts.shape)
            )

        weights = weights.astype(x.dtype)
        top_weights = top_weights.astype(x.dtype)
        return weights, top_weights, top_experts


class DbrxFFN(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.precision = precision
        self.router = DbrxRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.experts = DbrxExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: chex.Array, deterministic: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        hidden_states = control_mlp_sharding(
            hidden_states,
            self.config.partition_axis,
        )
        weights, top_weights, top_experts = self.router(
            hidden_states,
            deterministic=deterministic,
        )
        out = self.experts(
            hidden_states,
            weights,
            top_weights,
            top_experts,
        )
        return out, weights


class DbrxBlock(nnx.Module):
    def __init__(
        self,
        config: DbrxConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.precision = precision
        self.hidden_size = self.config.d_model
        self.resid_pdrop = self.config.resid_pdrop
        self.norm_attn_norm = DbrxNormAttentionNorm(
            config=config,
            dtype=dtype,
            layer_idx=layer_idx,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn = DbrxFFN(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
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
        resid_states, hidden_states, self_attn_weights = self.norm_attn_norm(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
        )

        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = resid_states + hidden_states

        return hidden_states, self_attn_weights, router_logits


class DbrxModel(BaseNNXModule):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config: DbrxConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.precision = precision
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.emb_pdrop = self.config.emb_pdrop

        self.wte = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = [
            DbrxBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=i,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ]
        self.norm_f = nnx.LayerNorm(
            config.d_model,
            use_bias=False,
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
            initial_rope_kwargs = dict(rope_type="none")
            if config.rope_scaling is not None:
                scaling_type = config.rope_scaling["type"]
                scaling_factor = config.rope_scaling["factor"]
                initial_rope_kwargs = dict(
                    scaling_factor=scaling_factor, rope_type=scaling_type
                )
            self._freqs_cis = precompute_freqs_cis(
                max_position_embeddings=(
                    getattr(
                        self.config,
                        "freq_max_position_embeddings",
                        self.config.max_position_embeddings,
                    )
                ),
                dim=config.hidden_size // config.num_attention_heads,
                base=config.rope_theta,
                **initial_rope_kwargs,
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
        output_router_logits: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        """The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
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
            output_router_logits: bool: Determine whether to return stack of router logits
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of: predictions
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        else:
            raise ValueError(
                "you should specify inputs_embeds or input_ids one of them"
            )
        batch_size, sequence_length, _ = inputs_embeds.shape
        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :sequence_length, :]
            )

        inputs_embeds = (
            inputs_embeds + extra_embedding
            if extra_embedding is not None
            else inputs_embeds
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        hidden_states = inputs_embeds
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, self_attn_weights, router_logits = block(
                hidden_states=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                freqs_cis=self.freqs_cis,
                past_key_values=past_key_values[idx],
            )

            if output_attentions:
                all_self_attns += (self_attn_weights,)
            if output_router_logits:
                all_router_logits += (router_logits,)
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class DbrxForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.rngs = rngs
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.transformer = DbrxModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            dtype=self.dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        inputs_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: bool = True,
    ) -> MoeCausalLMOutput | Tuple:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.wte.embedding.value.T
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        batch_size, seq_length, hd = logits.shape
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=tuple(  # type:ignore
                    [
                        logit.reshape(batch_size * seq_length, -1)
                        for logit in outputs.router_logits
                    ]  # type:ignore
                ),
                num_experts=self.config.ffn_config.moe_num_experts,
                top_k=self.config.ffn_config.moe_top_k,
                attention_mask=attention_mask,
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef
        if not return_dict:
            outputs = (logits,) + tuple(
                v
                for v in [
                    aux_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                ]
                if v is not None
            )
            return outputs

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @property
    def can_generate(self):
        return True
