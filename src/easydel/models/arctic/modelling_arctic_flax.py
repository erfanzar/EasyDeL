import functools
import math
from typing import List, Optional, Tuple, Union

import chex
import flax
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec

from easydel.models.arctic.arctic_configuration import ArcticConfig
from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.common import RMSNorm

# easydel.modules
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    get_gradient_checkpoint_policy,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.modeling_flax_outputs import (
    FlaxMaskedLMOutput,
)
from src.easydel.models.modelling_utils import BaseNNXModule


@flax.struct.dataclass
class MoeModelOutput:
    last_hidden_state: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    all_router_losses: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
    aux_loss: Optional[chex.Array] = None
    all_router_losses: Optional[Tuple[chex.Array]] = None


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class ArcticAttention(BaseAttentionModule):
    def __init__(
        self,
        config: ArcticConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        self.config: ArcticConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nnx.Linear(
            self.hidden_size,
            config.num_attention_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=getattr(self.config, "attention_bias", False),
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=getattr(self.config, "attention_bias", False),
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=getattr(self.config, "attention_bias", False),
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
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

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        """The __call__ function is the main function of a JAX module. It defines how the module behaves when called
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
            A tuple of two arrays
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        query_states, key_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
        )

        if past_key_values is not None:
            past_key_values.update(key_states=key_states, value_states=value_states)
            key_states, value_states, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

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

        attentions = self.attention_module(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            deterministic=self.resid_dropout.deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            segment_ids=segment_ids,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    self.config.partition_axis.batch_axis,
                    (
                        self.config.partition_axis.sequence_axis
                        if attn_output.shape[1] != 1
                        else None
                    ),
                    self.config.partition_axis.hidden_state_axis,
                ),
            )
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, attentions.attention_weights


class ArcticMLP(nnx.Module):
    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        is_residual_mlp: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: ArcticConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.hidden_dim = config.hidden_size
        self.ffn_dim = (
            config.intermediate_size if not is_residual_mlp else self.hidden_dim
        )
        linear = functools.partial(
            nnx.Linear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(),
        )
        self.w1 = linear(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w3 = linear(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.w2 = linear(self.hidden_dim, self.ffn_dim, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: chex.Array):
        """
        The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        Args:
            self: Represent the instance of the class
            hidden_states: jnp.ndarray: Pass in the input to the layer

        Returns:
            A tensor that is the result of applying a dropout function to `hidden_states`
        """
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class ArcticMoE(nnx.Module):
    def __init__(
        self,
        config: ArcticConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.top_k = config.num_experts_per_tok
        self.is_moe_layer = (layer_idx + 1) % config.moe_layer_frequency == 0

        if self.is_moe_layer:
            self.gate = nnx.Linear(
                config.hidden_size,
                config.num_local_experts,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.normal(),
                rngs=rngs,
            )
            self.experts = [
                ArcticMLP(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(config.num_local_experts)
            ]
        else:
            self.mlp = ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                is_residual_mlp=False,
                rngs=rngs,
            )

    def _call_moe(
        self,
        hidden_states: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

        router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
            jnp.promote_types(self.dtype, jnp.float32)
        )
        routing_weights, selected_experts = jax.lax.top_k(
            router_logits, k=self.config.num_experts_per_tok
        )
        routing_weights = jax.nn.softmax(
            routing_weights.astype(
                jnp.promote_types(self.dtype, jnp.float32),
            ),
            axis=-1,
        )
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_local_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(selected_experts == index, routing_weights), axis=-1
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp

        return final_hidden_state, auxiliary_load_balancing_loss_func(
            (router_logits,),  # type:ignore
            self.num_experts,
            self.top_k,
            None,
        )

    def __call__(self, hidden_states: chex.Array):
        if self.is_moe_layer:
            return self._call_moe(hidden_states=hidden_states)
        return self.mlp(hidden_states), jnp.array(0.0, dtype=hidden_states.dtype)


class ArcticSparseMoeBlock(nnx.Module):
    """
    This implementation isstrictly equivalent to standard MoE
    with full capacity (no dropped tokens). It's faster since it
    formulates MoE operations in terms of block-sparse operations
     to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(),
            rngs=rngs,
        )

        self.experts = [
            ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.num_local_experts)
        ]

    def __call__(
        self,
        hidden_states: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
            jnp.promote_types(self.dtype, jnp.float32)
        )
        routing_weights, selected_experts = jax.lax.top_k(
            router_logits, k=self.config.num_experts_per_tok
        )
        routing_weights = jax.nn.softmax(
            routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
        )
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_local_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(selected_experts == index, routing_weights), axis=-1
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp

        return (
            final_hidden_state,
            router_logits,
        )


class ArcticDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: ArcticConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        attn_block = ArcticAttention
        mlp_block = ArcticSparseMoeBlock
        if self.config.gradient_checkpointing != "":
            attn_block = nnx.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1, 4, 5),
            )
            mlp_block = nnx.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(),
            )
        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.block_sparse_moe = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.parallel_attn_mlp_res = (
            self.config.parallel_attn_mlp_res and self.block_sparse_moe.is_moe_layer
        )
        if self.parallel_attn_mlp_res:
            self.residual_layernorm = RMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.residual_mlp = ArcticMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                is_residual_mlp=True,
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
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # hidden_states: chex.Array,
        # freqs_cis: Tuple[chex.Array, chex.Array],
        # attention_mask: chex.Array,
        # position_ids: chex.Array,
        # past_key_values: Optional[KVCache] = None,
        # segment_ids: Optional[chex.Array] = None,

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )

        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states
        if self.parallel_attn_mlp_res:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_residual = residual_attn + hidden_states
            # parallel mlp moe part
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_residual + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states

        return hidden_states, self_attn_weights, gate_loss


class ArcticModel(BaseNNXModule):
    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.layers = [
            ArcticDecoderLayer(
                layer_idx=layer_idx,
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
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
            initial_rope_kwargs = dict(rope_type="none")
            if self.config.rope_scaling is not None:
                scaling_type = self.config.rope_scaling["type"]
                scaling_factor = self.config.rope_scaling["factor"]
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
                dim=self.config.hidden_size // self.config.num_attention_heads,
                base=self.config.rope_theta,
                **initial_rope_kwargs,
            )
        return self._freqs_cis

    @property
    def causal_mask(self):
        if self._causal_mask is None:
            self._causal_mask = flax.linen.make_causal_mask(
                jnp.ones(
                    (
                        1,
                        getattr(
                            self.config,
                            "causal_mask_max_position_embeddings",
                            self.config.max_position_embeddings,
                        ),
                    ),
                    dtype="bool",
                ),
                dtype="bool",
            )
        return self._causal_mask

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        input_embeds: Optional[chex.Array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ) -> MoeModelOutput | Tuple:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_losses = ()

        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )
        if input_embeds is None and input_ids is not None:
            input_embeds = self.embed_tokens(input_ids.astype("i4"))
        else:
            raise ValueError("you should specify input_embeds or input_ids one of them")
        batch_size, sequence_length, _ = input_embeds.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :sequence_length, :]
            )
        hidden_states = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            (hidden_states, self_attn_weights, gate_loss) = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                freqs_cis=self.freqs_cis,
                past_key_values=past_key_values,
                segment_ids=segment_ids,
            )

            if output_attentions:
                all_self_attns += (self_attn_weights,)

            all_router_losses += (gate_loss,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_router_losses,
                ]
                if v is not None
            )
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_router_losses=all_router_losses,
        )

    def set_input_embeddings(self, value):
        self.module.embed_tokens = value

    def get_input_embeddings(self):
        return self.embed_tokens.value


class ArcticForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: ArcticConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.model = ArcticModel(
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
            kernel_init=nnx.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        input_embeds: Optional[chex.Array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        extra_embedding: Optional[jax.Array] = None,
        return_dict: bool = True,
    ) -> MoeCausalLMOutput | Tuple:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
            extra_embedding=extra_embedding,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
            logits = self.lm_head(outputs.hidden_states)
        else:
            logits = self.lm_head(outputs.hidden_states)

        aux_loss = sum(outputs[-1]) * self.config.router_aux_loss_coef
        if not return_dict:
            outputs = (logits,) + tuple(
                v
                for v in [
                    aux_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.all_router_losses,
                ]
                if v is not None
            )
            return outputs

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_router_losses=outputs.all_router_losses,
        )

    def set_input_embeddings(self, value):
        self.module.model.embed_tokens = value

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_decoder(self, decoder):
        self.module.model = decoder

    def get_decoder(self):
        return self.model

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    @property
    def can_generate(self):
        return True
