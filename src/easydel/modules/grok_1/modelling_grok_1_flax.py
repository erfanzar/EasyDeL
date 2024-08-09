import math
from typing import Optional, Tuple, Union

import chex
import flax.linen.partitioning
import flax.struct
import jax
import jax.numpy as jnp
from fjformer.functions import auxiliary_load_balancing_loss_func
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import Dense, combine_masks
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec

from easydel.modules.attention_module import FlexibleAttentionModule
from easydel.modules.common import RMSNorm as FlaxGrok1RMSNorm

# easydel.modules
from easydel.modules.flax_modeling_utils import (
    FlaxAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    get_dot_general_by_bits,
    get_gradient_checkpoint_policy,
    precompute_frequencies,
    with_sharding_constraint,
)
from easydel.modules.grok_1.grok_1_configuration import Grok1Config as Grok1Config
from easydel.modules.modeling_flax_outputs import FlaxMaskedLMOutput
from easydel.modules.modeling_utils import EDPretrainedModel

re_mat = flax.linen.partitioning.remat


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


class FlaxGrok1Embedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, frequencies, position_ids):
        sin, cos = frequencies

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxGrok1Attention(FlaxAttentionModule):
    config: Grok1Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.k_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.v_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.o_proj = Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )

        self.rotary = FlaxGrok1Embedding(self.dtype)
        self.attention_performer = FlexibleAttentionModule(
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.head_dim,
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            mesh=self.config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
            base_config=self.config,
        )
        self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (chex.Array): The hidden states with separate head dimensions.

        Returns:
            chex.Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(self, query, key, frequencies, position_ids):
        """
        Applies rotary positional embeddings to the query and key tensors.

        Args:
            query (chex.Array): Query tensor.
            key (chex.Array): Key tensor.
            frequencies (Tuple[chex.Array, chex.Array]): Tuple containing cosine and sine components for rotary embeddings.
            position_ids (chex.Array): Position indices for the tokens.

        Returns:
            Tuple[chex.Array, chex.Array]: The modified query and key tensors after applying rotary embeddings.
        """

        query, key = self._transpose_sequence_head(
            query,
            key,
        )
        query, key = self.rotary(
            position_ids=position_ids,
            query=query,
            key=key,
            frequencies=frequencies,
        )
        return self._transpose_sequence_head(query, key)

    def __call__(
        self,
        hidden_states: chex.Array,
        frequencies: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask: Optional[chex.Array] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
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
            frequencies=frequencies,
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states,
                value_states,
                query_states,
                attention_mask,
            )

        key_states, value_states = self.repeat_key_value(
            key_states, value_states, self.num_key_value_groups
        )

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_performer(
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
            causal_mask=causal_mask,
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
        attn_output = self.o_proj(attn_output)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (
            (attn_output, attentions.attention_weights)
            if output_attentions
            else (attn_output,)
        )
        return outputs


class FlaxGrok1BLockSparseMLP(nn.Module):
    config: Grok1Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.linear = Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.linear_1 = Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.linear_v = Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        Args:
            self: Represent the instance of the class
            x: jnp.ndarray: Pass in the input to the layer
            deterministic: bool: Determine whether to use dropout #
                IGNORED

        Returns:
            A tensor that is the result of applying a dropout function
            to x
        """

        x = control_mlp_sharding(x, self.config.partition_axis)
        return self.linear_1(nn.gelu(self.linear(x)) * self.linear_v(x))


class FlaxGrok1BlocKSparesTop2MLPCollection(nn.Module):
    config: Grok1Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxGrok1BLockSparseMLP(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i),
            )
            for i in range(self.config.num_experts)
        ]

    def __call__(
        self,
        selected_experts: chex.Array,
        hidden_states: chex.Array,
        routing_weights: chex.Array,
        batch_size: int,
        sequence_length: int,
        hidden_dim: int,
    ) -> chex.Array:
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.layers[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                    False,
                )
                if self.config.use_scan_mlp
                else self.layers[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(selected_experts == index, routing_weights), axis=-1
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp
        return final_hidden_state


class FlaxGrok1SparseMoeBlock(nn.Module):
    """This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    config: Grok1Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[None, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.gate = Dense(
            self.config.num_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
        )

        self.experts = FlaxGrok1BlocKSparesTop2MLPCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

    def __call__(
        self, hidden_states: chex.Array, e: bool = False  # Ignored
    ) -> Tuple[chex.Array, chex.Array]:
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        router_logits = self.gate(hidden_states).astype(
            jnp.promote_types(self.dtype, jnp.float32)
        )
        routing_weights, selected_experts = jax.lax.top_k(
            router_logits, k=self.config.num_experts_per_tok
        )
        routing_weights = jax.nn.softmax(
            routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
        )

        return (
            self.experts(
                selected_experts=selected_experts,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_dim=hidden_dim,
                hidden_states=hidden_states,
                routing_weights=routing_weights,
            ),
            router_logits,
        )


class FlaxGrok1DecoderLayer(nn.Module):
    config: Grok1Config
    layer_index: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        # hidden_states: chex.Array
        # frequencies: Tuple[chex.Array, chex.Array],
        # attention_mask: chex.Array
        # causal_mask: chex.Array
        # position_ids: chex.Array
        # deterministic: bool = True
        # init_cache: bool = False
        # output_attentions: bool = True

        attn_block = FlaxGrok1Attention
        mlp_block = FlaxGrok1SparseMoeBlock
        if self.config.gradient_checkpointing != "":
            attn_block = re_mat(
                attn_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1, 3, 4, 6, 7, 8),
            )
            mlp_block = re_mat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1,),
            )
        self.attn = attn_block(
            config=self.config,
            layer_index=self.layer_index,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.moe_block = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.pre_attn_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_attn_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.pre_moe_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_moe_norm = FlaxGrok1RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        frequencies: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
        """
        Forward pass of the attentionNrom module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights.
            output_router_logits (bool): If True, outputs router logits.
            fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            Tuple[chex.Array, chex.Array, Optional[chex.Array]]: A tuple containing the residual_states, hidden states, and the attention weights.
        """
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states, attention_weights = self.attn(
            hidden_states,
            frequencies,
            attention_mask,
            position_ids,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
        )

        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        hidden_states, router_logits = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class FlaxGrok1DecoderLayerCollection(nn.Module):
    config: Grok1Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.blocks = [
            FlaxGrok1DecoderLayer(
                layer_index=layer_index,
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(layer_index),
            )
            for layer_index in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: chex.Array,
        frequencies: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[chex.Array, chex.Array, Optional[chex.Array]]:
        """
        Forward pass of the attentionNrom module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights.
            output_router_logits (bool): If True, outputs router logits.
            output_hidden_states (bool): If True, outputs all of hidden states.
        Returns:
            Tuple[chex.Array, Optional[chex.Array], Optional[chex.Array], Optional[chex.Array]]:
                A tuple containing the hidden_states, all_attentions, all_hidden_states, all_router_logits.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng("fcm"),
                shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio,
            )
            fcm_mask = (
                jax.random.uniform(
                    self.make_rng("fcm"),
                    shape=(batch_size, 1, seq_length, seq_length),
                )
                > fcm_ratio
            )
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype("bool")
        else:
            fcm_mask = None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                init_cache=init_cache,
                frequencies=frequencies,
                causal_mask=causal_mask,
                deterministic=deterministic,
                segment_ids=segment_ids,
                fcm_mask=fcm_mask,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_router_logits:
            outputs += (all_router_logits,)
        return outputs


class Grok1PreTrainedModel(EDPretrainedModel):
    """
    Base class for Grok1 models providing initialization and configuration.

    Attributes:
        config_class (Grok1Config): The configuration class for the model.
        module_class (nn.Module): The class representing the model's architecture.
        base_model_prefix (str): The prefix for the base model parameters.
    """

    config_class: Grok1Config = Grok1Config
    module_class: nn.Module = None
    base_model_prefix = "model"

    # main_input_name = "input_ids"

    def __init__(
        self,
        config: Grok1Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        input_shape: Tuple[int, int] = (1, 1),
        seed: int = 0,
        _do_init: bool = False,
        **kwargs,
    ):
        """
        Initializes the pre-trained model with the given configuration.

        Args:
            config (Grok1Config): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for model parameters.
            precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
            input_shape (Tuple[int, int]): Shape of the input tensor.
            seed (int): Seed for random number generation.
            _do_init (bool): If True, initialize model weights.
            **kwargs: Additional keyword arguments.
        """
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs,
        )

        super().__init__(
            dtype=dtype,
            _do_init=_do_init,
            module=module,
            config=config,
            input_shape=input_shape,
            seed=seed,
        )

    def init_weights(
        self,
        rng: jax.random.PRNGKey,
        input_shape: Tuple,
        params: Optional[FrozenDict] = None,
    ) -> FrozenDict:
        """The init_weights function is used to initialize the weights of a model.
        It takes in a rng, which is a random number generator key that can be used to generate random numbers.
        The input_shape parameter specifies the shape of the inputs that will be fed into this model.
        The params parameter allows you to pass in pre-trained weights for your model, if you have them available.

        Args:
            self: Access variables that belong to the class
            rng: jax.random.PRNGKey: Initialize the weights of the model
            input_shape: Tuple: Initialize the input_ids, attention_mask
                and position_ids
            params: flax.core.FrozenDict: Pass in the parameters of a
                pre-trained model

        Returns:
            A frozendict of parameters
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
            module_init_outputs = self.module.init(
                rngs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=False,
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
        """
        Initializes the cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum length for the cache.

        Returns:
            dict: Initialized cache.
        """

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return init_variables["cache"]

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        params: dict = None,
        past_key_values: Optional[dict] = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_params_field: bool = False,
        **kwargs,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids (chex.Array): Input tensor containing token IDs.
            attention_mask (Optional[chex.Array]): Mask for attention.
            position_ids (Optional[chex.Array]): Positional indices.
            segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
            input_embeds (Optional[chex.Array]): embedding inputs to be used instead of input_ids.
            params (dict, optional): Parameters for the model.
            past_key_values (dict, optional): Past key and value states for caching.
            dropout_rng (jax.random.PRNGKey, optional): RNG key for dropout.
            train (bool): If True, the model is in training mode.
            output_attentions (Optional[bool]): If True, output attention weights.
            output_hidden_states (Optional[bool]): If True, output hidden states.
            output_router_logits (Optional[bool]): If True, output router logits.
            return_dict (Optional[bool]): If True, return a dictionary of outputs.
            add_params_field (bool): If True, include the parameters in the input dictionary.
            **kwargs: Additional arguments.

        Returns:
            Output type depends on the model configuration.
        """

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
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rng_s = {}
        if dropout_rng is not None:
            rng_s["dropout"] = dropout_rng

        inputs = (
            {"params": params or self.params}
            if add_params_field
            else params or self.params
        )

        if self.config.bits is not None:
            rng_s["params"] = jax.random.key(0)
        if past_key_values is not None:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            segment_ids=segment_ids,
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            init_cache=False,
            deterministic=not train,
            return_dict=return_dict,
            rngs=rng_s,
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


class FlaxGrok1Module(nn.Module):
    config: Grok1Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.layers = FlaxGrok1DecoderLayerCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.norm = FlaxGrok1RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        initial_rope_kwargs = dict(rope_type="none")
        if self.config.rope_scaling is not None:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor, rope_type=scaling_type
            )
        self.frequencies = precompute_frequencies(
            max_position_embeddings=self.config.granted_freq_max_position_embedding,
            dim=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rope_theta,
            **initial_rope_kwargs,
        )
        self.causal_mask = flax.linen.make_causal_mask(
            jnp.ones(
                shape=(1, self.config.granted_mask_max_position_embedding),
                dtype="bool",
            ),
            dtype="bool",
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> MoeModelOutput | Tuple:
        """
        Forward pass through the Grok1 module.

        Args:
            input_ids (chex.Array): Input tensor containing token IDs.
            attention_mask (chex.Array): Mask for attention.
            position_ids (chex.Array): Positional indices.
            segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
            input_embeds (Optional[chex.Array]): Embedded input tensor.
            output_attentions (Optional[bool]): If True, output attention weights.
            output_hidden_states (Optional[bool]): If True, output hidden states.
            output_router_logits (Optional[bool]): If True, output router logits.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.
            return_dict (bool): If True, return a dictionary of outputs.

        Returns:
            MoeModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )

        if input_embeds is None and input_ids is not None:
            input_embeds = self.wte(input_ids.astype("i4"))
        else:
            raise ValueError("you should specify input_embeds or input_ids one of them")
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
        collection_outputs = self.layers(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            causal_mask=self.causal_mask,
            frequencies=self.frequencies,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            init_cache=init_cache,
            deterministic=deterministic,
        )
        all_self_attns = None
        all_hidden_states = None
        all_router_logits = None
        hidden_states = collection_outputs[0]
        if output_attentions:
            all_self_attns = collection_outputs[1]
        if output_hidden_states:
            all_hidden_states = collection_outputs[2 if output_attentions else 1]
        if output_router_logits:
            all_router_logits = collection_outputs[-1]
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


class FlaxGrok1Model(Grok1PreTrainedModel):
    module_class = FlaxGrok1Module


class FlaxGrok1ForCausalLMModule(nn.Module):
    """
    Grok1 model for causal language modeling, including the language model head.

    Attributes:
        config (Grok1Config): Configuration object with model hyperparameters.
        dtype (jnp.dtype): Data type for the computations.
        param_dtype (jnp.dtype): Data type for the model parameters.
        precision (Optional[jax.lax.Precision]): Precision setting for JAX operations.
    """

    config: Grok1Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.model = FlaxGrok1Module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.lm_head = Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )

        self.output_multiplier_scale = self.config.output_multiplier_scale

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> MoeCausalLMOutput | Tuple:
        """
        Forward pass through the Grok1 module.

        Args:
            input_ids (chex.Array): Input tensor containing token IDs.
            attention_mask (chex.Array): Mask for attention.
            position_ids (chex.Array): Positional indices.
            segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
            input_embeds (Optional[chex.Array]): Embedded input tensor.
            output_attentions (Optional[bool]): If True, output attention weights.
            output_hidden_states (Optional[bool]): If True, output hidden states.
            output_router_logits (Optional[bool]): If True, output router logits.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.
            return_dict (bool): If True, return a dictionary of outputs.

        Returns:
            MoeCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            init_cache=init_cache,
            deterministic=deterministic,
            return_dict=True,
            segment_ids=segment_ids,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        logits = logits * self.output_multiplier_scale
        batch_size, seq_length, hd = logits.shape
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=tuple(
                    [
                        logit.reshape(batch_size * seq_length, -1)
                        for logit in outputs.router_logits
                    ]
                ),
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
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


class FlaxGrok1ForCausalLM(Grok1PreTrainedModel):
    module_class = FlaxGrok1ForCausalLMModule

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        attention_mask: Optional[chex.Array] = None,
    ):
        """The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

        Args:
            self: Access variables that belong to the class
            input_ids: Pass in the input tokens
            max_length: Set the length of the sequence to be generated
            attention_mask: Optional[chex.Array]: Mask the attention
                weights

        Returns:
            A dictionary of the past_key_values, attention_mask and
            position ids
        """
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask,
                attention_mask,
                (0, 0),
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
