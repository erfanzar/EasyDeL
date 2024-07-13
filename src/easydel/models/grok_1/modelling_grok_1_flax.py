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

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.common import RMSNorm as FlaxGrok1RMSNorm

# easydel.modules
from easydel.models.flax_modelling_utils import (
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.grok_1.grok_1_configuration import Grok1Config as Grok1Config
from easydel.models.modeling_flax_outputs import MoeCausalLMOutput, MoeModelOutput
from easydel.models.modelling_utils import BaseNNXModule

re_mat = flax.linen.partitioning.remat


def apply_rope(
    query,
    key,
    freqs_cis,
    position_ids,
    dtype: jnp.dtype = jnp.float32,
):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class Grok1Attention(BaseAttentionModule):
    def __init__(
        self,
        config: Grok1Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = nnx.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.rotary = functools.partial(apply_rope, self.dtype)
        self.attention_module = FlexibleAttentionModule(
            head_dims=self.head_dim,
            precision=self.precision,
            attn_mechanism=self.config.attn_mechanism,
            mesh=self.config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
        )
        self.resid_dropout = nnx.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(self, query, key, freqs_cis, position_ids):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            query: Calculate the attention weights
            key: Calculate the attention
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


class Grok1BLockSparseMLP(nnx.Module):
    def __init__(
        self,
        config: Grok1Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.linear = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.linear_1 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.linear_v = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        Args:
            self: Represent the instance of the class
            hidden_states: jnp.ndarray: Pass in the input to the layer
        Returns:
            A tensor that is the result of applying a dropout function
            to hidden_states
        """

        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return self.linear_1(
            nnx.gelu(self.linear(hidden_states)) * self.linear_v(hidden_states)
        )


class Grok1SparseMoeBlock(nnx.Module):
    """This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        config: Grok1Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(),
            rngs=rngs,
        )

        self.experts = [
            Grok1BLockSparseMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_experts)
        ]

    def __call__(
        self, hidden_states: chex.Array, e: bool = False  # Ignored
    ) -> Tuple[chex.Array, chex.Array]:
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)

        router_logits = self.gate(hidden_states).astype(
            jnp.promote_types(self.dtype, jnp.float32)
        )
        routing_weights, selected_experts = jax.lax.top_k(
            router_logits,
            k=self.config.num_experts_per_tok,
        )
        routing_weights = jax.nn.softmax(
            routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)),
            axis=-1,
        )

        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.layers[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.layers[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(
                    jnp.multiply(selected_experts == index, routing_weights),
                    axis=-1,
                )[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp

        return (
            final_hidden_state,
            router_logits,
        )


class Grok1DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: Grok1Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = Grok1Attention
        mlp_block = Grok1SparseMoeBlock
        # if self.config.gradient_checkpointing != "":
        #     attn_block = re_mat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(3, 5, 6, 7, 8),
        #     )
        #     mlp_block = re_mat(
        #         mlp_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(1,),
        #     )
        self.attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.moe_block = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pre_attn_norm = FlaxGrok1RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attn_norm = FlaxGrok1RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_moe_norm = FlaxGrok1RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_moe_norm = FlaxGrok1RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
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
            A tuple of two arrays HiddenState, attentionWeight and and routerLogits.
        """
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states, attention_weights = self.attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )

        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        hidden_states, router_logits = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, attention_weights, router_logits)


class Grok1Model(BaseNNXModule):

    def __init__(
        self,
        config: Grok1Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            Grok1DecoderLayer(
                layer_idx=layer_idx,
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = FlaxGrok1RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self._freqs_cis = None
        self._causal_mask = None

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
        input_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ) -> MoeModelOutput | Tuple:
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

        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :sequence_length, :]
            )

        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        hidden_states = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
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
        hidden_states = (
            input_embeds if extra_embedding is None else input_embeds + extra_embedding
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weights, router_logits = block(
                attention_mask=attention_mask,
                freqs_cis=self.freqs_cis,
                hidden_states=hidden_states,
                past_key_values=(
                    past_key_values[idx] if past_key_values is None else None
                ),
                position_ids=position_ids,
                segment_ids=segment_ids,
            )
            if output_attentions:
                all_self_attns += (attn_weights,)
            if output_router_logits:
                all_router_logits += (router_logits,)

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


class Grok1ForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: Grok1Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.model = Grok1Model(
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

        self.output_multiplier_scale = self.config.output_multiplier_scale

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ) -> MoeCausalLMOutput | Tuple:
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
            extra_embedding=extra_embedding,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
            return_dict=True,
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

    @property
    def can_generate(self):
        return True
