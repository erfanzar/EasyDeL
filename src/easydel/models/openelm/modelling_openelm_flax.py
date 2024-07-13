import functools
import math
from typing import Optional, Tuple, Union, List

import chex
from flax import nnx
import jax
from easydel.models.caching_utils import KVCache
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.common import RMSNorm
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from easydel.models.openelm.openelm_configuration import OpenELMConfig as OpenELMConfig
from easydel.models.openelm.openelm_configuration import make_divisible


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    key_len = key.shape[2]
    query_len = query.shape[2]
    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin[..., :key_len, :], cos[..., :key_len, :])
    query = apply_rotary_pos_emb(
        query,
        sin[..., key_len - query_len : key_len, :],
        cos[..., key_len - query_len : key_len, :],
    )

    return query.astype(dtype), key.astype(dtype)


class OpenELMMultiHeadCausalAttention(BaseAttentionModule):
    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: OpenELMConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nnx.Linear(
            config.model_dim,
            (q_heads + k_heads + v_heads) * head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        if config.normalize_qk_projections:
            self.q_norm = RMSNorm(
                dim=config.head_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                dim=config.head_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nnx.Linear(
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.head_dim = head_dim
        self.rotary = functools.partial(apply_rope, dtype=dtype)
        self.attention_module = FlexibleAttentionModule(
            num_attention_heads=q_heads,
            attention_dropout=0.0,
            head_dims=head_dim,
            precision=precision,
            force_float32_tpu=True,
            attn_mechanism=config.attn_mechanism,
            dtype=config.attn_dtype,
            mesh=config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=config.attention_axis_name,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_q_heads * self.head_dim,)
        )

    def apply_rotary(self, query, key, freqs_cis, position_ids):
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

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(0, 2, 1, 3)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        query_states = qkv[
            :,
            : self.num_q_heads,
            :,
            :,
        ]
        key_states = qkv[
            :,
            self.num_q_heads : self.num_k_heads + self.num_q_heads,
            :,
            :,
        ]
        value_states = qkv[
            :,
            self.num_k_heads + self.num_q_heads :,
            :,
            :,
        ]
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        query_states, key_states, value_states = map(
            lambda x: x.transpose(0, 2, 1, 3),
            [query_states, key_states, value_states],
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
        attn_output = self.out_proj(attn_output)

        return attn_output, attentions.attention_weights


class OpenELMFeedForwardNetwork(nnx.Module):
    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: OpenELMConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,  # type:ignore
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nnx.Linear(
                config.model_dim,
                2 * intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.proj_2 = nnx.Linear(
                2 * intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.ffn_with_glu = True
        else:
            self.proj_1 = nnx.Linear(
                config.model_dim,
                intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.proj_2 = nnx.Linear(
                intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        if self.ffn_with_glu:
            y_12 = self.proj_1(hidden_states)
            y_1, y_2 = jnp.split(y_12, 2, axis=-1)
            return self.proj_2(self.act(y_1) * y_2)
        else:
            return self.proj_2(self.act(self.proj_1(hidden_states)))


class OpenELMDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: OpenELMConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        attn_block = OpenELMMultiHeadCausalAttention
        mlp_block = OpenELMFeedForwardNetwork
        # if self.config.gradient_checkpointing != "":
        #     # hidden_states: chex.Array,
        #     # freqs_cis: Tuple[chex.Array, chex.Array],
        #     # attention_mask: chex.Array,
        #     # position_ids: chex.Array,
        #     # causal_mask: chex.Array,
        #     # segment_ids: Optional[chex.Array] = None,
        #     # deterministic: bool = True,
        #     # init_cache: bool = False,
        #     # output_attentions: bool = False,
        #     # fcm_mask = None,
        #     attn_block = re_mat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(1, 3, 4, 6, 7, 8),
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
        self.ffn = mlp_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn_norm = RMSNorm(
            config.model_dim, dtype=dtype, param_dtype=param_dtype, eps=1e-6, rngs=rngs
        )
        self.attn_norm = RMSNorm(
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
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
    ) -> Tuple[chex.Array, chex.Array]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, self_attn_weights = self.attn(
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
        hidden_states = self.ffn_norm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.ffn,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.ffn(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states, self_attn_weights)

        return outputs


class OpenELMModel(BaseNNXModule):
    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.token_embeddings = nnx.Embed(
            config.vocab_size,
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            OpenELMDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=i,
                rngs=rngs,
            )
            for i in range(config.num_transformer_layers)
        ]
        self.norm = RMSNorm(
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
            rngs=rngs,
        )
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nnx.Dense(
                config.model_dim,
                config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.num_transformer_layers = config.num_transformer_layers

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
        input_embeds: Optional[chex.Array] = None,
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
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
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

        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )
        if input_embeds is None and input_ids is not None:
            input_embeds = self.token_embeddings(input_ids.astype("i4"))
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

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
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

        outputs = (hidden_states, all_hidden_states, all_attentions)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class OpenELMForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.transformer: OpenELMModel = OpenELMModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = nnx.Linear(
            config.model_dim,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        input_embeds: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ):
        """The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Return the attention weights
            output_hidden_states: bool: Determine whether to return the hidden states
            return_dict: bool: Return a dictionary of the outputs or not
            extra_embedding: (Optional(chex.Array)): Pass in the embedding of the word that we want to predict

        Returns:
            The logits and the hidden states
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
        )

        hidden_states = outputs[0]

        if self.config.share_input_output_layers:
            self.lm_head.kernel.value = (
                self.transformer.token_embeddings.embedding.value.T
            )
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits[:, : self.config.vocab_size]
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def can_generate(self):
        return True
