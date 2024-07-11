import math
from typing import List, Optional, Union

import chex
import jax
from einops import rearrange
from flax import nnx
from jax import lax
from jax import numpy as jnp
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.flax_modelling_utils import (
    BaseAttentionModule,
    control_mlp_sharding,
)
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.mosaic_mpt.mosaic_configuration import (
    MptConfig as MptConfig,
)


class MptMLP(nnx.Module):
    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.expansion_ratio * config.hidden_size,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.expansion_ratio * config.hidden_size,
            config.hidden_size,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.hidden_dropout = nnx.Dropout(
            config.attn_config.attn_pdrop,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        residual: chex.Array,
    ):
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return (
            self.hidden_dropout(
                self.down_proj(
                    jax.nn.gelu(self.up_proj(hidden_states), approximate=False)
                ),
            )
            + residual
        )


class MptAttention(BaseAttentionModule):
    def __init__(
        self,
        config: MptConfig,
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
        self.Wqkv = nnx.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(
            rate=config.attn_config.attn_pdrop,
            rngs=rngs,
        )

        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attention_module = FlexibleAttentionModule(
            attention_dropout=config.attn_config.attn_pdrop,
            num_attention_heads=config.num_attention_heads,
            head_dims=self.head_dim,
            precision=precision,
            attn_mechanism=config.attn_mechanism,
            dtype=config.attn_dtype,
            mesh=config.mesh,
            sm_scale=self.softmax_scale,
            axis_name=config.attention_axis_name,
            base_config=config,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_bias: chex.Array,
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
            attention_mask: (chex.Array): Mask out certain tokens in the input sequence
            past_key_values: (Optional(KVCache)): Past key and values used for generation
            position_bias: (Optional(chex.Array)):  Add a bias to the attention scores
            segment_ids: (Optional(chex.Array)): Determine the Segment.

        Returns:
            A tuple of two arrays
        """

        inp_shape = hidden_states.shape
        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

        query_states = rearrange(
            query_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )
        key_states = rearrange(
            key_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )
        value_states = rearrange(
            value_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )

        if past_key_values is not None:
            past_key_values.update(key_states=key_states, value_states=value_states)
            key_states, value_states, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if position_bias is not None:
            key_length = key_states.shape[1]

            position_bias_query_index = max(0, position_bias.shape[2] - query_length)
            position_bias_key_index = max(0, position_bias.shape[3] - key_length)

            position_bias = position_bias[
                :,
                :,
                position_bias_query_index:,
                position_bias_key_index:,
            ]
        attention_mask = attention_mask.repeat(position_bias.shape[1], 1)
        attention_bias = lax.select(
            attention_mask.astype("bool"),
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype)
            + position_bias.astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        attention = self.attention_module(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            bias=attention_bias,
            causal=False,
        )

        attn_output = self.out_proj(attention.attention_outputs.reshape(inp_shape))

        return attn_output, attention.attention_weights


class MptBlock(nnx.Module):
    def __init__(
        self,
        config: MptConfig,
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
        attn_block = MptAttention
        mlp_block = MptMLP
        # if self.config.gradient_checkpointing != "":
        #     mlp_block = remat(
        #         mlp_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(2,),
        #     )
        #     attn_block = remat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(3, 4, 5),
        #     )

        self.norm_1 = nnx.LayerNorm(
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.norm_2 = nnx.LayerNorm(
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.ffn = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout_rate = self.config.attn_config.attn_pdrop
        self.resid_attn_dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_bias: chex.Array,
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
            attention_mask: (chex.Array): Mask out certain tokens in the input sequence
            past_key_values: (Optional(KVCache)): Past key and values used for generation
            position_bias: (Optional(chex.Array)):  Add a bias to the attention scores
            segment_ids: (Optional(chex.Array)): Determine the Segment.

        Returns:
            A tuple of two arrays
        """
        attn_outputs, attn_weights = self.attn(
            hidden_states=self.norm_1(hidden_states),
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
        )
        hidden_states = self.resid_attn_dropout(attn_outputs) + hidden_states
        output = self.ffn(self.norm_2(hidden_states), hidden_states)

        return output, attn_weights


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    alibi = jnp.arange(1 - sequence_length, 1, dtype="i4").reshape(
        1, 1, 1, sequence_length
    )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / jnp.pow(2, base)
    slopes = slopes.reshape(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = jnp.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], axis=1)[
            :, :num_heads, ...
        ]

    alibi = alibi * slopes
    return alibi


class MptModel(BaseNNXModule):
    def __init__(
        self,
        config: MptConfig,
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
        self.wte = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.blocks = [
            MptBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ]
        self.norm_f = nnx.LayerNorm(
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=config.layer_norm_epsilon,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self._alibi = None
        self._causal_mask = None

    def alibi(self):
        if self._alibi is None:
            self._alibi = build_mpt_alibi_tensor(
                sequence_length=self.config.max_seq_len,
                num_heads=self.config.n_heads,
            )
        return self._alibi

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
        input_embeds: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ):
        """The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            input_embeds: (Optional(chex.Array)): input_embeds to be used instead of input_ids if passed.
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: (Optional(bool)): Return the attention weights.
            output_hidden_states: (Optional(bool)): Determine whether to return the hidden states.
            output_router_logits: (Optional(bool)): Determine whether to return the router logits.
            return_dict: bool: Return a dictionary of the outputs or not
            extra_embedding: (Optional(chex.Array)): Pass in the embedding of the word that we want to predict

        Returns:
            The logits and the hidden states
        """

        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )
        if input_embeds is None:
            input_embeds = self.wte(input_ids)
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
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attn_weight = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=(
                    past_key_values[idx] if past_key_values is not None else None
                ),
                position_bias=self.alibi,
                segment_ids=None,
            )
            if output_attentions:
                all_attentions += (attn_weight,)

        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )

        return tuple(
            [
                v
                for v in (hidden_states, all_hidden_states, all_attentions)
                if v is not None
            ]
        )

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class MptForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: MptConfig,
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
        self.transformer = MptModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ):
        """The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            input_embeds: (Optional(chex.Array)): input_embeds to be used instead of input_ids if passed.
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: (Optional(bool)): Return the attention weights.
            output_hidden_states: (Optional(bool)): Determine whether to return the hidden states.
            output_router_logits: (Optional(bool)): Determine whether to return the router logits.
            return_dict: bool: Return a dictionary of the outputs or not
            extra_embedding: (Optional(chex.Array)): Pass in the embedding of the word that we want to predict

        Returns:
            The logits and the hidden states
        """
        predict: FlaxBaseModelOutput = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            extra_embedding=extra_embedding,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        last_hidden_state = predict.last_hidden_state

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
            lm_logits = self.lm_head(last_hidden_state)
        else:
            lm_logits = self.lm_head(last_hidden_state)

        if return_dict:
            return FlaxCausalLMOutput(
                logits=lm_logits,
                hidden_states=predict.hidden_states,
                attentions=predict.attentions,
            )
        return tuple(
            v
            for v in [lm_logits, predict.hidden_states, predict.attentions]
            if v is not None
        )

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_decoder(self):
        return self.transformer

    def set_input_embeddings(self, value):
        self.module.transformer.wte = value

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        attention_mask: Optional[chex.Array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        return model_kwargs
