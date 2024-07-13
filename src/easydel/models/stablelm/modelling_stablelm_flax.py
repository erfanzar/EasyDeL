import math
from typing import Any, Optional, Tuple, Union, List

import chex
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec
from easydel.models.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxMaskedLMOutput,
)

from easydel.models.attention_module import FlexibleAttentionModule

# easydel.modules
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.caching_utils import KVCache
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.stablelm.stablelm_configuration import (
    StableLmConfig as StableLmConfig,
)
from flax import nnx


def repeat_kv(x: chex.Array, n_rep: int) -> chex.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jnp.newaxis, :, :]
    x = jnp.repeat(x, n_rep, axis=2)

    return x.reshape(bs, s, n_kv_heads * n_rep, head_dim)


class StableLmMLP(nnx.Module):

    def __init__(
        self,
        config: StableLmConfig,
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
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        Args:
            self: Represent the instance of the class
            x: jnp.ndarray: Pass in the input to the layer

        Returns:
            A tensor that is the result of function to x
        """

        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class StableLmLayerNormPerHead(nnx.Module):

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        eps: float = 1e-5,
        bias: bool = False,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.norms = [
            nnx.LayerNorm(
                head_dim,
                epsilon=eps,
                use_bias=bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for idx in range(num_heads)
        ]

    def __call__(self, hidden_states):
        states_per_heads = jnp.split(hidden_states, 1, axis=1)

        return jnp.concatenate(
            [
                norm(hidden_states)
                for norm, hidden_states in zip(self.norms, states_per_heads)
            ],
            axis=1,
        )


class StableLmAttention(BaseAttentionModule):
    def __init__(
        self,
        config: StableLmConfig,
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
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = nnx.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_qkv_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_qkv_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_qkv_bias,
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

        self.rotary_emb_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.attention_module: FlexibleAttentionModule = FlexibleAttentionModule(
            mesh=config.mesh,
            attn_mechanism=config.attn_mechanism,
            sm_scale=1 / math.sqrt(self.head_dim),
            num_attention_heads=config.num_attention_heads,
            head_dims=self.head_dim,
            precision=precision,
            base_config=config,
        )
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = StableLmLayerNormPerHead(
                self.head_dim,
                self.num_heads,
                eps=config.layer_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.k_layernorm = StableLmLayerNormPerHead(
                self.head_dim,
                self.num_key_value_heads,
                eps=config.layer_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )

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
            A tuple of 2 tensors: query_states, key
        """
        query, key = self._transpose_sequence_head(query, key)

        sin, cos = freqs_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        query_rot, query_pass = (
            query[..., : self.rotary_emb_dim],
            query[..., self.rotary_emb_dim :],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_emb_dim],
            key[..., self.rotary_emb_dim :],
        )

        key_rot = apply_rotary_pos_emb(key_rot, sin, cos)
        query_rot = apply_rotary_pos_emb(query_rot, sin, cos)

        query = jnp.concatenate((query_rot, query_pass), axis=-1)
        key = jnp.concatenate((key_rot, key_pass), axis=-1)

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
            segment_ids: (Optional(chex.Array)): Determine the Segment.

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

        if self.qk_layernorm:
            query_states = self.q_layernorm(
                query_states.transpose(0, 2, 1, 3)
            ).transpose(0, 2, 1, 3)
            key_states = self.k_layernorm(key_states.transpose(0, 2, 1, 3)).transpose(
                0, 2, 1, 3
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
        attn_output = self.o_proj(attn_output)

        return attn_output, attentions.attention_weights


class StableLmDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: StableLmConfig,
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
        attn_block = StableLmAttention
        mlp_block = StableLmMLP
        self.use_parallel_residual = self.config.use_parallel_residual
        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nnx.LayerNorm(
                config.hidden_size,
                epsilon=self.config.layer_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        self.dropout = nnx.Dropout(self.config.hidden_dropout, rngs=rngs)

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
            segment_ids: (Optional(chex.Array)): Determine the Segment.

        Returns:
            A tuple of two arrays HiddenState and attentionWeight
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )
        attn_out, self_attn_weights = (attn_out[0], attn_out[1])

        if self.use_parallel_residual:

            if self.config.use_scan_mlp:
                hidden_states = block_wise_ffn(
                    self.mlp,
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
            else:
                hidden_states = self.mlp(
                    hidden_states,
                )

            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + residual + attn_out
        else:

            residual = residual + attn_out
            if self.config.use_scan_mlp:
                hidden_states = block_wise_ffn(
                    self.mlp,
                    self.post_attention_layernorm(residual),
                    self.config.scan_mlp_chunk_size,
                )
            else:
                hidden_states = self.mlp(
                    self.post_attention_layernorm(residual),
                )
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + residual
        outputs = (hidden_states, self_attn_weights)
        return outputs


class StableLmModel(BaseNNXModule):
    def __init__(
        self,
        config: StableLmConfig,
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            StableLmDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
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
            if hasattr(config, "rope_scaling"):
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
                dim=int(
                    config.partial_rotary_factor
                    * (config.hidden_size // config.num_attention_heads)
                ),
                # dim=config.hidden_size // config.num_attention_heads,
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
        segment_ids: Optional[chex.Array] = None,
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
            segment_ids: (Optional(chex.Array)): Determine the Segment.
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            The logits and the hidden states
        """

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

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
                segment_ids=segment_ids
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


class StableLmForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: StableLmConfig,
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
        self.model = StableLmModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = self.config.vocab_size       
        self.lm_head = nnx.Linear(
            config.hidden_size,
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
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
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
            segment_ids: (Optional(chex.Array)): Determine the Segment.
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            The logits and the hidden states
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            input_embeds=input_embeds,
            segment_ids=segment_ids,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits.astype(jnp.float32)

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

