import functools
import math
from typing import Optional, Tuple, Union, List

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec

from easydel.models.attention_module import FlexibleAttentionModule
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
from easydel.models.llama.llama_configuration import LlamaConfig as LlamaConfig
from easydel.models.llama.vision_llama_configuration import (
    VisionLlamaConfig as VisionLlamaConfig,
)
from easydel.models.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxSequenceClassifierOutput,
)
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.caching_utils import KVCache


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class LlamaAttention(BaseAttentionModule):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: LlamaConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        if self.num_key_value_groups == 1:
            assert config.num_attention_heads == config.num_key_value_heads
        self.q_proj = nnx.Linear(
            self.hidden_size,
            config.num_attention_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.hidden_size,
            config.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
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
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            batch_size: Reshape the query, key and value tensors
            sequence_length: Reshape the query, key and value tensors
            query: Calculate the attention weights
            key: Calculate the attention
            value: Compute the attention weights
            freqs_cis: Calculate the frequency of each word in the
                vocabulary
            position_ids: Identify the position of each token in the
                sequence

        Returns:
            A tuple of 3 tensors: query, key and value
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

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if past_key_values is not None:
            past_key_values.update(key_states=key_length, value_states=value_states)
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
            causal_mask = attention_mask[:, :, :, :key_length]
            attention_bias = lax.select(
                causal_mask > 0,
                jnp.full(causal_mask.shape, 0.0).astype(self.dtype),
                jnp.full(
                    causal_mask.shape,
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


class LlamaMLP(nnx.Module):
    def __init__(
        self,
        config: LlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: LlamaConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
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
        return self.dropout(
            self.down_proj(
                ACT2FN[self.config.hidden_act](self.gate_proj(hidden_states))
                * self.up_proj(hidden_states),
            )
        )


class LlamaBlock(nnx.Module):
    def __init__(
        self,
        config: LlamaConfig,
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
        attn_block = LlamaAttention
        mlp_block = LlamaMLP
        if config.gradient_checkpointing != "":
            attn_block = nnx.remat(
                attn_block,
                static_argnums=(1, 4, 5),
                policy=get_gradient_checkpoint_policy(config.gradient_checkpointing),
            )
            mlp_block = nnx.remat(
                mlp_block,
                static_argnums=(),
                policy=get_gradient_checkpoint_policy(config.gradient_checkpointing),
            )

        self.self_attn = attn_block(
            config,
            layer_idx,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
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

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        """The __call__ function is the main function of a TransformerEncoderLayer.
        It takes in hidden states, frequency-domain inputs, and masks as input. It then
        applies self-attention to the hidden states using those inputs and returns an
        output tensor with shape (batch_size, sequence_length, model_dim).

        Args:
            self: Access variables that belong to the class
            hidden_states: (chex.Array): Pass the hidden states of the previous layer
            freqs_cis: (Tuple[chex.Array, chex.Array]),: Pass in the frequency coefficients for each position
            attention_mask: (chex.Array): Mask out certain tokens in the input sequence
            past_key_values: (Optional(KVCache)): Past key and values used for generation
            position_ids: (Optional(chex.Array)): Determine the position of each token in a sequence

        Returns:
            A tuple of two items
        """
        attn_output, attn_weight = self.self_attn(
            self.input_layernorm(hidden_states),
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(
                feed_forward_input,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, attn_weight


class LlamaModel(BaseNNXModule):
    def __init__(
        self,
        config: LlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(
            rate=config.embd_pdrop,
            rngs=rngs,
        )
        self.layers = [
            LlamaBlock(
                config=config,
                layer_idx=layer_idx,
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
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of:
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
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :seq_length, :]
            )

        batch_size, sequence_length, _ = inputs_embeds.shape

        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        inputs_embeds = (
            inputs_embeds + extra_embedding
            if extra_embedding is not None
            else inputs_embeds
        )
        hidden_states = self.dropout(inputs_embeds)
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


class LlamaForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: LlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.model = LlamaModel(
            config,
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
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"]
            self.lm_head.kernel.value = shared_kernel
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


class LlamaForSequenceClassification(BaseNNXModule):
    def __init__(
        self,
        num_classes: int,
        config: LlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.model = LlamaModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.classifier = nnx.Linear(
            config.hidden_size,
            num_classes,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array = None,
        position_ids: chex.Array = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        batch_size, seq_length = input_ids.shape

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            past_key_values=None,
        )

        hidden_states = outputs[0]
        prediction = self.classifier(hidden_states)
        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=prediction, hidden_states=hidden_states
            )
        else:
            return (prediction,)

    @property
    def can_generate(self):
        return True


class VisionLlamaModel(BaseNNXModule):
    def __init__(
        self,
        config: VisionLlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.embed_dim = config.hidden_size

        self.embed_vision = nnx.Embed(
            config.vision_vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.embd_pdrop)
        self.layers = [
            LlamaBlock(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
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
        vision_mask: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            vision_mask: (chex.Array): Mask out the vision padding tokens
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Return the attention weights
            output_hidden_states: bool: Determine whether to return the hidden states
            return_dict: bool: Return a dictionary of the outputs or not

        Returns:
            The logits and the hidden states
        """
        input_ids = input_ids.astype("i4")
        batch_size, seq_length = input_ids.shape
        if input_ids.shape[1] == 1:
            if self.config.sample_mode == "text":
                input_embeds = self.embed_tokens(input_ids)
            elif self.config.sample_mode == "vision":
                input_embeds = self.embed_vision(input_ids)
            elif self.config.sample_mode == "all":
                raise NotImplementedError
            else:
                raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
        else:
            input_text_embeds = self.embed_tokens(jnp.where(vision_mask, 0, input_ids))
            input_vision_embeds = self.embed_vision(
                jnp.where(vision_mask, input_ids, 0)
            )
            vision_mask = vision_mask[..., None].astype("f4")
            input_embeds = (
                input_text_embeds * (1 - vision_mask)
                + input_vision_embeds * vision_mask
            )

        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :seq_length, :]
            )

        hidden_states = self.dropout(input_embeds)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
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


class VisionLlamaForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: VisionLlamaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.model = VisionLlamaModel(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.vision_head = nnx.Linear(
            config.hidden_size,
            self.config.vision_vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        vision_mask: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            vision_mask: (chex.Array): Mask out the vision padding tokens
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Return the attention weights
            output_hidden_states: bool: Determine whether to return the hidden states
            return_dict: bool: Return a dictionary of the outputs or not

        Returns:
            The logits and the hidden states
        """
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            )

        outputs = self.model(
            input_ids=input_ids,
            vision_mask=vision_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        hidden_states = outputs[0]

        if self.config.tie_vision_embeddings:
            shared_kernel = self.transformer.variables["params"]["embed_vision"][
                "embedding"
            ].T
            vision_logits = self.vision_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states
            )
        else:
            vision_logits = self.vision_head(hidden_states)

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"]
            self.lm_head.kernel.value = shared_kernel
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.sample_mode == "all":
            if not return_dict:
                return (
                    vision_logits,
                    lm_logits,
                ) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=(vision_logits, lm_logits),
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif self.config.sample_mode == "vision":
            if not return_dict:
                return (vision_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=vision_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif self.config.sample_mode == "text":
            if not return_dict:
                return (lm_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=lm_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")

    @property
    def can_generate(self):
        return True
