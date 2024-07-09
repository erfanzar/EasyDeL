import math
from typing import List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx
from jax import Array, lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
)

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.chatglm.chatglm_configuration import ChatGLMConfig as ChatGLMConfig
from easydel.models.common import RMSNorm
from easydel.models.flax_modelling_utils import (
    BaseAttentionModule,
    with_sharding_constraint,
)


def flatten_axes(a: Array, start: int = 0, end: int = -1) -> Array:
    return a.reshape(a.shape[:start] + (-1,) + a.shape[end:][1:])


def split_tensor_along_last_dim(
    tensor: jax.Array,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> tuple[Array, ...] | list[Array]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = jnp.split(tensor, last_dim_size, axis=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(jax.lax.stop_gradient(chunk) for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        rope_ratio: float,
        dim: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype
        self.dim = dim
        self.rope_ratio = rope_ratio
        self._inv_freq = None

    @property
    def inv_freq(self):
        if self._inv_freq is None:
            self._inv_freq = 1.0 / (
                10000 ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim)
            )
        return self._inv_freq

    def __call__(self, seq_len: int, n_elem: int, base: int = 10000):
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (jnp.arange(0, n_elem, 2, dtype=jnp.float32) / n_elem))
        seq_idx = jnp.arange(seq_len, dtype=jnp.float32)
        idx_theta = jnp.outer(seq_idx, theta).astype(jnp.float32)

        cache = jnp.stack([jnp.cos(idx_theta), jnp.sin(idx_theta)], axis=-1)

        if self.dtype in (jnp.float16, jnp.bfloat16, jnp.int8):
            cache = (
                cache.astype(jnp.bfloat16)
                if self.dtype == jnp.bfloat16
                else cache.astype(jnp.float16)
            )
        return cache


def apply_rotary_pos_emb(x: jax.Array, rope_cache: jax.Array) -> jax.Array:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.reshape(-1, 1, sq, xshaped.shape[3], 2)
    x_out2 = jnp.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = flatten_axes(x_out2, 3)
    return jnp.concatenate((x_out2, x_pass), axis=-1)


class CoreAttention(nnx.Module):
    def __init__(
        self,
        config: ChatGLMConfig,
        layer_number: int,
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
        self.layer_number = layer_number
        config = self.config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = (
            projection_size // config.num_attention_heads
        )
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nnx.Dropout(config.attention_dropout, rngs=rngs)
        self.attention_module = FlexibleAttentionModule(
            attention_dropout=self.config.attention_dropout,
            num_attention_heads=self.config.num_attention_heads,
            head_dims=self.head_dim,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.config.attn_dtype,
            mesh=self.config.get_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            base_config=self.config,
        )

    def __call__(
        self,
        query_layer: jax.Array,
        key_layer: jax.Array,
        value_layer: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ):
        bias = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_layer.shape[1]]
            bias = lax.select(
                attention_mask,
                jnp.full(attention_mask.shape, 0, dtype=query_layer.dtype),
                jnp.full(
                    attention_mask.shape,
                    jnp.finfo(query_layer.dtype).min,
                    dtype=query_layer.dtype,
                ),
            )
        context_layer = self.attention_module(
            query_layer,
            key_layer,
            value_layer,
            bias=bias,
            attention_mask=attention_mask,
        ).attention_outputs
        new_context_layer_shape = context_layer.reshape[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


class ChatGLMAttention(BaseAttentionModule):
    def __init__(
        self,
        config: ChatGLMConfig,
        layer_number: int,
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
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size
                + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nnx.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            use_bias=config.add_bias_linear or config.add_qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.core_attention = CoreAttention(
            config,
            self.layer_number,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        # Output.
        self.dense = nnx.Linear(
            config.hidden_size,
            use_bias=config.add_bias_linear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
        )
        self.num_num_key_value_groupsreps = (
            self.num_attention_heads_per_partition
            // self.num_multi_query_groups_per_partition
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
            A tuple of 2 tensors: query, key
        """

        query, key = self._transpose_sequence_head(query, key)
        query, key = self.rotary(
            position_ids=position_ids, query=query, key=key, freqs_cis=freqs_cis
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
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        else:
            new_tensor_shape = mixed_x_layer.shape[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        query_layer, key_layer, value_layer = self.apply_rotary(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        if past_key_values is not None:
            past_key_values.update(key_layer, value_layer)
            key_layer, value_layer, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        key_layer, value_layer = self.repeat_key_value(
            key_layer,
            value_layer,
            self.num_key_value_groups,
        )

        attn_output = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.dense(attn_output)

        return attn_output, None


class MLP(nnx.Module):
    """
    MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        layer_number: int,
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
        self.layer_number = max(1, layer_number)
        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nnx.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            use_bias=self.add_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )

        def swiglu(x):
            x = jnp.split(x, 2, axis=-1)
            return jax.nn.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nnx.Linear(
            config.ffn_hidden_size * 2,
            config.hidden_size,
            bias=self.add_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
    ):
        """
        This function takes hidden states as input, applies some transformations, and
        returns the final output.

        Args:
          hidden_states: The `hidden_states` parameter in the code snippet represents the
        input hidden states that are passed to the neural network layer for processing.
        These hidden states are typically the output of the previous layer in a neural
        network model. The function processes these hidden states through a series of dense
        layers

        Returns:
          the result of passing the `hidden_states` through a series of dense layers and
        activation functions. The final output is the result of passing the output of the
        `dense_h_to_4h` layer through an activation function and then through the
        `dense_4h_to_h` layer.
        """
        return self.dense_4h_to_h(
            self.activation_func(self.dense_h_to_4h(hidden_states))
        )


class ChatGLMBlock(nnx.Module):
    def __init__(
        self,
        config: ChatGLMConfig,
        layer_number: int,
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
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else nnx.LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        attn_block = ChatGLMAttention
        mlp_block = MLP

        self.self_attention = attn_block(
            config=config,
            layer_number=layer_number,
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

        self.hidden_dropout = nnx.Dropout(config.hidden_dropout, rngs=rngs)

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
            A tuple of two items HiddenState and attentionWeight(if any)
        """
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attention(
            layernorm_output,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.hidden_dropout(attention_output)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.hidden_dropout(mlp_output)
        output = residual + output

        return output, None


class ChatGLMTransformer(nnx.Module):
    def __init__(
        self,
        config: ChatGLMConfig,
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
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        self.layers = [
            ChatGLMBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_number=i + 1,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else nnx.LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[List[KVCache]] = None,
        segment_ids: Optional[chex.Array] = None,
        output_hidden_states: Optional[bool] = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, _ = block(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=(
                    past_key_values[idx] if past_key_values is not None else None
                ),
                segment_ids=segment_ids,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states, None


class ChatGLMModel(nnx.Module):
    def __init__(
        self,
        config: ChatGLMConfig,
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

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )

        self.embedding = nnx.Embed(
            num_embeddings=config.padded_vocab_size,
            features=config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.rotary_pos_emb = RotaryEmbedding(
            dim=rotary_dim // 2,
            rope_ratio=config.rope_ratio,
            dtype=self.dtype,
        )
        self.encoder = ChatGLMTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.output_layer = nnx.Linear(
            config.hidden_size,
            config.padded_vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self._causal_mask = None

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
        """The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
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
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        input_embeds = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :sequence_length]

        hidden_states, all_hidden_states, all_self_attentions = self.encoder(
            attention_mask=attention_mask,
            causal_mask=self.causal_mask,
            hidden_states=input_embeds,
            freqs_cis=rotary_pos_emb,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            segment_ids=None,
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
