import warnings
from typing import Optional, Tuple, Union

import chex
import fjformer
from fjformer import linen as nn
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from fjformer import with_sharding_constraint
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from fjformer.linen import Linear
from flax.linen import combine_masks, make_causal_mask

from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...etils.etils import get_logger
from ..attention_module import AttentionModule
from ..flax_modelling_utils import (
    ACT2FN,
    BaseJAXAttentionModule,
    get_gradient_checkpoint_policy,
    get_dot_general_by_bits,
    block_wise_ffn,
    precompute_freq_cis,
    apply_rotary_pos_emb
)
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from .gemma_configuration import GemmaConfig

logger = get_logger(__name__)


def add_positional_embedding(
        input_embedding: jax.Array,
        position: int,
        theta: int = 10_000,
) -> jax.Array:
    """Adds positional embeddings to input embeddings. From DeepMind Gemma"""
    embed_dim = input_embedding.shape[-1]
    num_timescales = embed_dim // 2
    log_timescale_increment = jnp.log(float(theta)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
    signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
    position_embedding = signal.astype(jnp.float32)

    return input_embedding + position_embedding


def apply_rope(
        inputs: jax.Array,  # [B, L]
        positions: jax.Array,  # [B, L]
        head_dim: int,
        theta: int = 10_000,
) -> jax.Array:
    """Applies RoPE. From DeepMind Gemma"""
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = theta ** fraction

    sinusoid_inp = (
            positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class FlaxGemmaRMSNorm(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight_kernel = self.param("kernel", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + nn.linen.control_quantization(self.weight_kernel, self.dtype)) * jnp.asarray(
            hidden_states, dtype=self.dtype
        )


class FlaxGemmaRotaryEmbedding(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def __call__(self, freq_cis, key_states, query_states, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key_states, sin, cos)
        query = apply_rotary_pos_emb(query_states, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxGemmaAttention(BaseJAXAttentionModule):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        kernel = jax.nn.initializers.normal(self.config.initializer_range)
        self.q_proj = Linear(
            self.num_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.k_proj = Linear(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.v_proj = Linear(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.o_proj = Linear(
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.attention_performer = AttentionModule(
            use_sharding_constraint=self.config.use_sharding_constraint,
            block_k_major=self.config.block_k_major,
            block_b=self.config.block_b,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            block_q_major_dkv=self.config.block_q_major_dkv,
            block_k_major_dkv=self.config.block_k_major_dkv,
            block_k_major_dq=self.config.block_k_major_dq,
            block_k_dkv=self.config.block_k_dkv,
            block_q_dkv=self.config.block_q_dkv,
            block_q_dq=self.config.block_q_dq,
            block_k_dq=self.config.block_k_dq,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.head_dim,
            attention_partition_spec=self.config.attention_partition_spec,
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            generation_query_partition_spec=self.config.generation_query_partition_spec,
            generation_bias_partition_spec=self.config.generation_bias_partition_spec,
            generation_attention_partition_spec=self.config.generation_attention_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=self.head_dim ** -0.5,
            axis_name=self.config.attention_axis_name
        )

        self.rotary_emb = FlaxGemmaRotaryEmbedding(config, dtype=self.dtype)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads * self.head_dim,))

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    @staticmethod
    def _transpose_sequence_head(query, key, value):
        """
        The _transpose_sequence_head function transposes the query, key and value matrices.

        :param query: Get the attention weights for each of the heads
        :param key: Determine the number of heads
        :param value: Store the values of the input
        :return: The transpose of the query, key and value matrices

        """
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def apply_rotary(self, batch_size, sequence_length, query, key, value, freq_cis, position_ids):
        """
        The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freq_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        :param self: Access variables that belong to the class
        :param batch_size: Reshape the query, key and value tensors
        :param sequence_length: Reshape the query, key and value tensors
        :param query: Calculate the attention weights
        :param key: Calculate the attention
        :param value: Compute the attention weights
        :param freq_cis: Calculate the frequency of each word in the vocabulary
        :param position_ids: Identify the position of each token in the sequence
        :return: A tuple of 3 tensors: query, key and value

        """
        query = query.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )

        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary_emb(
            position_ids=position_ids, query_states=query, key_states=key, freq_cis=freq_cis
        )
        return self._transpose_sequence_head(query, key, value)

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        (
            query_states,
            key_states,
            value_states
        ) = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        query_states, key_states, value_states = self.apply_rotary(
            query_states.shape[0],
            query_states.shape[1],
            query_states,
            key_states,
            value_states,
            freq_cis,
            position_ids
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states,
                value_states,
                query_states,
                attention_mask
            )

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        key_states = jnp.repeat(key_states, repeats=self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, repeats=self.num_key_value_groups, axis=2)

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_performer.__call__(
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
            causal_mask=causal_mask
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output, PartitionSpec(
                    ("dp", "fsdp"),
                    "sp" if attn_output.shape[1] != 1 else None,
                    "tp"
                )
            )
        attn_output = self.o_proj(attn_output)

        return (
            attn_output,
            attentions.attention_weights
        ) if output_attentions else (attn_output,)


class FlaxGemmaMLP(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        if self.config.hidden_activation is None:
            warnings.warn(
                "Gemma's activation function should be approximate GeLU and not exact GeLU. "
                "Changing the activation function to `gelu_pytorch_tanh`."
                f"if you want to use the legacy `{self.config.hidden_act}`, "
                f"edit the `model.config` to set `hidden_activation={self.config.hidden_act}` "
            )
            hidden_activation = "gelu_pytorch_tanh"
        else:
            hidden_activation = self.config.hidden_activation
        self.act = ACT2FN[hidden_activation]

        self.gate_proj = Linear(
            inner_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.down_proj = Linear(
            embed_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.up_proj = Linear(
            inner_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(self, hidden_states, deterministic=False):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class FlaxGemmaDecoderLayer(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        mlp_block = FlaxGemmaMLP
        attn_block = FlaxGemmaAttention

        if self.config.gradient_checkpointing != "":
            mlp_block = flax.linen.partitioning.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )
            attn_block = flax.linen.partitioning.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(3, 4, 6, 7, 8)
            )
        self.input_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = attn_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.mlp = mlp_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            freq_cis,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
        )

        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.mlp(
                hidden_states,
                deterministic,
            )
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return (hidden_states,) + outputs[1:]


class FlaxGemmaPreTrainedModel(EasyDeLFlaxPretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GemmaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
            self,
            config: GemmaConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest"),
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            **kwargs
        )
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init
        )

    def init_cache(self, batch_size, max_length):
        """
        The init_cache function is used to initialize the cache for a given batch size and sequence length.
        The cache is a dictionary that contains all the intermediate states from each layer in the model.
        This allows us to run inference on multiple batches without having to re-run forward passes through every layer in
        the model, which would be very slow.

        :param self: Access the module
        :param batch_size: Define the batch size of the input tensors
        :param max_length: Set the length of the input sequence
        :return: A dictionary with the following keys:

        """
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params} if add_params_field else params or self.params

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
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


class FlaxGemmaLayerCollection(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.blocks = [
            FlaxGemmaDecoderLayer(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            causal_mask: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                freq_cis=freq_cis,
                causal_mask=causal_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGemmaModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.layers = FlaxGemmaLayerCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm = FlaxGemmaRMSNorm(
            self.config,
            dtype=self.dtype,
        )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=self.config.max_position_embeddings,
            dim=self.config.head_dim,
            base=self.config.rope_theta
        )
        self.causal_mask = make_causal_mask(
            jnp.ones((1, self.config.max_position_embeddings), dtype="bool"),
            dtype="bool"
        )

    # Ignore copy
    def __call__(
            self,
            input_ids,
            attention_mask: Optional[chex.Array] = None,
            position_ids: chex.Array = None,
            deterministic=True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        input_embeds = input_embeds * (self.config.hidden_size ** 0.5)

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            freq_cis=self.freq_cis,
            causal_mask=self.causal_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxGemmaModel(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaModule


class FlaxGemmaForCausalLMModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.model = FlaxGemmaModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = Linear(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),

            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids,
            attention_mask: Optional[chex.Array] = None,
            position_ids: chex.Array = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"]
            shared_kernel = fjformer.linen.linen.control_quantization(shared_kernel, self.param_dtype).T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxGemmaForCausalLM(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):

        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:

        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
