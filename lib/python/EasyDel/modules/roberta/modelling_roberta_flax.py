# Model is modified version from EasyLM
# Supports 8,6,4 BIT and flash attention
import math
from gc import unfreeze
from typing import Optional, Tuple
from flax import linen as nn
from flax.core import FrozenDict, freeze
from flax.linen.attention import make_attention_mask, make_causal_mask, combine_masks, dot_product_attention_weights
from flax.linen.partitioning import remat
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental.shard_map import shard_map
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxMaskedLMOutput, \
    FlaxSequenceClassifierOutput, FlaxMultipleChoiceModelOutput, FlaxTokenClassifierOutput, \
    FlaxQuestionAnsweringModelOutput, FlaxCausalLMOutputWithCrossAttentions, \
    FlaxBaseModelOutputWithPoolingAndCrossAttentions

from .roberta_configuration import RobertaConfig
import jax
from jax.sharding import PartitionSpec
from jax import lax, numpy as jnp
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
from ..easy_attention import EasyAttention
from ..flax_modelling_utils import get_gradient_checkpoint_policy, ACT2FN, get_dot_general_by_bits, \
    BaseJAXAttentionModule


class FlaxRobertaEmbeddings(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxRobertaSelfAttention(BaseJAXAttentionModule):
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )
        self.attention_performer = EasyAttention(
            attn_type="normal",
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
            attention_dropout=0.0,
            head_dims=self.head_dim,
            attention_partition_spec=self.config.attention_partition_spec,
            use_shard_map=self.config.use_shard_map,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim)
        )
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, getattr(self.config, "c_max_position_embeddings", self.config.max_position_embeddings)),
                         dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    def __call__(
            self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            key_value_states: Optional[jnp.array] = None,
            init_cache: bool = False,
            deterministic=True,
            output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        query_states = self.query(hidden_states)
        if is_cross_attention:
            key_states = self.key(key_value_states)
            value_states = self.value(key_value_states)
        else:
            key_states = self.key(hidden_states)
            value_states = self.value(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")
        if layer_head_mask is None:
            out = self.attention_performer.__call__(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                dropout_rng=dropout_rng,
                deterministic=deterministic,
                causal=False,
                bias=attention_bias,
                uses_cache=False,
                query_sequence_length=query_states.shape[1],
                key_value_sequence_length=key_states.shape[1]
            )
            attn_weights = out.attention_weights
            attn_output = out.attention_outputs
        else:

            attn_weights = dot_product_attention_weights(
                query_states,
                key_states,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attention_probs_dropout_prob,
                broadcast_dropout=True,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=None,
            )

            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.self = FlaxRobertaSelfAttention(
            self.config,
            causal=self.causal,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(
            self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            key_value_states=None,
            init_cache=False,
            deterministic=True,
            output_attentions: bool = False,
    ):
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            precision=self.precision,
            param_dtype=self.param_dtype,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.attention = FlaxRobertaAttention(
            self.config,
            causal=self.config.is_decoder,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxRobertaAttention(
                self.config,
                causal=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(
            self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            output_attentions: bool = False,
    ):
        # Self Attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs


class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        block = FlaxRobertaLayer
        if self.config.gradient_checkpointing != "":
            block = remat(
                block,
                static_argnums=(5, 6, 7),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )

        self.layers = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                deterministic,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.layer = FlaxRobertaLayerCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

    def __call__(
            self,
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        self.bias = self.param(
            "bias",
            jax.nn.initializers.zeros,
            (
                self.config.vocab_size,
            )
        )

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class FlaxRobertaPreTrainedModel(EasyDelFlaxPretrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    module_class: nn.Module = None

    def __init__(
            self,
            config: RobertaConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[lax.Precision] = None,
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.ones_like(input_ids)
        mask = (input_ids != self.config.pad_token_id).astype("i4")

        if mask.ndim > 2:
            mask = mask.reshape((-1, mask.shape[-1]))
            incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
            incremental_indices = incremental_indices.reshape(input_ids.shape)
        else:
            incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

        position_ids = incremental_indices.astype("i4") + self.config.pad_token_id

        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        random_params = module_init_outputs["params"]

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

        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            params: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            past_key_values: dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            mask = (input_ids != self.config.pad_token_id).astype("i4")

            if mask.ndim > 2:
                mask = mask.reshape((-1, mask.shape[-1]))
                incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
                incremental_indices = incremental_indices.reshape(input_ids.shape)
            else:
                incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

            position_ids = incremental_indices.astype("i4") + self.config.pad_token_id

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        if self.config.add_cross_attention:
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                token_type_ids=jnp.array(token_type_ids, dtype="i4"),
                position_ids=jnp.array(position_ids, dtype="i4"),
                head_mask=jnp.array(head_mask, dtype="i4"),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=not train,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rngs=rngs,
                mutable=mutable,
            )

            # add updated cache to model output
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        else:
            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                token_type_ids=jnp.array(token_type_ids, dtype="i4"),
                position_ids=jnp.array(position_ids, dtype="i4"),
                head_mask=jnp.array(head_mask, dtype="i4"),
                deterministic=not train,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rngs=rngs,
            )

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertModule with Bert->Roberta
class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxRobertaEmbeddings(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.encoder = FlaxRobertaEncoder(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.pooler = FlaxRobertaPooler(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids: Optional[jnp.ndarray] = None,
            position_ids: Optional[jnp.ndarray] = None,
            head_mask: Optional[jnp.ndarray] = None,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaModule


class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = FlaxRobertaLMHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.classifier = FlaxRobertaClassificationHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule


class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMultipleChoiceModule


class FlaxRobertaForTokenClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)

        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForTokenClassificationModule


class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.qa_outputs = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(bits=self.config.bits, mode=self.config.easy_method)
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForQuestionAnsweringModule


class FlaxRobertaForCausalLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[lax.Precision] = None

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = FlaxRobertaLMHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids: Optional[jnp.ndarray] = None,
            head_mask: Optional[jnp.ndarray] = None,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxRobertaForCausalLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
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
