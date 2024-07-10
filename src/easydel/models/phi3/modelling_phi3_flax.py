import functools
import math
from typing import Any, Optional, Tuple, Union

import chex
import fjformer
import flax.linen.partitioning
import jax.lax
from chex import Array
from fjformer import linen as nn
from fjformer.linen import Dense
from flax.core import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxMaskedLMOutput,
)

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.common import RMSNorm as RMSNorm
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    get_dot_general_by_bits,
    get_gradient_checkpoint_policy,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.phi3.phi3_configuration import Phi3Config as Phi3Config


class FlaxPhi3Embedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, freqs_cis, position_ids):
        sin, cos = freqs_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query_states = apply_rotary_pos_emb(query, sin, cos)

        return query_states.astype(self.dtype), key.astype(self.dtype)


class FlaxPhi3MLP(nn.Module):
    config: Phi3Config
    layer_idx: Optional[int] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    """Multi-Layer Perceptron.
    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.
    """

    def setup(self) -> None:
        self.gate_up_proj = Dense(
            2 * self.config.intermediate_size,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
        )
        self.down_proj = Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
        )
        self.activation_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Array, e: bool = False) -> Array:  # Ignored

        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = jnp.split(up_states, 2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class FlaxPhi3Attention(BaseAttentionModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    config: Phi3Config
    layer_idx: Optional[int] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        config = self.config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense_class = functools.partial(
            Dense,
            use_bias=False,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits),
        )

        op_size = self.num_heads * self.head_dim + 2 * (
            self.num_key_value_heads * self.head_dim
        )
        self.o_proj = dense_class(self.hidden_size)
        self.qkv_proj = dense_class(op_size)
        self.rotary = FlaxPhi3Embedding(self.dtype)
        self.attention_module = FlexibleAttentionModule(
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
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.config.attn_dtype,
            partition_axis=self.config.partition_axis,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            backward_pass_impl=self.config.flash_attention_backward_pass_impl,
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(
        self, batch_size, sequence_length, query, key, value, freqs_cis, position_ids
    ):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            batch_size: Reshape the query_states, key and value tensors
            sequence_length: Reshape the query_states, key and value
                tensors
            query: Calculate the attention weights
            key: Calculate the attention
            value: Compute the attention weights
            freqs_cis: Calculate the frequency of each word in the
                vocabulary
            position_ids: Identify the position of each token in the
                sequence

        Returns:
            A tuple of 3 tensors: query_states, key and value
        """
        query = query.reshape(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim
        )
        key = key.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )
        value = value.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )

        query, key, value = self._transpose_sequence_head(query, key, value)

        query, key = self.rotary(
            query=query, key=key, freqs_cis=freqs_cis, position_ids=position_ids
        )

        return self._transpose_sequence_head(query, key, value)

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        causal_mask: chex.Array,
        position_ids: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = True,
    ):
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[
            ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
        ]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            batch_size=batch_size,
            sequence_length=sequence_length,
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
        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
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

        attentions = self.attention_module.__call__(
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
            segment_ids=segment_ids,
            causal_mask=causal_mask,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.o_proj(attn_output)

        outputs = (
            (attn_output, attentions.attention_weights)
            if output_attentions
            else (attn_output,)
        )
        return outputs


class FlaxPhi3DecoderLayer(nn.Module):
    config: Phi3Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        # hidden_states: chex.Array,
        # freqs_cis: Tuple[chex.Array, chex.Array],
        # attention_mask: Optional[chex.Array],
        # position_ids: Optional[chex.Array],
        # causal_mask: Optional[chex.Array],
        # segment_ids: Optional[chex.Array] = None,
        # deterministic: bool = True,
        # output_attentions: bool = False,
        # init_cache: bool = False,
        attn_block = FlaxPhi3Attention
        mlp_block = FlaxPhi3MLP
        if self.config.gradient_checkpointing != "":
            # hidden_states: chex.Array,
            # freqs_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: Optional[chex.Array],
            # position_ids: Optional[chex.Array],
            # causal_mask: Optional[chex.Array],
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # output_attentions: bool = False,
            # init_cache: bool = False,
            attn_block = nn.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1, 3, 4, 6, 7, 8, 9),
            )
            mlp_block = nn.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
                static_argnums=(1,),
            )
        self.self_attn = attn_block(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.mlp = mlp_block(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.input_layernorm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.resid_attn_dropout = nn.Dropout(self.config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(self.config.resid_pdrop)
        self.post_attention_layernorm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: Optional[chex.Array],
        position_ids: Optional[chex.Array],
        causal_mask: Optional[chex.Array],
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        init_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(
            hidden_states,
            freqs_cis,  # type:ignore
            attention_mask,
            causal_mask,
            position_ids,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
        )
        attn_outputs, self_attn_weights = (
            (attn_out[0], attn_out[1]) if len(attn_out) == 2 else (attn_out[0], None)
        )

        hidden_states = (
            self.resid_attn_dropout(attn_outputs, deterministic=deterministic)
            + residual
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp, hidden_states, self.config.scan_mlp_chunk_size, deterministic
            )
        else:
            feed_forward_hidden_states = self.mlp(
                hidden_states,
                deterministic,
            )

        hidden_states = residual + self.resid_mlp_dropout(
            feed_forward_hidden_states, deterministic=deterministic
        )
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class FlaxPhiDecoderLayerCollection(nn.Module):
    config: Phi3Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxPhi3DecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(idx),
                layer_idx=idx,
            )
            for idx in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: Optional[chex.Array],
        position_ids: Optional[chex.Array],
        causal_mask: Optional[chex.Array],
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        init_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple[tuple, ...], FlaxBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # hidden_states: chex.Array,
            # freqs_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: Optional[chex.Array],
            # position_ids: Optional[chex.Array],
            # causal_mask: Optional[chex.Array],
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # output_attentions: bool = False,
            # init_cache: bool = False,
            layer_outputs = decoder_layer(
                hidden_states,
                freqs_cis,
                attention_mask,
                position_ids,
                causal_mask,
                segment_ids,
                deterministic,
                output_attentions,
                init_cache,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns]
                if v is not None
            )
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FlaxPhi3Module(nn.Module):
    config: Phi3Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.embed_dropout = flax.linen.Dropout(config.embd_pdrop)
        self.layers = FlaxPhiDecoderLayerCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.causal_mask = flax.linen.make_causal_mask(
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

        initial_rope_kwargs = dict(rope_type="none")
        if hasattr(config, "rope_scaling"):
            if config.rope_scaling is not None:
                original_max_position_embeddings = (
                    config.original_max_position_embeddings,
                )
                long_factor = (config.rope_scaling["long_factor"],)
                short_factor = (config.rope_scaling["short_factor"],)
                rope_type = config.rope_scaling["type"]
                initial_rope_kwargs = dict(
                    long_factor=long_factor,
                    short_factor=short_factor,
                    rope_type=rope_type,
                    original_max_position_embeddings=original_max_position_embeddings,
                )
        self.freqs_cis = precompute_freqs_cis(
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

    def __call__(
        self,
        input_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        extra_embedding: Optional[chex.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        init_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple[tuple[Any, ...], ...], FlaxBaseModelOutput]:
        if input_ids is None and input_embeds is None:
            raise RuntimeError("Both `input_ids` and `input_embeds` can not be None !")
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids.astype("i4"))
        input_embeds = self.embed_dropout(input_embeds, deterministic=deterministic)
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")
        if position_ids is None:
            position_ids = (
                (jnp.cumsum(attention_mask) - 1)
                .reshape(batch_size, sequence_length)
                .astype("i4")
            )
        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        input_embeds = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )

        outputs = self.layers(
            hidden_states=input_embeds,
            freqs_cis=self.freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            hidden_states=outputs[1] if output_hidden_states else None,
            attentions=outputs[-1] if output_attentions else None,
        )


class FlaxPhi3ForCausalLMModule(nn.Module):
    config: Phi3Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.model = FlaxPhi3Module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.vocab_size = self.config.vocab_size
        self.lm_head = Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nnx.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        extra_embedding: Optional[chex.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        init_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple[Any, ...], FlaxMaskedLMOutput]:
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            extra_embedding=extra_embedding,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        outputs = (res.last_hidden_state, res.hidden_states, res.attentions)
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"]
            shared_kernel = fjformer.linen.control_quantization(
                shared_kernel, self.param_dtype
            ).T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, res.last_hidden_state
            )
        else:
            lm_logits = self.lm_head(res.last_hidden_state)

        lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits, hidden_states=res.hidden_states, attentions=res.attentions
        )


class FlaxPhiPreTrainedModel(BaseNNXModule):
    """Phi pre-trained model."""

    module_class = None
    config_class = Phi3Config
    base_model_prefix = "transformer"

    def __init__(
        self,
        config: Phi3Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        input_shape=(1, 1),
        seed: int = 42,
        _do_init: bool = False,
    ) -> None:
        module = self.module_class(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        super().__init__(
            config=config,
            module=module,
            input_shape=input_shape,
            _do_init=_do_init,
            seed=seed,
        )

    def init_cache(self, batch_size, max_length):

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

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask)

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
        return_dict: Optional[bool] = True,
        extra_embedding: Optional[jnp.ndarray] = None,
        add_params_field: bool = False,
        **kwargs,
    ):

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

        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if self.config.bits is not None:
            rngs["params"] = jax.random.key(0)

        inputs = (
            {"params": params or self.params}
            if add_params_field
            else params or self.params
        )

        if past_key_values is not None:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids=input_ids,
            input_embeds=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            extra_embedding=extra_embedding,
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            init_cache=False,
            return_dict=return_dict,
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


class FlaxPhi3Model(FlaxPhiPreTrainedModel):
    module_class = FlaxPhi3Module


class FlaxPhi3ForCausalLM(FlaxPhiPreTrainedModel):
    module_class = FlaxPhi3ForCausalLMModule