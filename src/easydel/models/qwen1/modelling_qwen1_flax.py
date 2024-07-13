import functools
import math
from typing import Optional, List, Union

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax
from easydel.models.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxSequenceClassifierOutput,
)
from easydel.models.caching_utils import KVCache
from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.common import RMSNorm as RMSNorm
from easydel.models.flax_modelling_utils import (
    BaseAttentionModule,
    control_mlp_sharding,
    block_wise_ffn,
    rotate_half,
    with_sharding_constraint,
)
from jax.sharding import PartitionSpec
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.qwen1.qwen1_configuration import Qwen1Config as Qwen1Config


def apply_rotary_pos_emb(t: chex.Array, freqs):
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.astype(jnp.float32)
    t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
    t_rot = (t_rot * cos) + (rotate_half(t_rot) * sin)
    return jnp.concatenate((t_rot, t_pass), axis=-1).astype(t.dtype)


def apply_rope(
    query: chex.Array,
    key: chex.Array,
    rotary_pos_emb_list: list[chex.Array] | None = None,
    position_ids: chex.Array | None = None,
    dtype: jnp.dtype = jnp.float32,
):
    if rotary_pos_emb_list is not None:
        current_length = query.shape[1]
        if len(rotary_pos_emb_list) == 1:
            rotary_pos_emb = rotary_pos_emb_list[0]
            rotary_pos_emb = [i[:, -current_length:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)
        else:
            query_list = []
            key_list = []
            for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                rotary_pos_emb = [i[:, -current_length:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query_list += [
                    apply_rotary_pos_emb(
                        query[
                            i : i + 1,
                            :,
                            :,
                        ],
                        q_pos_emb[1],
                        q_pos_emb[0],
                    )
                ]
                key_list += [
                    apply_rotary_pos_emb(
                        key[
                            i : i + 1,
                            :,
                            :,
                        ],
                        k_pos_emb[1],
                        k_pos_emb[0],
                    )
                ]
            query = jnp.concatenate(query_list, axis=0)
            key = jnp.concatenate(key_list, axis=0)
    return query.astype(dtype), key.astype(dtype)


def compute_qwen1_rope(
    dim: int,
    seqlen,
    base: int | float = 10000,
    ntk_alpha=1,
):
    base = base * ntk_alpha ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    new_seq_len_cached = max(2 * seqlen, 16)
    seq = jnp.arange(new_seq_len_cached)
    freqs = jnp.outer(seq.astype(inv_freq.dtype), inv_freq)

    emb = jnp.concatenate([freqs, freqs], axis=-1)
    emb = rearrange(emb, "n d -> 1 n 1 d")

    return jnp.cos(emb), jnp.sin(emb)


class Qwen1MLP(nnx.Module):
    def __init__(
        self,
        config: Qwen1Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config

        self.w1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size // 2,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=not config.no_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.w2 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size // 2,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=not config.no_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.intermediate_size // 2,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=not config.no_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        Args:
            self: Represent the instance of the class
            hidden_states: jnp.ndarray: Pass in the input to the layer

        Returns:
            A tensor that is the result of applying a dropout function
            to x
        """

        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        hidden_states = self.c_proj(
            jax.nn.silu(self.w2(hidden_states)) * self.w1(hidden_states)
        )
        return hidden_states


class Qwen1Attention(BaseAttentionModule):
    def __init__(
        self,
        config: Qwen1Config,
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
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.projection_size = config.kv_channels * config.num_attention_heads
        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )

        self.c_attn = nnx.Linear(
            config.hidden_size,
            self.projection_size * 3,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.c_proj = nnx.Linear(
            self.projection_size * 3,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=not config.no_bias,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        # logn_list = [
        #     math.log(i, self.config.seq_length) if i > self.config.seq_length else 1
        #     for i in range(1, 32768)
        # ]
        # logn_tensor = jnp.asarray(logn_list)[None, :, None, None]
        # self.logn_tensor = logn_tensor
        self.rotary = functools.partial(apply_rope, dtype=self.dtype)
        self.attention_module = FlexibleAttentionModule(
            head_dims=self.head_dim,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.config.attn_dtype,
            partition_axis=self.config.partition_axis,
            mesh=self.config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def apply_rotary(self, query, key, rotary_pos_emb_list, position_ids):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, rotary_pos_emb_list, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            query_states: Calculate the attention weights
            key: Calculate the attention
            rotary_pos_emb_list: Calculate the frequency of each word in  the vocabulary
            position_ids: Identify the position of each token in the sequence

        Returns:
            A tuple of 3 tensors: query_states, key and value
        """
        query, key = self.rotary(
            position_ids=position_ids,
            query=query,
            key=key,
            rotary_pos_emb_list=rotary_pos_emb_list,
        )
        return query, key

    def __call__(
        self,
        hidden_states: chex.Array,
        rotary_pos_emb_list: chex.Array,
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
        mixed_x_layer: chex.Array = self.c_attn(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_x_layer, 3, 2)

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            rotary_pos_emb_list=rotary_pos_emb_list,
            batch_size=batch_size,
            sequence_length=sequence_length,
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
        attn_output = self.c_proj(attn_output)

        return attn_output, attentions.attention_weights


class Qwen1Block(nnx.Module):
    def __init__(
        self,
        config: Qwen1Config,
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
        attn_block = Qwen1Attention
        mlp_block = Qwen1MLP

        self.attn = attn_block(
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
        self.ln_1 = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ln_2 = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        rotary_pos_emb_list: chex.Array,
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
        attn_outputs = self.attn(
            hidden_states=self.ln_1(hidden_states),
            rotary_pos_emb_list=rotary_pos_emb_list,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ln_2(hidden_states)

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

        return (hidden_states,) + attn_outputs[1:]


class Qwen1Model(BaseNNXModule):
    def __init__(
        self,
        config: Qwen1Config,
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
            config.vocab_size,
            config.hidden_size,
            embedding_init=nnx.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.drop = nnx.Dropout(rate=config.emb_dropout_prob, rngs=rngs)
        self.h = [
            Qwen1Block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self._rope_cache = None
        self._causal_mask = None

    @property
    def rope_cache(self):
        if self._rope_cache is None:
            config = self.config
            if config.rotary_pct == 1.0:
                rotary_ndims = None
            else:
                assert config.rotary_pct < 1
                rotary_ndims = int(config.kv_channels * config.rotary_pct)
            self._rope_cache = compute_qwen1_rope(
                dim=(rotary_ndims if rotary_ndims is not None else config.kv_channels),
                base=self.config.rotary_emb_base,
                seqlen=getattr(
                    config,
                    "freq_max_position_embeddings",
                    config.seq_length,
                ),
            )
            return self._rope_cache

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
            segment_ids: (Optional(chex.Array)): Indicate the segments in input.input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
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
        hidden_states = self.drop(input_embeds)
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        for idx, layer in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weight = layer(
                hidden_states=hidden_states,
                rotary_pos_emb_list=[self.rope_cache],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values[idx],
                segment_ids=segment_ids,
            )
            if output_attentions:
                all_attentions += (attn_weight,)

        hidden_states = self.ln_f(hidden_states)

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

    def set_input_embeddings(self, value):
        self.wte = value

    def get_input_embeddings(self):
        return self.wte


class Qwen1ForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: Qwen1Config,
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
        self.transformer = Qwen1Model(
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
            segment_ids: (Optional(chex.Array)): Indicate the segments in input.
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of: logits and predictions
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
            segment_ids=segment_ids,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.embedding.embedding.value.T
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

    def set_input_embeddings(self, value):
        self.model.wte = value

    def get_input_embeddings(self):
        return self.model.wte

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class Qwen1ForSequenceClassification(BaseNNXModule):
    def __init__(
        self,
        config: Qwen1Config,
        num_classes: int,
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
        self.transformer = Qwen1Model(
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
            segment_ids: (Optional(chex.Array)): Indicate the segments in input.
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of: logits and predictions
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            extra_embedding=extra_embedding,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            segment_ids=segment_ids,
        )

        prediction = self.classifier(outputs.last_hidden_state)
        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=prediction,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return tuple(
            [
                s
                for s in [
                    prediction,
                    outputs.hidden_states,
                    outputs.attentions,
                ]
                if s is not None
            ]
        )
