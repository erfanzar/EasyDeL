import math
from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional, Union, Tuple

from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
import flax
from einops import rearrange
from flax.linen.partitioning import remat
from ..easy_attention import EasyAttention
from ..flax_modelling_utils import (
    get_gradient_checkpoint_policy,
    with_sharding_constraint,
    ACT2FN,
    get_dot_general_by_bits, BaseJAXAttentionModule, block_wise_ffn
)
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
import chex

from .mosaic_configuration import MptConfig


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.bfloat16))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxMptMLP(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.up = nn.Dense(
            self.config.d_model * self.config.expansion_ratio,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.down = nn.Dense(
            self.config.d_model,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.act = ACT2FN[self.config.act_fn]

    def __call__(
            self,
            hidden_states: chex.Array,
            e: bool = True  # Ignored
    ):
        return self.down(self.act(self.up(hidden_states)))


class FlaxMptAttention(BaseJAXAttentionModule):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:

        self.w_qkv = nn.Dense(
            self.config.d_model * 3,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision)
        self.wo = nn.Dense(
            self.config.d_model,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
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
            attention_dropout=self.config.attention_dropout,
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
            sm_scale=1 / math.sqrt(self.config.n_heads)
        )
        if self.config.qk_ln:
            self.q_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
            self.k_ln = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.causal_mask = nn.make_causal_mask(
            jnp.ones(
                (1, self.config.max_seq_len),
                dtype="bool"
            ), dtype="bool"
        )

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 position_ids: chex.Array,
                 attn_bias: chex.Array = None,
                 init_cache: bool = False
                 ):

        """
        The __call__ function is the main function of a JAX module.
        It takes in inputs and returns outputs, just like any other Python function.
        The difference is that __call__ can also take in state (e.g., parameters) from the module itself,
        and it can update that state as part of its computation.

        :param self: Access variables that belong to the class
        :param hidden_states: chex.Array: Pass the input to the attention layer
        :param attention_mask: chex.Array: Mask out certain positions in the sequence
        :param position_ids: chex.Array: Specify the position of each token in the sequence
        :param attn_bias: chex.Array: Add a bias to the attention scores
        :param init_cache: bool: Initialize the cache
        :return: The output of the attention layer
        
        """
        inp_shape = hidden_states.shape
        b, s, ds = inp_shape
        qkv = self.w_qkv(hidden_states)
        q, k, v = jnp.split(qkv, 3, -1)
        if self.config.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        if self.config.use_pjit_attention_force:
            q = with_sharding_constraint(q, PartitionSpec(("dp", "fsdp"), None, "sp"))
            k = with_sharding_constraint(k, PartitionSpec(("dp", "fsdp"), None, "sp"))
            v = with_sharding_constraint(v, PartitionSpec(("dp", "fsdp"), None, "sp"))
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.n_heads)
        attention_mask = attention_mask.reshape(b, 1, 1, -1)
        if self.has_variable('cache', 'key_states') or init_cache:
            k, v, attention_mask = self._concatenate_to_cache(key_states=k, value=v, query=q, attention_mask=attention_mask)
        # TODO: MPT WONT WORK CAUSE OF NEW ATTENTION MEC ON FJFORMER
        q_l = q.shape[1]
        k_l = k.shape[1]
        dropout_rng = None
        deterministic = False
        if deterministic:
            dropout_rng = self.make_rng("dropout")

        d = q.shape[-1]
        attn_output = jnp.einsum('...qhd,...khd->...hqk', q, k, precision=self.precision) * jax.lax.rsqrt(
            jnp.asarray(d).astype(v.dtype))
        if self.config.use_pjit_attention_force:
            attn_output = with_sharding_constraint(attn_output, PartitionSpec(("dp", "fsdp"), "sp", None, None))
        if attn_bias is not None:
            attn_output += attn_bias[:, :, :, :attn_output.shape[-1]]
        mask = jnp.where(self.causal_mask == 1, 0, jnp.finfo(attn_output).min)
        if attention_mask is not None:
            attention_mask = jnp.where(
                attention_mask == 1,
                0,
                jnp.finfo(attn_output).min
            )
            attn_output += attention_mask
        attn_output += mask[:, :, :attn_output.shape[-2], :attn_output.shape[-1]]
        attn_output = nn.softmax(attn_output, -1)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_output, v)
        return self.wo(attn_output.reshape(inp_shape))


class FlaxMptBlock(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.norm_1 = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.norm_2 = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        attn_block = FlaxMptAttention
        mlp_block = FlaxMptMLP
        if self.config.gradient_checkpointing != "":
            mlp_block = remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )
            # hidden_states: chex.Array
            # attention_mask: chex.Array
            # position_ids: chex.Array
            # attn_bias: chex.Array = None
            # init_cache: bool = False

            attn_block = remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(4,)
            )
        self.attn = attn_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ffn = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 position_ids: chex.Array,
                 attn_bias: chex.Array = None,
                 init_cache: bool = False
                 ):

        # hidden_states: chex.Array
        # attention_mask: chex.Array
        # position_ids: chex.Array
        # attn_bias: chex.Array = None
        # init_cache: bool = False

        hidden_states = (
                self.attn(
                    self.norm_1(hidden_states),
                    attention_mask,
                    position_ids,
                    attn_bias,
                    init_cache
                ) + hidden_states
        )
        ffn_input = self.norm_2(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.ffn,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                False
            )
        else:
            feed_forward_hidden_states = self.ffn(
                hidden_states,
                False,
            )
        return feed_forward_hidden_states + hidden_states


class FlaxMptCollection(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = FlaxMptBlock
        self.blocks = [
            block(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(
                self.config.n_layers
            )
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            attn_bias: chex.Array = None,
            init_cache: bool = False,
            output_hidden_states: bool = True
    ):

        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                position_ids=position_ids,
                init_cache=init_cache
            )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


def build_alibi(max_length, num_attention_heads, alibi_max: int = 8):
    w_range = jnp.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    # cp2 = jnp.power(2, jnp.ceil(jnp.log2(num_attention_heads)))
    cp2 = 2 ** math.ceil(math.log2(num_attention_heads))
    h_range = jnp.arange(1, 1 + num_attention_heads, ).reshape(1, -1, 1, 1)
    h_range = jnp.matmul(h_range, jnp.asarray(alibi_max / cp2).reshape(1, 1))
    slop = 1 / jnp.power(2, h_range)
    if cp2 != num_attention_heads:
        slop = jnp.concatenate([slop[1::2], slop[::2]], axis=-1)[:num_attention_heads]
    alibi = (w_range * slop).reshape(1, num_attention_heads, 1, max_length)
    return alibi


class FlaxMptModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.wte = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model)
        if not self.config.alibi:
            self.wpe = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.max_seq_len)
        self.h = FlaxMptCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_f = nn.LayerNorm(use_bias=self.config.use_norm_bias)
        self.alibi = build_alibi(self.config.max_seq_len, self.config.n_heads)

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            init_cache: bool = False,
            return_dict: bool = True,
            output_hidden_states: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        b, s = input_ids.shape
        hidden_states = self.wte(input_ids)
        hidden_states = hidden_states + extra_embedding if extra_embedding is not None else hidden_states

        if self.config.alibi:
            alibi = self.alibi
        else:
            pos_id = self.wpe(jnp.arange(s, dtype='i4').reshape(1, -1))
            hidden_states += pos_id
            alibi = None
        hidden_states, all_hidden_states = self.h(
            hidden_states,
            attn_bias=alibi,
            attention_mask=attention_mask,
            position_ids=position_ids,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states
        )
        hidden_states = self.norm_f(
            hidden_states
        )
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_dict:
            return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
        else:
            return hidden_states, all_hidden_states


class FlaxMptPretrainedModel(EasyDelFlaxPretrainedModel):
    module_class: nn.Module = None
    config_class: MptConfig = MptConfig

    def __init__(self,
                 config,
                 dtype: jnp.dtype = jnp.float32,
                 param_dtype: jnp.dtype = jnp.float32,
                 _do_init: bool = False,
                 precision: Optional[Union[jax.lax.Precision, None]] = jax.lax.Precision("fastest"),
                 input_shape: Tuple = (1, 16),
                 **kwargs
                 ):
        module = self.module_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision
        )
        super().__init__(_do_init=_do_init, config=config, input_shape=input_shape, module=module, **kwargs)

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
            init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.ones(input_shape, dtype='i4')
        if params is None:
            return self.module.init(
                rngs=rng,
                input_ids=input_ids,
                attention_mask=jnp.ones(input_shape, dtype='i4'),
                position_ids=jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape),
                init_cache=False
            )['params']
        else:
            return params

    def __call__(self,
                 input_ids,
                 attention_mask=None,
                 past_key_values=None,
                 position_ids=None,
                 output_hidden_states: Optional[bool] = None,
                 init_cache: bool = False,
                 params: dict = None,
                 add_params_field: bool = False,
                 return_dict: bool = True,
                 extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
                 **kwargs
                 ):

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        params = {'params': params or self.params} if add_params_field else params or self.params
        input_ids = jnp.asarray(input_ids, dtype='i4')
        mutable = False
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype='i4')
        if position_ids is None:
            position_ids = jnp.arange(0, attention_mask.shape[-1], 1, dtype='i4').reshape(
                1, -1
            ).repeat(input_ids.shape[0], axis=0)

        if past_key_values is not None:
            params['cache'] = past_key_values
            mutable = ['cache']
        rngs = {}
        if self.config.bits is not None:
            rngs['params'] = jax.random.key(0)
        predict = self.module.apply(
            params,
            input_ids=input_ids,
            attention_mask=jnp.asarray(attention_mask, dtype='i4'),
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            position_ids=position_ids,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states,
            mutable=mutable,
            rngs=rngs
        )
        if past_key_values is not None and return_dict:
            predict, past_key_values = predict
            predict["past_key_values"] = flax.core.unfreeze(past_key_values["cache"])
            return predict
        elif past_key_values is not None and not return_dict:
            predict, past_key_values = predict
            predict = predict[:1] + (flax.core.unfreeze(past_key_values["cache"]),) + predict[1:]
        return predict


class FlaxMptModel(FlaxMptPretrainedModel):
    module_class = FlaxMptModule

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class FlaxMptForCausalLMModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.transformer = FlaxMptModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        if self.config.use_lm_head:
            self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(),
                                    use_bias=self.config.use_bias,
                                    dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
                                    **get_dot_general_by_bits(self.config.bits, self.config.easy_method))

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            init_cache: bool = False,
            position_ids: chex.Array = None,
            return_dict: bool = True,
            output_hidden_states: bool = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None
    ):
        predict: FlaxBaseModelOutput = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            extra_embedding=extra_embedding,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
            init_cache=init_cache
        )
        if self.config.use_lm_head:
            logits = self.lm_head(predict.last_hidden_state)
        else:
            logits = predict.last_hidden_state @ self.transformer.wte.embedding.T
        if return_dict:

            return FlaxCausalLMOutput(
                logits=logits,
                hidden_states=predict.hidden_states
            )
        else:
            return logits, predict.hidden_states if output_hidden_states else (logits,)


class FlaxMptForCausalLM(FlaxMptPretrainedModel):
    module_class = FlaxMptForCausalLMModule

    def get_input_embeddings(self):
        return self.module.transformer.wte

    def get_decoder(self):
        return self.module.transformer

    def set_input_embeddings(self, value):
        self.module.transformer.wte = value

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.module.lm_head

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):

        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(
            batch_size, max_length
        )
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
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
