import math

import jax.nn.initializers
from jax import numpy as jnp

from functools import partial

from jax.experimental.pjit import with_sharding_constraint as _wish_sharding_constraint
from jax.interpreters import pxla
from flax import linen as nn
from transformers import FlaxPreTrainedModel, PretrainedConfig
from jax.sharding import PartitionSpec
import flax
from einops import rearrange
from typing import Dict, Any, Optional, Tuple, List
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

ACT2CLS = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu6": nn.relu6,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "swish": nn.swish,
    "tanh": nn.tanh,
}


class FlaxLTConfig(PretrainedConfig):
    def __init__(self,
                 initializer_range: float = 0.02,
                 hidden_size: int = 4096,
                 bos_token_id=2,
                 eos_token_id=1,
                 pad_token_id=0,
                 intermediate_size: int = 8192,
                 num_hidden_layers: int = 32,
                 vocab_size: int = 32000,
                 num_attention_heads: int = 32,
                 weight_decay: float = 0.02,
                 max_sequence_length: int = 2048,
                 softmax_scale: float = None,
                 alibi_bias_max: int = 8,
                 fsdp=False,
                 hidden_act="silu",
                 **kwargs
                 ):
        super().__init__(eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id)
        self.max_sequence_length = max_sequence_length
        self.weight_decay = weight_decay
        self.alibi_bias_max = alibi_bias_max
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.softmax_scale = softmax_scale
        self.fsdp = fsdp
        self.hidden_act = hidden_act
        self.__dict__.update(**kwargs)

    @staticmethod
    def get_partition_rules():
        return (
            # Emb
            ("model/wte/embedding", PartitionSpec("mp", "fsdp")),
            ("attn/(k_proj|v_proj|q_proj)/kernel", PartitionSpec("fsdp", "mp")),
            ("attn/o_proj/kernel", PartitionSpec("mp", "fsdp")),
            ("mlp/down/kernel", PartitionSpec("mp", "fsdp")),
            ("mlp/up/kernel", PartitionSpec("fsdp", "mp")),
            ("lm_head/kernel", PartitionSpec("fsdp", "mp")),
            ('.*', PartitionSpec(None)),
            ('ln/kernel', PartitionSpec(None)),
            ('ln1/kernel', PartitionSpec(None)),
            ('ln2/kernel', PartitionSpec(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')


def is_name(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def get_partition_names(partition):
    names = set()
    for name in partition:
        if name is None:
            continue
        elif isinstance(name, str):
            names.add(name)
        else:
            names.update(get_partition_names(name))
    return names


def with_sharding_constraint(x, p):
    names = get_partition_names(p)
    if is_name(*names):
        x = _wish_sharding_constraint(x, p)
    return x


class LTSelfAttention(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        dense = partial(nn.Dense,
                        features=self.config.hidden_size,
                        use_bias=False,
                        kernel_init=jax.nn.initializers.normal(self.config.initializer_range)
                        )
        self.q_proj = dense()
        self.o_proj = dense()
        self.k_proj = dense()
        self.v_proj = dense()
        self.scale = 1 / math.sqrt(self.config.hidden_size // self.config.hidden_size)

    def __call__(self, hidden_state: jnp.DeviceArray, attention_mask=None):
        b, t, c = hidden_state.shape
        wq, wk, wv = self.q_proj(hidden_state), self.k_proj(hidden_state), self.v_proj(hidden_state)

        if self.config.fsdp:
            wk = with_sharding_constraint(wk, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            wq = with_sharding_constraint(wq, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            wv = with_sharding_constraint(wv, PartitionSpec(('dp', 'fsdp'), None, 'mp'))

        wq = rearrange(wq, 'b s (h d) -> b h s d', h=self.config.num_attention_heads)
        wk = rearrange(wk, 'b s (h d) -> b h d s', h=self.config.num_attention_heads)
        wv = rearrange(wv, 'b s (h d) -> b h s d', h=self.config.num_attention_heads)

        attn_weights = jnp.matmul(wq, wk) / self.scale

        # attention_mask batch,head_size,seq,seq
        if attention_mask is not None:
            attn_weights = jnp.add(attention_mask, attn_weights)

        if self.config.fsdp:
            attn_weights = with_sharding_constraint(attn_weights, PartitionSpec(("dp", "fsdp"), "mp", None, None))

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        value = jnp.matmul(attn_weights, wv).reshape(b, t, c)
        return self.o_proj(value)


class LTMlp(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.up = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range
            )
        )
        self.down = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range
            )
        )
        self.act = ACT2CLS[self.config.hidden_act]

    def __call__(self, hidden_state: jnp.DeviceArray):
        return self.down(self.act(self.up(hidden_state)))


class LTBlock(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.attn = LTSelfAttention(
            config=self.config,
            param_dtype=self.param_dtype,
            dtype=self.dtype
        )
        self.mlp = LTMlp(
            config=self.config,
            param_dtype=self.param_dtype,
            dtype=self.dtype
        )
        self.ln1 = nn.LayerNorm(
            use_bias=False
        )
        self.ln2 = nn.LayerNorm(
            use_bias=False
        )

    def __call__(self, hidden_state: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None):
        hidden_state = hidden_state + self.attn(self.ln1(hidden_state), attention_mask)
        hidden_state = hidden_state + self.mlp(self.ln2(hidden_state))
        return hidden_state


class LTCollection(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.layers = [LTBlock(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype) for _ in
                       range(self.config.num_hidden_layers)]

    def __call__(self, hidden_state: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None):
        cache = []
        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)
            cache.append(hidden_state)
        return hidden_state, cache


class FlaxLTPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    module_config: FlaxLTConfig
    base_model_prefix: str = 'model'

    def __init__(
            self,
            config: FlaxLTConfig,
            input_shape=(1, 256),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = False,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape,
                     params: flax.core.FrozenDict = None, add_params_field=True) -> Dict:
        input_ids = jnp.ones(input_shape, dtype='i4')
        attention_mask = jnp.ones(input_shape, dtype='i4')
        if params is None:
            params = self.module.init(rng, input_ids, attention_mask)['params']
        return {'params': params} if add_params_field else params

    def __call__(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            params: dict = None,
            return_dict: Optional[bool] = None,
            add_params_field: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        inputs = {'params': params or self.params} if add_params_field else params or {'params': self.params}

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4") if attention_mask is not None else attention_mask,
            return_dict=return_dict,

        )

        return outputs


class FlaxLTModelModule(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.blocks = LTCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        self.ln = nn.LayerNorm(use_bias=False)

    def build_alibi(self, sequence_length: int):
        mxl = jnp.arange(1 - sequence_length, 1).reshape(1, 1, 1, -1)
        mxh = jnp.arange(1, 1 + self.config.num_attention_heads).reshape(1, -1, 1, 1)
        cp2 = 2 ** math.ceil(math.log2(self.config.num_attention_heads))
        base_mxl = mxh * (self.config.alibi_bias_max / cp2)
        slope = 1 / jnp.power(2, base_mxl)
        if self.config.num_attention_heads != cp2:
            slope = jnp.concatenate([slope[1::2], slope[::2]], axis=-1)[:self.config.num_attention_heads]
        mxl = (mxl * slope).reshape(1, self.config.num_attention_heads, 1, sequence_length)
        return mxl

    def __call__(self, input_ids: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None, return_dict: bool = True):
        b, s = input_ids.shape
        hidden_state = self.wte(input_ids.astype(dtype='i4'))
        alibi = self.build_alibi(s)
        if attention_mask is None:
            attention_mask = jnp.ones((b, s))
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :]
        assert attention_mask.ndim == 4
        attention_mask = jnp.where(attention_mask == 1, 0, jnp.finfo(hidden_state.dtype).min)

        attention_mask = jnp.add(attention_mask, alibi)

        causal_mask = nn.attention.make_causal_mask(input_ids)
        causal_mask = jnp.where(causal_mask == 1, 0, jnp.finfo(hidden_state.dtype).min)

        attention_mask = jnp.add(attention_mask, causal_mask)

        hidden_state, hidden_states = self.blocks(
            hidden_state=hidden_state, attention_mask=attention_mask
        )

        hidden_state = self.ln(hidden_state)
        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_state,
                hidden_states=hidden_states
            )
        else:
            return hidden_state, hidden_states


class FlaxLTModel(FlaxLTPretrainedModel):
    module_class = FlaxLTModelModule


class FlaxLTModelForCausalLMModule(nn.Module):
    config: FlaxLTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = FlaxLTModelModule(
            config=self.config, dtype=self.dtype, param_dtype=self.param_dtype
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range)
        )

    def __call__(self, input_ids: jnp.DeviceArray, attention_mask: jnp.DeviceArray = None, return_dict: bool = True):
        base_model_prediction = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        hidden_state = base_model_prediction.last_hidden_state if return_dict else base_model_prediction[0]
        hidden_states = base_model_prediction.hidden_states if return_dict else base_model_prediction[1]
        logits = self.lm_head(hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=logits,
                hidden_states=hidden_states,
                attentions=None
            )
        else:
            return logits, hidden_states


class FlaxLTForCausalLM(FlaxLTPretrainedModel):
    module_class = FlaxLTModelForCausalLMModule
