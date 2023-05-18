import math
import jax
from jax import jit, vmap, pmap
from flax import linen as nn
from jax import numpy as jnp
from transformers import PretrainedConfig, FlaxPreTrainedModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from functools import partial
from typing import Optional
from einops import rearrange


class FlaxLLamaConfig(PretrainedConfig):
    # HuggingFace FlaxLLamaConfig
    model_type = "LLama"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


ACT2CLS = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu6": nn.relu6,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "swish": nn.swish,
    "tanh": nn.tanh,
}


def compute_freq(dim: int, man_length: int, theta: int = 10000):
    freq = 1 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(man_length)
    m = jnp.einsum('i,j->ij', t, freq)
    m = jnp.concatenate([m, m], axis=-1)
    cos = jnp.cos(m)
    sin = jnp.sin(m)
    return cos, sin


def rotate_half(tensor):
    depth = tensor.shape[-1]
    x1 = tensor[..., :depth]
    x2 = tensor[..., depth:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_embedding(q, k, c, s):
    b, h, l, d = q.shape
    c = c[0, 0, :l, :]
    s = s[0, 0, :l, :]
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    return q, k


class PMSNorm(nn.Module):
    dim: int
    eps: float
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param('weight', nn.ones, (self.dim,), self.dtype)

    def norm(self, x):
        return x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps))

    def __call__(self, x):
        return self.weight * self.norm(x)


class RoEM(nn.Module):
    config: FlaxLLamaConfig

    def setup(self) -> None:
        dim = self.config.hidden_size // self.config.num_attention_heads
        self.cos, self.sin = compute_freq(dim, self.config.max_position_embeddings)
        self.dim = dim

    def __call__(self, x, max_l):
        if self.sin.shape[0] < max_l:
            self.cos, self.sin = compute_freq(self.dim, max_l)
        return self.cos[jnp.newaxis, jnp.newaxis, :, :], self.sin[jnp.newaxis, jnp.newaxis, :, :]


class LLamaSelfAttention(nn.Module):
    config: FlaxLLamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        dense = partial(nn.Dense,
                        features=self.config.hidden_size,
                        kernel_init=nn.initializers.normal(self.config.initializer_range),
                        use_bias=False, dtype=self.dtype, param_dtype=self.dtype
                        )
        self.k_proj = dense()
        self.v_proj = dense()
        self.q_proj = dense()
        self.o_proj = dense()
        self.rotary_embedding = RoEM(config=self.config)

    def __call__(self, input_ids: jnp.array, attention_mask=None):
        b, t, c = input_ids.shape
        vs = (b, t, self.config.num_attention_heads, c // self.config.num_attention_heads)
        k = self.k_proj(input_ids).reshape(vs).swapaxes(1, 2)
        q = self.q_proj(input_ids).reshape(vs).swapaxes(1, 2)
        v = self.v_proj(input_ids).reshape(vs).swapaxes(1, 2)

        cos, sin = self.rotary_embedding(x=k, max_l=t)

        k, q = apply_rotary_embedding(k=k, q=q, c=cos, s=sin)
        attn = q @ k.swapaxes(2, 3) / math.sqrt(k.shape[-1])
        if attention_mask is not None:
            # assert attention_mask.shape == [b, 1, t, kv_seq_length]
            attn += attention_mask
        attn = nn.softmax(attn, axis=-1)
        attn = (attn @ v).swapaxes(1, 2).reshape(b, t, c)
        return self.o_proj(attn)


class LLamaMLP(nn.Module):
    config: FlaxLLamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        dense = partial(nn.Dense,
                        kernel_init=nn.initializers.normal(self.config.initializer_range),
                        use_bias=False, dtype=self.dtype, param_dtype=self.dtype
                        )
        self.gate_proj = dense(features=self.config.intermediate_size)
        self.up_proj = dense(features=self.config.intermediate_size)
        self.down_proj = dense(features=self.config.hidden_size)
        self.act = ACT2CLS[self.config.hidden_act]

    def __call__(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class LLamaBlock(nn.Module):
    config: FlaxLLamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.self_attn = LLamaSelfAttention(config=self.config, dtype=self.dtype)
        self.mlp = LLamaMLP(config=self.config, dtype=self.dtype)
        self.input_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.post_attention_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps,
                                                dtype=self.dtype)

    def __call__(self, hidden_state, attention_mask=None):
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask)
        hidden_state = hidden_state + residual
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = hidden_state + residual
        return hidden_state


class FlaxLLamaPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    module_config: FlaxLLamaConfig
    base_model_prefix: str = 'model'

    def __init__(
            self,
            config: FlaxLLamaConfig,
            input_shape=(1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = False,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng, input_shape, params = None):
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            return_dict=False,
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

class FlaxLLamaModule(nn.Module):
    config: FlaxLLamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        self.layers = [LLamaBlock(self.config, dtype=self.dtype) for _ in range(self.config.num_hidden_layers)]
        self.norm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def __call__(self,
                 input_ids: jnp.array = None,
                 attention_mask: jnp.array = None,

                 return_dict=True):

        hidden_state = self.embed_tokens(input_ids)

        b, s, _ = hidden_state.shape
        if attention_mask is None:
            attention_mask = jnp.ones((b, 1, s, s))
        attention_mask = jnp.where(nn.make_causal_mask(input_ids) == 0, -6548, 0) + jnp.where(attention_mask > 0, 0,
                                                                                              -6548)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask=attention_mask)

        hidden_state = self.norm(hidden_state)
        if return_dict:
            return FlaxBaseModelOutput(
                hidden_states=hidden_state,
                last_hidden_state=hidden_state
            )
        else:
            return hidden_state


class FlaxLLamaModel(FlaxLLamaPretrainedModel):
    module_class = FlaxLLamaModule


class FlaxLLamaForCausalLMModule(nn.Module):
    config: FlaxLLamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = FlaxLLamaModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(features=self.config.vocab_size, use_bias=False, dtype=self.dtype,
                                param_dtype=self.dtype,
                                kernel_init=nn.initializers.normal(self.config.initializer_range))

    def __call__(self,
                 input_ids: jnp.array,
                 attention_mask: jnp.array = None,
                 return_dict: Optional[bool] = False,
                 ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=return_dict,
                            )

        hidden_state = output.last_hidden_state if return_dict else output

        pred = self.lm_head(hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=pred,
                hidden_states=hidden_state,
                # attentions=None,
            )
        else:
            return pred,


class FlaxLLamaForCausalLM(FlaxLLamaModel):
    module_class = FlaxLLamaForCausalLMModule