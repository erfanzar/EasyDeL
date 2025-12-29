# Adding Your Own Model

This guide walks you through the complete process of adding a new model to EasyDeL. Whether you're implementing a new architecture from scratch or porting an existing model, this guide covers all the steps.

## Overview

Adding a model to EasyDeL involves:

1. Creating a configuration class
2. Creating the model module(s)
3. Implementing weight conversion (if porting from PyTorch)
4. Registering the model
5. Testing

## Step 1: Create the Configuration

### Basic Configuration

```python
# easydel/modules/my_model/my_model_configuration.py

from easydel.infra import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("my_model")
class MyModelConfig(EasyDeLBaseConfig):
    """
    Configuration class for MyModel.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the hidden representations.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Dimension of the MLP intermediate layer.
        hidden_act: Activation function (e.g., "silu", "gelu").
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base for rotary position embeddings.
    """

    model_type: str = "my_model"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = None,
        intermediate_size: int = 11008,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def get_partition_rules(self, fully_sharded: bool = True):
        """Define how model parameters should be sharded.

        EasyDeL uses 5D sharding: (dp, fsdp, ep, tp, sp)
        - dp: Data Parallelism
        - fsdp: Fully Sharded Data Parallelism
        - ep: Expert Parallelism (for MoE)
        - tp: Tensor Parallelism
        - sp: Sequence Parallelism
        """
        return (
            # Embeddings - shard across tp and fsdp
            ("embed_tokens/embedding", PartitionSpec("tp", "fsdp")),

            # Attention projections
            ("self_attn/q_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/k_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/v_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", "fsdp")),

            # MLP layers
            ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("mlp/up_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", "fsdp")),

            # LM head
            ("lm_head/kernel", PartitionSpec("fsdp", "tp")),

            # Norms and biases (replicated)
            (".*norm.*", PartitionSpec()),
            (".*bias.*", PartitionSpec()),

            # Default
            (".*", PartitionSpec()),
        )
```

## Step 2: Create the Model Layers

### Attention Layer

```python
# easydel/modules/my_model/modeling_my_model.py

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from easydel.layers import EasyAttention, RMSNorm, Linear


class MyModelAttention(nnx.Module):
    """Multi-head attention layer for MyModel."""

    def __init__(
        self,
        config: MyModelConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projection layers
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Attention implementation
        self.attention = EasyAttention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            attn_mechanism=config.attn_mechanism,
            dtype=config.attn_dtype or dtype,
            precision=precision,
            blocksize_q=config.blocksize_q,
            blocksize_k=config.blocksize_k,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        frequencies: tuple = None,
        cache: dict = None,
        cache_index: int = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings
        if frequencies is not None:
            query, key = self.apply_rotary_embedding(query, key, frequencies, position_ids)

        # Handle KV cache for generation
        if cache is not None:
            key, value = self.update_cache(cache, cache_index, key, value)

        # Compute attention
        attn_output = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output

    def apply_rotary_embedding(self, query, key, frequencies, position_ids):
        """Apply rotary position embeddings."""
        sin, cos = frequencies
        # Implementation of RoPE
        query = self._apply_rope(query, sin, cos, position_ids)
        key = self._apply_rope(key, sin, cos, position_ids)
        return query, key
```

### MLP Layer

```python
class MyModelMLP(nnx.Module):
    """MLP layer for MyModel."""

    def __init__(
        self,
        config: MyModelConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Get activation function
        from easydel.infra.utils import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: jax.Array):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
```

### Decoder Layer

```python
class MyModelDecoderLayer(nnx.Module):
    """Single decoder layer for MyModel."""

    def __init__(
        self,
        config: MyModelConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MyModelAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = MyModelMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
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
        hidden_states: jax.Array,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        frequencies: tuple = None,
        cache: dict = None,
        cache_index: int = None,
    ):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            frequencies=frequencies,
            cache=cache,
            cache_index=cache_index,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

## Step 3: Create the Main Model

### Base Model

```python
from easydel.infra import EasyDeLBaseModule
from easydel.infra.modeling_outputs import BaseModelOutput


class MyModelModel(EasyDeLBaseModule):
    """Base MyModel without LM head."""

    def __init__(
        self,
        config: MyModelConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            MyModelDecoderLayer(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        cache: dict = None,
        return_dict: bool = True,
    ):
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Get frequencies for RoPE
        frequencies = self.frequencies

        # Process through layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                frequencies=frequencies,
                cache=cache.get(i) if cache else None,
                cache_index=i,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if return_dict:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
            )
        return hidden_states
```

### Causal LM Model

```python
from easydel.infra.modeling_outputs import CausalLMOutput
from easydel.infra.factory import register_module
from easydel.layers.base_modules import EasyDeLBaseModelForCausalLM


@register_module(
    "causal-lm",
    MyModelConfig,
    model_type="my_model",
    embedding_layer_names=["embed_tokens"],
    layernorm_names=["input_layernorm", "post_attention_layernorm", "norm"],
)
class MyModelForCausalLM(EasyDeLBaseModelForCausalLM):
    """MyModel for causal language modeling."""

    def __init__(
        self,
        config: MyModelConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.model = MyModelModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        labels: jax.Array = None,
        cache: dict = None,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.compute_loss(
                labels=labels,
                logits=logits,
            )

        if return_dict:
            return CausalLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
            )
        return (loss, logits) if loss is not None else logits

    def get_decoder(self):
        return self.model

    def get_lm_head(self):
        return self.lm_head
```

## Step 4: Implement Weight Conversion

```python
# In the model file, add the conversion logic

class MyModelForCausalLM(EasyDeLBaseModelForCausalLM):
    # ... other methods ...

    @classmethod
    def _get_torch_transform_fn(cls):
        """Return transformation function for PyTorch weights."""
        def transform_fn(key, value, config):
            # Key transformations
            key_mappings = {
                "model.embed_tokens.weight": "model/embed_tokens/embedding",
                "model.norm.weight": "model/norm/kernel",
                "lm_head.weight": "lm_head/kernel",
            }

            # Layer-wise transformations
            import re
            layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
            match = layer_pattern.match(key)
            if match:
                layer_idx = match.group(1)
                rest = match.group(2)
                key = f"model/layers/{layer_idx}/{rest}"

            # Apply static mappings
            for old, new in key_mappings.items():
                key = key.replace(old, new)

            # Value transformations
            if "kernel" in key and value.ndim == 2:
                value = value.T  # Transpose linear layers

            return key, value

        return transform_fn
```

## Step 5: Create Module Init File

```python
# easydel/modules/my_model/__init__.py

from .modeling_my_model import (
    MyModelAttention,
    MyModelMLP,
    MyModelDecoderLayer,
    MyModelModel,
    MyModelForCausalLM,
)
from .my_model_configuration import MyModelConfig

__all__ = [
    "MyModelConfig",
    "MyModelAttention",
    "MyModelMLP",
    "MyModelDecoderLayer",
    "MyModelModel",
    "MyModelForCausalLM",
]
```

## Step 6: Register in Main Modules Init

```python
# easydel/modules/__init__.py

from .my_model import (
    MyModelConfig,
    MyModelForCausalLM,
)
```

## Step 7: Testing

```python
# tests/modules/test_my_model.py

import pytest
import jax.numpy as jnp
import flax.nnx as nnx
from easydel.modules.my_model import MyModelConfig, MyModelForCausalLM


def test_model_creation():
    config = MyModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = MyModelForCausalLM(config, rngs=nnx.Rngs(0))
    assert model is not None


def test_forward_pass():
    config = MyModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = MyModelForCausalLM(config, rngs=nnx.Rngs(0))

    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    outputs = model(input_ids=input_ids)

    assert outputs.logits.shape == (1, 10, 1000)


def test_with_labels():
    config = MyModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = MyModelForCausalLM(config, rngs=nnx.Rngs(0))

    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    labels = jnp.ones((1, 10), dtype=jnp.int32)
    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs.loss is not None
```

## Complete File Structure

```md
easydel/modules/my_model/
├── __init__.py
├── my_model_configuration.py
└── modeling_my_model.py
```

## Tips and Best Practices

1. **Start Simple**: Implement a basic version first, then add features
2. **Reuse Components**: Use existing layers from `easydel.layers`
3. **Follow Conventions**: Match the style of existing models
4. **Test Incrementally**: Test each component as you build it
5. **Document**: Add docstrings and type hints
6. **Handle Edge Cases**: Consider batch size 1, sequence length 1, etc.

## Next Steps

- [eLargeModel Guide](elarge_model.md) - Use your model with high-level training API
- [Customization Guide](customization.md) - Further customize your model
