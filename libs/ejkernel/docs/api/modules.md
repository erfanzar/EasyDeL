# ejkernel.modules - High-Level Operations API

## Overview

The `ejkernel.modules` package provides high-level, user-friendly interfaces for attention mechanisms and other tensor operations. These modules abstract away the complexity of kernel selection, platform detection, and configuration management, allowing you to focus on your model architecture.

## Key Concepts

### Module vs Function API

Each operation is available in two forms:

1. **Module Class** (e.g., `FlashAttention`): For advanced use cases where you need fine-grained control over execution, configuration, or integration with the executor system.

2. **Function API** (e.g., `flash_attention`): For simple, direct usage with sensible defaults and automatic optimization.

```python
# Function API - simple and direct
from ejkernel.modules import flash_attention

output = flash_attention(query, key, value, causal=True)

# Module API - for advanced control
from ejkernel.modules import FlashAttention
from ejkernel.ops import Executor, ConfigSelectorChain, ConfigCache

executor = Executor(ConfigSelectorChain(cache=ConfigCache()))
attn_module = FlashAttention()
output = executor(attn_module, query=query, key=key, value=value, causal=True)
```

### Automatic Platform Selection

All modules automatically detect your hardware (GPU, TPU, or CPU) and select the optimal kernel implementation:

- **GPU**: Triton kernels (if available) or XLA fallback
- **TPU**: Pallas kernels optimized for TPU matrix units
- **CPU**: XLA-based implementation

You can override this with the `platform` parameter:

```python
output = flash_attention(query, key, value, platform="xla")  # Force XLA
output = flash_attention(query, key, value, platform="triton")  # Force Triton
```

Distributed matmul note (`all_gather_matmul`, `reduce_scatter_matmul`):

- These operations intentionally do **not** perform runtime fallback between distributed backends.
- If the chosen backend cannot execute the provided shape/constraints, the call fails.
- Choose `platform`/`cfg.platform` explicitly for production deployments.

### Configuration and Autotuning

Modules support automatic performance tuning. On first invocation with new input shapes, the system benchmarks multiple configurations and caches the optimal one:

```python
# First call: benchmarks and caches optimal config
output = flash_attention(query, key, value)

# Subsequent calls with same shapes: uses cached config
output = flash_attention(query, key, value)  # Fast!
```

---

## Attention Operations

### FlashAttention

Memory-efficient exact attention with O(N) memory complexity instead of O(N²).

```python
from ejkernel.modules import flash_attention

# Basic usage
output = flash_attention(query, key, value)

# With causal masking (for autoregressive models)
output = flash_attention(query, key, value, causal=True)

# With sliding window (local attention)
output = flash_attention(query, key, value, sliding_window=256)

# With attention bias
output = flash_attention(query, key, value, bias=attention_bias)

# Variable-length sequences (packed batching)
output = flash_attention(
    query, key, value,
    cum_seqlens_q=cu_seqlens_q,  # [0, 128, 300, 512]
    cum_seqlens_k=cu_seqlens_k,
)

# With MaskInfo for complex masking
from ejkernel.types import MaskInfo
mask_info = MaskInfo.from_segments(segment_ids)
output = flash_attention(query, key, value, mask_info=mask_info)
```

**Parameters:**

- `query`: Query tensor `[batch, seq_len_q, num_heads, head_dim]`
- `key`: Key tensor `[batch, seq_len_k, num_kv_heads, head_dim]`
- `value`: Value tensor `[batch, seq_len_k, num_kv_heads, head_dim]`
- `causal`: Apply causal masking (default: `False`)
- `softmax_scale`: Scaling factor (default: `1/sqrt(head_dim)`)
- `sliding_window`: Window size for local attention `int` or `(left, right)`
- `dropout_prob`: Dropout probability (default: `0.0`)
- `bias`: Optional attention bias tensor
- `mask_info`: Optional `MaskInfo` for complex masking patterns
- `cum_seqlens_q`, `cum_seqlens_k`: Cumulative sequence lengths for variable-length batching
- `logits_soft_cap`: Soft cap for attention logits (Gemma-2 style)

---

### RingAttention

Distributed attention using ring communication topology for ultra-long sequences across multiple devices.

```python
from ejkernel.modules import ring_attention
import jax

# Setup device mesh for distributed execution
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, axis_names=('sp',))

# Execute ring attention
with mesh:
    output = ring_attention(
        query, key, value,
        axis_name='sp',  # Required: name of sequence-parallel axis
        causal=True,
    )
```

**When to use:**

- Sequences too long to fit in single device memory
- Multi-device training with sequence parallelism
- Context lengths > 128K tokens

**Parameters:**

- `axis_name`: Name of the axis for collective operations (required)
- `chunk_size`: Optional chunk size for chunked causal attention
- `sliding_window`: Window size for local attention
- `softmax_aux`: Attention sink logits for long-context stability

---

### NativeSparseAttention

Sparse attention with explicit block index specification for maximum control over sparsity patterns.

```python
from ejkernel.modules import native_sparse_attention

# With explicit block indices
output = native_sparse_attention(
    query, key, value,
    block_indices=block_indices,  # Which key blocks each query attends to
    block_counts=16,  # Number of key blocks per query
)

# With variable-length sequences
output = native_sparse_attention(
    query, key, value,
    cu_seqlens=cu_seqlens,
    block_indices=block_indices,
)
```

**Parameters:**

- `block_indices`: Which key blocks each query block attends to
- `block_counts`: Number of key blocks per query block
- `cu_seqlens`: Cumulative sequence lengths for variable batching

---

### PageAttention

Paged KV-cache attention optimized for inference serving with dynamic batching.

```python
from ejkernel.modules import page_attention

output = page_attention(
    query,           # [batch, 1, num_heads, head_dim] for decode
    key_cache,       # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache,     # [num_blocks, block_size, num_kv_heads, head_dim]
    context_lens,    # [batch] - length of context for each sequence
    block_tables,    # [batch, max_blocks] - page table mapping
)
```

**When to use:**

- Inference serving with continuous batching
- Memory-efficient KV-cache management
- vLLM-style serving systems

---

### RaggedPageAttention (v2 and v3)

Advanced paged attention supporting variable-length sequences without padding.

```python
from ejkernel.modules import ragged_page_attention_v3

output = ragged_page_attention_v3(
    query,
    key_pages,
    value_pages,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    page_indices=page_indices,
)
```

**v3 improvements over v2:**

- Better memory efficiency
- Improved softmax numerics
- Support for attention sinks

---

### ScaledDotProductAttention

Standard scaled dot-product attention using XLA primitives. Good baseline and fallback.

```python
from ejkernel.modules import scaled_dot_product_attention

output = scaled_dot_product_attention(
    query, key, value,
    attention_mask=mask,
    causal=True,
)
```

---

### BlockSparseAttention

Block-sparse attention with configurable sparsity patterns.

```python
from ejkernel.modules import blocksparse_attention

output = blocksparse_attention(
    query, key, value,
    block_mask=block_mask,  # Which blocks can attend to which
    block_size=64,
)
```

---

### GLAttention (Gated Linear Attention)

Linear attention with gating for O(N) complexity.

```python
from ejkernel.modules import gla_attention

output = gla_attention(
    query, key, value,
    gate=gate_values,  # Gating tensor
)
```

---

### LightningAttention

Layer-aware attention with decay factors for efficient long-range modeling.

```python
from ejkernel.modules import lightning_attention

output = lightning_attention(
    query, key, value,
    decay=decay_factor,
)
```

---

### RecurrentAttention

Stateful recurrent attention for streaming inference.

```python
from ejkernel.modules import recurrent_attention

output, new_state = recurrent_attention(
    query, key, value,
    state=previous_state,  # Carries information from previous steps
)
```

---

## Other Operations

### GroupedMatmul

Efficient batched matrix multiplication with variable group sizes.

```python
from ejkernel.modules import grouped_matmul

output = grouped_matmul(
    inputs,     # [total_tokens, hidden]
    weights,    # [num_experts, hidden, output]
    group_ids,  # [total_tokens] - which expert each token uses
)
```

**When to use:**

- Mixture of Experts (MoE) layers
- Token routing to different experts
- Batched operations with variable sizes

---

### MeanPooling

Sequence mean pooling with variable-length support.

```python
from ejkernel.modules import mean_pooling

# With attention mask
pooled = mean_pooling(
    hidden_states,  # [batch, seq_len, hidden]
    attention_mask,  # [batch, seq_len]
)

# Result: [batch, hidden] - mean of valid tokens per sequence
```

---

## Distributed Execution

All operations support distributed execution via JAX's `shard_map`:

```python
from ejkernel.modules import flash_attention
from jax.sharding import Mesh, PartitionSpec as P
import jax

# Setup mesh
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), axis_names=('dp', 'tp'))

# Distributed attention with sharding specs
output = flash_attention(
    query, key, value,
    mesh=mesh,
    in_specs=(
        P('dp', None, 'tp', None),  # query sharding
        P('dp', None, 'tp', None),  # key sharding
        P('dp', None, 'tp', None),  # value sharding
    ),
    out_specs=P('dp', None, 'tp', None),
)
```

---

## Configuration Classes

Each operation has a corresponding configuration class:

```python
from ejkernel.modules import FlashAttentionConfig
from ejkernel.ops import FwdParams, BwdParams

# Create custom configuration
config = FlashAttentionConfig(
    fwd_params=FwdParams(
        q_blocksize=128,
        kv_blocksize=256,
        num_warps=4,
        num_stages=2,
    ),
    bwd_params=BwdParams(
        q_blocksize=64,
        kv_blocksize=128,
    ),
    platform="triton",
    backend="gpu",
)

# Use custom config
output = flash_attention(query, key, value, cfg=config)
```

**Common configuration parameters:**

- `platform`: `"triton"`, `"pallas"`, `"xla"`, `"cuda"`, or `"auto"`
- `backend`: `"gpu"`, `"tpu"`, or `"any"`
- `fwd_params`: Forward pass block sizes and tuning parameters
- `bwd_params`: Backward pass parameters

---

## Best Practices

### 1. Let autotuning work for you

First calls may be slower due to autotuning, but subsequent calls with the same shapes will be fast.

### 2. Use MaskInfo for complex masking

Instead of manually creating attention masks, use `MaskInfo` for segment-aware attention:

```python
from ejkernel.types import MaskInfo

# From segment IDs (packed sequences)
mask_info = MaskInfo.from_segments(segment_ids)

# From attention mask
mask_info = MaskInfo.from_attention_mask(attention_mask)

# Use with attention
output = flash_attention(query, key, value, mask_info=mask_info)
```

### 3. Choose the right attention variant

| Use Case                    | Recommended                                          |
| --------------------------- | ---------------------------------------------------- |
| Standard training           | `flash_attention`                                    |
| Inference serving           | `page_attention` or `ragged_page_attention_v3`       |
| Very long sequences (>128K) | `ring_attention`                                     |
| Sparse patterns             | `native_sparse_attention` or `blocksparse_attention` |
| Streaming inference         | `recurrent_attention`                                |
| MoE layers                  | `grouped_matmul`                                     |

### 4. Profile before optimizing

Use JAX profiling to identify bottlenecks before manually tuning configurations.
