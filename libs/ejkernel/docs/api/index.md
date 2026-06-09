# ejkernel API Documentation

Welcome to the ejkernel API documentation. ejkernel provides high-performance attention mechanisms and tensor operations optimized for JAX.

## Package Overview

ejkernel is organized into three main packages:

| Package | Purpose | Documentation |
|---------|---------|---------------|
| **ejkernel.modules** | High-level operations API | [modules.md](modules.md) |
| **ejkernel.types** | Masking and type utilities | [types.md](types.md) |
| **ejkernel.ops** | Kernel framework and config management | [ops.md](ops.md) |

---

## Quick Start

### Basic Attention

```python
from ejkernel.modules import flash_attention
import jax.numpy as jnp

# Create query, key, value tensors
q = jnp.ones((2, 128, 8, 64))  # (batch, seq_len, heads, head_dim)
k = jnp.ones((2, 128, 8, 64))
v = jnp.ones((2, 128, 8, 64))

# Run flash attention
output = flash_attention(q, k, v, causal=True)
```

### With Masking

```python
from ejkernel.types import MaskInfo

# Create mask from segment IDs
segment_ids = jnp.array([[1, 1, 2, 2, -1, -1]])  # -1 = padding
mask_info = MaskInfo.from_segments(segment_ids)

# Use with attention
output = flash_attention(q, k, v, mask_info=mask_info)
```

### Distributed Execution

```python
from jax.sharding import Mesh, PartitionSpec as P
import jax

devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), axis_names=('dp', 'tp'))

with mesh:
    output = flash_attention(
        q, k, v,
        mesh=mesh,
        in_specs=(P('dp', None, 'tp', None),) * 3,
        out_specs=P('dp', None, 'tp', None),
    )
```

---

## Documentation Guide

### For Most Users: Start with [modules.md](modules.md)

The `ejkernel.modules` package provides ready-to-use attention operations:

- `flash_attention` - Memory-efficient exact attention
- `ring_attention` - Distributed attention for long sequences
- `page_attention` - Paged KV-cache for inference
- And many more specialized variants

### For Complex Masking: See [types.md](types.md)

The `MaskInfo` class handles:

- Segment-based masking (packed sequences)
- Attention mask conversion
- Variable-length batching (cu_seqlens)
- Causal and sliding window patterns

### For Custom Kernels: See [ops.md](ops.md)

The `ejkernel.ops` framework enables:

- Custom kernel implementations
- Configuration autotuning
- Platform-specific optimizations
- Performance caching

---

## Common Patterns

### Packed Sequence Batching

```python
# Pack multiple sequences into one batch
segment_ids = jnp.array([
    [1, 1, 1, 2, 2, 3, 3, 3, 3, -1]  # 3 sequences, 1 padding
])
mask_info = MaskInfo.from_segments(segment_ids)

# Each sequence only attends within itself
output = flash_attention(q, k, v, mask_info=mask_info)
```

### Inference with KV Cache

```python
from ejkernel.modules import page_attention

output = page_attention(
    query,         # (batch, 1, heads, dim) - single token
    key_cache,     # (num_blocks, block_size, kv_heads, dim)
    value_cache,
    context_lens,  # (batch,) - current context length
    block_tables,  # (batch, max_blocks) - page table
)
```

### Long Sequence Training

```python
from ejkernel.modules import ring_attention
from jax.sharding import Mesh

# Distribute very long sequences across devices
mesh = Mesh(jax.devices(), axis_names=('sp',))

with mesh:
    output = ring_attention(
        q, k, v,
        axis_name='sp',
        causal=True,
    )
```

---

## Hardware Support

ejkernel automatically selects optimal implementations:

| Hardware | Platform | Key Features |
|----------|----------|--------------|
| NVIDIA GPU | Triton | Flash attention, tensor cores |
| Google TPU | Pallas | Matrix units, ring topology |
| CPU | XLA | Reference implementations |

Force a specific platform:

```python
output = flash_attention(q, k, v, platform="triton")  # GPU
output = flash_attention(q, k, v, platform="pallas")  # TPU
output = flash_attention(q, k, v, platform="xla")     # Fallback
```
