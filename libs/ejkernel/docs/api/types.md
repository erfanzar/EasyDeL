# ejkernel.types - Masking and Type Utilities

## Overview

The `ejkernel.types` package provides data structures and utilities for managing attention masks, segment IDs, and sequence metadata. The centerpiece is `MaskInfo`, a unified container that handles various mask representations and provides seamless conversion between them.

## Core Concept: Mask Representations

Attention mechanisms use masks to control which positions can attend to which. ejkernel supports multiple representations:

| Representation | Shape | Description |
|----------------|-------|-------------|
| **Attention Mask** | `(B, H, Q, K)` or `(B, Q, K)` | Pairwise mask: True = can attend |
| **Segment IDs** | `(B, Q)` + `(B, K)` | Group tokens: same ID = can attend |
| **Cumulative Lengths** | `(B*2,)` | Start/end positions for variable-length batching |

`MaskInfo` automatically converts between these representations as needed.

---

## MaskInfo

The `MaskInfo` dataclass is a smart container that holds mask information and lazily computes derived representations on demand.

### Creating MaskInfo

#### From Segment IDs

Segment IDs group tokens that can attend to each other. Use non-negative integers for valid tokens and -1 for padding.

```python
from ejkernel.types import MaskInfo
import jax.numpy as jnp

# Single segment (all tokens attend to all)
segment_ids = jnp.array([[0, 0, 0, 0, -1, -1]])  # 4 valid tokens, 2 padding
mask_info = MaskInfo.from_segments(segment_ids)

# Multiple segments (packed sequences)
segment_ids = jnp.array([[1, 1, 2, 2, 3, 3, 3, -1]])  # 3 sequences packed
mask_info = MaskInfo.from_segments(segment_ids)

# Cross-attention (different Q and KV)
q_segment_ids = jnp.array([[1, 1, 2, 2]])  # query segments
kv_segment_ids = jnp.array([[1, 1, 1, 2, 2, 2]])  # key-value segments
mask_info = MaskInfo.from_segments(q_segment_ids, kv_segment_ids)
```

#### From Attention Mask

For existing attention masks in various shapes:

```python
# 2D padding mask (batch, seqlen) - common for transformers
padding_mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])  # 1=valid, 0=padding
mask_info = MaskInfo.from_attention_mask(padding_mask)

# 3D pairwise mask (batch, qlen, kvlen)
pairwise_mask = jnp.tril(jnp.ones((1, 8, 8), dtype=jnp.bool_))  # causal
mask_info = MaskInfo.from_attention_mask(pairwise_mask)

# 4D multi-head mask (batch, heads, qlen, kvlen)
mh_mask = jnp.ones((2, 8, 64, 64), dtype=jnp.bool_)
mask_info = MaskInfo.from_attention_mask(mh_mask)

# PyTorch-style inverted mask (True = masked out)
inverted_mask = ~jnp.tril(jnp.ones((1, 8, 8), dtype=jnp.bool_))
mask_info = MaskInfo.from_attention_mask(inverted_mask, mask_is_valid=False)
```

#### From Cumulative Sequence Lengths

For variable-length batching (FlashAttention-style):

```python
# cu_seqlens format: [start_0, end_0, start_1, end_1, ...]
# Example: 3 sequences with lengths [3, 5, 4] packed to max_len=5
cu_seqlens_q = jnp.array([0, 3, 0, 5, 0, 4])  # start/end pairs
mask_info = MaskInfo.from_cu_seqlens(cu_seqlens_q, max_q_len=5)

# Cross-attention with different Q/KV lengths
cu_seqlens_kv = jnp.array([0, 6, 0, 8, 0, 7])
mask_info = MaskInfo.from_cu_seqlens(
    cu_seqlens_q, max_q_len=5,
    cu_seqlens_kv=cu_seqlens_kv, max_kv_len=8
)
```

#### Random Mask (for Testing)

```python
# Generate random attention pattern
mask_info = MaskInfo.from_random(
    batch_size=2,
    q_len=128,
    kv_len=256,  # None for self-attention
    sparsity=0.5,  # 50% masked out
    seed=42
)
```

#### Dynamic Initialization

For model implementations where mask format varies:

```python
# From input_ids (creates all-ones mask)
mask_info = MaskInfo.dynamic_init(input_ids=input_ids)

# From attention_mask (handles 2D/3D/4D)
mask_info = MaskInfo.dynamic_init(attention_mask=attention_mask)

# Pass through existing MaskInfo
mask_info = MaskInfo.dynamic_init(mask_info=existing_mask_info)
```

---

### Accessing Mask Properties

Properties are computed lazily on first access:

```python
mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2, -1]]))

# Shape information
mask_info.batch_size  # 1
mask_info.q_len       # 5
mask_info.kv_len      # 5
mask_info.shape       # (1, 5, 5)

# Mask representations
mask_info.attention_mask   # (B, 1, Q, K) boolean mask
mask_info.q_segment_ids    # (B, Q) segment IDs
mask_info.kv_segment_ids   # (B, K) segment IDs
mask_info.cu_seqlens_q     # Cumulative lengths for Q
mask_info.cu_seqlens_kv    # Cumulative lengths for KV

# Derived values
mask_info.q_lens           # Per-batch query lengths
mask_info.kv_lens          # Per-batch key-value lengths
mask_info.q_attention_mask # 2D query validity mask
mask_info.q_position_ids   # Position IDs computed from mask

# Attention bias (for score computation)
mask_info.bias             # 0.0 for valid, -inf for masked
mask_info.create_bias(dtype=jnp.float16)  # Custom dtype
```

---

### Applying Mask Transformations

MaskInfo provides chainable methods for modifying attention patterns:

#### Causal Masking

```python
# Standard causal (each position attends to itself and earlier)
causal_mask = mask_info.apply_causal()

# With offset (allow attending to future positions)
causal_mask = mask_info.apply_causal(offset=5)

# Per-batch offsets
offsets = jnp.array([0, 2, 4])  # different offset per batch
causal_mask = mask_info.apply_causal(offset=offsets)
```

#### Sliding Window

```python
# Symmetric window (attend to 256 positions left and right)
windowed = mask_info.apply_sliding_window(256)

# Asymmetric window (512 left, 0 right = causal local)
windowed = mask_info.apply_sliding_window((512, 0))

# Decode mode: slice mask for single-token generation
decode_mask = mask_info.apply_sliding_window(
    256,
    mode="decode",
    index=current_position
)

# Prefill mode: slice to last N positions
prefill_mask = mask_info.apply_sliding_window(
    256,
    mode="prefill"
)
```

#### Chunked Attention

```python
# Split attention into fixed-size chunks with causal ordering
chunked = mask_info.apply_chunked(chunk_size=128)
# Each query only attends within its chunk, with causal constraint
```

#### Token Type IDs

For segment-aware attention (e.g., BERT-style):

```python
# Same token types can attend to each other
token_types = jnp.array([[1, 1, 2, 2, 0, 0]])  # 0 = padding
typed_mask = mask_info.apply_token_type_ids(token_types)

# Different modes for combining with existing mask
typed_mask = mask_info.apply_token_type_ids(
    token_types,
    combine="intersect",  # AND with existing mask
    zero_policy="both"    # treat 0 as padding on both Q and KV
)
```

#### KV Length Limiting

For inference with variable context:

```python
# Mask out KV positions beyond per-example lengths
kv_lengths = jnp.array([100, 150, 80])  # valid KV length per batch
limited = mask_info.apply_kv_lengths(kv_lengths)

# With query windowing
limited = mask_info.apply_kv_lengths(
    kv_lengths,
    q_len=1,           # keep only 1 query position
    end_index=current_idx,  # window ends at this position
    sliding_window=256      # limit KV to last 256 positions
)
```

---

### Conversion and Materialization

```python
# Force computation of specific representations
mask_info = mask_info.materialize_attention_mask()
mask_info = mask_info.materialize_segment_ids()

# Convert attention mask dtype
mask_info = mask_info.to_dtype(jnp.float16)

# Get Q/KV masks separately
q_mask, kv_mask, attention_mask = mask_info.get_qkv_masks()

# Compute or retrieve representations
q_pos, kv_pos = mask_info.get_or_compute_positions()
cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens()
```

---

### Distributed Sharding

MaskInfo integrates with JAX's sharding system:

```python
from jax.sharding import Mesh

devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), axis_names=('dp', 'tp'))

# Get sharding specs for all mask components
shardings = mask_info.get_shardings(mesh=mesh)

# With sequence parallelism
shardings = mask_info.get_shardings(
    mesh=mesh,
    sequence_parallel=True
)

# Customize axis names
mask_info = MaskInfo.from_segments(
    segment_ids,
    batch_axis_name=('dp', 'fsdp'),
    sequence_axis_name='sp',
    qheads_axis_name='tp'
)
```

---

### Visualization

Debug attention patterns with ASCII/Unicode visualization:

```python
mask_info.visualize()  # Print to console

mask_info.visualize(
    batch=0,
    head=0,
    block_size=32,      # Aggregate into blocks
    charset="unicode",  # or "ascii"
    show_segments=True, # Show segment IDs
    return_str=True     # Return string instead of printing
)
```

Output:

```md
Attention mask (batch=0, head=0) block=(32x32) mask_shape=(1, 1, 128, 128)
              01 01 02 02
  ============
01 ||████..  ||
01 ||████..  ||
02 ||    ████||
02 ||    ████||
  ============
Legend mask: full='██'/##, partial='░░'/.., empty='  '
Legend seg: left=Q block ID, top=KV block ID, PAD=-1, MIX=??
```

---

### JAX Pytree Support

MaskInfo is registered as a JAX pytree, so it works seamlessly with JAX transformations:

```python
# Works with jit
@jax.jit
def process(mask_info, x):
    return x * mask_info.q_attention_mask[..., None]

# Works with vmap
vmapped = jax.vmap(process)

# Works with grad (for mask-dependent operations)
```

---

## Utility Functions

### Debug Mode

Enable verbose tracing for debugging:

```python
from ejkernel.types import set_debug_mode, get_debug_mode

set_debug_mode(True)  # Enable debug output
# ... operations will print trace info ...
set_debug_mode(False)  # Disable

if get_debug_mode():
    print("Debug mode is on")
```

---

## Best Practices

### 1. Choose the Right Constructor

| Situation | Constructor |
|-----------|-------------|
| Packed sequences with segment labels | `from_segments()` |
| Existing attention mask | `from_attention_mask()` |
| Variable-length batching | `from_cu_seqlens()` |
| Model forward with flexible input | `dynamic_init()` |
| Testing | `from_random()` |

### 2. Lazy Computation

MaskInfo computes representations lazily. Access only what you need:

```python
# Good: Only segment IDs computed
segment_q, segment_kv = mask_info.get_or_compute_segment_ids()

# Bad: Forces full mask materialization when not needed
full_mask = mask_info.attention_mask  # Avoid if not needed
```

### 3. Chain Transformations

```python
# Chain methods for complex patterns
mask_info = (
    MaskInfo.from_segments(segment_ids)
    .apply_causal()
    .apply_sliding_window(256)
)
```

### 4. Reuse MaskInfo

MaskInfo caches computed values. Reuse instances when possible:

```python
# Good: Reuse mask_info
for layer in layers:
    output = layer(x, mask_info=mask_info)

# Bad: Recreate each time
for layer in layers:
    output = layer(x, mask_info=MaskInfo.from_segments(ids))
```
