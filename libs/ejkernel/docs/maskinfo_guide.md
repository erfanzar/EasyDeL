# MaskInfo Guide: Comprehensive Attention Mask Management

## Overview

`MaskInfo` is a powerful dataclass in ejKernel that manages attention masks and their various representations for transformer models. It provides seamless conversion between different mask formats and supports both single-sequence and multi-sequence (packed) attention patterns.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Creating MaskInfo](#creating-maskinfo)
3. [Key Properties](#key-properties)
4. [Multi-Sequence Detection](#multi-sequence-detection)
5. [Common Use Cases](#common-use-cases)
6. [Advanced Features](#advanced-features)

---

## Core Concepts

### Segment IDs

Segment IDs are integer arrays that group tokens together:

- **Non-negative integers (0, 1, 2, ...)**: Tokens belonging to the same segment can attend to each other
- **-1**: Padding tokens (masked out)

**Examples:**

```python
# Single sequence with padding
[0, 0, 0, 0, -1, -1]  # 4 active tokens, 2 padding

# Multiple sequences (packed format)
[0, 0, 1, 1, 2, 2]    # 3 sequences: seg0(2 tokens), seg1(2), seg2(2)

# Mixed: active tokens and padding
[0, 0, 0, 1, 1, -1]   # 2 sequences: seg0(3 tokens), seg1(2), 1 padding
```

### Attention Masks

4D boolean/integer arrays of shape `(batch, heads, q_len, kv_len)`:

- **True/1**: Valid attention (query can attend to key)
- **False/0**: Masked (no attention allowed)

---

## Creating MaskInfo

### 1. From Segment IDs (Recommended)

The most common and efficient way to create MaskInfo:

```python
import jax.numpy as jnp
from ejkernel.types import MaskInfo

# Single sequence with padding
q_segment_ids = jnp.array([[0, 0, 0, 0, -1, -1]])
mask_info = MaskInfo.from_segments(q_segment_ids)

# Multi-sequence (packed format)
q_segment_ids = jnp.array([[0, 0, 1, 1, 2, 2]])
mask_info = MaskInfo.from_segments(q_segment_ids)

# Cross-attention: different Q and KV segments
q_segment_ids = jnp.array([[0, 0, 0, -1]])    # 3 query tokens
kv_segment_ids = jnp.array([[0, 0, 0, 0, 0]]) # 5 key-value tokens
mask_info = MaskInfo.from_segments(q_segment_ids, kv_segment_ids)
```

**When `kv_segment_ids` is None:**

- For self-attention: `kv_segment_ids = q_segment_ids`
- Tokens in the same segment can attend to each other
- This is the standard behavior for self-attention layers

### 2. From Attention Mask

When you already have an attention mask:

```python
# Create a causal attention mask
batch_size, seq_len = 2, 8
attention_mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len)))

# Convert to MaskInfo
mask_info = MaskInfo.from_attention_mask(attention_mask)

# Now you can access segment IDs and other representations
print(mask_info.q_segment_ids)    # Automatically computed
print(mask_info.kv_segment_ids)   # Automatically computed
```

### 3. From Cumulative Sequence Lengths (cu_seqlens)

Used in FlashAttention and packed sequence formats:

```python
# cu_seqlens defines sequence boundaries
# For 3 sequences of lengths [4, 3, 2]:
cu_seqlens_q = jnp.array([0, 4, 7, 9])   # Start positions + total length
cu_seqlens_kv = jnp.array([0, 5, 8, 10]) # Different KV lengths

mask_info = MaskInfo.from_cu_seqlens(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=cu_seqlens_kv,
    q_len=9,   # Total query tokens
    kv_len=10  # Total key-value tokens
)
```

---

## Key Properties

### Lazy Computation

MaskInfo uses lazy evaluation - representations are computed only when accessed:

```python
mask_info = MaskInfo.from_segments(q_segment_ids)

# These are computed on first access and cached:
attention_mask = mask_info.attention_mask      # 4D attention mask
q_segs = mask_info.q_segment_ids              # Query segment IDs
kv_segs = mask_info.kv_segment_ids            # Key-value segment IDs
cu_q = mask_info.cu_seqlens_q                 # Cumulative Q lengths
cu_kv = mask_info.cu_seqlens_kv               # Cumulative KV lengths
positions = mask_info.q_position_ids          # Position indices per segment
```

### Sequence Information

```python
# Get sequence lengths
q_lens = mask_info.q_lens          # Per-segment or per-batch lengths
kv_lens = mask_info.kv_lens        # Per-segment or per-batch lengths

# Shape information
batch = mask_info.batch_size       # Number of batches
q_len = mask_info.q_len           # Query sequence length
kv_len = mask_info.kv_len         # Key-value sequence length
shape = mask_info.shape           # (batch, q_len, kv_len)

# Check attention type
is_self = mask_info.is_self_attention()  # True if Q and KV are identical
```

---

## Multi-Sequence Detection

### Check if Packed Format

```python
# Single sequence: all valid tokens have segment ID 0
q_seg_single = jnp.array([[0, 0, 0, 0, -1, -1]])
mask_info = MaskInfo.from_segments(q_seg_single)
print(mask_info.is_multi_sequence)  # Array(False)

# Multi-sequence: tokens have different segment IDs
q_seg_multi = jnp.array([[0, 0, 1, 1, 2, 2]])
mask_info = MaskInfo.from_segments(q_seg_multi)
print(mask_info.is_multi_sequence)  # Array(True)
```

### Assert Single Sequence

Some operations don't support packed sequences. Use assertions to catch errors early:

```python
from ejkernel.types import MaskInfo
from ejkernel.modules.operations import attention

# This will pass
q_seg = jnp.array([[0, 0, 0, 0, -1, -1]])
mask_info = MaskInfo.from_segments(q_seg)
output = attention(query, key, value, mask_info=mask_info)

# This will fail with a clear error message
q_seg_packed = jnp.array([[0, 0, 1, 1, 2, 2]])
mask_info_packed = MaskInfo.from_segments(q_seg_packed)
try:
    output = attention(query, key, value, mask_info=mask_info_packed)
except ValueError as e:
    print(e)  # "attention does not support multi-sequence format..."
```

---

## Common Use Cases

### Use Case 1: Simple Self-Attention with Padding

```python
import jax.numpy as jnp
from ejkernel.types import MaskInfo
from ejkernel.modules.operations import flash_attention

# Batch of 2 sequences with different lengths (padded)
batch_size, max_seq_len = 2, 128
q_segment_ids = jnp.array([
    [0]*100 + [-1]*28,  # First sequence: 100 tokens
    [0]*80 + [-1]*48    # Second sequence: 80 tokens
], dtype=jnp.int32)

# Create mask info
mask_info = MaskInfo.from_segments(q_segment_ids)

# Use with attention
output = flash_attention(
    query, key, value,
    mask_info=mask_info,
    causal=True
)
```

### Use Case 2: Packed Multi-Sequence Attention (FlashAttention)

```python
# Pack 3 sequences into a single batch dimension
# Sequences: lengths [40, 35, 25]
q_segment_ids = jnp.array([
    [0]*40 + [1]*35 + [2]*25 + [-1]*156  # Total: 256 tokens
], dtype=jnp.int32)

mask_info = MaskInfo.from_segments(q_segment_ids)

# Verify it's multi-sequence
print(mask_info.is_multi_sequence)  # True

# Get cumulative lengths for FlashAttention
cu_seqlens_q = mask_info.cu_seqlens_q  # [0, 40, 75, 100]

# Use with flash attention (supports packed sequences)
output = flash_attention(
    query, key, value,
    mask_info=mask_info
)
```

### Use Case 3: Cross-Attention with Different Q/KV Lengths

```python
# Decoder queries attend to encoder keys/values
encoder_length = 512
decoder_length = 128

# All encoder tokens are valid (no padding)
kv_segment_ids = jnp.array([[0] * encoder_length], dtype=jnp.int32)

# Decoder has some padding
q_segment_ids = jnp.array([[0] * 100 + [-1] * 28], dtype=jnp.int32)

mask_info = MaskInfo.from_segments(
    q_segment_ids=q_segment_ids,
    kv_segment_ids=kv_segment_ids
)

# Cross-attention output
output = flash_attention(query, key, value, mask_info=mask_info)
```

### Use Case 4: Applying Causal Masking

```python
# Start with basic segment IDs
q_segment_ids = jnp.array([[0, 0, 0, 0, 0]], dtype=jnp.int32)
mask_info = MaskInfo.from_segments(q_segment_ids)

# Apply causal mask (queries can only attend to previous positions)
causal_mask_info = mask_info.apply_causal()

# The attention mask now has causal structure
print(causal_mask_info.attention_mask)
# [[[[1, 0, 0, 0, 0],
#    [1, 1, 0, 0, 0],
#    [1, 1, 1, 0, 0],
#    [1, 1, 1, 1, 0],
#    [1, 1, 1, 1, 1]]]]
```

### Use Case 5: Sliding Window Attention

```python
# Apply sliding window: each token attends to ±2 positions
q_segment_ids = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
mask_info = MaskInfo.from_segments(q_segment_ids)

# Window of 2 on each side
windowed_mask_info = mask_info.apply_sliding_window(window_size=(2, 2))

# Use with attention
output = flash_attention(
    query, key, value,
    mask_info=windowed_mask_info
)
```

### Use Case 6: JIT-Compatible Usage

```python
import jax

@jax.jit
def attention_forward(q_segment_ids, query, key, value):
    # Create mask info inside JIT
    mask_info = MaskInfo.from_segments(q_segment_ids)

    # Check if multi-sequence (returns JAX array, JIT-compatible)
    is_multi = mask_info.is_multi_sequence

    # Get position IDs (resets at segment boundaries)
    position_ids = mask_info.q_position_ids

    # Get cumulative lengths
    cu_seqlens = mask_info.cu_seqlens_q

    return cu_seqlens, position_ids, is_multi

# Call the JIT-compiled function
result = attention_forward(q_segment_ids, query, key, value)
```

### Use Case 7: Dynamic Sequence Lengths

```python
# Handle variable-length sequences at runtime
def process_batch(sequences_list):
    """Process a list of sequences with different lengths."""
    # Get max length
    max_len = max(len(seq) for seq in sequences_list)
    batch_size = len(sequences_list)

    # Create segment IDs with padding
    q_segment_ids = jnp.full((batch_size, max_len), -1, dtype=jnp.int32)

    for i, seq in enumerate(sequences_list):
        seq_len = len(seq)
        q_segment_ids = q_segment_ids.at[i, :seq_len].set(0)

    # Create mask info
    mask_info = MaskInfo.from_segments(q_segment_ids)

    # Get actual sequence lengths
    q_lens = mask_info.q_lens  # [len(seq1), len(seq2), ...]

    return mask_info, q_lens

# Example usage
sequences = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]]
mask_info, lengths = process_batch(sequences)
print(lengths)  # [5, 3, 4]
```

---

## Advanced Features

### Position IDs with Segment Resets

```python
# Position IDs reset at each segment boundary
q_segment_ids = jnp.array([[0, 0, 0, 1, 1, 2, 2, 2, -1]], dtype=jnp.int32)
mask_info = MaskInfo.from_segments(q_segment_ids)

position_ids = mask_info.q_position_ids
print(position_ids)
# [[0, 1, 2, 0, 1, 0, 1, 2, -1]]
#   ^seg0^  ^seg1^  ^seg2^  pad
```

### Custom Bias Addition

```python
# Add custom attention bias
mask_info = MaskInfo.from_segments(q_segment_ids)

# Get base bias from mask (0 for valid, -inf for masked)
bias = mask_info.bias

# Add positional bias
positional_bias = compute_positional_bias(seq_len)
combined_bias = bias + positional_bias

# Use with attention
output = attention(query, key, value, bias=combined_bias)
```

### Visualization

```python
# Visualize attention patterns
mask_info = MaskInfo.from_segments(q_segment_ids)
mask_info.visualize()  # Shows the attention mask structure
```

### Distributed Training with Sharding

```python
from jax.sharding import Mesh, PartitionSpec

# Define sharding configuration
mask_info = MaskInfo.from_segments(
    q_segment_ids,
    batch_axis_name="dp",           # Data parallel
    qheads_axis_name="tp",          # Tensor parallel
    sequence_axis_name="sp"         # Sequence parallel
)

# Get sharding specs
mesh = Mesh(devices, ("dp", "tp", "sp"))
shardings = mask_info.get_shardings(
    include_positions=True,
    mesh=mesh
)

# Use with distributed attention
output = attention(query, key, value, mask_info=mask_info, mesh=mesh)
```

---

## Best Practices

### 1. **Use Segment IDs When Possible**

Segment IDs are more compact and efficient than full attention masks:

```python
# Preferred ✓
mask_info = MaskInfo.from_segments(q_segment_ids)

# Less efficient ✗
attention_mask = create_large_4d_mask()  # Memory intensive
mask_info = MaskInfo.from_attention_mask(attention_mask)
```

### 2. **Check Multi-Sequence Before Operations**

```python
if mask_info.is_multi_sequence:
    # Use flash_attention (supports packed sequences)
    output = flash_attention(query, key, value, mask_info=mask_info)
else:
    # Can use any attention variant
    output = attention(query, key, value, mask_info=mask_info)
```

### 3. **Leverage Lazy Evaluation**

Don't compute what you don't need:

```python
# Only compute what you use
mask_info = MaskInfo.from_segments(q_segment_ids)

# If you only need cu_seqlens, don't access attention_mask
cu_seqlens = mask_info.cu_seqlens_q  # Efficient

# Avoid unnecessary conversions
# attention_mask = mask_info.attention_mask  # Skip if not needed
```

### 4. **Use Assertions for Debugging**

```python
# Catch errors early in development
mask_info = MaskInfo.from_segments(q_segment_ids)

# Assert expectations
assert mask_info.is_self_attention()   # For self-attention layers
```

---

## Summary

`MaskInfo` provides a unified interface for managing attention masks in ejKernel:

- ✅ **Multiple creation methods**: segment IDs, attention masks, cu_seqlens
- ✅ **Automatic conversions**: seamlessly convert between representations
- ✅ **Lazy evaluation**: compute only what you need
- ✅ **Multi-sequence support**: handle both single and packed sequences
- ✅ **JIT-compatible**: works with `jax.jit` compilation
- ✅ **Distributed ready**: built-in sharding support

For more examples and API details, see the [API documentation](api_docs/types/mask.md).
