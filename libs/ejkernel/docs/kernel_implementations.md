# Kernel Implementations Analysis

## Overview

The kernel implementations in ejKernel are organized by backend technology, with each backend optimized for specific hardware platforms. This modular organization allows for platform-specific optimizations while maintaining a consistent API across all implementations.

## Backend Organization

```md
ejkernel/kernels/
├── _triton/          # Triton GPU kernels (NVIDIA/AMD)
├── _cute/            # CUTLASS CuTe DSL kernels (NVIDIA)
├── _pallas/          # Pallas kernels (TPU/GPU)
│   ├── tpu/         # TPU-specific implementations
│   └── gpu/         # GPU Pallas implementations
├── _xla/            # XLA-based implementations (universal)
└── _cuda/           # Direct CUDA implementations
```

## Implementation Comparison

| Backend | Target Hardware | Programming Model | Performance       | Features                           |
| ------- | --------------- | ----------------- | ----------------- | ---------------------------------- |
| Triton  | NVIDIA/AMD GPU  | Block-based SIMD  | Best on GPU       | Full feature set                   |
| CUTE    | NVIDIA GPU      | CUTLASS CuTe DSL  | Correctness-first | Quantized matmul + flash attention |
| Pallas  | TPU/GPU         | Block-based       | Best on TPU       | Limited features                   |
| XLA     | All devices     | Functional        | Good everywhere   | Basic features                     |
| CUDA    | NVIDIA GPU      | Direct CUDA       | Maximum control   | Under development                  |

## Flash Attention Implementations

Flash Attention is the most complex and well-developed kernel, showcasing different implementation strategies across backends.

### Triton Implementation

**Location**: `ejkernel/kernels/_triton/flash_attention/_interface.py`

```python
@kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "..."] | None = None,
    bias: Float[Array, "..."] | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    cum_seqlens_q: Int[Array, "batch + 1"] | None = None,
    cum_seqlens_k: Int[Array, "batch + 1"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """
    Triton-based Flash Attention v2 implementation.

    Key features:
    - Memory-efficient O(N) memory complexity
    - Online softmax computation
    - Support for variable-length sequences (cu_seqlens)
    - Logit soft capping (Gemma-2 style)
    - Custom VJP for efficient gradients
    """
```

#### Custom VJP Structure

```python
@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9, 10, 13, 14))
@ejit(static_argnums=(5, 6, 7, 9, 10, 13, 14))
def flash_attention_call(query, key, value, ...):
    """Entry point with custom gradient"""
    return _fwd_attention_kernel_call(...)[0]

def _jax_fwd_attention_call(...):
    """Forward pass: computes output and saves residuals"""
    out, lse = _fwd_attention_kernel_call(...)
    residuals = (query, key, value, bias, attention_mask,
                out, lse, dropout_seed, cum_seqlens_q, cum_seqlens_k)
    return out, residuals

def _jax_bwd_attention_call(softmax_scale, dropout_prob, causal,
                            fwd_params, bwd_params, sliding_window,
                            logits_soft_cap, residuals, dO):
    """Backward pass: computes gradients"""
    query, key, value, bias, attention_mask, out, lse, ... = residuals
    dq, dk, dv = _bwd_attention_kernel_call(...)
    return dq, dk, dv, None, None, None, None, None, None

flash_attention_call.defvjp(_jax_fwd_attention_call, _jax_bwd_attention_call)
```

#### Triton Kernel Structure

```python
def _fwd_attention_kernel_call(...):
    """Forward kernel orchestration"""
    # 1. Validate inputs
    _validate_inputs(query, key, value)

    # 2. Setup configuration
    if fwd_params is None:
        fwd_params = _get_default_fwd_params(query, key, value)

    # 3. Prepare memory layout
    block_q = fwd_params.q_blocksize
    block_k = fwd_params.kv_blocksize
    grid = calculate_grid(batch_size, num_heads, seq_len_q, block_q)

    # 4. Launch Triton kernel
    output, lse = triton_flash_attention_fwd[grid](
        query, key, value,
        bias, attention_mask,
        block_q, block_k,
        softmax_scale, causal,
        dropout_prob, dropout_seed,
        sliding_window, logits_soft_cap
    )

    return output, lse

@triton.jit
def triton_flash_attention_fwd(...):
    """Actual Triton kernel implementation"""
    # Triton block-based implementation
    # Uses shared memory for efficient data access
    # Implements online softmax algorithm
    pass
```

### XLA Implementation

**Location**: `ejkernel/kernels/_xla/flash_attention/_interface.py`

```python
@kernel_registry.register("flash_attention", Platform.XLA, Backend.ANY)
def flash_attention(query, key, value, ...):
    """XLA-based implementation using JAX primitives"""
```

#### Function Caching Strategy

```python
_CORE_FUNC_CACHE = {}

def _make_core_func(precision_code, logits_dtype_code, chunk_size_q,
                   chunk_size_k, normalize_output, causal, dropout_prob):
    """
    Creates specialized function for static parameters.
    Caching prevents recompilation for common configurations.
    """
    @jax.custom_vjp
    def _flash_attention_core_specialized(query, key, value, ...):
        return _flash_attention_fwd(...)

    def _fwd(query, key, value, ...):
        y = _flash_attention_fwd(...)
        ctx = (bias, attention_mask, softmax_aux, sliding_window,
              softmax_scale, logits_soft_cap, chunk_size_q, chunk_size_k,
              normalize_output, query, key, value, causal,
              dropout_prob, dropout_key)
        return y, ctx

    def _bwd(ctx, g):
        # Unpack context
        bias, attention_mask, ... = ctx
        # Compute gradients
        dq, dk, dv = _flash_attention_bwd(...)
        return (dq, dk, dv, None, None, None, None, None, None, None)

    _flash_attention_core_specialized.defvjp(_fwd, _bwd)
    return _flash_attention_core_specialized

def flash_attention(...):
    """Main entry point"""
    # Create cache key from static parameters
    cache_key = (precision_code, logits_dtype_code, chunk_size_q,
                chunk_size_k, normalize_output, causal, dropout_prob)

    # Get or create specialized function
    if cache_key not in _CORE_FUNC_CACHE:
        _CORE_FUNC_CACHE[cache_key] = _make_core_func(*cache_key)

    # Execute specialized function
    return _CORE_FUNC_CACHE[cache_key](query, key, value, ...)
```

#### XLA Algorithm Implementation

```python
def _flash_attention_fwd(query, key, value, ...):
    """
    XLA Flash Attention using JAX operations.

    Algorithm:
    1. Chunk Q, K, V into blocks
    2. For each Q block:
       a. Compute attention with all K, V blocks
       b. Accumulate using log-sum-exp for numerical stability
    3. Normalize final output
    """
    # Chunk inputs
    q_chunks = chunk(query, chunk_size_q, axis=1)
    k_chunks = chunk(key, chunk_size_k, axis=1)
    v_chunks = chunk(value, chunk_size_k, axis=1)

    # Process chunks with scan
    def process_q_chunk(q_chunk):
        def process_kv_chunk(acc, kv_chunk):
            k_chunk, v_chunk = kv_chunk
            # Compute attention scores
            scores = jnp.einsum("...qhd,...khd->...qhk", q_chunk, k_chunk)
            scores = scores * softmax_scale

            # Apply causal mask if needed
            if causal:
                scores = apply_causal_mask(scores)

            # Softmax
            scores = jax.nn.softmax(scores, axis=-1)

            # Apply dropout
            if dropout_prob > 0:
                scores = apply_dropout(scores, dropout_prob, dropout_key)

            # Weighted sum
            output = jnp.einsum("...qhk,...khd->...qhd", scores, v_chunk)

            # Accumulate with log-sum-exp
            return update_accumulator(acc, output, scores)

        # Scan over KV chunks
        init_acc = initialize_accumulator()
        final_acc, _ = jax.lax.scan(process_kv_chunk, init_acc,
                                    (k_chunks, v_chunks))

        return normalize_output(final_acc)

    # Map over Q chunks
    outputs = jax.vmap(process_q_chunk)(q_chunks)
    return reshape_output(outputs)
```

### Pallas TPU Implementation

**Location**: `ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py`

```python
@kernel_registry.register("flash_attention", Platform.PALLAS, Backend.TPU)
def flash_attention(query, key, value, ...):
    """Pallas TPU implementation with TPU-specific optimizations"""
```

#### TPU-Specific Adaptations

```python
def flash_attention_tpu(query, key, value, ...):
    """
    TPU-specific implementation differences:
    1. Uses segment IDs instead of attention masks
    2. Different block size requirements (multiples of 128)
    3. No support for certain features (sliding window, soft cap)
    """
    # TPU limitations
    if cum_seqlens_q is not None:
        raise NotImplementedError(
            "Variable-length sequences not supported on TPU"
        )
    if sliding_window is not None:
        raise NotImplementedError(
            "Sliding window not supported on TPU"
        )
    if logits_soft_cap is not None:
        raise NotImplementedError(
            "Logits soft cap not supported on TPU"
        )

    # Convert attention mask to segment IDs (TPU optimization)
    if attention_mask is not None:
        from ejkernel.xla_utils import mask_to_segment_ids
        q_segment_ids, kv_segment_ids = mask_to_segment_ids(attention_mask)
    else:
        q_segment_ids = kv_segment_ids = None

    # TPU block configuration
    block_sizes = BlockSizes(
        block_q=fwd_params.q_blocksize or 256,      # Larger blocks on TPU
        block_k_major=fwd_params.kv_blocksize or 256,
        block_k=128,  # Fixed for TPU MXU efficiency
        block_b=1,
        block_h=num_heads,
        block_q_dkv=fwd_params.q_blocksize or 256,
        block_k_dkv=fwd_params.kv_blocksize or 256,
        block_k_major_dkv=fwd_params.kv_blocksize or 256,
        block_k_dkv=128,
        block_q_dq=bwd_params.q_blocksize or 256,
        block_k_dq=bwd_params.kv_blocksize or 256,
    )

    # Use Pallas kernel
    return pallas_flash_attention(
        query, key, value,
        segment_ids=SegmentIds(q=q_segment_ids, kv=kv_segment_ids),
        block_sizes=block_sizes,
        causal=causal,
        softmax_scale=softmax_scale
    )
```

#### Pallas Kernel Definition

```python
@functools.partial(
    jax.pallas.pallas_call,
    out_shape=compute_output_shape,
    grid=compute_grid,
    debug=False,
)
def pallas_flash_attention_kernel(
    q_ref,      # Input refs
    k_ref,
    v_ref,
    segment_ids_ref,
    o_ref,      # Output ref
    lse_ref,    # Log-sum-exp ref
    block_sizes,
    causal,
    softmax_scale,
):
    """
    Pallas kernel implementation for TPU.

    Uses TPU-specific features:
    - Matrix Multiply Units (MXU)
    - High-bandwidth memory (HBM)
    - Efficient segment ID handling
    """
    # Get block indices
    block_q_idx = jax.pallas.program_id(0)
    block_b_idx = jax.pallas.program_id(1)

    # Load Q block
    q_block = q_ref[block_b_idx, block_q_idx, :, :]

    # Initialize accumulator
    acc = jnp.zeros_like(o_ref[block_b_idx, block_q_idx, :, :])
    lse = jnp.full((block_sizes.block_q,), -jnp.inf)

    # Iterate over K, V blocks
    for block_k_idx in range(num_k_blocks):
        # Load K, V blocks
        k_block = k_ref[block_b_idx, block_k_idx, :, :]
        v_block = v_ref[block_b_idx, block_k_idx, :, :]

        # Compute attention scores using TPU MXU
        scores = jax.lax.dot_general(
            q_block, k_block.T,
            (((2,), (2,)), ((), ())),
            preferred_element_type=jnp.float32
        )

        # Scale scores
        scores = scores * softmax_scale

        # Apply causal mask
        if causal:
            mask = compute_causal_mask(block_q_idx, block_k_idx)
            scores = jnp.where(mask, scores, -jnp.inf)

        # Check segment IDs (TPU-specific)
        if segment_ids_ref is not None:
            q_segments = segment_ids_ref.q[block_b_idx, block_q_idx]
            kv_segments = segment_ids_ref.kv[block_b_idx, block_k_idx]
            segment_mask = (q_segments[:, None] == kv_segments[None, :])
            scores = jnp.where(segment_mask, scores, -jnp.inf)

        # Online softmax with log-sum-exp
        scores_max = jnp.max(scores, axis=-1, keepdims=True)
        scores_exp = jnp.exp(scores - scores_max)
        scores_sum = jnp.sum(scores_exp, axis=-1, keepdims=True)

        # Update accumulator
        prev_scale = jnp.exp(lse - scores_max)
        acc = acc * prev_scale + jax.lax.dot_general(
            scores_exp, v_block,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32
        )

        # Update log-sum-exp
        lse = jnp.logaddexp(lse, scores_max + jnp.log(scores_sum))

    # Normalize output
    o_ref[block_b_idx, block_q_idx, :, :] = acc / jnp.exp(lse)
    lse_ref[block_b_idx, block_q_idx, :] = lse
```

## Other Attention Mechanisms

### Page Attention

Optimized for KV-cache in inference scenarios with paged memory management.

```python
@kernel_registry.register("page_attention", Platform.TRITON, Backend.GPU)
def page_attention_triton(
    query: Float[Array, "batch num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_heads block_size head_dim"],
    context_lens: Int[Array, "batch"],
    block_tables: Int[Array, "batch max_blocks_per_seq"],
    softmax_scale: float,
    block_size: int,
):
    """
    Paged attention for efficient KV-cache management.

    Features:
    - Block-wise memory management
    - Dynamic context lengths
    - Efficient cache reuse
    """
```

### Ring Attention

Distributed attention for sequence parallelism across devices.

```python
@kernel_registry.register("ring_attention", Platform.XLA, Backend.ANY)
def ring_attention_xla(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    axis_name: str = "ring",
    causal: bool = True,
):
    """
    Ring attention for distributed computation.

    Algorithm:
    1. Split Q, K, V across devices
    2. Rotate K, V chunks in ring pattern
    3. Compute local attention at each step
    4. Accumulate results with proper normalization
    """
    # Get ring communicator
    axis_index = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    # Initialize accumulator
    output = jnp.zeros_like(query)
    lse = jnp.full(query.shape[:-1], -jnp.inf)

    # Ring iterations
    for step in range(axis_size):
        # Rotate K, V
        k_chunk = jax.lax.ppermute(
            key, axis_name,
            perm=[(i, (i + step) % axis_size) for i in range(axis_size)]
        )
        v_chunk = jax.lax.ppermute(
            value, axis_name,
            perm=[(i, (i + step) % axis_size) for i in range(axis_size)]
        )

        # Compute local attention
        local_out, local_lse = compute_attention(
            query, k_chunk, v_chunk,
            causal=causal and (step == 0)
        )

        # Accumulate with log-sum-exp
        output, lse = accumulate_attention(output, lse, local_out, local_lse)

    return output
```

### Block Sparse Attention

Efficient sparse patterns for long-context processing.

```python
@kernel_registry.register("blocksparse_attention", Platform.TRITON, Backend.GPU)
def blocksparse_attention_triton(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    sparsity_mask: Bool[Array, "num_blocks_q num_blocks_k"],
    block_size: int = 64,
):
    """
    Block-sparse attention with configurable sparsity patterns.

    Supports:
    - Fixed sparsity patterns (local, strided, etc.)
    - Learned sparsity patterns
    - Combination patterns (local + global)
    """
```

## Non-Attention Operations

### Recurrent Kernels

```python
@kernel_registry.register("recurrent", Platform.TRITON, Backend.GPU)
def recurrent_triton(
    inputs: Float[Array, "batch seq_len input_dim"],
    hidden: Float[Array, "batch hidden_dim"],
    weights: dict[str, Array],
    cell_type: str = "lstm",
):
    """
    Optimized recurrent operations.

    Supports:
    - LSTM, GRU, vanilla RNN
    - Bidirectional processing
    - Custom activation functions
    """
```

### Mean Pooling

```python
@kernel_registry.register("mean_pooling", Platform.XLA, Backend.ANY)
def mean_pooling_xla(
    inputs: Float[Array, "batch seq_len dim"],
    mask: Bool[Array, "batch seq_len"] | None = None,
):
    """
    Variable-length sequence pooling.

    Features:
    - Proper masking for padded sequences
    - Numerically stable averaging
    - Custom gradients for efficiency
    """
    if mask is not None:
        # Masked mean
        masked_inputs = inputs * mask[:, :, None]
        sum_inputs = jnp.sum(masked_inputs, axis=1)
        count = jnp.sum(mask, axis=1, keepdims=True)
        return sum_inputs / jnp.maximum(count, 1.0)
    else:
        # Standard mean
        return jnp.mean(inputs, axis=1)
```

### Grouped Matrix Multiplication

```python
@kernel_registry.register("grouped_matmul", Platform.PALLAS, Backend.TPU)
def grouped_matmul_pallas(
    a: Float[Array, "batch groups m k"],
    b: Float[Array, "batch groups k n"],
):
    """
    Efficient batched matrix multiplication.

    Optimizations:
    - Fused operations for multiple groups
    - Efficient memory access patterns
    - TPU MXU utilization
    """
    return jax.vmap(jax.vmap(jnp.dot, in_axes=0), in_axes=0)(a, b)
```

## Implementation Patterns

### 1. Custom VJP Pattern

All performance-critical kernels implement custom VJP:

```python
@jax.custom_vjp
def kernel(inputs, params):
    return forward(inputs, params)

def kernel_fwd(inputs, params):
    output = forward(inputs, params)
    residuals = (inputs, params, intermediate_state)
    return output, residuals

def kernel_bwd(residuals, grad_output):
    inputs, params, intermediate = residuals
    grad_inputs = backward_inputs(grad_output, intermediate)
    grad_params = backward_params(grad_output, intermediate)
    return grad_inputs, grad_params

kernel.defvjp(kernel_fwd, kernel_bwd)
```

### 2. Static Parameter Caching

For XLA implementations, cache specialized functions:

```python
_CACHE = {}

def get_specialized_func(static_params):
    if static_params not in _CACHE:
        _CACHE[static_params] = create_specialized(static_params)
    return _CACHE[static_params]
```

### 3. Platform-Specific Adaptations

```python
def kernel(inputs, **kwargs):
    platform = detect_platform()

    if platform == "tpu":
        # TPU-specific preprocessing
        kwargs = adapt_for_tpu(kwargs)

    if platform == "gpu":
        # GPU-specific optimizations
        kwargs = optimize_for_gpu(kwargs)

    return execute_kernel(inputs, **kwargs)
```

### 4. Block Size Selection

```python
def select_block_sizes(seq_len, head_dim, platform):
    """Heuristic block size selection"""
    if platform == "gpu":
        if head_dim <= 64:
            block_q = 128
            block_k = 256
        else:
            block_q = 64
            block_k = 128
    elif platform == "tpu":
        # TPU prefers larger blocks
        block_q = 256
        block_k = 512
    else:
        # Conservative defaults
        block_q = 32
        block_k = 64

    return block_q, block_k
```

## Performance Characteristics

### Triton (GPU)

**Strengths**:

- Direct control over shared memory
- Efficient warp-level operations
- Flexible block sizes
- Custom data types

**Limitations**:

- NVIDIA/AMD specific
- Compilation overhead
- Version compatibility

### Pallas (TPU)

**Strengths**:

- TPU MXU utilization
- Efficient for large blocks
- Native bfloat16 support
- High memory bandwidth

**Limitations**:

- Fixed block size constraints
- Limited feature set
- TPU-specific code

### XLA (Universal)

**Strengths**:

- Works on all platforms
- JAX integration
- Automatic optimization
- Good baseline performance

**Limitations**:

- Less control over low-level details
- May not achieve peak performance
- Limited by XLA compiler capabilities

## Testing Strategy

### Cross-Backend Validation

```python
def test_attention_consistency():
    """Ensure all backends produce equivalent results"""
    # Generate test inputs
    inputs = create_test_inputs()

    # Get all implementations
    triton_impl = kernel_registry.get("attention", Platform.TRITON)
    pallas_impl = kernel_registry.get("attention", Platform.PALLAS)
    xla_impl = kernel_registry.get("attention", Platform.XLA)

    # Compute outputs
    out_triton = triton_impl(**inputs)
    out_pallas = pallas_impl(**inputs)
    out_xla = xla_impl(**inputs)

    # Compare
    assert_allclose(out_triton, out_xla, rtol=1e-5)
    assert_allclose(out_pallas, out_xla, rtol=1e-5)
```

### Gradient Verification

```python
def test_custom_vjp():
    """Verify custom gradients match autograd"""
    # Create test case
    inputs = create_differentiable_inputs()

    # Custom VJP gradient
    custom_grads = jax.grad(kernel_with_custom_vjp)(inputs)

    # Autograd gradient (reference)
    auto_grads = jax.grad(kernel_without_vjp)(inputs)

    # Compare
    assert_allclose(custom_grads, auto_grads, rtol=1e-5)
```

## Conclusion

The kernel implementations in ejKernel demonstrate:

1. **Platform Optimization**: Each backend leverages platform-specific features
2. **Consistent API**: Same interface across all implementations
3. **Performance Focus**: Custom VJP, caching, and optimizations
4. **Extensibility**: Clear patterns for adding new kernels
5. **Robustness**: Comprehensive testing and validation

This multi-backend approach ensures optimal performance across diverse hardware while maintaining ease of use and reliability.
