# Module Operations Layer Analysis

## Overview

The module operations layer provides high-level interfaces to kernel implementations, abstracting away the complexity of platform selection, configuration management, and execution orchestration. This layer serves as the primary API for users, offering clean interfaces with automatic optimization.

## Architecture

```md
ejkernel/modules/
├── operations/
│   ├── __init__.py                        # Public API exports
│   ├── configs.py                         # Configuration dataclasses
│   ├── attention.py                       # Standard attention
│   ├── flash_attention.py                 # Flash attention v2
│   ├── blocksparse_attention.py           # Block-sparse patterns
│   ├── ring_attention.py                  # Distributed attention
│   ├── page_attention.py                  # Paged KV-cache
│   ├── native_sparse_attention.py         # Native sparse ops
│   ├── recurrent.py                       # RNN operations
│   ├── pooling.py                         # Sequence pooling
│   ├── grouped_matmul.py                  # Batched GEMM
│   ├── gated_linear_attention.py          # GLA implementation
│   ├── lightning_attention.py             # Lightning attention
│   ├── multi_head_latent_attention.py     # Multi-head latent attention
│   ├── ragged_decode_attention.py         # Variable-length decode
│   ├── ragged_page_attention_v2.py        # Variable-length paged v2
│   ├── ragged_page_attention_v3.py        # Variable-length paged v3
│   ├── scaled_dot_product_attention.py    # Basic SDPA
│   └── unified_attention.py               # Unified attention interface
```

## Available Operations

### Attention Variants

- **Attention**: Standard multi-head attention with XLA optimization

- **FlashAttention**: Memory-efficient O(N) complexity attention

- **FlashMLA**: Multi-head latent attention with low-rank compression

- **GLAttention**: Gated linear attention mechanism

- **LightningAttention**: Layer-aware attention optimization

- **NativeSparseAttention**: Sparse attention with block patterns

- **PageAttention**: Paged KV cache for serving workloads

- **RaggedDecodeAttention**: Variable-length decode attention

- **RaggedPageAttentionv2**: Page attention for variable-length sequences v2

- **RaggedPageAttentionv3**: Advanced page attention for variable-length sequences v3

- **RecurrentAttention**: Stateful recurrent attention

- **RingAttention**: Distributed attention with ring topology

- **ScaledDotProductAttention**: Standard scaled dot-product attention

- **UnifiedAttention**: Unified attention interface

### Additional Operations

- **BlockSparseAttention**: Block-sparse attention patterns

- **GroupedMatmul**: Efficient grouped matrix multiplication

- **MeanPooling**: Sequence mean pooling operation

## Base Configuration Classes

### BaseOperationConfig

```python
@dataclass
class BaseOperationConfig:
    """Base configuration for all operations"""
    platform: Literal["triton", "pallas", "cuda", "xla", "auto", "cute"] = "auto"
    backend: str = "any"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)
```

### Operation-Specific Configurations

```python
@dataclass
class FlashAttentionConfig(BaseOperationConfig):
    """Flash Attention configuration"""
    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        # Auto-convert dicts to dataclasses
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)

@dataclass
class BlockSparseAttentionConfig(BaseOperationConfig):
    """Block-sparse attention configuration"""
    block_size: int = 64
    sparsity_pattern: Literal["local", "strided", "custom"] = "local"
    local_window_size: int | None = None
    num_global_blocks: int | None = None
    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

@dataclass
class RingAttentionConfig(BaseOperationConfig):
    """Ring attention configuration for distributed execution"""
    axis_name: str = "ring"
    chunk_size: int | None = None
    overlap_communication: bool = True

@dataclass
class PageAttentionConfig(BaseOperationConfig):
    """Paged attention configuration"""
    block_size: int = 16
    max_blocks_per_seq: int = 256
    sliding_window: int | None = None
```

## Module Implementation Pattern

Each module follows a consistent pattern inheriting from the `Kernel` base class:

```python
class OperationModule(Kernel[OperationConfig, OutputType]):
    """High-level operation interface"""

    def __init__(self):
        super().__init__(op_id="operation_name")

    def get_impl(self, cfg: OperationConfig) -> Callable:
        """Fetch implementation from kernel registry"""
        return kernel_registry.get(
            algorithm=self.op_id,
            platform=detect_platform(self.op_id, cfg.platform),
            backend=cfg.backend,
        )

    def run(self, *args, cfg: OperationConfig, **kwargs) -> OutputType:
        """Execute with configuration"""
        impl = self.get_impl(cfg)
        return impl(*args, **process_kwargs(kwargs, cfg))

    def heuristic_cfg(self, inv: Invocation) -> OperationConfig:
        """Default configuration based on input characteristics"""
        return OperationConfig(...)

    def candidate_cfgs(self, inv: Invocation) -> list[OperationConfig]:
        """Autotuning candidates"""
        return generate_candidates(inv)

    # Optional platform-specific candidates
    def candidate_cfgs_gpu(self, inv: Invocation) -> list[OperationConfig]:
        """GPU-optimized candidates"""
        return generate_gpu_candidates(inv)

    def candidate_cfgs_tpu(self, inv: Invocation) -> list[OperationConfig]:
        """TPU-optimized candidates"""
        return generate_tpu_candidates(inv)
```

## Flash Attention Module

The most sophisticated module implementation:

```python
class FlashAttention(Kernel[FlashAttentionConfig, Array]):
    """Flash Attention v2 with automatic optimization"""

    def __init__(self):
        super().__init__(op_id="flash_attention")

    def run(self, query, key, value, attention_mask=None, bias=None,
            causal=False, softmax_scale=None, dropout_prob=0.0,
            dropout_seed=None, cfg: FlashAttentionConfig = None, **kwargs):
        """Execute flash attention"""

        # Get implementation
        impl = self.get_impl(cfg)

        # Process inputs
        softmax_scale = softmax_scale or (1.0 / math.sqrt(query.shape[-1]))

        # Execute
        return impl(
            query=query, key=key, value=value,
            attention_mask=attention_mask,
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            dropout_seed=dropout_seed,
            fwd_params=cfg.fwd_params,
            bwd_params=cfg.bwd_params,
            **kwargs
        )

    def candidate_cfgs_gpu(self, inv: Invocation) -> list[FlashAttentionConfig]:
        """GPU-specific autotuning candidates"""

        # Extract metadata
        query = inv.kwargs["query"]
        head_dim = int(query.shape[-1])
        seq_len_q = int(query.shape[-2])
        sliding_window = inv.kwargs.get("sliding_window", None)

        # Adaptive block size selection based on head dimension
        if head_dim <= 64:
            q_opts = [32, 64, 128]
            kv_opts = [64, 128, 256]
        elif head_dim <= 128:
            q_opts = [32, 64, 128]
            kv_opts = [32, 64, 128, 256]
        else:
            q_opts = [32, 64]
            kv_opts = [32, 64, 128]

        # Sliding window constraints
        if sliding_window is not None:
            # Limit KV blocks to window size
            max_kv = min(256, sliding_window)
            kv_opts = [k for k in kv_opts if k <= max_kv]

        # Generate configurations
        configs = []
        for q_block in q_opts:
            for kv_block in kv_opts:
                # Estimate shared memory usage
                smem_bytes = estimate_smem_usage(q_block, kv_block, head_dim)

                # Skip if exceeds hardware limit
                if smem_bytes > get_smem_limit():
                    continue

                # Select warp configuration
                num_warps = select_num_warps(q_block, kv_block, head_dim)
                num_stages = select_num_stages(smem_bytes)

                configs.append(FlashAttentionConfig(
                    fwd_params=FwdParams(
                        q_blocksize=q_block,
                        kv_blocksize=kv_block,
                        num_warps=num_warps,
                        num_stages=num_stages
                    ),
                    bwd_params=BwdParams(
                        q_blocksize=min(q_block, 64),
                        kv_blocksize=min(kv_block, 64),
                        num_warps=num_warps,
                        num_stages=num_stages
                    ),
                    platform="triton",
                    backend="gpu"
                ))

        return configs

    def create_shard_map_wrapper(self, query, key, value, mesh,
                                in_specs, out_specs, check_vma=False, **kwargs):
        """Create distributed execution wrapper"""

        def wrapped_flash_attn(query, key, value, **fixed_kwargs):
            return self.run(query, key, value, **fixed_kwargs)

        # Create shard_map function
        shard_map_fn = shard_map(
            wrapped_flash_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma
        )

        return shard_map_fn, (query, key, value), kwargs
```

## Executor and Convenience Functions

Each module provides a convenience function with pre-configured executor:

```python
# Create executor with configuration chain
_flash_executor = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("flash-attn", cfg_type=FlashAttentionConfig),
    )
)

# Public API function
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
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    cfg: FlashAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple | None = None,
    out_specs: Any | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """
    Flash Attention v2 with automatic optimization.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        attention_mask: Optional boolean mask
        bias: Optional attention bias
        causal: Enable causal masking
        softmax_scale: Softmax temperature (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability
        dropout_seed: Random seed for dropout
        sliding_window: Local attention window size
        logits_soft_cap: Soft capping value (Gemma-2 style)
        cfg: Optional configuration override
        mesh: Optional mesh for distributed execution
        in_specs: Input partition specs for shard_map
        out_specs: Output partition specs for shard_map

    Returns:
        Attention output [batch, seq_len_q, num_heads, head_dim]
    """
    method = "shard_map" if mesh is not None else None

    return _flash_executor(
        FlashAttention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        dropout_seed=dropout_seed,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg
    )
```

## Other Module Examples

### Page Attention

```python
class PageAttention(Kernel[PageAttentionConfig, Array]):
    """Paged attention for KV-cache management"""

    def run(self, query, key_cache, value_cache, context_lens,
            block_tables, cfg: PageAttentionConfig, **kwargs):
        """Execute paged attention"""

        impl = self.get_impl(cfg)

        return impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            block_size=cfg.block_size,
            **kwargs
        )

    def heuristic_cfg(self, inv: Invocation) -> PageAttentionConfig:
        """Default configuration for paged attention"""
        # Analyze cache dimensions
        key_cache = inv.kwargs.get("key_cache")
        if key_cache is not None:
            block_size = key_cache.shape[2]  # Infer from cache shape
        else:
            block_size = 16  # Default

        return PageAttentionConfig(
            block_size=block_size,
            max_blocks_per_seq=256,
            platform="auto"
        )
```

### Ring Attention

```python
class RingAttention(Kernel[RingAttentionConfig, Array]):
    """Distributed ring attention"""

    def run(self, query, key, value, cfg: RingAttentionConfig, **kwargs):
        """Execute ring attention"""

        # Ensure we're in a distributed context
        if cfg.axis_name not in jax.experimental.maps.thread_resources.env.shape:
            raise ValueError(f"Ring attention requires axis '{cfg.axis_name}'")

        impl = self.get_impl(cfg)

        return impl(
            query=query,
            key=key,
            value=value,
            axis_name=cfg.axis_name,
            chunk_size=cfg.chunk_size,
            **kwargs
        )

    def candidate_cfgs(self, inv: Invocation) -> list[RingAttentionConfig]:
        """Generate chunk size candidates"""
        seq_len = inv.kwargs["query"].shape[1]

        # Try different chunk sizes
        chunk_sizes = []
        for divisor in [2, 4, 8, 16]:
            if seq_len % divisor == 0:
                chunk_sizes.append(seq_len // divisor)

        return [
            RingAttentionConfig(
                chunk_size=cs,
                overlap_communication=overlap
            )
            for cs in chunk_sizes
            for overlap in [True, False]
        ]
```

### Mean Pooling

```python
class MeanPooling(Kernel[MeanPoolingConfig, Array]):
    """Variable-length sequence pooling"""

    def run(self, inputs, mask=None, cfg: MeanPoolingConfig = None, **kwargs):
        """Execute mean pooling"""

        impl = self.get_impl(cfg)

        if mask is None and cfg.sequence_lengths is not None:
            # Create mask from sequence lengths
            mask = create_mask_from_lengths(
                cfg.sequence_lengths,
                inputs.shape[1]
            )

        return impl(inputs=inputs, mask=mask, **kwargs)

    def fwd_with_residuals(self, inputs, mask, cfg):
        """Forward with saved mask for gradient"""
        output = self.run(inputs, mask, cfg=cfg)
        residuals = (mask, inputs.shape)
        return output, residuals

    def vjp(self, residuals, y, dy, inputs, mask, cfg):
        """Custom backward for mean pooling"""
        mask_saved, input_shape = residuals

        # Gradient is dy broadcasted and masked
        if mask_saved is not None:
            # Compute normalization factor
            counts = jnp.sum(mask_saved, axis=1, keepdims=True)
            # Broadcast gradient
            grad = dy[:, None, :] / jnp.maximum(counts[:, :, None], 1.0)
            # Apply mask
            grad = grad * mask_saved[:, :, None]
        else:
            # No mask: uniform gradient
            seq_len = input_shape[1]
            grad = dy[:, None, :] / seq_len

        return (grad, None)  # gradient for inputs, None for mask
```

## Autotuning Heuristics

### Shared Memory Estimation

```python
def estimate_smem_usage(q_block: int, kv_block: int, head_dim: int,
                       dtype: jnp.dtype = jnp.float16) -> int:
    """Estimate shared memory usage for flash attention"""
    bytes_per_element = dtype.itemsize

    # Q, K, V blocks in shared memory
    q_smem = q_block * head_dim * bytes_per_element
    k_smem = kv_block * head_dim * bytes_per_element
    v_smem = kv_block * head_dim * bytes_per_element

    # Softmax statistics
    stats_smem = q_block * 4  # float32 for max and sum

    # Total with padding for alignment
    total = q_smem + k_smem + v_smem + stats_smem
    return align_to(total, 128)  # Align to 128 bytes
```

### Warp Selection

```python
def select_num_warps(q_block: int, kv_block: int, head_dim: int) -> int:
    """Select optimal number of warps"""
    # Estimate parallelism
    total_work = q_block * kv_block * head_dim

    if total_work <= 4096:
        return 4
    elif total_work <= 8192:
        return 8
    else:
        return 16
```

### Platform Detection

```python
def detect_platform(algorithm: str, requested: str = "auto",
                   maybe_pallas: bool = False) -> str:
    """Detect best platform for current hardware"""

    if requested != "auto":
        return requested

    # Get JAX backend
    backend = jax.default_backend()

    if backend == "gpu":
        # Check available implementations
        has_triton = kernel_registry.has(algorithm, Platform.TRITON)
        has_pallas = kernel_registry.has(algorithm, Platform.PALLAS)

        if has_triton:
            return Platform.TRITON
        elif has_pallas and maybe_pallas:
            return Platform.PALLAS
        else:
            return Platform.XLA

    elif backend == "tpu":
        if kernel_registry.has(algorithm, Platform.PALLAS):
            return Platform.PALLAS
        else:
            return Platform.XLA

    else:  # CPU
        return Platform.XLA
```

## Module Design Patterns

### 1. Configuration Auto-Conversion

```python
def __post_init__(self):
    """Auto-convert nested configurations"""
    if isinstance(self.fwd_params, dict):
        self.fwd_params = FwdParams(**self.fwd_params)
    if isinstance(self.bwd_params, dict):
        self.bwd_params = BwdParams(**self.bwd_params)
```

### 2. Platform-Specific Candidate Generation

```python
def candidate_cfgs(self, inv: Invocation) -> list[Config]:
    """Dispatch to platform-specific generation"""
    platform = get_device_platform()

    # Try platform-specific method first
    method_name = f"candidate_cfgs_{platform}"
    if hasattr(self, method_name):
        return getattr(self, method_name)(inv)

    # Fallback to generic
    return self._generic_candidate_cfgs(inv)
```

### 3. Input Validation and Processing

```python
def prepare(self, *args, **kwargs):
    """Validate and preprocess inputs"""
    query, key, value = args[:3]

    # Validate shapes
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must have same head dimension")

    # Add default softmax scale
    if "softmax_scale" not in kwargs:
        kwargs["softmax_scale"] = 1.0 / math.sqrt(query.shape[-1])

    return args, kwargs
```

### 4. Distributed Execution Support

```python
def run_shard_map(self, *args, cfg, **kwargs):
    """Specialized distributed execution"""
    # Create wrapper that binds configuration
    def local_compute(*local_args):
        return self.run(*local_args, cfg=cfg, **kwargs)

    # Apply shard_map
    return shard_map(
        local_compute,
        mesh=kwargs.pop("mesh"),
        in_specs=kwargs.pop("in_specs"),
        out_specs=kwargs.pop("out_specs")
    )(*args)
```

## Public API Design

### Consistent Function Signatures

All public functions follow a consistent pattern:

```python
def operation_name(
    # Required positional arguments
    primary_input: Array,
    *secondary_inputs: Array,

    # Common optional arguments
    mask: Array | None = None,
    bias: Array | None = None,

    # Operation-specific arguments
    specific_param: type = default,

    # Configuration override
    cfg: OperationConfig | None = None,

    # Distributed execution
    mesh: Mesh | None = None,
    in_specs: tuple | None = None,
    out_specs: Any | None = None,

    # Additional kwargs
    **kwargs
) -> Array:
    """Operation with automatic optimization."""
```

### Error Messages

```python
def validate_inputs(query, key, value):
    """Provide helpful error messages"""
    if query.ndim != 4:
        raise ValueError(
            f"Query must be 4D [batch, seq_len, num_heads, head_dim], "
            f"got shape {query.shape}"
        )

    if key.shape[-1] != value.shape[-1]:
        raise ValueError(
            f"Key and value must have same head dimension, "
            f"got key: {key.shape[-1]}, value: {value.shape[-1]}"
        )
```

## Testing Support

### Configuration Override

```python
def test_flash_attention():
    """Test with specific configuration"""
    test_config = FlashAttentionConfig(
        fwd_params=FwdParams(q_blocksize=64, kv_blocksize=64),
        platform="xla"  # Force XLA for testing
    )

    output = flash_attention(
        query, key, value,
        cfg=test_config  # Override autotuning
    )
```

### Mock Executor

```python
def test_with_mock_executor():
    """Test with mock executor"""
    mock_executor = Executor(
        ConfigSelectorChain(
            policy=AutotunePolicy(
                allow_autotune=False,  # Disable autotuning
                allow_heuristics=True
            )
        )
    )

    # Inject mock executor
    with mock.patch("ejkernel.modules.operations._flash_executor", mock_executor):
        output = flash_attention(query, key, value)
```

## Performance Optimization

### Lazy Implementation Loading

```python
class LazyModule(Kernel):
    def __init__(self):
        super().__init__(op_id="lazy_op")
        self._impl_cache = {}

    def get_impl(self, cfg):
        """Cache implementation lookup"""
        cache_key = (cfg.platform, cfg.backend)

        if cache_key not in self._impl_cache:
            self._impl_cache[cache_key] = kernel_registry.get(
                self.op_id, cfg.platform, cfg.backend
            )

        return self._impl_cache[cache_key]
```

### Configuration Reuse

```python
# Global default configurations
_DEFAULT_CONFIGS = {
    "flash_attention": FlashAttentionConfig(
        fwd_params=FwdParams(q_blocksize=128, kv_blocksize=128)
    ),
    "page_attention": PageAttentionConfig(block_size=16)
}

def get_default_config(operation: str) -> BaseOperationConfig:
    """Get cached default configuration"""
    return copy.deepcopy(_DEFAULT_CONFIGS[operation])
```

## Integration Examples

### With Model Code

```python
class TransformerBlock:
    def __call__(self, x, mask=None):
        # Use flash attention transparently
        attn_output = flash_attention(
            query=self.q_proj(x),
            key=self.k_proj(x),
            value=self.v_proj(x),
            attention_mask=mask,
            causal=True,
            dropout_prob=self.dropout_rate
        )
        return self.out_proj(attn_output)
```

### With Distributed Training

```python
# Setup mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=("data", "model"))

# Distributed attention
output = flash_attention(
    query, key, value,
    mesh=mesh,
    in_specs=(P("data", None), P("data", None), P("data", None)),
    out_specs=P("data", None)
)
```

### With Mixed Precision

```python
# Cast inputs for mixed precision
query_fp16 = query.astype(jnp.float16)
key_fp16 = key.astype(jnp.float16)
value_fp16 = value.astype(jnp.float16)

# Attention in FP16
output_fp16 = flash_attention(
    query_fp16, key_fp16, value_fp16,
    causal=True
)

# Cast back to FP32 if needed
output = output_fp16.astype(jnp.float32)
```

## Conclusion

The module operations layer provides:

1. **Clean API**: Consistent, well-documented public functions
2. **Automatic Optimization**: Transparent autotuning and caching
3. **Platform Flexibility**: Seamless multi-backend execution
4. **Type Safety**: Full type annotations with validation
5. **Extensibility**: Clear patterns for adding new operations

This layer successfully abstracts the complexity of the underlying kernel system while providing flexibility for advanced users through configuration overrides and custom executors.
