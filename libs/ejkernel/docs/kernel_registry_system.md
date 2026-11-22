# Kernel Registry System Analysis

## Overview

The kernel registry system is the heart of ejKernel's multi-backend architecture, providing a centralized mechanism for managing multiple implementations of the same algorithm across different platforms and backends. It enables automatic platform-specific kernel selection and ensures consistency across implementations.

## Core Components

### 1. Platform and Backend Enumerations

```python
class Platform(str, Enum):
    TRITON = "triton"  # GPU kernels via Triton
    PALLAS = "pallas"  # TPU/GPU kernels via Pallas
    CUDA = "cuda"      # Direct CUDA implementations
    XLA = "xla"        # XLA compiler-based

class Backend(str, Enum):
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"
    ANY = "any"        # Platform-agnostic
```

### 2. KernelSpec Dataclass

```python
@dataclass(frozen=True)
class KernelSpec:
    platform: Platform
    backend: Backend
    algorithm: str
    implementation: Callable
    priority: int = 0  # Higher values are preferred
```

The KernelSpec encapsulates all metadata about a kernel implementation, including its priority for selection when multiple implementations are available.

### 3. KernelRegistry Class

The central registry managing all kernel implementations with sophisticated selection logic.

## Key Methods

### 1. Registration

```python
@kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU, priority=10)
def flash_attention_triton(q, k, v, **kwargs):
    # Triton implementation
    pass
```

**Features:**

- Decorator-based registration for clean syntax
- Priority-based ordering (higher values preferred)
- Automatic sorting of implementations by priority
- Prevents duplicate registrations for same (algorithm, platform, backend) tuple

### 2. Kernel Retrieval

```python
def get(self, algorithm: str, platform: Platform | str | None = None,
        backend: Backend | str | None = None) -> Callable:
    """
    Retrieves the best matching kernel implementation.

    Selection hierarchy:
    1. Exact match: (algorithm, platform, backend)
    2. Platform match with Backend.ANY
    3. Auto-detection based on JAX backend
    4. Raise ValueError if no match found
    """
```

**Selection Logic:**

1. **Explicit Request**: If platform and backend specified, find exact match
2. **Platform-Only**: If only platform specified, match platform with any backend
3. **Auto-Detection**: If platform="auto" or None, detect from JAX backend:
   - JAX backend "gpu" → prefer Triton, fallback to Pallas/XLA
   - JAX backend "tpu" → prefer Pallas, fallback to XLA
   - JAX backend "cpu" → use XLA
4. **Priority Resolution**: When multiple matches exist, select highest priority

### 3. Signature Validation

```python
def validate_signatures(self, algorithm: str | None = None, verbose: bool = False):
    """
    Validates parameter consistency across all implementations of an algorithm.

    Checks:
    - Parameter names match
    - Parameter order matches
    - Parameter kinds match (positional, keyword-only, etc.)
    - Default values match
    - Type annotations are compatible
    """
```

**Validation Process:**

1. Groups all implementations by algorithm
2. For each algorithm with multiple implementations:
   - Extract signatures using `inspect.signature`
   - Normalize type annotations (handles different import paths)
   - Compare parameter names, order, kinds, defaults
   - Report discrepancies if verbose=True
   - Raise ValueError on mismatch

### 4. Type Normalization

```python
def _normalize_type_string(type_annotation: Any) -> str:
    """
    Normalizes type annotation strings for comparison.

    Examples:
    - 'jaxtyping.Float[Array, "..."]' → 'Float[Array, "..."]'
    - 'ejkernel.ops.utils.datacarrier.FwdParams' → 'FwdParams'
    - 'typing.Union[X, None]' → 'Optional[X]'
    """
```

This enables signature validation even when implementations use different import styles for the same types.

## Platform Detection

```python
def detect_platform(algorithm: str, platform: str = "auto",
                    maybe_pallas: bool = False) -> Platform:
    """
    Intelligently detects the best platform for the current hardware.

    Detection logic:
    1. If explicit platform requested, return it
    2. Get JAX backend (GPU/TPU/CPU)
    3. Check available implementations in registry
    4. Apply platform preferences:
       - GPU: Triton > Pallas > XLA
       - TPU: Pallas > XLA
       - CPU: XLA only
    """
```

## Registration Patterns

### Basic Registration

```python
@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU)
def my_kernel_gpu(x, y, z):
    return triton_implementation(x, y, z)
```

### Multi-Backend Registration

```python
# GPU implementation
@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU, priority=100)
def my_kernel_gpu(x, y, z):
    return gpu_optimized(x, y, z)

# TPU implementation
@kernel_registry.register("my_kernel", Platform.PALLAS, Backend.TPU, priority=100)
def my_kernel_tpu(x, y, z):
    return tpu_optimized(x, y, z)

# Universal fallback
@kernel_registry.register("my_kernel", Platform.XLA, Backend.ANY, priority=50)
def my_kernel_xla(x, y, z):
    return xla_generic(x, y, z)
```

### Platform-Specific Features

```python
@kernel_registry.register("attention", Platform.PALLAS, Backend.TPU)
def attention_tpu(query, key, value, **kwargs):
    # TPU-specific: use segment IDs instead of attention masks
    if "attention_mask" in kwargs:
        segment_ids = mask_to_segment_ids(kwargs["attention_mask"])
        kwargs["segment_ids"] = segment_ids
        del kwargs["attention_mask"]
    return pallas_attention(query, key, value, **kwargs)
```

## Usage Examples

### Direct Registry Usage

```python
from ejkernel import kernel_registry, Platform, Backend

# Get specific implementation
flash_attn_gpu = kernel_registry.get(
    algorithm="flash_attention",
    platform=Platform.TRITON,
    backend=Backend.GPU
)

# Auto-detect platform
flash_attn_auto = kernel_registry.get("flash_attention")

# Use the kernel
output = flash_attn_gpu(query, key, value, causal=True)
```

### Module Integration

```python
class FlashAttention(Kernel):
    def get_impl(self, cfg: FlashAttentionConfig):
        """Get implementation from registry based on config"""
        return kernel_registry.get(
            algorithm="flash_attention",
            platform=detect_platform("flash_attention", cfg.platform),
            backend=cfg.backend,
        )
```

## Design Decisions

### 1. Priority-Based Selection

**Rationale**: Different implementations may have different performance characteristics. Priority allows expressing preferences while maintaining flexibility.

**Example**: Triton implementations typically have priority=100 on GPU, while XLA fallbacks have priority=50.

### 2. Signature Validation

**Rationale**: Ensures that different implementations of the same algorithm are drop-in replacements for each other.

**Benefits**:

- Catches API mismatches early
- Enforces consistency across backends
- Simplifies testing and maintenance

### 3. Type Normalization

**Rationale**: Different files may import the same type differently (e.g., absolute vs relative imports).

**Solution**: Strip module paths and normalize common patterns to enable comparison.

### 4. Backend.ANY

**Rationale**: Some implementations are platform-specific but work across backends (e.g., XLA works on CPU/GPU/TPU).

**Usage**: Typically for fallback implementations that prioritize correctness over performance.

## Thread Safety

The registry implementation uses thread-safe operations:

- Registration uses list operations which are thread-safe in Python
- Retrieval only reads from immutable data structures
- No mutable shared state after initialization

## Error Handling

### Registration Errors

```python
# Duplicate registration attempt
@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU)
def impl1(): pass

@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU)
def impl2(): pass  # Raises ValueError
```

### Retrieval Errors

```python
# No matching implementation
kernel = kernel_registry.get("nonexistent_kernel")  # Raises ValueError

# No implementation for requested platform
kernel = kernel_registry.get("cpu_only_kernel", platform=Platform.TRITON)  # Raises ValueError
```

## Performance Considerations

### Registration (One-time)

- O(n log n) sorting after each registration
- Happens at module import time
- No runtime overhead

### Retrieval (Runtime)

- O(n) search through implementations
- Typically n < 10 per algorithm
- Result can be cached by caller

## Extension Points

### Custom Platform

```python
class Platform(str, Enum):
    TRITON = "triton"
    PALLAS = "pallas"
    CUDA = "cuda"
    XLA = "xla"
    CUSTOM = "custom"  # Add new platform
```

### Custom Selection Logic

```python
class CustomRegistry(KernelRegistry):
    def get(self, algorithm, platform=None, backend=None):
        # Add custom selection logic
        if should_use_custom_logic():
            return self.get_custom_impl(algorithm)
        return super().get(algorithm, platform, backend)
```

### Dynamic Registration

```python
def register_dynamic_kernel(algorithm, impl, platform, backend):
    """Register kernel at runtime"""
    kernel_registry._implementations[algorithm].append(
        KernelSpec(platform, backend, algorithm, impl, priority=0)
    )
    kernel_registry._implementations[algorithm].sort(
        key=lambda x: x.priority, reverse=True
    )
```

## Best Practices

### 1. Consistent Signatures

Always ensure all implementations of an algorithm have identical signatures:

```python
# Good - consistent signatures
@kernel_registry.register("matmul", Platform.TRITON, Backend.GPU)
def matmul_gpu(a: Array, b: Array, transpose_a: bool = False, transpose_b: bool = False):
    pass

@kernel_registry.register("matmul", Platform.XLA, Backend.ANY)
def matmul_xla(a: Array, b: Array, transpose_a: bool = False, transpose_b: bool = False):
    pass
```

### 2. Platform-Specific Optimizations

Use platform-specific features when beneficial:

```python
@kernel_registry.register("attention", Platform.TRITON, Backend.GPU)
def attention_gpu(q, k, v, **kwargs):
    # Use Triton's shared memory optimizations
    return triton_attention_kernel(q, k, v, **kwargs)

@kernel_registry.register("attention", Platform.PALLAS, Backend.TPU)
def attention_tpu(q, k, v, **kwargs):
    # Use TPU's matrix units
    return pallas_attention_kernel(q, k, v, **kwargs)
```

### 3. Graceful Degradation

Always provide a fallback implementation:

```python
@kernel_registry.register("my_operation", Platform.XLA, Backend.ANY, priority=0)
def my_operation_fallback(x, y):
    """Basic implementation that works everywhere"""
    return jnp.dot(x, y)
```

### 4. Documentation

Document platform-specific limitations:

```python
@kernel_registry.register("sparse_attention", Platform.PALLAS, Backend.TPU)
def sparse_attention_tpu(q, k, v, sparsity_mask):
    """
    TPU implementation of sparse attention.

    Note: TPU version requires sparsity_mask to be block-aligned
    with blocks of size 128x128.
    """
    pass
```

## Testing Strategies

### 1. Cross-Backend Validation

```python
def test_kernel_consistency():
    """Test that all backends produce identical results"""
    algorithms = ["flash_attention", "page_attention"]

    for algo in algorithms:
        impls = kernel_registry.get_all_implementations(algo)

        # Test with same inputs
        inputs = generate_test_inputs(algo)
        outputs = [impl(*inputs) for impl in impls]

        # Verify all outputs are equivalent
        for out in outputs[1:]:
            assert_allclose(outputs[0], out, rtol=1e-5)
```

### 2. Signature Validation Fn

```python
def test_signature_consistency():
    """Ensure all implementations have consistent signatures"""
    kernel_registry.validate_signatures(verbose=True)
```

### 3. Platform Detection

```python
def test_platform_detection():
    """Test automatic platform detection"""
    with mock_jax_backend("gpu"):
        platform = detect_platform("flash_attention")
        assert platform == Platform.TRITON

    with mock_jax_backend("tpu"):
        platform = detect_platform("flash_attention")
        assert platform == Platform.PALLAS
```

## Conclusion

The kernel registry system provides a robust, extensible foundation for multi-backend kernel management in ejKernel. Its key strengths include:

1. **Flexibility**: Easy to add new implementations and platforms
2. **Consistency**: Signature validation ensures API compatibility
3. **Intelligence**: Automatic platform detection and priority-based selection
4. **Simplicity**: Clean decorator-based registration API
5. **Reliability**: Multiple fallback mechanisms ensure availability

This design enables ejKernel to provide optimal performance across diverse hardware platforms while maintaining a consistent API for users.
