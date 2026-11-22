# ejKernel Project Overview

## Executive Summary

ejKernel is a sophisticated, high-performance kernel library for JAX that provides multi-backend support for various deep learning operations, with a particular focus on efficient attention mechanisms. The project demonstrates advanced software engineering practices with a modular architecture designed for extensibility and performance.

## Project Information

- **Name**: ejKernel (EasyDeL JAX Kernels)
- **Version**: 0.0.1
- **Author**: Erfan Zare Chavoshi
- **License**: Apache License 2.0
- **Python Support**: 3.11 - 3.13
- **Primary Dependencies**: JAX, Triton, Pallas, CUTLASS CuTe DSL (optional), jaxtyping, beartype

## Core Objectives

1. **Multi-Platform Support**: Provide optimized kernel implementations for GPU (NVIDIA/AMD), TPU, and CPU
2. **Performance Optimization**: Automatic kernel selection and configuration tuning for optimal performance
3. **Extensibility**: Easy addition of new kernel implementations through registry system
4. **Type Safety**: Comprehensive type annotations with runtime validation
5. **Developer-Friendly**: Clean API with sensible defaults and progressive disclosure of advanced features

## Architecture Overview

```md
ejkernel/
├── kernels/           # Core kernel implementations
│   ├── _triton/      # Triton GPU kernels
│   ├── _cute/        # CUTLASS CuTe DSL kernels
│   ├── _pallas/      # Pallas TPU kernels
│   ├── _xla/         # XLA CPU/fallback kernels
│   └── _cuda/        # Native CUDA kernels
├── modules/          # High-level operation modules
│   └── operations/   # Wrapped kernels with auto-selection
├── ops/              # Kernel execution framework
│   ├── config/       # Configuration management
│   ├── core/         # Base kernel classes
│   ├── execution/    # Execution orchestration
│   └── utils/        # Utilities and helpers
├── xla_utils/        # XLA-specific utilities
└── callib/           # Calibration library
```

## Key Features

### 1. Multi-Backend Kernel Registry

- **Automatic Platform Detection**: Seamlessly selects optimal implementation based on hardware
- **Priority-based Selection**: Configurable kernel selection with fallback mechanisms
- **Signature Validation**: Ensures consistency across implementations

### 2. Configuration Management Hierarchy

- Override → Overlay → Cache → Persistent → Autotune → Heuristics
- In-memory and persistent caching for optimal configurations
- Sophisticated autotuning with backward pass validation

### 3. Attention Mechanism Zoo

- **Flash Attention v2**: Memory-efficient O(N) attention with causal masking, dropout, sliding windows
- **Page Attention**: Optimized for KV-cache in inference scenarios
- **Ring Attention**: Distributed attention for sequence parallelism
- **Block Sparse Attention**: Efficient sparse patterns for long-context processing
- **GLA (Gated Linear Attention)**: Linear complexity attention alternative
- **Lightning Attention**: Layer-dependent decay attention mechanism
- **MLA (Multi-head Latent Attention)**: Efficient latent attention implementation
- **Ragged Attention**: Variable-length sequence support

### 4. Advanced Operations

- **Recurrent Kernels**: Optimized RNN-like operations with custom gradients
- **Mean Pooling**: Variable-length sequence pooling with proper masking
- **Grouped Matrix Multiplication**: Efficient batched matrix operations
- **Native Sparse Operations**: Block-sparse matrix computations

### 5. Developer Experience

- **Full Type Hints**: jaxtyping annotations for better IDE support
- **Modular Architecture**: Easy to extend with new kernel implementations
- **Comprehensive Testing**: Extensive test coverage with XLA vs Triton comparisons
- **Automatic Differentiation**: Custom VJP rules for efficient gradients
- **Profiling Integration**: Built-in support for JAX profiling tools

## Platform Support Matrix

| Algorithm | Triton GPU | CUTE GPU | CUDA | Pallas TPU | XLA (CPU/Fallback) |
|-----------|------------|----------|------|------------|--------------------|
| Flash Attention v2 | ✅ | ✅ | 🚧 | ✅ | ✅ |
| Page Attention | ✅ | ❌ | 🚧 | ✅ | ✅ |
| Ring Attention | ✅ | ❌ | 🚧 | ✅ | ✅ |
| Native Sparse | ✅ | ❌ | 🚧 | ❌ | ✅ |
| GLA | ✅ | ❌ | ❌ | 🚧 | ✅ |
| Lightning Attention | ✅ | ❌ | 🚧 | ❌ | ✅ |
| MLA | ✅ | ❌ | ❌ | 🚧 | ❌ |
| Ragged Page Attention | ✅ | ❌ | 🚧 | ✅ | ✅ |
| Recurrent | ✅ | ❌ | 🚧 | 🚧 | ✅ |
| Mean Pooling | ✅ | ❌ | 🚧 | 🚧 | ✅ |
| Grouped MatMul | 🚧 | ❌ | 🚧 | ✅ | ✅ |
| Quantized MatMul | ✅ | ✅ | ✅ | ❌ | ✅ |

✅ = Implemented and optimized
🚧 = Under development
❌ = Not yet implemented
* CUTE backend is correctness-first and currently includes quantized matmul and flash attention.

## Design Principles

1. **Convention over Configuration**: Sensible defaults everywhere with optional overrides
2. **Progressive Disclosure**: Simple API for common cases, advanced features when needed
3. **Fail Gracefully**: Multiple fallback layers ensure reliability
4. **Optimize Lazily**: Cache results, autotune on demand
5. **Type Everything**: Static and runtime validation for correctness

## Usage Example

```python
import jax
import jax.numpy as jnp
from ejkernel.modules import FlashAttention, create_default_executor

# Initialize
executor = create_default_executor()
attention = FlashAttention()

# Create inputs
batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
key = jax.random.PRNGKey(0)
q = k = v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

# Execute attention with automatic optimization
output = executor(
    attention, q, k, v,
    causal=True,
    dropout_prob=0.1,
    logits_soft_cap=30.0  # Gemma-2 style soft capping
)
```

## Environment Variables

```bash
# Autotuning
EJKERNEL_AUTOTUNE_POLICY=autotune|heuristics  # Default: autotune
EJKERNEL_LOG_AUTOTUNE=1                        # Log candidate testing

# Profiling
EJKERNEL_OPS_STAMP=hash|json|none             # Default: none
EJKERNEL_OPS_RECORD=1                          # Enable invocation recording

# Triton-specific
EJKERNEL_TRITON_SMEM_LIMIT=101376             # Shared memory limit (bytes)
```

## Integration Points

The library integrates seamlessly with:

- **JAX Ecosystem**: Full compatibility with JAX transformations
- **EasyDeL Framework**: Parent framework for JAX deep learning
- **JAX Profiling Tools**: XProf, TensorBoard integration
- **Distributed Training**: shard_map support for model parallelism

## Future Roadmap

### Near-term

- Flash Attention 3 implementation
- Flash Decoding for optimized inference
- Quantized attention (INT8/INT4)
- Fused LayerNorm + Attention kernels

### Long-term

- Sliding Window Attention with Sinks
- Differential Attention with learnable patterns
- Mixture of Attention mechanisms
- Speculative decoding kernels
- Continuous batching support

## Conclusion

ejKernel represents a production-quality implementation of high-performance machine learning kernels with a focus on:

- **Modularity**: Clean separation of concerns
- **Performance**: State-of-the-art optimizations
- **Reliability**: Multiple fallback mechanisms
- **Extensibility**: Easy to add new features
- **Maintainability**: Well-documented and tested

The project serves as an excellent foundation for building high-performance deep learning systems in JAX.
