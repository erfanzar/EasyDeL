# ejkernel.ops - Kernel Framework and Configuration Management

## Overview

The `ejkernel.ops` package provides a sophisticated framework for implementing high-performance JAX operations with automatic configuration management, caching, and autotuning. It's the foundation upon which `ejkernel.modules` operations are built.

**Key Features:**

- Abstract kernel interface with platform-specific implementations
- Multi-tier configuration selection and caching
- Automatic performance autotuning
- Batch processing utilities (vmap/pmap with shared config)
- Custom gradient support via VJP

---

## Architecture

```md
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│            kernel(args) or executor(kernel, args)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Executor                                │
│    Coordinates config selection, kernel dispatch, and caching   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   ConfigSelectorChain   │     │         Kernel          │
│                         │     │                         │
│  1. ConfigCache         │     │  - heuristic_cfg()      │
│  2. PersistentCache     │     │  - config_space()       │
│  3. Heuristic           │     │  - run()                │
│  4. Autotuning          │     │  - run_{platform}()     │
└─────────────────────────┘     └─────────────────────────┘
```

---

## Kernel: The Building Block

A `Kernel` is an abstract base class for implementing operations. Each kernel defines:

1. **How to run the operation** (`run()` or platform-specific methods)
2. **How to generate default configs** (`heuristic_cfg()`)
3. **What configs to explore** (`config_space()` for autotuning)

### Basic Kernel Implementation

```python
from dataclasses import dataclass
from ejkernel.ops import Kernel
import jax.numpy as jnp

# 1. Define your configuration
@dataclass(frozen=True)
class MyConfig:
    block_size: int = 128
    num_warps: int = 4
    algorithm: str = "default"

# 2. Implement the kernel
class MyKernel(Kernel[MyConfig, jnp.ndarray]):
    def __init__(self):
        super().__init__(name="my_kernel")

    def run(self, a, b, *, cfg: MyConfig, **kwargs) -> jnp.ndarray:
        """Default implementation."""
        # Your operation logic here
        return jnp.dot(a, b)

    def heuristic_cfg(self, inv: "Invocation") -> MyConfig:
        """Generate config based on input shapes."""
        # inv.args contains the input arrays
        a = inv.args[0]
        block = 128 if a.shape[0] >= 128 else 64
        return MyConfig(block_size=block)
```

### Platform-Specific Implementations

Kernels can have different implementations for different platforms:

```python
class AttentionKernel(Kernel[AttentionConfig, jnp.ndarray]):
    def run_triton(self, q, k, v, *, cfg, **kwargs):
        """Triton GPU implementation."""
        return triton_flash_attention(q, k, v, **cfg.triton_params)

    def run_pallas(self, q, k, v, *, cfg, **kwargs):
        """Pallas TPU implementation."""
        return pallas_attention(q, k, v, **cfg.pallas_params)

    def run_xla(self, q, k, v, *, cfg, **kwargs):
        """XLA fallback implementation."""
        return xla_attention(q, k, v)

    def run(self, q, k, v, *, cfg, **kwargs):
        """Default fallback."""
        return self.run_xla(q, k, v, cfg=cfg, **kwargs)
```

The executor automatically selects the appropriate platform method based on available hardware.

### Custom Gradients (VJP)

Kernels can define custom backward passes:

```python
class MyKernel(Kernel[MyConfig, jnp.ndarray]):
    def run(self, a, b, *, cfg, **kwargs):
        return custom_forward(a, b, cfg)

    def custom_vjp(self, primals, cfg, kwargs, cotangents):
        """Custom backward pass."""
        a, b = primals
        ct = cotangents[0]
        grad_a = custom_grad_a(a, b, ct, cfg)
        grad_b = custom_grad_b(a, b, ct, cfg)
        return (grad_a, grad_b)
```

---

## Invocation: Capturing Call Context

An `Invocation` captures everything about a kernel call:

```python
from ejkernel.ops import Invocation

# Created automatically by the executor
inv = Invocation(
    kernel=my_kernel,
    args=(a, b),       # Positional arguments
    kwargs={'scale': 1.0},  # Keyword arguments
    metadata={...}     # Shapes, dtypes, device info
)

# Access abstracted shapes (for config lookup)
inv.abstract_args  # JAX ShapeDtypeStruct versions
inv.device_fingerprint  # Unique device identifier
```

---

## Executor: Running Kernels

The `Executor` is the main interface for running kernels with automatic config selection:

```python
from ejkernel.ops import Executor, ConfigSelectorChain, ConfigCache

# Create the execution stack
cache = ConfigCache()
selector = ConfigSelectorChain(cache=cache)
executor = Executor(selector)

# Run a kernel
result = executor(my_kernel, a, b)

# Run with explicit config
result = executor(my_kernel, a, b, cfg=my_config)
```

### How Config Selection Works

When you call `executor(kernel, args)`:

1. **Check ConfigCache** - Fast in-memory lookup
2. **Check PersistentCache** - Disk-based storage for cross-session persistence
3. **Use Heuristic** - Call `kernel.heuristic_cfg()` for shape-based default
4. **Autotune** - If enabled, benchmark multiple configs and cache the best

---

## Configuration Selection and Caching

### ConfigCache (In-Memory)

Fast lookup for repeated calls with same shapes:

```python
from ejkernel.ops import ConfigCache

cache = ConfigCache()

# Store a config
cache.put(key, config)

# Retrieve
config = cache.get(key)  # Returns None if not found

# Check existence
if key in cache:
    ...
```

### PersistentCache (Disk-Based)

Survives across sessions:

```python
from ejkernel.ops import PersistentCache

# Stored in ~/.cache/ejkernel/ by default
persistent = PersistentCache(path="~/.cache/ejkernel/configs")

# Same interface as ConfigCache
persistent.put(key, config)
config = persistent.get(key)
```

### ConfigSelectorChain

Combines multiple selection strategies:

```python
from ejkernel.ops import ConfigSelectorChain, ConfigCache, PersistentCache

selector = ConfigSelectorChain(
    cache=ConfigCache(),           # Level 1: In-memory
    persistent=PersistentCache(),  # Level 2: Disk
    autotune=True                  # Level 3: Benchmark if not found
)
```

### Temporary Overrides

```python
from ejkernel.ops import overlay_cache, policy_override

# Temporarily use different cache
with overlay_cache(my_custom_cache):
    result = executor(kernel, args)

# Temporarily change autotuning policy
with policy_override(autotune=False):
    result = executor(kernel, args)  # No autotuning
```

---

## Autotuning

ejkernel can automatically find optimal configurations by benchmarking:

### Defining Config Space

```python
class MyKernel(Kernel[MyConfig, jnp.ndarray]):
    def config_space(self, inv: Invocation) -> list[MyConfig]:
        """Return configs to try during autotuning."""
        return [
            MyConfig(block_size=64, num_warps=2),
            MyConfig(block_size=128, num_warps=4),
            MyConfig(block_size=256, num_warps=8),
        ]
```

### Manual Autotuning

```python
from ejkernel.ops import autotune, benchmark

# Autotune a specific kernel call
best_config = autotune(
    kernel, a, b,
    configs=kernel.config_space(inv),
    num_warmup=3,
    num_iters=10
)

# Simple function benchmarking
time_ms = benchmark(lambda: kernel(a, b), num_iters=100)
```

### Batch Autotuning

Autotune all recorded invocations at once:

```python
from ejkernel.ops import autotune_recorded, record_invocation, get_invocations

# Enable recording
record_invocation(kernel, inv)

# ... run your model with various inputs ...

# Get all recorded invocations
invocations = get_invocations()

# Autotune all at once
results = autotune_recorded(invocations)
```

---

## Batch Processing

### vmap_with_config

Vectorized execution with shared config selection (avoids re-selecting config for each batch element):

```python
from ejkernel.ops import vmap_with_config

# Standard vmap would select config per element
# vmap_with_config selects once and broadcasts
batched_result = vmap_with_config(
    kernel,
    in_axes=(0, 0),  # Batch over first axis
)(batched_a, batched_b)
```

### pmap_with_config

Parallel execution across devices:

```python
from ejkernel.ops import pmap_with_config

# Run kernel across multiple devices
parallel_result = pmap_with_config(
    kernel,
    axis_name='devices'
)(sharded_a, sharded_b)
```

---

## Configuration Parameters

### FwdParams and BwdParams

Standard parameter containers for attention kernels:

```python
from ejkernel.ops import FwdParams, BwdParams

fwd = FwdParams(
    q_blocksize=128,      # Query block size
    kv_blocksize=256,     # Key-value block size
    num_warps=4,          # GPU warps
    num_stages=2,         # Pipeline stages
)

bwd = BwdParams(
    q_blocksize=64,
    kv_blocksize=128,
)
```

---

## Utility Functions

### Device Detection

```python
from ejkernel.ops import device_kind, get_device_platform, device_fingerprint

# Get device type
device_kind()  # Returns "gpu", "tpu", or "cpu"

# Get platform for dispatch
get_device_platform()  # Returns "triton", "pallas", or "xla"

# Unique device identifier (for caching)
fingerprint = device_fingerprint()
```

### Sharding Utilities

```python
from ejkernel.ops import sharding_fingerprint, default_key_builder_with_sharding

# Get fingerprint of array sharding
fingerprint = sharding_fingerprint(array)

# Build cache key including sharding info
key = default_key_builder_with_sharding(kernel, args, kwargs)
```

### Serialization

```python
from ejkernel.ops import to_json, from_json, stable_json

# Serialize config to JSON
json_str = to_json(config)

# Deserialize
config = from_json(json_str, MyConfig)

# Stable JSON (deterministic ordering for cache keys)
stable_str = stable_json(config)
```

### HLO Analysis

```python
from ejkernel.ops import (
    find_labels_in_lowered,
    extract_labels_from_hlo_text,
    labels_to_configs
)

# Find labeled operations in lowered JAX code
lowered = jax.jit(fn).lower(args)
labels = find_labels_in_lowered(lowered)

# Extract from HLO text
labels = extract_labels_from_hlo_text(hlo_text)

# Convert labels to configs
configs = labels_to_configs(labels)
```

### Labeling Operations

```python
from ejkernel.ops import label

# Add label for HLO analysis
@label("my_attention")
def my_attention(q, k, v):
    return attention_impl(q, k, v)
```

---

## Best Practices

### 1. Use the Module API for Common Operations

For standard attention operations, use `ejkernel.modules` which handles all the complexity:

```python
# Recommended for most users
from ejkernel.modules import flash_attention
output = flash_attention(q, k, v, causal=True)
```

### 2. Implement Kernel Only When Needed

Create custom kernels only for:

- New operations not in `ejkernel.modules`
- Operations needing special configuration
- Performance-critical custom implementations

### 3. Define Good Heuristics

Good heuristic configs reduce the need for autotuning:

```python
def heuristic_cfg(self, inv):
    # Consider input shapes
    q = inv.args[0]
    batch, seq_len, heads, dim = q.shape

    # Larger sequences benefit from larger blocks
    if seq_len >= 2048:
        block = 256
    elif seq_len >= 512:
        block = 128
    else:
        block = 64

    return MyConfig(block_size=block)
```

### 4. Leverage Caching

Configs are cached automatically. Ensure consistent key generation:

```python
# Shapes and dtypes are used for cache keys
# Actual values don't matter, only structure
```

### 5. Profile Before Optimizing

Use JAX's built-in profiler before manual autotuning:

```python
with jax.profiler.trace("/tmp/jax-trace"):
    result = executor(kernel, args)
```

---

## Type Variables

| Type Variable | Description |
|---------------|-------------|
| `Cfg` | Configuration type (e.g., `MyConfig`) |
| `Out` | Output type (e.g., `jnp.ndarray`) |

---

## Module Structure

```md
ejkernel/ops/
├── __init__.py           # Public API exports
├── core/
│   ├── kernel.py         # Kernel, Invocation
│   └── types.py          # Cfg, Out type variables
├── config/
│   ├── cache.py          # ConfigCache, overlay_cache
│   ├── persistent.py     # PersistentCache
│   └── selection.py      # ConfigSelectorChain, AutotunePolicy
├── execution/
│   ├── executor.py       # Executor, ConfigChooser
│   ├── batch.py          # vmap_with_config, pmap_with_config
│   └── tuning.py         # Autotuner, benchmark utilities
├── registry.py           # Invocation recording
└── utils/                # Utility functions
```
