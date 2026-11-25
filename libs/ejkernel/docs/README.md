# ejKernel Architecture & Documentation Guide

## Overview

This directory contains architectural documentation and API references for **ejKernel** — a multi-backend JAX kernel library for deep learning operations (attention, paged serving, sparse/linear/recurrent attention, state-space models, grouped/quantized matrix multiplication, fused losses, and collective matmuls).

## Documentation Structure

### Architecture & Design Docs

1. **[project_overview.md](./project_overview.md)**
   - Project summary, goals, and layered architecture overview
   - Platform support matrix and integration points

2. **[kernel_registry_system.md](./kernel_registry_system.md)**
   - Multi-platform kernel registry (`Platform`, `Backend`, `KernelSpec`, `KernelRegistry`)
   - Priority-sorted registration, runtime unsupported-feature guards (`del <param>`, `_TUNING_PARAM_NAMES`, `__ejkernel_unsupported__`), and signature validation
   - Automatic platform detection (`detect_platform`)

3. **[ops_system_architecture.md](./ops_system_architecture.md)**
   - Core execution infrastructure (`Kernel`, `Invocation`, `Executor`)
   - 7-tier `ConfigSelectorChain`, autotuning, and caching architecture
   - Custom VJP and distributed execution integration

4. **[kernel_implementations.md](./kernel_implementations.md)**
   - Backend-specific implementation strategies across Triton, Pallas, CUDA, CuTe, TileLang, and XLA
   - Performance characteristics and hardware trade-offs

5. **[module_operations.md](./module_operations.md)**
   - High-level `ejkernel.modules` API design (`Kernel` subclasses + functional wrappers)
   - Operation dataclasses (`BaseOperationConfig`), heuristics, and candidate generation

6. **[utilities_and_helpers.md](./utilities_and_helpers.md)**
   - XLA utilities, device fingerprinting, `FwdParams`/`BwdParams`, and numerical helpers

7. **[test_suite_and_examples.md](./test_suite_and_examples.md)**
   - Test organization, cross-backend XLA parity validation, and benchmarking

8. **[comprehensive_architecture_report.md](./comprehensive_architecture_report.md)**
   - End-to-end architectural report and design rationale

### API Guides

- **[api/ops.md](./api/ops.md)** — `ejkernel.ops` framework, 7-tier `ConfigSelectorChain`, `PersistentCache`, and `ejkernel.ops.tuned`
- **[api/modules.md](./api/modules.md)** — high-level `ejkernel.modules` operation wrappers
- **[api/types.md](./api/types.md)** — `MaskInfo` and shared type definitions
- **[maskinfo_guide.md](./maskinfo_guide.md)** — attention mask / segment-ID representation guide

## Key Capabilities

### Architectural Strengths

- **Layered Architecture**: Clean separation between user-facing modules (`ejkernel.modules`), execution/tuning infrastructure (`ejkernel.ops`), and backend kernel implementations (`ejkernel.kernels`).
- **6 Implementation Platforms across 5 Hardware Backends**: Unified dispatch across **Triton**, **Pallas** (TPU & GPU), **CUDA** (FFI), **CuTe** (CUTLASS CuTe DSL), **TileLang** (TVM-FFI), and universal **XLA** fallback, targeting `Backend.GPU`, `Backend.MPS`, `Backend.TPU`, `Backend.CPU`, and wildcard `Backend.ANY`.
- **7-Tier Configuration Selection**: Override → Overlay → Memory Cache → Persistent Cache (provenance-enforced `~/ejkernel-presistent-cache/`) → Autotune → Heuristics → Error.
- **Shipped Pre-Tuned Database**: `ejkernel.ops.tuned` ships a read-only SQLite table (`tuned_kernels.db`) recording winning `(platform, config)` pairs with bucketed shape lookups and runner-up/baseline timings.
- **Runtime Safety & Signature Consistency**: Automatic cross-backend signature validation (excluding `_TUNING_PARAM_NAMES`) and AST-based unsupported-feature guards raising `EjkernelRuntimeError`.
- **Custom Gradients & Distributed Execution**: Automatic `jax.custom_vjp` wiring via `fwd_with_residuals()` + `vjp()`, plus `jax.shard_map` support via `create_shard_map_wrapper()`.

### Design Patterns

- **Registry Pattern**: `kernel_registry` for algorithm discovery, priority sorting, and validation
- **Strategy Pattern**: `ConfigChooser` / `ConfigSelectorChain` for configuration selection
- **Chain of Responsibility**: 7-tier config resolution and platform fallback (`detect_platform`)
- **Template Method**: Hardware-platform and context method specialization on `Kernel` (`*_gpu`, `*_tpu`, `*_shard_map`)

## Project Statistics

Verified against `libs/ejkernel/ejkernel/kernels/` and `libs/ejkernel/ejkernel/modules/operations/`:

- **Registered Algorithms**: **105** algorithms in `kernel_registry` (every algorithm has a mandatory `Platform.XLA` / `Backend.ANY` implementation)
- **High-Level Operation Modules**: **79** operation modules under `ejkernel.modules.operations`
- **Implementation Platforms (6)** — **253** total registered kernel implementations in source:
  - `Platform.XLA` (`"xla"`): 105 (`Backend.ANY`)
  - `Platform.PALLAS` (`"pallas"`): 71 (69 `Backend.TPU`, 2 `Backend.GPU`)
  - `Platform.TILELANG` (`"tilelang"`): 47 (`Backend.GPU`)
  - `Platform.TRITON` (`"triton"`): 21 (`Backend.GPU`)
  - `Platform.CUDA` (`"cuda"`): 5 (`Backend.GPU`)
  - `Platform.CUTE` (`"cute"`): 4 (`Backend.GPU`)
  - *(Optional GPU backends `triton`, `cute`, and `tilelang` register conditionally at import time when their optional dependencies are installed.)*
- **Target Hardware Backends (5)**: `Backend.GPU`, `Backend.MPS` (Apple Silicon Metal/MPS), `Backend.TPU`, `Backend.CPU`, `Backend.ANY`
- **Hardware Support**: GPU (NVIDIA CUDA / AMD ROCm), Apple Silicon (MPS), Google TPU, CPU

## Usage Guide

### For Developers

1. Start with [project_overview.md](./project_overview.md) to understand the stack
2. Review [kernel_registry_system.md](./kernel_registry_system.md) for registration, validation, and platform detection
3. Study [api/ops.md](./api/ops.md) and [ops_system_architecture.md](./ops_system_architecture.md) for `Kernel`, `Executor`, caching, and `ejkernel.ops.tuned`
4. Examine [kernel_implementations.md](./kernel_implementations.md) for backend-specific implementation details

### For Users

1. Read [module_operations.md](./module_operations.md) and [api/modules.md](./api/modules.md) for high-level API usage
2. Check [test_suite_and_examples.md](./test_suite_and_examples.md) for practical patterns and testing
3. Reference [utilities_and_helpers.md](./utilities_and_helpers.md) and [maskinfo_guide.md](./maskinfo_guide.md) for helper utilities

### For Contributors

1. Review [comprehensive_architecture_report.md](./comprehensive_architecture_report.md)
2. Ensure every new kernel provides a `Platform.XLA` (`Backend.ANY`) fallback and passes `kernel_registry.validate_signatures(...)`
3. Follow testing requirements in [test_suite_and_examples.md](./test_suite_and_examples.md)

## Quick Reference

### Adding a New Kernel

```python
from dataclasses import dataclass
from jaxtyping import Array
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules.base import detect_platform
from ejkernel.ops import ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel


# 1. Register backend implementations (XLA fallback is mandatory)
@kernel_registry.register("my_kernel", Platform.XLA, Backend.ANY)
def my_kernel_xla(x: Array) -> Array:
    return xla_implementation(x)


@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU)
def my_kernel_triton(x: Array) -> Array:
    return triton_implementation(x)


# 2. Define config and Kernel subclass
@dataclass
class MyConfig:
    platform: str = "auto"
    backend: str = "any"


class MyKernel(Kernel[MyConfig, Array]):
    def __init__(self):
        super().__init__(op_id="my_kernel")

    def get_impl(self, cfg: MyConfig):
        platform = detect_platform(self.op_id, cfg.platform)
        backend = Backend.ANY if platform == Platform.XLA else cfg.backend
        return kernel_registry.get(self.op_id, platform=platform, backend=backend)

    def run(self, x: Array, *, cfg: MyConfig) -> Array:
        impl = self.get_impl(cfg)
        return impl(x)

    def heuristic_cfg(self, inv: Invocation[MyConfig, Array]) -> MyConfig:
        return MyConfig()


# 3. Provide a convenience function backed by an Executor
_executor = Executor(ConfigSelectorChain(cache=ConfigCache()))


def my_kernel(x: Array, *, cfg: MyConfig | None = None) -> Array:
    return _executor(MyKernel(), x, cfg=cfg)
```

### Using Flash Attention

```python
from ejkernel.modules import FlashAttentionConfig, flash_attention
from ejkernel.ops import FwdParams

# Basic usage with automatic platform detection and optimization
output = flash_attention(query, key, value, causal=True)

# With explicit configuration
config = FlashAttentionConfig(
    fwd_params=FwdParams(q_blocksize=128, kv_blocksize=128, num_warps=4, num_stages=2),
    platform="triton",
    backend="gpu",
)
output = flash_attention(query, key, value, causal=True, cfg=config)
```

---

*Package Version: 0.0.82*
