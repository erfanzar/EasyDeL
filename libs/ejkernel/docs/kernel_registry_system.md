# Kernel Registry System Analysis

## Overview

The kernel registry system (`ejkernel/kernels/_registry.py`) is the backbone of ejKernel's multi-platform architecture. It provides centralized decorator-based registration, priority-sorted lookup, runtime unsupported-feature guards, and cross-implementation signature validation across all implementation platforms and hardware backends.

## Core Components

### 1. `Platform` and `Backend` Enumerations

Both enums inherit from `enum.StrEnum`:

```python
from enum import StrEnum


class Platform(StrEnum):
    TRITON = "triton"      # OpenAI Triton GPU kernels
    PALLAS = "pallas"      # JAX Pallas kernels (TPU and GPU)
    CUDA = "cuda"          # Native ahead-of-time CUDA C/C++ FFI kernels
    CUTE = "cute"          # CUTLASS CuTe DSL kernels
    TILELANG = "tilelang"  # TileLang DSL kernels (TVM-FFI backed)
    XLA = "xla"            # Portable XLA / JAX primitive implementations


class Backend(StrEnum):
    GPU = "gpu"  # GPU backend (CUDA / ROCm)
    MPS = "mps"  # Apple Silicon Metal/MPS backend (jax.default_backend() == "mps")
    TPU = "tpu"  # Google TPU backend
    CPU = "cpu"  # CPU-only backend
    ANY = "any"  # Wildcard backend matching any hardware query
```

### 2. `KernelSpec` Dataclass

```python
@dataclass(frozen=True)
class KernelSpec:
    platform: Platform
    backend: Backend
    algorithm: str
    implementation: Callable
    priority: int = 0  # Higher values are preferred during lookup
```

Each `KernelSpec` binds a wrapped kernel callable to its algorithm name, platform, backend, and selection priority.

### 3. `KernelRegistry` Class and `kernel_registry` Singleton

`KernelRegistry` stores `self._registry: dict[str, list[KernelSpec]]` keyed by lower-cased algorithm name. Built-in kernels register against the global singleton `kernel_registry` (re-exported from `ejkernel`, `ejkernel.kernels`, and `ejkernel.kernels._registry`) at import time.

---

## Key Methods and Semantics

### 1. Registration and Priority-Sorted Multiple-Registration Semantics

```python
@kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU, priority=10)
def flash_attention_triton(query, key, value, **kwargs):
    ...
```

**Actual registration behavior:**

- **Case-insensitive algorithm keys**: stored under `algorithm.lower()`.
- **String-to-enum coercion**: `platform` and `backend` accept either enum members or their lowercase string values (raising `ValueError` on invalid enum strings).
- **Multiple registrations coexist (no duplicate error)**: `register()` does **not** reject duplicate `(algorithm, platform, backend)` tuples. Every registration appends a `KernelSpec` to `self._registry[key]` and sorts the list in descending `priority` order:
  ```python
  self._registry[key].append(spec)
  self._registry[key].sort(key=lambda x: x.priority, reverse=True)
  ```
- **Tie-breaking**: Because Python's `list.sort()` is stable, higher `priority` values always win during lookup; when multiple registrations share the same `(algorithm, platform, backend, priority)`, the earliest-registered spec comes first.
- **Validation wrapper**: `register()` wraps the function in `_wrapped`, preserving metadata (`@functools.wraps`) and signature (`_wrapped.__signature__ = inspect.signature(func)`), and enforces runtime unsupported-feature checks before invoking the underlying implementation.

### 2. Runtime Unsupported-Feature Guards

Every call to a registered kernel passes through `_wrapped(*args, **kwargs)`, which runs `_collect_unsupported_reasons(func, args, kwargs)` before calling `func`. If any unsupported reason is found, it raises `EjkernelRuntimeError(f"{algorithm} (platform={plat.value}): " + "; ".join(reasons))`.

Two mechanisms feed into `_collect_unsupported_reasons`:

#### A. AST-Based `del <param>` Detection (`_get_ignored_params`)

The registry parses the source of `func` with `inspect.getsource` + `ast.parse` (cached per function in `_IGNORED_PARAM_CACHE`) and collects every bare `ast.Name` target in a `del` statement (`_DelVisitor.visit_Delete`).

- If a parameter name appears in `del <param>` inside `func`, is **not** in `_TUNING_PARAM_NAMES`, and the caller supplied a **non-default** value (checked via `_is_non_default`), the wrapper records `f"{name} is not supported"` and raises `EjkernelRuntimeError`.
- Passing the parameter's default value (e.g., `dropout_prob=0.0` or `bias=None`) is allowed.
- Deleting a dict entry such as `del kwargs["foo"]` is an `ast.Subscript`, not an `ast.Name`, so it does not trigger this guard.

```python
@kernel_registry.register("my_attention", Platform.TRITON, Backend.GPU)
def my_attention_triton(query, key, value, dropout_prob: float = 0.0, block_q: int = 64):
    del dropout_prob  # Non-default dropout_prob != 0.0 -> raises EjkernelRuntimeError!
    del block_q       # In _TUNING_PARAM_NAMES -> safely ignored without error
    return run_triton(query, key, value)
```

#### B. `_TUNING_PARAM_NAMES` Exemption

The following 31 tuning/tiling parameter names in `_TUNING_PARAM_NAMES` are exempt from **both** the AST `del <param>` guard and cross-backend `validate_signatures()` comparisons:

```python
_TUNING_PARAM_NAMES: set[str] = {
    "block_m",
    "block_n",
    "block_k",
    "tile_m",
    "tile_n",
    "tile_i",
    "chunk",
    "block_q",
    "block_kv",
    "block_size",
    "block_v",
    "block_d",
    "block_e",
    "block_dim",
    "num_warps",
    "num_stages",
    "num_ctas",
    "num_waves",
    "num_sms",
    "num_splits",
    "split_k",
    "use_bf16",
    "seq_threshold_3d",
    "num_par_softmax_segments",
    "max_warps",
    "num_queries_per_block",
    "num_kv_pages_per_block",
    "num_kv_splits",
    "gemm_block",
    "fwd_params",
    "bwd_params",
}
```

This allows backends to accept backend-specific tiling/launch kwargs (or `del` unused tuning hints passed by a shared caller) without breaking signature parity or tripping the unsupported-feature guard.

#### C. Explicit `__ejkernel_unsupported__` Hook

A kernel function may attach an `__ejkernel_unsupported__` attribute for conditional or dynamic feature checks that cannot be expressed with `del <param>` alone. `_collect_unsupported_reasons` handles three forms:

- **Callable**: invoked as `extra(**bound.arguments)` (falling back to `extra(bound.arguments)` on `TypeError`); may return a `str`, an `Iterable` of truthy reason strings, or `None`/empty if supported.
- **String (`str`)**: appended unconditionally as a reason.
- **Iterable**: every truthy element is converted via `str(...)` and appended.

```python
def _check_unsupported(**bound):
    reasons = []
    if bound.get("causal") and bound.get("bias") is not None:
        reasons.append("combining causal=True with additive bias is not supported")
    return reasons


@kernel_registry.register("my_attention", Platform.CUDA, Backend.GPU)
def my_attention_cuda(query, key, value, causal: bool = False, bias=None):
    ...


my_attention_cuda.__ejkernel_unsupported__ = _check_unsupported
```

#### D. Exception Normalization

During execution of `func(*args, **kwargs)`, `_wrapped` lets `EjkernelRuntimeError` propagate as-is and wraps any `NotImplementedError` or `ValueError` whose message contains `"not supported"` or `"unsupported"` (case-insensitive) into an `EjkernelRuntimeError` annotated with `f"{algorithm} (platform={plat.value}): ..."`.

### 3. Kernel Retrieval (`get`)

```python
def get(
    self,
    algorithm: str,
    platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None,
    backend: Backend | Literal["gpu", "mps", "tpu", "cpu", "any"] | None = None,
) -> Callable:
```

**Matching rules (in order):**

1. Look up `candidates = self._registry[algorithm.lower()]` (already sorted by descending `priority`). Raises `ValueError` if the algorithm is not registered.
2. Convert string `platform` / `backend` arguments via `Platform(platform)` / `Backend(backend)`.
   - **Important:** `"auto"` is **not** a member of `Platform`; passing `platform="auto"` directly to `kernel_registry.get()` raises `ValueError`. Use `ejkernel.modules.base.detect_platform(algorithm, "auto")` first to resolve `"auto"` to a concrete `Platform`.
3. **Primary scan (priority order):** Return the first `spec.implementation` where:
   - `platform is None` or `spec.platform == platform`, AND
   - `backend is None` or `spec.backend == backend` or `spec.backend == Backend.ANY`.
4. **Iterative fallbacks** (non-recursive):
   - If `platform == Platform.XLA` and `backend != Backend.ANY`, retry scanning for `(Platform.XLA, Backend.ANY)`.
   - If `backend == Backend.ANY`, resolve `default_backend = Backend(jax.default_backend())` and scan for a matching `platform` with `spec.backend in (default_backend, Backend.ANY)`.
5. If no match is found, raise `ValueError`.

### 4. Inspection (`list_algorithms` and `list_implementations`)

```python
# Sorted list of all registered lowercase algorithm names
algos: list[str] = kernel_registry.list_algorithms()

# Shallow copy of KernelSpec entries for an algorithm, sorted by descending priority
specs: list[KernelSpec] = kernel_registry.list_implementations("flash_attention")
```

### 5. Signature Validation (`validate_signatures`)

```python
def validate_signatures(self, algorithm: str | None, verbose: bool = False) -> bool:
```

**Validation process:**

1. When `algorithm is None`, validates every algorithm in `self.list_algorithms()` (returning `None`).
2. For a given algorithm with fewer than 2 implementations, returns `True`.
3. Uses `specs[0]` (the highest-priority registration) as the reference and compares every `spec` in `specs[1:]` against it.
4. **Filters out `_TUNING_PARAM_NAMES`** from both reference and candidate parameter lists before comparing.
5. Compares parameter count, parameter names in order, parameter kinds (`POSITIONAL_OR_KEYWORD`, `KEYWORD_ONLY`, etc.), default values, and type annotations via `_types_are_equivalent()`.
6. Emits a `UserWarning` for each mismatch (does **not** raise `ValueError` on signature mismatch; only raises `ValueError` if `algorithm` is not registered) and returns `True` if all matched, `False` otherwise.
7. `ejkernel/__init__.py` automatically calls `kernel_registry.validate_signatures(None)` at package import time.

### 6. Type Normalization (`_normalize_type_string`)

`_normalize_type_string` normalizes annotation strings so implementations using different import paths compare equal:

- Strips `<class '...'>` wrappers
- Strips `jaxtyping.`, `jax._src.lax.lax.`, `jax._src.typing.`, `jax.typing.`, and `ejkernel.<subpkg>.` prefixes
- Normalizes `jaxlib._jax.Array`, `jax.jaxlib._jax.Array`, and `jax.Array` to `Array`
- Normalizes `jax.lax.` to `lax.`
- Maps any annotation containing `PrecisionLike`, `DotAlgorithm`, or `DotAlgorithmPreset` to `"PrecisionLike"`
- Maps any annotation containing `DTypeLike` or `SupportsDType` to `"DTypeLike"`

---

## Automatic Platform Detection (`detect_platform`)

Hardware-aware platform selection lives in `ejkernel/modules/base.py`, separate from `KernelRegistry.get()`:

```python
from ejkernel.modules.base import detect_platform

platform = detect_platform(
    algorithm="flash_attention",
    platform="auto",          # "triton" | "pallas" | "cuda" | "cute" | "tilelang" | "xla" | "auto" | None
    prefer_pallas=False,
    prefer_cuda=False,
    prefer_triton=False,
    prefer_cute=False,
    prefer_tilelang=False,
)
```

**Resolution order when `platform in ("auto", None)`:**

1. If `prefer_pallas=True`: `Platform.PALLAS` when a matching TPU (on `"tpu"`) or GPU (on `"gpu"`) Pallas spec exists.
2. TPU backend + Pallas TPU available → `Platform.PALLAS`.
3. Explicit GPU preference flags (checked in order on `"gpu"` / `"cuda"`):
   - `prefer_cute=True` (NVIDIA GPU + CuTe registered) → `Platform.CUTE`
   - `prefer_cuda=True` (NVIDIA GPU + CUDA registered) → `Platform.CUDA`
   - `prefer_triton=True` (GPU + Triton registered) → `Platform.TRITON`
   - `prefer_tilelang=True` (GPU + TileLang registered) → `Platform.TILELANG`
4. Default GPU auto-selection (when no `prefer_*` flag matched):
   - NVIDIA GPU + CuTe available → `Platform.CUTE`
   - NVIDIA GPU + CUDA available → `Platform.CUDA`
   - GPU + Triton available → `Platform.TRITON`
   - *(Note: `Platform.TILELANG` and GPU `Platform.PALLAS` are not auto-selected unless explicitly requested or enabled via `prefer_tilelang=True` / `prefer_pallas=True`.)*
5. Fallback on any backend (CPU, MPS, or GPU/TPU without a specialized kernel) → `Platform.XLA`.

---

## Usage Examples

### Direct Registry Usage

```python
from ejkernel import Backend, Platform, kernel_registry
from ejkernel.modules.base import detect_platform

# Retrieve a specific backend implementation
flash_attn_triton = kernel_registry.get(
    algorithm="flash_attention",
    platform=Platform.TRITON,
    backend=Backend.GPU,
)

# Resolve platform automatically first, then look up in the registry
plat = detect_platform("flash_attention", platform="auto")
backend = Backend.ANY if plat == Platform.XLA else None
flash_attn_auto = kernel_registry.get("flash_attention", platform=plat, backend=backend)

output = flash_attn_auto(query, key, value, causal=True)
```

### Multi-Platform Registration Pattern

```python
@kernel_registry.register("my_kernel", Platform.CUTE, Backend.GPU, priority=20)
def my_kernel_cute(x, y, *, block_m: int = 128):
    return cute_impl(x, y, block_m=block_m)


@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU, priority=10)
def my_kernel_triton(x, y, *, block_m: int = 128, num_warps: int = 4):
    return triton_impl(x, y, block_m=block_m, num_warps=num_warps)


@kernel_registry.register("my_kernel", Platform.PALLAS, Backend.TPU, priority=10)
def my_kernel_pallas(x, y, *, block_m: int = 128):
    return pallas_impl(x, y, block_m=block_m)


@kernel_registry.register("my_kernel", Platform.XLA, Backend.ANY, priority=0)
def my_kernel_xla(x, y):
    # Tuning parameters (block_m, num_warps) are in _TUNING_PARAM_NAMES, so
    # XLA does not need to declare them to pass validate_signatures().
    return xla_impl(x, y)
```

### Dynamic Runtime Registration

Use `kernel_registry.register(...)` directly rather than mutating internal attributes:

```python
def register_dynamic_kernel(algorithm: str, impl, platform: Platform, backend: Backend, priority: int = 0):
    return kernel_registry.register(algorithm, platform, backend, priority=priority)(impl)
```

---

## Error Handling Summary

| Condition | Exception / Signal |
|---|---|
| Invalid `platform` or `backend` string (including `"auto"` passed directly to `get()`) | `ValueError` |
| Algorithm not found, or no matching `(platform, backend)` implementation in `get()` | `ValueError` |
| Non-default value passed to a `del <param>` parameter (not in `_TUNING_PARAM_NAMES`) | `EjkernelRuntimeError` |
| `__ejkernel_unsupported__` returns/yields one or more reasons | `EjkernelRuntimeError` |
| Underlying kernel raises `NotImplementedError`/`ValueError` containing `"not supported"` / `"unsupported"` | Wrapped as `EjkernelRuntimeError` |
| Cross-backend signature mismatch in `validate_signatures()` | `UserWarning` (returns `False`) |

---

## Testing Patterns

### Signature Consistency Check

```python
from ejkernel import kernel_registry


def test_signature_consistency():
    for algo in kernel_registry.list_algorithms():
        assert kernel_registry.validate_signatures(algo, verbose=True)
```

### Cross-Backend Parity Against XLA Reference

```python
from ejkernel import Backend, Platform, kernel_registry


def test_kernel_parity(algorithm: str, inputs):
    xla_impl = kernel_registry.get(algorithm, platform=Platform.XLA, backend=Backend.ANY)
    ref = xla_impl(*inputs)

    for spec in kernel_registry.list_implementations(algorithm):
        if spec.platform == Platform.XLA:
            continue
        out = spec.implementation(*inputs)
        # Compare numerically against the XLA reference on the target accelerator
```
