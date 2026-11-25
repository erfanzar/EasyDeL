# ejkernel.ops - Kernel Framework and Configuration Management

## Overview

The `ejkernel.ops` package provides the core execution framework for implementing high-performance JAX operations with automatic configuration management, multi-tier caching, autotuning, shipped pre-tuned lookup tables, custom gradients, and distributed execution. It is the foundation upon which `ejkernel.modules` operations are built.

**Key Features:**

- Abstract `Kernel[Cfg, Out]` interface with hardware-platform and execution-context dispatch (`gpu` / `tpu` / `cpu`, `shard_map`)
- 7-tier `ConfigSelectorChain` configuration selection and caching
- Provenance-enforced disk persistence (`PersistentCache` in `~/ejkernel-presistent-cache/`)
- Shipped SQLite pre-tuned kernel table (`ejkernel.ops.tuned` / `tuned_kernels.db`)
- Online, recorded, and offline-HLO autotuning with optional backward-pass validation
- Batch processing utilities (`vmap_with_config`, `pmap_with_config`) and `shard_map` integration
- Transparent custom gradient wiring via `fwd_with_residuals()` + `vjp()`

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                                User Code                                  │
│         flash_attention(...)  or  executor(kernel, *args, **kwargs)       │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            Executor[Cfg, Out]                             │
│  kernel.prepare() ─► Invocation ─► Config selection ─► custom_vjp / run   │
│  (plus optional shard_map wrapper, HLO stamping, and invocation recording)│
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│     ConfigSelectorChain (7-tier)  │   │          Kernel[Cfg, Out]         │
│                                   │   │                                   │
│  1. Override (inv.override_cfg)   │   │  - op_id, version, key_builder    │
│  2. Overlay  (overlay_cache)      │   │  - prepare(*args, **kwargs)       │
│  3. Memory   (ConfigCache)        │   │  - run(*args, cfg, **kwargs)      │
│  4. Persistent (PersistentCache)  │   │  - heuristic_cfg*(inv)            │
│  5. Autotune (Tuner + candidates) │   │  - candidate_cfgs*(inv)           │
│  6. Heuristics (heuristic_cfg*)   │   │  - fwd_with_residuals() + vjp()   │
│  7. Error (RuntimeError)          │   │  - create_shard_map_wrapper()     │
└───────────────────────────────────┘   │  - get_impl(cfg) [modules pattern]│
                                        └───────────────────────────────────┘
```

---

## `Kernel[Cfg, Out]`: The Building Block

`Kernel[Cfg, Out]` (in `ejkernel/ops/core/kernel.py`) is the generic base class for configurable operations.

**Class attributes and constructor:**

- `op_id: str` — unique operation identifier used in cache keys and profiling labels. Set as a class attribute, passed to `super().__init__(op_id="...")`, or auto-generated from `"{module}.{ClassName}"`.
- `version: str = "0"` — version tag appended to cache keys as `f"{op_id}@v{version}"`. Bump this to invalidate stale cached configurations when implementation semantics change.
- `key_builder: Callable[[Invocation[Cfg, Out]], str] | None = None` — optional custom cache-key function (e.g., `default_key_builder_with_sharding`).

**Required overrides:**

1. `run(self, *args, cfg: Cfg, **kwargs) -> Out` — executes the operation using `cfg`.
2. `heuristic_cfg(self, inv: Invocation[Cfg, Out]) -> Cfg` — returns a safe, always-valid default configuration for `inv`.

**Optional overrides:**

- `prepare(self, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]` — preprocesses/validates arguments before creating `Invocation`.
- `candidate_cfgs(self, inv: Invocation[Cfg, Out]) -> Iterable[Cfg]` — yields candidate configs for autotuning (defaults to `[self.heuristic_cfg(inv)]`).
- `fwd_with_residuals(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]` and `vjp(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs)` — custom backward pass pair.
- `create_shard_map_wrapper(...)` — wraps execution in `jax.shard_map` when called with `method="shard_map"`.

### Basic Kernel Implementation

```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from ejkernel.ops import Invocation, Kernel


@dataclass(frozen=True)
class MatMulConfig:
    precision: str = "default"
    transpose_a: bool = False


class MatMulKernel(Kernel[MatMulConfig, jax.Array]):
    version = "1"

    def __init__(self):
        super().__init__(op_id="custom_matmul")

    def prepare(self, a, b, **kwargs):
        return (jnp.asarray(a), jnp.asarray(b)), kwargs

    def run(self, a, b, *, cfg: MatMulConfig, **kwargs) -> jax.Array:
        if cfg.transpose_a:
            a = a.T
        return jnp.dot(a, b, precision=cfg.precision)

    def heuristic_cfg(self, inv: Invocation[MatMulConfig, jax.Array]) -> MatMulConfig:
        a = inv.args[0]
        precision = "bfloat16" if a.dtype == jnp.bfloat16 else "default"
        return MatMulConfig(precision=precision)

    def candidate_cfgs(self, inv: Invocation[MatMulConfig, jax.Array]):
        return [
            MatMulConfig(precision="default"),
            MatMulConfig(precision="bfloat16"),
            MatMulConfig(precision="highest"),
        ]
```

### Hardware-Platform and Context Method Dispatch

`Executor` and `ConfigSelectorChain` resolve specialized `Kernel` methods at runtime via `_get_platform_method(kernel, method_name, platform, context)`, where `platform` is the active **hardware** platform from `get_device_platform()` (`"gpu"`, `"tpu"`, or `"cpu"`) and `context` is `"shard_map"` (when `method="shard_map"`) or `None`.

Resolution priority (highest to lowest):

| Priority | Pattern | Examples |
|---|---|---|
| 1 | `{method}_{context}_{platform}` | `run_shard_map_gpu`, `candidate_cfgs_shard_map_gpu`, `heuristic_cfg_shard_map_tpu` |
| 2 | `{method}_{context}` | `run_shard_map`, `fwd_with_residuals_shard_map` + `vjp_shard_map` |
| 3 | `{method}_{platform}` | `run_gpu`, `heuristic_cfg_gpu`, `heuristic_cfg_tpu`, `candidate_cfgs_gpu`, `candidate_cfgs_tpu` |
| 4 | `{method}` | `run`, `heuristic_cfg`, `candidate_cfgs`, `fwd_with_residuals` + `vjp` |

> **Note:** `Kernel` does **not** dispatch to `run_triton()`, `run_pallas()`, or `run_xla()`. Hardware-platform suffixes (`_gpu`, `_tpu`, `_cpu`) specialize `run`, `heuristic_cfg`, `candidate_cfgs`, `fwd_with_residuals`, and `vjp`. Dispatch across implementation frameworks (`Platform.TRITON`, `Platform.PALLAS`, `Platform.CUDA`, `Platform.CUTE`, `Platform.TILELANG`, `Platform.XLA`) is handled through the kernel registry via `get_impl(cfg)`.

### Dispatching to Registered Backends with `get_impl(cfg)`

Operations in `ejkernel.modules` follow the `get_impl(cfg)` pattern: `cfg` carries a `platform` and `backend`, `get_impl(cfg)` resolves the concrete implementation from `kernel_registry` via `detect_platform`, and `run()` invokes it:

```python
from dataclasses import dataclass
from typing import Literal
import jax
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules.base import detect_platform
from ejkernel.ops import FwdParams, Invocation, Kernel


@dataclass
class MyAttentionConfig:
    fwd_params: FwdParams | None = None
    platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] = "auto"
    backend: str = "any"


class MyAttention(Kernel[MyAttentionConfig, jax.Array]):
    def __init__(self):
        super().__init__(op_id="flash_attention")

    def get_impl(self, cfg: MyAttentionConfig):
        platform = detect_platform(self.op_id, cfg.platform)
        backend = Backend.ANY if platform == Platform.XLA else cfg.backend
        return kernel_registry.get(algorithm=self.op_id, platform=platform, backend=backend)

    def run(self, query, key, value, *, causal: bool = False, cfg: MyAttentionConfig) -> jax.Array:
        impl = self.get_impl(cfg)
        return impl(query=query, key=key, value=value, causal=causal, fwd_params=cfg.fwd_params)

    def heuristic_cfg(self, inv: Invocation[MyAttentionConfig, jax.Array]) -> MyAttentionConfig:
        return MyAttentionConfig(fwd_params=FwdParams(q_blocksize=128, kv_blocksize=128), platform="auto")

    def heuristic_cfg_gpu(self, inv: Invocation[MyAttentionConfig, jax.Array]) -> MyAttentionConfig:
        return MyAttentionConfig(
            fwd_params=FwdParams(q_blocksize=64, kv_blocksize=64, num_warps=4, num_stages=2),
            platform="auto",
            backend="gpu",
        )

    def candidate_cfgs_gpu(self, inv: Invocation[MyAttentionConfig, jax.Array]):
        return [
            MyAttentionConfig(
                fwd_params=FwdParams(q_blocksize=qb, kv_blocksize=kb, num_warps=4, num_stages=2),
                platform="triton",
                backend="gpu",
            )
            for qb, kb in ((64, 64), (128, 64), (128, 128))
        ]
```

### Custom Gradients (`fwd_with_residuals` + `vjp`)

When a `Kernel` overrides **both** `fwd_with_residuals` and `vjp` (or a matching platform/context pair such as `fwd_with_residuals_gpu` + `vjp_gpu`), `Executor` detects this via `_has_custom_vjp` and automatically wraps the call in `jax.custom_vjp`:

```python
class CustomDotKernel(Kernel[MatMulConfig, jax.Array]):
    def __init__(self):
        super().__init__(op_id="custom_dot")

    def run(self, x, y, *, cfg: MatMulConfig, **kwargs) -> jax.Array:
        return jnp.dot(x, y, precision=cfg.precision)

    def heuristic_cfg(self, inv: Invocation[MatMulConfig, jax.Array]) -> MatMulConfig:
        return MatMulConfig()

    def fwd_with_residuals(self, x, y, *, cfg: MatMulConfig, **kwargs):
        out = self.run(x, y, cfg=cfg, **kwargs)
        return out, (x, y)

    def vjp(self, residuals, y_out, dy, *args, cfg: MatMulConfig, **kwargs):
        x, y = residuals
        dx = jnp.dot(dy, y.T, precision=cfg.precision)
        dy_in = jnp.dot(x.T, dy, precision=cfg.precision)
        # Return one gradient per positional arg (None for non-differentiated args)
        return (dx, dy_in)
```

> **Note:** `vjp()` must return a tuple with one gradient per positional argument (`args`); gradients for array-valued `kwargs` are zero-filled automatically by `Executor`. When `method="shard_map"`, `Executor` invokes `create_shard_map_wrapper` directly and bypasses this automatic `custom_vjp` wrapper.

### Distributed Execution (`create_shard_map_wrapper`)

`Kernel.create_shard_map_wrapper(fn, *args, mesh, in_specs, out_specs, check_vma=False, **kwargs)` is a base helper that binds `kwargs` and wraps `fn` with `jax.shard_map`, returning `(shard_map_fn, call_args)`.

When `Executor` is called with `method="shard_map"`, it invokes:

```python
kernel.create_shard_map_wrapper(
    *args2,
    mesh=mesh,
    in_specs=in_specs,
    out_specs=out_specs,
    check_vma=check_vma,
    cfg=chosen,
    **kwargs2,
)
```

Operations that support `method="shard_map"` (such as `FlashAttention`) override `create_shard_map_wrapper` to accept `cfg` and return either `(shard_map_fn, call_args)` or `(shard_map_fn, call_args, post_callback)`.

---

## `Invocation`: Capturing Call Context

`Invocation[Cfg, Out]` is an immutable (`@dataclass(frozen=True)`) snapshot created by `Executor` after `kernel.prepare(*args, **kwargs)`.

```python
from ejkernel.ops import Invocation, device_fingerprint

inv = Invocation(
    op_id="custom_matmul",
    args=(a, b),
    kwargs={"scale": 1.0},
    batch_axes=None,       # Optional Mapping[str, int]
    override_cfg=None,     # Optional Cfg bypassing cache/autotune
    stamp=True,            # Whether Executor may attach HLO profiling labels
    method=None,           # None or "shard_map"
    mesh=None,             # jax.sharding.Mesh when method="shard_map"
    in_specs=None,         # tuple[PartitionSpec, ...] when method="shard_map"
    out_specs=None,        # PartitionSpec when method="shard_map"
    check_vma=False,       # Passed to jax.shard_map
)

# 16-char SHA-256 hex digest over abstractify(args), abstractify(kwargs), batch_axes, method
call_key = inv.call_key

# Or with a custom key builder (e.g., default_key_builder_with_sharding)
custom_key = inv.make_key(key_builder=kernel.key_builder)

# Device identity is queried via the standalone helper, not an Invocation field
dev = device_fingerprint()
```

Array values are stripped via `abstractify()` (keeping only shape and dtype), so arrays with identical structure share the same `call_key`.

---

## `Executor`: Running Kernels

`Executor[Cfg, Out]` orchestrates preprocessing, config selection, custom VJP wiring, profiling stamps, invocation recording, and execution:

```python
from ejkernel.ops import ConfigCache, ConfigSelectorChain, Executor

cache = ConfigCache()
selector = ConfigSelectorChain(cache=cache)
executor = Executor(selector, stamp_prefix="ejkernel_ops")

# Execute with automatic config selection
result = executor(my_kernel, a, b)

# Execute with an explicit configuration override (legacy `_cfg=...` is also accepted)
result = executor(my_kernel, a, b, cfg=my_config)

# Inspect the selected config without running the kernel
chosen_cfg = executor.choose_config(my_kernel, a, b)

# Pre-select a config for example inputs and return a jax.jit-compiled callable
compiled_fn = executor.compile(my_kernel, a, b)
result = compiled_fn(a_real, b_real)
```

**Execution notes:**

- **Heuristics-only fast path:** When `selector.policy.cache_miss_fallback == "heuristics"`, `Executor` uses `_choose_heuristics_only()` (checking override → overlay → memory cache → persistent cache → `heuristic_cfg*()`), skipping autotuning completely.
- **Config coercion & NVIDIA platform preference:** After selection, `Executor` coerces dict-serialized configs back to the kernel's dataclass (`_coerce_cfg`) and runs `_prefer_cuda_cfg`, which upgrades `"auto"` / `"triton"` configs to `"cute"` or `"cuda"` on NVIDIA GPUs when no explicit `cfg` override or explicit `platform` kwarg was passed and a CuTe/CUDA implementation is registered.
- **HLO labeling (`EJKERNEL_OPS_STAMP`):** When `stamp=True`, `EJKERNEL_OPS_STAMP` controls label injection (`"none"` default, `"hash"` for compact `ejkernel_ops#<op_id>@v<ver>:<call_key>`, `"json"` for full JSON payload).
- **Invocation recording (`EJKERNEL_OPS_RECORD=1`):** Records `(device_fingerprint(), op_id_v, call_key, kernel, args, kwargs)` into the in-memory registry for batch or offline autotuning.

---

## Configuration Selection and Caching

### `ConfigSelectorChain` (7-Tier Selection Order)

`ConfigSelectorChain.choose(inv, kernel)` resolves a configuration in strict priority order:

1. **Override** — `inv.override_cfg` (passed via `cfg=` / `_cfg=`). Cached in memory only (never persisted to disk).
2. **Overlay** — active `overlay_cache(...)` stack, checked in reverse order (most-recently-pushed wins).
3. **In-Memory Cache** — `ConfigCache.get(dev, op_id_v, call_key)`.
4. **Persistent Cache** — `PersistentCache.get(dev, op_id_v, call_key)` (trusted provenance only); on hit, promotes into `ConfigCache`.
5. **Autotune** — if `policy.cache_miss_fallback == "autotune"` and `policy.allow_autotune` is `True`, resolves `candidate_cfgs*()` via `_get_platform_method`, benchmarks them with `Tuner`, caches the winner in memory, and writes it to `PersistentCache` with `provenance=PROVENANCE_AUTOTUNE` (when `persist_autotune=True`). Respects `forbid_reautotune=True` to prevent repeated autotuning of the same key.
6. **Heuristics** — if `policy.allow_heuristics` is `True` (or all autotune candidates failed), resolves `heuristic_cfg*()` via `_get_platform_method` and stores it in memory cache only.
7. **Error** — raises `RuntimeError` if every tier is unavailable or disabled.

```python
import os
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    PersistentCache,
    Tuner,
)

selector = ConfigSelectorChain(
    cache=ConfigCache(),
    policy=AutotunePolicy(
        allow_autotune=True,
        allow_heuristics=True,
        cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
        validate_backward=False,
    ),
    tuner=Tuner(warmup=1, iters=3),
    persistent=PersistentCache("my-op", cfg_type=MatMulConfig),
    persist_autotune=True,
    forbid_reautotune=True,
)
executor = Executor(selector)
```

### `ConfigCache` (In-Memory)

Keyed by `(device_fingerprint, op_id_v, call_key)`:

```python
from ejkernel.ops import ConfigCache

cache = ConfigCache()
cache.put(dev, "custom_matmul@v1", call_key, cfg)
hit = cache.get(dev, "custom_matmul@v1", call_key)  # None on miss
n = cache.size()
all_keys = cache.keys()
cache.clear()
```

### `PersistentCache` (Disk-Based, Provenance-Enforced)

`PersistentCache` stores autotuned configurations as atomic JSON files keyed by `"{device}|{op_id}|{call_key}"`.

**Default path resolution** (when `path=None`):

1. `$EJKERNEL_PERSISTENT_CACHE_DIR/{opname}.json` (if `EJKERNEL_PERSISTENT_CACHE_DIR` is set)
2. `~/ejkernel-presistent-cache/{opname}.json` (default)
3. `./.ejkernel-presistent-cache/{opname}.json` (fallback if home directory is unwritable)
4. `$TMPDIR/ejkernel-presistent-cache/{opname}.json` (last resort; silently disables disk I/O if none is writable)

**Provenance & typed reconstruction:**

- `put()` requires an explicit keyword-only `provenance` argument and wraps each entry as `{"__ejk_provenance__": provenance, "__ejk_cfg_class__": "ejkernel....:CfgClass", "cfg": ...}`.
- `get()` only returns entries whose provenance is in `TRUSTED_PROVENANCE` (currently `{PROVENANCE_AUTOTUNE}`, where `PROVENANCE_AUTOTUNE = "autotune"`). Legacy pre-provenance entries and unmeasured entries are treated as cache misses so shared per-host files cannot be poisoned by heuristic defaults or explicit caller overrides.
- On read, JSON lists inside config dicts are converted back to tuples and reconstructed via `loader`, `cfg_type`, the recorded `ejkernel.*` dataclass/Pydantic class (`__ejk_cfg_class__`), or `argparse.Namespace`.
- Writes are atomic (temporary file in the target directory + `os.replace`).

```python
from ejkernel.ops import PROVENANCE_AUTOTUNE, PersistentCache

persistent = PersistentCache("my-op", cfg_type=MatMulConfig)
persistent.put(dev, "custom_matmul@v1", call_key, best_cfg, provenance=PROVENANCE_AUTOTUNE)
loaded = persistent.get(dev, "custom_matmul@v1", call_key)
```

### Temporary Overrides and Progress Context Managers

```python
from ejkernel.ops import forward_autotune_only, overlay_cache, policy_override
from ejkernel.ops.config import log_autotune_progress, set_autotune_progress

# Temporarily override specific (device, op_id_v, call_key) entries in the selector chain
with overlay_cache({(dev, "custom_matmul@v1", call_key): custom_cfg}):
    result = executor(my_kernel, a, b)

# Temporarily change AutotunePolicy fields on a selector
with policy_override(selector, allow_autotune=False, cache_miss_fallback="heuristics"):
    result = executor(my_kernel, a, b)

# Time forward-only even when policy.validate_backward=True
with forward_autotune_only():
    cfg = selector.choose(inv, my_kernel)

# Show tqdm progress bars during autotuning (also enabled via EJKERNEL_AUTOTUNE_PROGRESS=1)
with log_autotune_progress():
    result = executor(my_kernel, a, b)

# Or toggle imperatively
set_autotune_progress(True)
```

Set `EJKERNEL_LOG_AUTOTUNE=1` to log each measured candidate and winning config during autotuning.

---

## Pre-Tuned Kernel Database (`ejkernel.ops.tuned`)

While `PersistentCache` caches per-machine autotuning results in `~/ejkernel-presistent-cache/`, `ejkernel.ops.tuned` ships a read-only SQLite table (`ejkernel/ops/tuned/tuned_kernels.db`, schema version `2`) inside the installed wheel so fresh accelerators start with measured winners immediately.

Each row maps `(kernel, device, dtypes, shape_key)` to:

- `platform` — winning backend (`"pallas"`, `"xla"`, `"triton"`, etc.)
- `config` — winning config dictionary
- `ms` — measured latency in milliseconds
- `runner_up` — next-best `{"platform", "config", "ms"}`
- `baseline` — untuned default `{"config", "ms"}` (rows are only kept when tuning beats default by `--min-gain`, default `1.05x`)
- `provenance` — sweep metadata (stored deduplicated in a `sources` table)

Dimensions in `shape_key` are bucketed down to powers of two via `bucket()` / `shape_signature()`, and `dtypes` are canonicalized via `dtype_signature()`. Lookups check for an exact key match first and, with `allow_nearest=True` (default), fall back to the nearest measured shape for the same `(kernel, device, dtypes)`.

### Python Lookup API

```python
from ejkernel.ops.tuned import (
    TunedEntry,
    TunedStore,
    current_device_kind,
    set_tuned_store,
    tuned_choice,
    tuned_store,
)

choice: TunedEntry | None = tuned_choice(
    "grouped_matmul",
    dtypes={"lhs": lhs.dtype, "rhs": rhs.dtype},
    shape={"m": m, "k": k, "n": n, "e": num_groups},
)
if choice is not None:
    platform, cfg = choice.platform, choice.config
    speedup = choice.speedup_over_default()

# Point lookups at an alternate database (or set EJKERNEL_TUNED_DB=/path/to/db)
set_tuned_store("/tmp/custom_tuned.db")
set_tuned_store(None)  # Restore the shipped tuned_kernels.db
```

### CLI (`python -m ejkernel.ops.tuned`)

```bash
# Summarize the shipped table (or pass --db PATH)
python -m ejkernel.ops.tuned stats

# Print rows as diffable TSV or JSON
python -m ejkernel.ops.tuned dump --kernel grouped_matmul
python -m ejkernel.ops.tuned dump --kernel grouped_matmul --json

# List registered sweeps and run a hardware sweep
python -m ejkernel.ops.tuned sweep --list
python -m ejkernel.ops.tuned sweep --kernel grouped_matmul --out /tmp/gmm.db --reps 30 --min-gain 1.05

# Merge sweep databases (keeps the faster measured row by default; pass --last-wins to override)
python -m ejkernel.ops.tuned merge --into libs/ejkernel/ejkernel/ops/tuned/tuned_kernels.db /tmp/gmm.db
```

---

## Autotuning and Benchmarking

### Candidate Generation on `Kernel`

Override `candidate_cfgs(self, inv)` — and optionally platform/context variants like `candidate_cfgs_gpu`, `candidate_cfgs_tpu`, or `candidate_cfgs_shard_map_gpu` — to define the tuning search space for `ConfigSelectorChain`:

```python
class MyKernel(Kernel[MyConfig, jax.Array]):
    def candidate_cfgs_gpu(self, inv: Invocation[MyConfig, jax.Array]):
        return [
            MyConfig(block_size=64, num_warps=2),
            MyConfig(block_size=128, num_warps=4),
            MyConfig(block_size=256, num_warps=8),
        ]
```

### Benchmarking and Standalone Function Autotuning

```python
from ejkernel.ops import Autotuner, autotune, benchmark

# Compile with jax.jit and return average seconds per iteration
seconds = benchmark(lambda x, y: x @ y, a, b, warmup=2, iters=10)

# Low-level candidate benchmarking via Autotuner
tuner = Autotuner(warmup=1, iters=3)
data = tuner.autotune(lambda c: (lambda x, y: my_kernel.run(x, y, cfg=c)), (a, b), {}, candidates)
best_cfg = data.fastest_config

# Standalone JAX function hyperparameter autotuning decorator (FNAutotuner)
@autotune(hyperparams={"block_size": [64, 128, 256]})
def custom_fn(x, y, block_size=128):
    return compute(x, y, block_size)

out = custom_fn(a, b)
print(custom_fn.optimal_hyperparams)
```

> **Note:** Built-in hardware kernels under `ejkernel/kernels/` never use `@autotune` or `@triton.autotune`; all tuning for registered kernels flows through `Executor`, `ConfigSelectorChain`, and `candidate_cfgs*()`.

### Recorded and Offline-HLO Autotuning

```python
import jax
from ejkernel.ops import autotune_recorded, device_fingerprint, get_invocations
from ejkernel.ops.execution import autotune_lowered

# 1. Run once with EJKERNEL_OPS_RECORD=1 (and EJKERNEL_OPS_STAMP=hash for offline HLO mode)
recorded = get_invocations(device_fingerprint())

# 2a. Autotune all recorded invocations for the current device
result = autotune_recorded(selector, show_progress=True)
with result:  # Applies tuned winners as an overlay_cache
    out = model(inputs)

# 2b. Or autotune only the operations embedded in a lowered JAX computation
lowered = jax.jit(model).lower(example_inputs)
result = autotune_lowered(selector, lowered)
with result:
    out = jax.jit(model)(real_inputs)
```

---

## Batch Processing

`vmap_with_config` and `pmap_with_config` select a configuration once using a representative slice (index `0`) and then apply `jax.vmap` / `jax.pmap` with that cached configuration:

```python
from ejkernel.ops import pmap_with_config, vmap_with_config

# Vectorized execution with shared config selection
vmapped = vmap_with_config(executor, my_kernel, in_axes=0)
batched_out = vmapped(batched_a, batched_b)

# Multi-device parallel execution with shared config selection
pmapped = pmap_with_config(executor, my_kernel, in_axes=0, axis_name="devices")
sharded_out = pmapped(sharded_a, sharded_b)
```

---

## Configuration Data Carriers

### `FwdParams` and `BwdParams`

Standard block-size and launch-parameter dataclasses (all fields optional, defaulting to `None`, hashable via `ejkernel.callib.hash_fn`):

```python
from ejkernel.ops import BwdParams, FwdParams

fwd = FwdParams(
    blocksize_m=128,
    blocksize_k=64,
    blocksize_n=128,
    q_blocksize=128,
    kv_blocksize=256,
    blocksize_heads=None,
    blocksize_keys=None,
    num_key_splits=None,
    num_warps=4,
    num_stages=2,
)

bwd = BwdParams(
    blocksize_m=64,
    blocksize_k=64,
    blocksize_n=64,
    q_blocksize=64,
    kv_blocksize=128,
    num_warps=4,
    num_stages=2,
)
```

---

## Utility Functions

### Device Detection and Fingerprinting

```python
from ejkernel.ops import device_fingerprint, device_kind, get_device_platform

device_kind()          # Default device kind, e.g. "NVIDIA H100 80GB HBM3", "TPU v5p", "cpu"
get_device_platform()  # Hardware platform for Kernel method dispatch: "gpu", "tpu", "cpu", or "unknown"
device_fingerprint()   # Cache device key: "<device_kind>|<platform_version>"
```

### Sharding and Hashing Utilities

```python
from ejkernel.ops import (
    abstractify,
    default_key_builder_with_sharding,
    sharding_fingerprint,
    short_hash,
)

# Extract sharding repr for a JAX array (None for non-arrays)
s_fp = sharding_fingerprint(array)

# Assign as Kernel.key_builder to include input shardings in cache keys
key = default_key_builder_with_sharding(inv)

# Replace array leaves with ShapeDtypeStruct and hash deterministically
abstract = abstractify((a, b))
digest = short_hash(abstract)  # 16-char hex digest
```

### Serialization

```python
from ejkernel.ops import from_json, stable_json, to_json

json_str = to_json(config)      # Dataclass-aware JSON string
raw_obj = from_json(json_str)   # Thin wrapper over json.loads() -> plain Python dict/list/primitives
stable = stable_json(obj)       # Deterministic JSON handling JAX arrays, dtypes, dataclasses, Pydantic, callables
```

### HLO Label Extraction

```python
import jax
from ejkernel.ops import (
    extract_labels_from_hlo_text,
    find_labels_in_lowered,
    label,
    labels_to_configs,
)

# Build a standardized HLO label string: "ejkernel_ops#custom_matmul@v1:<16-hex>"
op_label = label("custom_matmul@v1", call_key)

# Extract embedded labels from a lowered computation or raw HLO text
lowered = jax.jit(fn).lower(a, b)
labels = find_labels_in_lowered(lowered)
labels_from_text = extract_labels_from_hlo_text(hlo_text)

# Resolve each label in `lowered` back to its cached config in `selector`
label_cfg_pairs = labels_to_configs(lowered, selector)
```

---

## Type Variables

| Type Variable | Description |
|---|---|
| `Cfg` | Configuration type (e.g., `FlashAttentionConfig`, `MatMulConfig`) |
| `Out` | Kernel output type (e.g., `jax.Array` or `tuple[jax.Array, jax.Array]`) |

---

## Module Structure

```text
ejkernel/ops/
├── __init__.py               # Public API exports
├── core/
│   ├── __init__.py
│   ├── kernel.py             # Kernel, Invocation, _has_custom_vjp, _get_platform_method
│   └── types.py              # Cfg, Out type variables
├── config/
│   ├── __init__.py
│   ├── cache.py              # ConfigCache, overlay_cache
│   ├── persistent.py         # PersistentCache, PROVENANCE_AUTOTUNE, TRUSTED_PROVENANCE
│   └── selection.py          # ConfigSelectorChain, AutotunePolicy, Tuner, policy_override,
│                             # forward_autotune_only, log_autotune_progress, set_autotune_progress
├── execution/
│   ├── __init__.py
│   ├── executor.py           # Executor, ConfigChooser
│   ├── batch.py              # vmap_with_config, pmap_with_config
│   ├── offline.py            # autotune_lowered
│   ├── profiler.py           # XPlane Profiler for FNAutotuner
│   └── tuning.py             # Autotuner, FNAutotuner, autotune, autotune_recorded, benchmark
├── tuned/
│   ├── __init__.py           # Public exports for shipped pre-tuned kernel table
│   ├── __main__.py           # CLI: python -m ejkernel.ops.tuned {stats,dump,sweep,merge}
│   ├── _lookup.py            # tuned_choice, tuned_store, set_tuned_store, current_device_kind
│   ├── _store.py             # TunedStore, TunedEntry, bucket, dtype_signature, shape_signature, merge
│   ├── _sweep.py             # SweepSpec, SweepPoint, Candidate, run_sweep
│   ├── _sweeps/              # Registered sweep specifications
│   └── tuned_kernels.db      # Shipped SQLite database of pre-tuned winners
├── registry.py               # record_invocation, get_invocations
└── utils/
    ├── __init__.py
    ├── datacarrier.py        # FwdParams, BwdParams
    ├── fingerprint.py        # device_fingerprint, device_kind, get_device_platform,
    │                         # sharding_fingerprint, abstractify, short_hash, stable_json
    ├── meta.py               # label, find_labels_in_lowered, extract_labels_from_hlo_text, labels_to_configs
    └── serialize.py          # to_json, from_json
```
