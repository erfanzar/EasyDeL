# Ops System Architecture Analysis

## Overview

The ops system provides the core infrastructure for kernel execution in ejKernel. It implements a sophisticated multi-tier configuration management system with automatic optimization, caching, and device-aware execution. This system orchestrates the complete lifecycle of kernel invocation from configuration selection to execution with custom gradients.

## Architecture Components

```md
ejkernel/ops/
├── config/           # Configuration management
│   ├── selection.py  # Multi-tier config selection
│   ├── cache.py      # In-memory caching
│   └── persistent.py # Disk-based persistence
├── core/             # Base kernel classes
│   └── kernel.py     # Kernel base class and invocation
├── execution/        # Execution orchestration
│   └── executor.py   # Main executor implementation
└── utils/            # Utilities
    ├── fingerprint.py    # Device fingerprinting
    └── datacarrier.py    # Configuration data structures
```

## Core Components

### 1. Kernel Base Class

The `Kernel[Cfg, Out]` generic class provides the foundation for all operations:

```python
class Kernel(Generic[Cfg, Out]):
    """Base class for all kernel operations"""

    def __init__(self, op_id: str):
        self.op_id = op_id

    # Required methods
    def run(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Core operation execution with configuration"""
        raise NotImplementedError

    def heuristic_cfg(self, inv: Invocation[Cfg, Out]) -> Cfg:
        """Default configuration for given invocation"""
        raise NotImplementedError

    # Optional methods
    def prepare(self, *args, **kwargs) -> tuple[tuple, dict]:
        """Preprocess arguments before execution"""
        return args, kwargs

    def candidate_cfgs(self, inv: Invocation[Cfg, Out]) -> Iterable[Cfg]:
        """Configurations for autotuning"""
        return []

    # Custom gradient support
    def fwd_with_residuals(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass with residuals for custom VJP"""
        output = self.run(*args, cfg=cfg, **kwargs)
        return output, None

    def vjp(self, residuals: Any, y: Out, dy: Any, *args, cfg: Cfg, **kwargs):
        """Backward pass for custom gradients"""
        raise NotImplementedError
```

### 2. Platform-Specific Method Dispatch

The system supports hierarchical method dispatch for platform-specific optimizations:

```python
# Method resolution order (most specific to least specific):
# 1. run_shard_map_gpu  (context + platform)
# 2. run_shard_map      (context only)
# 3. run_gpu            (platform only)
# 4. run               (generic fallback)

class MyKernel(Kernel):
    def run(self, x, cfg):
        """Generic implementation"""
        return generic_impl(x, cfg)

    def run_gpu(self, x, cfg):
        """GPU-optimized implementation"""
        return gpu_optimized_impl(x, cfg)

    def run_shard_map(self, x, cfg):
        """Distributed implementation"""
        return distributed_impl(x, cfg)

    def run_shard_map_gpu(self, x, cfg):
        """Distributed GPU-optimized implementation"""
        return distributed_gpu_impl(x, cfg)
```

### 3. Invocation Dataclass

Captures complete execution context:

```python
@dataclasses.dataclass(frozen=True)
class Invocation(Generic[Cfg, Out]):
    """Complete execution context for a kernel invocation"""

    # Core identification
    op_id: str                                    # Kernel identifier
    args: tuple[Any, ...]                        # Positional arguments
    kwargs: Mapping[str, Any]                    # Keyword arguments

    # Configuration
    override_cfg: Cfg | None = None              # Explicit configuration override

    # Execution context
    method: str | None = None                    # e.g., "shard_map"
    stamp: bool = True                           # Enable profiling metadata
    batch_axes: Mapping[str, int] | None = None  # Batching information

    # Distributed execution
    mesh: jax.sharding.Mesh | None = None
    in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None
    out_specs: jax.sharding.PartitionSpec | None = None
    check_vma: bool = False                      # Verify memory alignment

    @property
    def call_key(self) -> str:
        """16-character hash for caching based on arg shapes/types"""
        return short_hash(abstractify((self.args, self.kwargs)))

    @property
    def versioned_op_id(self) -> str:
        """op_id with version suffix for cache invalidation"""
        return f"{self.op_id}@v1"
```

## Configuration Selection System

### ConfigSelectorChain

Implements a sophisticated 7-tier fallback hierarchy for configuration selection:

```python
class ConfigSelectorChain(Generic[Cfg]):
    """Multi-tier configuration selection with fallback"""

    def __init__(self,
                 cache: ConfigCache[Cfg] | None = None,
                 policy: AutotunePolicy | None = None,
                 tuner: Tuner[Cfg] | None = None,
                 persistent: PersistentCache[Cfg] | None = None,
                 overlay: dict | None = None):
        self.cache = cache or ConfigCache()
        self.policy = policy or AutotunePolicy()
        self.tuner = tuner or Tuner()
        self.persistent = persistent
        self.overlay = overlay or {}

    def choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """
        Configuration selection priority:
        1. Override: Explicit cfg in invocation
        2. Overlay: Temporary context-specific configurations
        3. Memory Cache: Fast in-memory lookup
        4. Persistent Cache: Disk-based storage
        5. Autotune: Benchmark candidate configurations
        6. Heuristics: Kernel-provided defaults
        7. Error: No configuration available
        """
```

### Selection Flow

```python
def choose(self, inv: Invocation, kernel: Kernel) -> Cfg:
    device = device_fingerprint()

    # 1. Override - highest priority
    if inv.override_cfg is not None:
        return inv.override_cfg

    # 2. Overlay - temporary overrides
    cache_key = (device, inv.versioned_op_id, inv.call_key)
    if cache_key in self.overlay:
        return self.overlay[cache_key]

    # 3. Memory Cache - fast lookup
    if self.cache:
        cfg = self.cache.get(device, inv.versioned_op_id, inv.call_key)
        if cfg is not None:
            return cfg

    # 4. Persistent Cache - disk storage
    if self.persistent:
        cfg = self.persistent.get(device, inv.versioned_op_id, inv.call_key)
        if cfg is not None:
            self.cache.put(device, inv.versioned_op_id, inv.call_key, cfg)
            return cfg

    # 5. Autotune - benchmark candidates
    if self.policy.allow_autotune:
        candidates = get_candidates(kernel, inv, get_device_platform())
        if candidates:
            best_cfg = self.tuner.autotune(make_fn, args, kwargs, candidates)
            # Cache the result
            self.cache.put(device, inv.versioned_op_id, inv.call_key, best_cfg)
            if self.persistent:
                self.persistent.put(device, inv.versioned_op_id, inv.call_key, best_cfg)
            return best_cfg

    # 6. Heuristics - default configuration
    if self.policy.allow_heuristics:
        return kernel.heuristic_cfg(inv)

    # 7. Error - no configuration available
    raise ValueError(f"No configuration available for {inv.op_id}")
```

## Tuner Class

Sophisticated performance benchmarking system:

```python
class Tuner(Generic[Cfg]):
    """Performance benchmarking for configuration selection"""

    def __init__(self, warmup: int = 5, iters: int = 100):
        self.warmup = warmup
        self.iters = iters

    def measure(self, fn: Callable, *args, **kwargs) -> float:
        """
        Measure execution time with proper handling of JAX specifics:
        - Deep-flattens args/kwargs to separate arrays from constants
        - Handles JAX tracers by converting to concrete arrays
        - Supports backward pass validation
        - Falls back gracefully for non-transformable functions
        """
        # Flatten to separate arrays from constants
        leaves, treedef = tree_flatten((args, kwargs))
        array_leaves = [x for x in leaves if isinstance(x, Array)]

        # Convert tracers to concrete arrays
        concrete_arrays = [np.array(x) if hasattr(x, '__jax_array__')
                          else x for x in array_leaves]

        # Rebuild args/kwargs with concrete arrays
        concrete_leaves = [concrete_arrays.pop(0) if isinstance(x, Array)
                          else x for x in leaves]
        args, kwargs = tree_unflatten(treedef, concrete_leaves)

        # JIT compile if possible
        try:
            jitted_fn = jax.jit(fn)
        except:
            jitted_fn = fn

        # Warmup runs
        for _ in range(self.warmup):
            jitted_fn(*args, **kwargs).block_until_ready()

        # Timed runs
        start = time.perf_counter()
        for _ in range(self.iters):
            jitted_fn(*args, **kwargs).block_until_ready()
        end = time.perf_counter()

        return (end - start) / self.iters

    def autotune(self, make_fn: Callable[[Cfg], Callable],
                 args: tuple, kwargs: dict,
                 candidates: list[Cfg]) -> Cfg:
        """Benchmark all candidates and return the fastest"""
        best_cfg = None
        best_time = float('inf')

        for cfg in candidates:
            try:
                fn = make_fn(cfg)
                time = self.measure(fn, *args, **kwargs)

                if time < best_time:
                    best_time = time
                    best_cfg = cfg

                if os.getenv("EJKERNEL_LOG_AUTOTUNE"):
                    print(f"Config {cfg}: {time:.6f}s")
            except Exception as e:
                if os.getenv("EJKERNEL_LOG_AUTOTUNE"):
                    print(f"Config {cfg} failed: {e}")

        return best_cfg
```

### Backward Pass Validation

```python
def validate_backward(fn, args, kwargs):
    """Validate gradient computation"""
    if not kwargs.get("_ejk_validate_backward", False):
        return True

    try:
        # Extract array leaves for gradient computation
        leaves, _ = tree_flatten((args, kwargs))
        array_indices = [i for i, x in enumerate(leaves)
                        if isinstance(x, Array)]

        # Create value_and_grad function
        def wrapped(*array_args):
            # Rebuild full args/kwargs
            full_leaves = list(leaves)
            for i, idx in enumerate(array_indices):
                full_leaves[idx] = array_args[i]
            args, kwargs = tree_unflatten(treedef, full_leaves)
            return fn(*args, **kwargs).sum()

        value_and_grad_fn = jax.value_and_grad(wrapped, argnums=range(len(array_indices)))

        # Test gradient computation
        array_args = [leaves[i] for i in array_indices]
        _, grads = value_and_grad_fn(*array_args)

        return True
    except:
        return False
```

## Executor

The main orchestrator for kernel execution:

```python
class Executor(Generic[Cfg, Out]):
    """Orchestrates complete kernel execution pipeline"""

    def __init__(self, chooser: ConfigSelectorChain[Cfg]):
        self.chooser = chooser

    def __call__(self, kernel: Kernel[Cfg, Out], *args,
                 cfg: Cfg | None = None,
                 stamp: bool = True,
                 method: str | None = None,
                 mesh: Mesh | None = None,
                 in_specs: tuple | None = None,
                 out_specs: Any | None = None,
                 check_vma: bool = False,
                 **kwargs) -> Out:
        """
        Complete execution flow:
        1. Preprocess arguments via kernel.prepare()
        2. Create Invocation with metadata
        3. Select configuration via chooser.choose()
        4. Setup custom VJP if kernel implements it
        5. Add profiling metadata if stamp=True
        6. Execute with chosen configuration
        7. Record invocation if EJKERNEL_OPS_RECORD=1
        """
```

### Execution Flow

```python
def execute(self, kernel, *args, **kwargs):
    # 1. Preprocessing
    args, kwargs = kernel.prepare(*args, **kwargs)

    # 2. Create invocation context
    inv = Invocation(
        op_id=kernel.op_id,
        args=args,
        kwargs=kwargs,
        override_cfg=kwargs.pop("_cfg", None),
        method=method,
        stamp=stamp,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma
    )

    # 3. Configuration selection
    cfg = self.chooser.choose(inv, kernel)

    # 4. Determine execution method
    platform = get_device_platform()
    context = inv.method or "default"

    # Try methods in order of specificity
    method_name = f"run_{context}_{platform}"
    if not hasattr(kernel, method_name):
        method_name = f"run_{context}"
    if not hasattr(kernel, method_name):
        method_name = f"run_{platform}"
    if not hasattr(kernel, method_name):
        method_name = "run"

    # 5. Custom VJP setup
    if _has_custom_vjp(kernel, platform, context):
        runner = create_custom_vjp_wrapper(kernel, cfg, platform, context)
    else:
        runner = getattr(kernel, method_name)

    # 6. Add profiling metadata
    if stamp:
        runner = add_profiling_metadata(runner, inv, cfg)

    # 7. Execute
    output = runner(*args, cfg=cfg, **kwargs)

    # 8. Record invocation
    if os.getenv("EJKERNEL_OPS_RECORD"):
        record_invocation(inv, cfg, output)

    return output
```

### Custom VJP Integration

```python
def create_custom_vjp_wrapper(kernel, cfg, platform, context):
    """Create custom VJP wrapper for gradient computation"""

    @jax.custom_vjp
    def wrapped(*array_args, **array_kwargs):
        # Forward pass
        return kernel.fwd_with_residuals(*array_args, cfg=cfg, **array_kwargs)[0]

    def fwd(*array_args, **array_kwargs):
        output, residuals = kernel.fwd_with_residuals(*array_args, cfg=cfg, **array_kwargs)
        return output, (residuals, array_args, array_kwargs)

    def bwd(ctx, dy):
        residuals, array_args, array_kwargs = ctx
        grads = kernel.vjp(residuals, None, dy, *array_args, cfg=cfg, **array_kwargs)

        # Map gradients to correct positions
        grad_dict = {}
        for i, arg in enumerate(array_args):
            grad_dict[id(arg)] = grads[i] if i < len(grads) else None

        return tuple(grad_dict.get(id(arg), None) for arg in array_args)

    wrapped.defvjp(fwd, bwd)
    return wrapped
```

## Caching System

### In-Memory Cache

```python
class ConfigCache(Generic[Cfg]):
    """Thread-safe in-memory configuration cache"""

    def __init__(self):
        self._cache: dict[tuple[str, str, str], Cfg] = {}
        self._lock = threading.RLock()

    def get(self, dev: str, op_id: str, call_key: str) -> Cfg | None:
        """Thread-safe cache lookup"""
        with self._lock:
            return self._cache.get((dev, op_id, call_key))

    def put(self, dev: str, op_id: str, call_key: str, cfg: Cfg):
        """Thread-safe cache insertion"""
        with self._lock:
            self._cache[(dev, op_id, call_key)] = cfg

    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Number of cached entries"""
        with self._lock:
            return len(self._cache)
```

### Persistent Cache

```python
class PersistentCache(Generic[Cfg]):
    """Disk-based configuration persistence"""

    def __init__(self, opname: str, path: str | None = None,
                 loader: Callable | None = None,
                 dumper: Callable | None = None,
                 cfg_type: type | None = None):
        """
        Args:
            opname: Operation name for default path
            path: Custom cache file path
            loader: Custom deserialization function
            dumper: Custom serialization function
            cfg_type: Configuration type for automatic ser/deser
        """
        if path is None:
            # Default: ~/ejkernel-persistent-cache/{opname}.json
            cache_dir = Path.home() / "ejkernel-persistent-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.path = cache_dir / f"{opname}.json"
        else:
            self.path = Path(path)

        self.loader = loader or self._default_loader
        self.dumper = dumper or self._default_dumper
        self.cfg_type = cfg_type
        self._lock = threading.RLock()
        self._data = self._load()

    def _default_loader(self, data: dict) -> Cfg:
        """Default deserialization"""
        if self.cfg_type:
            if dataclasses.is_dataclass(self.cfg_type):
                return self.cfg_type(**data)
            elif hasattr(self.cfg_type, 'model_validate'):  # Pydantic
                return self.cfg_type.model_validate(data)
        return data

    def _default_dumper(self, cfg: Cfg) -> dict:
        """Default serialization"""
        if dataclasses.is_dataclass(cfg):
            return dataclasses.asdict(cfg)
        elif hasattr(cfg, 'model_dump'):  # Pydantic
            return cfg.model_dump()
        return cfg

    def _load(self) -> dict:
        """Load cache from disk"""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save(self):
        """Atomic save to disk"""
        with tempfile.NamedTemporaryFile('w', dir=self.path.parent,
                                         delete=False) as tmp:
            json.dump(self._data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, self.path)  # Atomic on POSIX

    def get(self, dev: str, op_id: str, call_key: str) -> Cfg | None:
        """Retrieve configuration from disk"""
        with self._lock:
            key = f"{dev}|{op_id}|{call_key}"
            data = self._data.get(key)
            if data is not None:
                return self.loader(data)
            return None

    def put(self, dev: str, op_id: str, call_key: str, cfg: Cfg):
        """Store configuration to disk"""
        with self._lock:
            key = f"{dev}|{op_id}|{call_key}"
            self._data[key] = self.dumper(cfg)
            self._save()
```

## Utility Functions

### Device Fingerprinting

```python
def device_fingerprint(dev: Device | None = None) -> str:
    """
    Generate unique device identifier

    Returns:
        'device_kind|platform_version'

    Examples:
        'gpu|cuda_12.0'
        'tpu|v4'
        'cpu|'
    """
    if dev is None:
        dev = jax.devices()[0]

    device_kind = dev.platform
    platform_version = ""

    if device_kind == "gpu":
        # Get CUDA/ROCm version
        platform_version = get_cuda_version()
    elif device_kind == "tpu":
        # Get TPU generation
        platform_version = get_tpu_version()

    return f"{device_kind}|{platform_version}"

def get_device_platform(dev: Device | None = None) -> str:
    """Get device platform: 'gpu', 'tpu', 'cpu', or 'unknown'"""
    if dev is None:
        dev = jax.devices()[0]
    return dev.platform
```

### Stable Hashing

```python
def abstractify(pytree: Any) -> Any:
    """Convert arrays to ShapeDtypeStruct for caching"""
    def _abstractify_leaf(x):
        if isinstance(x, Array):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        return x

    return tree_map(_abstractify_leaf, pytree)

def short_hash(obj: Any) -> str:
    """Generate 16-character hash for object"""
    json_str = stable_json(obj)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

def stable_json(obj: Any) -> str:
    """
    Deterministic JSON serialization

    Handles:
    - Functions: (module, name, source_position)
    - functools.partial: (func, args, kwargs)
    - Dataclasses: asdict conversion
    - Pydantic models: model_dump
    - JAX/NumPy arrays: (shape, dtype)
    - Primitives: direct serialization
    """
    if callable(obj):
        return json.dumps({
            "type": "function",
            "module": obj.__module__,
            "name": obj.__name__,
            "position": str(inspect.getsourcefile(obj))
        }, sort_keys=True)
    elif isinstance(obj, functools.partial):
        return json.dumps({
            "type": "partial",
            "func": stable_json(obj.func),
            "args": [stable_json(arg) for arg in obj.args],
            "kwargs": {k: stable_json(v) for k, v in obj.keywords.items()}
        }, sort_keys=True)
    # ... handle other types
    return json.dumps(obj, sort_keys=True)
```

### Configuration Data Carriers

```python
@dataclass
class FwdParams:
    """Forward pass configuration parameters"""
    # Block sizes
    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None

    # Thread configuration
    blocksize_heads: int | None = None
    blocksize_keys: int | None = None
    num_key_splits: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    def __hash__(self):
        """Custom hash for caching"""
        return hash(tuple(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
        ))

@dataclass
class BwdParams:
    """Backward pass configuration parameters"""
    # Typically smaller block sizes for memory efficiency
    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    def __hash__(self):
        """Custom hash for caching"""
        return hash(tuple(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
        ))
```

## Context Managers

### Policy Override

```python
@contextmanager
def policy_override(selector: ConfigSelectorChain,
                    allow_autotune: bool | None = None,
                    allow_heuristics: bool | None = None,
                    cache_miss_fallback: str | None = None):
    """Temporarily override autotuning policy"""
    old_policy = selector.policy
    new_policy = AutotunePolicy(
        allow_autotune=allow_autotune if allow_autotune is not None
                      else old_policy.allow_autotune,
        allow_heuristics=allow_heuristics if allow_heuristics is not None
                        else old_policy.allow_heuristics,
        cache_miss_fallback=cache_miss_fallback or old_policy.cache_miss_fallback
    )
    selector.policy = new_policy
    try:
        yield
    finally:
        selector.policy = old_policy
```

### Cache Overlay

```python
@contextmanager
def overlay_cache(overrides: dict):
    """Temporarily override cache entries"""
    old_overlay = selector.overlay
    selector.overlay = {**old_overlay, **overrides}
    try:
        yield
    finally:
        selector.overlay = old_overlay
```

## Design Patterns

### 1. Multi-Tier Configuration Selection

**Pattern**: Hierarchical fallback with caching at multiple levels

**Benefits**:

- Performance: Fast path for cached configurations
- Flexibility: Multiple ways to provide configurations
- Robustness: Always has a fallback (heuristics)

### 2. Device-Aware Caching

**Pattern**: Cache key includes device fingerprint

**Benefits**:

- Correctness: Device-specific optimizations
- Performance: Optimal configuration per device
- Portability: Works across heterogeneous hardware

### 3. Atomic Persistence

**Pattern**: Write to temp file + atomic replace

**Benefits**:

- Safety: No partial writes or corruption
- Concurrency: Multiple processes can safely read/write
- Reliability: Filesystem-level guarantees

### 4. Custom VJP Integration

**Pattern**: Extract array leaves, rebuild in fwd/bwd

**Benefits**:

- Efficiency: Specialized gradient computation
- Correctness: Proper handling of constants
- Compatibility: Works with JAX transformations

## Performance Considerations

### Configuration Selection

- **First call**: May trigger autotuning (seconds)
- **Subsequent calls**: Cache lookup (microseconds)
- **Persistent cache**: Survives process restart (milliseconds to load)

### Autotuning

- **Warmup runs**: Eliminates JIT compilation overhead
- **Multiple iterations**: Reduces measurement noise
- **Parallel testing**: Can test multiple configs concurrently
- **Early stopping**: Can stop when clear winner emerges

### Memory Management

- **In-memory cache**: Unbounded growth (consider LRU eviction)
- **Persistent cache**: JSON file size grows with entries
- **Configuration objects**: Lightweight dataclasses

## Testing and Debugging

### Force Specific Configuration

```python
# Direct override
output = executor(kernel, *args, _cfg=my_config)

# Via overlay cache
with overlay_cache({cache_key: my_config}):
    output = executor(kernel, *args)
```

### Disable Autotuning

```python
with policy_override(selector, allow_autotune=False):
    output = executor(kernel, *args)
```

### Inspect Cache

```python
# In-memory
print(f"Cached configs: {cache.size()}")
for key, cfg in cache._cache.items():
    print(f"{key}: {cfg}")

# Persistent
with open(persistent.path) as f:
    data = json.load(f)
    print(f"Persistent configs: {len(data)}")
```

## Conclusion

The ops system provides a sophisticated, production-ready infrastructure for kernel execution with:

1. **Flexibility**: Multiple configuration sources with fallback
2. **Performance**: Automatic optimization with caching
3. **Extensibility**: Easy to add new kernels and configurations
4. **Robustness**: Error handling and graceful degradation
5. **Debugging**: Rich context and inspection capabilities

This architecture enables ejKernel to deliver optimal performance across diverse hardware while maintaining simplicity for end users.
