# Utilities and Helper Functions Analysis

## Overview

The utilities and helper functions in ejKernel provide essential infrastructure for device management, type conversions, sequence handling, and various computational helpers. These utilities are organized across multiple modules to support the core kernel operations.

## Module Organization

```md
ejkernel/
├── xla_utils/
│   └── utils.py          # XLA-specific utilities
├── ops/utils/
│   ├── fingerprint.py    # Device fingerprinting and hashing
│   └── datacarrier.py    # Configuration data structures
└── callib/
    └── __init__.py       # Calibration utilities
```

## XLA Utilities

### Sequence Length Handling

```python
def prepare_lens(cu_seqlens: Int[Array, "batch + 1"]) -> Int[Array, "batch"]:
    """
    Convert cumulative sequence lengths to individual lengths.

    Args:
        cu_seqlens: Cumulative lengths [0, len1, len1+len2, ...]

    Returns:
        Individual lengths [len1, len2, ...]

    Example:
        >>> cu_seqlens = jnp.array([0, 5, 12, 18])
        >>> prepare_lens(cu_seqlens)
        Array([5, 7, 6], dtype=int32)
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_lens_from_mask(mask: Bool[Array, "batch seq_len"]) -> Int[Array, "batch"]:
    """
    Extract sequence lengths from boolean mask.

    Args:
        mask: Boolean mask where True indicates valid positions

    Returns:
        Sequence lengths per batch element

    Example:
        >>> mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        >>> prepare_lens_from_mask(mask)
        Array([3, 2], dtype=int32)
    """
    return mask.sum(axis=-1, dtype=jnp.int32)

def prepare_cu_seqlens_from_mask(mask: Bool[Array, "batch seq_len"],
                                out_dtype: jnp.dtype = jnp.int32) -> Int[Array, "batch + 1"]:
    """
    Convert boolean mask to cumulative sequence lengths.

    Args:
        mask: Boolean mask
        out_dtype: Output data type

    Returns:
        Cumulative lengths with leading 0

    Example:
        >>> mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        >>> prepare_cu_seqlens_from_mask(mask)
        Array([0, 3, 5], dtype=int32)
    """
    lens = prepare_lens_from_mask(mask)
    cumsum_lens = lens.cumsum(axis=0, dtype=out_dtype)
    return jnp.pad(cumsum_lens, (1, 0))
```

### Segment ID Utilities

Segment IDs are an efficient representation for attention masking, particularly on TPUs:

```python
def segment_ids_to_mask(
    segment_ids: Int[Array, "batch seq_len"] | tuple[Int[Array, "..."], Int[Array, "..."]],
    dtype: jnp.dtype = jnp.bool_,
    return_separate_masks: bool = False
) -> Bool[Array, "batch seq_len_q seq_len_k"] | tuple:
    """
    Convert segment IDs to attention mask.

    Segment ID conventions:
    - -1 or 0: padding token
    - Positive integers: segment identifiers
    - Same segment ID = can attend

    Args:
        segment_ids: Either single array for self-attention or
                    tuple (q_segments, kv_segments) for cross-attention
        dtype: Output mask dtype
        return_separate_masks: Return (q_mask, kv_mask, attention_mask)

    Returns:
        Attention mask or tuple of masks

    Example:
        >>> # Self-attention
        >>> segment_ids = jnp.array([[1, 1, 2, 0], [1, 2, 0, 0]])
        >>> mask = segment_ids_to_mask(segment_ids)
        >>> # mask[0, 0, 1] = True (same segment 1)
        >>> # mask[0, 0, 2] = False (different segments 1 vs 2)

        >>> # Cross-attention
        >>> q_segments = jnp.array([[1, 1, 2, 0]])
        >>> kv_segments = jnp.array([[1, 2, 2, 0]])
        >>> mask = segment_ids_to_mask((q_segments, kv_segments))
    """
    if isinstance(segment_ids, tuple):
        q_segment_ids, kv_segment_ids = segment_ids

        # Create masks for valid positions
        q_mask = (q_segment_ids > 0).astype(dtype)
        kv_mask = (kv_segment_ids > 0).astype(dtype)

        # Create attention mask
        q_seg = q_segment_ids[:, :, None]
        kv_seg = kv_segment_ids[:, None, :]
        attention_mask = (q_seg == kv_seg) & (q_seg > 0) & (kv_seg > 0)
        attention_mask = attention_mask.astype(dtype)
    else:
        # Self-attention case
        q_mask = (segment_ids > 0).astype(dtype)
        kv_mask = q_mask

        seg_q = segment_ids[:, :, None]
        seg_kv = segment_ids[:, None, :]
        attention_mask = (seg_q == seg_kv) & (seg_q > 0) & (seg_kv > 0)
        attention_mask = attention_mask.astype(dtype)

    if return_separate_masks:
        return q_mask, kv_mask, attention_mask
    return attention_mask
```

### Mask to Segment ID Conversion

```python
def mask_to_segment_ids(
    mask: ndarray,
    per_head: bool = False
) -> tuple[ndarray, ndarray]:
    """
    JIT-friendly conversion from attention mask to segment IDs.

    Algorithm:
    1. Pack each row/column into bit representation
    2. Find equal rows/columns via bit comparison
    3. Assign contiguous segment IDs
    4. Mark padding as -1

    Args:
        mask: Attention mask with shape:
             - (Q, K): 2D mask
             - (B, Q, K): Batched mask
             - (B, H, Q, K): Per-head mask
        per_head: Process each head independently

    Returns:
        (q_segment_ids, kv_segment_ids) with appropriate shapes

    Example:
        >>> mask = jnp.array([
        ...     [[1, 1, 0, 0],
        ...      [1, 1, 1, 0],
        ...      [0, 1, 1, 1],
        ...      [0, 0, 1, 1]]
        ... ])
        >>> q_seg, kv_seg = mask_to_segment_ids(mask)
        >>> # q_seg[0] = [1, 2, 3, 4] (unique patterns)
        >>> # kv_seg[0] = [1, 2, 3, 4] (matching patterns)
    """

    def pack_bits(arr):
        """Pack boolean array into bit representation"""
        # Reshape to chunks of 8 for packbits
        pad_len = (8 - arr.shape[-1] % 8) % 8
        padded = jnp.pad(arr, ((0, 0),) * (arr.ndim - 1) + ((0, pad_len),))
        packed = jnp.packbits(padded, axis=-1)
        return packed

    def find_unique_patterns(packed_rows):
        """Find unique bit patterns and assign IDs"""
        n_rows = packed_rows.shape[0]
        segment_ids = jnp.zeros(n_rows, dtype=jnp.int32)

        current_id = 1
        for i in range(n_rows):
            # Check if this pattern has been seen
            is_new = True
            for j in range(i):
                if jnp.all(packed_rows[i] == packed_rows[j]):
                    segment_ids = segment_ids.at[i].set(segment_ids[j])
                    is_new = False
                    break

            if is_new:
                # Check if it's padding (all zeros)
                if jnp.any(packed_rows[i]):
                    segment_ids = segment_ids.at[i].set(current_id)
                    current_id += 1
                else:
                    segment_ids = segment_ids.at[i].set(-1)

        return segment_ids

    # Handle different input shapes
    if mask.ndim == 2:
        # Simple 2D case
        q_packed = pack_bits(mask)
        kv_packed = pack_bits(mask.T)
        q_segment_ids = find_unique_patterns(q_packed)
        kv_segment_ids = find_unique_patterns(kv_packed)

    elif mask.ndim == 3:
        # Batched case
        batch_size = mask.shape[0]
        q_segment_ids = []
        kv_segment_ids = []

        for b in range(batch_size):
            q_packed = pack_bits(mask[b])
            kv_packed = pack_bits(mask[b].T)
            q_segment_ids.append(find_unique_patterns(q_packed))
            kv_segment_ids.append(find_unique_patterns(kv_packed))

        q_segment_ids = jnp.stack(q_segment_ids)
        kv_segment_ids = jnp.stack(kv_segment_ids)

    elif mask.ndim == 4 and per_head:
        # Per-head processing
        batch_size, num_heads = mask.shape[:2]
        q_segment_ids = []
        kv_segment_ids = []

        for b in range(batch_size):
            q_head_ids = []
            kv_head_ids = []
            for h in range(num_heads):
                q_packed = pack_bits(mask[b, h])
                kv_packed = pack_bits(mask[b, h].T)
                q_head_ids.append(find_unique_patterns(q_packed))
                kv_head_ids.append(find_unique_patterns(kv_packed))

            q_segment_ids.append(jnp.stack(q_head_ids))
            kv_segment_ids.append(jnp.stack(kv_head_ids))

        q_segment_ids = jnp.stack(q_segment_ids)
        kv_segment_ids = jnp.stack(kv_segment_ids)

    return q_segment_ids, kv_segment_ids
```

### Type Conversion Utilities

```python
def identity_dtype_convert(dtype: jnp.dtype) -> Callable:
    """
    Create identity function with custom gradient dtype conversion.

    This is useful for mixed-precision training where forward pass
    uses one dtype but gradients need to be in another.

    Args:
        dtype: Target dtype for gradients

    Returns:
        Identity function with dtype conversion in backward

    Example:
        >>> # Ensure gradients match query dtype in mixed precision
        >>> convert_fn = identity_dtype_convert(query.dtype)
        >>> key = convert_fn(key.astype(jnp.float16))
        >>> # Forward: key stays in float16
        >>> # Backward: gradients converted to query.dtype
    """
    @jax.custom_vjp
    def identity_fn(x):
        return x

    def identity_fn_fwd(x):
        return x, None

    def identity_fn_bwd(res, g):
        return (g.astype(dtype),)

    identity_fn.defvjp(identity_fn_fwd, identity_fn_bwd)
    return identity_fn
```

## Device Fingerprinting Utilities

### Device Identification

```python
def device_fingerprint(dev: Device | None = None) -> str:
    """
    Generate unique device identifier for caching.

    Args:
        dev: JAX device (defaults to first device)

    Returns:
        String identifier: 'device_kind|platform_version'

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
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                platform_version = f"cuda_{result.stdout.strip()}"
        except:
            platform_version = "cuda_unknown"

    elif device_kind == "tpu":
        # Get TPU generation
        try:
            tpu_type = dev.device_kind
            if "v2" in tpu_type:
                platform_version = "v2"
            elif "v3" in tpu_type:
                platform_version = "v3"
            elif "v4" in tpu_type:
                platform_version = "v4"
            else:
                platform_version = "unknown"
        except:
            platform_version = "unknown"

    return f"{device_kind}|{platform_version}"

def get_device_platform(dev: Device | None = None) -> str:
    """
    Get device platform type.

    Returns:
        'gpu', 'tpu', 'cpu', or 'unknown'
    """
    if dev is None:
        dev = jax.devices()[0]

    platform = dev.platform.lower()

    if "gpu" in platform or "cuda" in platform or "rocm" in platform:
        return "gpu"
    elif "tpu" in platform:
        return "tpu"
    elif "cpu" in platform:
        return "cpu"
    else:
        return "unknown"
```

### Hashing Utilities

```python
def abstractify(pytree: Any) -> Any:
    """
    Convert arrays to ShapeDtypeStruct for cache keys.

    This creates a lightweight representation that can be hashed
    without accessing array data.

    Args:
        pytree: Nested structure with arrays

    Returns:
        Same structure with arrays replaced by ShapeDtypeStruct

    Example:
        >>> arr = jnp.ones((2, 3))
        >>> abstract = abstractify(arr)
        >>> # abstract = ShapeDtypeStruct((2, 3), float32)
    """
    def _abstractify_leaf(x):
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        return x

    return jax.tree_map(_abstractify_leaf, pytree)

def short_hash(obj: Any) -> str:
    """
    Generate 16-character hash for any object.

    Args:
        obj: Object to hash

    Returns:
        16-character hex string

    Example:
        >>> config = {"block_size": 128, "num_warps": 4}
        >>> short_hash(config)
        'a1b2c3d4e5f6g7h8'
    """
    json_str = stable_json(obj)
    full_hash = hashlib.sha256(json_str.encode()).hexdigest()
    return full_hash[:16]

def stable_json(obj: Any) -> str:
    """
    Create deterministic JSON representation.

    Handles special types:
    - Functions: (module, name, source_file)
    - functools.partial: (func, args, kwargs)
    - Dataclasses: converted to dict
    - Pydantic models: model_dump()
    - Arrays: (shape, dtype) only

    Args:
        obj: Object to serialize

    Returns:
        Deterministic JSON string
    """
    def _serialize(o):
        if callable(o):
            # Serialize function reference
            return {
                "_type": "function",
                "module": o.__module__,
                "name": o.__name__,
                "file": inspect.getsourcefile(o)
            }

        elif isinstance(o, functools.partial):
            # Serialize partial application
            return {
                "_type": "partial",
                "func": _serialize(o.func),
                "args": [_serialize(arg) for arg in o.args],
                "kwargs": {k: _serialize(v) for k, v in o.keywords.items()}
            }

        elif dataclasses.is_dataclass(o):
            # Convert dataclass to dict
            return {
                "_type": "dataclass",
                "class": o.__class__.__name__,
                "data": dataclasses.asdict(o)
            }

        elif hasattr(o, 'model_dump'):
            # Pydantic model
            return {
                "_type": "pydantic",
                "class": o.__class__.__name__,
                "data": o.model_dump()
            }

        elif isinstance(o, (jnp.ndarray, np.ndarray)):
            # Array metadata only
            return {
                "_type": "array",
                "shape": list(o.shape),
                "dtype": str(o.dtype)
            }

        elif isinstance(o, jax.ShapeDtypeStruct):
            # Already abstracted
            return {
                "_type": "shape_dtype",
                "shape": list(o.shape),
                "dtype": str(o.dtype)
            }

        elif isinstance(o, (list, tuple)):
            return [_serialize(item) for item in o]

        elif isinstance(o, dict):
            return {k: _serialize(v) for k, v in sorted(o.items())}

        else:
            # Primitive types
            return o

    serialized = _serialize(obj)
    return json.dumps(serialized, sort_keys=True, separators=(',', ':'))
```

## Configuration Data Carriers

### Forward Pass Parameters

```python
@dataclass
class FwdParams:
    """
    Forward pass configuration parameters for kernels.

    These parameters control block sizes and thread configurations
    for optimal performance on specific hardware.
    """
    # Block dimensions for matrix operations
    blocksize_m: int | None = None  # M dimension block size
    blocksize_k: int | None = None  # K dimension block size
    blocksize_n: int | None = None  # N dimension block size

    # Attention-specific block sizes
    q_blocksize: int | None = None   # Query block size
    kv_blocksize: int | None = None  # Key/Value block size

    # Thread block configuration
    blocksize_heads: int | None = None  # Heads per block
    blocksize_keys: int | None = None   # Keys per block
    num_key_splits: int | None = None   # Key splitting factor

    # GPU thread configuration
    num_warps: int | None = None   # Warps per block (GPU)
    num_stages: int | None = None  # Pipeline stages (GPU)

    def __hash__(self):
        """Custom hash for use in caching"""
        return hash(tuple(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
        ))

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v is not None}

    @classmethod
    def merge(cls, base: "FwdParams", override: "FwdParams") -> "FwdParams":
        """Merge two parameter sets, override takes precedence"""
        base_dict = base.to_dict() if base else {}
        override_dict = override.to_dict() if override else {}
        return cls(**{**base_dict, **override_dict})
```

### Backward Pass Parameters

```python
@dataclass
class BwdParams:
    """
    Backward pass configuration parameters.

    Typically uses smaller block sizes than forward pass
    for better memory efficiency during gradient computation.
    """
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

    @classmethod
    def from_fwd_params(cls, fwd: FwdParams, scale: float = 0.5) -> "BwdParams":
        """
        Create backward params from forward params.

        Typically uses smaller blocks for memory efficiency.
        """
        return cls(
            q_blocksize=int((fwd.q_blocksize or 128) * scale),
            kv_blocksize=int((fwd.kv_blocksize or 128) * scale),
            num_warps=fwd.num_warps,
            num_stages=fwd.num_stages
        )
```

## Calibration Utilities

### Shared Memory Estimation

```python
def estimate_shared_memory_bytes(
    q_block: int,
    kv_block: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.float16,
    include_softmax: bool = True
) -> int:
    """
    Estimate shared memory usage for attention kernels.

    Args:
        q_block: Query block size
        kv_block: Key/value block size
        head_dim: Attention head dimension
        dtype: Data type
        include_softmax: Include softmax statistics

    Returns:
        Estimated bytes of shared memory

    Example:
        >>> smem = estimate_shared_memory_bytes(128, 256, 64)
        >>> if smem > 49152:  # 48KB limit
        ...     print("Block sizes too large for shared memory")
    """
    bytes_per_element = np.dtype(dtype).itemsize

    # Q, K, V tiles in shared memory
    q_smem = q_block * head_dim * bytes_per_element
    k_smem = kv_block * head_dim * bytes_per_element
    v_smem = kv_block * head_dim * bytes_per_element

    # Softmax statistics (max and sum)
    if include_softmax:
        # Float32 for numerical stability
        stats_smem = q_block * 2 * 4
    else:
        stats_smem = 0

    # Alignment padding (GPU typically requires 128-byte alignment)
    def align(x, alignment=128):
        return ((x + alignment - 1) // alignment) * alignment

    total = q_smem + k_smem + v_smem + stats_smem
    return align(total)

def get_shared_memory_limit() -> int:
    """
    Get shared memory limit for current device.

    Returns:
        Shared memory limit in bytes
    """
    limit_env = os.getenv("EJKERNEL_TRITON_SMEM_LIMIT")
    if limit_env:
        return int(limit_env)

    # Default limits by GPU architecture
    device = jax.devices()[0]
    if "A100" in str(device):
        return 164 * 1024  # 164KB for A100
    elif "V100" in str(device):
        return 96 * 1024   # 96KB for V100
    elif "T4" in str(device):
        return 64 * 1024   # 64KB for T4
    else:
        return 48 * 1024   # Conservative default
```

### Warp Configuration Selection

```python
def select_num_warps(
    q_block: int,
    kv_block: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.float16
) -> int:
    """
    Select optimal number of warps for GPU kernels.

    Args:
        q_block: Query block size
        kv_block: Key/value block size
        head_dim: Head dimension
        dtype: Data type

    Returns:
        Number of warps (4, 8, or 16)

    Algorithm:
        - Estimate compute intensity
        - Balance parallelism vs register pressure
        - Consider occupancy limits
    """
    # Estimate operations per block
    flops = q_block * kv_block * head_dim * 2  # Multiply-add

    # Estimate data movement
    bytes_moved = (q_block + kv_block) * head_dim * np.dtype(dtype).itemsize

    # Compute intensity
    intensity = flops / bytes_moved

    # High intensity = more warps for compute
    # Low intensity = fewer warps to reduce conflicts
    if intensity > 10:
        num_warps = 8
    elif intensity > 5:
        num_warps = 4
    else:
        num_warps = 2

    # Adjust for block size
    total_threads = num_warps * 32
    if q_block * kv_block < 1024:
        num_warps = min(num_warps, 4)
    elif q_block * kv_block > 16384:
        num_warps = max(num_warps, 8)

    return num_warps

def select_num_stages(smem_bytes: int) -> int:
    """
    Select pipeline stages for GPU kernels.

    Args:
        smem_bytes: Shared memory usage

    Returns:
        Number of pipeline stages (1, 2, or 3)

    Note:
        More stages enable better latency hiding but
        require more shared memory and registers.
    """
    smem_limit = get_shared_memory_limit()

    # Available memory for additional stages
    available = smem_limit - smem_bytes

    if available > smem_bytes * 2:
        return 3  # Can afford 3 stages
    elif available > smem_bytes:
        return 2  # Can afford 2 stages
    else:
        return 1  # Single stage only
```

## Array Manipulation Helpers

### Chunking and Padding

```python
def chunk_array(
    arr: Array,
    chunk_size: int,
    axis: int = 0,
    pad_value: float = 0.0
) -> Array:
    """
    Chunk array along axis with padding if needed.

    Args:
        arr: Input array
        chunk_size: Size of chunks
        axis: Axis to chunk along
        pad_value: Value for padding

    Returns:
        Chunked array with shape [..., num_chunks, chunk_size, ...]

    Example:
        >>> x = jnp.arange(10)
        >>> chunk_array(x, 4)
        Array([[0, 1, 2, 3],
               [4, 5, 6, 7],
               [8, 9, 0, 0]], dtype=int32)
    """
    shape = arr.shape
    axis_len = shape[axis]

    # Calculate padding needed
    num_chunks = (axis_len + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - axis_len

    if pad_len > 0:
        # Pad array
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, pad_len)
        arr = jnp.pad(arr, pad_width, constant_values=pad_value)

    # Reshape to chunks
    new_shape = list(shape)
    new_shape[axis:axis+1] = [num_chunks, chunk_size]
    return arr.reshape(new_shape)

def unchunk_array(
    arr: Array,
    original_length: int,
    axis: int = 0
) -> Array:
    """
    Reverse chunking operation.

    Args:
        arr: Chunked array
        original_length: Original length before chunking
        axis: Chunked axis

    Returns:
        Unchunked array with original length

    Example:
        >>> chunked = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 0]])
        >>> unchunk_array(chunked, 10)
        Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
    """
    # Flatten chunks
    shape = list(arr.shape)
    shape[axis:axis+2] = [-1]
    arr = arr.reshape(shape)

    # Trim to original length
    indices = [slice(None)] * arr.ndim
    indices[axis] = slice(original_length)
    return arr[tuple(indices)]
```

### Numerical Stability Helpers

```python
def log_sum_exp(x: Array, axis: int | None = None,
                keepdims: bool = False) -> Array:
    """
    Numerically stable log-sum-exp.

    Args:
        x: Input array
        axis: Reduction axis
        keepdims: Keep reduced dimensions

    Returns:
        log(sum(exp(x)))

    Example:
        >>> x = jnp.array([1000., 1001., 1002.])
        >>> log_sum_exp(x)  # Stable computation
        1002.407
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    sum_exp = jnp.sum(exp_x, axis=axis, keepdims=keepdims)
    lse = jnp.log(sum_exp)

    if not keepdims and axis is not None:
        x_max = jnp.squeeze(x_max, axis=axis)

    return lse + x_max

def safe_divide(numerator: Array, denominator: Array,
                min_denominator: float = 1e-10) -> Array:
    """
    Safe division with small denominator handling.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        min_denominator: Minimum denominator value

    Returns:
        numerator / max(denominator, min_denominator)
    """
    safe_denom = jnp.maximum(denominator, min_denominator)
    return numerator / safe_denom
```

## Testing Utilities

### Array Comparison

```python
def assert_allclose(actual: Array, expected: Array,
                   rtol: float = 1e-5, atol: float = 1e-8,
                   equal_nan: bool = True):
    """
    Assert arrays are close within tolerances.

    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Consider NaN values equal

    Raises:
        AssertionError: If arrays differ beyond tolerance
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    if actual.dtype != expected.dtype:
        print(f"Warning: dtype mismatch: {actual.dtype} vs {expected.dtype}")

    close = jnp.allclose(actual, expected, rtol=rtol, atol=atol,
                        equal_nan=equal_nan)

    if not close:
        # Find maximum difference for debugging
        diff = jnp.abs(actual - expected)
        max_diff = jnp.max(diff)
        max_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)

        raise AssertionError(
            f"Arrays not close\n"
            f"Max difference: {max_diff} at index {max_idx}\n"
            f"Actual: {actual[max_idx]}\n"
            f"Expected: {expected[max_idx]}"
        )
```

### Mock Data Generation

```python
def generate_attention_inputs(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: jnp.dtype = jnp.float32,
    key: jax.random.PRNGKey | None = None
) -> dict[str, Array]:
    """
    Generate random attention inputs for testing.

    Returns:
        Dictionary with query, key, value arrays
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    keys = jax.random.split(key, 3)

    query = jax.random.normal(
        keys[0], (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    ) * 0.02

    key_array = jax.random.normal(
        keys[1], (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    ) * 0.02

    value = jax.random.normal(
        keys[2], (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    ) * 0.02

    return {"query": query, "key": key_array, "value": value}
```

## Performance Monitoring

### Timer Context Manager

```python
@contextmanager
def timer(name: str = "Operation"):
    """
    Simple timer context manager.

    Example:
        >>> with timer("Flash Attention"):
        ...     output = flash_attention(q, k, v)
        Flash Attention took 0.123 seconds
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.3f} seconds")
```

### Memory Profiling

```python
def get_memory_usage() -> dict[str, int]:
    """
    Get current memory usage on device.

    Returns:
        Dictionary with allocated and reserved memory in bytes
    """
    device = jax.devices()[0]

    if device.platform == "gpu":
        # GPU memory tracking
        stats = device.memory_stats()
        return {
            "allocated": stats.get("bytes_in_use", 0),
            "reserved": stats.get("bytes_reserved", 0),
            "limit": stats.get("bytes_limit", 0)
        }
    else:
        # Basic memory info for other platforms
        import psutil
        process = psutil.Process()
        return {
            "allocated": process.memory_info().rss,
            "reserved": 0,
            "limit": psutil.virtual_memory().total
        }
```

## Conclusion

The utilities and helper functions in ejKernel provide:

1. **Device Management**: Fingerprinting and platform detection
2. **Data Conversion**: Segment IDs, masks, and type conversions
3. **Configuration**: Parameter dataclasses with merging and hashing
4. **Numerical Stability**: Safe operations and stability helpers
5. **Testing Support**: Comparison, generation, and profiling utilities

These utilities form the foundation that enables the kernel system's flexibility, performance, and reliability across different hardware platforms.
