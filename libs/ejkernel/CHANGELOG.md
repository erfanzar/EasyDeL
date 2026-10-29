# Changelog

All notable changes to ejKernel will be documented in this file.

## [0.0.23] - 2025-11-30

### Added

- New `_ring_splash.py` implementation using JAX Splash Attention kernels for TPU ring attention
- `RingSplashAttentionKernel` class with PyTree registration for JAX compatibility
- `make_ring_attention()` factory function for creating ring attention kernels
- `ring_splash_attention()` low-level API for direct splash attention access
- Platform detection utilities: `get_platform()`, `is_tpu()`, `is_gpu()`, `is_cpu()` in `ejkernel/utils.py`
- `replicate_to_local_devices()` utility in `ejkernel/xla_utils/shardings.py`
- `chunk_size` parameter for chunked causal attention (Llama4 style)
- `mask_builder` parameter for custom mask construction
- `fwd_params` and `bwd_params` for fine-grained block size control
- Initial Triton ring attention implementation scaffolding

### Changed

- **BREAKING**: Ring attention TPU implementation now uses Splash Attention kernels instead of custom Pallas kernels
- **BREAKING**: `softmax_aux` parameter simplified from `Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"]` to `Float[Array, "num_sinks"]` across all attention kernels
- **BREAKING**: `bias` parameter in TPU ring attention now raises `NotImplementedError`
- Blocksparse attention now uses `make_splash_mha` instead of `make_splash_mqa_single_device`
- `FwdParams` and `BwdParams` configs are now hashable
- Simplified `SplashCustomReturnType` type definition formatting

### Removed

- `_pallas_impl_fwd.py` (769 lines) - replaced by splash attention
- `_pallas_impl_bwd.py` (1003 lines) - replaced by splash attention
- `_utils.py` (192 lines) - utilities merged into new implementation
- Ring attention parameters: `float32_logits`, `deterministic`, `dropout_rng`, `pdrop`, `policy`, `prevent_cse`, `precision`, `dtype`, `cache_idx`, `attention_mask`

### Fixed

- Configs not being hashable issue (bed7c39)

### Migration Guide

#### Ring Attention API Changes

**Before (0.0.22):**

```python
ring_attention(
    query, key, value,
    attention_mask=mask,
    float32_logits=True,
    deterministic=True,
    dropout_rng=None,
    pdrop=0.0,
    ...
)
```

**After (0.0.23):**

```python
ring_attention(
    query, key, value,
    causal=True,  # or use mask_builder for custom masks
    fwd_params=FwdParams(q_blocksize=512, kv_blocksize=512),
    ...
)
```

#### softmax_aux Shape Change

**Before:**

```python
softmax_aux = jnp.zeros((num_kv_heads, num_sinks))  # 2D allowed
```

**After:**

```python
softmax_aux = jnp.zeros((num_sinks,))  # Must be 1D
```
