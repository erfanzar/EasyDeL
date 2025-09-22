# EasyDeL Caching Layer Tests

Comprehensive test suite for the EasyDeL caching system, covering all cache types and utilities.

## Test Structure

### Core Tests

- `test_abstracts.py` - Tests for abstract base classes (BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata)
- `test_specs.py` - Tests for cache specifications (KVCacheSpec, AttentionSpec, FullAttentionSpec, SlidingWindowSpec, etc.)
- `test_cache_utils.py` - Tests for utility classes (AttnMaskDetail)

### Cache Implementation Tests

- `test_transformer_cache.py` - Tests for transformer key-value caching
- `test_pages_kv_cache.py` - Tests for paged attention caching
- `test_linear_cache.py` - Tests for linear attention (Gated Delta) caching
- `test_mamba_cache.py` - Tests for Mamba and Mamba2 state-space model caching

### Integration Tests

- `test_all_caching.py` - Import tests and module-level validation

## Running Tests

Run all caching tests:

```bash
pytest tests/layers/caching/
```

Run specific test file:

```bash
pytest tests/layers/caching/test_transformer_cache.py
```

Run with verbose output:

```bash
pytest tests/layers/caching/ -v
```

Run with coverage:

```bash
pytest tests/layers/caching/ --cov=easydel.layers.caching
```

## Test Coverage

The test suite covers:

- ✅ Abstract base class interfaces and contracts
- ✅ Cache metadata creation and validation
- ✅ Cache view initialization and updates
- ✅ Multi-layer cache orchestration
- ✅ Memory size calculations and specifications
- ✅ Different attention patterns (full, sliding, chunked)
- ✅ State-space model caching (Mamba, Mamba2)
- ✅ Linear attention caching
- ✅ Paged attention for memory efficiency
- ✅ JAX pytree compatibility
- ✅ Sharding and distributed execution support
- ✅ Quantization support
- ✅ Runtime metadata handling

## Key Test Patterns

### Metadata Validation

```python
def test_metadata_validation_errors(self):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TransformerCacheMetaData.create(batch_size=0, ...)
```

### Cache Initialization

```python
def test_cache_initialization(self, setup):
    cache = TransformerCache.init_cache(
        mesh=mesh,
        metadata=metadata,
        partition_manager=partition_manager,
        dtype=jnp.float32,
    )
    assert len(cache) == num_layers
```

### Cache Updates

```python
def test_view_concatenate_to_cache(self, setup):
    key_cache, value_cache, mask, new_view = view.concatenate_to_cache(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
    )
    assert jnp.all(new_view.layer_index == seq_len)
```

### PyTree Compatibility

```python
def test_metadata_is_pytree(self):
    leaves, treedef = tree.tree_flatten(metadata)
    reconstructed = tree.tree_unflatten(treedef, leaves)
    assert reconstructed == metadata
```

## Notes

- Tests use JAX's device mesh for distributed testing
- Most tests include both unit tests and integration tests
- All cache types follow consistent testing patterns
- Tests validate both functional correctness and performance characteristics
