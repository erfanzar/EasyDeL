# PyTree Utilities Documentation

## Overview

The `eformer.pytree` module provides comprehensive utilities for working with JAX PyTrees - nested tree-like structures containing JAX arrays, NumPy arrays, and Python containers (dicts, lists, tuples). This documentation covers the tree manipulation utilities and serialization functionality.

## Table of Contents

1. [Tree Utilities (`_tree_util.py`)](#tree-utilities)
2. [Serialization (`_serialization.py`)](#serialization)
3. [Usage Examples](#usage-examples)
4. [API Reference](#api-reference)

---

## Tree Utilities

The `_tree_util.py` module provides over 50 utility functions for manipulating PyTrees. These functions are organized into several categories:

### Core Functions

#### Basic Operations

- **`is_array(element)`**: Check if element is a JAX or NumPy array
- **`is_array_like(element)`**: Check if element is array-like (includes scalars)
- **`is_flatten(tree)`**: Check if dictionary represents a flattened tree
- **`is_iterable(obj)`**: Check if object is iterable

#### Tree Structure

- **`tree_structure_equal(tree1, tree2)`**: Compare tree structures
- **`tree_equal(*pytrees, typematch=False, rtol=0.0, atol=0.0)`**: Compare tree values and structure
- **`deepcopy_tree(model)`**: Create a deep copy of a PyTree

### Statistical Functions

#### Size and Memory

- **`tree_size(tree)`**: Calculate total number of elements
- **`tree_bytes(tree)`**: Calculate memory usage in bytes

#### Aggregation

- **`tree_sum(tree, axis=None)`**: Sum all values in tree
- **`tree_mean(tree, axis=None)`**: Compute mean of values
- **`tree_min(tree)`**: Find minimum value
- **`tree_max(tree)`**: Find maximum value
- **`tree_norm(tree, ord=2)`**: Compute norm (L1, L2, L∞)
- **`tree_reduce(reducer, tree, initializer=None)`**: Custom reduction

### Arithmetic Operations

- **`tree_add(tree1, tree2)`**: Element-wise addition
- **`tree_subtract(tree1, tree2)`**: Element-wise subtraction
- **`tree_multiply(tree1, tree2)`**: Element-wise or scalar multiplication
- **`tree_divide(tree1, tree2)`**: Element-wise or scalar division
- **`tree_dot(tree1, tree2)`**: Dot product of two trees

### Mathematical Functions

- **`tree_abs(tree)`**: Absolute values
- **`tree_sign(tree)`**: Sign function (-1, 0, 1)
- **`tree_sqrt(tree)`**: Square root
- **`tree_exp(tree)`**: Exponential
- **`tree_log(tree)`**: Natural logarithm
- **`tree_reciprocal(tree)`**: Compute 1/x
- **`tree_clip(tree, min_val=None, max_val=None)`**: Clip values
- **`tree_round(tree, decimals=0)`**: Round values

### Array Transformations

- **`tree_cast(tree, dtype)`**: Cast arrays to dtype
- **`tree_reshape(tree, shape)`**: Reshape arrays
- **`tree_transpose(tree, axes=None)`**: Transpose arrays
- **`tree_squeeze(tree, axis=None)`**: Remove single dimensions
- **`tree_expand_dims(tree, axis)`**: Add new dimensions

### Boolean Operations

- **`tree_any(tree)`**: Check if any value is True
- **`tree_all(tree)`**: Check if all values are True
- **`tree_where(condition, x, y)`**: Conditional selection

### NaN/Inf Handling

- **`tree_isnan(tree)`**: Check for NaN values
- **`tree_isinf(tree)`**: Check for infinite values
- **`tree_isfinite(tree)`**: Check for finite values
- **`tree_replace_nans(tree, value=0.0)`**: Replace NaN values
- **`tree_replace_infs(tree, value=0.0)`**: Replace infinite values

### Tree Creation

- **`tree_zeros_like(tree)`**: Create tree of zeros
- **`tree_ones_like(tree)`**: Create tree of ones
- **`tree_random_like(tree, key, distribution="normal", **kwargs)`**: Create random tree

### Advanced Operations

#### Splitting and Merging

- **`split(pytree, filter_spec, replace=None, is_leaf=None)`**: Split tree based on filter
- **`merge(*pytrees, is_leaf=None)`**: Merge multiple trees
- **`recursive_merge(full_tree, updates)`**: Recursively merge trees

#### Stacking and Concatenation

- **`tree_concatenate(trees, axis=0)`**: Concatenate arrays in trees
- **`tree_stack(trees, axis=0)`**: Stack arrays in trees

#### Filtering

- **`tree_filter(tree, predicate)`**: Filter tree elements

### Dictionary Operations

#### Flattening

- **`flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None, fumap=False)`**: Flatten nested dictionary
- **`unflatten_dict(xs, sep=None)`**: Unflatten dictionary
- **`flatten_mapping(xs, keep_empty_nodes=False, is_leaf=None, sep=None)`**: Flatten nested mapping
- **`unflatten_mapping(xs, sep=None)`**: Unflatten mapping
- **`flatten_to_sequence(xs, is_leaf=None)`**: Flatten to list of (path, value) pairs
- **`flatten_tree(xs, is_leaf=None, sep=None)`**: Flatten JAX tree to dict

#### Key Conversion

- **`int_key_to_string(xs)`**: Convert integer keys to strings
- **`string_key_to_int(xs)`**: Convert string keys to integers

### Path Operations

- **`tree_map_with_path(f, tree, is_leaf=None)`**: Map function with path information
- **`tree_flatten_with_paths(tree, is_leaf=None)`**: Flatten with path tracking
- **`tree_leaves_with_paths(tree, is_leaf=None)`**: Get leaves with paths
- **`tree_path_to_string(path, sep=None)`**: Convert tree path to string
- **`named_tree_map(f, tree, *rest, is_leaf=None, sep=None)`**: Map with named paths

### Sharding

- **`specs_to_name_sharding(tree, mesh=None)`**: Convert specs to NamedSharding

### Function Application

- **`tree_apply(fns, tree)`**: Apply dictionary of functions to tree

---

## Serialization

The `_serialization.py` module provides robust serialization capabilities for PyTrees using MessagePack format.

### Core Serialization

#### State Dict Operations

- **`to_state_dict(target)`**: Convert object to state dictionary
- **`from_state_dict(target, state, name=".")`**: Restore object from state dict
- **`register_serialization_state(ty, ty_to_state_dict, ty_from_state_dict, override=False)`**: Register custom type serialization

#### Binary Serialization

- **`to_bytes(target)`**: Serialize to MessagePack bytes
- **`from_bytes(target, encoded_bytes)`**: Deserialize from bytes
- **`msgpack_serialize(pytree, in_place=False)`**: Low-level MessagePack serialization
- **`msgpack_restore(encoded_pytree)`**: Low-level MessagePack deserialization

### File Operations

- **`save_to_file(target, filepath)`**: Save PyTree to file
- **`load_from_file(target, filepath)`**: Load PyTree from file

### Compression

- **`compress_bytes(data)`**: Compress bytes using gzip
- **`decompress_bytes(data)`**: Decompress gzipped bytes
- **`to_compressed_bytes(target)`**: Serialize and compress
- **`from_compressed_bytes(target, compressed_data)`**: Decompress and deserialize

### Validation and Information

- **`is_serializable(target)`**: Check if object is serializable
- **`validate_serializable(pytree)`**: Validate PyTree serializability
- **`get_serialization_info(pytree)`**: Get serialization metadata
- **`current_path()`**: Get current deserialization path (for debugging)

### Features

#### Automatic Chunking
Arrays larger than 2^30 bytes are automatically chunked for serialization to avoid MessagePack limits.

#### Type Support
Built-in support for:
- NumPy arrays and JAX arrays
- Python primitives (int, float, complex, bool)
- Containers (dict, list, tuple, namedtuple)
- JAX Partial functions

#### Array Addressability
The serialization system checks that JAX arrays are fully addressable before serialization, preventing issues with sharded arrays across multiple devices.

---

## Usage Examples

### Basic Tree Operations

```python
import jax
import jax.numpy as jnp
from eformer import pytree

# Create a sample tree
tree = {
    "layer1": {"weights": jnp.ones((10, 5)), "bias": jnp.zeros(5)},
    "layer2": {"weights": jnp.ones((5, 3)), "bias": jnp.zeros(3)}
}

# Get tree statistics
size = pytree.tree_size(tree)  # Total number of elements
bytes_used = pytree.tree_bytes(tree)  # Memory usage

# Arithmetic operations
tree2 = pytree.tree_multiply(tree, 2.0)  # Scale all values
tree_sum = pytree.tree_add(tree, tree2)  # Add trees

# Statistical operations
total = pytree.tree_sum(tree)
mean = pytree.tree_mean(tree)
l2_norm = pytree.tree_norm(tree, ord=2)
```

### Gradient Clipping

```python
# Clip gradients by value
gradients = {...}  # Your gradients
clipped = pytree.tree_clip(gradients, min_val=-1.0, max_val=1.0)

# Clip by norm
grad_norm = pytree.tree_norm(gradients)
max_norm = 5.0
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    gradients = pytree.tree_multiply(gradients, scale)
```

### Model Initialization

```python
key = jax.random.PRNGKey(42)

# Create template with desired structure
template = {
    "encoder": {"w": jnp.zeros((784, 256)), "b": jnp.zeros(256)},
    "decoder": {"w": jnp.zeros((256, 784)), "b": jnp.zeros(784)}
}

# Initialize with random values
model = pytree.tree_random_like(template, key, "normal")
```

### Serialization

```python
# Save model to file
pytree.save_to_file(model, "model.msgpack")

# Load model from file
loaded_model = pytree.load_from_file(template, "model.msgpack")

# Compression
compressed = pytree.to_compressed_bytes(model)
decompressed = pytree.from_compressed_bytes(template, compressed)

# Validation
is_valid, issues = pytree.validate_serializable(model)
if not is_valid:
    print("Serialization issues:", issues)

# Get serialization info
info = pytree.get_serialization_info(model)
print(f"Leaves: {info['num_leaves']}, Bytes: {info['memory_bytes']}")
```

### Path Operations

```python
# Map with path information
def path_printer(path, value):
    print(f"Path: {path}, Shape: {value.shape if hasattr(value, 'shape') else 'scalar'}")
    return value

pytree.tree_map_with_path(path_printer, tree)

# Get all paths and values
paths_and_values = pytree.tree_leaves_with_paths(tree)
for path, value in paths_and_values:
    print(f"{path}: {value}")
```

### Dictionary Flattening

```python
nested = {
    "model": {
        "encoder": {"weight": jnp.array([1, 2])},
        "decoder": {"weight": jnp.array([3, 4])}
    }
}

# Flatten with tuple keys
flat = pytree.flatten_dict(nested)
# {('model', 'encoder', 'weight'): array([1, 2]), ...}

# Flatten with string separator
flat_str = pytree.flatten_dict(nested, sep=".")
# {'model.encoder.weight': array([1, 2]), ...}

# Unflatten
restored = pytree.unflatten_dict(flat)
```

### Advanced Tree Manipulation

```python
# Split tree based on condition
def is_bias(x):
    return hasattr(x, 'shape') and len(x.shape) == 1

biases, weights = pytree.split(tree, is_bias)

# Merge trees
merged = pytree.merge(biases, weights)

# Filter tree
large_arrays = pytree.tree_filter(tree, lambda x: hasattr(x, 'size') and x.size > 10)

# Check for NaN/Inf
has_nans = pytree.tree_any(pytree.tree_isnan(tree))
tree_clean = pytree.tree_replace_nans(tree, 0.0)
```

---

## API Reference

### Type Aliases

```python
PyTree = Any  # Nested structure of arrays and containers
FilterSpec = bool | Callable[[Any], bool]
IsLeafFn = Callable[[Any], bool]
FnDict = dict[Any, Callable[[Any], Any]]
TreeDict = dict[Any, Any]
Path = tuple[Any, ...]
```

### Important Classes

#### `MetaValueRecreator`
Helper for recreating meta values with state tracking:
- `get_count()`: Get incrementing counter
- `get_rng()`: Get new random key

#### `StateValidationResult`
Validation result container:
- `is_valid`: Boolean validity flag
- `missing_keys`: Set of missing keys
- `invalid_types`: Dictionary of invalid types

### Error Handling

The serialization module provides detailed error messages with path information when deserialization fails, making debugging easier.

### Performance Considerations

1. **Memory**: Large trees are automatically chunked during serialization
2. **Sharding**: Arrays must be fully addressable for serialization
3. **Compression**: Use compressed variants for large models to save disk space
4. **In-place operations**: Some functions support in-place modification to save memory

### Best Practices

1. Always validate serializability before saving critical models
2. Use compression for long-term storage
3. Keep template structures for deserialization
4. Use path operations for debugging tree structures
5. Leverage tree arithmetic for optimizer implementations
6. Use `tree_clip` and `tree_norm` for gradient clipping
7. Prefer tree operations over manual loops for better performance

---

## Contributing

When adding new tree utilities:
1. Follow the existing naming convention (`tree_*` for operations)
2. Include comprehensive docstrings
3. Add type hints
4. Write corresponding tests
5. Update this documentation

## License

Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
Licensed under the Apache License, Version 2.0.