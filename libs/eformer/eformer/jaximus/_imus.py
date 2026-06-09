# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implicit Array System and JAX Primitive Interception.

This module provides a framework for creating and manipulating "implicit arrays" -
custom array-like objects that defer expensive operations (like materialization from
quantized formats) until absolutely necessary. It enables transparent integration with
JAX by intercepting primitive operations and routing them to custom handlers.

Core Components:
    - ImplicitArray: Abstract base class for lazy/deferred array representations
    - ste (Straight-Through Estimator): Decorator for gradient pass-through in quantization
    - use_implicit/implicit: Context manager for enabling implicit array dispatch
    - register: Decorator for registering custom primitive handlers
    - _CustomTrace: JAX trace implementation for intercepting operations

The system uses JAX's tracing infrastructure to intercept operations on ImplicitArray
instances and dispatch them to registered handlers, allowing custom behavior while
maintaining compatibility with JAX transformations (jit, grad, vmap, etc.).

Example:
    >>> from eformer.jaximus import ImplicitArray, register, implicit
    >>> from eformer.ops.quantization import ArrayNF4
    >>>
    >>> # Create a quantized weight array
    >>> weight = jnp.ones((128, 64), dtype=jnp.float32)
    >>> nf4_weight = ArrayNF4.quantize(weight, block_size=64)
    >>>
    >>> # Use implicit dispatch to avoid premature materialization
    >>> @implicit
    ... def linear(x, w):
    ...     return x @ w  # Uses custom dot_general handler for NF4
    >>>
    >>> output = linear(inputs, nf4_weight)  # NF4 kernel used, no materialization
"""

from __future__ import annotations

import abc
import dataclasses
import functools as ft
import itertools as it
import os
import typing as tp
import warnings
from abc import ABC
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, is_dataclass
from typing import TypeGuard

import chex
import jax
import jax._src
import jax._src.core as core
import jax._src.lax
import jax._src.pjit
import jax.extend
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import plum
from jax.custom_derivatives import SymbolicZero as SZ

WARN_ON_MATTER = os.environ.get("WARN_ON_MATTER", "true") in ["true", "yes", "1", "on"]
CT = tp.TypeVar("CT", bound=Callable)

StaticScalar = np.bool_ | np.number | bool | int | float | complex
ArrayLike = jax.Array | np.ndarray | StaticScalar


class OrginArray(ABC):  # noqa:B024
    """Abstract base class for array-like types in the implicit array system.

    This class serves as a registry for types that should be treated as
    "original" array types. jax.Array is automatically registered.

    The class is used in type checking and dispatch logic to determine
    whether a value is an array-like type that can participate in
    implicit array operations.
    """

    ...


OrginArray.register(jax.Array)


def ste(func):
    """
    Straight-Through Estimator (STE) decorator for quantization-aware training.

    This decorator enables gradient flow through non-differentiable quantization operations
    by using a custom VJP (vector-Jacobian product) that passes gradients straight through
    to the input, ignoring the quantization step in the backward pass.

    The STE is critical for training with quantized weights:
    - Forward pass: Uses the quantized representation (e.g., NF4, INT4)
    - Backward pass: Gradients flow to the full-precision master weights

    This allows training with quantization awareness while maintaining gradient-based
    optimization of the underlying float32 parameters.

    Args:
        func: A function that performs quantization or other non-differentiable operations.
              Signature: func(x: Array, *args, **kwargs) -> Array | ImplicitArray

    Returns:
        Wrapped function with straight-through gradient behavior.

    Example:
        >>> @ste
        ... def quantize_nf4(weights, block_size=64):
        ...     return ArrayNF4.quantize(weights, block_size)
        >>>
        >>> # Forward: uses quantized weights
        >>> # Backward: gradients flow to original float32 weights
        >>> quantized = quantize_nf4(fp32_weights)

    Technical Details:
        - Uses jax.custom_vjp to define custom backward pass behavior
        - Automatically materializes ImplicitArray cotangents to ensure valid gradients
        - Supports arbitrary positional and keyword arguments
        - Returns None for cotangents of non-differentiable arguments

    Note:
        The STE is a biased gradient estimator - it assumes the gradient of the
        quantization function is the identity. This works well in practice for
        quantization-aware training despite the theoretical bias.
    """

    @jax.custom_vjp
    @ft.wraps(func)
    def _wrapped(x: jax.Array | OrginArray, *args, **kwargs):
        return func(x, *args, **kwargs)

    def _fwd(x, *args, **kwargs):
        y = func(x, *args, **kwargs)
        return y, (len(args), tuple(kwargs.keys()))

    def _bwd(res, g):
        num_args, kw_keys = res

        def _materialize_if_needed(val):
            if isinstance(val, ImplicitArray):
                return val.materialize()
            return val

        if isinstance(g, ImplicitArray):
            g = g.materialize()
        else:
            g = tu.tree_map(_materialize_if_needed, g)
        cot_args = (g,) + (None,) * num_args
        if kw_keys:
            cot_kwargs = {k: None for k in kw_keys}
            return cot_args, cot_kwargs
        return cot_args

    _wrapped.defvjp(_fwd, _bwd)
    return _wrapped


def default_handler(primitive, *args, **params):
    """Default handler that executes a JAX primitive normally.

    Args:
        primitive: JAX primitive to execute.
        *args: Arguments to the primitive.
        **params: Parameters for the primitive.

    Returns:
        Result of executing the primitive with the given arguments.
    """
    return _bind_primitive(primitive, args, params)


def _materialize_all(vals):
    """Materialize all ImplicitArray instances in a sequence.

    Args:
        vals: Sequence of values, some of which may be ImplicitArrays.

    Returns:
        List with all ImplicitArrays replaced by their materialized form.
    """
    outs = []
    for val in vals:
        if hasattr(val, "materialize"):
            val = val.materialize()
        outs.append(val)
    return outs


def materialize_handler(primitive, *vals, params):
    """Handler that materializes all ImplicitArrays before executing primitive.

    This is the fallback handler used when no custom handler is registered
    for a primitive operation involving ImplicitArrays.

    Args:
        primitive: JAX primitive to execute.
        *vals: Values that may include ImplicitArrays.
        params: Parameters for the primitive.

    Returns:
        Result of executing the primitive after materializing all values.
    """
    vals = _materialize_all(vals)
    result = _bind_primitive(primitive, vals, params)
    return result


class UninitializedAval(Exception):
    """Exception raised when accessing uninitialized abstract value attributes.

    This is raised when trying to access shape or dtype on an ImplicitArray
    before these values have been computed or set.
    """

    ...


def aux_field(metadata=None, **kwargs):
    """
    Create a dataclass field marked as auxiliary (non-pytree) data.

    In ImplicitArray subclasses, fields can be either:
    1. Pytree children: Arrays and nested structures that should be traced by JAX
    2. Auxiliary data: Static metadata (ints, strings, dtypes) that don't participate in tracing

    This function creates fields marked as auxiliary, which means:
    - Not included in pytree flattening/unflattening
    - Not traced by JAX transformations (jit, grad, vmap)
    - Typically used for static configuration (block_size, dtype, mesh info)

    Args:
        metadata: Optional existing metadata dict to extend
        **kwargs: Additional arguments passed to dataclasses.field()

    Returns:
        A dataclass field with auxiliary metadata set

    Example:
        >>> from dataclasses import dataclass
        >>> from eformer.jaximus import ImplicitArray, aux_field
        >>>
        >>> @dataclass
        >>> class ArrayNF4(ImplicitArray):
        ...     # These are pytree children (traced by JAX)
        ...     packed: jax.Array
        ...     absmax: jax.Array
        ...
        ...     # These are auxiliary (static metadata)
        ...     block_size: int = aux_field()
        ...     dtype: jnp.dtype = aux_field(default=jnp.float32)
        ...     mesh_config: tuple | None = aux_field(default=None)

    Technical Details:
        - Sets metadata["implicit_array_aux"] = True
        - Used by tree_flatten_with_keys and tree_unflatten
        - Auxiliary fields are passed as aux_data in pytree registration
        - Must use this for non-array fields to avoid JAX tracing errors

    See Also:
        - ImplicitArray.tree_flatten_with_keys: Uses aux_field metadata
        - _get_names_and_aux: Helper that reads this metadata
    """

    metadata = dict(metadata) if metadata else {}
    metadata["implicit_array_aux"] = True
    return field(metadata=metadata, **kwargs)


class _AvalDescriptor:
    """Descriptor for lazy abstract value (aval) attributes.

    This descriptor provides lazy initialization for shape and dtype attributes
    on ImplicitArray subclasses. It stores values in a private attribute and
    raises UninitializedAval if accessed before being set.

    The descriptor enables ImplicitArray to defer shape/dtype computation
    until actually needed, supporting cases where these values are derived
    from materialization.
    """

    def __set_name__(self, owner, name):
        """Store the private attribute name when the descriptor is assigned."""
        self._name = f"_{name}"

    def __get__(self, obj, owner=None):
        """Get the attribute value, raising UninitializedAval if not set."""
        if obj is None:
            return None
        result = getattr(obj, self._name, None)
        if result is None:
            raise UninitializedAval()
        return result

    def __set__(self, obj, value):
        """Set the attribute value."""
        setattr(obj, self._name, value)


_aval_discovery = ContextVar("aval_discovery", default=False)
"""Context variable tracking whether we're in aval discovery mode."""


def _def_leaf(x):
    """Check if a value is an ImplicitArray leaf node for tree operations.

    Args:
        x: Value to check.

    Returns:
        True if x is an ImplicitArray instance.
    """
    return isinstance(x, ImplicitArray)


def use_implicit(fn):
    """
    Enable implicit array dispatch for a function.

    This decorator/wrapper sets up a custom JAX trace that intercepts operations on
    ImplicitArray instances and routes them to registered handlers. This allows
    transparent use of quantized or lazy arrays without manual materialization.

    Args:
        fn: Function to wrap with implicit array support.

    Returns:
        Wrapped function that handles ImplicitArray instances transparently.

    Example:
        >>> @use_implicit
        ... def matmul(x, w):
        ...     return x @ w  # Automatically uses custom dot_general for NF4
        >>>
        >>> # Or use as context:
        >>> with implicit:
        ...     output = inputs @ nf4_weights  # Custom handler dispatched

    Technical Details:
        - Creates a custom JAX trace (_CustomTrace) that intercepts primitive operations
        - Wraps ImplicitArray instances in _CustomTracer for operation interception
        - Dispatches to registered handlers via the @register decorator
        - Falls back to materialization if no handler is registered
        - Maintains compatibility with JAX transformations (jit, grad, vmap)

    See Also:
        - register: Decorator for registering custom primitive handlers
        - ImplicitArray: Base class for implicit array implementations
        - _CustomTrace: The trace implementation that handles dispatch
    """

    def implicit_f(*args, **kwargs):
        leaves, struct = tu.tree_flatten(
            (fn, args, kwargs),
            is_leaf=_def_leaf,
        )

        tag = core.TraceTag()
        with core.take_current_trace() as ctrace:
            trace = _CustomTrace(parent_trace=ctrace, tag=tag)
            leaves = tu.tree_map(
                ft.partial(_wrap_tracer, trace=trace),
                leaves,
                is_leaf=_def_leaf,
            )
            func, args, kwargs = tu.tree_unflatten(struct, leaves)
            with core.set_current_trace(trace):
                outs = func(*args, **kwargs)
            outs = tu.tree_map(ft.partial(_unwrap_tracer, trace=trace), outs)
        return outs

    return implicit_f


implicit = use_implicit  # Alias for convenience


def materialize_nested(implicit_arr, full=False):
    """Recursively materialize nested ImplicitArray structures.

    Args:
        implicit_arr: ImplicitArray or nested ImplicitArray to materialize.
        full: If True, recursively materialize until reaching a regular array.
              If False, only materialize one level.

    Returns:
        Materialized array. If materialization fails, returns an array of ones
        with the expected shape and dtype.
    """
    while isinstance(implicit_arr, ImplicitArray):
        try:
            implicit_arr = implicit_arr.materialize()
        except Exception:
            aval = implicit_arr.aval
            implicit_arr = jnp.ones(aval.shape, aval.dtype)
            break
        if not full:
            break
    return implicit_arr


def _get_materialization_aval(imp_arr):
    """Get the abstract value (shape/dtype) of a materialized ImplicitArray.

    Uses jax.eval_shape to determine the output shape without actually
    performing materialization.

    Args:
        imp_arr: ImplicitArray to get aval for.

    Returns:
        ShapedArray representing the materialized array's shape and dtype.
    """
    with _aval_discovery_context():
        result = jax.eval_shape(ft.partial(materialize_nested, full=True), imp_arr)
    return result


@contextmanager
def _aval_discovery_context():
    """Context manager for abstract value discovery mode.

    Sets the _aval_discovery context variable to True, which signals to
    tree operations that we're discovering avals and should handle
    uninitialized values gracefully.
    """
    token = _aval_discovery.set(True)
    try:
        yield
    finally:
        _aval_discovery.reset(token)


def is_array(element: tp.Any) -> bool:
    """Check if an element is an array type (NumPy or JAX).

    Args:
        element: Value to check.

    Returns:
        True if element is a NumPy array, NumPy scalar, or JAX array.
    """
    return isinstance(element, np.ndarray | np.generic | jax.Array)


@dataclass
class _ArrayBase(OrginArray, abc.ABC):
    """Base class providing common array attributes for ImplicitArray.

    This abstract base class defines the core attributes and class-level
    configuration that all ImplicitArray subclasses share.

    Class Attributes:
        commute_ops: If True, operations may be reordered for optimization.
        warn_on_materialize: If True, warn when falling back to materialization.
        default_shape: Default shape for the array type, if known statically.
        default_dtype: Default dtype for the array type, if known statically.

    Instance Attributes:
        shape: The logical shape of the array.
        dtype: The data type of the materialized array.
    """

    commute_ops: tp.ClassVar[bool] = True
    warn_on_materialize: tp.ClassVar[bool] = True

    default_shape: tp.ClassVar[tp.Sequence[int] | None] = None
    default_dtype: tp.ClassVar[jnp.dtype | None] = None

    shape: tp.Sequence[int] | None = aux_field(kw_only=True, default=None)
    dtype: jnp.dtype = aux_field(kw_only=True, default=None)  # noqa


@dataclass
class ImplicitArray(_ArrayBase):
    """
    Abstract base class for implicit (lazy/deferred) array representations.

    ImplicitArray enables custom array-like types that defer expensive operations
    until materialization is required. This is particularly useful for quantized
    arrays (NF4, INT4, INT8) where you want to:
    1. Store data in compressed format to save memory
    2. Perform operations directly on compressed data when possible (via custom kernels)
    3. Only materialize to full precision when necessary

    The ImplicitArray system integrates with JAX's tracing infrastructure to intercept
    operations and dispatch to custom handlers registered via @register decorator.

    Attributes:
        shape: Tuple representing the logical shape of the array.
               Uses _AvalDescriptor for lazy initialization.
        dtype: JAX dtype of the materialized array.
               Uses _AvalDescriptor for lazy initialization.

    Subclass Requirements:
        1. Must be a dataclass
        2. Must implement materialize() method
        3. Should register custom handlers for primitives via @register
        4. Non-array fields should use aux_field() to mark them as auxiliary

    Example Subclass:
        >>> from dataclasses import dataclass
        >>> from eformer.jaximus import ImplicitArray, aux_field, register
        >>>
        >>> @dataclass
        >>> class ArrayNF4(ImplicitArray):
        ...     packed: jax.Array           # 4-bit packed data
        ...     absmax: jax.Array           # Scale factors
        ...     block_size: int = aux_field()  # Static metadata
        ...
        ...     def materialize(self):
        ...         # Dequantize back to float32
        ...         return dequantize_nf4(self.packed, self.absmax, self.block_size)
        >>>
        >>> # Register custom handler for matrix multiplication
        >>> @register("dot_general")
        >>> def nf4_matmul(lhs, rhs: ArrayNF4, **kwargs):
        ...     # Use optimized NF4 kernel instead of materializing
        ...     return nf4_kernel(lhs, rhs.packed, rhs.absmax)

    Usage:
        >>> # Create quantized weight
        >>> weight = jnp.randn(128, 64)
        >>> nf4_weight = ArrayNF4.quantize(weight, block_size=64)
        >>>
        >>> # Use with implicit dispatch
        >>> @implicit
        ... def linear(x, w):
        ...     return x @ w  # Automatically uses nf4_matmul handler
        >>>
        >>> output = linear(inputs, nf4_weight)  # No materialization!

    Technical Details:
        - Registered as a JAX pytree for compatibility with transformations
        - Integrates with JAX's abstract value (aval) system for shape inference
        - Uses _AvalDescriptor for lazy shape/dtype discovery
        - Supports nested implicit arrays via tree_flatten_with_keys
        - Automatic materialization fallback when no custom handler exists

    See Also:
        - register: Decorator for registering custom primitive handlers
        - use_implicit/implicit: Enable implicit array dispatch
        - ste: Straight-through estimator for quantization-aware training
    """

    shape = _AvalDescriptor()
    dtype = _AvalDescriptor()

    def __post_init__(self):
        try:
            aval = _get_materialization_aval(self)
        except UninitializedAval:
            aval = None
        shape = None
        try:
            shape = self.shape
        except UninitializedAval:
            shape = self.shape = self.compute_shape()

        if aval is not None:
            if shape is None:
                self.shape = aval.shape
            elif shape != aval.shape:
                warnings.warn(
                    f"ImplicitArray shape {shape} does not match materialization shape {aval.shape}",
                    stacklevel=1,
                )
        elif shape is None:
            raise UninitializedAval("shape")

        dtype = None
        try:
            dtype = self.dtype
        except UninitializedAval:
            dtype = self.dtype = self.compute_dtype()

        if dtype is None and aval is None:
            aval = _get_materialization_aval(self)

        if aval is not None:
            if dtype is None:
                self.dtype = aval.dtype
            elif dtype != aval.dtype:
                warnings.warn(
                    f"ImplicitArray dtype {dtype} does not match materialization dtype {aval.dtype}",
                    stacklevel=1,
                )
        elif dtype is None:
            raise UninitializedAval("dtype")

    def compute_shape(self):
        return self.default_shape

    def compute_dtype(self):
        return self.default_dtype

    @property
    def aval(self):
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    @abc.abstractmethod
    def materialize(self): ...

    def tree_flatten_with_keys(self):
        children = []
        aux_data = []
        for name, is_aux in _get_names_and_aux(self):
            try:
                value = getattr(self, name)
            except UninitializedAval:
                if not _aval_discovery.get():
                    raise
                value = None
            if is_aux:
                aux_data.append(value)
            else:
                children.append((tu.GetAttrKey(name), value))

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_it = iter(children)
        aux_it = iter(aux_data)
        obj = cls.__new__(cls)
        for name, is_aux in _get_names_and_aux(cls):
            value = next(aux_it if is_aux else child_it)
            setattr(obj, name, value)

        return obj

    def astype(self, new_dtype):
        self.dtype = new_dtype
        return self

    def __init_subclass__(cls, commute_ops=True, warn_on_materialize=True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.commute_ops = commute_ops
        cls.warn_on_materialize = warn_on_materialize

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        core.pytype_aval_mappings[cls] = lambda x: x.aval
        tu.register_pytree_with_keys_class(cls)
        return cls


def _get_names_and_aux(obj):
    """Get field names and auxiliary flags for a dataclass.

    Yields tuples of (field_name, is_auxiliary) for each field in the dataclass.

    Args:
        obj: Dataclass instance or type.

    Yields:
        Tuples of (str, bool) where the bool indicates if the field is auxiliary.
    """
    for val in dataclasses.fields(obj):
        yield val.name, bool(val.metadata.get("implicit_array_aux"))


def combine_leaf_predicate(base_fn, is_leaf):
    """Wrap a tree function to include ImplicitArray as a leaf type.

    Creates a new function that combines the given is_leaf predicate with
    an additional predicate, allowing custom leaf detection.

    Args:
        base_fn: Original tree function (e.g., tu.tree_map).
        is_leaf: Predicate function to treat values as leaves.

    Returns:
        Wrapped function that uses the combined leaf predicate.
    """

    @ft.wraps(base_fn)
    def new_fn(*args, new_is_leaf=None):
        if new_is_leaf is None:
            combined_is_leaf = is_leaf
        else:

            def combined_is_leaf(arg):
                return is_leaf(arg) or new_is_leaf(arg)

        return base_fn(*args, is_leaf=combined_is_leaf)

    return new_fn


def leaf_predicate(x):
    """Predicate for identifying ImplicitArray leaves in tree operations.

    Args:
        x: Value to check.

    Returns:
        True if x is an ImplicitArray instance.
    """
    return isinstance(x, ImplicitArray)


# Tree utilities that treat ImplicitArray as leaves
# These functions are versions of jax.tree_util functions that recognize
# ImplicitArray instances as leaf nodes rather than traversing into them.

tree_map_with_implicit = combine_leaf_predicate(
    tu.tree_map,
    leaf_predicate,
)
"""Like jax.tree_util.tree_map but treats ImplicitArray as leaves."""

tree_map_with_path_with_implicit = combine_leaf_predicate(
    tu.tree_map_with_path,
    leaf_predicate,
)
"""Like jax.tree_util.tree_map_with_path but treats ImplicitArray as leaves."""

tree_flatten_with_implicit = combine_leaf_predicate(
    tu.tree_flatten,
    leaf_predicate,
)
"""Like jax.tree_util.tree_flatten but treats ImplicitArray as leaves."""

tree_flatten_with_path_with_implicit = combine_leaf_predicate(
    tu.tree_flatten_with_path,
    leaf_predicate,
)
"""Like jax.tree_util.tree_flatten_with_path but treats ImplicitArray as leaves."""

tree_leaves_with_implicit = combine_leaf_predicate(
    tu.tree_leaves,
    leaf_predicate,
)
"""Like jax.tree_util.tree_leaves but treats ImplicitArray as leaves."""

tree_structure_with_implicit = combine_leaf_predicate(
    tu.tree_structure,
    leaf_predicate,
)
"""Like jax.tree_util.tree_structure but treats ImplicitArray as leaves."""


def flatten_one_implicit_layer(tree):
    """Flatten one layer of nested ImplicitArrays in a tree.

    For nested ImplicitArray structures, this function flattens just one level,
    treating nested ImplicitArrays as leaves.

    Args:
        tree: Pytree potentially containing nested ImplicitArrays.

    Returns:
        Tuple of (leaves, structure) where leaves are flattened one level
        and structure is the pytree structure.
    """

    def is_leaf_below_node(node, x):
        return isinstance(x, ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return tu.tree_map(
            lambda _: 1,
            node,
            is_leaf=ft.partial(is_leaf_below_node, node),
        )

    prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = tu.tree_structure(prototype)

    leaves = tree_leaves_with_implicit(tree)
    leaves = list(
        it.chain.from_iterable(
            (
                tu.tree_leaves(leaf, is_leaf=ft.partial(is_leaf_below_node, leaf))
                if isinstance(leaf, ImplicitArray)
                else [leaf]
            )
            for leaf in leaves
        )
    )
    return leaves, struct


def implicit_depth(tree):
    """Calculate the maximum nesting depth of ImplicitArrays in a tree.

    Args:
        tree: Pytree potentially containing nested ImplicitArrays.

    Returns:
        Integer depth of nesting. 0 means no ImplicitArrays, 1 means
        ImplicitArrays with no nesting, etc.
    """
    leaves = tree_leaves_with_implicit(tree)
    depth = 0
    while True:
        next_leaves = []
        any_implicit = False
        for leaf in leaves:
            if not isinstance(leaf, ImplicitArray):
                continue
            any_implicit = True
            next_leaves.extend(flatten_one_implicit_layer(leaf)[0])

        if not any_implicit:
            return depth

        depth += 1
        leaves = next_leaves


def _map_leaves_with_implicit_path(f, leaves, is_leaf, path_prefix=()):
    """Map a function over leaves with path tracking for nested ImplicitArrays.

    Recursively maps a function over leaves in a tree structure, keeping track
    of the path to each leaf for nested ImplicitArray structures.

    Args:
        f: Function to apply to each leaf.
        leaves: List of leaf values.
        is_leaf: Predicate function (path, leaf) -> bool to stop recursion.
        path_prefix: Tuple prefix for the current path in the tree.

    Returns:
        List of mapped leaf values with nested structures preserved.
    """
    mapped_leaves = []
    for idx, leaf in enumerate(leaves):
        path = (*path_prefix, idx)
        if not isinstance(leaf, ImplicitArray) or is_leaf(path, leaf):
            mapped_leaves.append(f(leaf))
            continue

        subtree, substruct = flatten_one_implicit_layer(leaf)
        mapped_subtree = _map_leaves_with_implicit_path(
            f,
            subtree,
            is_leaf=is_leaf,
            path_prefix=path,
        )
        mapped_leaves.append(tu.tree_unflatten(substruct, mapped_subtree))
    return mapped_leaves


_rules: dict[core.Primitive, plum.Function] = {}


def register(
    primitive: core.Primitive | str,
    *,
    precedence: int = 0,
) -> Callable[[CT], CT]:
    """
    Register a custom handler for a JAX primitive operation.

    This decorator allows you to define custom behavior for JAX primitives (like
    dot_general, add, mul, etc.) when operating on ImplicitArray instances. Handlers
    are dispatched via multiple dispatch based on argument types.

    Args:
        primitive: JAX primitive to register handler for. Can be:
                  - A core.Primitive object (e.g., jax.lax.dot_general_p)
                  - A string name (e.g., "dot_general") - automatically resolved to primitive
        precedence: Handler precedence for multiple dispatch (higher = higher priority).
                   Default is 0.

    Returns:
        Decorator that registers the handler function.

    Example - Basic Usage:
        >>> from eformer.jaximus import register
        >>> from jax.extend.core import Primitive
        >>>
        >>> @register("dot_general")
        >>> def nf4_matmul(lhs: jax.Array, rhs: ArrayNF4, **kwargs):
        ...     # Custom matmul for dense @ NF4
        ...     return nf4_kernel(lhs, rhs.packed, rhs.absmax)

    Example - Multiple Dispatch:
        >>> @register("dot_general")
        >>> def nf4_lhs_matmul(lhs: ArrayNF4, rhs: jax.Array, **kwargs):
        ...     # Handle NF4 @ dense (different from above)
        ...     return nf4_transpose_kernel(rhs, lhs.packed, lhs.absmax)
        >>>
        >>> # Both handlers registered; dispatched based on argument types
        >>> dense @ nf4_weight  # Uses first handler
        >>> nf4_weight @ dense  # Uses second handler

    Example - Precedence:
        >>> @register("add", precedence=10)
        >>> def high_priority_add(x: ArrayNF4, y: ArrayNF4, **kwargs):
        ...     # This handler takes precedence over lower-priority ones
        ...     return optimized_nf4_add(x, y)

    Technical Details:
        - Uses plum for multiple dispatch based on type signatures
        - Handlers are called within _CustomTrace.process_primitive
        - If no handler matches, falls back to materialization
        - Supports string primitive names (auto-resolved from jax.lax)
        - Multiple handlers per primitive allowed (dispatched by type)

    Handler Signature:
        Handlers receive:
        - primitive: The JAX primitive being executed (sometimes)
        - *args: Operands (ImplicitArray or regular arrays)
        - **kwargs: Primitive parameters (dimension_numbers, precision, etc.)

        And should return:
        - Result array (can be ImplicitArray or regular array)

    Common Primitives to Register:
        - "dot_general": Matrix multiplication (x @ y)
        - "add", "sub", "mul", "div": Arithmetic operations
        - "reshape", "transpose": Shape operations
        - "reduce": Reductions (sum, max, etc.)
        - "convert_element_type": Type conversions

    See Also:
        - ImplicitArray: Base class for custom array types
        - use_implicit/implicit: Enable handler dispatch
        - _CustomTrace.process_primitive: Dispatch implementation
    """

    if isinstance(primitive, str):
        lac = getattr(jax.lax, f"{primitive}_p", None)
        if lac is None:
            raise ValueError(f"couldn't verify given string primitive {primitive}_p")
        primitive = lac

    def _register(rule: CT) -> CT:
        try:
            existing_rule = _rules[primitive]
        except KeyError:

            def existing_rule():
                raise AssertionError()

            existing_rule.__name__ = f"{primitive}_dispatcher"
            existing_rule.__qualname__ = f"{primitive}_dispatcher"
            existing_rule = plum.Dispatcher().abstract(existing_rule)

            _rules[primitive] = existing_rule
        existing_rule.dispatch(rule, precedence=precedence)
        return rule

    return _register


def _default_process(
    primitive: core.Primitive,
    values: Sequence[chex.Array | ImplicitArray],
    params,
):
    arrays = _materialize_values(values)
    return _bind_primitive(primitive, arrays, params)


def _bind_primitive(primitive: core.Primitive, args: Sequence[tp.Any], params):
    bind_spec = primitive.get_bind_params(params)
    if isinstance(bind_spec, tuple) and len(bind_spec) == 2:
        subfuns, bind_params = bind_spec
    else:
        bind_params = dict(bind_spec)
        subfuns = bind_params.pop("subfuns", ())
    return primitive.bind(*subfuns, *args, **bind_params)


def _materialize_values(values: Sequence[chex.Array | ImplicitArray]) -> list[tp.Any]:
    arrays: list[tp.Any] = []
    for x in values:
        if _is_value(x):
            arrays.append(x.materialize())
        elif is_array(x):
            arrays.append(tp.cast(chex.Array, x))
        else:
            arrays.append(x)
    return arrays


def _wrap_tracer(x, trace: _CustomTrace):
    """Wrap an ImplicitArray value in a CustomTracer for dispatch.

    Args:
        x: Value to potentially wrap.
        trace: CustomTrace instance to associate with the tracer.

    Returns:
        _CustomTracer if x is an ImplicitArray, otherwise x unchanged.
    """
    if _is_value(x):
        return _CustomTracer(trace, x)
    else:
        return x


def _unwrap_tracer(x, trace):
    """Unwrap a CustomTracer to get the underlying value.

    Args:
        x: Value to potentially unwrap.
        trace: CustomTrace instance for raising arrays.

    Returns:
        The underlying value if x is a CustomTracer, otherwise x unchanged.
    """
    if is_array(x):
        x = trace.full_raise(x)
    if isinstance(x, _CustomTracer):
        return x.value
    else:
        return x


class _CustomTracer(core.Tracer):
    """JAX tracer that wraps ImplicitArray values for operation interception.

    This tracer is used within the implicit array dispatch system to wrap
    ImplicitArray instances. When operations are performed on traced values,
    they are intercepted by the associated _CustomTrace and routed to
    registered handlers.

    Attributes:
        value: The wrapped ImplicitArray instance.
    """

    __slots__ = ("value",)

    def __init__(self, trace: _CustomTrace, value: ImplicitArray) -> None:
        """Initialize the tracer with a trace and value.

        Args:
            trace: The CustomTrace this tracer belongs to.
            value: The ImplicitArray to wrap.
        """
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        """Get the abstract value (shape/dtype) of the wrapped array."""
        return self.value.aval

    def full_lower(self):
        """Lower the tracer to a concrete value if possible."""
        if isinstance(self.value, ImplicitArray):
            return self
        else:
            return core.full_lower(self.value)


class _CustomTrace(core.Trace):
    """JAX trace implementation for implicit array dispatch.

    This trace intercepts primitive operations and routes them to registered
    handlers when ImplicitArray instances are involved. It integrates with
    JAX's tracing infrastructure to provide transparent operation dispatch.

    Attributes:
        tag: Unique identifier for this trace instance.
        parent_trace: The enclosing trace to delegate to when needed.
    """

    def __init__(self, parent_trace, tag):
        """Initialize the trace with a parent and unique tag.

        Args:
            parent_trace: Enclosing JAX trace.
            tag: Unique TraceTag for this trace instance.
        """
        super().__init__()
        self.tag = tag
        self.parent_trace = parent_trace

    def to_value(self, val):
        """Extract the value from a tracer if it belongs to this trace.

        Args:
            val: Tracer or value to extract.

        Returns:
            The underlying value if val is a matching tracer, otherwise val.
        """
        if isinstance(val, _CustomTracer) and val._trace.tag is self.tag:
            return val.value
        return val

    def process_primitive(self, primitive, tracers, params):
        """Process a JAX primitive operation, dispatching to custom handlers.

        This is the core dispatch method. It extracts values from tracers,
        looks up registered handlers for the primitive, and either:
        1. Calls a matching handler if registered
        2. Falls back to materializing ImplicitArrays and calling the primitive

        Args:
            primitive: JAX primitive being executed.
            tracers: List of traced values (operands).
            params: Parameters for the primitive.

        Returns:
            _CustomTracer wrapping the result (or list of tracers for multi-output).
        """
        values = [self.to_value(t) for t in tracers]
        values = tuple(values)
        implicit_idx = next(
            (i for i, v in enumerate(values) if isinstance(v, ImplicitArray)),
            None,
        )
        implicit_name = None
        if implicit_idx is not None:
            implicit_name = values[implicit_idx].__class__.__name__
            try:
                rule = _rules[primitive]
            except KeyError:
                with core.set_current_trace(self.parent_trace):
                    if WARN_ON_MATTER and implicit_name is not None:
                        warnings.warn(
                            f"No Custom Primitive been found for {primitive} (materializing {implicit_name})",
                            stacklevel=1,
                        )
                    out = _default_process(primitive, values, params)
            else:
                include_prim = False
                with core.set_current_trace(self.parent_trace):
                    try:
                        try:
                            method, _ = rule.resolve_method(values)
                        except (plum.NotFoundLookupError, plum.AmbiguousLookupError):
                            inhint = (primitive, *tuple(values))
                            include_prim = True
                            method, _ = rule.resolve_method(inhint)
                    except (plum.NotFoundLookupError, plum.AmbiguousLookupError):
                        if WARN_ON_MATTER and implicit_name is not None:
                            warnings.warn(
                                f"No Custom Primitive could match for {primitive} (materializing {implicit_name})",
                                stacklevel=1,
                            )
                        out = _default_process(primitive, values, params)
                    else:
                        if include_prim:
                            values = (primitive, *tuple(values))
                        out = method(*values, **params)
        else:
            with core.set_current_trace(self.parent_trace):
                out = _bind_primitive(primitive, values, params)
        if primitive.multiple_results:
            out = [_CustomTracer(self, x) for x in out]
        else:
            out = _CustomTracer(self, out)
        return out

    def process_shard_map(self: _CustomTrace, primitive, fun, tracers, **params):
        """Process shard_map primitives by materializing and delegating.

        Shard maps require concrete arrays, so ImplicitArrays are materialized
        before execution.
        """
        tracers = [(arr.materialize() if _is_value(arr) else arr) for arr in [self.to_value(t) for t in tracers]]
        out = primitive.bind_with_trace(self.parent_trace, (fun, *tracers), params)
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_map(self, map_primitive, f, tracers, params=None, /, **kwargs):
        """Process map primitives (pmap, etc.) by materializing inputs."""
        if params is None:
            params = kwargs
        elif kwargs:
            params = {**params, **kwargs}
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(map_primitive, in_values, params)
        if map_primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_custom_transpose(self, prim, call, tracers, **params):
        """Process custom transpose primitives by materializing inputs."""
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(prim, in_values, params)
        if prim.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_call(self, call_primitive, f, tracers, params):
        """Process call primitives by materializing inputs."""
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(call_primitive, in_values, params)
        if call_primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_custom_jvp_call(self, primitive, fun, *args, symbolic_zeros=None, **kwargs):
        """Process custom JVP calls by materializing inputs.

        Custom JVP requires concrete arrays, so all ImplicitArrays are
        materialized before the forward pass is executed.
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments in process_custom_jvp_call: {unexpected}")
        if len(args) == 2:
            jvp, tracers = args
        elif len(args) == 5:
            _fwd, _bwd, tracers, _out_trees, symbolic_zeros = args
            jvp = _fwd
        else:
            raise TypeError(f"Unexpected process_custom_jvp_call arguments: {len(args)} positional values")
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            arrays = _materialize_values(in_values)
            out_leaves = primitive.bind_with_trace(
                self.parent_trace,
                tuple(arrays),
                tuple(core.typeof(x) for x in arrays),
                {"subfuns": (fun, jvp), "symbolic_zeros": symbolic_zeros},
            )
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out_leaves]
        else:
            return _CustomTracer(self, out_leaves)

    def process_custom_vjp_call(
        self,
        primitive,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees=None,
        symbolic_zeros=None,
        **kwargs,
    ):
        """Process custom VJP calls by materializing inputs.

        Custom VJP requires concrete arrays for the forward pass,
        so all ImplicitArrays are materialized before execution.
        """
        if out_trees is None and "out_trees" in kwargs:
            out_trees = kwargs.pop("out_trees")
        if symbolic_zeros is None and "symbolic_zeros" in kwargs:
            symbolic_zeros = kwargs.pop("symbolic_zeros")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments in process_custom_vjp_call: {unexpected}")
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            arrays = _materialize_values(in_values)
            out_leaves = primitive.bind_with_trace(
                self.parent_trace,
                tuple(arrays),
                tuple(core.typeof(x) for x in arrays),
                {
                    "subfuns": (fun, fwd, bwd),
                    "out_trees": out_trees,
                    "symbolic_zeros": symbolic_zeros,
                },
            )
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out_leaves]
        else:
            return _CustomTracer(self, out_leaves)


def _custom_vjp_fwd_wrap(fwd, tag, in_treedef):
    """Wrap a custom VJP forward function for use with ImplicitArrays.

    This wrapper handles the flattening/unflattening of pytrees to work
    with JAX's custom VJP infrastructure.

    Note: This function is defined twice; the second definition is the
    active one that returns the output tree structure.
    """

    def wrapped(*args):
        inputs = tu.tree_unflatten(in_treedef, args)
        out = fwd(*inputs)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, out_tree = tu.tree_flatten(out)
        return out_flat, out_tree

    return wrapped, None


def _custom_vjp_bwd_wrap(bwd, tag, in_treedef):
    """Wrap a custom VJP backward function for use with ImplicitArrays.

    This wrapper handles the flattening/unflattening of pytrees to work
    with JAX's custom VJP infrastructure for backward passes.
    """

    def wrapped(*args):
        res_and_cts = tu.tree_unflatten(in_treedef, args)
        out = bwd(*res_and_cts)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, out_tree = tu.tree_flatten(out)
        return out_flat, out_tree

    return wrapped, None


@lu.transformation_with_aux
def _custom_jvp_fun_wrap(tag, in_treedef, *in_leaves):
    in_values = tu.tree_unflatten(in_treedef, in_leaves)
    with core.take_current_trace() as parent_trace:
        trace = _CustomTrace(parent_trace, tag)
        in_tracers = [x if type(x) is SZ else _CustomTracer(trace, x) for x in in_values]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            out_tracers = [jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t for t in out_tracers]
            out_values = [trace.to_value(t) for t in out_tracers]
            del out_tracers
        del trace, in_tracers
    out_leaves, out_treedef = tu.tree_flatten(out_values)
    yield out_leaves, out_treedef


@lu.transformation_with_aux
def _custom_jvp_jvp_wrap(tag, in_treedef, *in_primals_and_tangents):
    in_primals = in_primals_and_tangents[: len(in_primals_and_tangents) // 2]
    in_tangents = in_primals_and_tangents[len(in_primals_and_tangents) // 2 :]
    in_primal_values = tu.tree_unflatten(in_treedef, in_primals)
    in_tangent_values = tu.tree_unflatten(in_treedef, in_tangents)
    with core.take_current_trace() as parent_trace:
        trace = _CustomTrace(parent_trace, tag)
        in_tracers = [_CustomTracer(trace, x) for x in it.chain(in_primal_values, in_tangent_values)]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            out_tracers = [jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t for t in out_tracers]
            out_values = [trace.to_value(t) for t in out_tracers]
            out_primal_values = out_values[: len(out_values) // 2]
            out_tangent_values = out_values[len(out_values) // 2 :]
            out_primal_values2 = []
            out_tangent_values2 = []
            if len(out_primal_values) != len(out_tangent_values):
                raise ValueError("Primals and tangents length mismatch.")
            for primal, tangent in zip(out_primal_values, out_tangent_values):  # noqa
                if primal.__class__ != tangent.__class__:
                    primal = primal.materialize()
                    tangent = tangent.materialize()
                out_primal_values2.append(primal)
                out_tangent_values2.append(tangent)
            del out_tracers
        del trace, in_tracers
    out_primals, out_primal_treedef = tu.tree_flatten(out_primal_values2)
    out_tangents, out_tangent_treedef = tu.tree_flatten(out_tangent_values2)
    if out_primal_treedef != out_tangent_treedef:
        raise ValueError("Primals and tangents had the same class, but different flattened results.")
    yield out_primals + out_tangents, out_primal_treedef


def _is_value(x) -> TypeGuard[ImplicitArray]:
    """Type guard to check if a value is an ImplicitArray.

    Args:
        x: Value to check.

    Returns:
        True if x is an ImplicitArray, with type narrowing for static analysis.
    """
    return isinstance(x, ImplicitArray)


@register(jax._src.pjit.jit_p)
def _(
    *args: ImplicitArray | ArrayLike,
    jaxpr,
    inline,
    **kwargs,
):
    del kwargs
    fun = use_implicit(jax.extend.core.jaxpr_as_fun(jaxpr))
    return fun(*args)
    if inline:
        return fun(*args)
    else:
        leaves, treedef = tu.tree_flatten(args)
        flat_fun = lambda x: fun(*tu.tree_unflatten(treedef, x))  # noqa
        return jax.jit(flat_fun)(leaves)


_sentinel = object()


@register(jax.lax.while_p)
def _(
    *args: ImplicitArray | ArrayLike,
    cond_nconsts: int,
    cond_jaxpr,
    body_nconsts: int,
    body_jaxpr,
):
    cond_consts = args[:cond_nconsts]
    body_consts = args[cond_nconsts : cond_nconsts + body_nconsts]
    init_vals = args[cond_nconsts + body_nconsts :]

    quax_cond_fn = implicit(core.jaxpr_as_fun(cond_jaxpr))
    quax_cond_jaxpr = jax.make_jaxpr(quax_cond_fn)(*cond_consts, *init_vals)
    quax_body_fn = implicit(core.jaxpr_as_fun(body_jaxpr))
    quax_body_jaxpr = jax.make_jaxpr(quax_body_fn)(*body_consts, *init_vals)

    cond_leaves, _ = tu.tree_flatten(cond_consts)
    body_leaves, _ = tu.tree_flatten(body_consts)
    init_val_leaves, val_treedef = tu.tree_flatten(init_vals)
    try:
        out_val = jax.lax.while_p.bind(
            *cond_leaves,
            *body_leaves,
            *init_val_leaves,
            cond_nconsts=cond_nconsts,
            cond_jaxpr=quax_cond_jaxpr,
            body_nconsts=body_nconsts,
            body_jaxpr=quax_body_jaxpr,
        )
    except Exception as e:
        raise RuntimeError("You should customize while prim for your usecase") from e
    result = tu.tree_unflatten(val_treedef, out_val)
    return result


@register("cond")
def _(
    index: chex.Array,
    *args: tp.Any,
    branches: tuple,
    linear=_sentinel,
):
    flat_args, in_tree = tu.tree_flatten(args)

    out_trees = []
    _branches = []
    for jaxpr in branches:

        def flat__call(flat_args):
            args = tu.tree_unflatten(in_tree, flat_args)
            out = use_implicit(core.jaxpr_as_fun(jaxpr))(*args)  # noqa
            flat_out, out_tree = tu.tree_flatten(out)
            out_trees.append(out_tree)
            return flat_out

        _jaxpr = jax.make_jaxpr(flat__call)(flat_args)
        _branches.append(_jaxpr)

    if tp.Any(tree_outs_i != out_trees[0] for tree_outs_i in out_trees[1:]):
        raise TypeError("all branches output must have the same pytree.")

    if linear is _sentinel:
        maybe_linear = {}
    else:
        maybe_linear = dict(linear=linear)
    out_val = jax.lax.cond_p.bind(index, *flat_args, branches=tuple(_branches), **maybe_linear)
    result = tu.tree_unflatten(out_trees[0], out_val)
    return result
