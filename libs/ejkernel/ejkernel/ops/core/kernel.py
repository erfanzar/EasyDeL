# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Core kernel infrastructure for configurable JAX operations.

This module provides the foundational classes for implementing high-performance
JAX operations with automatic configuration optimization and caching.

Key Classes:
    Invocation: Represents a specific call to a kernel with arguments and metadata
    Kernel: Abstract base class for implementing configurable operations

The kernel system enables:
    - Automatic hyperparameter optimization through configuration testing
    - Caching of optimal configurations for performance
    - Custom gradient implementations with VJP support
    - Flexible argument preprocessing and transformation
    - Device-aware configuration management

Kernel Implementation Pattern:
    1. Inherit from Kernel[ConfigType, OutputType]
    2. Implement run() method for the core operation
    3. Implement heuristic_cfg() for default configuration
    4. Optionally implement candidate_cfgs() for autotuning
    5. Optionally implement custom VJP methods for gradients

Example Implementation:
    >>> @dataclass
    >>> class MatMulConfig:
    ...     precision: str = 'default'
    ...     transpose_a: bool = False
    >>>
    >>> class MatMulKernel(Kernel[MatMulConfig, jax.Array]):
    ...     def run(self, a, b, cfg: MatMulConfig) -> jax.Array:
    ...         return jnp.dot(a, b, precision=cfg.precision)
    ...
    ...     def heuristic_cfg(self, inv) -> MatMulConfig:
    ...         return MatMulConfig()
    ...
    ...     def candidate_cfgs(self, inv):
    ...         return [MatMulConfig(p) for p in ['float32', 'bfloat16']]

Invocation Usage:
    The Invocation class captures all information needed for a kernel call:
    - Arguments and their shapes/types
    - Optional configuration overrides
    - Batching information for vmapping
    - Profiling and caching metadata

This design enables seamless integration with the autotuning and caching
system while providing a clean interface for operation implementers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic

import jax
import jax.sharding
from jax import shard_map

from ..utils.fingerprint import abstractify, short_hash
from .types import Cfg, Out


@dataclasses.dataclass(frozen=True)
class Invocation(Generic[Cfg, Out]):
    """Immutable snapshot of a single kernel call, including its arguments and execution context.

    :class:`Invocation` is the unit of work flowing through the config-selection
    and execution pipeline.  It is created by :class:`~ejkernel.ops.execution.Executor`
    after calling :meth:`~ejkernel.ops.core.Kernel.prepare`, then passed to
    :class:`~ejkernel.ops.config.ConfigSelectorChain` to look up or compute the
    optimal configuration.

    The :attr:`call_key` and :meth:`make_key` methods derive a stable hash from
    argument shapes/dtypes (not values) so the same configuration can be reused
    for different arrays that share the same structure.

    Attributes:
        op_id: Unique operation identifier, used as part of the cache key.
        args: Positional arguments for the kernel (after ``kernel.prepare()``).
        kwargs: Keyword arguments for the kernel (after ``kernel.prepare()``).
        batch_axes: Optional mapping of parameter names to batch axes, set when
            the invocation is issued inside a vmap context.
        override_cfg: If not ``None``, bypasses all cache and autotuning logic
            and uses this configuration directly.  The chosen value is also
            written back to the in-memory and persistent caches.
        stamp: Whether :class:`~ejkernel.ops.execution.Executor` should inject
            profiling metadata into the compiled graph (default ``True``).
        method: Execution mode.  Currently the only non-default value is
            ``'shard_map'``, which causes the executor to wrap the call with
            ``jax.shard_map``.
        mesh: :class:`jax.sharding.Mesh` required when ``method='shard_map'``.
        in_specs: Per-argument ``PartitionSpec`` tuple for ``shard_map`` inputs.
        out_specs: ``PartitionSpec`` for the ``shard_map`` output.
        check_vma: Passed as ``check_vma`` to ``jax.shard_map``; when ``True``,
            JAX verifies that non-sharded inputs are replicated across all devices.
    """

    op_id: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    batch_axes: Mapping[str, int] | None = None
    override_cfg: Cfg | None = None
    stamp: bool = True
    method: str | None = None
    mesh: jax.sharding.Mesh | None = None
    in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None
    out_specs: jax.sharding.PartitionSpec | None = None
    check_vma: bool = False

    @property
    def call_key(self) -> str:
        """Generate a stable hash key for this invocation based on argument shapes and types.

        Creates a 16-character hash that uniquely identifies this invocation
        based on the abstract shapes and types of arguments, not their values.
        This enables caching of configurations based on operation signature.

        Returns:
            16-character hexadecimal hash string representing the call signature

        Note:
            The hash includes argument shapes/types, keyword argument shapes/types,
            batch axes information, and execution method. Array values are not included,
            allowing the same configuration to be reused for arrays with the same structure.
        """
        spec = dict(
            args_spec=abstractify(self.args),
            kwargs_spec=abstractify(dict(self.kwargs)),
            batch_axes=self.batch_axes,
            method=self.method,
        )
        return short_hash(spec)

    def make_key(self, key_builder=None) -> str:
        """Generate a cache key for this invocation, optionally using a custom key builder.

        Provides flexibility in cache key generation by allowing custom key builders
        while falling back to the default implementation.

        Args:
            key_builder: Optional function that takes an Invocation and returns a key

        Returns:
            Cache key string for this invocation

        Note:
            Custom key builders can include additional information like sharding
            or device placement for more sophisticated caching strategies.
        """
        if key_builder is not None:
            return key_builder(self)
        spec = dict(
            args_spec=abstractify(self.args),
            kwargs_spec=abstractify(dict(self.kwargs)),
            batch_axes=self.batch_axes,
            method=self.method,
        )
        return short_hash(spec)


class Kernel(Generic[Cfg, Out]):
    """Abstract base class for configurable JAX operations with autotuning support.

    A :class:`Kernel` encapsulates the logic for a single operation together
    with all the information required to select, cache, and validate its
    execution configuration.  Subclasses are registered with an
    :class:`~ejkernel.ops.execution.Executor` that handles the full lifecycle:

    1. **Preprocessing** — :meth:`prepare` transforms raw call arguments.
    2. **Config selection** — :class:`~ejkernel.ops.config.ConfigSelectorChain`
       looks up or autotuners the best :class:`Cfg`.
    3. **Execution** — :meth:`run` (or a context/platform-specific variant) is
       called with the chosen config.
    4. **Gradients** — if :meth:`fwd_with_residuals` and :meth:`vjp` are both
       overridden, JAX's ``custom_vjp`` mechanism is used automatically.

    **Required overrides:**

    * :meth:`run` — core computation.
    * :meth:`heuristic_cfg` — fast, always-correct default configuration.

    **Optional overrides:**

    * :meth:`prepare` — argument preprocessing / validation.
    * :meth:`candidate_cfgs` — configurations to benchmark during autotuning.
    * :meth:`fwd_with_residuals` + :meth:`vjp` — custom gradient pair.
    * ``run_{context}``, ``run_{platform}``, ``run_{context}_{platform}`` — specialised
      execution variants (see method-naming convention below).
    * Same naming pattern for ``fwd_with_residuals`` / ``vjp`` variants.

    **Method-naming convention:**

    The execution system resolves the most specific available method at runtime
    using :func:`_get_platform_method`:

    ============================================  ================================
    Name pattern                                  Example
    ============================================  ================================
    ``{method}_{context}_{platform}`` (highest)  ``run_shard_map_gpu``
    ``{method}_{context}``                        ``run_shard_map``
    ``{method}_{platform}``                       ``run_gpu``
    ``{method}`` (lowest / generic)               ``run``
    ============================================  ================================

    Known platforms: ``'gpu'``, ``'tpu'``, ``'cpu'``.
    Known contexts: ``'shard_map'``.

    Attributes:
        op_id: Unique string identifier for this operation.  Used as part of
            every cache key and in profiling labels.  May be set as a class
            attribute or passed to :meth:`__init__`.
        key_builder: Optional callable ``(Invocation) -> str`` for generating
            custom cache keys.  Useful when sharding or other metadata not
            captured by the default hash should affect caching.
        version: String version tag appended to ``op_id`` in cache keys as
            ``"{op_id}@v{version}"``.  Increment this to invalidate existing
            cached configurations when the kernel implementation changes
            (default: ``"0"``).
    """

    op_id: str
    key_builder: Callable[[Invocation[Cfg, Out]], str] | None = None
    version: str = "0"

    def __init__(self, op_id: str | None = None):
        """Initialize a new Kernel instance with an operation identifier.

        Sets up the kernel with a unique operation ID for caching, profiling,
        and identification purposes. The op_id can be explicitly provided,
        inherited from a class attribute, or auto-generated from the class name.

        Args:
            op_id: Optional unique identifier for this operation. If not provided,
                uses the class attribute op_id if set, otherwise generates an ID
                from the module and class name (e.g., 'mymodule.MyKernel').

        Example:
            >>> class MatMulKernel(Kernel):
            ...     pass
            >>>
            >>> # Explicit op_id
            >>> kernel = MatMulKernel("custom_matmul")
            >>> kernel.op_id
            'custom_matmul'
            >>>
            >>> # Auto-generated op_id
            >>> kernel = MatMulKernel()
            >>> kernel.op_id
            '__main__.MatMulKernel'
        """
        if op_id is not None:
            self.op_id = op_id
        elif getattr(self, "op_id", None):
            pass
        else:
            self.op_id = f"{type(self).__module__}.{type(self).__name__}"

    def prepare(self, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Preprocess arguments before execution. Override to modify args/kwargs.

        This method is called before the run() method to allow transformation
        of arguments. Common use cases include shape validation, type conversion,
        or argument reordering.

        Args:
            *args: Positional arguments to preprocess
            **kwargs: Keyword arguments to preprocess

        Returns:
            Tuple of (processed_args, processed_kwargs)

        Example:
            >>> def prepare(self, x, y, **kwargs):
            ...
            ...     x = jnp.asarray(x)
            ...     y = jnp.asarray(y)
            ...     return (x, y), kwargs
        """
        return args, kwargs

    def run(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute the operation with the given configuration. Must be implemented.

        This is the core method that performs the actual computation. It receives
        the preprocessed arguments and a configuration object, and must return
        the operation result.

        Args:
            *args: Positional arguments (after prepare() preprocessing)
            cfg: Configuration object specifying how to execute the operation
            **kwargs: Keyword arguments (after prepare() preprocessing)

        Returns:
            Result of the operation

        Raises:
            NotImplementedError: Must be overridden in subclasses

        Example:
            >>> def run(self, x, y, cfg: MatMulConfig) -> jax.Array:
            ...     if cfg.transpose_a:
            ...         x = x.T
            ...     return jnp.dot(x, y, precision=cfg.precision)
        """
        raise NotImplementedError

    def heuristic_cfg(self, inv: Invocation[Cfg, Out]) -> Cfg:
        """Return a reasonable default configuration for this invocation. Must be implemented.

        Provides a sensible default configuration based on the invocation context.
        This configuration should work correctly for the given arguments, though
        it may not be optimal for performance.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration object

        Raises:
            NotImplementedError: Must be overridden in subclasses

        Example:
            >>> def heuristic_cfg(self, inv) -> MatMulConfig:
            ...
            ...     dtype = inv.args[0].dtype
            ...     precision = 'bfloat16' if dtype == jnp.bfloat16 else 'float32'
            ...     return MatMulConfig(precision=precision)
        """
        raise NotImplementedError

    def candidate_cfgs(self, inv: Invocation[Cfg, Out]) -> Iterable[Cfg]:
        """Return alternative configurations for autotuning. Defaults to just heuristic_cfg.

        Provides a set of configurations to test during autotuning. The autotuning
        system will benchmark each configuration and select the fastest one.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Iterable of configuration objects to test

        Example:
            >>> def candidate_cfgs(self, inv):
            ...     return [
            ...         MatMulConfig(precision='float32', transpose_a=False),
            ...         MatMulConfig(precision='bfloat16', transpose_a=False),
            ...         MatMulConfig(precision='float32', transpose_a=True),
            ...     ]

        Note:
            The default implementation returns only the heuristic configuration.
            Override this method to enable autotuning with multiple options.
        """
        return [self.heuristic_cfg(inv)]

    def fwd_with_residuals(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass that returns residuals for custom VJP. Implement for custom gradients.

        When implementing custom gradients, this method performs the forward pass
        and returns both the result and any residual values needed for the
        backward pass.

        Args:
            *args: Positional arguments for the operation
            cfg: Configuration object
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, residuals)
            - operation_result: Same as run() method output
            - residuals: Any values needed for the backward pass

        Raises:
            NotImplementedError: Only implement if providing custom gradients

        Example:
            >>> def fwd_with_residuals(self, x, y, cfg):
            ...     result = jnp.dot(x, y)
            ...     residuals = (x, y, cfg)
            ...     return result, residuals

        Note:
            Must be implemented together with vjp() method for custom gradients.
        """
        raise NotImplementedError

    def vjp(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for custom VJP. Return gradients for positional args only.

        Computes vector-Jacobian products (gradients) for the custom operation.
        This method is called during backpropagation to compute gradients with
        respect to the positional arguments.

        Args:
            residuals: Values returned from fwd_with_residuals()
            y: Forward pass output (from fwd_with_residuals())
            dy: Incoming gradients (cotangents) with respect to y
            *args: Original positional arguments
            cfg: Configuration object
            **kwargs: Original keyword arguments

        Returns:
            Tuple of gradients for each positional argument
            (None for arguments that don't need gradients)

        Raises:
            NotImplementedError: Only implement if providing custom gradients

        Example:
            >>> def vjp(self, residuals, y, dy, *args, cfg, **kwargs):
            ...     x, y_orig, _ = residuals
            ...     dx = jnp.dot(dy, y_orig.T)
            ...     dy_orig = jnp.dot(x.T, dy)
            ...     return dx, dy_orig

        Note:
            Must be implemented together with fwd_with_residuals() method.
        """
        raise NotImplementedError

    def run_shard_map(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute the operation within a shard_map context. Optional override.

        Implement this method to provide a version of the operation that runs
        efficiently inside ``jax.shard_map``.  When this override is present,
        :func:`_get_platform_method` will select it over the generic :meth:`run`
        whenever ``context='shard_map'`` (i.e. when the executor is called with
        ``method='shard_map'``).

        If this method is not overridden, the execution system falls back to the
        generic :meth:`run` method.

        Args:
            *args: Positional arguments (after :meth:`prepare` preprocessing).
            cfg: Configuration object specifying how to execute the operation.
            **kwargs: Keyword arguments (after :meth:`prepare` preprocessing).

        Returns:
            Result of the operation.

        Raises:
            NotImplementedError: Base implementation raises this; override in
                subclasses to provide shard_map-specific execution.

        Example:
            >>> def run_shard_map(self, x, y, cfg: MatMulConfig) -> jax.Array:
            ...     return custom_sharded_matmul(x, y, cfg)
        """
        raise NotImplementedError

    def fwd_with_residuals_shard_map(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass with residuals for shard_map context. Optional override.

        Specialized forward pass for shard_map contexts that returns both the
        result and residuals needed for the backward pass.

        Args:
            *args: Positional arguments for the operation
            cfg: Configuration object
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, residuals)
            - operation_result: Same as run_shard_map() method output
            - residuals: Any values needed for the backward pass

        Raises:
            NotImplementedError: Only implement if providing custom gradients for shard_map

        Note:
            Must be implemented together with vjp_shard_map() method for custom
            gradients in shard_map contexts.
        """
        raise NotImplementedError

    def vjp_shard_map(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for shard_map context. Optional override.

        Specialized backward pass for shard_map contexts that computes gradients
        with respect to positional arguments.

        Args:
            residuals: Values returned from fwd_with_residuals_shard_map()
            y: Forward pass output (from fwd_with_residuals_shard_map())
            dy: Incoming gradients (cotangents) with respect to y
            *args: Original positional arguments
            cfg: Configuration object
            **kwargs: Original keyword arguments

        Returns:
            Tuple of gradients for each positional argument
            (None for arguments that don't need gradients)

        Raises:
            NotImplementedError: Only implement if providing custom gradients for shard_map

        Note:
            Must be implemented together with fwd_with_residuals_shard_map() method.
        """
        raise NotImplementedError

    def run_shard_map_gpu(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute operation in shard_map context on GPU. Optional override.

        Most specific implementation combining shard_map execution context with
        GPU platform optimizations. This method is automatically selected when
        running with method='shard_map' on a GPU device.

        Args:
            *args: Positional arguments (after prepare() preprocessing)
            cfg: Configuration object specifying how to execute the operation
            **kwargs: Keyword arguments (after prepare() preprocessing)

        Returns:
            Result of the operation

        Raises:
            NotImplementedError: Only implement if providing GPU-specific shard_map execution

        Example:
            >>> def run_shard_map_gpu(self, x, y, cfg: MyConfig) -> jax.Array:
            ...     # Use GPU-specific CUDA kernels in sharded context
            ...     return gpu_sharded_operation(x, y, cfg)

        Note:
            This is the highest priority method for shard_map execution on GPU,
            following the priority chain: composite > context > platform > generic.
        """
        raise NotImplementedError

    def fwd_with_residuals_shard_map_gpu(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass with residuals for shard_map on GPU. Optional override.

        Specialized forward pass that combines shard_map context with GPU platform
        optimizations. Returns both the result and residuals needed for the backward
        pass in custom gradient computations.

        Args:
            *args: Positional arguments for the operation
            cfg: Configuration object
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, residuals)
            - operation_result: Same as run_shard_map_gpu() method output
            - residuals: Any values needed for the backward pass

        Raises:
            NotImplementedError: Only implement if providing custom gradients for
                GPU-specific shard_map execution

        Note:
            Must be implemented together with vjp_shard_map_gpu() method for custom
            gradients in GPU shard_map contexts.
        """
        raise NotImplementedError

    def vjp_shard_map_gpu(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for shard_map on GPU. Optional override.

        Specialized backward pass that combines shard_map context with GPU platform
        optimizations. Computes gradients with respect to positional arguments.

        Args:
            residuals: Values returned from fwd_with_residuals_shard_map_gpu()
            y: Forward pass output (from fwd_with_residuals_shard_map_gpu())
            dy: Incoming gradients (cotangents) with respect to y
            *args: Original positional arguments
            cfg: Configuration object
            **kwargs: Original keyword arguments

        Returns:
            Tuple of gradients for each positional argument
            (None for arguments that don't need gradients)

        Raises:
            NotImplementedError: Only implement if providing custom gradients for
                GPU-specific shard_map execution

        Note:
            Must be implemented together with fwd_with_residuals_shard_map_gpu() method.
        """
        raise NotImplementedError

    def create_shard_map_wrapper(
        self,
        fn: Callable,
        *args,
        mesh: jax.sharding.Mesh,
        in_specs: tuple[jax.sharding.PartitionSpec, ...],
        out_specs: jax.sharding.PartitionSpec,
        check_vma: bool = False,
        **kwargs,
    ) -> tuple[Callable, tuple]:
        """Create a shard_map wrapper around a function with fixed kwargs.

        This helper method simplifies the pattern of wrapping functions with shard_map
        by handling the functools.partial and shard_map setup automatically.

        Args:
            fn: Function to wrap with shard_map (e.g., flash_attention)
            *args: Positional arguments that will be passed through shard_map
            mesh: JAX device mesh for distributed execution
            in_specs: Partition specs for input arguments (must match len(args))
            out_specs: Partition spec for output
            check_vma: Whether to check replication in shard_map
            **kwargs: Keyword arguments to fix via functools.partial

        Returns:
            Tuple of (shard_map_wrapped_function, call_arguments)
            - shard_map_wrapped_function: Function that takes positional args
            - call_arguments: Tuple of args to pass to the wrapped function

        Example:
            >>> from ejkernel.modules.operations import flash_attention
            >>> attn = FlashAttention()
            >>>
            >>>
            >>> shard_map_fn, call_args = attn.create_shard_map_wrapper(
            ...     flash_attention,
            ...     query, key, value,
            ...     mesh=my_mesh,
            ...     in_specs=(q_spec, k_spec, v_spec),
            ...     out_specs=out_spec,
            ...     causal=True,
            ...     softmax_scale=0.125
            ... )
            >>>
            >>>
            >>> output = shard_map_fn(*call_args)

        Note:
            This follows the EasyDeL pattern where attention operations are wrapped
            with shard_map for distributed execution. The function `fn` is called
            with the positional args and fixed kwargs inside the shard_map context.
        """

        def _wrapped(*call_args):
            """Inner function that calls fn with positional args and fixed kwargs inside shard_map."""
            return fn(*call_args, **kwargs)

        shard_map_fn = shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, args


def _has_custom_vjp(
    k: Kernel,
    platform: str | None = None,
    context: str | None = None,
) -> bool:
    """Determine whether a kernel provides a custom VJP implementation.

    Returns ``True`` if *both* the ``fwd_with_residuals`` and ``vjp`` methods
    (or their context/platform-specific variants) have been overridden from the
    base :class:`Kernel` stubs.  The same priority chain as
    :func:`_get_platform_method` is followed:

    1. Composite ``{method}_{context}_{platform}`` pair (e.g.
       ``fwd_with_residuals_shard_map_gpu`` + ``vjp_shard_map_gpu``).
    2. Context-specific pair (e.g. ``fwd_with_residuals_shard_map`` + ``vjp_shard_map``).
    3. Platform-specific pair (e.g. ``fwd_with_residuals_gpu`` + ``vjp_gpu``).
    4. Generic pair (``fwd_with_residuals`` + ``vjp``).

    *Both* methods of a pair must be overridden; overriding only one is not
    sufficient.

    Args:
        k: Kernel instance to inspect.
        platform: Optional platform identifier (e.g. ``'gpu'``, ``'tpu'``, ``'cpu'``).
        context: Optional execution-context identifier (e.g. ``'shard_map'``).

    Returns:
        ``True`` if the kernel provides a matching custom VJP pair at any
        specificity level; ``False`` otherwise (including on ``AttributeError``).
    """
    try:
        if context and platform:
            composite_fwd = f"fwd_with_residuals_{context}_{platform}"
            composite_vjp = f"vjp_{context}_{platform}"

            has_composite_fwd = hasattr(type(k), composite_fwd) and getattr(type(k), composite_fwd) is not getattr(
                Kernel, composite_fwd, None
            )
            has_composite_vjp = hasattr(type(k), composite_vjp) and getattr(type(k), composite_vjp) is not getattr(
                Kernel, composite_vjp, None
            )

            if has_composite_fwd and has_composite_vjp:
                return True

        if context:
            context_fwd = f"fwd_with_residuals_{context}"
            context_vjp = f"vjp_{context}"

            has_context_fwd = hasattr(type(k), context_fwd) and getattr(type(k), context_fwd) is not getattr(
                Kernel, context_fwd, None
            )
            has_context_vjp = hasattr(type(k), context_vjp) and getattr(type(k), context_vjp) is not getattr(
                Kernel, context_vjp, None
            )

            if has_context_fwd and has_context_vjp:
                return True

        if platform:
            platform_fwd = f"fwd_with_residuals_{platform}"
            platform_vjp = f"vjp_{platform}"

            has_platform_fwd = hasattr(type(k), platform_fwd) and getattr(type(k), platform_fwd) is not getattr(
                Kernel, platform_fwd, None
            )
            has_platform_vjp = hasattr(type(k), platform_vjp) and getattr(type(k), platform_vjp) is not getattr(
                Kernel, platform_vjp, None
            )

            if has_platform_fwd and has_platform_vjp:
                return True

        return type(k).fwd_with_residuals is not Kernel.fwd_with_residuals and type(k).vjp is not Kernel.vjp
    except AttributeError:
        return False


def _get_platform_method(
    k: Kernel,
    method_name: str,
    platform: str | None = None,
    context: str | None = None,
) -> Callable | None:
    """Look up the most specific method override on a kernel, following the priority chain.

    Supports execution contexts (e.g. ``'shard_map'``) combined with hardware
    platforms (e.g. ``'gpu'``, ``'tpu'``, ``'cpu'``).  The lookup priority is:

    1. ``{method}_{context}_{platform}`` — e.g. ``run_shard_map_gpu``
    2. ``{method}_{context}``           — e.g. ``run_shard_map``
    3. ``{method}_{platform}``          — e.g. ``run_gpu``
    4. ``{method}``                     — e.g. ``run`` (generic fallback)

    A method is only considered an "override" if it is not the same object as
    the corresponding method on the base :class:`Kernel` class.  This prevents
    the base ``NotImplementedError`` stubs from being returned as valid overrides.

    Args:
        k: Kernel instance to inspect.
        method_name: Base method name (e.g. ``'run'``, ``'candidate_cfgs'``,
            ``'fwd_with_residuals'``).
        platform: Optional platform identifier (``'gpu'``, ``'tpu'``, ``'cpu'``).
            Derived from :func:`~ejkernel.ops.utils.fingerprint.get_device_platform`.
        context: Optional execution-context identifier (currently only
            ``'shard_map'`` is recognised).

    Returns:
        The most specific bound method that overrides the base class, or ``None``
        if no override exists at any priority level.

    Example:
        >>> method = _get_platform_method(kernel, 'run', platform='gpu', context='shard_map')
        >>> if method:
        ...     result = method(*args, cfg=cfg)
    """

    if context and platform:
        name = f"{method_name}_{context}_{platform}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if context:
        name = f"{method_name}_{context}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if platform:
        name = f"{method_name}_{platform}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if hasattr(k, method_name):
        method = getattr(k, method_name)
        base = getattr(Kernel, method_name, None)
        if method is not base:
            return method
    return None
