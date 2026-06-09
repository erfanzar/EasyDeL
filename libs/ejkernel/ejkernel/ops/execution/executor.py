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


"""Main execution engine for kernels with configuration management.

This module provides the Executor class, which serves as the central orchestrator
for running kernel operations with automatic configuration selection, custom
gradient support, and comprehensive profiling capabilities.

Key Components:
    Executor: Main execution engine coordinating the entire execution pipeline
    ConfigChooser: Protocol defining configuration selection interface

The Executor handles:
    - Argument preprocessing via kernel.prepare()
    - Configuration selection through ConfigChooser strategies
    - Custom VJP (Vector-Jacobian Product) implementation for gradients
    - Profiling metadata injection for performance analysis
    - Invocation recording for batch optimization
    - JAX compilation with pre-selected configurations

Execution Flow:
    1. Preprocess arguments using kernel.prepare()
    2. Create Invocation object with argument metadata
    3. Select configuration via ConfigChooser.choose()
    4. Set up custom VJP if kernel implements it
    5. Add profiling metadata based on environment settings
    6. Execute kernel with chosen configuration
    7. Record invocation for future optimization (if enabled)

Environment Variables:
    EJKERNEL_OPS_RECORD: Set to "1" to enable invocation recording
    EJKERNEL_OPS_STAMP: Controls profiling metadata format:
        - "hash": Use operation hash for labeling (default)
        - "json": Use full JSON payload for labeling
        - "none": Disable profiling metadata

Example Usage:
    >>> cache = ConfigCache()
    >>> selector = ConfigSelectorChain(cache)
    >>> executor = Executor(selector)
    >>>
    >>>
    >>> result = executor(my_kernel, input_data)
    >>>
    >>>
    >>> compiled_fn = executor.compile(my_kernel, example_input)
    >>> result = compiled_fn(actual_input)
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from enum import Enum
from typing import Generic, Literal, Protocol

import jax
import jax.numpy as jnp
import jax.sharding
import jax.tree_util as jtu

from ...kernels._registry import Backend, Platform, kernel_registry
from ..config.cache import _cache_overlay
from ..core import Invocation, Kernel, _get_platform_method, _has_custom_vjp
from ..core.types import Cfg, Out
from ..utils.fingerprint import abstractify, device_fingerprint, get_device_platform, stable_json


class ConfigChooser(Protocol):
    """Protocol for configuration selection strategies.

    Defines the interface that configuration selection strategies must implement.
    The primary implementer is ConfigSelectorChain, which provides a sophisticated
    multi-tier selection system with caching and autotuning.

    Methods:
        choose: Select optimal configuration for the given invocation and kernel
    """

    def choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Select optimal configuration for the given invocation.

        Args:
            inv: Invocation object containing arguments and metadata
            kernel: Kernel implementation requiring configuration

        Returns:
            Configuration object suitable for the kernel and invocation
        """
        ...


class Executor(Generic[Cfg, Out]):
    """Main execution engine for kernels with automatic configuration selection.

    The Executor coordinates the entire execution process:
    1. Preprocess arguments via kernel.prepare()
    2. Select configuration via the ConfigChooser
    3. Handle custom VJP if implemented by the kernel
    4. Add profiling metadata if requested
    5. Execute the kernel with the chosen configuration

    Supports both regular operations and custom gradient implementations.

    Attributes:
        chooser: Configuration selection strategy (typically ConfigSelectorChain)
        stamp_prefix: Prefix for profiling metadata labels
    """

    def __init__(self, chooser: ConfigChooser, stamp_prefix: str = "ejkernel_ops"):
        """Initialize executor with configuration chooser and profiling settings.

        Args:
            chooser: Configuration selection strategy (typically ConfigSelectorChain)
            stamp_prefix: Prefix for profiling metadata labels in compiled code
        """
        self.chooser = chooser
        self.stamp_prefix = stamp_prefix

    @staticmethod
    def _platform_value(val) -> str | None:
        """Convert a platform value to a lowercase string representation.

        Normalizes platform identifiers from various formats (Enum, string, etc.)
        into a consistent lowercase string for comparison and matching.

        Args:
            val: Platform value to normalize. Can be None, an Enum member,
                or any object convertible to string.

        Returns:
            Lowercase string representation of the platform, or None if val is None.
        """
        if val is None:
            return None
        if isinstance(val, Enum):
            return str(val.value).lower()
        return str(val).lower()

    @staticmethod
    def _is_nvidia_gpu() -> bool:
        """Detect whether the current GPU device is an NVIDIA GPU.

        Checks the JAX backend platform version and device properties to
        determine if the system has an NVIDIA (CUDA-capable) GPU. Distinguishes
        NVIDIA GPUs from AMD (ROCm) GPUs by examining platform version strings
        and device kind identifiers.

        Returns:
            True if an NVIDIA GPU is detected, False otherwise (including AMD
            GPUs, no GPUs, or detection failure).

        Note:
            First checks the XLA backend platform version for 'cuda' or 'rocm'
            strings. If inconclusive, falls back to inspecting individual device
            kind attributes for NVIDIA-specific keywords (nvidia, tesla, geforce,
            rtx, quadro).
        """
        try:
            from jax.lib import xla_bridge

            platform_version = xla_bridge.get_backend().platform_version
            if isinstance(platform_version, str):
                pv = platform_version.lower()
                if "cuda" in pv:
                    return True
                if "rocm" in pv:
                    return False
        except Exception:
            pass

        try:
            devices = jax.devices("gpu")
        except Exception:
            devices = []
        if not devices:
            try:
                devices = jax.devices()
            except Exception:
                devices = []
        for dev in devices:
            kind = (getattr(dev, "device_kind", "") or "").lower()
            if any(token in kind for token in ("nvidia", "tesla", "geforce", "rtx", "quadro")):
                return True
        return False

    @staticmethod
    def _has_cuda_impl(algorithm: str) -> bool:
        """Check if a native CUDA implementation exists for the given algorithm.

        Queries the kernel registry to determine whether a CUDA-platform
        implementation is registered for the specified algorithm name.

        Args:
            algorithm: Algorithm/kernel name to look up (typically kernel.op_id).

        Returns:
            True if a CUDA implementation exists in the registry, False if not
            found or if the registry query fails.
        """
        try:
            specs = kernel_registry.list_implementations(algorithm)
        except Exception:
            return False
        return any(spec.platform == Platform.CUDA and spec.backend in (Backend.GPU, Backend.ANY) for spec in specs)

    @staticmethod
    def _has_cute_impl(algorithm: str) -> bool:
        """Check if a CuTe DSL implementation exists for the given algorithm.

        Queries the kernel registry to determine whether a CUTE-platform
        implementation is registered for the specified algorithm name.

        Args:
            algorithm: Algorithm/kernel name to look up (typically ``kernel.op_id``).

        Returns:
            True if a CuTe implementation exists in the registry, False if not
            found or if the registry query fails.
        """
        try:
            specs = kernel_registry.list_implementations(algorithm)
        except Exception:
            return False
        return any(spec.platform == Platform.CUTE and spec.backend in (Backend.GPU, Backend.ANY) for spec in specs)

    def _prefer_cuda_cfg(self, cfg: Cfg, kernel: Kernel[Cfg, Out], inv: Invocation[Cfg, Out]) -> Cfg:
        """Upgrade configuration to prefer CUTE/CUDA when conditions are met.

        Automatically switches the platform field in a configuration from 'auto'
        or 'triton' to 'cute'/'cuda' when all of the following conditions are satisfied:
        - The current configuration platform is 'auto' or 'triton'
        - No explicit override configuration was provided
        - No explicit platform was specified in invocation kwargs
        - The current device is a GPU
        - The GPU is an NVIDIA GPU (not AMD ROCm)
        - A CuTe or CUDA implementation exists in the kernel registry

        This enables transparent use of optimized CUTE/CUDA kernels when
        available, while gracefully falling back to Triton implementations
        on unsupported platforms.

        Args:
            cfg: Current configuration object (may have 'platform' and/or
                'backend' attributes).
            kernel: Kernel instance being executed.
            inv: Current invocation with arguments and metadata.

        Returns:
            Modified configuration with platform set to 'cuda' and backend
            set to 'gpu' if upgrade conditions are met, otherwise the
            original configuration unchanged.
        """
        platform_val = self._platform_value(getattr(cfg, "platform", None))
        if platform_val is None or platform_val in ("cuda", "cute"):
            return cfg
        if platform_val not in ("auto", "triton"):
            return cfg
        if inv.override_cfg is not None:
            return cfg
        explicit_platform = None
        if isinstance(inv.kwargs, dict):
            explicit_platform = self._platform_value(inv.kwargs.get("platform", None))
        if explicit_platform not in (None, "auto"):
            return cfg
        if get_device_platform() != "gpu":
            return cfg
        if not self._is_nvidia_gpu():
            return cfg
        has_cute = self._has_cute_impl(kernel.op_id)
        has_cuda = self._has_cuda_impl(kernel.op_id)
        if not has_cute and not has_cuda:
            return cfg

        if dataclasses.is_dataclass(cfg):
            fields = {field.name for field in dataclasses.fields(cfg)}
            updates = {}
            if "platform" in fields:
                updates["platform"] = "cute" if has_cute else "cuda"
            if "backend" in fields:
                updates["backend"] = "gpu"
            if updates:
                try:
                    return dataclasses.replace(cfg, **updates)
                except Exception:
                    pass
        try:
            if hasattr(cfg, "platform"):
                cfg.platform = "cute" if has_cute else "cuda"
            if hasattr(cfg, "backend"):
                cfg.backend = "gpu"
        except Exception:
            return cfg
        return cfg

    def _stamp_hash(self, kernel, inv, fn, cfg):
        """Add hash-based profiling metadata to function.

        Creates a compact label using operation ID and call signature hash
        for performance profiling and debugging.

        Args:
            kernel: Kernel being executed
            inv: Invocation object
            fn: Function to wrap with profiling metadata
            cfg: Configuration being used

        Returns:
            Function wrapped with hash-based profiling label
        """
        call_key = inv.make_key(kernel.key_builder)
        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        label = f"{self.stamp_prefix}#{op_id_v}:{call_key}"
        return self._stamp(label, fn)

    def _stamp_json(self, kernel, inv, fn, cfg):
        """Add JSON-based profiling metadata to function.

        Creates detailed profiling metadata including full operation context,
        arguments, and configuration for comprehensive debugging.

        Args:
            kernel: Kernel being executed
            inv: Invocation object
            fn: Function to wrap with profiling metadata
            cfg: Configuration being used

        Returns:
            Function wrapped with JSON profiling metadata

        Note:
            This mode provides more detailed information but may impact
            performance due to larger metadata payloads.
        """

        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        payload = stable_json(
            dict(
                op_id=op_id_v,
                args=abstractify(inv.args),
                kwargs=abstractify(dict(inv.kwargs)),
                cfg=cfg,
            )
        )

        def wrapped(*a, **k):
            """Execute the function within a JAX named scope containing the JSON payload."""
            with jax.named_scope(f"{self.stamp_prefix}:{payload}"):
                return fn(*a, **k)

        return wrapped

    def _stamp(self, name: str, fn: Callable) -> Callable:
        """Add profiling metadata to function using JAX naming primitives.

        Uses JAX's named_call if available, otherwise falls back to named_scope
        for adding operation labels to compiled code.

        Args:
            name: Label to attach to the operation
            fn: Function to wrap with profiling metadata

        Returns:
            Function wrapped with profiling label
        """
        if hasattr(jax, "named_call"):
            return jax.named_call(fn, name=name)

        def wrapped(*a, **k):
            """Execute the function within a JAX named scope for profiling."""
            with jax.named_scope(name):
                return fn(*a, **k)

        return wrapped

    def __call__(
        self,
        kernel: Kernel[Cfg, Out],
        *args,
        cfg: Cfg | None = None,
        stamp: bool = True,
        method: Literal["shard_map"] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None,
        out_specs: jax.sharding.PartitionSpec | None = None,
        check_vma: bool = False,
        **kwargs,
    ) -> Out:
        """Execute kernel with automatic configuration selection and management.

        This is the main execution method that orchestrates the complete execution
        pipeline including preprocessing, configuration selection, custom gradients,
        profiling, and invocation recording.

        Args:
            kernel: Kernel implementation to execute.
            *args: Positional arguments forwarded to ``kernel.prepare()`` and
                then to ``kernel.run()``.
            cfg: Optional configuration override.  When provided, bypasses all
                cache and autotuning logic (equivalent to setting
                ``Invocation.override_cfg``).  Can also be passed as the
                keyword argument ``_cfg`` for legacy call-sites.
            stamp: Whether to inject profiling metadata into the compiled graph.
                The format is controlled by the ``EJKERNEL_OPS_STAMP``
                environment variable:
                ``'hash'`` (default when stamp=True) — compact ``op_id:call_key`` label;
                ``'json'`` — full JSON payload with args/kwargs/cfg details;
                ``'none'`` — no label (same as stamp=False).
            method: Execution mode.  ``'shard_map'`` wraps the call with
                ``jax.shard_map`` using the provided ``mesh``/``in_specs``/
                ``out_specs``.  ``None`` (default) uses standard execution.
            mesh: :class:`jax.sharding.Mesh` required when ``method='shard_map'``.
            in_specs: Per-argument ``PartitionSpec`` tuple, required when
                ``method='shard_map'``.
            out_specs: ``PartitionSpec`` for the output, required when
                ``method='shard_map'``.
            check_vma: Passed to ``jax.shard_map`` as ``check_vma``; when
                ``True``, JAX checks that non-sharded inputs are replicated.
            **kwargs: Keyword arguments forwarded to ``kernel.prepare()`` and
                then to ``kernel.run()``.

        Returns:
            Result of kernel execution with the chosen configuration.

        Note:
            Custom VJP is detected automatically via
            :func:`~ejkernel.ops.core._has_custom_vjp`.  If the kernel overrides
            both ``fwd_with_residuals`` and ``vjp``, a ``jax.custom_vjp`` wrapper
            is created transparently so that ``jax.grad`` works correctly.

            When ``method='shard_map'``, the executor calls
            ``kernel.create_shard_map_wrapper`` instead of ``kernel.run``,
            so custom VJP is bypassed.  Kernels must handle gradients inside the
            shard_map wrapper in that case.

            Invocations are recorded to the global registry when the environment
            variable ``EJKERNEL_OPS_RECORD=1`` is set.
        """

        if "_cfg" in kwargs:
            cfg = kwargs.pop("_cfg")

        if method == "shard_map":
            if mesh is None:
                raise ValueError("mesh must be provided when method='shard_map'")
            if in_specs is None:
                raise ValueError("in_specs must be provided when method='shard_map'")
            if out_specs is None:
                raise ValueError("out_specs must be provided when method='shard_map'")

        args2, kwargs2 = kernel.prepare(*args, **kwargs)
        inv = Invocation(
            op_id=kernel.op_id,
            args=args2,
            kwargs=kwargs2,
            override_cfg=cfg,
            stamp=stamp,
            method=method,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        policy = getattr(self.chooser, "policy", None)
        if policy is not None and getattr(policy, "cache_miss_fallback", "heuristics") == "heuristics":
            chosen = self._choose_heuristics_only(inv, kernel)
        else:
            chosen = self.chooser.choose(inv, kernel)
        chosen = self._prefer_cuda_cfg(chosen, kernel, inv)

        platform = get_device_platform()
        context = "shard_map" if method == "shard_map" else None

        if _has_custom_vjp(kernel, platform, context):
            fwd_method = (
                _get_platform_method(kernel, "fwd_with_residuals", platform, context) or kernel.fwd_with_residuals
            )
            vjp_method = _get_platform_method(kernel, "vjp", platform, context) or kernel.vjp

            full_leaves, treedef = jtu.tree_flatten((args2, kwargs2))
            is_arr = [isinstance(x, jax.Array) for x in full_leaves]

            const_leaves = [None if m else x for m, x in zip(is_arr, full_leaves, strict=False)]

            def _restore_args_kwargs(array_leaves):
                """Rebuild (args, kwargs) by merging dynamic array leaves into closed constants."""
                it = iter(array_leaves)
                merged = [next(it) if m else v for m, v in zip(is_arr, const_leaves, strict=False)]
                return jtu.tree_unflatten(treedef, merged)

            def fwd_arrays(*array_leaves):
                """Forward rule: takes only array leaves, rebuilds args/kwargs inside."""
                a, k = _restore_args_kwargs(array_leaves)
                y, res = fwd_method(*a, cfg=chosen, **k)
                return y, (tuple(array_leaves), res)

            def bwd_arrays(payload, dy):
                """Backward rule: rebuild args/kwargs, call kernel.vjp, and map grads to array inputs."""
                array_leaves, res = payload
                a, k = _restore_args_kwargs(array_leaves)

                grads = vjp_method(res, dy, *a, cfg=chosen, **k)
                if isinstance(grads, dict):
                    raise TypeError("kernel.vjp must return a tuple of grads for positional args.")
                grads = tuple(grads)
                if len(grads) != len(a):
                    raise TypeError(
                        f"kernel.vjp must return one grad per positional arg; got {len(grads)} for {len(a)} args."
                    )

                def align_arg_grad(x, g):
                    """Align gradient structure with argument structure, using None for missing grads."""
                    if g is None:
                        return jtu.tree_map(lambda t: None, x)
                    return jtu.tree_map(lambda _t, gg: gg, x, g)

                aligned_args_grads = tuple(align_arg_grad(x, g) for x, g in zip(a, grads, strict=False))

                zeros_kwargs = {
                    name: jtu.tree_map(lambda t: jnp.zeros_like(t) if isinstance(t, jax.Array) else None, val)
                    for name, val in k.items()
                }

                full_grads = (aligned_args_grads, zeros_kwargs)
                flat_grads, _ = jtu.tree_flatten(full_grads)

                grad_out = []
                itg = iter(flat_grads)
                for m in is_arr:
                    gleaf = next(itg)
                    if m:
                        if gleaf is None:
                            gleaf = 0.0
                        grad_out.append(gleaf)
                return tuple(grad_out)

            def primal_only_arrays(*array_inputs):
                """Compute forward pass output only, discarding residuals for custom VJP."""
                return fwd_arrays(*array_inputs)[0]

            g = jax.custom_vjp(primal_only_arrays)
            g.defvjp(fwd_arrays, bwd_arrays)

            def fn(*a, **k):
                """Extract array leaves from args/kwargs and route through the custom VJP wrapper."""
                flat_call, _ = jtu.tree_flatten((a, k))
                array_in = [x for x, m in zip(flat_call, is_arr, strict=False) if m]
                return g(*array_in)

        else:
            run_method = _get_platform_method(kernel, "run", platform, context) or kernel.run

            def fn(*a, **k):
                """Execute the kernel run method with the pre-selected configuration."""
                return run_method(*a, cfg=chosen, **k)

        if os.getenv("EJKERNEL_OPS_RECORD", "0") == "1":
            try:
                from ..registry import record_invocation

                call_key = inv.make_key(kernel.key_builder)
                op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
                record_invocation(device_fingerprint(), op_id_v, call_key, kernel, args2, kwargs2)
            except Exception:
                pass
        if stamp:
            mode = os.getenv("EJKERNEL_OPS_STAMP", "none").lower()
            if mode == "json":
                fn = self._stamp_json(kernel, inv, fn, chosen)
            elif mode == "hash":
                fn = self._stamp_hash(kernel, inv, fn, chosen)
            elif mode == "none":
                pass

        if method == "shard_map":
            if not hasattr(kernel, "create_shard_map_wrapper"):
                raise AttributeError(f"Kernel {kernel.op_id} does not implement create_shard_map_wrapper")

            callback = None
            eagers = kernel.create_shard_map_wrapper(
                *args2,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
                cfg=chosen,
                **kwargs2,
            )

            if len(eagers) == 2:
                shard_map_fn, call_args = eagers
            elif len(eagers) == 3:
                shard_map_fn, call_args, callback = eagers
            outs = shard_map_fn(*call_args)
            if callback is not None:
                outs = callback(outs, cfg=chosen)
            return outs

        return fn(*args2, **kwargs2)

    def choose_config(self, kernel: Kernel[Cfg, Out], *args, cfg: Cfg | None = None, **kwargs) -> Cfg:
        """Select configuration for kernel without executing it.

        Useful for inspecting what configuration would be chosen for given
        arguments, or for pre-selecting configurations for compilation.

        Args:
            kernel: Kernel implementation requiring configuration
            *args: Positional arguments for the kernel
            cfg: Optional configuration override
            **kwargs: Keyword arguments for the kernel

        Returns:
            Configuration that would be selected for this invocation
        """

        if "_cfg" in kwargs:
            cfg = kwargs.pop("_cfg")

        args2, kwargs2 = kernel.prepare(*args, **kwargs)

        method = kwargs2.pop("method", None)
        mesh = kwargs2.pop("mesh", None)
        in_specs = kwargs2.pop("in_specs", None)
        out_specs = kwargs2.pop("out_specs", None)
        check_vma = kwargs2.pop("check_vma", False)

        inv = Invocation(
            op_id=kernel.op_id,
            args=args2,
            kwargs=kwargs2,
            override_cfg=cfg,
            stamp=False,
            method=method,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        policy = getattr(self.chooser, "policy", None)
        if policy is not None and getattr(policy, "cache_miss_fallback", "heuristics") == "heuristics":
            cfg = self._choose_heuristics_only(inv, kernel)
        else:
            cfg = self.chooser.choose(inv, kernel)
        return self._prefer_cuda_cfg(cfg, kernel, inv)

    def _choose_heuristics_only(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Select configuration using fast heuristics path without autotuning.

        This method provides a fast configuration selection path that bypasses
        the full autotuning system. It checks caches in the following order:
        1. Cache overlay (temporary overrides)
        2. In-memory cache
        3. Persistent cache
        4. Kernel heuristics (fallback)

        This is used when the policy is set to use heuristics-only on cache miss,
        avoiding expensive autotuning operations in production scenarios.

        Args:
            inv: Invocation object containing operation arguments and metadata
            kernel: Kernel implementation requiring configuration

        Returns:
            Configuration selected from cache or computed via heuristics

        Note:
            Unlike the full selector.choose path, this method never triggers
            autotuning. Selected configurations are written back to caches
            for future use.
        """
        dev = device_fingerprint()
        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        call_key = inv.make_key(kernel.key_builder)

        for overlay in reversed(_cache_overlay.get()):
            if (cfg := overlay.get((dev, op_id_v, call_key))) is not None:
                return cfg

        if (cfg := self.chooser.cache.get(dev, op_id_v, call_key)) is not None:
            return cfg

        if self.chooser.persistent is not None:
            if (cfg := self.chooser.persistent.get(dev, op_id_v, call_key)) is not None:
                self.chooser.cache.put(dev, op_id_v, call_key, cfg)
                return cfg

        platform = get_device_platform()
        context = "shard_map" if getattr(inv, "method", None) == "shard_map" else None
        heuristic_cfg_method = _get_platform_method(kernel, "heuristic_cfg", platform, context) or kernel.heuristic_cfg
        cfg = heuristic_cfg_method(inv)

        self.chooser.cache.put(dev, op_id_v, call_key, cfg)
        if self.chooser.persistent is not None and self.chooser.persist_autotune:
            self.chooser.persistent.put(dev, op_id_v, call_key, cfg)
        return cfg

    def compile(self, kernel: Kernel[Cfg, Out], *example_args, cfg: Cfg | None = None, **example_kwargs):
        """Compile kernel with pre-selected configuration.

        Selects optimal configuration based on example arguments, then returns
        a JAX-compiled function that uses that configuration for all subsequent
        calls. This avoids configuration selection overhead during execution.

        Args:
            kernel: Kernel implementation to compile
            *example_args: Example positional arguments for configuration selection
            cfg: Optional configuration override
            **example_kwargs: Example keyword arguments for configuration selection

        Returns:
            JAX-compiled function with pre-selected configuration

        Example:
            >>> compiled_matmul = executor.compile(matmul_kernel, x_example, y_example)
            >>>
            >>> result = compiled_matmul(x_actual, y_actual)
        """

        if "_cfg" in example_kwargs:
            cfg = example_kwargs.pop("_cfg")

        chosen = self.choose_config(kernel, *example_args, cfg=cfg, **example_kwargs)

        def run(*a, **k):
            """Execute the kernel with the pre-selected optimal configuration."""
            return kernel.run(*a, cfg=chosen, **k)

        return jax.jit(run)
