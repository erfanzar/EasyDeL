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


"""Configuration selection and autotuning system for kernel optimization.

This module provides a comprehensive configuration selection framework that
intelligently chooses optimal kernel configurations through a multi-tier
fallback chain. The system prioritizes cached results while supporting
automatic performance optimization when needed.

Key Components:
    ConfigSelectorChain: Main selection coordinator with fallback hierarchy
    AutotunePolicy: Configuration policy for autotuning behavior
    Tuner: Performance benchmarking and autotuning engine
    policy_override: Context manager for temporary policy changes

Selection Hierarchy (in order of priority):
    1. Override: Explicit configuration provided by caller
    2. Overlay: Temporary context-specific configuration overrides
    3. Memory Cache: Fast lookup for recently used configurations
    4. Persistent Cache: Disk-based storage across program runs
    5. Autotune: Benchmark candidates to find optimal configuration
    6. Heuristics: Kernel-provided default configuration
    7. Error: No configuration available (throws exception)

This design ensures optimal performance by:
    - Prioritizing fastest lookup methods (memory cache)
    - Preserving optimization results across runs (persistent cache)
    - Automatically finding optimal configurations (autotuning)
    - Providing sensible defaults (heuristics) as fallback

Example Usage:
    >>> cache = ConfigCache()
    >>> policy = AutotunePolicy(allow_autotune=True)
    >>> selector = ConfigSelectorChain(cache, policy)
    >>>
    >>>
    >>> config = selector.choose(invocation, kernel)
    >>>
    >>>
    >>> with policy_override(selector, allow_autotune=False):
    ...     config = selector.choose(invocation, kernel)
"""

from __future__ import annotations

import os
import pprint
import time
import traceback
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jcore
from jax import tree_util as jtu

from ejkernel.loggings import get_logger

from ..core import Invocation, Kernel, _get_platform_method
from ..utils.fingerprint import device_fingerprint, get_device_platform
from .cache import ConfigCache, _cache_overlay
from .persistent import PersistentCache

Cfg = TypeVar("Cfg")
Out = TypeVar("Out")

autotune_logger = get_logger("ejKernel-Selection")
_backward_autotune_enabled: ContextVar[bool] = ContextVar("ejkernel_backward_autotune_enabled", default=True)
_autotune_progress_enabled: ContextVar[bool] = ContextVar("ejkernel_autotune_progress", default=False)


@dataclass
class AutotunePolicy:
    """Configuration policy for autotuning behavior.

    Controls how the configuration selection system behaves when making
    optimization decisions, including whether to run autotuning, use
    heuristics, and validate backward pass correctness.

    Attributes:
        allow_autotune: Whether autotuning is permitted. When True, the system
            can benchmark multiple configurations to find the optimal one.
        allow_heuristics: Whether heuristic configurations are allowed as a
            fallback when no cached configuration is available.
        cache_miss_fallback: Strategy when no cached config is found. Either
            "autotune" to benchmark candidates or "heuristics" to use defaults.
        validate_backward: Whether to validate backward pass during autotuning.
            When True, autotuning will measure gradient computation time in
            addition to forward pass, ensuring the selected configuration
            performs well for training workloads.
    """

    allow_autotune: bool = True
    allow_heuristics: bool = True
    cache_miss_fallback: Literal["autotune", "heuristics"] = "autotune"
    validate_backward: bool = False


class policy_override:
    """Context manager for temporarily overriding autotuning policy settings.

    Allows temporary modification of AutotunePolicy attributes within a context,
    automatically restoring the original values when exiting the context.

    This is useful for:
    - Disabling autotuning for specific operations
    - Forcing use of heuristics during debugging
    - Testing different policy configurations

    Args:
        selector: ConfigSelectorChain instance to modify
        **updates: Policy attributes to override with new values

    Example:
        >>> with policy_override(selector, allow_autotune=False):
        ...     result = executor(kernel, *args)
        >>>

        >>> with policy_override(selector, cache_miss_fallback="heuristics"):
        ...     config = selector.choose(inv, kernel)
    """

    def __init__(self, selector: ConfigSelectorChain, **updates):
        """Initialize policy override context manager.

        Args:
            selector: ConfigSelectorChain to modify
            **updates: Policy attributes to override
        """
        self.selector = selector
        self.updates = updates
        self._prev = {}

    def __enter__(self):
        """Enter context and apply policy overrides.

        Returns:
            Self for use in with statements
        """
        for k, v in self.updates.items():
            self._prev[k] = getattr(self.selector.policy, k)
            setattr(self.selector.policy, k, v)
        return self

    def __exit__(self, *exc):
        """Exit context and restore original policy values.

        Args:
            *exc: Exception information (ignored)
        """
        for k, v in self._prev.items():
            setattr(self.selector.policy, k, v)


class forward_autotune_only:
    """Context manager that disables backward validation during autotuning.

    While active, autotune measurements run forward-only even when
    ``AutotunePolicy.validate_backward`` is True. This keeps autotuning focused
    on forward latency and avoids gradient-timing overhead.

    Example:
        >>> with forward_autotune_only():
        ...     cfg = selector.choose(inv, kernel)
    """

    def __init__(self):
        self._token = None

    def __enter__(self):
        self._token = _backward_autotune_enabled.set(False)
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _backward_autotune_enabled.reset(self._token)


def _is_backward_autotune_enabled() -> bool:
    """Return whether backward-pass validation is currently enabled for autotuning.

    Reads the ``_backward_autotune_enabled`` context variable which is set to
    ``False`` by :class:`forward_autotune_only` context managers.

    Returns:
        True unless :class:`forward_autotune_only` is active in the current
        context, in which case False.
    """
    return bool(_backward_autotune_enabled.get())


def _is_autotune_progress_enabled() -> bool:
    """Return whether autotune progress bars are enabled.

    Checks the ``ContextVar`` first, then falls back to the
    ``EJKERNEL_AUTOTUNE_PROGRESS`` environment variable.
    """
    if _autotune_progress_enabled.get():
        return True
    return os.getenv("EJKERNEL_AUTOTUNE_PROGRESS", "0") == "1"


class log_autotune_progress:
    """Context manager that enables tqdm progress bars during autotuning.

    Safe to use inside or outside ``jax.jit`` — autotuning itself always
    runs at Python level so the ``ContextVar`` is never traced.

    Example::

        with log_autotune_progress():
            result = model(x)           # any autotune triggered here shows a bar

        # or nest with other context managers
        with log_autotune_progress(), forward_autotune_only():
            result = model(x)
    """

    def __init__(self):
        self._token = None

    def __enter__(self):
        self._token = _autotune_progress_enabled.set(True)
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _autotune_progress_enabled.reset(self._token)


def set_autotune_progress(enabled: bool = True) -> None:
    """Imperatively enable or disable autotune progress bars.

    Unlike :class:`log_autotune_progress`, this persists until changed
    again (or until the ``ContextVar`` context is exited).  Useful for
    one-off toggling in notebooks or scripts::

        from ejkernel.ops.config import set_autotune_progress
        set_autotune_progress(True)
        # ... all subsequent autotune calls show progress ...
        set_autotune_progress(False)

    Args:
        enabled: Whether to show progress bars.
    """
    _autotune_progress_enabled.set(enabled)


class Tuner(Generic[Cfg]):
    """Performance benchmarking engine used by :class:`ConfigSelectorChain`.

    Compiles each candidate configuration under ``jax.jit``, runs warmup
    iterations to let the XLA runtime reach steady-state, and then times
    ``iters`` iterations.  The measured average is compared across all
    candidates and the fastest is returned.

    Deep-flattening of ``(args, kwargs)`` ensures that only array leaves are
    dynamic JIT parameters; everything else (dtypes, strings, callables) is
    captured as a Python constant so comparisons are JIT-stable.

    When ``_ejk_validate_backward=True`` is set on the function tag, the tuner
    differentiates a scalar sum-reduction loss over the output and times the
    resulting gradient computation instead of the forward pass alone.

    Attributes:
        warmup: Number of warmup iterations before timing begins (default: 1).
        iters: Number of timed iterations; the average is reported (default: 3).
    """

    def __init__(self, warmup=1, iters=3):
        """Initialize tuner with warmup and iteration settings.

        Args:
            warmup: Number of warmup iterations before timing (default: 1)
            iters: Number of timed iterations to average over (default: 3)
        """
        self.warmup, self.iters = warmup, iters

    def measure(self, fn, *args, **kwargs) -> float:
        """Measure average execution time with optional backward validation.

        Deep-flatten (args, kwargs) so only array-like leaves are dynamic:
        - Arrays or JAX tracers become dynamic parameters to the jitted function.
        - Everything else (dtype, strings, bools, callables, nested containers)
          is captured as Python constants in the closure.
        - Tracer-like arrays (e.g., ShardMapTracer, DynamicJaxprTracer) are converted
          to concrete zeros of the same shape/dtype before compile and timing.
        - If _ejk_validate_backward=True, we differentiate a scalar loss w.r.t.
          float/complex array leaves only; others are treated as non-diff.
        - If a kernel uses precompiled functions that can't be transformed, we fall back
          to forward-only timing, and if needed, to non-jitted forward timing.

        Args:
            fn: Function to measure (possibly tagged with _ejk_validate_backward)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Average execution time per iteration in seconds
        """

        def _is_arrayish(x) -> bool:
            """Return True for JAX Arrays, NumPy arrays, and JAX Tracers (abstract values)."""
            return isinstance(x, jax.Array | np.ndarray) or isinstance(x, jcore.Tracer)

        def _to_concrete(x):
            """Convert a tracer or abstract value to a concrete JAX array.

            Handles JAX tracers by extracting shape/dtype from their abstract
            value and creating zero-filled arrays. Passes through concrete arrays unchanged.

            Args:
                x: Value to convert (array, tracer, or scalar).

            Returns:
                Concrete JAX array with the same shape and dtype as the input.
            """
            if isinstance(x, jax.Array | np.ndarray):
                return x
            shape = getattr(x, "shape", None)
            dtype = getattr(x, "dtype", None)
            aval = getattr(x, "aval", None)
            if (shape is None or dtype is None) and aval is not None:
                shape = getattr(aval, "shape", None)
                dtype = getattr(aval, "dtype", None)
            if shape is not None and dtype is not None:
                return jnp.zeros(shape, dtype)
            return jnp.asarray(x)

        def _block_all(x):
            """Block until all arrays in the pytree are ready for synchronous timing."""
            return jtu.tree_map(lambda t: t.block_until_ready() if hasattr(t, "block_until_ready") else t, x)

        leaves, treedef = jtu.tree_flatten((args, kwargs))
        is_arr = [_is_arrayish(x) for x in leaves]
        const_leaves = [None if m else x for m, x in zip(is_arr, leaves, strict=False)]
        arr_leaves = [x for m, x in zip(is_arr, leaves, strict=False) if m]
        arr0 = tuple(_to_concrete(x) for x in arr_leaves)

        def _restore_args_kwargs(array_leaves):
            """Rebuild (args, kwargs) by merging dynamic array leaves with closed-over constants."""
            it = iter(array_leaves)
            merged = [next(it) if m else v for m, v in zip(is_arr, const_leaves, strict=False)]
            return jtu.tree_unflatten(treedef, merged)

        method = getattr(fn, "_ejk_method", "regular")
        validate_bwd = bool(getattr(fn, "_ejk_validate_backward", False))

        if method == "shard_map" and not getattr(fn, "_ejk_validate_backward", False):
            validate_bwd = False

        def _time_forward(jitted: bool = True) -> float:
            """Time forward-only execution with optional JIT compilation.

            Args:
                jitted: If True, JIT-compile the function before timing.
                    Falls back to non-jitted execution if False.

            Returns:
                Average execution time per iteration in seconds.
            """

            def core(*arrs):
                """Reconstruct args/kwargs from array leaves and call the target function."""
                aa, kk = _restore_args_kwargs(arrs)
                return fn(*aa, **kk)

            if jitted:
                c = jax.jit(core).lower(*arr0).compile()
                for _ in range(self.warmup):
                    _block_all(c(*arr0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(c(*arr0))
                return (time.perf_counter() - t0) / self.iters
            else:
                for _ in range(self.warmup):
                    _block_all(core(*arr0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(core(*arr0))
                return (time.perf_counter() - t0) / self.iters

        if validate_bwd:

            def _is_diff(x):
                """Check if a value is differentiable (floating-point or complex type)."""
                try:
                    dt = np.dtype(getattr(x, "dtype", None))
                    return np.issubdtype(dt, np.inexact)
                except Exception:
                    return False

            diff_mask = [_is_diff(x) for x in arr0]
            has_diff = any(diff_mask)
            if not has_diff:
                try:
                    return _time_forward(jitted=True)
                except Exception:
                    return _time_forward(jitted=False)

            def _split(arrs):
                """Split array leaves into differentiable and non-differentiable groups."""
                theta, nondiff = [], []
                for m, v in zip(diff_mask, arrs, strict=False):
                    (theta if m else nondiff).append(v)
                return tuple(theta), tuple(nondiff)

            def _merge(theta, nondiff):
                """Merge differentiable and non-differentiable arrays back into original order."""
                it_t, it_n = iter(theta), iter(nondiff)
                return tuple(next(it_t) if m else next(it_n) for m in diff_mask)

            def _scalarize_output(output):
                """Reduce array/scalar leaves from arbitrary pytrees into a scalar loss.

                Some kernels return tuple/dict outputs (e.g. ``(output, aux)``). Backward
                validation only needs a scalar objective, so we sum every numeric leaf.
                """
                leaves = jtu.tree_leaves(output)
                total = None
                for leaf in leaves:
                    if isinstance(leaf, (jax.Array, np.ndarray, jcore.Tracer)):
                        arr = jnp.asarray(leaf)
                    elif np.isscalar(leaf):
                        arr = jnp.asarray(leaf)
                    else:
                        continue
                    if not np.issubdtype(arr.dtype, np.number):
                        continue
                    part = jnp.sum(arr)
                    total = part if total is None else (total + part)
                if total is None:
                    raise TypeError(
                        "Autotune backward validation requires at least one numeric array/scalar in function output."
                    )
                if jnp.iscomplexobj(total):
                    total = jnp.real(total)
                return total

            def loss(theta, nondiff):
                """Compute scalar loss for backward pass validation timing."""
                arrs = _merge(theta, nondiff)
                aa, kk = _restore_args_kwargs(arrs)
                y = fn(*aa, **kk)
                return _scalarize_output(y)

            try:
                grad_core = jax.jit(jax.grad(loss, argnums=0))
                theta0, nondiff0 = _split(arr0)
                c = grad_core.lower(theta0, nondiff0).compile()
                for _ in range(self.warmup):
                    _block_all(c(theta0, nondiff0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(c(theta0, nondiff0))
                return (time.perf_counter() - t0) / self.iters
            except Exception as e:
                msg = str(e)
                if ("Cannot apply JAX transformations" in msg) or ("Leaked trace" in msg):
                    try:
                        return _time_forward(jitted=True)
                    except Exception:
                        return _time_forward(jitted=False)

                raise

        try:
            return _time_forward(jitted=True)
        except Exception:
            return _time_forward(jitted=False)

    def autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg]) -> Cfg:
        """Benchmark all candidate configurations and return the fastest one.

        Tests each candidate configuration by measuring its execution time
        and selects the configuration with the lowest average execution time.

        Set ``EJKERNEL_AUTOTUNE_PROGRESS=1`` to display a live tqdm progress
        bar showing the current candidate, best time so far, and remaining
        steps.

        Args:
            make_fn: Factory function that creates a function given a config
            args: Positional arguments for the function being benchmarked
            kwargs: Keyword arguments for the function being benchmarked
            candidates: Iterable of candidate configurations to test

        Returns:
            The configuration that achieved the fastest execution time

        Raises:
            RuntimeError: If no candidates are provided for testing
        """
        candidates = list(candidates)
        show_progress = _is_autotune_progress_enabled()
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=len(candidates),
                    desc="autotune",
                    unit="cfg",
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                autotune_logger.warning("EJKERNEL_AUTOTUNE_PROGRESS=1 but tqdm is not installed; progress bar disabled.")

        best_cfg, best_t = None, float("inf")
        last_err = None
        try:
            for cfg in candidates:
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"cfg={cfg}  best={best_t * 1e3:.3f}ms" if best_t < float("inf") else f"cfg={cfg}"
                    )
                try:
                    t = self.measure(make_fn(cfg), *args, **kwargs)
                    if os.getenv("EJKERNEL_LOG_AUTOTUNE", "0") == "1":
                        autotune_logger.info(pprint.pformat({"config": cfg, "time": t}))
                except Exception as e:
                    last_err = e
                    if pbar is not None:
                        pbar.update(1)
                    continue
                if t < best_t:
                    best_cfg, best_t = cfg, t
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                if best_cfg is not None:
                    pbar.set_postfix_str(f"best={best_t * 1e3:.3f}ms  cfg={best_cfg}")
                pbar.close()

        if best_cfg is None:
            if last_err:
                traceback.print_exception(last_err)
            autotune_logger.warning("All candidates failed during autotune; falling back to heuristics.")
            return None
        if os.getenv("EJKERNEL_LOG_AUTOTUNE", "0") == "1":
            autotune_logger.info(pprint.pformat({"best_config": best_cfg, "best_time": best_t}))
        return best_cfg


class ConfigSelectorChain(Generic[Cfg, Out]):
    """Multi-tier configuration selection system with fallback chain.

    Selection order:
    1. Override (explicit configuration provided)
    2. Overlay (temporary context-specific overrides)
    3. In-memory cache (fast lookup for recently used configs)
    4. Persistent cache (disk-based storage across runs)
    5. Autotune (benchmark candidates to find optimal config)
    6. Heuristics (kernel-provided default configuration)
    7. Error (no configuration available)

    Attributes:
        cache: In-memory configuration cache
        policy: Autotuning behavior policy
        tuner: Performance benchmarking tool
        persistent: Optional disk-based cache
        persist_autotune: Whether to save autotuned configs to persistent storage
        on_event: Optional callback for selection events
        forbid_reautotune: Prevent re-autotuning the same operation
    """

    def __init__(
        self,
        cache: ConfigCache[Cfg],
        policy: AutotunePolicy | None = None,
        tuner: Tuner[Cfg] | None = None,
        persistent: PersistentCache[Cfg] | None = None,
        persist_autotune: bool = True,
        on_event: callable | None = None,
        forbid_reautotune: bool = True,
    ):
        """Initialize configuration selector with cache and policy settings.

        Args:
            cache: In-memory configuration cache for fast lookups
            policy: Autotuning behavior policy (default: AutotunePolicy())
            tuner: Performance benchmarking tool (default: Tuner())
            persistent: Optional disk-based cache for cross-run persistence
            persist_autotune: Save autotuned configs to persistent storage (default: True)
            on_event: Optional callback for selection events (monitoring/debugging)
            forbid_reautotune: Prevent re-autotuning same operation (default: True)
        """
        self.cache = cache
        self.policy = policy or AutotunePolicy()
        self.tuner = tuner or Tuner()
        self.persistent = persistent
        self.persist_autotune = persist_autotune
        self.on_event = on_event
        self.forbid_reautotune = forbid_reautotune
        self._autotuned_keys: set[tuple[str, str, str]] = set()

    def choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Select optimal configuration using the fallback hierarchy.

        Implements the complete configuration selection algorithm, trying
        each method in order until a suitable configuration is found.

        Selection Priority (highest to lowest):
        1. Override: Explicit configuration in invocation
        2. Overlay: Temporary context-specific overrides
        3. Memory Cache: Previously computed optimal configurations
        4. Persistent Cache: Disk-stored configurations from previous runs
        5. Autotune: Benchmark candidates to find optimal configuration
        6. Heuristics: Kernel-provided default configuration

        Args:
            inv: Function invocation containing arguments and context
            kernel: Kernel implementation with candidate configurations

        Returns:
            Optimal configuration for this invocation

        Raises:
            RuntimeError: If no configuration can be determined
        """
        dev = device_fingerprint()
        op_id = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        call_key = inv.make_key(kernel.key_builder)

        if inv.override_cfg is not None:
            cfg = inv.override_cfg
            self._emit("override", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

        for overlay in reversed(_cache_overlay.get()):
            if (cfg := overlay.get((dev, op_id, call_key))) is not None:
                self._emit("overlay_hit", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
                return cfg

        if (cfg := self.cache.get(dev, op_id, call_key)) is not None:
            self._emit(
                "cache_hit",
                level="memory",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                cfg=cfg,
            )
            return cfg

        if self.persistent is not None:
            if (cfg := self.persistent.get(dev, op_id, call_key)) is not None:
                self._emit(
                    "cache_hit",
                    level="persistent",
                    device=dev,
                    op_id=op_id,
                    call_key=call_key,
                    cfg=cfg,
                )

                self.cache.put(dev, op_id, call_key, cfg)
                return cfg

        if self.policy.cache_miss_fallback == "autotune" and self.policy.allow_autotune:
            if self.forbid_reautotune and (dev, op_id, call_key) in self._autotuned_keys:
                raise RuntimeError(f"Re-autotune requested for {(dev, op_id, call_key)}")

            platform = get_device_platform()
            context = "shard_map" if inv.method == "shard_map" else None

            candidate_cfgs_method = _get_platform_method(kernel, "candidate_cfgs", platform, context)
            if candidate_cfgs_method:
                candidates = tuple(candidate_cfgs_method(inv))
            else:
                candidates = tuple(kernel.candidate_cfgs(inv))

            self._emit(
                "autotune_start",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                candidates=len(candidates),
                platform=platform,
                method=inv.method,
            )

            kw = dict(inv.kwargs)

            def _is_arrayish(x) -> bool:
                """Return True for JAX Arrays, NumPy arrays, and JAX Tracers."""
                return isinstance(x, jax.Array | np.ndarray) or isinstance(x, jcore.Tracer)

            static_fun_kwargs = {k: v for k, v in kw.items() if callable(v)}
            dyn_kwargs = kw

            validate_backward = self.policy.validate_backward and _is_backward_autotune_enabled()

            if inv.method == "shard_map":
                if not hasattr(kernel, "create_shard_map_wrapper"):
                    raise RuntimeError(
                        f"Kernel {kernel.op_id} does not implement create_shard_map_wrapper for shard_map benchmarking"
                    )

                def mk(c, _static=static_fun_kwargs):
                    """Return a callable that benchmarks the kernel under shard_map with config ``c``."""

                    def f(*a, **k):
                        """Build the shard_map wrapper for config ``c`` and run it, applying any callback."""
                        callback = None
                        eagers = kernel.create_shard_map_wrapper(
                            *a,
                            cfg=c,
                            mesh=inv.mesh,
                            in_specs=inv.in_specs,
                            out_specs=inv.out_specs,
                            check_vma=inv.check_vma,
                            **(k | _static),
                        )
                        if len(eagers) == 2:
                            shard_map_fn, call_args = eagers
                        elif len(eagers) == 3:
                            shard_map_fn, call_args, callback = eagers

                        outs = shard_map_fn(*call_args)
                        if callback is not None:
                            outs = callback(outs, cfg=c)
                        return outs

                    f._ejk_method = "shard_map"
                    if validate_backward and getattr(kernel, "supports_grad_validation", False):
                        f._ejk_validate_backward = True
                    return f

            else:
                run_method = _get_platform_method(kernel, "run", platform, context) or kernel.run

                def mk(c, _run=run_method, _static=static_fun_kwargs):
                    """Return a callable that benchmarks ``kernel.run`` with config ``c``."""

                    def f(*a, **k):
                        """Call the resolved run method with config ``c`` and closed-over static kwargs."""
                        return _run(*a, cfg=c, **(k | _static))

                    f._ejk_method = "regular"
                    if validate_backward:
                        f._ejk_validate_backward = True
                    return f

            best = self.tuner.autotune(mk, inv.args, dyn_kwargs, candidates)
            if best is not None:
                self._autotuned_keys.add((dev, op_id, call_key))
                self.cache.put(dev, op_id, call_key, best)
                if self.persistent is not None and self.persist_autotune:
                    self.persistent.put(dev, op_id, call_key, best)
                self._emit(
                    "autotune_finish",
                    device=dev,
                    op_id=op_id,
                    call_key=call_key,
                    cfg=best,
                    platform=platform,
                    method=inv.method,
                )
                return best

        if self.policy.allow_heuristics:
            platform = get_device_platform()
            context = "shard_map" if inv.method == "shard_map" else None

            heuristic_cfg_method = _get_platform_method(kernel, "heuristic_cfg", platform, context)
            if heuristic_cfg_method:
                cfg = heuristic_cfg_method(inv)
            else:
                cfg = kernel.heuristic_cfg(inv)
            self._emit(
                "heuristics",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                cfg=cfg,
                platform=platform,
                method=inv.method,
            )

            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None and self.persist_autotune:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

        self._emit("error", device=dev, op_id=op_id, call_key=call_key, reason="no_config")
        raise RuntimeError("No config found: override/overlay/cache/persistent/autotune/heuristics all unavailable.")

    def _emit(self, event: str, **data):
        """Invoke the optional monitoring callback with a selection event.

        Called at every decision point inside :meth:`choose` so that callers
        can trace cache hits, autotune starts/finishes, heuristic fallbacks,
        and errors without instrumenting the selector themselves.

        Args:
            event: Event type string.  Known values:
                ``'override'``, ``'overlay_hit'``, ``'cache_hit'``,
                ``'autotune_start'``, ``'autotune_finish'``,
                ``'heuristics'``, ``'error'``.
            **data: Event-specific keyword arguments, which vary by event but
                always include ``device``, ``op_id``, and ``call_key``.

        Note:
            Does nothing when :attr:`on_event` is ``None`` (the default).
        """
        if self.on_event:
            self.on_event(event, **data)
