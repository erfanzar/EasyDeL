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


"""Autotuning and benchmarking utilities for JAX kernel optimization.

This module provides comprehensive tools for automatic hyperparameter optimization
of JAX functions through systematic benchmarking and configuration testing.

Key Features:
    - Profiler-based timing with Python-level fallback for accuracy
    - Parallel compilation and testing of hyperparameter configurations
    - Statistical timing analysis with outlier removal
    - Thread-safe caching with configurable size limits
    - Support for distributed computation with sharding specifications

Classes:
    Measurement: Container for a single performance measurement
    AutotuneData: Container for all optimization measurements and results
    Autotuner: Core autotuning engine for hyperparameter optimization
    Entry: Cache entry for storing optimal configurations
    AutotuningResult: Device-specific optimization results container
    TimingResult: Statistical timing result with mean and standard deviation
    FNAutotuner: Advanced class-based autotuner with profiler integration

Functions:
    autotune: Decorator for automatic hyperparameter optimization
    autotune_recorded: Autotune all recorded invocations
    benchmark: Simple function benchmarking utility

Example:
    >>> @autotune(hyperparams={'block_size': [64, 128, 256]})
    ... def matrix_op(x, y, block_size=128):
    ...     return compute(x, y, block_size)
    >>>
    >>> result = matrix_op(x, y)
    >>> print(matrix_op.optimal_hyperparams)
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import random as pyrandom
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Generic, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, Sharding, SingleDeviceSharding

from ejkernel.loggings import get_logger

from ..config.cache import overlay_cache
from ..config.selection import _is_autotune_progress_enabled
from ..utils.fingerprint import device_fingerprint
from .profiler import Profiler

autotune_logger = get_logger("ejkernel.autotune", "WARNING")


Cfg = TypeVar("Cfg")


@dataclass
class Measurement:
    """Container for a single performance measurement.

    Stores the configuration and corresponding execution time
    for a single hyperparameter combination during optimization.

    Attributes:
        cfg: The hyperparameter configuration that was tested
        seconds: Execution time in seconds for this configuration
    """

    cfg: Any
    seconds: float


@dataclass
class AutotuneData(Generic[Cfg]):
    """Container for all optimization measurements and results.

    Stores performance measurements for all tested hyperparameter
    configurations and provides utilities to analyze the results.

    Type Parameters:
        Cfg: Configuration type (e.g., dict, dataclass, etc.)

    Attributes:
        measurements: List of all performance measurements taken
    """

    measurements: list[Measurement]

    @property
    def fastest_config(self) -> Cfg:
        """Get the configuration with the fastest execution time.

        Finds the measurement with the minimum execution time among all
        recorded measurements and returns its configuration.

        Returns:
            The configuration that achieved the lowest execution time

        Raises:
            ValueError: If no measurements are available
        """
        if not self.measurements:
            raise ValueError("No measurements available to determine fastest config")
        return min(self.measurements, key=lambda m: m.seconds).cfg


class Autotuner(Generic[Cfg]):
    """Core autotuning engine for hyperparameter optimization.

    This class provides the fundamental optimization algorithm that tests
    different configurations and measures their performance to find the
    optimal hyperparameter settings.

    Type Parameters:
        Cfg: Configuration type for hyperparameters

    Attributes:
        warmup: Number of warmup iterations before timing
        iters: Number of timing iterations for measurement accuracy
    """

    def __init__(self, warmup=1, iters=3):
        """Initialize the autotuner with timing parameters.

        Args:
            warmup: Number of warmup calls to stabilize performance before timing
            iters: Number of timing iterations for statistical accuracy
        """
        self.warmup, self.iters = warmup, iters

    def autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg]) -> AutotuneData[Cfg]:
        """Optimize hyperparameters by testing candidate configurations.

        Tests each candidate configuration by compiling and timing the
        function execution, then returns all measurements for analysis.

        Set ``EJKERNEL_AUTOTUNE_PROGRESS=1`` to display a live tqdm progress
        bar showing the current candidate, best time so far, and remaining
        steps.

        Args:
            make_fn: Factory function that creates a function given a config
            args: Positional arguments for the function being optimized
            kwargs: Keyword arguments for the function being optimized
            candidates: Iterable of candidate configurations to test

        Returns:
            AutotuneData containing all performance measurements

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

        measures = []
        best_t = float("inf")
        try:
            for cfg in candidates:
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"cfg={cfg}  best={best_t * 1e3:.3f}ms" if best_t < float("inf") else f"cfg={cfg}"
                    )
                try:
                    fn = make_fn(cfg)
                    c = jax.jit(fn).lower(*args, **kwargs).compile()

                    for _ in range(self.warmup):
                        _ = c(*args, **kwargs).block_until_ready()

                    t0 = time.perf_counter()
                    for _ in range(self.iters):
                        _ = c(*args, **kwargs).block_until_ready()
                    dt = (time.perf_counter() - t0) / self.iters
                    measures.append(Measurement(cfg, dt))
                    if dt < best_t:
                        best_t = dt
                except Exception as e:
                    autotune_logger.warning(f"Configuration {cfg} failed: {e}")
                    measures.append(Measurement(cfg, float("inf")))
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                if best_t < float("inf"):
                    pbar.set_postfix_str(f"best={best_t * 1e3:.3f}ms")
                pbar.close()

        if not measures or all(m.seconds == float("inf") for m in measures):
            autotune_logger.warning("All candidate configurations failed to execute; returning empty measurements.")
        return AutotuneData(measures)


@dataclass(frozen=True)
class Entry:
    """Cache entry for storing optimal configurations.

    Represents a single cached optimization result with the operation
    identifier, call signature, and optimal configuration.

    Attributes:
        op_id_v: Operation identifier for the optimized function
        call_key: Hash key representing the function call signature
        cfg: The optimal configuration found for this operation
    """

    op_id_v: str
    call_key: str
    cfg: Any


@dataclass(frozen=True)
class AutotuningResult:
    """Result container for device-specific optimization results.

    Stores all optimized configurations for a specific device and provides
    context manager functionality for temporary cache overlays.

    Attributes:
        device: Device identifier these results apply to
        entries: Tuple of optimization entries (operation -> config mappings)
    """

    device: str
    entries: tuple[Entry, ...]

    def as_overlay(self):
        """Convert results to cache overlay mapping format.

        Creates a dictionary mapping that can be used with the cache overlay
        system to temporarily apply these optimization results.

        Returns:
            Dictionary mapping (device, op_id, call_key) tuples to configurations
        """
        mapping = {(self.device, e.op_id_v, e.call_key): e.cfg for e in self.entries}
        return mapping

    def __enter__(self):
        """Enter context manager to apply optimization results as cache overlay.

        Activates the cache overlay with the optimization results, temporarily
        overriding any existing cache entries with the optimized configurations.

        Returns:
            Self for use in with statements
        """
        self._ctx = overlay_cache(self.as_overlay())
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager and restore previous cache state.

        Deactivates the cache overlay and restores the previous cache state,
        ensuring clean cleanup even if exceptions occur.

        Args:
            exc_type: Exception type (if any)
            exc: Exception instance (if any)
            tb: Exception traceback (if any)
        """
        self._ctx.__exit__(exc_type, exc, tb)
        delattr(self, "_ctx")


def autotune_recorded(hyperparameter_selector, *, show_progress=False, repetition_count=1):
    """Autotune all kernel invocations recorded for the current device.

    Iterates over every ``(op_id, call_key)`` entry stored in the global
    invocation registry for the current device (see
    :func:`~ejkernel.ops.registry.get_invocations`), benchmarks each
    candidate configuration via :func:`benchmark`, and writes the winner to
    both the in-memory and persistent caches.

    The registry is populated only when ``EJKERNEL_OPS_RECORD=1`` is set in the
    environment before the initial (non-tuning) run.

    Autotuning process:
        1. Retrieve all invocations recorded for the current device.
        2. For each ``(op_id_v, call_key, kernel, args, kwargs)`` entry:
           a. Call ``kernel.prepare(*args, **kwargs)`` to get processed inputs.
           b. Call ``kernel.candidate_cfgs()`` for candidate configurations.
           c. Benchmark each candidate with :func:`benchmark`.
           d. Write the fastest configuration to the selector's cache (and
              persistent cache, if configured).
        3. Return an :class:`AutotuningResult` wrapping all winners.

    Args:
        hyperparameter_selector: :class:`~ejkernel.ops.config.ConfigSelectorChain`
            instance whose ``.cache`` and ``.persistent`` attributes receive
            the optimised configurations.
        show_progress: Whether to display tqdm progress bars during
            benchmarking.  Also enabled by the ``EJKERNEL_AUTOTUNE_PROGRESS=1``
            environment variable or the
            :class:`~ejkernel.ops.config.log_autotune_progress` context manager.
            *Note:* the ``repetition_count`` argument is accepted but currently
            has no effect (reserved for future multi-pass support).
        repetition_count: Reserved for future use.  Accepted but ignored.

    Returns:
        :class:`AutotuningResult` containing one :class:`Entry` per tuned
        operation.  Can be used as a context manager to apply the results as a
        cache overlay for a subsequent run.

    Example:
        >>> from ejkernel.ops.config import ConfigCache, ConfigSelectorChain
        >>> cache = ConfigCache()
        >>> selector = ConfigSelectorChain(cache)
        >>>
        >>> # First run: record invocations
        >>> # EJKERNEL_OPS_RECORD=1 python my_script.py
        >>>
        >>> # Then autotune all recorded operations:
        >>> result = autotune_recorded(selector)
        >>> with result:
        ...     output = my_model(input_data)

    Note:
        Requires invocations to have been previously recorded.  If the registry
        is empty, this function returns an :class:`AutotuningResult` with an
        empty ``entries`` tuple.
    """
    from ..registry import get_invocations

    dev = device_fingerprint()
    invs = get_invocations(dev)
    entries = []

    show_progress = show_progress or _is_autotune_progress_enabled()
    _tqdm = None
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            autotune_logger.warning("EJKERNEL_AUTOTUNE_PROGRESS=1 but tqdm is not installed; progress bar disabled.")

    for op_id_v, d in invs.items():
        for call_key, (kernel, args, kwargs) in d.items():
            inv_args, inv_kwargs = kernel.prepare(*args, **kwargs)

            static_fun_kwargs = {k: v for k, v in inv_kwargs.items() if callable(v)}
            dyn_kwargs = {k: v for k, v in inv_kwargs.items() if k not in static_fun_kwargs}

            tmp_inv = type(
                "Tmp",
                (),
                dict(
                    op_id=kernel.op_id, args=inv_args, kwargs=dyn_kwargs, batch_axes=None, override_cfg=None, stamp=False
                ),
            )()
            candidates = tuple(kernel.candidate_cfgs(tmp_inv))

            def mk(c, _run=kernel.run, _static=static_fun_kwargs):
                """Create a partial function binding a specific configuration to the kernel run method."""
                return partial(_run, cfg=c, **_static)

            best_cfg, best_t = None, float("inf")
            pbar = None
            if _tqdm is not None and len(candidates) > 0:
                pbar = _tqdm(
                    total=len(candidates),
                    desc=f"autotune {op_id_v}",
                    unit="cfg",
                    dynamic_ncols=True,
                    leave=True,
                )
            try:
                for c in candidates:
                    if pbar is not None:
                        pbar.set_postfix_str(
                            f"cfg={c}  best={best_t * 1e3:.3f}ms" if best_t < float("inf") else f"cfg={c}"
                        )
                    t = benchmark(mk(c), *inv_args, **dyn_kwargs)
                    if t < best_t:
                        best_cfg, best_t = c, t
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    if best_cfg is not None:
                        pbar.set_postfix_str(f"best={best_t * 1e3:.3f}ms  cfg={best_cfg}")
                    pbar.close()

            hyperparameter_selector.cache.put(dev, op_id_v, call_key, best_cfg)
            if hyperparameter_selector.persistent and hyperparameter_selector.persist_autotune:
                hyperparameter_selector.persistent.put(dev, op_id_v, call_key, best_cfg)
            entries.append(Entry(op_id_v, call_key, best_cfg))
    return AutotuningResult(dev, tuple(entries))


def _split_static_callable_kwargs(kwargs):
    """Split keyword arguments into static and dynamic components.

    Separates callable arguments (static) from regular arguments (dynamic)
    for proper JAX compilation and execution.

    Args:
        kwargs: Dictionary of keyword arguments to split

    Returns:
        Tuple of (static_kwargs, dynamic_kwargs)
    """
    static = {k: v for k, v in kwargs.items() if callable(v)}
    return static, {k: v for k, v in kwargs.items() if k not in static}


def benchmark(fn, *args, warmup=1, iters=5, **kwargs) -> float:
    """Benchmark function execution time with JAX compilation.

    Compiles the function with JAX and measures its execution time
    over multiple iterations, handling both static and dynamic arguments.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for the function
        warmup: Number of warmup iterations before timing
        iters: Number of timing iterations for measurement
        **kwargs: Keyword arguments for the function

    Returns:
        Average execution time per iteration in seconds
    """
    static, dyn = _split_static_callable_kwargs(kwargs)

    if static:

        def fn_wrapped(*a, _fn=fn, _static=static, **k):
            """Wrap function to merge static callable kwargs with dynamic kwargs at call time."""
            return _fn(*a, **(k | _static))

        c = jax.jit(fn_wrapped).lower(*args, **dyn).compile()
        for _ in range(warmup):
            _ = c(*args, **dyn).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = c(*args, **dyn).block_until_ready()
        return (time.perf_counter() - t0) / iters
    else:
        c = jax.jit(fn).lower(*args, **kwargs).compile()
        for _ in range(warmup):
            _ = c(*args, **kwargs).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = c(*args, **kwargs).block_until_ready()
        return (time.perf_counter() - t0) / iters


@dataclass
class TimingResult:
    """Statistical timing result for a single hyperparameter configuration.

    Stores the measured execution time statistics for a specific set of
    hyperparameters, including both mean and standard deviation for
    reliability analysis.

    Attributes:
        hyperparams: Dictionary of hyperparameter names to their tested values
        t_mean: Mean execution time in seconds across timing iterations
        t_std: Standard deviation of execution times in seconds
    """

    hyperparams: dict[Any, Any]
    t_mean: float
    t_std: float


def _get_global_mesh():
    """Retrieve the currently active global JAX mesh from thread-local state.

    Reads the physical mesh from ``jax.interpreters.pxla.thread_resources``.
    This is the mesh set by the innermost active ``jax.sharding.Mesh`` context
    manager, or ``None`` when no mesh is active.

    Returns:
        The active :class:`jax.sharding.Mesh`, or ``None`` if no mesh is
        configured or the mesh is empty (``mesh.empty`` is ``True``).

    Note:
        This accesses JAX internals (``pxla.thread_resources``) which may
        change across JAX releases.
    """
    env = pxla.thread_resources.env
    mesh = env.physical_mesh
    return None if mesh.empty else mesh


def _get_default_device():
    """Return the default JAX compute device.

    Reads ``jax.config.values['jax_default_device']`` first; falls back to
    the first device returned by ``jax.devices()`` when no explicit default
    has been set.

    Returns:
        A :class:`jax.Device` instance representing the default compute device.
    """
    if jax.config.values["jax_default_device"] is not None:
        return jax.config.values["jax_default_device"]
    return jax.devices()[0]


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Context manager to temporarily suppress stdout and stderr output.

    Redirects both stdout and stderr to /dev/null for the duration of the
    context, then restores them afterward. Useful for silencing noisy
    library output during compilation or benchmarking.

    Yields:
        None - This is a context manager with no return value

    Note:
        Uses file descriptor duplication to ensure proper restoration
        even if exceptions occur within the context.
    """
    devnull = open(os.devnull, "w")
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        try:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)
            devnull.close()


def _normalize_sharding(
    arg: jax.Array | np.ndarray | Any,
    sharding_or_spec: PartitionSpec | Sharding | None,
    default_device: jax.Device,  # type: ignore
):
    """Convert a sharding specification into a concrete :class:`~jax.sharding.Sharding`.

    Used inside :class:`FNAutotuner` to resolve the ``in_shardings`` argument
    so that concrete random arrays can be placed on the correct devices before
    benchmarking.

    Args:
        arg: The array (or other value) for which sharding is being resolved.
            Non-array values receive ``None`` (they are not sharded).
        sharding_or_spec: How to shard ``arg``:

            - :class:`~jax.sharding.Sharding` instance — returned as-is.
            - :class:`~jax.sharding.PartitionSpec` — converted to a
              :class:`~jax.sharding.NamedSharding` using the current global
              mesh.  Raises ``ValueError`` if no global mesh is active.
            - ``None`` — resolved to :class:`~jax.sharding.SingleDeviceSharding`
              on ``default_device``.

        default_device: Fallback device when no explicit sharding is given.

    Returns:
        A concrete :class:`~jax.sharding.Sharding`, or ``None`` if ``arg`` is
        not an array.

    Raises:
        ValueError: If a :class:`~jax.sharding.PartitionSpec` is provided but
            no global mesh is currently active.
    """
    if not isinstance(arg, jax.Array | np.ndarray):
        return None
    if isinstance(sharding_or_spec, Sharding):
        return sharding_or_spec
    global_mesh = _get_global_mesh()
    if isinstance(sharding_or_spec, PartitionSpec) and global_mesh is not None:
        return jax.NamedSharding(global_mesh, sharding_or_spec)
    elif isinstance(sharding_or_spec, PartitionSpec) and global_mesh is None:
        raise ValueError("If specifying shardings via ParitionSpec, a global mesh must be defined")
    else:
        return SingleDeviceSharding(default_device)


def _ensure_dtype(dt):
    """Return the ``.dtype`` of an array, or the input itself if not an array.

    Used to normalise entries in a shape/dtype specification that might be
    either concrete arrays or already-extracted dtypes.

    Args:
        dt: A :class:`jax.Array`, :class:`numpy.ndarray`, or any dtype-like
            object.

    Returns:
        ``dt.dtype`` if ``dt`` is a JAX or NumPy array; otherwise ``dt`` itself.
        Returns ``dt`` unchanged on any exception.
    """
    try:
        return dt.dtype if isinstance(dt, jax.Array | np.ndarray) else dt
    except Exception:
        return dt


@partial(jax.jit, static_argnames=("sds", "sharding"))
def _get_random_value(sds, sharding=None):
    """Generate a concrete random array matching a shape/dtype specification.

    Used by :class:`FNAutotuner` to materialise abstract input specifications
    (e.g. ``jax.ShapeDtypeStruct``) into concrete arrays suitable for JIT
    compilation and benchmarking.

    Both arguments are treated as *static* by the surrounding ``@jax.jit``
    wrapper so that different shape/sharding combinations each produce a
    separately-compiled function.

    Args:
        sds: A shape/dtype specification with ``.shape`` and ``.dtype``
            attributes (e.g. :class:`jax.ShapeDtypeStruct`), or any other
            value that should be returned unchanged.
        sharding: Optional :class:`~jax.sharding.Sharding` passed as
            ``out_shardings`` to the inner JIT call.  ``None`` means no
            explicit sharding is requested.

    Returns:
        For floating-point ``sds``: a normally-distributed random array of the
        specified shape and dtype.
        For integer ``sds``: a zero-filled array of the specified shape and dtype.
        For any other ``sds`` (no ``.shape``/``.dtype``): returns ``sds`` unchanged.

    Raises:
        ValueError: If ``sds`` has ``.shape`` and ``.dtype`` but the dtype is
            neither floating-point nor integer.
    """
    if hasattr(sds, "shape") and hasattr(sds, "dtype"):
        dt = _ensure_dtype(sds.dtype)
        if jnp.issubdtype(dt, jnp.floating):
            return jax.jit(lambda key: random.normal(key, sds.shape, dt), out_shardings=sharding)(random.key(0))
        elif jnp.issubdtype(dt, jnp.integer):
            return jax.jit(lambda: jnp.zeros(sds.shape, dt), out_shardings=sharding)()
        else:
            raise ValueError(f"Unsupported dtype {dt}")
    else:
        return sds


def _try_hash_input(args, kws, must_be_concrete: bool = True):
    """Attempt to create a hashable signature key from function input arguments.

    Builds a hash of the PyTree *structure* plus the shape, dtype, and sharding
    of every array leaf.  Non-array leaves are included directly.  Array values
    are intentionally ignored so the same key is produced for arrays that have
    the same structure but different data.

    Args:
        args: Positional arguments to hash.
        kws: Keyword arguments to hash.
        must_be_concrete: If ``True`` (default), returns ``None`` when any JAX
            array leaf is abstract (i.e. is a tracer, as happens inside
            ``jax.jit`` or ``jax.vmap``).  Setting to ``False`` allows hashing
            inside traced contexts, but the resulting key may be less stable.

    Returns:
        An integer hash uniquely identifying the input signature, or ``None`` if:

        - ``must_be_concrete`` is ``True`` and at least one array leaf is abstract.
        - Hashing fails for any reason (e.g. unhashable leaf types).
    """
    flat_vals, struct = jax.tree.flatten((args, kws))
    all_concrete = all(jax.core.is_concrete(x) for x in flat_vals if isinstance(x, jax.Array))
    if not all_concrete and must_be_concrete:
        return None

    def _get_sharding(x):
        """Extract sharding from array or its abstract type."""
        try:
            return x.sharding
        except AttributeError:
            return jax.typeof(x).sharding

    def array_to_hashable(x):
        """Convert array to hashable representation based on type and sharding."""
        return x if not isinstance(x, jax.Array) else hash((jax.typeof(x), _get_sharding(x)))

    try:
        return hash((struct, tuple(array_to_hashable(x) for x in flat_vals)))
    except Exception:
        return None


class FNAutotuner:
    """Advanced JAX autotuner with profiler-first timing and Python fallback.

    This is the engine behind the :func:`autotune` decorator.  It provides a
    full hyperparameter optimisation pipeline:

    1. Generate all hyperparameter combinations (up to ``sample_num``).
    2. Compile each configuration in parallel (``ThreadPoolExecutor``).
    3. Time execution using :class:`~ejkernel.ops.execution.profiler.Profiler`
       (JAX XPlane profiler) when available, or fall back to Python-level
       ``time.perf_counter`` timing.
    4. Remove outliers and rank by a composite score
       (mean + 0.1 × std, see :meth:`_calculate_timing_score`).
    5. Return (and cache) the fastest configuration.

    Thread-safety:
        The per-function result cache (inside :meth:`decorate`) is protected by
        a :class:`threading.Lock`.  The ``FNAutotuner`` instance itself is
        stateless between :meth:`tune` calls and may be shared across threads.

    Attributes:
        allow_fallback_timing: Fall back to Python timing when the profiler is
            unavailable or raises an error (default: ``True``).
        profiling_samples: Number of full profile captures for statistical
            aggregation.  More samples reduce noise but increase tuning time
            (default: ``5``).
        must_find_profiler_fraction: Minimum fraction of compiled configurations
            that must appear in the profiler output before the profiler results
            are accepted.  Below this threshold the system falls back to Python
            timing (default: ``0.5``).
        enable_detailed_logging: When ``True``, failed compilation and execution
            errors are logged with full tracebacks (default: ``False``).
        find_optimal_layouts_automatically: When ``True``, query
            ``compiled.input_formats`` to discover device-preferred memory
            layouts and place inputs accordingly before timing (default:
            ``False``).
        max_compilation_time_seconds: Timeout hint passed to ``_try_call``.
            Currently accepted but not enforced (default: ``300.0``).
        timing_warmup_iterations: Number of un-timed warm-up calls before
            Python-level timing begins (default: ``2``).
        timing_rounds: Number of timed rounds for Python-level timing; the
            outer two (min and max) are discarded before averaging (default:
            ``5``).
        calls_per_round: Calls executed per Python timing round; each round's
            wall-clock time is divided by this value (default: ``3``).
        cache_size_limit: Maximum number of entries in the per-decorated-function
            LRU-style cache inside :meth:`decorate` (default: ``1000``).
        profiler: :class:`~ejkernel.ops.execution.profiler.Profiler` instance
            configured with the ``profiler_*`` constructor arguments.
    """

    PREFIX_FN = "autotune_fn_{}"

    def __init__(
        self,
        *,
        allow_fallback_timing: bool = True,
        profiling_samples: int = 5,
        must_find_profiler_fraction: float = 0.5,
        enable_detailed_logging: bool = False,
        find_optimal_layouts_automatically: bool = False,
        max_compilation_time_seconds: float = 300.0,
        timing_warmup_iterations: int = 2,
        timing_rounds: int = 5,
        calls_per_round: int = 3,
        cache_size_limit: int = 1000,
        profiler_prefix_filter: str = "jit_",
        profiler_event_regex: str | None = None,
        profiler_min_duration_ns: float = 1000.0,
        profiler_max_events: int | None = 10000,
        profiler_verbose: bool = False,
    ):
        """Initialize the autotuner with timing and profiling configuration.

        Args:
            allow_fallback_timing: Enable Python timing fallback if profiler fails
            profiling_samples: Number of profiling iterations for statistical accuracy
            must_find_profiler_fraction: Minimum fraction of configs needing profiler
                results (0.0-1.0). Falls back to Python timing if not met.
            enable_detailed_logging: Enable detailed error logging with tracebacks
            find_optimal_layouts_automatically: Auto-discover optimal memory layouts
                for distributed computation
            max_compilation_time_seconds: Maximum compilation time per configuration
            timing_warmup_iterations: Warmup iterations before timing measurements
            timing_rounds: Number of timing rounds for statistical analysis
            calls_per_round: Function calls per timing round
            cache_size_limit: Maximum number of cached optimization results
            profiler_prefix_filter: Event name prefix filter for profiler
            profiler_event_regex: Optional regex filter for profiler events
            profiler_min_duration_ns: Minimum event duration for profiler inclusion
            profiler_max_events: Maximum events per profile to prevent memory issues
            profiler_verbose: Enable verbose profiler output
        """
        self.allow_fallback_timing = allow_fallback_timing
        self.profiling_samples = profiling_samples
        self.must_find_profiler_fraction = must_find_profiler_fraction
        self.enable_detailed_logging = enable_detailed_logging
        self.find_optimal_layouts_automatically = find_optimal_layouts_automatically
        self.max_compilation_time_seconds = max_compilation_time_seconds
        self.timing_warmup_iterations = timing_warmup_iterations
        self.timing_rounds = timing_rounds
        self.calls_per_round = calls_per_round
        self.cache_size_limit = cache_size_limit

        self.profiler = Profiler(
            prefix_filter=profiler_prefix_filter,
            event_filter_regex=profiler_event_regex,
            min_duration_ns=profiler_min_duration_ns,
            max_events_per_profile=profiler_max_events,
            verbose=profiler_verbose or (autotune_logger._level <= logging.INFO),
        )

        self._cache_lock = threading.Lock()

    @staticmethod
    def _calculate_timing_score(tr: TimingResult) -> float:
        """Calculate a composite timing score for ranking configurations.

        Combines mean execution time with standard deviation to create a single
        score that balances speed with consistency. Penalizes configurations
        with high variance even if they have good mean performance.
        The weighting factor of 0.1 for standard deviation provides a reasonable
        balance between speed and stability.

        Args:
            tr: TimingResult containing mean and standard deviation

        Returns:
            Composite score (lower is better) = mean + 0.1 * std
        """
        return tr.t_mean + 0.1 * tr.t_std

    def _create_parameterized_function(
        self,
        target_function: Callable[..., Any],
        hyperparameter_values: dict[str, Any],
        output_shardings: Any = None,
        function_id: int = 0,
    ) -> Callable[..., Any]:
        """Create a JIT-compiled function with embedded hyperparameters.

        Wraps the target function with specific hyperparameter values and
        compiles it using JAX JIT. Sets a unique name for profiler identification
        and optionally configures output sharding for distributed computation.
        The function name follows the pattern 'autotune_fn_{function_id}' for
        easy identification in profiler output.

        Args:
            target_function: Original function to parameterize
            hyperparameter_values: Dictionary of hyperparameter names to values
            output_shardings: Optional sharding specification for outputs
            function_id: Unique identifier for profiler tracking

        Returns:
            JIT-compiled function with embedded hyperparameters and unique name
        """
        jax_compiler = partial(jax.jit, out_shardings=output_shardings)

        def parameterized_function(*function_args, **function_kwargs):
            """Execute the target function with embedded hyperparameter values merged into kwargs."""
            combined_kwargs = dict(function_kwargs, **hyperparameter_values)
            return target_function(*function_args, **combined_kwargs)

        function_name = self.PREFIX_FN.format(function_id)
        parameterized_function.__name__ = function_name
        parameterized_function.__qualname__ = function_name

        return jax_compiler(parameterized_function)

    def _try_call(
        self,
        fn: Callable[..., Any],
        resolved_args,
        resolved_kwargs,
        compile_only: bool = False,
        compute_layouts: bool = False,
        optimal_formats: Any | None = None,
        timeout: float | None = None,
    ):
        """Safely compile or execute a function with comprehensive error handling.

        Attempts to compile or execute a function with the provided arguments,
        handling various failure modes gracefully. Supports layout optimization
        and optimal device placement for distributed computation.

        Args:
            fn: Function to compile or execute
            resolved_args: Positional arguments for the function
            resolved_kwargs: Keyword arguments for the function
            compile_only: If True, only compile without execution
            compute_layouts: If True, compute optimal memory layouts
            optimal_formats: Optional pre-computed optimal device formats
            timeout: Optional compilation timeout (currently unused)

        Returns:
            Tuple of (success_bool, error_message, optimal_input_formats)
            where success_bool indicates if operation succeeded,
            error_message contains failure details if any,
            and optimal_input_formats contains discovered layouts if computed
        """
        optimal_input_formats = None
        try:
            if compile_only:
                if compute_layouts:

                    def to_shape(x):
                        """Convert a JAX array to its ShapeDtypeStruct representation for layout optimization."""
                        return (
                            jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
                            if isinstance(x, jax.Array)
                            else x
                        )

                    argument_shapes, keyword_shapes = jax.tree.map(to_shape, (resolved_args, resolved_kwargs))
                    try:
                        compiled_function = jax.jit(fn).lower(*argument_shapes, **keyword_shapes).compile()
                        optimal_input_formats = getattr(compiled_function, "input_formats", None)
                    except Exception as compilation_error:
                        autotune_logger.warning(
                            f"Layout optimization failed during compilation: "
                            f"{compilation_error.__class__.__name__}: {compilation_error}"
                        )
                        optimal_input_formats = None
                else:
                    _ = jax.jit(fn).lower(*resolved_args, **resolved_kwargs).compile()
            else:
                if optimal_formats is not None:

                    def place_array_on_optimal_device(array_data, target_format):
                        """Place an array on its optimal device layout, passing non-arrays through."""
                        return (
                            jax.device_put(array_data, target_format)
                            if isinstance(array_data, jax.Array)
                            else array_data
                        )

                    try:
                        optimally_placed_args, optimally_placed_kwargs = jax.tree.map(
                            place_array_on_optimal_device, (resolved_args, resolved_kwargs), optimal_formats
                        )
                        _ = jax.block_until_ready(fn(*optimally_placed_args, **optimally_placed_kwargs))
                    except Exception:
                        autotune_logger.warning(
                            "Failed to place arrays on optimal devices - falling back to original argument placement"
                        )
                        _ = jax.block_until_ready(fn(*resolved_args, **resolved_kwargs))
                else:
                    _ = jax.block_until_ready(fn(*resolved_args, **resolved_kwargs))
            return True, None, optimal_input_formats
        except Exception as e:
            msg = f"{type(e).__name__}: {e!s}"
            if self.enable_detailed_logging:
                import traceback

                msg = traceback.format_exc()
            return False, msg, optimal_input_formats

    def _time_fn(self, target_function: Callable[[], None]) -> tuple[float, float]:
        """Perform high-precision Python-level timing with statistical analysis.

        Executes the target function multiple times with warmup iterations,
        measures execution times, and provides statistical analysis with
        outlier removal for reliable performance measurements. Uses
        block_until_ready() to ensure accurate timing of JAX computations.

        Args:
            target_function: Zero-argument function to time (should return JAX arrays)

        Returns:
            Tuple of (mean_time, std_time) in seconds after outlier removal,
            or (inf, inf) if all timing attempts failed
        """

        def _execute_and_block():
            """Execute the target function and block until all outputs are ready."""
            return jax.block_until_ready(target_function())

        for _ in range(self.timing_warmup_iterations):
            try:
                _execute_and_block()
            except Exception as warmup_error:
                autotune_logger.warning(f"Warmup failed: {warmup_error.__class__.__name__}: {warmup_error}")
                return float("inf"), float("inf")

        times = []
        for _ in range(self.timing_rounds):
            t0 = time.perf_counter()
            try:
                for _ in range(self.calls_per_round):
                    _execute_and_block()
                times.append(time.perf_counter() - t0)
            except Exception as timing_error:
                autotune_logger.warning(f"Timing round failed: {timing_error.__class__.__name__}: {timing_error}")
                times.append(float("inf"))

        valid = [t for t in times if not np.isinf(t)]
        if not valid:
            return float("inf"), float("inf")

        arr = np.array(valid) / self.calls_per_round
        arr = np.sort(arr)
        if len(arr) > 2:
            arr = arr[1:-1]
        return float(np.mean(arr)), float(np.std(arr))

    def tune(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None,
        max_workers: int = 32,
        in_shardings: Any = None,
        out_shardings: Any = None,
        device: jax.Device | str | None = None,  # type: ignore
        example_args: tuple[Any, ...] | None = None,
        example_kws: dict[Any, Any] | None = None,
        sample_num: int = 2**63 - 1,
        event_filter_regex: str | None = None,
        timeout: float | None = None,
    ) -> tuple[Callable[..., Any], dict[str, Any], list[tuple[int, TimingResult]]]:
        """Tune hyperparameters for a function and return optimal configuration.

        Performs comprehensive hyperparameter optimization by testing all
        candidate configurations in parallel, measuring their performance,
        and returning the best-performing configuration.

        Args:
            fn: Function to optimize (must accept hyperparameters as keyword args)
            args: Positional arguments for the function (can be abstract shapes)
            kwargs: Keyword arguments for the function (excluding hyperparameters)
            hyperparams: Dictionary mapping hyperparameter names to lists of candidate
                values to test. Each combination will be evaluated.
            max_workers: Maximum number of parallel workers for compilation
            in_shardings: Input sharding specifications for distributed computation
            out_shardings: Output sharding specifications for distributed computation
            device: Target device or device string for computation
            example_args: Concrete example arguments when args are abstract
            example_kws: Concrete example kwargs when kwargs are abstract
            sample_num: Maximum number of hyperparameter combinations to test.
                If fewer than total combinations, samples randomly.
            event_filter_regex: Optional regex filter for profiler events
            timeout: Optional compilation timeout override

        Returns:
            Tuple of (parameterized_fn, optimal_hyperparams, timing_results_sorted):
            - parameterized_fn: JIT-compiled function with optimal hyperparameters
            - optimal_hyperparams: Dictionary of optimal hyperparameter values
            - timing_results_sorted: List of (index, TimingResult) sorted by performance

        Raises:
            TypeError: If fn is not callable
            ValueError: If max_workers <= 0, sample_num < 0, or hyperparameters invalid
            RuntimeError: If all hyperparameter configurations fail to compile
        """
        if not callable(fn):
            raise TypeError("fn must be callable")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if sample_num < 0:
            raise ValueError("sample_num must be non-negative")

        if event_filter_regex is not None:
            self.profiler._pattern = re.compile(event_filter_regex)

        def _extract_array_type(x):
            """Extract the abstract type of a JAX array, or pass through non-array values."""
            return x if not isinstance(x, jax.Array) else jax.typeof(x)

        if len(args) == 0 or all(x is None or jax.core.is_concrete(x) for x in jax.tree.leaves(args)):
            resolved_args = args
        elif example_args is not None:
            if in_shardings is not None or device is not None:
                raise ValueError(
                    "Cannot combine example_args with explicit in_shardings or device configuration. "
                    "Example arguments should already be properly sharded and placed."
                )
            resolved_args = example_args
        else:
            resolved_device = device if isinstance(device, jax.Device) else _get_default_device()
            if isinstance(resolved_device, str):
                resolved_device = jax.devices(resolved_device)[0]
            input_shardings = in_shardings if in_shardings is not None else jax.tree.map(lambda _: None, args)
            normalized_shardings = jax.tree.map(
                partial(_normalize_sharding, default_device=resolved_device),
                args,
                input_shardings,
            )
            resolved_args = jax.tree.map(
                lambda x, s: _get_random_value(_extract_array_type(x), s), args, normalized_shardings
            )

        if len(kwargs) == 0 or all(v is None or jax.core.is_concrete(v) for v in kwargs.values()):
            resolved_kwargs = kwargs
        elif example_kws is not None:
            resolved_kwargs = example_kws
        else:
            resolved_kwargs = jax.tree.map(lambda x: _get_random_value(_extract_array_type(x)), kwargs)

        hyperparams = hyperparams if hyperparams is not None else {}
        hyperparams_norm: dict[str, tuple[Any, ...]] = {}
        for k, v in hyperparams.items():
            if isinstance(v, tuple | list):
                if len(v) == 0:
                    raise ValueError(f"Hyperparameter '{k}' has empty list of values")
                hyperparams_norm[k] = tuple(v)
            else:
                hyperparams_norm[k] = (v,)

        if hyperparams_norm:
            hyperparam_settings = dict(enumerate(itertools.product(*hyperparams_norm.values())))
            total_combinations = len(hyperparam_settings)

            if sample_num < total_combinations:
                if sample_num == 0:
                    hyperparam_settings = {0: tuple()}
                else:
                    sample_idx = sorted(
                        pyrandom.sample(list(range(total_combinations)), k=min(sample_num, total_combinations))
                    )
                    hyperparam_settings_ = list(hyperparam_settings.items())
                    hyperparam_settings = dict([hyperparam_settings_[idx] for idx in sample_idx])

            autotune_logger.info(
                f"Testing {len(hyperparam_settings)} hyperparameter combinations out of {total_combinations} possible"
            )
        else:
            hyperparam_settings = {0: tuple()}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fns: dict[int, Callable[..., Any]] = {}
            optimal_formats: dict[int, Any] = {}
            for phase in range(2):
                compile_only = phase == 0
                compute_layouts = self.find_optimal_layouts_automatically
                compiles = {}

                for i, vals in hyperparam_settings.items():
                    hs = dict(zip(hyperparams_norm.keys(), vals, strict=True))
                    fns[i] = self._create_parameterized_function(fn, hs, output_shardings=out_shardings, function_id=i)

                    opts = dict(
                        optimal_formats=optimal_formats.get(i, None),
                        compute_layouts=compute_layouts,
                        timeout=self.max_compilation_time_seconds if timeout is None else timeout,
                    )
                    fut = executor.submit(
                        self._try_call, fns[i], resolved_args, resolved_kwargs, compile_only=compile_only, **opts
                    )
                    compiles[fut] = i

                successful = {}
                for fut in as_completed(compiles):
                    status, err, optf = fut.result()
                    if status:
                        successful[compiles[fut]] = (status, err, optf)

                if compile_only and compute_layouts:
                    for i, (_, _, optf) in successful.items():
                        optimal_formats[i] = optf

                if not successful:
                    for fut, i in compiles.items():
                        _, err, _ = fut.result()
                        autotune_logger.error(
                            f"Hyperparameters {hyperparam_settings[i]} failed to compile with message:\n{err}"
                        )
                    raise ValueError("No hyperparameters compiled successfully")

                hyperparam_settings = {i: hyperparam_settings[i] for i in successful.keys()}
                fns = {i: fns[i] for i in successful.keys()}

        results: dict[int, TimingResult] = {}
        try:
            args_with_device = [
                next(iter(arg.devices())) for arg in jax.tree.leaves(resolved_args) if hasattr(arg, "devices")
            ]
            platform = args_with_device[0].platform if len(args_with_device) > 0 else _get_default_device().platform

            def _timing_closure():
                """Execute all hyperparameter configurations in random order for profiler timing."""
                settings = list(hyperparam_settings.items())
                pyrandom.shuffle(settings)
                for i, _ in settings:
                    self._try_call(
                        fns[i],
                        resolved_args,
                        resolved_kwargs,
                        compile_only=False,
                        optimal_formats=optimal_formats.get(i, None),
                    )

            profiler_timings = self.profiler.profile_time_by_function_id(
                _timing_closure,
                platform,
                self.profiling_samples,
            )
            fraction_measured = sum(1 for i in hyperparam_settings.keys() if i in profiler_timings) / len(
                hyperparam_settings
            )
            if fraction_measured < self.must_find_profiler_fraction:
                missing = [i for i in hyperparam_settings.keys() if i not in profiler_timings]
                msg = "Could not find profiler results for some hyperparameter settings:\n" + "\n".join(
                    f"  - {i}: {hyperparam_settings[i]}" for i in missing
                )
                raise RuntimeError(msg)

            for i in hyperparam_settings.keys():
                if i not in profiler_timings:
                    autotune_logger.warning(
                        f"Could not find profiler results for hyperparameter settings: {hyperparam_settings[i]}"
                    )
                    profiler_timings[i] = (float("inf"), float("inf"))

            for i, hs in hyperparam_settings.items():
                hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
                t_mean, t_std = profiler_timings[i]
                results[i] = TimingResult(hs, float(t_mean), float(t_std))
        except Exception as e:
            if not self.allow_fallback_timing:
                raise RuntimeError(
                    f"Need to fall back to the python-level timing, but allow_fallback_timing=False. Error: {e}"
                ) from None

            for i, hs in hyperparam_settings.items():
                hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
                t_mean, t_std = self._time_fn(partial(lambda fn: fn(*resolved_args, **resolved_kwargs), fns[i]))
                results[i] = TimingResult(hs, t_mean, t_std)

        results_sorted = sorted(results.items(), key=lambda x: self._calculate_timing_score(x[1]))
        idx, optimal_hyperparams = results_sorted[0][0], results_sorted[0][1].hyperparams
        return fns[idx], optimal_hyperparams, results_sorted

    def decorate(
        self,
        fn: Callable[..., Any],
        *,
        hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None,
        max_workers: int = 32,
        in_shardings: Any = None,
        out_shardings: Any = None,
        device: jax.Device | str | None = None,  # type: ignore
        example_args: tuple[Any, ...] | None = None,
        example_kws: dict[str, Any] | None = None,
        sample_num: int = 2**63 - 1,
        event_filter_regex: str | None = None,
        timeout: float | None = None,
        cache_key: str | None = None,
    ):
        """Create a decorated version of a function with automatic hyperparameter tuning.

        Wraps a function so that the first call triggers hyperparameter optimization,
        and subsequent calls with the same input signature use cached optimal values.

        Args:
            fn: Function to decorate with autotuning capabilities
            hyperparams: Dictionary mapping hyperparameter names to lists of candidate
                values to test during optimization
            max_workers: Maximum number of parallel workers for compilation
            in_shardings: Input sharding specifications for distributed computation
            out_shardings: Output sharding specifications for distributed computation
            device: Target device or device string for computation
            example_args: Concrete example arguments when function args are abstract
            example_kws: Concrete example kwargs when function kwargs are abstract
            sample_num: Maximum number of hyperparameter combinations to test
            event_filter_regex: Optional regex filter for profiler events
            timeout: Optional compilation timeout override
            cache_key: Optional custom cache key prefix for disambiguation

        Returns:
            Decorated function with automatic hyperparameter optimization.
            The returned function has additional attributes after first execution:
            - timing_results: List of all timing measurements from optimization
            - optimal_hyperparams: Dictionary of optimal parameter values

        Raises:
            TypeError: If fn is not callable
        """
        if not callable(fn):
            raise TypeError("fn must be callable")

        cache: dict[Any, tuple[dict[str, Any], list[tuple[int, TimingResult]]]] = {}
        cache_lock = threading.Lock()

        @wraps(fn)
        def wrapped(*args, **kws):
            """Wrapper function that performs autotuning on first call and caches results.

            On first call with a given input signature, performs hyperparameter
            optimization and caches the results. Subsequent calls with the same
            signature use the cached optimal hyperparameters.

            Args:
                *args: Positional arguments for the wrapped function
                **kws: Keyword arguments for the wrapped function

            Returns:
                Result of executing the function with optimal hyperparameters

            Raises:
                Exception: Any exception from the underlying function execution
            """
            input_hash = _try_hash_input(args, kws)
            lookup_key = f"{cache_key}:{input_hash}" if cache_key and input_hash else input_hash

            with cache_lock:
                hit = lookup_key is not None and lookup_key in cache
                if hit:
                    optimal_hyperparams, results = cache[lookup_key]
                else:
                    optimal_hyperparams = None
                    results = None

            if optimal_hyperparams is None:
                flat_vals = jax.tree.leaves((args, kws))
                has_tracers = any(not jax.core.is_concrete(x) for x in flat_vals if isinstance(x, jax.Array))
                if has_tracers:
                    if wrapped.optimal_hyperparams:
                        optimal_hyperparams = wrapped.optimal_hyperparams.copy()
                        results = wrapped.timing_results
                    else:
                        optimal_hyperparams = {
                            k: v[0] if isinstance(v, list) else v for k, v in (hyperparams or {}).items()
                        }
                        results = []
                else:
                    with jax.core.eval_context():
                        _, optimal_hyperparams, results = self.tune(
                            fn,
                            args=args,
                            kwargs=kws,
                            hyperparams=hyperparams,
                            max_workers=max_workers,
                            in_shardings=in_shardings,
                            out_shardings=out_shardings,
                            device=device,
                            example_args=example_args,
                            example_kws=example_kws,
                            sample_num=sample_num,
                            event_filter_regex=event_filter_regex,
                            timeout=timeout,
                        )

                    with cache_lock:
                        if lookup_key is not None:
                            if len(cache) >= self.cache_size_limit:
                                oldest_key = next(iter(cache))
                                del cache[oldest_key]
                            cache[lookup_key] = (optimal_hyperparams, results)

            wrapped.timing_results = results
            wrapped.optimal_hyperparams = optimal_hyperparams

            try:
                return fn(*args, **dict(kws, **optimal_hyperparams))
            except Exception as e:
                autotune_logger.error(f"Execution failed with optimal hyperparameters {optimal_hyperparams}: {e}")
                raise

        wrapped.timing_results = []
        wrapped.optimal_hyperparams = {}
        return wrapped


def autotune(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    allow_fallback_timing: bool = True,
    profiling_samples: int = 5,
    must_find_profiler_fraction: float = 0.5,
    enable_detailed_logging: bool = False,
    find_optimal_layouts_automatically: bool = False,
    max_compilation_time_seconds: float = 300.0,
    timing_warmup_iterations: int = 2,
    timing_rounds: int = 5,
    calls_per_round: int = 3,
    cache_size_limit: int = 1000,
    profiler_prefix_filter: str = "jit_",
    profiler_event_regex: str | None = None,
    profiler_min_duration_ns: float = 1000.0,
    profiler_max_events: int | None = 10000,
    profiler_verbose: bool = False,
    hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None,
    max_workers: int = 32,
    in_shardings: Any = None,
    out_shardings: Any = None,
    device: jax.Device | str | None = None,  # type: ignore
    example_args: tuple[Any, ...] | None = None,
    example_kws: dict[str, Any] | None = None,
    sample_num: int = 2**63 - 1,
    event_filter_regex: str | None = None,
    timeout: float | None = None,
    cache_key: str | None = None,
):
    """Advanced JAX function autotuning decorator with comprehensive optimization features.

    A flexible decorator that automatically optimizes JAX functions by testing different
    hyperparameter configurations. Uses profiler-based timing for accuracy with Python
    fallback, supports parallel execution, and provides intelligent caching.

    This decorator can be used in three different ways:
        1) @autotune()               -> decorator factory with custom parameters
        2) @autotune                 -> plain decorator with default parameters
        3) autotune(fn, kw=...)      -> direct function call returning wrapped function

    The decorator automatically optimizes hyperparameters on the first function call
    and caches results for subsequent calls with the same input signature.

    Key Features:
    - Profiler-based timing with Python-level fallback for maximum accuracy
    - Parallel compilation and testing of hyperparameter configurations
    - Automatic optimal memory layout discovery for distributed computation
    - Thread-safe caching with configurable size limits
    - Statistical timing analysis with outlier removal
    - Comprehensive error handling and detailed logging options
    - Support for both concrete and abstract input specifications

    Args:
        fn: Function to autotune (when used as direct call)


        allow_fallback_timing: Enable Python timing fallback if profiler fails
        profiling_samples: Number of profiling iterations for statistical accuracy
        must_find_profiler_fraction: Minimum fraction of configs needing profiler results (0.0-1.0)
        enable_detailed_logging: Enable detailed error logging with full tracebacks
        timing_warmup_iterations: Warmup iterations before timing measurements
        timing_rounds: Number of timing rounds for statistical analysis
        calls_per_round: Function calls per timing round


        profiler_prefix_filter: Event name prefix filter for profiler (default: "jit_")
        profiler_event_regex: Optional regex filter for profiler events
        profiler_min_duration_ns: Minimum event duration for profiler inclusion (nanoseconds)
        profiler_max_events: Maximum events per profile to prevent memory issues
        profiler_verbose: Enable verbose profiler output


        find_optimal_layouts_automatically: Auto-discover optimal memory layouts
        max_compilation_time_seconds: Maximum compilation time per configuration
        cache_size_limit: Maximum cached optimization results
        hyperparams: Dictionary mapping parameter names to candidate value lists
        max_workers: Maximum parallel workers for optimization
        sample_num: Maximum hyperparameter combinations to test


        in_shardings: Input sharding specifications for distributed computation
        out_shardings: Output sharding specifications for distributed computation
        device: Target device or device string for computation
        example_args: Concrete example arguments for abstract input shapes
        example_kws: Concrete example kwargs for abstract input values
        event_filter_regex: Optional regex filter for profiler events (alias for profiler_event_regex)
        timeout: Optional compilation timeout override
        cache_key: Optional custom cache key prefix for disambiguation

    Returns:
        When used as decorator: Decorated function with automatic hyperparameter optimization
        When used directly: Wrapper function that performs optimization and execution

        The returned function has additional attributes after first execution:
        - timing_results: List of all timing measurements from optimization
        - optimal_hyperparams: Dictionary of optimal parameter values

    Raises:
        TypeError: If fn is not callable (when used as direct call)
        ValueError: If hyperparameter specifications are invalid
        RuntimeError: If all hyperparameter configurations fail to compile

    Examples:
        Basic usage with hyperparameter optimization:
        ```python
        @autotune(hyperparams={'block_size': [64, 128, 256, 512]})
        def matrix_multiply(a, b, block_size=128):

            return jnp.dot(a, b)

        result = matrix_multiply(x, y)
        print(f"Optimal config: {matrix_multiply.optimal_hyperparams}")
        ```

        Advanced usage with custom timing and sharding:
        ```python
        @autotune(
            hyperparams={
                'chunk_size': [32, 64, 128],
                'algorithm': ['parallel', 'sequential']
            },
            profiling_samples=10,
            max_workers=16,
            in_shardings=jax.sharding.PartitionSpec('data', None),
            enable_detailed_logging=True
        )
        def compute_function(data, chunk_size=64, algorithm='parallel'):

            return processed_data
        ```

        Direct call usage:
        ```python
        optimized_fn = autotune(
            my_function,
            hyperparams={'param': [1, 2, 3]},
            profiling_samples=5
        )
        result = optimized_fn(input_data)
        ```

        Using with abstract input specifications:
        ```python
        @autotune(
            hyperparams={'tile_size': [16, 32, 64]},
            example_args=(jnp.zeros((1000, 1000)),),
            device='gpu'
        )
        def process_matrix(matrix, tile_size=32):

            return result
        ```

    Note:
        - First call triggers optimization and may take longer
        - Subsequent calls with same input signature use cached optimal parameters
        - Profiler-based timing requires TensorFlow backend; falls back to Python timing
        - For distributed computation, ensure proper mesh configuration before use
        - Cache keys are based on input signatures; use cache_key for manual disambiguation
    """
    tuner = FNAutotuner(
        allow_fallback_timing=allow_fallback_timing,
        profiling_samples=profiling_samples,
        must_find_profiler_fraction=must_find_profiler_fraction,
        enable_detailed_logging=enable_detailed_logging,
        find_optimal_layouts_automatically=find_optimal_layouts_automatically,
        max_compilation_time_seconds=max_compilation_time_seconds,
        timing_warmup_iterations=timing_warmup_iterations,
        timing_rounds=timing_rounds,
        calls_per_round=calls_per_round,
        cache_size_limit=cache_size_limit,
        profiler_prefix_filter=profiler_prefix_filter,
        profiler_event_regex=profiler_event_regex,
        profiler_min_duration_ns=profiler_min_duration_ns,
        profiler_max_events=profiler_max_events,
        profiler_verbose=profiler_verbose,
    )

    def decorator(func: Callable[..., Any]):
        """Internal decorator function that applies autotuning to a target function.

        Args:
            func: Function to wrap with autotuning capabilities

        Returns:
            Function decorated with automatic hyperparameter optimization
        """
        return tuner.decorate(
            func,
            hyperparams=hyperparams,
            max_workers=max_workers,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            device=device,
            example_args=example_args,
            example_kws=example_kws,
            sample_num=sample_num,
            event_filter_regex=event_filter_regex,
            timeout=timeout,
            cache_key=cache_key,
        )

    if callable(fn):
        return decorator(fn)

    return decorator
