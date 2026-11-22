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


"""Execution engine and batch processing for ejkernel.ops.

This module provides the main execution engine for running kernels with automatic
configuration selection, as well as utilities for batch processing, autotuning,
and profiling.

Classes:
    Executor: Main execution engine coordinating config selection and kernel
        execution.  Handles argument preprocessing, custom VJP wiring,
        profiling stamps, and invocation recording.
    ConfigChooser: Protocol that :class:`~ejkernel.ops.config.ConfigSelectorChain`
        implements.  Defines the ``choose(inv, kernel) -> cfg`` interface.
    Autotuner: Lightweight autotuner that JIT-compiles each candidate and
        times it.  Returns :class:`AutotuneData` with all measurements.
        Also exported as the alias ``Tuner`` for backward compatibility.
    FNAutotuner: Advanced autotuner with profiler-based timing, parallel
        compilation, statistical analysis, and thread-safe caching.
    AutotuneData: Container for all :class:`Measurement` objects from an
        :class:`Autotuner` run.
    AutotuningResult: Frozen container of :class:`Entry` objects returned
        by :func:`autotune_recorded` or :func:`autotune_lowered`.  Acts as
        a context manager that applies results as a cache overlay.
    Entry: Single ``(op_id_v, call_key, cfg)`` record inside
        :class:`AutotuningResult`.
    Measurement: Single ``(cfg, seconds)`` record from :class:`Autotuner`.

Functions:
    vmap_with_config: Vectorized execution that selects a configuration once
        (using the first element) and then applies ``jax.vmap``.
    pmap_with_config: Multi-device parallel execution that selects a
        configuration once (using data from device 0) and then applies
        ``jax.pmap``.
    autotune: Decorator (or direct wrapper) that uses :class:`FNAutotuner`
        to optimize hyperparameters on the first call and cache results.
    autotune_recorded: Autotune all kernel invocations stored in the global
        registry for the current device.  Requires ``EJKERNEL_OPS_RECORD=1``
        during initial runs to populate the registry.
    autotune_lowered: Autotune ejkernel operations found in the HLO of a
        ``jax.jit(...).lower(...)`` result.  Matches HLO labels to registry
        entries and benchmarks candidate configurations.
    benchmark: Compile a function with ``jax.jit`` and time ``iters``
        iterations after ``warmup`` warmup calls.

Note:
    ``Tuner`` is an alias for ``Autotuner`` kept for backward compatibility.

Example:
    >>> from ejkernel.ops.execution import Executor, autotune
    >>>
    >>> executor = Executor(selector)
    >>> result = executor(my_kernel, input_data)
    >>>
    >>> @autotune(hyperparams={'block_size': [64, 128, 256]})
    ... def my_function(x, block_size=128):
    ...     return process(x, block_size)
"""

from .batch import pmap_with_config, vmap_with_config
from .executor import ConfigChooser, Executor
from .offline import autotune_lowered
from .tuning import (
    AutotuneData,
    Autotuner,
    AutotuningResult,
    Entry,
    FNAutotuner,
    Measurement,
    autotune,
    autotune_recorded,
    benchmark,
)

Tuner = Autotuner

__all__ = (
    "AutotuneData",
    "Autotuner",
    "AutotuningResult",
    "ConfigChooser",
    "Entry",
    "Executor",
    "FNAutotuner",
    "Measurement",
    "Tuner",
    "autotune",
    "autotune_lowered",
    "autotune_recorded",
    "benchmark",
    "pmap_with_config",
    "vmap_with_config",
)
