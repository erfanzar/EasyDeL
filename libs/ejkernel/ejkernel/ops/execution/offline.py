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


"""Offline autotuning driven by the HLO of lowered JAX computations.

This module provides :func:`autotune_lowered`, which extracts the ejkernel
operation labels embedded in a ``jax.jit(...).lower(...)`` result, looks up
matching invocations in the global registry, and benchmarks every candidate
configuration under ``jax.core.eval_context`` so that autotuning runs outside
the JAX tracing stack.

The offline tuning workflow:
    1. Lower a JAX function to obtain its HLO representation.
    2. Parse the HLO text to extract all ``ejkernel_ops#...`` labels.
    3. For each label, look up the recorded ``(kernel, args, kwargs)`` in the
       global registry (populated when ``EJKERNEL_OPS_RECORD=1`` is set).
    4. Benchmark each candidate configuration returned by
       ``kernel.candidate_cfgs()`` using :class:`~ejkernel.ops.config.Tuner`.
    5. Write the fastest configuration to the in-memory and persistent caches.
    6. Return an :class:`~ejkernel.ops.execution.tuning.AutotuningResult` that
       can be used as a context manager to apply all results at once.

Note:
    Operations not present in the registry are silently skipped.  Enable
    ``EJKERNEL_OPS_RECORD=1`` during an initial non-tuning run to populate
    the registry.

Example:
    >>> lowered = jax.jit(my_function).lower(example_args)
    >>> result = autotune_lowered(selector, lowered)
    >>> with result:  # temporarily apply the optimised configurations
    ...     output = jax.jit(my_function)(real_args)
"""

from __future__ import annotations

import jax

from ..config.selection import ConfigSelectorChain
from ..core import _get_platform_method
from ..registry import get_invocations
from ..utils.fingerprint import device_fingerprint, get_device_platform
from ..utils.meta import LABEL_RE, extract_labels_from_hlo_text
from .tuning import AutotuningResult, Entry


def _labels_to_invocations(lowered) -> list[tuple[str, str]]:
    """Parse ejkernel labels from a lowered computation into (op_id_v, call_key) pairs.

    Extracts the HLO text from ``lowered`` (falling back to ``str(lowered)``
    if the HLO dialect is unavailable) and applies :data:`~ejkernel.ops.utils.meta.LABEL_RE`
    to find every embedded ``ejkernel_ops#<op>:<key>`` label.

    Args:
        lowered: A ``jax.stages.Lowered`` object (the result of
            ``jax.jit(...).lower(...)``).

    Returns:
        List of ``(op_id_v, call_key)`` tuples, one per matching label found
        in the HLO text.  ``op_id_v`` has the form ``'<name>@v<version>'``
        and ``call_key`` is a 16-character hexadecimal hash.
    """
    try:
        hlo_text = lowered.compiler_ir(dialect="hlo").as_text()
    except Exception:
        hlo_text = str(lowered)
    labels = extract_labels_from_hlo_text(hlo_text)
    pairs = []
    for lab in labels:
        m = LABEL_RE.search(lab)
        if not m:
            continue
        pairs.append((m.group("op"), m.group("key")))
    return pairs


def autotune_lowered(selector: ConfigSelectorChain, lowered) -> AutotuningResult:
    """Autotune all ejkernel operations found in a lowered JAX computation.

    Parses the HLO of ``lowered`` to find embedded ``ejkernel_ops#`` labels,
    matches each label against the global invocation registry, and benchmarks
    the candidate configurations for every matched operation inside a
    ``jax.core.eval_context`` so that execution is not traced.

    Winning configurations are written to ``selector.cache`` and, when
    ``selector.persistent is not None and selector.persist_autotune``, to
    ``selector.persistent`` as well.

    Args:
        selector: :class:`~ejkernel.ops.config.ConfigSelectorChain` that owns
            the ``tuner``, ``cache``, and (optionally) ``persistent`` attributes
            used during benchmarking and result storage.
        lowered: A ``jax.stages.Lowered`` object — the result of
            ``jax.jit(fn).lower(*example_args)``.

    Returns:
        :class:`~ejkernel.ops.execution.tuning.AutotuningResult` wrapping all
        tuned ``(op_id_v, call_key, best_cfg)`` entries for the current device.
        Can be used as a context manager to apply the results as a cache overlay.

    Example:
        >>> lowered = jax.jit(my_model).lower(example_input)
        >>> result = autotune_lowered(selector, lowered)
        >>> with result:
        ...     output = jax.jit(my_model)(real_input)

    Note:
        Operations not present in the invocation registry are silently skipped.
        Set ``EJKERNEL_OPS_RECORD=1`` and run the model once to populate the
        registry before calling this function.
    """
    dev = device_fingerprint()
    invs = get_invocations(dev)
    targets = _labels_to_invocations(lowered)
    entries: list[Entry] = []
    platform = get_device_platform()

    for op_id_v, call_key in targets:
        if op_id_v not in invs or call_key not in invs[op_id_v]:
            continue
        kernel, args, kwargs = invs[op_id_v][call_key]
        inv_args, inv_kwargs = kernel.prepare(*args, **kwargs)
        static_fun_kwargs = {k: v for k, v in inv_kwargs.items() if callable(v)}
        dyn_kwargs = {k: v for k, v in inv_kwargs.items() if not callable(v)}
        tmp_inv = type(
            "Tmp",
            (),
            dict(op_id=kernel.op_id, args=inv_args, kwargs=dyn_kwargs, batch_axes=None, override_cfg=None, stamp=False),
        )()

        cand_method = _get_platform_method(kernel, "candidate_cfgs", platform, None) or kernel.candidate_cfgs
        candidates = tuple(cand_method(tmp_inv))
        run_method = _get_platform_method(kernel, "run", platform, None) or kernel.run

        def mk(c, _run=run_method, _static=static_fun_kwargs):
            """Return a callable that runs ``kernel.run`` with configuration ``c``."""

            def f(*a, **k):
                """Call the resolved run method with config ``c`` and closed-over static kwargs."""
                return _run(*a, cfg=c, **(k | _static))  # noqa: B023

            return f

        with jax.core.eval_context():
            best = selector.tuner.autotune(mk, inv_args, dyn_kwargs, candidates)
        selector.cache.put(dev, op_id_v, call_key, best)
        if selector.persistent is not None and selector.persist_autotune:
            selector.persistent.put(dev, op_id_v, call_key, best)
        entries.append(Entry(op_id_v, call_key, best))

    return AutotuningResult(dev, tuple(entries))
