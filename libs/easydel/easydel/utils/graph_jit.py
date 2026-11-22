# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Weight-safe ``jax.jit`` over ``spectrax`` modules.

JIT-compiling a function that reaches an :class:`spectrax.Module`'s weights
through a Python *closure* bakes those weight arrays into the compiled program
as captured constants. For real models that is gigabytes per executable: it
slows compilation, stacks a second weight copy in HBM, and pushes persistent
compilation-cache entries past protobuf's 2 GB parse limit (a poisoned cache
then SIGILL-crashes every later process that reads it). This exact bug bit the
MTP drafter, the speculative program, and the VLM vision jits.

:func:`module_jit` makes that impossible by construction: the module's weights
are threaded through the jit as a :class:`spectrax.State` *input* rather than
captured. The ``GraphDef`` (static structure) is split out once with
``spx.export`` and closed over — it holds no arrays — while the ``State`` is
re-exported from a live getter on every call, so hot weight swaps are observed
and the arrays never enter the compile key. ``spx.export`` only traverses the
module pytree and references the already-resident device buffers; it copies
nothing.

:func:`assert_no_large_constants` is the paired test guard: it fails if a
lowered program embeds a dense constant above a size threshold, catching a
weight-capture regression.
"""

from __future__ import annotations

import re
import typing as tp

import jax
import spectrax as spx

__all__ = ["assert_no_large_constants", "max_constant_elements", "module_jit"]

ModuleGetter = tp.Callable[[], tp.Any]


def module_jit(
    fn: tp.Callable[..., tp.Any],
    module_getters: ModuleGetter | tp.Sequence[ModuleGetter],
    *,
    static_argnames: tp.Sequence[str] = (),
) -> tp.Callable[..., tp.Any]:
    """Build a ``jax.jit`` closure that threads module weights as ``State`` inputs.

    The returned callable runs ``fn(*bound_modules, *args, **static_kwargs)``
    inside ``jax.jit``, where each ``bound_module`` is reconstructed *inside*
    the trace via ``spx.bind(graphdef, state)`` from a ``State`` passed as a
    jit input. Nothing about the model is captured as a constant.

    Args:
        fn: The traced body. It is called as
            ``fn(bound_module_0, bound_module_1, ..., *array_args, **static_kwargs)``
            — one leading bound module per entry in ``module_getters``, then the
            positional (traced) arguments, then the static keyword arguments.
        module_getters: A single zero-arg getter returning an ``spx.Module``, or
            a sequence of them (one per module ``fn`` expects). Getters are
            re-invoked on every call so a hot-swapped model is observed. Each
            getter must return a valid module by the first call.
        static_argnames: Names of keyword arguments to the returned callable
            that are compile-time constants (grid shapes, boolean flags, ...).
            Each distinct combination of static values gets its own compiled
            program. Values must be hashable.

    Returns:
        A callable ``run(*array_args, **static_kwargs)``. It also exposes
        ``run.lower(*array_args, **static_kwargs)`` which returns the lowered
        inner jit (weights threaded as a leading ``State`` tuple), for use with
        :func:`assert_no_large_constants` and ahead-of-time compilation.
    """
    getters: list[ModuleGetter] = [module_getters] if callable(module_getters) else list(module_getters)
    static_argnames = tuple(static_argnames)

    graphdefs_cell: list[tuple[spx.GraphDef, ...] | None] = [None]
    jit_cache: dict[tuple[tuple[str, tp.Any], ...], tp.Callable] = {}

    def _graphdefs() -> tuple[spx.GraphDef, ...]:
        if graphdefs_cell[0] is None:
            graphdefs_cell[0] = tuple(spx.export(getter())[0] for getter in getters)
        return graphdefs_cell[0]

    def _jit_for(static_items: tuple[tuple[str, tp.Any], ...]) -> tp.Callable:
        cached = jit_cache.get(static_items)
        if cached is not None:
            return cached
        graphdefs = _graphdefs()
        static_kwargs = dict(static_items)

        @jax.jit
        def _inner(states: tuple[spx.State, ...], *args: tp.Any) -> tp.Any:
            bound = tuple(spx.bind(gd, st) for gd, st in zip(graphdefs, states, strict=True))
            return fn(*bound, *args, **static_kwargs)

        jit_cache[static_items] = _inner
        return _inner

    def _split_static(kwargs: dict[str, tp.Any]) -> tuple[tuple[tuple[str, tp.Any], ...], dict[str, tp.Any]]:
        static_items = tuple((name, kwargs[name]) for name in static_argnames if name in kwargs)
        leftover = {k: v for k, v in kwargs.items() if k not in static_argnames}
        if leftover:
            raise TypeError(f"module_jit call got unexpected keyword argument(s): {', '.join(sorted(leftover))}.")
        return static_items, leftover

    def _states() -> tuple[spx.State, ...]:
        return tuple(spx.export(getter())[1] for getter in getters)

    def run(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        static_items, _ = _split_static(kwargs)
        return _jit_for(static_items)(_states(), *args)

    def lower(*args: tp.Any, **kwargs: tp.Any):
        """Lower the inner jit with current weights threaded as a ``State`` tuple."""
        static_items, _ = _split_static(kwargs)
        return _jit_for(static_items).lower(_states(), *args)

    run.lower = lower  # type: ignore[attr-defined]
    return run


def max_constant_elements(mlir_text: str) -> int:
    """Return the element count of the largest dense constant in lowered MLIR.

    Args:
        mlir_text: The lowered module text (``lowered.as_text()``).

    Returns:
        The number of elements in the largest ``constant`` dense tensor, or 0
        when the module has no dense constants.
    """
    worst = 0
    for line in mlir_text.splitlines():
        if "constant" not in line:
            continue
        for shape in re.findall(r"tensor<([0-9x]+)x[a-z]", line):
            elems = 1
            for dim in shape.split("x"):
                if dim:
                    elems *= int(dim)
            worst = max(worst, elems)
    return worst


def assert_no_large_constants(lowered: tp.Any, *, max_elements: int = 10_000) -> int:
    """Assert a lowered program embeds no dense constant at/above a size bound.

    A weight-capture bug shows up as a large dense constant in the lowered
    program (the weights were baked in instead of arriving as arguments). This
    guard fails when the largest constant reaches ``max_elements``.

    Args:
        lowered: A ``jax.stages.Lowered`` (e.g. from ``module_jit(...).lower(...)``)
            or any object with an ``as_text()`` method, or the MLIR text itself.
        max_elements: Exclusive upper bound on the largest dense constant's
            element count. Choose it below the smallest real weight tensor of
            the model under test and above its index/position constants.

    Returns:
        The largest dense-constant element count found (below ``max_elements``).

    Raises:
        AssertionError: If a dense constant with ``>= max_elements`` elements is
            present (weight capture regressed).
    """
    mlir_text = lowered if isinstance(lowered, str) else lowered.as_text()
    worst = max_constant_elements(mlir_text)
    if worst >= max_elements:
        raise AssertionError(
            f"lowered program embeds a {worst}-element dense constant "
            f"(>= {max_elements}); model weights were captured as constants "
            f"instead of threaded as a State input."
        )
    return worst
