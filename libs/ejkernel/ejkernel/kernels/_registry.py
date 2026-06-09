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


"""Kernel registry system for managing multi-platform implementations.

This module provides the core registry infrastructure for ejkernel's
multi-platform kernel dispatch system. It enables registration and
lookup of kernel implementations across different platforms (Triton,
Pallas, CUDA, XLA) and backends (GPU, TPU, CPU).

Key Components:
    - Platform: Enum for implementation platforms (TRITON, PALLAS, CUDA, CUTE, XLA)
    - Backend: Enum for hardware backends (GPU, TPU, CPU, ANY)
    - KernelSpec: Dataclass describing a registered kernel implementation
    - KernelRegistry: Registry class for managing kernel implementations
    - kernel_registry: Global singleton registry instance

Usage:
    Registration:
        @kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU)
        def flash_attention_triton(q, k, v): ...

    Lookup:
        impl = kernel_registry.get("flash_attention", platform="triton", backend="gpu")
        result = impl(q, k, v)

Priority System:
    Multiple implementations of the same algorithm can coexist with different
    priorities. When looking up a kernel, the highest priority match is returned.
    This enables optimized implementations to take precedence over fallbacks.

Signature Validation:
    The validate_signatures() method ensures all implementations of an algorithm
    have compatible signatures, catching registration errors early.

Example:
    >>> from ejkernel.kernels._registry import kernel_registry, Platform, Backend
    >>>
    >>> # List all registered algorithms
    >>> algorithms = kernel_registry.list_algorithms()
    >>>
    >>> # Get implementations for an algorithm
    >>> impls = kernel_registry.list_implementations("flash_attention")
    >>>
    >>> # Get best implementation for current platform
    >>> impl = kernel_registry.get("flash_attention", platform=Platform.XLA)
"""

from __future__ import annotations

import ast
import functools
import inspect
import re
import textwrap
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeVar, overload

import jax

from ejkernel.errors import EjkernelRuntimeError

F = TypeVar("F", bound=Callable)
"""TypeVar bound to Callable, used for preserving function signatures in decorators."""

_IGNORED_PARAM_CACHE: dict[Callable[..., Any], set[str]] = {}
"""Cache mapping functions to the set of parameter names explicitly deleted within their body."""

_TUNING_PARAM_NAMES: set[str] = {
    "block_m",
    "block_n",
    "block_k",
    "block_q",
    "block_kv",
    "block_size",
    "block_v",
    "block_d",
    "block_e",
    "block_dim",
    "num_warps",
    "num_stages",
    "num_ctas",
    "num_waves",
    "num_sms",
    "num_splits",
    "split_k",
    "use_bf16",
    "seq_threshold_3d",
    "num_par_softmax_segments",
    "max_warps",
    "num_queries_per_block",
    "num_kv_pages_per_block",
    "num_kv_splits",
    "gemm_block",
    "fwd_params",
    "bwd_params",
}


def _get_ignored_params(func: Callable[..., Any]) -> set[str]:
    """Detect parameter names explicitly deleted inside a function body.

    Parses the source code of *func* using the ``ast`` module and collects
    all names that appear in ``del`` statements. These names typically
    represent parameters the implementation chooses to discard, signaling
    that the corresponding feature is unsupported.

    Results are cached per function in ``_IGNORED_PARAM_CACHE`` to avoid
    repeated source parsing.

    Args:
        func: The callable whose source code will be inspected.

    Returns:
        A set of parameter names found in ``del`` statements within
        *func*. Returns an empty set if the source cannot be retrieved
        or parsed.
    """
    cached = _IGNORED_PARAM_CACHE.get(func)
    if cached is not None:
        return cached
    try:
        source = inspect.getsource(func)
    except OSError:
        _IGNORED_PARAM_CACHE[func] = set()
        return _IGNORED_PARAM_CACHE[func]
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        _IGNORED_PARAM_CACHE[func] = set()
        return _IGNORED_PARAM_CACHE[func]

    ignored: set[str] = set()

    class _DelVisitor(ast.NodeVisitor):
        """AST visitor that collects names from ``del`` statements."""

        def visit_Delete(self, node: ast.Delete) -> None:
            """Record every ``ast.Name`` target in a ``del`` statement."""
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ignored.add(target.id)
            self.generic_visit(node)

    _DelVisitor().visit(tree)
    _IGNORED_PARAM_CACHE[func] = ignored
    return ignored


def _is_non_default(value: Any, default: Any) -> bool:
    """Determine whether a supplied argument value differs from its default.

    Uses identity checks for ``None`` and ``inspect._empty``, equality
    checks for common scalar types (``bool``, ``int``, ``float``, ``str``),
    and falls back to identity comparison for all other types.

    Args:
        value: The argument value that was actually passed by the caller.
        default: The parameter's default value from its signature. If the
            parameter has no default, this should be ``inspect._empty``.

    Returns:
        ``True`` if *value* is considered different from *default*,
        ``False`` otherwise.
    """
    if default is inspect._empty:
        return True
    if default is None:
        return value is not None
    if isinstance(default, (bool, int, float, str)):
        return value != default
    return value is not default


def _collect_unsupported_reasons(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[str]:
    """Collect human-readable reasons why a kernel call is unsupported.

    This function checks two sources of unsupported-feature information:

    1. **Ignored parameters** -- Parameters whose names appear in ``del``
       statements within *func* (detected by ``_get_ignored_params``).  If the
       caller supplied a non-default value for such a parameter and it is not
       a tuning parameter, a reason string is generated.
    2. **Explicit unsupported hook** -- If *func* carries an
       ``__ejkernel_unsupported__`` attribute, it is invoked (or iterated)
       to produce additional reason strings.

    Args:
        func: The kernel implementation function to inspect.
        args: Positional arguments passed to the kernel call.
        kwargs: Keyword arguments passed to the kernel call.

    Returns:
        A list of human-readable reason strings. An empty list means the
        call is fully supported.
    """
    reasons: list[str] = []
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    ignored = _get_ignored_params(func)
    for name in ignored:
        if name in _TUNING_PARAM_NAMES:
            continue
        if name in bound.arguments and _is_non_default(bound.arguments[name], sig.parameters[name].default):
            reasons.append(f"{name} is not supported")

    extra = getattr(func, "__ejkernel_unsupported__", None)
    if extra is not None:
        if callable(extra):
            try:
                result = extra(**bound.arguments)
            except TypeError:
                result = extra(bound.arguments)
            if isinstance(result, str):
                reasons.append(result)
            elif isinstance(result, Iterable):
                reasons.extend([str(item) for item in result if item])
        elif isinstance(extra, str):
            reasons.append(extra)
        elif isinstance(extra, Iterable):
            reasons.extend([str(item) for item in extra if item])
    return reasons


def _normalize_type_string(type_annotation: Any) -> str:
    """Normalize type annotation string for comparison.

    Handles cases where the same type is imported differently:
    - 'jaxtyping.Float' -> 'Float'
    - 'ejkernel.ops.utils.datacarrier.FwdParams' -> 'FwdParams'

    Args:
        type_annotation: The type annotation to normalize

    Returns:
        Normalized string representation of the type
    """
    if type_annotation is inspect._empty:
        return "inspect._empty"

    type_str = type_annotation if isinstance(type_annotation, str) else str(type_annotation)

    type_str = re.sub(r"<class '(.+)'>", r"\1", type_str)

    type_str = re.sub(r"\bjaxtyping\.", "", type_str)

    type_str = re.sub(r"\b(?:jaxlib\._jax\.Array|jax\.jaxlib\._jax\.Array|jax\.Array)\b", "Array", type_str)
    type_str = re.sub(r"\bjax\._src\.lax\.lax\.", "", type_str)
    type_str = re.sub(r"\bjax\._src\.typing\.", "", type_str)
    type_str = re.sub(r"\bjax\.typing\.", "", type_str)
    type_str = re.sub(r"\bjax\.lax\.", "lax.", type_str)

    type_str = re.sub(r"\bejkernel\.[\w\.]+\.(\w+)", r"\1", type_str)

    if "PrecisionLike" in type_str or "DotAlgorithm" in type_str or "DotAlgorithmPreset" in type_str:
        return "PrecisionLike"
    if "DTypeLike" in type_str or "SupportsDType" in type_str:
        return "DTypeLike"

    return type_str


def _types_are_equivalent(type1: Any, type2: Any) -> bool:
    """Check if two type annotations are equivalent.

    This handles cases where the same type might be imported differently
    in different modules, e.g., 'Float' vs 'jaxtyping.Float'.

    Args:
        type1: First type annotation
        type2: Second type annotation

    Returns:
        True if types are equivalent, False otherwise
    """

    if type1 is inspect._empty and type2 is inspect._empty:
        return True

    if (type1 is inspect._empty) != (type2 is inspect._empty):
        return False

    normalized1 = _normalize_type_string(type1)
    normalized2 = _normalize_type_string(type2)

    return normalized1 == normalized2


class Platform(StrEnum):
    """Supported kernel implementation platforms.

    Each member identifies a compilation/execution framework used to
    implement a kernel.

    Attributes:
        TRITON: OpenAI Triton GPU kernels.
        PALLAS: JAX Pallas kernels (supports both GPU and TPU).
        CUDA: Native CUDA C/C++ kernels compiled ahead-of-time.
        CUTE: CUTLASS CuTe DSL kernels.
        TILELANG: tile-lang DSL kernels (Microsoft/TileLang, TVM-FFI backed).
        XLA: XLA HLO-based implementations using JAX primitives.
    """

    TRITON = "triton"
    PALLAS = "pallas"
    CUDA = "cuda"
    CUTE = "cute"
    TILELANG = "tilelang"
    XLA = "xla"


class Backend(StrEnum):
    """Target hardware backends for kernel execution.

    Used to tag kernel implementations with the hardware they target.
    During lookup, ``Backend.ANY`` acts as a wildcard that matches every
    backend query, making it suitable for platform-agnostic implementations.

    Attributes:
        GPU: GPU backend (typically CUDA/ROCm-class).
        MPS: Apple Silicon Metal/MPS backend (``jax.default_backend() == "mps"``).
        TPU: Google TPU backend.
        CPU: CPU-only backend.
        ANY: Wildcard backend matching any hardware target.
    """

    GPU = "gpu"
    MPS = "mps"
    TPU = "tpu"
    CPU = "cpu"
    ANY = "any"


@dataclass(frozen=True)
class KernelSpec:
    """Immutable specification describing a single registered kernel implementation.

    Each ``KernelSpec`` binds a concrete callable to its algorithm name,
    target platform, and hardware backend.  The ``priority`` field governs
    selection order when multiple implementations match a lookup query --
    higher values are preferred.

    Attributes:
        platform: The implementation platform (e.g., ``Platform.TRITON``).
        backend: Target hardware backend (e.g., ``Backend.GPU``).
        algorithm: Canonical algorithm name (e.g., ``'flash_attention'``).
        implementation: The wrapped kernel callable.
        priority: Selection priority; higher values are preferred during
            lookup.  Defaults to ``0``.
    """

    platform: Platform
    backend: Backend
    algorithm: str
    implementation: Callable
    priority: int = 0


class KernelRegistry:
    """Central registry for managing kernel implementations across platforms and backends.

    ``KernelRegistry`` is the backbone of ejkernel's multi-platform dispatch
    system.  It stores ``KernelSpec`` objects keyed by algorithm name and
    provides decorator-based registration, priority-aware lookup, and
    cross-implementation signature validation.

    The typical usage pattern is:

    1. **Register** implementations via the ``register`` decorator.
    2. **Look up** the best implementation for a given algorithm / platform /
       backend combination via ``get``.
    3. (Optional) **Validate** that all implementations of an algorithm share
       a compatible signature via ``validate_signatures``.

    Attributes:
        _registry: Internal mapping from lower-cased algorithm names to
            lists of ``KernelSpec`` objects sorted by descending priority.

    Example:
        >>> registry = KernelRegistry()
        >>> @registry.register("flash_attention", Platform.TRITON, Backend.GPU)
        ... def flash_attention_triton(q, k, v): ...
        >>>
        >>> impl = registry.get("flash_attention", platform="triton", backend="gpu")
    """

    def __init__(self) -> None:
        """Initialize an empty kernel registry with no registered algorithms."""
        self._registry: dict[str, list[KernelSpec]] = {}

    @overload
    def register(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla"],
        backend: Backend | Literal["gpu", "mps", "tpu", "cpu", "any"],
        priority: int = 0,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla"],
        backend: Backend | Literal["gpu", "tpu", "cpu", "any"],
        priority: int = 0,
    ) -> Callable[[F], F]:
        """Decorator to register a kernel implementation.

        Wraps *func* in a validation layer that checks for unsupported
        parameters before dispatch and converts certain runtime exceptions
        into ``EjkernelRuntimeError``.  The wrapped function is stored in a
        ``KernelSpec`` and appended to the internal registry under
        *algorithm* (case-insensitive).

        Args:
            algorithm: Name of the algorithm (e.g., ``'flash_attention'``).
            platform: Implementation platform.  Accepts a ``Platform``
                enum member or its string value.
            backend: Target hardware backend.  Accepts a ``Backend``
                enum member or its string value.
            priority: Selection priority (default: ``0``). Higher values
                are preferred during lookup.

        Returns:
            A decorator that registers the kernel and returns the
            wrapped callable (preserving the original signature).

        Example:
            >>> @registry.register("flash_attention", Platform.TRITON, Backend.GPU, priority=10)
            ... def flash_attention_impl(q, k, v):
            ...     return compute_attention(q, k, v)
        """

        def decorator(func: F) -> F:
            """Inner decorator that wraps and registers *func*."""
            key = algorithm.lower()
            if key not in self._registry:
                self._registry[key] = []

            plat = Platform(platform) if isinstance(platform, str) else platform
            back = Backend(backend) if isinstance(backend, str) else backend

            @functools.wraps(func)
            def _wrapped(*args, **kwargs):
                """Validation wrapper that guards the kernel call.

                Before forwarding to the original implementation, this
                wrapper checks for unsupported parameter usage and
                re-raises certain exceptions as ``EjkernelRuntimeError``.
                """
                reasons = _collect_unsupported_reasons(func, args, kwargs)
                if reasons:
                    raise EjkernelRuntimeError(f"{algorithm} (platform={plat.value}): " + "; ".join(reasons))
                try:
                    return func(*args, **kwargs)
                except EjkernelRuntimeError:
                    raise
                except (NotImplementedError, ValueError) as exc:
                    msg = str(exc)
                    if "not supported" in msg.lower() or "unsupported" in msg.lower():
                        raise EjkernelRuntimeError(f"{algorithm} (platform={plat.value}): {msg}") from exc
                    raise

            _wrapped.__signature__ = inspect.signature(func)

            spec = KernelSpec(
                platform=plat,
                backend=back,
                algorithm=algorithm,
                implementation=_wrapped,
                priority=priority,
            )
            self._registry[key].append(spec)

            self._registry[key].sort(key=lambda x: x.priority, reverse=True)
            return _wrapped  # type: ignore[return-value]

        return decorator

    def get(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None,
        backend: Backend | Literal["gpu", "mps", "tpu", "cpu", "any"] | None = None,
    ) -> Callable:
        """Retrieve the best matching kernel implementation.

        Searches for implementations matching the specified algorithm,
        platform, and backend.  Returns the highest-priority match among
        all candidates.

        Matching rules:
            - If *platform* is ``None``, any platform matches.
            - If *backend* is ``None``, any backend matches.
            - ``Backend.ANY`` implementations match every backend query.
            - When *platform* is ``Platform.XLA`` and no direct match is
              found, a fallback lookup with ``Backend.ANY`` is attempted.
            - When *backend* is ``Backend.ANY`` and no match is found, a
              fallback lookup using ``jax.default_backend()`` is attempted.

        Args:
            algorithm: Algorithm name to look up (case-insensitive).
            platform: Optional platform filter.  Accepts a ``Platform``
                enum member, its string value, or ``"auto"``.
            backend: Optional backend filter.  Accepts a ``Backend``
                enum member or its string value.

        Returns:
            The matching kernel implementation callable.

        Raises:
            ValueError: If no matching implementation is found after all
                fallback attempts.

        Example:
            >>> impl = registry.get("flash_attention", platform="triton", backend="gpu")
            >>> result = impl(q, k, v)
        """
        key = algorithm.lower()
        if key not in self._registry:
            raise ValueError(f"No implementation found for algorithm: {algorithm}")

        candidates = self._registry[key]

        if isinstance(platform, str):
            platform = Platform(platform)
        if isinstance(backend, str):
            backend = Backend(backend)

        for spec in candidates:
            if platform is not None and spec.platform != platform:
                continue
            if backend is not None and spec.backend != backend and spec.backend != Backend.ANY:
                continue
            return spec.implementation

        if platform == Platform.XLA:
            return self.get(algorithm=algorithm, platform=platform, backend=Backend.ANY)
        if backend == Backend.ANY:
            return self.get(algorithm=algorithm, platform=platform, backend=jax.default_backend())
        raise ValueError(f"No implementation found for algorithm={algorithm}, platform={platform}, backend={backend}")

    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names.

        Returns:
            A sorted list of lower-cased algorithm name strings currently
            present in the registry.
        """
        return sorted(self._registry.keys())

    def list_implementations(self, algorithm: str) -> list[KernelSpec]:
        """List all registered implementations for a given algorithm.

        Args:
            algorithm: Algorithm name to query (case-insensitive).

        Returns:
            A shallow copy of the ``KernelSpec`` list for the algorithm,
            sorted by priority in descending order.  Returns an empty
            list if the algorithm has not been registered.
        """
        key = algorithm.lower()
        return self._registry.get(key, []).copy()

    def validate_signatures(self, algorithm: str | None, verbose: bool = False) -> bool:
        """Validate that all implementations of an algorithm have compatible signatures.

        Uses the first registered implementation (highest priority) as the
        reference and compares every subsequent implementation against it.
        The following properties are checked for each parameter:

        - **Name** -- parameter names must match in order.
        - **Kind** -- positional-only, keyword-only, etc. must agree.
        - **Default value** -- default values must be equal.
        - **Type annotation** -- annotations are compared after
          normalization via ``_types_are_equivalent``.

        Any mismatch emits a ``UserWarning`` describing the discrepancy.

        When *algorithm* is ``None``, **all** registered algorithms are
        validated in turn (the return value is ``None`` in this case).

        Args:
            algorithm: Algorithm name to validate (case-insensitive), or
                ``None`` to validate every registered algorithm.
            verbose: If ``True``, print detailed parameter information for
                every implementation before running comparisons.

        Returns:
            ``True`` if all signatures match, ``False`` otherwise.
            Returns ``None`` implicitly when *algorithm* is ``None``.

        Raises:
            ValueError: If the specified algorithm has not been registered.
        """
        if algorithm is None:
            for algo in self.list_algorithms():
                self.validate_signatures(algo)
            return
        key = algorithm.lower()
        if key not in self._registry:
            raise ValueError(f"No implementation found for algorithm: {algorithm}")

        specs = self._registry[key]
        if len(specs) < 2:
            return True

        reference_spec = specs[0]
        reference_sig = inspect.signature(reference_spec.implementation)
        reference_params = [
            param for param in reference_sig.parameters.values() if param.name not in _TUNING_PARAM_NAMES
        ]

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Algorithm: {algorithm}")
            print(f"{'=' * 80}")
            for spec in specs:
                sig = inspect.signature(spec.implementation)
                print(f"\n{spec.platform}/{spec.backend} (priority={spec.priority}):")
                print(f"  Signature: {sig}")
                for param_name, param in sig.parameters.items():
                    print(f"    {param_name}:")
                    print(f"      kind: {param.kind.name}")
                    print(f"      default: {param.default}")
                    print(f"      annotation: {param.annotation}")
            print(f"{'=' * 80}\n")

        all_match = True

        for spec in specs[1:]:
            sig = inspect.signature(spec.implementation)
            params = [param for param in sig.parameters.values() if param.name not in _TUNING_PARAM_NAMES]

            if len(params) != len(reference_params):
                warnings.warn(
                    f"Signature mismatch for algorithm '{algorithm}':\n"
                    f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                    f"{len(reference_params)} parameters\n"
                    f"  Implementation ({spec.platform}/{spec.backend}): {len(params)} parameters",
                    UserWarning,
                    stacklevel=2,
                )
                all_match = False
                continue

            for ref_param, param in zip(reference_params, params, strict=False):
                if ref_param.name != param.name:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"parameter '{ref_param.name}'\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): parameter '{param.name}'",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if ref_param.kind != param.kind:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): {ref_param.kind.name}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): {param.kind.name}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if ref_param.default != param.default:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"default={ref_param.default}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): default={param.default}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if not _types_are_equivalent(ref_param.annotation, param.annotation):
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"type={ref_param.annotation} = {ref_param.default}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): type={param.annotation} = {param.default}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

        return all_match


kernel_registry = KernelRegistry()
"""Global singleton ``KernelRegistry`` instance.

All built-in kernel implementations register themselves against this
instance at import time.  User code should typically interact with this
object rather than creating a new ``KernelRegistry``.
"""
