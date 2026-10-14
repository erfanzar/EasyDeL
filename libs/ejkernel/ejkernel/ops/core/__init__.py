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


"""Core abstractions for the ejkernel.ops framework.

This module provides the fundamental building blocks for implementing JAX operations
with automatic configuration management and performance optimization.

Classes:
    Kernel: Abstract base class for implementing custom operations with configuration
        management, autotuning support, and optional custom gradient implementations.
        Supports platform-specific and context-specific method dispatch.
    Invocation: Represents a specific call to a kernel with arguments, metadata,
        and execution context (standard, shard_map, etc.). Provides cache key
        generation for configuration lookups.

Type Variables:
    Cfg: Configuration type parameter for kernels. Can be any type (dataclass,
        dict, NamedTuple, etc.) that specifies how the operation should execute.
    Out: Output type parameter for kernels. Can be any type (JAX arrays, tuples,
        nested structures) that the operation returns.

Functions:
    _has_custom_vjp: Utility to detect if a kernel has implemented custom VJP
        (vector-Jacobian product) methods for gradient computation. Supports
        checking for context-specific and platform-specific implementations.
    _get_platform_method: Utility to retrieve the most specific method implementation
        for a given platform and/or execution context, following the priority chain:
        composite ({method}_{context}_{platform}) > context ({method}_{context}) >
        platform ({method}_{platform}) > generic ({method}).

Example:
    >>> from ejkernel.ops.core import Kernel, Invocation, Cfg, Out
    >>>
    >>> class MyKernel(Kernel[dict, jax.Array]):
    ...     def run(self, x, cfg: dict) -> jax.Array:
    ...         return x * cfg['scale']
    ...
    ...     def heuristic_cfg(self, inv: Invocation) -> dict:
    ...         return {'scale': 1.0}
"""

from .kernel import Invocation, Kernel, _get_platform_method, _has_custom_vjp
from .types import Cfg, Out

__all__ = ("Cfg", "Invocation", "Kernel", "Out", "_get_platform_method", "_has_custom_vjp")
