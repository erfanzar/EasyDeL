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


"""Type definitions for the ejkernel operations system.

This module defines the core type variables used throughout the operations
framework to provide type safety and generic programming support.

Type Variables:
    Cfg: Configuration type for kernel operations
        Represents the configuration parameter type for kernels.
        Can be any type (dict, dataclass, NamedTuple, etc.)

    Out: Output type for kernel operations
        Represents the return type of kernel operations.
        Can be any type (JAX arrays, tuples, complex structures)

These type variables enable:
    - Type-safe kernel implementations
    - Generic configuration handling
    - Clear documentation of input/output types
    - IDE support for type checking and autocompletion

Example Usage:
    >>> class MyKernel(Kernel[dict, jax.Array]):
    ...     def run(self, x: jax.Array, cfg: dict) -> jax.Array:
    ...         return x * cfg['multiplier']
    >>>
    >>>
    >>> kernel = MyKernel()
"""

from __future__ import annotations

from typing import TypeVar

Cfg = TypeVar("Cfg")
Out = TypeVar("Out")
