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

"""Compilation utilities for JAX function optimization.

Thin EasyDeL-facing layer over the hardened ejkernel ``callib`` implementation:
the enhanced ``ejit`` decorator, the compiled-function save/load helpers, the
hashing utilities, and the cache-control constants all live in
``ejkernel.callib`` (the single source of truth) and are re-exported here for
backward-compatible ``easydel.utils.compiling_utils`` imports.

Functions:
    ejit: Enhanced JIT with persistent caching
    save_compiled_fn: Save compiled function to disk
    load_compiled_fn: Load compiled function from disk
    load_cached_functions: Load multiple cached functions
    hash_fn / get_safe_hash_int / get_hash_of_lowering: hashing helpers

Constants:
    RECOMPILE_FORCE: Force recompilation flag
    ECACHE_COMPILES: Enable compilation caching
    CACHE_DIR: EasyDeL cache directory path
    COMPILE_FUNC_DIR: Environment-namespaced compiled-functions directory
    COMPILED_FILE_NAME / COMPILED_CACHE: on-disk name + in-memory cache

Example:
    >>> from easydel.utils.compiling_utils import ejit
    >>>
    >>> @ejit
    ... def optimized_fn(x, y):
    ...     return x @ y + x.T @ y.T
    >>>
    >>> # First call compiles and caches
    >>> result = optimized_fn(a, b)
    >>> # Next run loads from cache
    >>> result = optimized_fn(a, b)
"""

from __future__ import annotations

from ejkernel.callib import (  # pyright: ignore[reportMissingTypeStubs]
    COMPILED_CACHE,
    COMPILED_FILE_NAME,
    ECACHE_COMPILES,
    RECOMPILE_FORCE,
    ejit,
    get_effective_compile_dir,
    get_hash_of_lowering,
    get_safe_hash_int,
    hash_fn,
    load_cached_functions,
    load_compiled_fn,
    save_compiled_fn,
)

from .helpers import get_cache_dir

ejit = ejit

__all__ = [
    "COMPILED_CACHE",
    "COMPILED_FILE_NAME",
    "ECACHE_COMPILES",
    "RECOMPILE_FORCE",
    "ejit",
    "get_hash_of_lowering",
    "get_safe_hash_int",
    "hash_fn",
    "load_cached_functions",
    "load_compiled_fn",
    "save_compiled_fn",
]

CACHE_DIR = get_cache_dir()

# The single source of truth for compiled-function persistence is
# ejkernel.callib._ejit: an environment-fingerprinted cache directory
# (jax/jaxlib/libtpu versions + host CPU identity) with entry-size guards and
# atomic writes, so stale or oversized entries can never crash a reader.
COMPILE_FUNC_DIR = get_effective_compile_dir()
