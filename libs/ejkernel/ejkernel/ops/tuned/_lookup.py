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

"""The call kernels make to ask "has anyone measured this shape?".

One function, :func:`tuned_choice`, so a kernel never grows its own schema:

    choice = tuned_choice("grouped_matmul", dtypes={"lhs": lhs.dtype, "rhs": rhs.dtype},
                          shape={"m": m, "k": k, "n": n, "e": num_groups})
    if choice is not None:
        platform, cfg = choice.platform, choice.config

``None`` means "not measured" and the caller keeps whatever it does today, so
adding the table is strictly additive: an unswept kernel behaves exactly as
before, and a swept one stops guessing.
"""

from __future__ import annotations

import os
import typing as tp
from functools import lru_cache
from pathlib import Path

from ._store import TunedEntry, TunedStore, dtype_signature, shape_signature

__all__ = ["current_device_kind", "set_tuned_store", "tuned_choice", "tuned_store"]

_override: TunedStore | None = None


@lru_cache(maxsize=1)
def _shipped_store() -> TunedStore:
    return TunedStore()


def tuned_store() -> TunedStore:
    """The active store: an explicit override, else the shipped table.

    ``EJKERNEL_TUNED_DB`` re-points it, which exists so a sweep can be validated
    against a real workload before it is committed. It follows the established
    ``EJKERNEL_PERSISTENT_CACHE_DIR`` convention rather than inventing a new one.
    """
    if _override is not None:
        return _override
    env = os.environ.get("EJKERNEL_TUNED_DB")
    if env:
        return TunedStore(Path(env))
    return _shipped_store()


def set_tuned_store(store: TunedStore | str | Path | None) -> None:
    """Point lookups at a specific database, or ``None`` to restore the default."""
    global _override
    if store is None or isinstance(store, TunedStore):
        _override = store
    else:
        _override = TunedStore(store)


@lru_cache(maxsize=1)
def current_device_kind() -> str:
    """JAX's device kind for the default device, e.g. ``"TPU v5p"``.

    Cached: it cannot change within a process, and the lookup sits on paths that
    run per kernel invocation at trace time.
    """
    try:
        import jax

        return str(getattr(jax.devices()[0], "device_kind", "unknown"))
    except Exception:
        return "unknown"


def tuned_choice(
    kernel: str,
    *,
    dtypes: tp.Mapping[str, tp.Any] | str | None = None,
    shape: tp.Mapping[str, int] | str | None = None,
    device: str | None = None,
    allow_nearest: bool = True,
) -> TunedEntry | None:
    """Look up the measured winner for this call, or ``None`` if untabulated.

    Args:
        kernel: Kernel identifier, matching what sweeps recorded.
        dtypes: Named dtypes that affect the choice, or a prebuilt signature.
        shape: Named dimensions that affect the choice, or a prebuilt key.
            Sizes are bucketed to powers of two.
        device: Device kind; defaults to the current device.
        allow_nearest: Fall back to the nearest measured shape.

    Returns:
        The tuned entry, carrying both ``platform`` and ``config``.
    """
    store = tuned_store()
    if not store.available():
        return None
    dt = dtypes if isinstance(dtypes, str) else dtype_signature(**dict(dtypes or {}))
    sh = shape if isinstance(shape, str) else shape_signature(**dict(shape or {}))
    return store.lookup(
        kernel,
        device or current_device_kind(),
        dt,
        sh,
        allow_nearest=allow_nearest,
    )
