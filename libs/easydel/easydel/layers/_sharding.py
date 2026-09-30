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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer sharding compatibility exports.

Generic sharding resolution lives in ``easydel.infra.sharding``. This module
keeps the historical layer import path without maintaining a second
implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from easydel.infra.sharding import pick_mesh, resolve_safe_sharding


def all_gather_then_dot(x: Array, kernel: Array, axis_name: str, *, precision: jax.lax.PrecisionLike) -> Array:
    """All-gather the trailing feature axis, then dot against a local column shard.

    Intended for use inside ``jax.shard_map`` when ``x`` is feature-sharded
    and ``kernel`` is sharded on its output axis.
    """
    gathered = jax.lax.all_gather(x, axis_name, axis=x.ndim - 1, tiled=True)
    return jnp.einsum("...i,io->...o", gathered, kernel, precision=precision)


__all__ = ["all_gather_then_dot", "pick_mesh", "resolve_safe_sharding"]
