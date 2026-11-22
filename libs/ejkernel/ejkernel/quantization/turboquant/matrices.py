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

"""Random matrix generation for TurboQuant.

Provides deterministic generation of:
- Haar-distributed random orthogonal rotation matrices (via QR of Gaussian)
- Random Gaussian projection matrices for QJL
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def generate_rotation_matrix(
    seed: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Generate a Haar-distributed random orthogonal matrix.

    Constructs a uniformly random orthogonal matrix by applying QR
    decomposition to a matrix of i.i.d. Gaussian entries, with sign
    correction on the diagonal of R for uniqueness.

    Args:
        seed: Random seed for deterministic generation.
        head_dim: Dimension of the rotation matrix (head_dim x head_dim).
        dtype: Output dtype (default float32).

    Returns:
        Orthogonal matrix Pi of shape [head_dim, head_dim].
    """
    key = jax.random.PRNGKey(seed)
    G = jax.random.normal(key, (head_dim, head_dim), dtype=jnp.float32)
    Q, R = jnp.linalg.qr(G)
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    Pi = Q * signs[None, :]
    return Pi.astype(dtype)


def generate_projection_matrix(
    seed: int,
    projection_dim: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Generate a random Gaussian projection matrix for QJL.

    Each entry is drawn i.i.d. from N(0, 1/projection_dim) so that
    the projection approximately preserves inner products.

    Args:
        seed: Random seed for deterministic generation.
        projection_dim: Number of projection dimensions (qjl_dim).
        head_dim: Input dimension (head_dim).
        dtype: Output dtype (default float32).

    Returns:
        Projection matrix S of shape [projection_dim, head_dim].
    """
    key = jax.random.PRNGKey(seed)
    S = jax.random.normal(key, (projection_dim, head_dim), dtype=jnp.float32)
    S = S / jnp.sqrt(jnp.float32(projection_dim))
    return S.astype(dtype)
