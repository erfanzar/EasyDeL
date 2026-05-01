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

"""TurboQuant KV cache quantization configuration and constants.

TurboQuant (ICLR 2026) compresses KV caches using two-stage vector
quantization: random rotation + Lloyd-Max scalar quantization, with
QJL residual correction for unbiased key attention scores.

This module provides:
- TurboQuantConfig: Configuration dataclass for TurboQuant parameters
- TurboQuantConstants: Precomputed per-layer constants (rotation, codebook, etc.)
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from easydel.infra.sharding import MeshLike


@functools.lru_cache(maxsize=32)
def _solve_lloyd_max_cached(bits: int, dim: int):
    """Cached Lloyd-Max solver — codebooks depend only on (bits, dim)."""
    from ejkernel.quantization.turboquant.codebook import solve_lloyd_max

    return solve_lloyd_max(bits=bits, dim=dim)


@dataclasses.dataclass(frozen=True)
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization.

    Controls the bit budget, QJL projection dimension, and random seeds
    for generating the rotation and projection matrices.

    At b total bits per coordinate:
    - Keys use (b-1) bits for Lloyd-Max codebook + 1 bit for QJL sign
    - Values use b bits for Lloyd-Max codebook (no QJL needed)

    Compression ratios (vs bf16, head_dim=128):
    - 4-bit: ~3.8x
    - 3-bit: ~5.0x
    - 2-bit: ~7.3x

    Attributes:
        bits: Total bits per coordinate (2-8). Default 4.
        qjl_dim: QJL projection dimension for keys. If None, defaults
            to head_dim. Larger values reduce inner product variance.
        seed: Base seed for deterministic generation of rotation matrix
            Pi and QJL projection matrix S. Different layers use
            seed + layer_index for independence.
    """

    bits: int = 4
    qjl_dim: int | None = None
    seed: int = 42

    def __post_init__(self):
        if self.bits < 2 or self.bits > 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")

    @property
    def key_codebook_bits(self) -> int:
        """Bits used for key codebook indices (bits - 1 for QJL sign)."""
        return self.bits - 1

    @property
    def value_codebook_bits(self) -> int:
        """Bits used for value codebook indices (all bits, no QJL)."""
        return self.bits

    @property
    def key_codebook_size(self) -> int:
        """Number of Lloyd-Max centroids for keys."""
        return 1 << self.key_codebook_bits

    @property
    def value_codebook_size(self) -> int:
        """Number of Lloyd-Max centroids for values."""
        return 1 << self.value_codebook_bits


@dataclasses.dataclass(frozen=True)
class TurboQuantConstants:
    """Precomputed constants for TurboQuant, generated once at cache init.

    These are deterministically generated from the config and head_dim.
    All layers share the same constants (same head_dim), but different
    seeds can be used per layer for independence.

    Attributes:
        key_codebook: Lloyd-Max centroids for keys, shape [2^(bits-1)].
        value_codebook: Lloyd-Max centroids for values, shape [2^bits].
        rotation_matrix: Orthogonal rotation Pi, shape [head_dim, head_dim].
        qjl_projection: QJL projection S, shape [qjl_dim, head_dim].
        qjl_dim: Effective QJL projection dimension.
        bits: Total bit budget per coordinate.
    """

    key_codebook: jax.Array
    value_codebook: jax.Array
    rotation_matrix: jax.Array
    qjl_projection: jax.Array
    qjl_dim: int
    bits: int

    def replicate(self, mesh: "MeshLike") -> "TurboQuantConstants":
        """Return a copy with all arrays replicated across the mesh.

        Shard-map kernels pass these constants with ``PartitionSpec()``
        (fully replicated).  If the arrays sit on a single device the
        shard-map will fail, so this helper broadcasts them first.
        """
        from easydel.infra.sharding import replicated_named_sharding

        replicated = replicated_named_sharding(mesh)
        return TurboQuantConstants(
            key_codebook=jax.device_put(self.key_codebook, replicated),
            value_codebook=jax.device_put(self.value_codebook, replicated),
            rotation_matrix=jax.device_put(self.rotation_matrix, replicated),
            qjl_projection=jax.device_put(self.qjl_projection, replicated),
            qjl_dim=self.qjl_dim,
            bits=self.bits,
        )

    @staticmethod
    def generate(
        config: TurboQuantConfig,
        head_dim: int,
        layer_index: int = 0,
        mesh: "MeshLike | None" = None,
    ) -> TurboQuantConstants:
        """Generate all precomputed constants from config.

        Args:
            config: TurboQuant configuration.
            head_dim: Attention head dimension.
            layer_index: Layer index for seed derivation (0 = shared).
            mesh: If provided, replicate all arrays across this mesh so
                they are compatible with shard-map ``PartitionSpec()`` specs.

        Returns:
            TurboQuantConstants with all arrays populated.
        """
        from ejkernel.quantization.turboquant.matrices import (
            generate_projection_matrix,
            generate_rotation_matrix,
        )

        qjl_dim = config.qjl_dim if config.qjl_dim is not None else head_dim

        # Codebooks are shared across layers (same distribution) — use cached solver
        key_cb = _solve_lloyd_max_cached(bits=config.key_codebook_bits, dim=head_dim)
        value_cb = _solve_lloyd_max_cached(bits=config.value_codebook_bits, dim=head_dim)

        # Rotation and projection matrices use per-layer seeds
        rotation_seed = config.seed + layer_index * 1000
        projection_seed = config.seed + layer_index * 1000 + 500

        Pi = generate_rotation_matrix(seed=rotation_seed, head_dim=head_dim)
        S = generate_projection_matrix(seed=projection_seed, projection_dim=qjl_dim, head_dim=head_dim)

        constants = TurboQuantConstants(
            key_codebook=jnp.array(key_cb.centroids),
            value_codebook=jnp.array(value_cb.centroids),
            rotation_matrix=Pi,
            qjl_projection=S,
            qjl_dim=qjl_dim,
            bits=config.bits,
        )
        if mesh is not None:
            constants = constants.replicate(mesh)
        return constants
