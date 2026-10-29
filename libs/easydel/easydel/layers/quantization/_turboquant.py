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
    """Memoized Lloyd-Max codebook solver keyed by ``(bits, dim)``.

    The Lloyd-Max algorithm produces optimal scalar-quantization centroids
    for a given source distribution and bit budget. For TurboQuant we apply
    Lloyd-Max to the rotated KV coordinates, whose distribution is
    fully determined by ``(bits, head_dim)`` — so the same codebook is
    reused for every layer with the same head dim, and ``functools.lru_cache``
    avoids the (relatively expensive) iterative solve on every layer init.

    Args:
        bits: Bit budget for the codebook (``2**bits`` centroids).
        dim: Head dimension — controls the variance of the rotated
            distribution that Lloyd-Max optimizes against.

    Returns:
        The :class:`ejkernel.quantization.turboquant.codebook.LloydMaxResult`
        with ``.centroids`` of length ``2**bits``.
    """
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
        """Validate the bit budget after dataclass initialization.

        Raises:
            ValueError: If ``bits`` is outside the supported ``[2, 8]`` range.
        """
        if self.bits < 2 or self.bits > 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")

    @property
    def key_codebook_bits(self) -> int:
        """Bit budget for the *key* Lloyd-Max codebook.

        TurboQuant spends one of the ``bits`` per key coordinate on the
        QJL sign (used to recover an unbiased estimator for ``q · k``);
        the remaining ``bits - 1`` bits index the Lloyd-Max codebook for
        the rotated key coordinates.

        Returns:
            ``self.bits - 1`` — the index width into ``key_codebook``.
        """
        return self.bits - 1

    @property
    def value_codebook_bits(self) -> int:
        """Bit budget for the *value* Lloyd-Max codebook.

        Values do not require the QJL sign trick (they participate in
        attention only as the right operand of the softmax×V matmul, so
        sign-preservation isn't needed for unbiasedness), and the full
        ``self.bits`` indexes the Lloyd-Max codebook for rotated values.

        Returns:
            ``self.bits`` — the index width into ``value_codebook``.
        """
        return self.bits

    @property
    def key_codebook_size(self) -> int:
        """Number of Lloyd-Max centroids in the *key* codebook.

        Returns:
            ``2 ** key_codebook_bits``.
        """
        return 1 << self.key_codebook_bits

    @property
    def value_codebook_size(self) -> int:
        """Number of Lloyd-Max centroids in the *value* codebook.

        Returns:
            ``2 ** value_codebook_bits``.
        """
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
        """Broadcast every constant array across ``mesh`` (fully replicated).

        The TurboQuant shard-map kernels declare these constants with
        ``PartitionSpec()`` (i.e. fully replicated on every device). When
        the constants are originally produced on a single device — which is
        the common case during model load — the shard-map dispatcher will
        refuse them because their physical placement does not match the
        declared partitioning. This helper performs the necessary
        ``jax.device_put`` against the mesh's replicated NamedSharding for
        each of the four arrays (key/value codebooks, rotation, projection)
        while leaving the integer fields untouched.

        Args:
            mesh: Target mesh — typically the same one the attention
                kernels run under.

        Returns:
            A new :class:`TurboQuantConstants` whose array fields all live
            on every device in ``mesh``.
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
        """Materialize a full :class:`TurboQuantConstants` set for one layer.

        Steps performed:

        1. Resolve ``qjl_dim`` (defaults to ``head_dim`` when the config
           leaves it as ``None``).
        2. Solve Lloyd-Max for keys (bits = ``config.key_codebook_bits``)
           and values (bits = ``config.value_codebook_bits``) via the
           shared :func:`_solve_lloyd_max_cached` — codebooks are layer
           independent so the solve runs at most once per ``(bits, head_dim)``.
        3. Build the orthogonal random rotation ``Pi`` and the QJL random
           projection ``S`` from per-layer seeds derived as
           ``seed + layer_index * 1000`` and ``seed + layer_index * 1000 + 500``
           respectively, ensuring each layer gets independent matrices when
           ``layer_index`` varies.
        4. Optionally replicate the resulting arrays across ``mesh`` so the
           constants can be passed straight into the shard-map kernels.

        Args:
            config: TurboQuant configuration (bit budget, qjl_dim, base seed).
            head_dim: Attention head dimension; sets the rotation matrix
                size and the source distribution variance fed to Lloyd-Max.
            layer_index: Index of the transformer layer the constants are
                being generated for. Pass ``0`` to share matrices across
                layers, or distinct indices to give each layer its own
                rotation/projection.
            mesh: Optional target mesh; when provided, every array field
                is broadcast via :meth:`replicate` so the shard-map
                kernels' ``PartitionSpec()`` declarations are satisfied.

        Returns:
            A frozen :class:`TurboQuantConstants` with all four array
            fields populated and ready for cache initialization.
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
