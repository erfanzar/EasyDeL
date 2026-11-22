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

"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After a random rotation by a Haar-distributed orthogonal matrix, each
coordinate of a d-dimensional unit vector follows a distribution well
approximated by N(0, 1/d). The Lloyd-Max quantizer finds the optimal
scalar quantization centroids for this distribution by iteratively
solving the continuous 1-D k-means conditions.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as gamma_fn


@dataclasses.dataclass(frozen=True)
class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given (bits, dim) pair.

    Attributes:
        bits: Number of quantization bits.
        dim: Dimensionality of the original vectors (used for variance = 1/dim).
        n_levels: Number of quantization levels (2^bits).
        centroids: Array of centroid values, shape [n_levels].
        boundaries: Array of decision boundaries, shape [n_levels - 1].
    """

    bits: int
    dim: int
    n_levels: int
    centroids: np.ndarray
    boundaries: np.ndarray


def solve_lloyd_max(
    bits: int,
    dim: int = 128,
    max_iters: int = 200,
    tol: float = 1e-10,
    use_gaussian_approx: bool = True,
) -> LloydMaxCodebook:
    """Solve for optimal Lloyd-Max centroids for N(0, 1/dim) distribution.

    Uses iterative optimization: alternate between computing decision
    boundaries (midpoints between adjacent centroids) and updating
    centroids as conditional expectations ``E[X | X in partition_i]``.
    Centroids are initialized uniformly in ``[-3σ, 3σ]`` where ``σ = 1/sqrt(dim)``.

    Requires ``scipy`` at call time (imported inside the function).

    Args:
        bits: Number of quantization bits, typically 3 or 4 for TurboQuant.
            Determines ``n_levels = 2^bits``.
        dim: Vector dimensionality; sets the distribution variance to ``1/dim``.
            Defaults to 128 (common attention head dimension).
        max_iters: Maximum Lloyd-Max iterations before stopping even without
            convergence.  200 is sufficient for most practical settings.
        tol: Convergence threshold on the max absolute centroid change between
            iterations.
        use_gaussian_approx: If ``True`` (or if ``dim >= 64``), the coordinate
            distribution is modelled as ``N(0, 1/dim)`` using ``scipy.stats.norm``.
            If ``False`` and ``dim < 64``, uses the exact marginal PDF of a uniform
            unit-sphere projection (Beta-type), computed via ``scipy.special.gamma``.
            Note: the ``dim >= 64`` condition takes precedence — even when this
            argument is ``False``, the Gaussian path is used whenever ``dim >= 64``.

    Returns:
        ``LloydMaxCodebook`` with ``centroids`` and ``boundaries`` stored as
        float32 numpy arrays of shapes ``[n_levels]`` and ``[n_levels - 1]``.
    """
    from scipy import integrate
    from scipy.stats import norm

    n_levels = 1 << bits
    sigma = 1.0 / np.sqrt(dim)

    if use_gaussian_approx or dim >= 64:

        def pdf(x):
            return norm.pdf(x, loc=0, scale=sigma)

        def x_pdf(x):
            return x * norm.pdf(x, loc=0, scale=sigma)

    else:
        coeff = gamma_fn(dim / 2.0) / (np.sqrt(np.pi) * gamma_fn((dim - 1) / 2.0))

        def pdf(x):
            x2 = np.clip(x * x, 0, 1.0 - 1e-15)
            return coeff * (1.0 - x2) ** ((dim - 3) / 2.0)

        def x_pdf(x):
            return x * pdf(x)

    lo, hi = -3.0 * sigma, 3.0 * sigma
    centroids = np.linspace(lo, hi, n_levels)

    for _ in range(max_iters):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(x_pdf, a, b, limit=100)
            den, _ = integrate.quad(pdf, a, b, limit=100)
            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return LloydMaxCodebook(
        bits=bits,
        dim=dim,
        n_levels=n_levels,
        centroids=centroids.astype(np.float32),
        boundaries=boundaries.astype(np.float32),
    )


@functools.lru_cache(maxsize=32)
def get_codebook(bits: int, dim: int = 128) -> LloydMaxCodebook:
    """Get or compute a cached Lloyd-Max codebook.

    Results are memoized via ``functools.lru_cache`` (up to 32 distinct
    ``(bits, dim)`` pairs).  Calls :func:`solve_lloyd_max` with default
    ``use_gaussian_approx=True``, which uses the Gaussian approximation
    for all ``dim`` values.  Requires ``scipy`` on first call for any
    given ``(bits, dim)`` pair.

    Args:
        bits: Number of quantization bits, e.g. 3 or 4.
        dim: Vector dimensionality (sets the quantizer distribution variance
            to ``1/dim``).  Defaults to 128.

    Returns:
        Cached :class:`LloydMaxCodebook` instance.
    """
    return solve_lloyd_max(bits=bits, dim=dim)


def quantize_to_indices(
    x: jax.Array,
    centroids: jax.Array,
) -> jax.Array:
    """Quantize each element of x to the index of the nearest centroid.

    Computes the L1 distance from each element to every centroid by broadcasting
    ``x[..., None] - centroids`` into a ``[..., n_levels]`` array and taking
    ``argmin`` along the last axis.  Memory cost is O(numel(x) * n_levels).

    Args:
        x: Input float array of arbitrary shape.
        centroids: 1-D float array of centroid values, shape ``[n_levels]``.
            Must have ``n_levels <= 256`` so that indices fit in uint8.

    Returns:
        uint8 array of the same shape as ``x``, with values in ``[0, n_levels)``.
    """
    diffs = jnp.abs(x[..., None] - centroids)
    return jnp.argmin(diffs, axis=-1).astype(jnp.uint8)


def dequantize_from_indices(
    indices: jax.Array,
    centroids: jax.Array,
) -> jax.Array:
    """Map centroid indices back to their float values.

    Uses a one-hot matrix multiply rather than a gather/index operation.
    This avoids Mosaic's restriction to 2-D indexing on TPU Pallas backends
    and is practical for small codebooks (typically 8–16 entries at 3–4 bits).

    Args:
        indices: Integer array of centroid indices, arbitrary shape.
            Values must be in ``[0, n_levels)``.
        centroids: 1-D float array of centroid values, shape ``[n_levels]``.

    Returns:
        float32 array of the same shape as ``indices``, where each element is
        replaced by its corresponding centroid value.
    """
    return jnp.sum(
        jax.nn.one_hot(indices, centroids.shape[0], dtype=jnp.float32) * centroids,
        axis=-1,
    )
