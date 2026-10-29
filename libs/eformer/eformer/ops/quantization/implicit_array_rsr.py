# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""
Randomized Sparse Representation (RSR) Operators for Binary and Ternary Matrices.

This module provides efficient implicit array representations for binary {0, 1} and
ternary {-1, 0, 1} matrices using the one-hot encoding RSR method. This enables
fast vector-matrix multiplication without materializing the full dense matrix.

Key Classes:
    - RSROperatorBinary: Implicit representation for binary matrices
    - RSROperatorTernary: Implicit representation for ternary matrices

The RSR method works by:
    1. Grouping matrix columns into blocks of size k
    2. Converting each block row to an integer index (0 to 2^k - 1)
    3. Creating one-hot encoded lookup tables for each block
    4. Using matrix multiplication with lookup tables for efficient computation

This approach reduces memory and computation for sparse binary/ternary matrices
commonly found in quantized neural networks.

Example:
    >>> import jax.numpy as jnp
    >>> from eformer.ops.quantization import RSROperatorBinary, RSROperatorTernary
    >>>
    >>> # Binary matrix
    >>> A = jnp.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=jnp.int32)
    >>> rsr = RSROperatorBinary.from_matrix(A, k=4)
    >>> v = jnp.array([1.0, 2.0])
    >>> result = rsr.dot(v)  # Efficient v @ A
    >>>
    >>> # Ternary matrix
    >>> B = jnp.array([[-1, 0, 1], [1, -1, 0]], dtype=jnp.int32)
    >>> rsr_ternary = RSROperatorTernary.from_matrix(B, k=4)
"""

import functools
from dataclasses import dataclass

import jax
from jax import lax
from jax import numpy as jnp

from eformer.jaximus import ImplicitArray, aux_field

from .quantization_functions import pack_weights_1bit

Array = jax.Array


@functools.partial(jax.jit, static_argnames=["k"])
def _generate_binary_matrix(k: int) -> Array:
    """
    Generate a binary matrix mapping integers 0 to 2^k-1 to their bit representations.

    Creates a lookup table where row i contains the k-bit binary representation
    of integer i, with the most significant bit first.

    Args:
        k (int): Number of bits (columns in output matrix).

    Returns:
        jax.Array: Integer array of shape (2**k, k) where each row i contains
            the binary digits of i.

    Example:
        >>> _generate_binary_matrix(3)
        Array([[0, 0, 0],   # 0
               [0, 0, 1],   # 1
               [0, 1, 0],   # 2
               [0, 1, 1],   # 3
               [1, 0, 0],   # 4
               [1, 0, 1],   # 5
               [1, 1, 0],   # 6
               [1, 1, 1]], dtype=int32)  # 7
    """
    indices = jnp.arange(2**k)
    exponents = jnp.arange(k - 1, -1, -1)
    return ((indices[:, None] >> exponents) & 1).astype(jnp.int32)


def _preprocess_matrix_jax(A: Array, k: int) -> tuple[Array, int]:
    """
    Preprocess a binary matrix into one-hot encoded block representations.

    This function converts a binary matrix into a form suitable for efficient
    vector-matrix multiplication using the RSR method. Each block of k columns
    is converted to a one-hot encoding based on the integer value of the binary
    pattern in that block.

    Args:
        A (jax.Array): Binary matrix with values in {0, 1}.
            Shape: (n, m) where n is rows and m is columns.
        k (int): Block size for grouping columns. The matrix is processed
            in blocks of k columns.

    Returns:
        tuple[jax.Array, int]: A tuple containing:
            - one_hot_maps (jax.Array): Stacked one-hot encoded representations.
              Shape: (num_blocks, n, 2**k).
            - padding (int): Number of zero columns added to make m divisible by k.

    Note:
        This preprocessing should be done once (typically on CPU) as it involves
        Python loops. The resulting one_hot_maps are used for efficient JIT-compiled
        dot products.
    """
    n, m = A.shape
    padding = (k - (m % k)) % k

    if padding > 0:
        A_padded = jnp.pad(A, ((0, 0), (0, padding)), mode="constant")
    else:
        A_padded = A

    num_blocks = A_padded.shape[1] // k
    powers_of_2 = 2 ** jnp.arange(k - 1, -1, -1, dtype=jnp.int32)

    one_hot_maps = []
    for i in range(num_blocks):
        block = lax.dynamic_slice_in_dim(A_padded, i * k, k, axis=1)
        integer_values = block @ powers_of_2
        L = jnp.zeros((n, 2**k), dtype=A.dtype)
        L = L.at[jnp.arange(n), integer_values].set(1)
        one_hot_maps.append(L)
    return jnp.stack(one_hot_maps), padding


@functools.partial(jax.jit, static_argnames=["k", "m"])
def _rsr_v_dot_a_binary(v: Array, one_hot_maps: Array, padding: int, k: int, m: int) -> Array:
    """
    Compute vector-matrix product v @ A using the one-hot RSR method.

    This is the JIT-compiled core computation for RSR-based matrix multiplication.
    It uses the preprocessed one-hot maps to efficiently compute the product
    without materializing the full matrix.

    Args:
        v (jax.Array): Input vector of shape (n,).
        one_hot_maps (jax.Array): Preprocessed one-hot encoded blocks.
            Shape: (num_blocks, n, 2**k).
        padding (int): Number of padded columns (unused in computation but
            needed for proper slicing).
        k (int): Block size used during preprocessing.
        m (int): Original number of columns in matrix A.

    Returns:
        jax.Array: Result vector of shape (m,) representing v @ A.

    Note:
        This function is optimized for JIT compilation and should not be
        called directly. Use RSROperatorBinary.dot() instead.
    """
    segmented_sums = jnp.einsum("n,bnz->bz", v, one_hot_maps, optimize="optimal")
    binary_matrix = _generate_binary_matrix(k).astype(v.dtype)
    block_results = segmented_sums @ binary_matrix
    results_flat = block_results.flatten()
    return lax.slice(results_flat, (0,), (m,))


@dataclass
class RSROperatorBinary(ImplicitArray):
    """
    Implicit Array for binary matrices using the one-hot RSR method.

    This class provides an efficient implicit representation of binary matrices
    (containing only 0s and 1s) using Randomized Sparse Representation (RSR).
    The matrix is never stored in dense form; instead, it uses one-hot encoded
    lookup tables for efficient vector-matrix multiplication.

    Attributes:
        one_hot_maps (jax.Array): Preprocessed one-hot encoded block representations.
            Shape: (num_blocks, n, 2**k).
        k (int): Block size for column grouping. Larger k uses more memory but
            may be faster for some matrices.
        padding (int): Number of zero columns added during preprocessing.
        org_dtype (jnp.dtype): Original dtype of the source matrix.

    Example:
        >>> import jax.numpy as jnp
        >>> A = jnp.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=jnp.int32)
        >>> rsr = RSROperatorBinary.from_matrix(A, k=4)
        >>> v = jnp.array([1.0, 2.0])
        >>> result = rsr.dot(v)  # Computes v @ A efficiently
    """

    one_hot_maps: Array
    k: int = aux_field()
    padding: int = aux_field()
    org_dtype: jnp.dtype = aux_field()  # noqa

    @classmethod
    def quantize(cls, A: Array, k: int = 8):
        """
        Create an RSROperatorBinary from an array by packing to 1-bit first.

        Args:
            A (jax.Array): Input array to quantize.
            k (int): Block size for RSR encoding. Defaults to 8.

        Returns:
            RSROperatorBinary: The RSR-encoded binary matrix.
        """
        return cls.from_matrix(pack_weights_1bit(A), k, A.dtype)

    @classmethod
    def from_matrix(cls, A: Array, k: int = 8, org_dtype: jnp.dtype = jnp.float32):
        """
        Create an RSROperatorBinary from a binary integer matrix.

        Args:
            A (jax.Array): Binary matrix with values in {0, 1}.
                Must have dtype int32 or int8.
            k (int): Block size for column grouping. Defaults to 8.
                The matrix columns are processed in groups of k.
            org_dtype (jnp.dtype): Original dtype to preserve for materialization.
                Defaults to float32.

        Returns:
            RSROperatorBinary: The RSR-encoded representation.

        Raises:
            TypeError: If input matrix is not integer type.
        """
        if A.dtype not in (jnp.int32, jnp.int8):
            raise TypeError("Input matrix must be integer type.")

        one_hot_maps, padding = _preprocess_matrix_jax(A.astype(jnp.int32), k)

        return cls(one_hot_maps=one_hot_maps, k=k, padding=padding, shape=A.shape, dtype=A.dtype, org_dtype=org_dtype)

    def dot(self, v: Array) -> Array:
        """
        Compute vector-matrix product v @ A without materializing A.

        Args:
            v (jax.Array): Input vector of shape (n,) where n is the number
                of rows in the original matrix.

        Returns:
            jax.Array: Result vector of shape (m,) where m is the number
                of columns in the original matrix.

        Raises:
            ValueError: If input vector shape doesn't match matrix dimensions.
        """
        n, m = self.shape
        if v.shape != (n,):
            raise ValueError(f"Input vector shape mismatch. Expected ({n},), got {v.shape}")

        return _rsr_v_dot_a_binary(v.astype(self.one_hot_maps.dtype), self.one_hot_maps, self.padding, self.k, m)

    def materialize(self) -> Array:
        """
        Reconstruct the original dense binary matrix.

        This method reverses the RSR preprocessing to produce the original
        dense matrix. Useful for debugging, verification, or when the full
        matrix is needed for operations not supported by RSR.

        Returns:
            jax.Array: Dense binary matrix of shape (n, m) with the stored dtype.
        """
        n, m = self.shape
        num_blocks = self.one_hot_maps.shape[0]
        binary_matrix_B = _generate_binary_matrix(self.k).astype(self.dtype)
        reconstructed_blocks = jnp.einsum("bnz,zk->bnk", self.one_hot_maps, binary_matrix_B, optimize="optimal")
        reconstructed_padded = reconstructed_blocks.transpose((1, 0, 2))
        reconstructed_padded = reconstructed_padded.reshape((n, num_blocks * self.k))
        E = lax.slice(reconstructed_padded, (0, 0), (n, m))
        return E


@dataclass
class RSROperatorTernary(ImplicitArray):
    """
    Implicit Array for ternary matrices using decomposed binary RSR operators.

    This class provides an efficient implicit representation of ternary matrices
    (containing values {-1, 0, 1}) by decomposing them into two binary matrices:
    one for positive values (+1) and one for negative values (-1). The ternary
    dot product is computed as: v @ A = v @ B1 - v @ B2 where B1 marks +1 positions
    and B2 marks -1 positions.

    Attributes:
        rsr_b1 (RSROperatorBinary): RSR representation of the positive mask (A == 1).
        rsr_b2 (RSROperatorBinary): RSR representation of the negative mask (A == -1).

    Example:
        >>> import jax.numpy as jnp
        >>> A = jnp.array([[-1, 0, 1], [1, -1, 0]], dtype=jnp.int32)
        >>> rsr = RSROperatorTernary.from_matrix(A, k=4)
        >>> v = jnp.array([1.0, 2.0])
        >>> result = rsr.dot(v)  # Computes v @ A efficiently

    Note:
        Memory usage is approximately 2x that of RSROperatorBinary since
        two binary RSR operators are stored internally.
    """

    rsr_b1: RSROperatorBinary
    rsr_b2: RSROperatorBinary

    @classmethod
    def from_matrix(cls, A: Array, k: int = 8):
        """
        Create an RSROperatorTernary from a ternary integer matrix.

        Args:
            A (jax.Array): Ternary matrix with values in {-1, 0, 1}.
                Must have dtype int32 or int8.
            k (int): Block size for column grouping in the underlying binary
                RSR operators. Defaults to 8.

        Returns:
            RSROperatorTernary: The decomposed RSR representation.

        Raises:
            TypeError: If input matrix is not integer type.
        """
        if A.dtype not in (jnp.int32, jnp.int8):
            raise TypeError("Input matrix must be integer type.")

        B1 = (A == 1).astype(jnp.int32)
        B2 = (A == -1).astype(jnp.int32)

        rsr_b1 = RSROperatorBinary.from_matrix(B1, k=k)
        rsr_b2 = RSROperatorBinary.from_matrix(B2, k=k)

        return cls(rsr_b1=rsr_b1, rsr_b2=rsr_b2, shape=A.shape, dtype=A.dtype)

    def dot(self, v: Array) -> Array:
        """
        Compute vector-matrix product v @ A without materializing A.

        Computes the ternary dot product as the difference of two binary
        dot products: v @ B1 - v @ B2.

        Args:
            v (jax.Array): Input vector of shape (n,) where n is the number
                of rows in the original matrix.

        Returns:
            jax.Array: Result vector of shape (m,) where m is the number
                of columns in the original matrix.
        """
        return self.rsr_b1.dot(v) - self.rsr_b2.dot(v)

    def materialize(self) -> Array:
        """
        Reconstruct the original dense ternary matrix.

        Materializes both binary components and computes their difference
        to reconstruct the original ternary matrix.

        Returns:
            jax.Array: Dense ternary matrix of shape (n, m) with values in {-1, 0, 1}.
        """
        B1 = self.rsr_b1.materialize()
        B2 = self.rsr_b2.materialize()
        return (B1 - B2).astype(self.dtype)
