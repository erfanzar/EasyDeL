# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Mixture of Experts (MoE) layer implementations for EasyDeL.

This module provides a comprehensive implementation of Mixture of Experts (MoE) layers
for large-scale neural networks. It includes support for various routing strategies,
load balancing techniques, and distributed training optimizations.

Key Components:
    - **BaseMoeModule**: Abstract base class for MoE implementations with common
      utilities for routing, permutation, and metric computation.
    - **ParallelMoELinear**: Batched linear transformation layer for expert networks
      with support for ragged and grouped matrix multiplication.
    - **Routing Strategies**: Multiple routing algorithms including top-k, switch,
      expert choice, and hash-based routing.
    - **Load Balancing**: Various strategies to ensure balanced expert utilization
      including standard, switch transformer, and expert choice methods.
    - **Distributed Support**: Full support for expert parallelism (EP), tensor
      parallelism (TP), and data parallelism (DP) with optimized all-to-all
      communication patterns.

The module is designed for efficient execution on TPUs and GPUs with optimizations
for:
    - Custom VJP for gradient-efficient sorting operations
    - Pallas-based grouped matrix multiplication kernels for TPUs
    - Ragged tensor operations for variable-length expert assignments
    - Automatic sharding and partitioning for distributed training

Example:
    >>> from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
    >>> # Create a custom MoE layer by extending BaseMoeModule
    >>> class CustomMoE(BaseMoeModule):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         # Initialize gate and expert layers
    ...     def __call__(self, hidden_states):
    ...         # Implement forward pass using _moe_call helper
    ...         return self._moe_call(...)
"""

from __future__ import annotations

import enum
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
from eformer import common_types
from eformer.escale import PartitionManager
from ejkernel.modules import grouped_matmul
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

BATCH = common_types.BATCH
EMPTY = common_types.EMPTY
EMBED = common_types.EMBED
EXPERT = common_types.EXPERT
MODE_TRAIN = common_types.MODE_TRAIN
EP = common_types.EP
DP = common_types.DP
FSDP = common_types.FSDP
TP = common_types.TP
SP = common_types.SP

ExpertColumnWiseAlt = common_types.ExpertColumnWiseAlt
ExpertRowWiseAlt = common_types.ExpertRowWiseAlt
DynamicShardingAxes = common_types.DynamicShardingAxes


GMM_PLATFORM = None


def _psum32(x: jax.Array, axes: str | tuple[str, ...]) -> jax.Array:
    """Performs precision-safe parallel sum across specified axes using float32 accumulation.

    This function temporarily casts the input to float32 for the reduction operation
    to avoid numerical precision issues, then casts back to the original dtype.
    This is particularly important for bfloat16 and float16 reductions which can
    suffer from accumulation errors.

    Args:
        x: Input array to be reduced across devices.
        axes: Name(s) of the mesh axes to reduce over. Can be a single axis name
            or a tuple of axis names.

    Returns:
        Array with the same shape and dtype as input, containing the sum across
        specified device axes.

    Example:
        >>> # Sum gradients across data-parallel devices
        >>> reduced_grads = _psum32(grads, axes="dp")
        >>> # Sum across multiple axes
        >>> reduced = _psum32(x, axes=("dp", "fsdp"))
    """
    return lax.psum(x.astype(jnp.float32), axes).astype(x.dtype)


def _rsum_scatter32(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool = True) -> jax.Array:
    """Performs reduce-scatter operation with float32 accumulation for precision.

    This function combines a reduction across devices with a scatter operation that
    splits the result. It uses float32 accumulation to maintain numerical precision,
    then converts back to the original dtype.

    Args:
        x: Input array to be reduced and scattered.
        axis_name: Name of the mesh axis along which to reduce-scatter.
        scatter_dimension: Dimension along which to split the reduced result.
        tiled: If True, the scatter dimension is divided by the axis size.
            Defaults to True.

    Returns:
        Array containing the local shard of the reduced-scattered result,
        in the original dtype.

    Example:
        >>> # Reduce-scatter along TP axis, splitting the last dimension
        >>> result = _rsum_scatter32(x, axis_name="tp", scatter_dimension=-1)
    """
    return lax.psum_scatter(
        x.astype(jnp.float32),
        axis_name,
        scatter_dimension=scatter_dimension,
        tiled=tiled,
    ).astype(x.dtype)


def _argsort(x: jax.Array) -> jax.Array:
    """Returns indices that would sort the array along the last axis.

    Args:
        x: Input array to compute sorting indices for.

    Returns:
        Array of integer indices that would sort x along axis=-1.

    Example:
        >>> x = jnp.array([3, 1, 2])
        >>> indices = _argsort(x)  # Returns [1, 2, 0]
    """
    return jnp.argsort(x, axis=-1)


def _take1d(x: jax.Array, idx: jax.Array) -> jax.Array:
    """Extracts elements from array using indices along axis 0.

    Args:
        x: Input array to index into.
        idx: Integer indices specifying which elements to extract.

    Returns:
        Array containing the selected elements.

    Example:
        >>> x = jnp.array([[1, 2], [3, 4], [5, 6]])
        >>> idx = jnp.array([2, 0])
        >>> result = _take1d(x, idx)  # Returns [[5, 6], [1, 2]]
    """
    return jnp.take(x, idx, axis=0)


def _repeat_take_sorted(x: jax.Array, sort_idx: jax.Array, k: int) -> jax.Array:
    """Repeats and reorders array elements based on sorted indices divided by k.

    This is used in MoE to map tokens to their corresponding experts when each
    token is replicated k times (once for each of its top-k experts).

    Args:
        x: Input array to be indexed.
        sort_idx: Sorted indices representing flattened token-expert pairs.
        k: Number of experts per token (num_experts_per_tok).

    Returns:
        Array where each element x[i] appears k times, reordered according
        to sort_idx // k.

    Example:
        >>> tokens = jnp.array([[1, 2], [3, 4]])  # 2 tokens
        >>> sort_idx = jnp.array([0, 2, 1, 3])  # sorted expert indices
        >>> result = _repeat_take_sorted(tokens, sort_idx, k=2)
    """
    return jnp.take(x, sort_idx // k, axis=0)


def _bincount(x: jax.Array, length: int) -> jax.Array:
    """Counts occurrences of each integer value in the array.

    Args:
        x: Input integer array to count values in.
        length: Number of bins (maximum value + 1). All values in x should
            be less than length.

    Returns:
        Array of counts where result[i] is the number of times i appears in x.
        Shape: (length,).

    Example:
        >>> x = jnp.array([0, 1, 1, 2, 2, 2])
        >>> counts = _bincount(x, length=4)  # Returns [1, 2, 3, 0]
    """
    return jnp.bincount(x, length=length)


def _all_i32(*xs: jax.Array) -> tuple[jax.Array, ...]:
    """Casts all input arrays to int32 dtype.

    This is commonly used before calling operations that require int32 inputs,
    such as dynamic slicing or indexing operations.

    Args:
        *xs: Variable number of arrays to cast to int32.

    Returns:
        Tuple containing all input arrays cast to int32.

    Example:
        >>> a = jnp.array([1, 2, 3], dtype=jnp.int64)
        >>> b = jnp.array([4, 5], dtype=jnp.float32)
        >>> a_i32, b_i32 = _all_i32(a, b)  # Both are now int32
    """
    return tuple(z.astype(jnp.int32) for z in xs)


def _sort_activations(inputs: jax.Array, sort_indices: jax.Array, use_custom_vjp: bool = True) -> jax.Array:
    """Sorts activations according to provided indices with optional custom VJP.

    This function reorders the input tensor along its first dimension according to
    the provided sort indices. It optionally uses a custom VJP implementation for
    more efficient gradient computation during backpropagation.

    Args:
        inputs: Input tensor to be sorted. Shape: (N, ...).
        sort_indices: Integer array containing the sorting order. Shape: (N,).
            Each element specifies which index from inputs should appear at that
            position in the output.
        use_custom_vjp: If True, uses a custom VJP implementation that avoids
            materializing the full Jacobian during backpropagation. This is more
            memory-efficient for large tensors. Defaults to True.

    Returns:
        Sorted tensor with the same shape as inputs, where output[i] = inputs[sort_indices[i]].

    Raises:
        AssertionError: If the first dimension of inputs doesn't match the length
            of sort_indices.

    Example:
        >>> inputs = jnp.array([[1, 2], [3, 4], [5, 6]])
        >>> sort_indices = jnp.array([2, 0, 1])
        >>> sorted_inputs = _sort_activations(inputs, sort_indices)
        >>> # sorted_inputs will be [[5, 6], [1, 2], [3, 4]]
    """
    inputs_first_dim: int = inputs.shape[0]
    indices_first_dim: int = sort_indices.shape[0]
    dims_match: bool = inputs_first_dim == indices_first_dim
    assert dims_match

    result: jax.Array
    if use_custom_vjp:
        sorted_custom: jax.Array = _sort_activations_custom(inputs, sort_indices)
        result = sorted_custom
    else:
        sorted_direct: jax.Array = inputs[sort_indices, ...]
        result = sorted_direct
    return result


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    """Custom VJP implementation for sorting activations.

    This function provides a custom vector-Jacobian product (VJP) for the sorting
    operation, which is more memory-efficient than the default automatic differentiation.
    The custom VJP avoids materializing the full Jacobian matrix by directly computing
    the inverse permutation during the backward pass.

    Args:
        inputs: Input tensor to be sorted. Shape: (N, ...).
        sort_indices: Integer array containing the sorting order. Shape: (N,).

    Returns:
        Sorted tensor where output[i] = inputs[sort_indices[i]].

    Note:
        This function is decorated with @jax.custom_vjp, which means it has custom
        forward and backward implementations defined in _sort_activations_custom_fwd
        and _sort_activations_custom_bwd.
    """
    return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Forward pass for custom VJP sorting.

    Computes the sorted output and stores the sort indices as residuals for the
    backward pass.

    Args:
        inputs: Input tensor to be sorted.
        sort_indices: Sorting indices.

    Returns:
        Tuple of (sorted_output, residuals) where residuals contains the sort_indices
        needed for the backward pass.
    """
    sorted_output: jax.Array = _sort_activations_custom(inputs, sort_indices)
    residuals: jax.Array = sort_indices
    result: tuple[jax.Array, jax.Array] = (sorted_output, residuals)
    return result


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
    """Backward pass for custom VJP sorting.

    Applies the inverse permutation to gradients to route them back to their
    original positions.

    Args:
        residuals: Stored sort_indices from the forward pass.
        grads: Gradients flowing backward from the sorted output.

    Returns:
        Tuple of (input_grads, None) where input_grads are the gradients with
        respect to the original unsorted inputs, and None indicates no gradient
        for sort_indices.
    """
    sort_indices: jax.Array = residuals
    inverse_indices: jax.Array = jnp.argsort(sort_indices)
    input_grads: jax.Array = _sort_activations_custom(grads, inverse_indices)
    indices_grad: None = None
    result: tuple[jax.Array, None] = (input_grads, indices_grad)
    return result


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


@dataclass(frozen=True)
class MoeFusedPolicy:
    """Configuration policy for fused MoE operations with advanced sharding strategies.

    This dataclass defines the execution policy for fused MoE implementations, controlling
    how computations are distributed across different parallelism dimensions and which
    optimizations are applied. It enables fine-grained control over tensor parallel (TP),
    sequence parallel (SP), and fully-sharded data parallel (FSDP) strategies.

    Attributes:
        gate_slice_k_axes: Tuple of axis names that shard the contracting dimension (K/hidden)
            for the gate/router computation. Order matters as it's linearized row-major.
            Defaults to ("sp", "fsdp") to split the hidden dimension across SP and FSDP.
        wiwu_slice_k_axes: Tuple of axis names that shard the contracting dimension for
            the expert wi (input) and wu (up) projections. Defaults to ("sp", "fsdp").
        combine_axes: Axes to reduce over (psum) after gate/wi/wu computations to combine
            results from split-K or parameter shards. Defaults to ("sp", "fsdp").
        tp_axis: Name of the tensor parallel axis. Defaults to "tp".
        rs_dim: Dimension for reduce-scatter operation. Defaults to -1 (last axis).
        rs_enabled: Whether reduce-scatter is enabled for TP. Defaults to True.
        gather_gate_on_tp: If True, gather activations along the hidden dimension before
            gate computation. This ensures full precision routing at the cost of
            communication. Defaults to True.
        gather_logits_on_tp: If True, gather router logits across TP before top-k selection.
            This ensures consistent routing across TP shards. Defaults to True.
        gmm_platform: Backend platform for grouped matrix multiplication. Options are:
            None (automatic), 'xla' (XLA default), 'pallas' (TPU-optimized kernel),
            'cuda' (GPU-optimized), or 'auto'. Defaults to None.
        reduce_dtype: Data type for reduction accumulators to balance precision and
            performance. Defaults to jnp.bfloat16.

    Example:
        >>> # Custom policy for aggressive TP sharding
        >>> policy = MoeFusedPolicy(
        ...     tp_axis="tp",
        ...     gather_gate_on_tp=False,  # Skip gather for speed
        ...     gmm_platform="pallas",  # Use TPU kernel
        ...     reduce_dtype=jnp.float32  # Higher precision
        ... )
    """

    # Which axes shard the contracting K (hidden) dim; order matters (row-major linearized)
    gate_slice_k_axes: tuple[str, ...] = ("sp", "fsdp")
    wiwu_slice_k_axes: tuple[str, ...] = ("sp", "fsdp")

    # Axes to psum (combine split-K/param shards) after gate/wi/wu
    combine_axes: tuple[str, ...] = ("sp", "fsdp")

    # TP reduce-scatter
    tp_axis: str = "tp"
    rs_dim: int = -1  # last axis by default
    rs_enabled: bool = True

    # Gather controls
    gather_gate_on_tp: bool = True  # gather activations on H before gate
    gather_logits_on_tp: bool = True  # gather logits on E across TP for top-k

    # GMM backend choice for grouped_matmul
    gmm_platform: str | None = None  # None|'xla'|'pallas'|'cuda'|'auto'

    # Reduction accumulator dtype
    reduce_dtype = jnp.bfloat16


@dataclass(frozen=True)
class MoeFusedHooks:
    """Hook system for custom interventions during fused MoE execution.

    This dataclass provides a comprehensive set of hook points throughout the fused
    MoE forward pass, allowing users to inject custom logic, apply transformations,
    or collect debugging information at specific stages of execution. Each hook is
    an optional callable that receives relevant tensors and context as input and
    returns modified tensors.

    The hooks are called in the following order during MoE execution:
        1. before_gate: Before router computation
        2. after_gate: After router logits computation
        3. before_topk: Before top-k expert selection
        4. after_topk: After expert selection
        5. before_ep_dispatch: Before expert-parallel all-to-all
        6. after_ep_receive: After receiving from expert-parallel communication
        7. before_wiwu: Before wi/wu expert projections
        8. after_wiwu: After wi/wu, typically for activation function
        9. before_wo: Before output projection
        10. after_wo: After output projection
        11. before_combine: Before combining expert outputs with routing weights
        12. finalize_output: Final processing before returning output

    Attributes:
        before_gate: Called before router computation. Receives activations and gate weights.
        after_gate: Called after router logits computation. Receives and returns router logits.
        before_topk: Called before top-k selection. Receives router probabilities.
        after_topk: Called after expert selection. Receives and returns (weights, expert_ids).
        before_ep_dispatch: Called before expert-parallel dispatch. Receives sorted activations.
        after_ep_receive: Called after EP communication. Receives local expert inputs.
        before_wiwu: Called before wi/wu projections. Can modify inputs and weights.
        after_wiwu: Called after wi/wu, typically to apply custom activation. Receives (y0, y1).
        before_wo: Called before output projection. Receives intermediate activations.
        after_wo: Called after output projection. Receives expert outputs.
        before_combine: Called before combining expert outputs. Receives outputs and weights.
        finalize_output: Called as final step. Receives and returns (output, router_logits).

    Example:
        >>> def debug_router(logits, ctx):
        ...     print(f"Router logits stats: mean={jnp.mean(logits)}, std={jnp.std(logits)}")
        ...     return logits
        >>>
        >>> def custom_activation(y0, y1, ctx):
        ...     # Custom gating beyond standard GLU
        ...     return jax.nn.silu(y0) * jax.nn.sigmoid(y1)
        >>>
        >>> hooks = MoeFusedHooks(
        ...     after_gate=debug_router,
        ...     after_wiwu=custom_activation
        ... )

    Note:
        All hooks receive a MoeContext object as their last parameter, providing
        access to mesh information, batch size, and other execution details.
    """

    before_gate: Callable | None = None
    after_gate: Callable | None = None
    before_topk: Callable | None = None
    after_topk: Callable | None = None
    before_ep_dispatch: Callable | None = None
    after_ep_receive: Callable | None = None
    before_wiwu: Callable | None = None
    after_wiwu: Callable | None = None
    before_wo: Callable | None = None
    after_wo: Callable | None = None
    before_combine: Callable | None = None
    finalize_output: Callable | None = None

    def __hash__(self) -> int:
        """Makes the hooks dataclass hashable for NNX graph hashing.

        Returns:
            Hash value based on the identity of all hook callables.

        Note:
            Uses id() of callables rather than hashing the functions themselves,
            which works correctly for functions, lambdas, and partials.
        """
        # Use identities of callables (works for functions, lambdas, partials).
        return hash(
            (
                id(self.before_gate),
                id(self.after_gate),
                id(self.before_topk),
                id(self.after_topk),
                id(self.before_ep_dispatch),
                id(self.after_ep_receive),
                id(self.before_wiwu),
                id(self.after_wiwu),
                id(self.before_wo),
                id(self.after_wo),
                id(self.before_combine),
                id(self.finalize_output),
            )
        )


@dataclass
class MoeContext:
    """Read-only runtime context information passed to MoE hooks during execution.

    This dataclass encapsulates the execution environment and tensor dimensions
    available during MoE forward pass. It provides hooks with essential information
    about the distributed mesh configuration and tensor shapes, enabling context-aware
    custom logic.

    Attributes:
        mesh: JAX device mesh defining the multi-device topology for distributed
            execution. Contains device assignments and axis names for parallelism.
        axis_names: Tuple of axis names present in the mesh (e.g., ('dp', 'tp', 'ep')).
            Used for identifying available parallelism dimensions.
        axis_sizes: Dictionary mapping axis names to their sizes (number of devices).
            Example: {'dp': 4, 'tp': 2, 'ep': 8} indicates 4-way data parallel,
            2-way tensor parallel, and 8-way expert parallel.
        B_loc: Local batch size on this device/shard. The actual batch size seen
            by computations after batch-dimension sharding.
        S_loc: Local sequence length on this device/shard. The sequence length
            after sequence-parallel sharding, if applied.
        H_full: Full hidden dimension size before any tensor-parallel sharding.
            This is the complete hidden size of the model.
        K: Number of experts per token (num_experts_per_tok). Typically 1-2 for
            top-k routing.
        E: Total number of experts in the MoE layer.

    Example:
        >>> def custom_hook(logits, ctx):
        ...     print(f"Running on mesh with axes: {ctx.axis_names}")
        ...     print(f"Local batch size: {ctx.B_loc}, sequence length: {ctx.S_loc}")
        ...     print(f"Processing {ctx.K} experts per token from {ctx.E} total experts")
        ...     return logits

    Note:
        This context is immutable (frozen=False but intended for read-only use)
        and should not be modified by hooks. It provides a consistent view of
        the execution environment across all hook invocations.
    """

    # Read-only runtime info passed to hooks
    mesh: jax.sharding.Mesh
    axis_names: tuple[str, ...]
    axis_sizes: dict
    B_loc: int
    S_loc: int
    H_full: int
    K: int
    E: int


# Helpers
def _canon_dim(ndim: int, dim: int) -> int:
    """Canonicalizes a dimension index to be non-negative.

    Converts negative dimension indices (counting from the end) to their
    equivalent positive indices.

    Args:
        ndim: Total number of dimensions in the array.
        dim: Dimension index to canonicalize. Can be negative.

    Returns:
        Non-negative dimension index in range [0, ndim).

    Example:
        >>> _canon_dim(ndim=3, dim=-1)  # Returns 2
        >>> _canon_dim(ndim=3, dim=1)   # Returns 1
    """
    return dim if dim >= 0 else (ndim + dim)


def _psum_maybe(
    x: jax.Array, axes: tuple[str, ...], mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Conditionally performs parallel sum across specified axes if they exist in mesh.

    This function filters the requested axes to only include those that exist in
    the mesh and have size > 1, then performs a psum reduction. If no valid axes
    are found, returns the input unchanged.

    Args:
        x: Input array to reduce.
        axes: Tuple of axis names to potentially reduce over.
        mesh: JAX device mesh containing available axes.
        dtype: Data type for accumulation during reduction. Defaults to float32
            for numerical stability.

    Returns:
        Reduced array if valid axes exist, otherwise the original array.
        Always returns in the original dtype of x.

    Example:
        >>> # Only reduces if 'sp' and 'fsdp' exist in mesh
        >>> result = _psum_maybe(x, axes=("sp", "fsdp"), mesh=mesh)
    """
    names = tuple(a for a in axes if a in mesh.axis_names and mesh.shape.get(a, 1) > 1)
    return lax.psum(x.astype(dtype), names).astype(x.dtype) if names else x


def _rsum_scatter_maybe(
    x: jax.Array, axis_name: str, dim: int, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Conditionally performs reduce-scatter if the axis exists and has size > 1.

    This function checks if the specified axis exists in the mesh with size > 1,
    and if so, performs a reduce-scatter operation. Otherwise, returns input unchanged.

    Args:
        x: Input array to reduce-scatter.
        axis_name: Name of the axis to reduce-scatter along.
        dim: Dimension of x to split during scatter. Can be negative.
        mesh: JAX device mesh containing available axes.
        dtype: Data type for accumulation during reduction. Defaults to float32.

    Returns:
        Reduce-scattered array with the scatter dimension divided by axis size,
        or the original array if the axis doesn't exist or has size 1.

    Example:
        >>> # Reduce-scatter on last dim if 'tp' axis exists
        >>> result = _rsum_scatter_maybe(x, axis_name="tp", dim=-1, mesh=mesh)
    """
    if axis_name not in mesh.axis_names or mesh.shape.get(axis_name, 1) == 1:
        return x
    dim = _canon_dim(x.ndim, dim)
    return lax.psum_scatter(x.astype(dtype), axis_name, scatter_dimension=dim, tiled=True).astype(x.dtype)


def _slice_k_for_param_shards(
    x_mat: jax.Array, chunk: int, axes: tuple[str, ...], mesh: jax.sharding.Mesh, axis: int = 1
) -> jax.Array:
    """Slices activation tensor to match the chunk size of parameter shards.

    When parameters are sharded across multiple axes (e.g., sequence parallel + FSDP),
    this function computes which chunk of the activation tensor should be used by
    each device based on its multi-dimensional position in the mesh. The axes are
    linearized in row-major order to compute a single index.

    Args:
        x_mat: Activation tensor to slice.
        chunk: Size of each parameter chunk (local size after sharding).
        axes: Tuple of axis names that shard the parameter dimension, in row-major
            order. For example, ("sp", "fsdp") means SP varies faster than FSDP.
        mesh: JAX device mesh.
        axis: Dimension of x_mat to slice. Defaults to 1 (typically the hidden dimension).

    Returns:
        Sliced tensor of size chunk along the specified axis, or the original tensor
        if no sharding is needed (stride=1 or chunk matches full size).

    Example:
        >>> # Slice hidden dim for device at position (sp=1, fsdp=0)
        >>> # With sp_size=2, fsdp_size=4: linear_idx = 1*1 + 0*2 = 1
        >>> x_sliced = _slice_k_for_param_shards(
        ...     x, chunk=256, axes=("sp", "fsdp"), mesh=mesh, axis=1)
    """
    # Compute linear index across provided axes in row-major order
    idx = 0
    stride = 1
    for ax in axes:
        size = mesh.shape.get(ax, 1)
        if size > 1:
            idx = idx + lax.axis_index(ax) * stride
            stride *= size
    if stride == 1 or chunk == x_mat.shape[axis]:
        return x_mat
    start = idx * chunk
    return lax.dynamic_slice_in_dim(x_mat, start_index=start, slice_size=chunk, axis=axis)


class _Transform(enum.Enum):
    """Enumeration of transformation strategies for all-to-all communication parameters.

    This enum defines different strategies for computing offsets and sizes used in
    ragged all-to-all communication patterns during expert parallel execution.

    Attributes:
        INPUT_OFFSET: Strategy for computing input buffer offsets.
        SEND_SIZE: Strategy for computing send buffer sizes.
        OUTPUT_OFFSET: Strategy for computing output buffer offsets.
        RECV_SIZE: Strategy for computing receive buffer sizes.
    """

    INPUT_OFFSET = enum.auto()
    SEND_SIZE = enum.auto()
    OUTPUT_OFFSET = enum.auto()
    RECV_SIZE = enum.auto()


def _get_all_to_all_params(
    all_shards_group_sizes: jax.Array,  # [num_batch_shards, num_experts] or [num_experts]
    shard_id: int,
    num_expert_parallelism: int,
    is_batch_sharded: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Computes parameters for ragged all-to-all communication in expert parallelism.

    This function calculates the offsets and sizes needed for ragged all-to-all
    communication when distributing tokens across expert-parallel devices. It handles
    both batch-sharded and non-batch-sharded configurations.

    Args:
        all_shards_group_sizes: Array containing the number of tokens assigned to each
            expert across all shards. Shape is either [num_batch_shards, num_experts]
            when batch is sharded, or [num_experts] when batch is replicated.
        shard_id: The ID of the current shard/device in the expert parallel dimension.
        num_expert_parallelism: Total number of expert parallel shards.
        is_batch_sharded: Whether the batch dimension is also sharded across devices.
            If True, tokens are distributed across both batch and expert dimensions.
            If False, the batch is replicated and only experts are distributed.

    Returns:
        A tuple containing four arrays:
            - input_offsets: Offsets into the input buffer for each expert. Shape: (num_experts,).
            - send_sizes: Number of tokens to send to each expert shard. Shape: (num_experts,).
            - output_offsets: Offsets into the output buffer for received tokens. Shape: (num_experts,).
            - recv_sizes: Number of tokens to receive from each shard. Shape: (num_experts,).

    Example:
        >>> # For 4 experts distributed across 2 EP shards
        >>> group_sizes = jnp.array([10, 15, 8, 12])  # tokens per expert
        >>> in_off, send, out_off, recv = _get_all_to_all_params(
        ...     group_sizes, shard_id=0, num_expert_parallelism=2, is_batch_sharded=False)
        >>> # Shard 0 handles experts 0-1, shard 1 handles experts 2-3
    """

    def transform(inp, shard_id, strategy, is_batch_sharded):
        if is_batch_sharded:
            if strategy == _Transform.INPUT_OFFSET:
                local = inp[shard_id]
                return jnp.concatenate([jnp.array([0], dtype=inp.dtype), jnp.cumsum(local)[:-1]])
            elif strategy == _Transform.SEND_SIZE:
                return inp[shard_id]
            elif strategy == _Transform.OUTPUT_OFFSET:
                zero = jnp.zeros((1, *inp.shape[1:]), dtype=inp.dtype)
                cum = jnp.cumsum(jnp.concatenate([zero, inp], axis=0), axis=0, dtype=inp.dtype)
                return cum[shard_id]
            elif strategy == _Transform.RECV_SIZE:
                return inp[:, shard_id]
        else:
            if strategy == _Transform.INPUT_OFFSET:
                return jnp.zeros(num_expert_parallelism, dtype=inp.dtype)
            elif strategy == _Transform.SEND_SIZE:
                return jnp.repeat(inp[shard_id], num_expert_parallelism)
            elif strategy == _Transform.OUTPUT_OFFSET:
                base = jnp.concatenate([jnp.array([0], dtype=inp.dtype), jnp.cumsum(inp[:-1])])
                return jnp.repeat(base[shard_id], num_expert_parallelism)
            elif strategy == _Transform.RECV_SIZE:
                return inp
        raise ValueError("Unknown transform")

    in_off = transform(all_shards_group_sizes, shard_id, _Transform.INPUT_OFFSET, is_batch_sharded)
    send = transform(all_shards_group_sizes, shard_id, _Transform.SEND_SIZE, is_batch_sharded)
    out_off = transform(all_shards_group_sizes, shard_id, _Transform.OUTPUT_OFFSET, is_batch_sharded)
    recv = transform(all_shards_group_sizes, shard_id, _Transform.RECV_SIZE, is_batch_sharded)
    return in_off, send, out_off, recv


def _local_permute(
    inputs: jax.Array,
    global_group_sizes: jax.Array,
    local_expert_size: int,
    shard_index: int,
    is_offset: bool = False,
    global_sorted_experts: jax.Array | None = None,
    use_custom_sort_vjp: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Performs local permutation of tokens to group them by expert on a single shard.

    This function reorders tokens on a local shard so that all tokens assigned to
    the same expert are grouped together. This is a crucial step after all-to-all
    communication in expert parallel execution.

    Args:
        inputs: Local token representations. Shape: (tokens_local, hidden_dim).
        global_group_sizes: Number of tokens per expert across all shards.
            Shape: (num_batch_shards, num_experts) or (1, num_experts).
        local_expert_size: Number of experts handled by this shard.
        shard_index: Index of the current expert-parallel shard.
        is_offset: If True, uses global_sorted_experts to determine expert assignments.
            If False, generates expert indices based on token counts.
        global_sorted_experts: Optional pre-computed expert assignments for each token.
            Required when is_offset=True. Shape: (tokens_local,).
        use_custom_sort_vjp: Whether to use custom VJP for sorting operations.

    Returns:
        A tuple containing:
            - sorted_inputs: Tokens sorted by expert assignment. Shape: (tokens_local, hidden_dim).
            - sorted_indices: Indices used for sorting, needed for unpermutation.
            - local_group_size: Number of tokens per local expert. Shape: (local_expert_size,).
            - sorted_experts_ids: Expert IDs after sorting. Shape: (tokens_local,).

    Example:
        >>> # Permute tokens for experts 0-1 on shard 0
        >>> inputs = jnp.ones((100, 768))  # 100 tokens, 768 hidden dim
        >>> group_sizes = jnp.array([[50, 50, 30, 20]])  # 4 experts total
        >>> sorted_inputs, indices, sizes, expert_ids = _local_permute(
        ...     inputs, group_sizes, local_expert_size=2, shard_index=0)
    """
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
    )
    local_sizes = all_shard_local_sizes.reshape(-1)
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    if is_offset:
        divided = jnp.floor_divide(global_sorted_experts, local_expert_size)
        expert_indices = jnp.where(
            divided == shard_index,
            jnp.mod(global_sorted_experts, local_expert_size),
            local_expert_size,
        )
    else:
        base = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
        expert_indices = jnp.repeat(base, local_sizes, total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    sorted_inputs = _sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids


class MoeRoutingStrategy(enum.Enum):
    """Defines the available strategies for routing tokens to experts in an MoE layer.

    Attributes:
        TOP_K: Standard top-k routing, where each token is routed to the k experts
            with the highest router scores.
        TOP_K_NDIV: Top-k routing without dividing the weights by their sum.
        SWITCH: Switch Transformer-style routing, where each token is routed to
            only the top-1 expert.
        EMPTY_CHOICE: Expert Choice routing, where each expert selects the top-k
            tokens with the highest scores for that expert.
        HASH: A simple hashing-based routing for debugging or baseline comparison.
    """

    TOP_K = "top_k"
    TOP_K_NDIV = "top_k_ndiv"
    SWITCH = "switch"
    EMPTY_CHOICE = "expert_choice"
    HASH = "hash"


class MoeLoadBalancingStrategy(enum.Enum):
    """Defines the available strategies for calculating the load balancing loss.

    Attributes:
        STANDARD: A common load balancing loss based on the product of expert
            loads and mean router probabilities.
        SWITCH_TRANSFORMER: The load balancing loss used in the Switch
            Transformer paper.
        EMPTY_CHOICE: A load balancing loss variant suitable for Expert Choice
            routing, often based on the variance of expert loads.
        NONE: No load balancing loss is applied.
    """

    STANDARD = "standard"
    SWITCH_TRANSFORMER = "switch_transformer"
    EMPTY_CHOICE = "expert_choice"
    NONE = "none"


@dataclass
class MoeMetrics:
    """A container for storing metrics and auxiliary losses from an MoE layer.

    Attributes:
        expert_loads: An array representing the number of tokens routed to each
            expert. Shape: (num_experts,).
        router_probs: The probabilities output by the router for each token and
            expert. Shape: (num_tokens, num_experts).
        selected_experts: The indices of the experts selected for each token.
            Shape: (num_tokens, num_experts_per_tok).
        selected_weights: The weights assigned to the selected experts for each
            token. Shape: (num_tokens, num_experts_per_tok).
        load_balancing_loss: The calculated auxiliary loss to encourage balanced
            load across experts.
        router_z_loss: The calculated auxiliary loss to encourage small router
            logits, promoting stability.
        expert_utilization: The fraction of experts that were utilized (i.e.,
            received at least one token).
        routing_entropy: The entropy of the router probabilities, measuring routing
            confidence.
    """

    expert_loads: Float[Array, "num_experts"]  # type:ignore #noqa
    router_probs: Float[Array, "batch_seq num_experts"]
    selected_experts: Int[Array, "batch_seq num_experts_per_tok"]
    selected_weights: Float[Array, "batch_seq num_experts_per_tok"]
    load_balancing_loss: float | None = None
    router_z_loss: float | None = None
    expert_utilization: float | None = None
    routing_entropy: float | None = None


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros
Initializer = nn.initializers.Initializer


class BaseMoeModule(nn.Module, ABC):
    """An abstract base class for Mixture of Experts (MoE) modules.

    This class provides a foundational structure and common utilities for
    implementing various MoE architectures. It includes methods for token routing,
    data permutation for efficient expert computation, load balancing loss
    calculation, and sharding for distributed environments. Subclasses are

    expected to implement the `__call__` method to define the specific MoE forward
    pass.

    Attributes:
        config: The configuration object for the MoE module.
        mesh: The JAX device mesh for distributed computation.
        n_routed_experts: The total number of experts available for routing.
        num_experts_per_tok: The number of experts each token is routed to (k).
        hidden_size: The dimension of the hidden states.
        lbl_coef: The coefficient for the load balancing loss.
        rzl_coef: The coefficient for the router z-loss.
        routing_strategy: The strategy used for routing tokens to experts.
        load_balancing_strategy: The strategy used for calculating the load
            balancing loss.
    """

    def __init__(
        self,
        config: EasyDeLBaseConfig,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        hidden_size: int | None = None,
        lbl_coef: float | None = None,
        rzl_coef: float | None = None,
        routing_strategy: MoeRoutingStrategy = MoeRoutingStrategy.TOP_K,
        load_balancing_strategy: MoeLoadBalancingStrategy = MoeLoadBalancingStrategy.STANDARD,
        moe_policy: MoeFusedPolicy | None = None,
        moe_hooks: MoeFusedHooks | None = None,
    ):
        """Initializes the BaseMoeModule.

        Args:
            config: The configuration object for this MoE module.
            n_routed_experts: The total number of experts. If None, it's taken
                from `config.n_routed_experts`.
            num_experts_per_tok: The number of experts to route each token to. If
                None, it's taken from `config.num_experts_per_tok`.
            hidden_size: The hidden dimension of the input and output. If None,
                it's taken from `config.hidden_size`.
            lbl_coef: The coefficient for the load balancing loss.
            rzl_coef: The coefficient for the router z-loss.
            routing_strategy: The strategy for routing tokens to experts.
            load_balancing_strategy: The strategy for load balancing.
        """
        super().__init__()
        self.config = config
        self.mesh = config.mesh
        self.partition_manager = config.partition_manager
        self.n_routed_experts = n_routed_experts or config.n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok or config.num_experts_per_tok
        self.hidden_size = hidden_size or config.hidden_size
        self.lbl_coef = lbl_coef
        self.rzl_coef = rzl_coef
        self.routing_strategy = routing_strategy
        self.load_balancing_strategy = load_balancing_strategy
        self.moe_policy = MoeFusedPolicy() if moe_policy is None else moe_policy
        self.moe_hooks = MoeFusedHooks() if moe_hooks is None else moe_hooks

    def _route(
        self,
        router_probs: jax.Array,
        routing_strategy: MoeRoutingStrategy | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Selects experts for each token based on the specified routing strategy.

        This method wraps the sharded routing implementation.

        Args:
            router_probs: An array of router probabilities with shape
                `(batch_size * seq_len, num_experts)`.
            routing_strategy: The routing strategy to use. If None, the default
                strategy from the constructor is used.

        Returns:
            A tuple containing:
                - selected_weights: The weights for the selected experts. Shape
                    `(batch_size * seq_len, num_experts_per_tok)`.
                - selected_experts: The indices of the selected experts. Shape
                    `(batch_size * seq_len, num_experts_per_tok)`.
        """
        return self._route_sharded(router_probs, routing_strategy or self.routing_strategy)

    def _route_sharded(
        self, router_probs: Float[Array, "batch_seq num_experts"], strategy: MoeRoutingStrategy
    ) -> tuple[Float[Array, "batch_seq k"], Int[Array, "batch_seq k"]]:
        """Performs sharded routing of tokens to experts with improved partitioning.

        Router probs shape: (batch * seq_len, num_experts)
        Partitioned as: ((dp, fsdp), sp) for batch dimension
        """
        pmag = self.partition_manager

        if router_probs.ndim == 2:
            pspec = pmag.resolve(axes=[BATCH, EMPTY], mode=MODE_TRAIN, shape=router_probs.shape)
            in_specs = pspec
            out_specs = (pspec, pspec)
        elif router_probs.ndim == 3:
            pspec = pmag.resolve(axes=[BATCH, EMPTY, EMPTY], mode=MODE_TRAIN, shape=router_probs.shape)
            in_specs = pspec
            out_specs = (pspec, pspec)
        else:
            in_specs = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)
            out_specs = (in_specs, in_specs)

        @partial(shard_map, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        def sharded_route(router_probs_):
            return self._route_local(router_probs_, strategy)

        return sharded_route(router_probs)

    def _route_local(
        self, router_probs: Float[Array, "batch_seq num_experts"], strategy: MoeRoutingStrategy
    ) -> tuple[Float[Array, "batch_seq k"], Int[Array, "batch_seq k"]]:
        """Implements the routing logic on a local device shard.

        Args:
            router_probs: A shard of router probabilities.
            strategy: The routing strategy to apply.

        Returns:
            A tuple containing the selected weights and expert indices for the
            local shard.

        Raises:
            ValueError: If an unknown routing strategy is provided.
        """
        if strategy == MoeRoutingStrategy.TOP_K:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
            selected_weights /= selected_weights.sum(-1, keepdims=True)
        elif strategy == MoeRoutingStrategy.TOP_K_NDIV:
            selected_weights, selected_experts = jax.lax.top_k(router_probs, self.num_experts_per_tok)
        elif strategy == MoeRoutingStrategy.SWITCH:
            selected_experts = jnp.argmax(router_probs, axis=-1, keepdims=True)
            selected_weights = jnp.take_along_axis(router_probs, selected_experts, axis=-1)
        elif strategy == MoeRoutingStrategy.EMPTY_CHOICE:
            k = router_probs.shape[0] // self.n_routed_experts
            selected_weights, selected_experts = jax.lax.top_k(router_probs.T, k=k)
            selected_weights = selected_weights.T
            selected_experts = selected_experts.T
        elif strategy == MoeRoutingStrategy.HASH:
            token_ids = jnp.arange(router_probs.shape[0])
            selected_experts = (token_ids % self.n_routed_experts)[..., None]
            selected_weights = jnp.ones_like(selected_experts, dtype=router_probs.dtype)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        return selected_weights, selected_experts

    def _moe_call_sparse_ep(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        use_custom_sort_vjp: bool = True,
        batch_sharded_by_expert: bool = False,
    ):
        """Executes MoE forward pass with expert parallelism using all-to-all communication.

        This method implements an optimized MoE forward pass specifically designed for
        expert parallel (EP) execution. It uses ragged all-to-all communication to
        efficiently distribute tokens to their assigned experts across devices.

        The execution flow:
        1. Route tokens to experts using the gate layer
        2. Replicate and sort tokens globally by expert ID
        3. Use all-to-all to redistribute tokens to expert-owning devices
        4. Locally permute tokens on each device to group by expert
        5. Execute expert computation
        6. Reverse the communication pattern to return results
        7. Combine expert outputs using routing weights

        Args:
            gate_layer: Router module that produces expert selection logits.
            expert_layer: Module containing the expert networks (ParallelMoELinear).
            hidden_state: Input tensor. Shape: (batch_size, seq_len, hidden_dim).
            output_metrics: If True, returns detailed metrics; otherwise returns logits.
            use_custom_sort_vjp: Whether to use memory-efficient custom VJP for sorting.
            batch_sharded_by_expert: If True, batch dimension is sharded across EP devices
                for additional parallelism. If False, batch is replicated.

        Returns:
            Tuple of (output, metrics/logits) where:
                - output: Processed tensor. Shape: (batch_size, seq_len, hidden_dim).
                - Second element is MoeMetrics if output_metrics=True, else router logits.

        Note:
            This method automatically falls back to standard _moe_call if expert
            parallelism is not available (no 'expert' or 'ep' axis in mesh).

        Example:
            >>> # With expert parallelism across 4 devices
            >>> output, metrics = self._moe_call_sparse_ep(
            ...     gate_layer, expert_layer, hidden_states,
            ...     output_metrics=True, batch_sharded_by_expert=True)
        """
        mesh = self.mesh
        pmag = self.partition_manager
        expert_axis_name = "expert" if "expert" in mesh.axis_names else ("ep" if "ep" in mesh.axis_names else None)
        if expert_axis_name is None or mesh.shape[expert_axis_name] <= 1:
            return self._moe_call(gate_layer, expert_layer, hidden_state, output_metrics=output_metrics)

        B, S, H = hidden_state.shape

        hs_flat = hidden_state.reshape(-1, H)
        router_logits = gate_layer(hs_flat).astype(jnp.float32)
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        selected_weights, selected_experts = self._route(router_probs)
        sorted_inputs, sorted_order, group_sizes, sorted_experts = self._replicate_and_sort_tokens(
            hs_flat, selected_experts, use_custom_sort_vjp
        )

        x_in_specs = pmag.resolve(axes=[BATCH, TP], mode=MODE_TRAIN, shape=sorted_inputs.shape)
        gs_in_specs = pmag.resolve(axes=[EXPERT], mode=MODE_TRAIN, shape=group_sizes.shape)
        se_in_specs = pmag.resolve(axes=[BATCH], mode=MODE_TRAIN, shape=sorted_experts.shape)
        out_specs = pmag.resolve(axes=[BATCH, TP], mode=MODE_TRAIN, shape=(sorted_inputs.shape[0], H))

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(x_in_specs, gs_in_specs, se_in_specs),
            out_specs=out_specs,
            check_vma=False,
        )
        def ep_wrapper(x_sorted, g_sizes, s_experts):
            shard_id = jax.lax.axis_index(expert_axis_name)
            num_ep = mesh.shape[expert_axis_name]
            assert self.n_routed_experts % num_ep == 0, "n_routed_experts must be divisible by EP size"
            local_E = self.n_routed_experts // num_ep
            batch_axis = next((ax for ax in getattr(mesh, "axis_names", ()) if ax in ("dp", "data", "fsdp")), None)

            def _i32(x):
                return x.astype(jnp.int32)

            if batch_sharded_by_expert and batch_axis is not None:
                all_shards_group_sizes = jax.lax.all_gather(
                    jnp.sum(g_sizes.reshape(-1, local_E), axis=1), axis_name=batch_axis
                )
                input_offsets, send_sizes, output_offsets, recv_sizes = _get_all_to_all_params(
                    all_shards_group_sizes, shard_id, num_ep, is_batch_sharded=True
                )
                input_offsets, send_sizes, output_offsets, recv_sizes = map(
                    _i32, (input_offsets, send_sizes, output_offsets, recv_sizes)
                )

                buffer_size = int(num_ep * B * S * self.num_experts_per_tok)
                output_shape = jnp.zeros((buffer_size, H), dtype=x_sorted.dtype)
                x_dispatched = jax.lax.ragged_all_to_all(
                    x_sorted,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=expert_axis_name,
                )

                global_group_sizes = jax.lax.all_gather(g_sizes, axis_name=expert_axis_name)
                x_local, local_sorted_idx, local_group_sizes, _local_experts = _local_permute(
                    x_dispatched,
                    global_group_sizes,
                    local_E,
                    shard_index=shard_id,
                    use_custom_sort_vjp=use_custom_sort_vjp,
                )

            else:
                reshaped_group_sizes = jnp.sum(g_sizes.reshape(-1, local_E), axis=1)
                input_offsets, send_sizes, output_offsets, recv_sizes = _get_all_to_all_params(
                    reshaped_group_sizes, shard_id, num_ep, is_batch_sharded=False
                )
                input_offsets, send_sizes, output_offsets, recv_sizes = map(
                    _i32, (input_offsets, send_sizes, output_offsets, recv_sizes)
                )

                buffer_size = int(num_ep * B * S * self.num_experts_per_tok)
                output_shape = jnp.zeros((buffer_size, H), dtype=x_sorted.dtype)
                x_local = jax.lax.ragged_all_to_all(
                    x_sorted,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=expert_axis_name,
                )

                x_local, local_sorted_idx, local_group_sizes, _local_experts = _local_permute(
                    x_local,
                    g_sizes[None, :],
                    local_E,
                    shard_index=shard_id,
                    is_offset=True,
                    global_sorted_experts=s_experts,
                    use_custom_sort_vjp=use_custom_sort_vjp,
                )

            y_local = expert_layer(x_local, local_group_sizes)

            y_local = _sort_activations(y_local, jnp.argsort(local_sorted_idx), use_custom_sort_vjp)

            if batch_sharded_by_expert and batch_axis is not None:
                input_offsets, send_sizes, output_offsets, recv_sizes = _get_all_to_all_params(
                    jnp.transpose(all_shards_group_sizes), shard_id, num_ep, is_batch_sharded=True
                )
                input_offsets, send_sizes, output_offsets, recv_sizes = map(
                    _i32, (input_offsets, send_sizes, output_offsets, recv_sizes)
                )
            else:
                input_offsets, send_sizes, output_offsets, recv_sizes = _get_all_to_all_params(
                    reshaped_group_sizes, shard_id, num_ep, is_batch_sharded=False
                )
                input_offsets, send_sizes, output_offsets, recv_sizes = map(
                    _i32, (input_offsets, send_sizes, output_offsets, recv_sizes)
                )

            y = jax.lax.ragged_all_to_all(
                y_local,
                jnp.zeros_like(x_sorted),
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=expert_axis_name,
            )
            return y

        out_sorted = ep_wrapper(sorted_inputs, group_sizes, sorted_experts)
        out_unsorted = _sort_activations(out_sorted, jnp.argsort(sorted_order), use_custom_sort_vjp)
        out_unflat = out_unsorted.reshape(-1, self.num_experts_per_tok, H)
        output = jnp.sum(out_unflat * selected_weights[..., None], axis=1).reshape(B, S, H)
        if output_metrics:
            group_sizes_all = group_sizes
            metrics = self._compute_metrics(
                router_logits,
                router_probs,
                selected_experts,
                selected_weights,
                group_sizes_all,
            )
            return output, metrics
        return output, router_logits

    def _replicate_and_sort_tokens(
        self,
        inputs_flat: jax.Array,
        selected_experts: jax.Array,
        use_custom_sort_vjp: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Replicates tokens for each selected expert and sorts them by expert ID.

        This method handles the case where each token can be routed to multiple experts
        (k > 1). It replicates each token k times (once for each selected expert) and
        then sorts all token-expert pairs by expert ID to group tokens by expert.

        Args:
            inputs_flat: Flattened input tokens. Shape: (batch_size * seq_len, hidden_dim).
            selected_experts: Expert indices selected for each token.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            use_custom_sort_vjp: Whether to use custom VJP for sorting operations.
                Defaults to True for memory efficiency.

        Returns:
            A tuple containing:
                - sorted_inputs: Replicated and sorted tokens.
                    Shape: (batch_size * seq_len * num_experts_per_tok, hidden_dim).
                - sort_order: Indices used for sorting, needed for unpermutation.
                    Shape: (batch_size * seq_len * num_experts_per_tok,).
                - group_sizes: Number of tokens assigned to each expert.
                    Shape: (num_experts,).
                - sorted_experts: Expert IDs after sorting.
                    Shape: (batch_size * seq_len * num_experts_per_tok,).

        Example:
            >>> # Route 100 tokens to top-2 experts each
            >>> inputs = jnp.ones((100, 768))
            >>> experts = jnp.array([[0, 2], [1, 3], ...])  # shape: (100, 2)
            >>> sorted_inputs, order, sizes, sorted_exp = self._replicate_and_sort_tokens(
            ...     inputs, experts)
            >>> # sorted_inputs.shape = (200, 768)  # 100 tokens * 2 experts each
        """
        k = selected_experts.shape[-1]
        flat_idx = selected_experts.reshape(-1)
        sort_order = jnp.argsort(flat_idx)
        base_indices = sort_order // k
        replicated = jnp.reshape(
            jnp.broadcast_to(inputs_flat[None, ...], (k, *inputs_flat.shape)),
            (k * inputs_flat.shape[0], inputs_flat.shape[1]),
        )
        sorted_inputs = _sort_activations(replicated, base_indices, use_custom_sort_vjp)
        group_sizes = jnp.bincount(flat_idx, length=self.n_routed_experts)
        sorted_experts = jnp.take(flat_idx, sort_order)
        return sorted_inputs, sort_order, group_sizes, sorted_experts

    def _apply_capacity_mask(
        self,
        topk_indices: jax.Array,
        weights: jax.Array,
        capacity_factor: float,
    ) -> jax.Array:
        """Applies capacity constraints to expert assignments by masking overflow tokens.

        This method implements capacity constraints to prevent any single expert from
        being overwhelmed with too many tokens. Tokens that exceed an expert's capacity
        are masked out (their weights are set to zero).

        Args:
            topk_indices: Indices of selected experts for each token.
                Shape: (batch_size, seq_len, num_experts_per_tok).
            weights: Weights for the selected experts.
                Shape: (batch_size, seq_len, num_experts_per_tok).
            capacity_factor: Factor to determine expert capacity.
                capacity = capacity_factor * (total_tokens / num_experts).

        Returns:
            Modified weights with capacity constraints applied. Tokens exceeding
            capacity have their weights set to zero.
            Shape: (batch_size, seq_len, num_experts_per_tok).

        Note:
            The capacity is enforced using a cumulative sum approach. Tokens are
            processed in order, and once an expert reaches capacity, subsequent
            tokens assigned to it are masked out.

        Example:
            >>> # Apply capacity with factor 1.5 (50% overhead)
            >>> indices = jnp.array([[[0, 1], [0, 2], [0, 3]]])  # 3 tokens, top-2 each
            >>> weights = jnp.ones((1, 3, 2))
            >>> capped_weights = self._apply_capacity_mask(indices, weights, 1.5)
        """
        B, S, k = topk_indices.shape
        tokens_per_batch = S * k
        cap = int(max(jnp.ceil(tokens_per_batch / self.n_routed_experts) * capacity_factor, capacity_factor))
        expert_mask = jax.nn.one_hot(topk_indices, num_classes=self.n_routed_experts, dtype=jnp.int32)
        fused = expert_mask.reshape(B, S * k, self.n_routed_experts)
        counts = jnp.cumsum(fused, axis=1)
        counts = counts.reshape(B, S, k, self.n_routed_experts)
        keep = (counts <= cap).astype(weights.dtype)
        keep_for_slot = jnp.sum(keep, axis=-1)
        return weights * keep_for_slot

    def _expert_group_mask(self, gate_logits: jax.Array, n_groups: int, topk_groups: int) -> jax.Array:
        """Creates a mask for expert groups based on top-k group selection.

        This method implements hierarchical routing where experts are organized into
        groups. First, the top-k groups are selected based on aggregated scores, then
        only experts within those groups can be selected. This reduces computational
        cost by limiting the routing search space.

        Args:
            gate_logits: Router logits for all experts.
                Shape: (batch_size * seq_len, num_experts).
            n_groups: Number of expert groups to divide experts into.
                Must evenly divide num_experts.
            topk_groups: Number of top groups to select for each token.

        Returns:
            Binary mask where 1 indicates an expert is in a selected group
            and 0 indicates it should be masked out.
            Shape: (batch_size * seq_len, num_experts).

        Example:
            >>> # 8 experts divided into 4 groups, select top 2 groups
            >>> logits = jnp.randn(100, 8)  # 100 tokens, 8 experts
            >>> mask = self._expert_group_mask(logits, n_groups=4, topk_groups=2)
            >>> # mask will have 1s for experts in the top-2 groups, 0s elsewhere

        Note:
            Group scores are computed as the sum of top-2 expert scores within
            each group, promoting groups with multiple strong experts.
        """
        BS = gate_logits.shape[0]
        experts_per_group = self.n_routed_experts // n_groups
        scores_grouped = gate_logits.reshape(BS, n_groups, experts_per_group)
        top2_vals, _ = jax.lax.top_k(scores_grouped, k=2)
        group_scores = jnp.sum(top2_vals.astype(jnp.float32), axis=-1)
        _, group_idx = jax.lax.top_k(group_scores, k=topk_groups)
        mask_groups = jax.nn.one_hot(group_idx, num_classes=n_groups, dtype=jnp.float32).sum(axis=-2)
        mask = jnp.broadcast_to(mask_groups[..., None], (BS, n_groups, experts_per_group)).reshape(BS, -1)
        return mask

    def _permute(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Permutes tokens to group them by their assigned expert.

        This operation is crucial for efficient batch computation by the experts.
        It sorts the tokens based on their expert index, allowing for a single
        batched computation per expert.

        Args:
            hidden_states_flat: A flattened array of input tokens with shape
                `(batch_size * seq_len, hidden_size)`.
            topk_idx_flat: A flattened array of expert indices for each token-
                expert pair, with shape `(batch_size * seq_len * num_experts_per_tok,)`.

        Returns:
            A tuple containing:
                - x_repeat_sort: The permuted hidden states, ready for expert
                    computation.
                - group_sizes: An array indicating how many tokens are assigned to
                    each expert.
                - sort_idx: The indices used for sorting, needed for the un-permutation step.
        """
        return self._permute_sharded(hidden_states_flat, topk_idx_flat)

    def _permute_sharded(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Performs a sharded permutation of tokens with improved partitioning.

        Hidden states: (batch * seq_len, hidden_size)
        Partitioned as: ((dp, fsdp), sp) for batch/seq, tp for hidden dimension
        """
        pmag = self.partition_manager

        if hidden_states_flat.ndim == 2:
            x_in_specs = pmag.resolve(axes=[BATCH, TP], mode=MODE_TRAIN, shape=hidden_states_flat.shape)
        else:
            x_in_specs = pmag.resolve(axes=[BATCH, EMPTY, TP], mode=MODE_TRAIN, shape=hidden_states_flat.shape)

        idx_in_specs = pmag.resolve(axes=[BATCH], mode=MODE_TRAIN, shape=topk_idx_flat.shape)

        x_out_specs = pmag.resolve(
            axes=[BATCH, TP],
            mode=MODE_TRAIN,
            shape=(topk_idx_flat.shape[0], hidden_states_flat.shape[-1]),
        )
        gs_out_specs = pmag.resolve(axes=[EXPERT], mode=MODE_TRAIN, shape=(self.n_routed_experts,))
        sortidx_out_specs = pmag.resolve(axes=[BATCH], mode=MODE_TRAIN, shape=topk_idx_flat.shape)

        batch_axis_names = tuple(n for n in getattr(self.mesh, "axis_names", ()) if n in ("dp", "fsdp"))

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(x_in_specs, idx_in_specs),
            out_specs=(x_out_specs, gs_out_specs, sortidx_out_specs),
            check_vma=False,
        )
        def permute_sharded(x_flat_: jax.Array, topk_idx_flat_: jax.Array):
            x_repeat_sort, group_sizes_local, sort_idx = self._permute_local(x_flat_, topk_idx_flat_)
            for ax in batch_axis_names:
                group_sizes_local = jax.lax.psum(group_sizes_local, axis_name=ax)
            return x_repeat_sort, group_sizes_local, sort_idx

        return permute_sharded(hidden_states_flat, topk_idx_flat)

    def _permute_local(
        self,
        hidden_states_flat: jax.Array,
        topk_idx_flat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Implements the permutation logic on a local device shard.

        Args:
            hidden_states_flat: A shard of flattened hidden states.
            topk_idx_flat: A shard of flattened expert indices.

        Returns:
            A tuple containing the locally permuted hidden states, group sizes,
            and sorting indices.
        """
        sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
        x_repeat_sort = jnp.take(hidden_states_flat, sort_idx // self.num_experts_per_tok, axis=0)
        group_sizes = jnp.bincount(topk_idx_flat, length=self.n_routed_experts)
        return x_repeat_sort, group_sizes, sort_idx

    def _unpermute(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Restores the original order of tokens after expert processing.

        This is the inverse operation of `_permute`.

        Args:
            out_repeat_sort: The output from the experts, in sorted (permuted)
                order.
            sort_idx: The sorting indices generated by the `_permute` step.
            original_shape: The original shape of the hidden states before
                flattening, typically `(batch_size, seq_len, hidden_size)`.

        Returns:
            The un-permuted expert outputs, reshaped to match the input structure.
        """
        return self._unpermute_sharded(out_repeat_sort, sort_idx, original_shape)

    def _unpermute_sharded(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Performs a sharded un-permutation of tokens with auto-sharding."""
        pmag = self.partition_manager

        if out_repeat_sort.ndim == 2:
            out_in_specs = pmag.resolve(axes=[BATCH, TP], mode=MODE_TRAIN, shape=out_repeat_sort.shape)
        else:
            out_in_specs = pmag.resolve(axes=[BATCH, EMPTY, TP], mode=MODE_TRAIN, shape=out_repeat_sort.shape)

        idx_in_specs = pmag.resolve(axes=[BATCH], mode=MODE_TRAIN, shape=sort_idx.shape)

        batch_size, seq_len, _hidden_size = original_shape
        out_dim = out_repeat_sort.shape[-1]
        output_shape = (batch_size * seq_len, self.num_experts_per_tok, out_dim)

        out_specs = pmag.resolve(axes=[BATCH, EMPTY, TP], mode=MODE_TRAIN, shape=output_shape)

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(out_in_specs, idx_in_specs),
            out_specs=out_specs,
            check_vma=False,
        )
        def unpermute_sharded(out_repeat_sort_: jax.Array, sort_idx_: jax.Array):
            return self._unpermute_local(out_repeat_sort_, sort_idx_, original_shape)

        return unpermute_sharded(out_repeat_sort, sort_idx)

    def _unpermute_local(
        self,
        out_repeat_sort: jax.Array,
        sort_idx: jax.Array,
        original_shape: tuple[int, ...],
    ) -> jax.Array:
        """Implements the un-permutation logic on a local device shard.

        Args:
            out_repeat_sort: A shard of expert outputs in sorted order.
            sort_idx: A shard of sorting indices.
            original_shape: The original shape of the hidden states.

        Returns:
            The locally un-permuted and reshaped expert outputs.
        """

        out_repeat = jnp.take(out_repeat_sort, jnp.argsort(sort_idx), axis=0)
        out_dim = out_repeat.shape[-1]
        return jnp.reshape(out_repeat, (-1, self.num_experts_per_tok, out_dim))

    def _compute_load_balancing_loss(
        self,
        router_probs: jax.Array,
        expert_loads: jax.Array,
        strategy: MoeLoadBalancingStrategy | None = None,
    ) -> float | None:
        """Computes the load balancing auxiliary loss.

        This loss encourages the router to distribute tokens evenly across all
        experts, preventing situations where some experts are overloaded while
        others are underutilized.

        Args:
            router_probs: The probabilities output by the router.
            expert_loads: The number of tokens assigned to each expert.
            strategy: The load balancing strategy to use. If None, the default
                strategy from the constructor is used.

        Returns:
            The computed load balancing loss as a scalar float, or None if the
            strategy is `NONE` or the loss coefficient is not set.

        Raises:
            ValueError: If an unknown load balancing strategy is provided.
        """
        strategy = strategy or self.load_balancing_strategy

        if strategy == MoeLoadBalancingStrategy.NONE or self.lbl_coef is None:
            return None

        if strategy == MoeLoadBalancingStrategy.STANDARD:
            f = expert_loads * self.n_routed_experts / self.num_experts_per_tok
            p = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * jnp.sum(f * p)

        elif strategy == MoeLoadBalancingStrategy.SWITCH_TRANSFORMER:
            num_tokens = router_probs.shape[0]
            expert_fraction = expert_loads / num_tokens
            router_fraction = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * self.n_routed_experts * jnp.sum(expert_fraction * router_fraction)

        elif strategy == MoeLoadBalancingStrategy.EMPTY_CHOICE:
            return self.lbl_coef * jnp.var(expert_loads)

        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

    def _compute_router_z_loss(self, router_logits: Float[Array, "batch_seq num_experts"]) -> float | None:
        """Computes the router z-loss.

        This auxiliary loss encourages the router to produce logits with small
        magnitudes, which can improve training stability. It is calculated as the
        mean of the squared log-sum-exp of the router logits.

        Args:
            router_logits: The raw logits produced by the router.

        Returns:
            The computed router z-loss as a scalar float, or None if the loss
            coefficient is not set.
        """
        if self.rzl_coef is None:
            return None

        log_z = jax.nn.logsumexp(router_logits, axis=-1)
        return self.rzl_coef * jnp.mean(log_z**2)

    def _compute_metrics(
        self,
        router_logits: jax.Array,
        router_probs: jax.Array,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        expert_loads: jax.Array,
    ) -> MoeMetrics:
        """Computes and aggregates all MoE-related metrics.

        This method consolidates the calculation of various metrics and auxiliary
        losses into a single `MoeMetrics` object. These metrics are essential for
        monitoring the health of the MoE layer during training and inference.

        Args:
            router_logits: The raw logits from the router before softmax.
                Shape: (batch_size * seq_len, num_experts).
            router_probs: The probabilities from the router after softmax.
                Shape: (batch_size * seq_len, num_experts).
            selected_experts: The indices of the chosen experts for each token.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            selected_weights: The weights for the chosen experts.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            expert_loads: The number of tokens assigned to each expert.
                Shape: (num_experts,).

        Returns:
            An `MoeMetrics` object containing:
                - expert_loads: Token counts per expert
                - router_probs: Router probability distribution
                - selected_experts: Chosen expert indices
                - selected_weights: Weights for chosen experts
                - load_balancing_loss: Auxiliary loss for balanced routing
                - router_z_loss: Auxiliary loss for stable routing
                - expert_utilization: Fraction of experts used
                - routing_entropy: Entropy of routing decisions

        Example:
            >>> metrics = self._compute_metrics(
            ...     router_logits, router_probs, selected_experts,
            ...     selected_weights, expert_loads)
            >>> print(f"Load balancing loss: {metrics.load_balancing_loss}")
            >>> print(f"Expert utilization: {metrics.expert_utilization}")
        """
        metrics = MoeMetrics(
            expert_loads=expert_loads,
            router_probs=router_probs,
            selected_experts=selected_experts,
            selected_weights=selected_weights,
        )
        metrics.load_balancing_loss = self._compute_load_balancing_loss(router_probs, expert_loads)
        metrics.router_z_loss = self._compute_router_z_loss(router_logits)
        metrics.expert_utilization = jnp.mean(expert_loads > 0)
        metrics.routing_entropy = jnp.mean(-jnp.sum(router_probs * jnp.log(router_probs + 1e-8), axis=-1))
        return metrics

    def _apply_expert_sharding(self, tensor: Float[Array, ...], tensor_type: str = "weight") -> Float[Array, ...]:
        """Applies expert parallel sharding to a tensor with auto-sharding.

        This method configures the sharding specification for expert-related tensors
        to ensure they are properly distributed across devices in expert parallel
        configurations.

        Args:
            tensor: The tensor to be sharded. Can be weights, biases, or activations.
                Common shapes:
                - Weights: (num_experts, in_features, out_features)
                - Biases: (num_experts, out_features)
                - Activations: (batch, seq_len, hidden_dim)
            tensor_type: Type of tensor for specialized sharding patterns.
                Options: "weight", "weight_col", "weight_row", "bias".

        Returns:
            The tensor with appropriate sharding specification applied.

        Note:
            The sharding pattern depends on the tensor type and shape:
            - "weight_col": Column-wise partitioning for output parallelism
            - "weight_row": Row-wise partitioning for input parallelism
            - "bias": Follows the output dimension sharding
            - "weight" (default): Standard expert-wise sharding
        """
        pmag = self.partition_manager

        if tensor_type == "weight_col":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "weight_row":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "bias":
            if tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=tensor.shape)

        else:
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            elif tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape)
            else:
                sharding_spec = pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        return jax.device_put(tensor, jax.sharding.NamedSharding(self.mesh, sharding_spec))

    def _get_gate_layer_sharding(self, weight_shape: tuple) -> PartitionSpec:
        """Returns the partition spec for the gate/router layer weights.

        Args:
            weight_shape: Shape of the gate layer weights, typically
                (hidden_size, num_experts).

        Returns:
            PartitionSpec defining how to shard the gate weights across devices.
        """
        pmag = self.partition_manager
        return pmag.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=weight_shape)

    def _get_gate_layer_bias_sharding(self, bias_shape: tuple) -> PartitionSpec:
        """Returns the partition spec for the gate/router layer bias.

        Args:
            bias_shape: Shape of the gate layer bias, typically (num_experts,).

        Returns:
            PartitionSpec defining how to shard the gate bias across devices.
        """
        pmag = self.partition_manager
        return pmag.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=bias_shape)

    def _validate_routing_inputs(
        self, hidden_states: Float[Array, "batch seq hidden_dim"], router_logits: Float[Array, "batch_seq num_experts"]
    ) -> None:
        """Validates the shapes of inputs for routing operations.

        This method performs sanity checks to ensure that the input tensors have
        the expected shapes based on the module's configuration. It helps catch
        configuration mismatches early.

        Args:
            hidden_states: The input tensor to the MoE layer.
                Expected shape: (batch_size, seq_len, hidden_size).
            router_logits: The logits produced by the router.
                Expected shape: (batch_size * seq_len, num_experts).

        Raises:
            ValueError: If any of the following conditions are met:
                - Hidden dimension doesn't match config.hidden_size
                - Router output dimension doesn't match config.n_routed_experts
                - Batch dimensions are inconsistent between inputs

        Example:
            >>> # This will raise ValueError if shapes don't match config
            >>> self._validate_routing_inputs(hidden_states, router_logits)
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input hidden dimension {hidden_states.shape[-1]} doesn't "
                f"match config hidden dimension {self.hidden_size}"
            )

        if router_logits.shape[-1] != self.n_routed_experts:
            raise ValueError(
                f"Router logits expert dimension {router_logits.shape[-1]} doesn't match "
                f"config expert count {self.n_routed_experts}"
            )

        if router_logits.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]:
            raise ValueError(
                f"Router logits batch dimension {router_logits.shape[0]} doesn't match "
                f"flattened input batch dimension {hidden_states.shape[0] * hidden_states.shape[1]}"
            )

    def _apply_capacity_constraint(
        self,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        capacity_factor: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Applies a capacity constraint to expert routing.

        This method implements a soft capacity constraint that limits the number of
        tokens each expert can process. Unlike hard capacity constraints that drop
        tokens, this implementation scales down weights for over-capacity experts
        and renormalizes to maintain the sum of weights.

        Args:
            selected_experts: The indices of the selected experts for each token.
                Shape: (num_tokens, num_experts_per_tok).
            selected_weights: The weights for the selected experts.
                Shape: (num_tokens, num_experts_per_tok).
            capacity_factor: The factor to determine the maximum capacity.
                max_capacity = capacity_factor * num_tokens / num_experts.
                Common values: 1.0 (balanced), 1.25 (25% overhead), 2.0 (2x overhead).
                If None, defaults to 1.0.

        Returns:
            A tuple containing:
                - constrained_experts: The expert indices (unchanged).
                - constrained_weights: The weights adjusted for capacity and
                    renormalized to sum to 1 for each token.

        Example:
            >>> # Apply capacity constraint with 25% overhead
            >>> experts = jnp.array([[0, 1], [0, 2]])  # 2 tokens, top-2 experts
            >>> weights = jnp.array([[0.6, 0.4], [0.7, 0.3]])
            >>> new_experts, new_weights = self._apply_capacity_constraint(
            ...     experts, weights, capacity_factor=1.25)

        Note:
            The constraint is applied globally across all tokens. Experts that
            exceed capacity have their weights scaled by 1/over_capacity_ratio.
        """
        if capacity_factor is None:
            capacity_factor = 1.0
        num_tokens = selected_experts.shape[0]
        max_capacity = int(capacity_factor * num_tokens / self.n_routed_experts)
        expert_counts = jnp.bincount(selected_experts.flatten(), length=self.n_routed_experts)
        over_capacity_ratio = jnp.maximum(expert_counts / max_capacity, 1.0)
        weight_adjustments = 1.0 / over_capacity_ratio[selected_experts]
        constrained_weights = selected_weights * weight_adjustments
        weight_sum = jnp.sum(constrained_weights, axis=-1, keepdims=True)
        constrained_weights = jnp.where(weight_sum > 0, constrained_weights / weight_sum, constrained_weights)
        return selected_experts, constrained_weights

    def _create_expert_mask(
        self, selected_experts: Int[Array, "batch_seq k"], expert_id: int
    ) -> Bool[Array, "batch_seq k"]:
        """Creates a boolean mask for tokens assigned to a specific expert.

        This utility method is useful for analyzing expert assignment patterns
        or implementing expert-specific operations.

        Args:
            selected_experts: An array of selected expert indices for each token.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            expert_id: The ID of the expert for which to create the mask.
                Must be in range [0, num_experts).

        Returns:
            A boolean array where True indicates that the token is routed to
            the specified expert. Shape: (batch_size * seq_len,).

        Example:
            >>> # Check which tokens are routed to expert 3
            >>> selected = jnp.array([[0, 1], [2, 3], [3, 4]])  # 3 tokens
            >>> mask = self._create_expert_mask(selected, expert_id=3)
            >>> # mask will be [False, True, True]

        Note:
            If a token is routed to the same expert multiple times (rare but
            possible), it will still only appear once in the mask.
        """
        return jnp.any(selected_experts == expert_id, axis=-1)

    def _moe_call_fused_shard_map(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_kernel: jax.Array,  # [H, E]
        wi_kernel: jax.Array,  # [E, H, M]
        wu_kernel: jax.Array,  # [E, H, M]
        wd_kernel: jax.Array,  # [E, M, H]
        act_fn: Callable[[jax.Array], jax.Array],
        *,
        output_metrics: bool = False,
        batch_sharded_by_expert: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Executes a fully-fused MoE forward pass with advanced sharding and hook support.

        This is the most optimized MoE implementation in EasyDeL, combining router computation,
        expert-parallel communication, expert computation, and output combination into a single
        fused operation with shard_map. It supports:
        - Split-K contracting dimension across SP/FSDP axes
        - Tensor-parallel (TP) sharding with reduce-scatter
        - Expert-parallel (EP) distribution with ragged all-to-all
        - Custom hook injection at every execution stage
        - Configurable grouped matrix multiplication backends

        The execution flow:
        1. Compute router logits with optional K-dimension splitting and TP gathering
        2. Apply softmax and perform top-k expert selection
        3. Replicate tokens by k, sort by expert ID, compute group sizes
        4. Dispatch tokens to expert-owning devices via ragged all-to-all (EP)
        5. Locally permute tokens to group by expert
        6. Execute expert wi/wu projections with split-K and GLU activation
        7. Execute expert output projection (wo) with TP reduce-scatter
        8. Reverse all-to-all to return expert outputs to original devices
        9. Unsort and combine expert outputs with routing weights

        Args:
            hidden_state: Input activations. Shape: [batch, seq_len, hidden_size].
            gate_kernel: Router weight matrix. Shape: [hidden_size, num_experts].
            wi_kernel: Expert input projection weights. Shape: [num_experts, hidden_size, intermediate_size].
            wu_kernel: Expert up projection weights for GLU. Shape: [num_experts, hidden_size, intermediate_size].
            wd_kernel: Expert output projection weights. Shape: [num_experts, intermediate_size, hidden_size].
            act_fn: Activation function to apply in the expert FFN (e.g., jax.nn.silu).
                Takes a single array and returns the activated array.
            output_metrics: If True, returns MoeMetrics with detailed statistics.
                If False, returns router logits for auxiliary loss computation.
            batch_sharded_by_expert: Currently unused in this implementation.
                Reserved for future batch-sharded EP strategies.

        Returns:
            A tuple containing:
                - output: Processed hidden states. Shape: [batch, seq_len, hidden_size].
                - metrics_or_logits: Either router logits [batch, seq_len, num_experts]
                    or the full logits reshaped to [batch, seq_len, num_experts].

        Note:
            This method requires:
            - An 'ep' or 'expert' axis in the mesh for expert parallelism
            - Properly configured moe_policy and moe_hooks (optional)
            - Compatible sharding specifications in partition_manager

        Example:
            >>> # Execute fused MoE with custom policy and hooks
            >>> output, logits = self._moe_call_fused_shard_map(
            ...     hidden_state=inputs,
            ...     gate_kernel=self.gate.kernel.value,
            ...     wi_kernel=self.experts.wi.kernel.value,
            ...     wu_kernel=self.experts.wu.kernel.value,
            ...     wd_kernel=self.experts.wo.kernel.value,
            ...     act_fn=jax.nn.silu,
            ...     output_metrics=False
            ... )

        Raises:
            AssertionError: If num_experts is not evenly divisible by EP size.
        """
        mesh = self.mesh
        E = wi_kernel.shape[0]
        K = self.num_experts_per_tok

        policy = getattr(self, "moe_policy", MoeFusedPolicy())
        hooks = getattr(self, "moe_hooks", MoeFusedHooks())

        pm = self.partition_manager
        x_spec = pm.resolve([DP, EMPTY, TP], mode=MODE_TRAIN)  # [B, S, H]
        w_gate_sp = pm.resolve([[SP, FSDP], TP], mode=MODE_TRAIN)  # [H, E_shard]
        wi_sp = pm.resolve([EP, [SP, FSDP], TP], mode=MODE_TRAIN)  # [E, H, M_shard]
        wu_sp = pm.resolve([EP, [SP, FSDP], TP], mode=MODE_TRAIN)  # [E, H, M_shard]
        wd_sp = pm.resolve([EP, TP, EMPTY], mode=MODE_TRAIN)  # [E, M_shard, H]
        out_spec = pm.resolve([DP, EMPTY, TP], mode=MODE_TRAIN)  # [B, S, H_shard]
        rx = pm.paxis.resolve_axis

        def _as_tuple_axes(ax):
            if isinstance(ax, tuple | list):
                return tuple(ax)
            elif ax is None:
                return ()
            else:
                return (ax,)

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(x_spec, w_gate_sp, wi_sp, wu_sp, wd_sp),
            out_specs=(out_spec, pm.resolve([DP, EMPTY, EMPTY], mode=MODE_TRAIN)),
            check_vma=False,
        )
        def _fused(x, w_gate, w0, w1, w2):
            # Local dynamic sizes (handles nnx micro-batching)
            B_loc, S_loc, H_shard = x.shape
            bs = B_loc * S_loc

            # Axis names/sizes
            axis_sizes = {n: mesh.shape[n] for n in mesh.axis_names}
            tp_axis = policy.tp_axis
            combine_axes = _as_tuple_axes(policy.combine_axes)
            gate_k_axes = _as_tuple_axes(policy.gate_slice_k_axes)
            wiwu_k_axes = _as_tuple_axes(policy.wiwu_slice_k_axes)

            # 1) Router logits on full H/E
            x = x.astype(self.dtype)
            x_flat = x.reshape(bs, H_shard)  # [bs, H/tp]
            x_fullH = lax.all_gather(x_flat, tp_axis, axis=-1, tiled=True) if policy.gather_gate_on_tp else x_flat
            H_full = x_fullH.shape[1]

            ctx = MoeContext(
                mesh=mesh,
                axis_names=mesh.axis_names,
                axis_sizes=axis_sizes,
                B_loc=B_loc,
                S_loc=S_loc,
                H_full=H_full,
                K=K,
                E=E,
            )
            if hooks.before_gate:
                x_fullH, w_gate = hooks.before_gate(x_fullH, w_gate, ctx)

            # Slice activations to local K-chunk if gate weight shards K across (sp,fsdp)
            k_chunk_gate = w_gate.shape[0]
            x_gate_k = (
                _slice_k_for_param_shards(x_fullH, k_chunk_gate, gate_k_axes, mesh, axis=1)
                if k_chunk_gate != H_full
                else x_fullH
            )

            logits_partial = x_gate_k @ w_gate  # [bs, E_shard]
            logits_full = _psum_maybe(logits_partial, combine_axes, mesh, dtype=policy.reduce_dtype)
            if policy.gather_logits_on_tp:
                logits_full = lax.all_gather(logits_full, tp_axis, axis=-1, tiled=True)  # [bs, E]
            router_logits = logits_full.astype(jnp.float32)
            if hooks.after_gate:
                router_logits = hooks.after_gate(router_logits, ctx)

            # 2) Routing (top-k)
            router_probs = jax.nn.softmax(router_logits, axis=-1).astype(self.dtype)
            if hooks.before_topk:
                router_probs = hooks.before_topk(router_probs, ctx)
            topk_weights, topk_idx = lax.top_k(router_probs, K)
            if getattr(self.config, "norm_topk_prob", False):
                topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
            if hooks.after_topk:
                topk_weights, topk_idx = hooks.after_topk(topk_weights, topk_idx, ctx)

            # 3) Replicate and sort by expert, compute global group sizes
            flat_idx = topk_idx.reshape(-1)  # [bs*K]
            sort_idx = jnp.argsort(flat_idx)  # [bs*K]
            x_rep_sorted = jnp.take(x_fullH, sort_idx // K, axis=0)  # [bs*K, H]
            group_sizes = jnp.bincount(flat_idx, length=E)  # [E]
            if hooks.before_ep_dispatch:
                x_rep_sorted, group_sizes = hooks.before_ep_dispatch(x_rep_sorted, group_sizes, ctx)

            # 4) Dispatch to EP via ragged_all_to_all
            ep_axis = rx([EP], mode=MODE_TRAIN)[0]
            ep_size = axis_sizes[ep_axis]
            assert E % ep_size == 0
            local_E = E // ep_size
            ep_idx = lax.axis_index(ep_axis)

            reshaped_gs = jnp.sum(group_sizes.reshape(-1, local_E), axis=1)  # [ep_size]
            in_off, send_sz, out_off, recv_sz = _get_all_to_all_params(
                reshaped_gs, ep_idx, ep_size, is_batch_sharded=False
            )
            in_off = in_off.astype(jnp.int32)
            send_sz = send_sz.astype(jnp.int32)
            out_off = out_off.astype(jnp.int32)
            recv_sz = recv_sz.astype(jnp.int32)

            buf_len = int(ep_size * bs * K)  # conservative buffer per call
            x_buf = jnp.zeros((buf_len, H_full), dtype=x_rep_sorted.dtype)
            x_local = lax.ragged_all_to_all(x_rep_sorted, x_buf, in_off, send_sz, out_off, recv_sz, axis_name=ep_axis)

            # 5) Local permute: derive local ids from local sizes (no global offsets)
            x_local, local_sorted_idx, local_group_sizes, _ = _local_permute(
                x_local,
                group_sizes[None, :],  # [1, E]
                local_E,
                shard_index=ep_idx,
                is_offset=False,
                global_sorted_experts=None,
                use_custom_sort_vjp=True,
            )
            local_group_sizes = local_group_sizes.astype(jnp.int32)
            if hooks.after_ep_receive:
                x_local, local_group_sizes = hooks.after_ep_receive(x_local, local_group_sizes, ctx)

            # 6) Experts: wi/up with psum over combine_axes; slice activations to local k-chunk if needed
            k_chunk_w = w0.shape[1]  # local k per device for w0/w1 across (sp,fsdp)
            x_k = (
                x_local
                if k_chunk_w == x_local.shape[1]
                else _slice_k_for_param_shards(x_local, k_chunk_w, wiwu_k_axes, mesh, axis=1)
            )
            if hooks.before_wiwu:
                x_k, w0, w1, local_group_sizes = hooks.before_wiwu(x_k, w0, w1, local_group_sizes, ctx)

            y0 = grouped_matmul(
                x_k, w0, local_group_sizes, preferred_element_type=jnp.bfloat16, platform=policy.gmm_platform
            )
            y0 = _psum_maybe(y0, combine_axes, mesh, dtype=policy.reduce_dtype)
            y1 = grouped_matmul(
                x_k, w1, local_group_sizes, preferred_element_type=jnp.bfloat16, platform=policy.gmm_platform
            )
            y1 = _psum_maybe(y1, combine_axes, mesh, dtype=policy.reduce_dtype)
            inter = hooks.after_wiwu(y0, y1, ctx) if hooks.after_wiwu else (act_fn(y0) * y1)  # default: GLU

            # 7) wo: reduce-scatter over TP on the last axis
            if hooks.before_wo:
                inter, w2 = hooks.before_wo(inter, w2, ctx)
            y2 = grouped_matmul(
                inter,
                w2,
                local_group_sizes,
                preferred_element_type=jnp.bfloat16,
                platform=policy.gmm_platform,
            )
            if policy.rs_enabled:
                y2 = _rsum_scatter_maybe(y2, policy.tp_axis, policy.rs_dim, mesh, dtype=policy.reduce_dtype)
            if hooks.after_wo:
                y2 = hooks.after_wo(y2, ctx)

            # Undo local sort
            y_local_unsorted = _sort_activations(y2, jnp.argsort(local_sorted_idx), use_custom_vjp=True)

            # 8) Return from EP (inverse ragged_all_to_all)
            y_back = lax.ragged_all_to_all(
                y_local_unsorted,
                jnp.zeros((x_rep_sorted.shape[0], y_local_unsorted.shape[-1]), dtype=y_local_unsorted.dtype),
                in_off,
                send_sz,
                out_off,
                recv_sz,
                axis_name=ep_axis,
            )  # [bs*K, H/tp]

            # 9) Unsort to token order and combine with weights
            y_unsorted = _sort_activations(y_back, jnp.argsort(sort_idx), use_custom_vjp=True)  # [bs*K, H/tp]
            y_unflat = y_unsorted.reshape(bs, K, -1)  # [bs, K, H/tp]
            if hooks.before_combine:
                y_unflat, topk_weights = hooks.before_combine(y_unflat, topk_weights, ctx)
            out = jnp.sum(y_unflat * topk_weights[..., None].astype(y_unflat.dtype), axis=1)  # [bs, H/tp]
            out = out.reshape(B_loc, S_loc, -1)  # [B_loc, S_loc, H/tp]

            # Finalize
            logits_bsh = router_logits.reshape(B_loc, S_loc, -1)
            if hooks.finalize_output:
                out, logits_bsh = hooks.finalize_output(out, logits_bsh, ctx)
            return out, logits_bsh

        return _fused(hidden_state, gate_kernel, wi_kernel, wu_kernel, wd_kernel)

    def _moe_call(
        self,
        gate_layer: nn.Module,
        expert_layer: nn.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        apply_capacity_constraint: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """A generic forward pass implementation for a standard MoE block.

        This method orchestrates the entire MoE process: routing, permutation,
        expert computation, un-permutation, and optional metric calculation.
        It serves as the main execution path for MoE layers without expert parallelism.

        The execution flow:
        1. Flatten input and compute router logits
        2. Apply softmax to get routing probabilities
        3. Select top-k experts for each token
        4. Apply optional capacity constraints
        5. Replicate and sort tokens by expert assignment
        6. Execute expert computation on grouped tokens
        7. Unsort and combine expert outputs using routing weights
        8. Compute optional metrics for monitoring

        Args:
            gate_layer: The module that acts as the router (e.g., a Linear layer).
                Should map from hidden_size to num_experts.
            expert_layer: The module containing the expert logic (e.g., `ParallelMoELinear`).
                Should accept (tokens, group_sizes) and return processed tokens.
            hidden_state: The input tensor of shape
                (batch_size, seq_len, hidden_size).
            output_metrics: If True, returns a `MoeMetrics` object along with the
                output. Otherwise, returns the router logits for auxiliary loss.
            validate_inputs: If True, validates the shapes of routing inputs
                to catch configuration errors early.
            apply_capacity_constraint: If True, applies capacity constraints to
                prevent expert overload. Uses soft constraints by default.
            reform_router_probs_fn: An optional function to apply to the router
                probabilities after the softmax. Can be used for custom routing
                logic or probability adjustments.

        Returns:
            A tuple containing:
                - output: The final output tensor of shape
                    (batch_size, seq_len, hidden_size).
                - metrics: If `output_metrics` is True, a `MoeMetrics` object
                    containing detailed routing statistics and auxiliary losses.
                    Otherwise, the raw router logits for external loss computation.

        Example:
            >>> # Standard MoE forward pass with metrics
            >>> output, metrics = self._moe_call(
            ...     gate_layer=self.gate,
            ...     expert_layer=self.experts,
            ...     hidden_state=inputs,
            ...     output_metrics=True,
            ...     apply_capacity_constraint=True
            ... )
            >>> total_loss = main_loss + metrics.load_balancing_loss

        Note:
            This method automatically applies logical sharding for distributed
            training using the partition manager from the config.
        """
        batch_size, seq_len, hidden_size = hidden_state.shape
        hidden_state_flat = hidden_state.reshape(-1, hidden_size)
        router_logits = gate_layer(hidden_state_flat).astype(jnp.promote_types(self.dtype, jnp.float32))
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        if reform_router_probs_fn is not None:
            router_probs = reform_router_probs_fn(router_probs)

        if validate_inputs:
            self._validate_routing_inputs(hidden_state, router_logits)

        selected_weights, selected_experts = self._route(router_probs)
        if apply_capacity_constraint:
            selected_experts, selected_weights = self._apply_capacity_constraint(selected_experts, selected_weights)

        sorted_inputs, sort_order, group_sizes, _ = self._replicate_and_sort_tokens(hidden_state_flat, selected_experts)
        out_sorted = expert_layer(sorted_inputs, group_sizes)
        out_unsorted = _sort_activations(out_sorted, jnp.argsort(sort_order))
        out_unflat = out_unsorted.reshape(batch_size * seq_len, self.num_experts_per_tok, hidden_size)
        output = jnp.sum(out_unflat * selected_weights[..., None], axis=1).reshape(batch_size, seq_len, hidden_size)
        if output_metrics:
            metrics = self._compute_metrics(
                router_logits,
                router_probs,
                selected_experts,
                selected_weights,
                group_sizes,
            )
            return output, metrics
        return output, router_logits

    @abstractmethod
    def __call__(
        self, hidden_states: Float[Array, "batch seq hidden_dim"], **kwargs
    ) -> tuple[Float[Array, "batch seq hidden_dim"], MoeMetrics]:
        """Performs the forward pass of the MoE module.

        Subclasses must implement this method to define the specific logic of their
        MoE layer.

        Args:
            hidden_states: The input tensor.
            **kwargs: Additional keyword arguments that may be required by the
                specific implementation.

        Returns:
            A tuple containing:
                - output: The output tensor from the MoE layer.
                - metrics: A `MoeMetrics` object containing metrics and auxiliary losses.
        """
        pass


class ParallelMoELinear(nn.Module):
    """A batched linear transformation layer for Mixture of Experts (MoE) models.

    This layer applies separate linear transformations for each expert in a MoE setup.
    The inputs are assumed to be sorted and grouped by expert, with `group_sizes`
    specifying how many tokens belong to each expert. It supports:
    - **Ragged Matrix Multiplication** via `jax.lax.ragged_dot_general`.
    - **Grouped Matrix Multiplication (GMM)** via a Pallas kernel for TPUs.

    Can optionally integrate with a `PartitionManager` to shard parameters and
    use `shard_map` for distributed execution.

    Attributes:
        num_experts (int): Number of experts.
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        use_pallas_group_matmul (bool): Whether to use the optimized GMM kernel (TPU-optimized).
        out_first (bool): If True, kernel shape is `(num_experts, out_features, in_features)`;
            otherwise `(num_experts, in_features, out_features)`.
        dtype (jax.numpy.dtype | None): Data type for computation.
        param_dtype (jax.numpy.dtype): Data type for parameters.
        kernel (nn.Param): Weight matrix parameter for the transformation.
        bias (nn.Param | None): Optional bias parameter.
        partition_manager (PartitionManager | None): Handles sharding of parameters.
        _direction (Literal["row", "column"] | None): Sharding direction for ALT sharding.
    """

    _direction: typing.Literal["row", "column"] | None = None

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        out_first: bool = False,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        use_pallas_group_matmul: bool = False,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        partition_manager: PartitionManager | None = None,
        direction: typing.Literal["row", "column"] | None = None,
        rngs: nn.Rngs,
    ):
        """Initializes a `ParallelMoELinear` layer.

        Args:
            num_experts: Number of experts in the layer.
            in_features: Size of the input feature dimension.
            out_features: Size of the output feature dimension.
            use_bias: Whether to include a bias term. Defaults to True.
            out_first: If True, kernel shape is `(num_experts, out_features, in_features)`,
                otherwise `(num_experts, in_features, out_features)`.
            kernel_init: Initializer for the kernel weights.
            bias_init: Initializer for the bias.
            use_pallas_group_matmul: Whether to use the TPU-optimized grouped matrix multiplication kernel.
            dtype: Data type for computation. Defaults to None (inherits from inputs).
            param_dtype: Data type for parameters (weights, biases).
            partition_manager: Partition manager for parameter sharding and mapping.
            direction: ALT-sharding direction, either `"row"`, `"column"`, or None.
            rngs: Random number generators for parameter initialization.
        """
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.use_pallas_group_matmul = use_pallas_group_matmul and (jax.default_backend() == "tpu")
        self.out_first = out_first
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.partition_manager = partition_manager

        self.kernel_init = kernel_init
        self.bias_init = bias_init

        if direction is not None:
            assert direction in ["row", "column"]
            self._direction = direction
        kshape = (num_experts, out_features, in_features) if out_first else (num_experts, in_features, out_features)
        self.kernel = nn.Param(kernel_init(rngs.param(), kshape, param_dtype))
        if use_bias:
            bshape = (num_experts, out_features)
            self.bias = nn.Param(bias_init(rngs.param(), bshape, self.param_dtype))
        else:
            self.bias = None

    @property
    def direction(self) -> typing.Literal["row", "column"] | None:
        """Returns the parallelism direction for this layer.

        Returns:
            "row" for row-wise parallelism (input dimension partitioned),
            "column" for column-wise parallelism (output dimension partitioned),
            or None if no parallelism direction is set.
        """
        return self._direction

    @property
    def can_use_shard_map(self) -> bool:
        """Checks if this layer can use shard_map for distributed execution.

        Returns:
            True if both a partition manager and parallelism direction are configured,
            indicating the layer is ready for distributed execution with shard_map.
        """
        return self.partition_manager is not None and self._direction is not None

    @property
    def alt_sharding(self) -> ExpertRowWiseAlt | ExpertColumnWiseAlt | None:
        """Returns the ALT (Alternative) sharding configuration for this layer.

        ALT sharding provides pre-defined sharding patterns for common parallelism
        strategies, simplifying the configuration of distributed execution.

        Returns:
            ExpertRowWiseAlt for row parallelism,
            ExpertColumnWiseAlt for column parallelism,
            or None if no direction is set.

        Raises:
            NotImplementedError: If an unsupported direction is configured.
        """
        if self.direction is None:
            return None
        elif self.direction == "row":
            return ExpertRowWiseAlt
        elif self.direction == "column":
            return ExpertColumnWiseAlt
        else:
            direction = self.direction
            raise NotImplementedError(f"ALT-Sharding Rule for {direction=} is not implemented!.")

    @property
    def alt_sharding_axis(self) -> list[str] | None:
        """Returns the axis names for ALT sharding configuration.

        Returns:
            List of axis names (e.g., ["expert", "tp", "dp"]) for the configured
            ALT sharding pattern, or None if no ALT sharding is configured.
        """
        if self.alt_sharding is None:
            return None
        return self.alt_sharding.axes

    def __call__(
        self,
        inputs: Float[Array, "tokens_ragged hidden_dim"],
        group_sizes: Int[Array, "num_groups"],  # noqa
    ) -> Float[Array, "tokens_ragged out_dim"]:
        """Applies the batched linear transformation.

        Args:
            inputs: The input array, which is a batch of tokens sorted and grouped
                by expert. Shape: `(total_tokens, in_features)`.
            group_sizes: An array indicating the number of tokens assigned to each
                expert. Shape: `(num_experts,)`.

        Returns:
            The output array after the linear transformation.
            Shape: `(total_tokens, out_features)`.
        """
        weight = self.kernel.value

        if weight.dtype in (
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        ):
            weight = weight.astype("f4")

        inputs, weight = promote_dtype((inputs, weight), dtype=self.dtype)

        output = grouped_matmul(
            inputs,
            weight,
            group_sizes,
            platform="xla",
            preferred_element_type=jnp.bfloat16,
            transpose_rhs=self.out_first,
        )

        if self.bias is not None:
            output += self._expand_bias_ragged(group_sizes)

        return output

    def _expand_bias_ragged(self, group_sizes: Int[Array, "num_groups"]) -> Float[Array, "tokens_ragged out_dim"]:  # noqa
        """Expands the bias to match the ragged batch structure.

        This method repeats the bias for each expert according to the number of
        tokens assigned to it. This is necessary because tokens are grouped by
        expert, and each group needs its corresponding expert's bias.

        Args:
            group_sizes: The sizes of token groups for each expert.
                Shape: (num_experts,). Each element indicates how many tokens
                are assigned to that expert.

        Returns:
            The expanded bias array where each expert's bias is repeated
            according to its group size. Shape: (total_tokens, out_features).

        Example:
            If expert 0 has 3 tokens, expert 1 has 2 tokens, and expert 2 has 4 tokens,
            this will repeat bias[0] 3 times, bias[1] 2 times, and bias[2] 4 times.
        """
        return self.bias.value[jnp.repeat(jnp.arange(self.num_experts), group_sizes)]


class RowParallelMoELinear(ParallelMoELinear):
    """Row-parallel variant of ParallelMoELinear.

    This class specializes ParallelMoELinear for row-wise parallelism, where the
    input dimension is partitioned across devices. In row parallelism, each device
    holds a subset of input features and computes partial results that are then
    reduced across devices.

    The weight matrix is partitioned along the input dimension (rows), and an
    all-reduce operation is performed after the matrix multiplication to combine
    partial results.

    Attributes:
        _direction: Fixed to "row" to indicate row-wise parallelism.

    Example:
        >>> # Create a row-parallel MoE linear layer
        >>> layer = RowParallelMoELinear(
        ...     num_experts=8,
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=rngs
        ... )
    """

    _direction: typing.Literal["row", "column"] | None = "row"


class ColumnParallelMoELinear(ParallelMoELinear):
    """Column-parallel variant of ParallelMoELinear.

    This class specializes ParallelMoELinear for column-wise parallelism, where the
    output dimension is partitioned across devices. In column parallelism, each device
    computes a subset of output features independently without requiring reduction.

    The weight matrix is partitioned along the output dimension (columns), and each
    device produces its portion of the output directly without communication.

    Attributes:
        _direction: Fixed to "column" to indicate column-wise parallelism.

    Example:
        >>> # Create a column-parallel MoE linear layer
        >>> layer = ColumnParallelMoELinear(
        ...     num_experts=8,
        ...     in_features=768,
        ...     out_features=3072,
        ...     rngs=rngs
        ... )
    """

    _direction: typing.Literal["row", "column"] | None = "column"
