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

"""Utilities for MoE routing, permutation, and distributed execution.

This module provides helper functions and small data containers used by the MoE
layers, including sorting utilities with custom VJP, communication parameter
computation for expert-parallel all-to-all, and convenience enums/contexts for
fused policies and metrics.
"""

from __future__ import annotations

import enum
import os
import typing
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
from eformer import common_types
from eformer.escale import PartitionManager
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

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

EP_DISPATCH = os.getenv("EP_DISPATCH", "auto")
EP_AUTO_TRESHOLD = int(os.getenv("EP_AUTO_TRESHOLD", 0))
GMM_PLATFORM = None


class MoEMethods(str, enum.Enum):
    """Enumeration of available MoE execution methods.

    This enum defines different strategies for executing Mixture of Experts layers,
    each optimized for different use cases and hardware configurations.

    Attributes:
        FUSED_MOE: Fused execution path using grouped matrix multiplication and shard_map.
            Optimal for distributed training with expert parallelism on TPUs/GPUs.
            Uses ragged tensor operations and custom kernels for maximum efficiency.
            Automatically falls back to STANDARD_MOE when FSDP*SP axis size > 1.

        STANDARD_MOE: Standard token-by-token execution path.
            More flexible and easier to debug. Uses traditional permutation,
            expert computation, and unpermutation steps. Supports all sharding
            configurations and is the fallback when fused path is unavailable.

        DENSE_MOE: Dense batched execution using per-token matrix multiplications.
            Instead of ragged/grouped operations, uses dense einsum operations
            with expert selection via indexing. Useful for debugging or when
            grouped matmul kernels are not available.

    Example:
        >>> from easydel.layers.moe import MoEMethods
        >>> # Configure in model config
        >>> config.moe_method = MoEMethods.FUSED_MOE
    """

    FUSED_MOE = "fused_moe"
    STANDARD_MOE = "standard_moe"
    DENSE_MOE = "dense_moe"


def rsum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool = True) -> jax.Array:
    """Performs reduce-scatter collective operation with float32 accumulation.

    This function wraps `jax.lax.psum_scatter` to provide a reduce-scatter operation,
    which combines reduction (sum) across devices with scattering of results. Each
    device receives a different slice of the reduced result.

    Args:
        x: Input array to reduce and scatter.
        axis_name: Name of the mesh axis to reduce-scatter along (e.g., "dp", "tp", "ep").
        scatter_dimension: Dimension of `x` along which to scatter the result.
            Each device receives a slice of size `x.shape[scatter_dimension] / num_devices`.
        tiled: If True, the output is tiled (concatenated) across devices in the
            scatter dimension. Defaults to True.

    Returns:
        Reduced and scattered array. Shape is the same as input, but the
        scatter_dimension is divided by the number of devices in the axis.

    Example:
        >>> # Reduce-scatter across data parallel axis
        >>> x = jnp.ones((8, 1024))  # On each device
        >>> result = rsum_scatter(x, "dp", scatter_dimension=0)
        >>> # result.shape = (8 // dp_size, 1024) with summed values
    """
    return lax.psum_scatter(x, axis_name, scatter_dimension=scatter_dimension, tiled=tiled)


def argsort(x: jax.Array) -> jax.Array:
    """Returns indices that would sort the input array along its last axis.

    Convenience wrapper around `jnp.argsort` for sorting along the last dimension,
    commonly used in MoE routing to sort tokens by expert assignment.

    Args:
        x: Input array to sort. Can have any number of dimensions.

    Returns:
        Integer array of indices that would sort `x` along axis=-1.
        Has the same shape as `x`.

    Example:
        >>> x = jnp.array([[3, 1, 2], [6, 4, 5]])
        >>> indices = argsort(x)
        >>> # indices = [[1, 2, 0], [1, 2, 0]]
        >>> sorted_x = jnp.take_along_axis(x, indices, axis=-1)
    """
    return jnp.argsort(x, axis=-1)


def take1d(x: jax.Array, idx: jax.Array) -> jax.Array:
    """Indexes an array along axis 0 using the provided indices.

    Convenience wrapper for extracting rows from a 2D array or elements from a 1D array.

    Args:
        x: Input array to index. Shape: (N, ...).
        idx: Integer array of indices to extract. Shape: (M,).
            Values should be in range [0, N).

    Returns:
        Array with selected elements. Shape: (M, ...) where ... matches
        the trailing dimensions of `x`.

    Example:
        >>> x = jnp.array([[10, 20], [30, 40], [50, 60]])
        >>> idx = jnp.array([2, 0, 1])
        >>> result = take1d(x, idx)
        >>> # result = [[50, 60], [10, 20], [30, 40]]
    """
    return jnp.take(x, idx, axis=0)


def repeat_take_sorted(x: jax.Array, sort_idx: jax.Array, k: int) -> jax.Array:
    """Repeats and reorders rows of an array based on sorted indices.

    This function is used in MoE to replicate token representations k times
    (once per selected expert) and then sort them by expert assignment.
    The sort indices are assumed to be for a k-times-repeated array, where
    `sort_idx // k` maps back to the original row indices.

    Args:
        x: Input array to repeat and sort. Shape: (N, ...).
        sort_idx: Sorted indices for the repeated array. Shape: (N*k,).
            Each group of k consecutive original indices can map to any position.
        k: Repetition factor (typically `num_experts_per_tok`).

    Returns:
        Sorted and implicitly repeated array. Shape: (N*k, ...) where each
        original row appears k times, reordered according to `sort_idx`.

    Example:
        >>> x = jnp.array([[1, 2], [3, 4]])  # 2 tokens
        >>> sort_idx = jnp.array([2, 0, 3, 1])  # Sorted for 2*2=4 entries
        >>> result = repeat_take_sorted(x, sort_idx, k=2)
        >>> # Maps indices [2,0,3,1] -> [1,0,1,0] -> rows [3,4], [1,2], [3,4], [1,2]
    """
    return jnp.take(x, sort_idx // k, axis=0)


def bincount(x: jax.Array, length: int) -> jax.Array:
    """Counts occurrences of non-negative integers in an array.

    Wrapper around `jnp.bincount` with explicit output length specification.
    Used in MoE to count how many tokens are assigned to each expert.

    Args:
        x: Integer array to count. Shape: (N,). Values should be in range [0, length).
            Negative values are ignored.
        length: Length of the output array. Determines the number of bins.

    Returns:
        Count array where result[i] = number of times i appears in x.
        Shape: (length,).

    Example:
        >>> expert_ids = jnp.array([0, 2, 1, 0, 2, 2])
        >>> counts = bincount(expert_ids, length=4)
        >>> # counts = [2, 1, 3, 0] - expert 0: 2 tokens, expert 1: 1 token, etc.
    """
    return jnp.bincount(x, length=length)


def all_i32(*xs: jax.Array) -> tuple[jax.Array, ...]:
    """Casts all input arrays to int32 dtype.

    Utility function for ensuring integer arrays use consistent int32 type,
    which is often required for indexing operations and compatibility with
    certain XLA operations.

    Args:
        *xs: Variable number of JAX arrays to cast.

    Returns:
        Tuple of arrays with all elements cast to jnp.int32.

    Example:
        >>> a = jnp.array([1, 2, 3], dtype=jnp.int64)
        >>> b = jnp.array([4, 5], dtype=jnp.int16)
        >>> a32, b32 = all_i32(a, b)
        >>> # Both are now int32
    """
    return tuple(z.astype(jnp.int32) for z in xs)


def sort_activations(inputs: jax.Array, sort_indices: jax.Array, use_custom_vjp: bool = True) -> jax.Array:
    """Reorders activations using provided sort indices with optional custom gradient.

    This function permutes the first dimension of `inputs` according to `sort_indices`.
    When `use_custom_vjp=True`, it uses a memory-efficient custom VJP that avoids
    materializing the full permutation matrix during backpropagation.

    Args:
        inputs: Input activations to sort. Shape: (N, ...).
        sort_indices: Integer array of indices defining the permutation. Shape: (N,).
            Must be a valid permutation of range(N).
        use_custom_vjp: If True, uses custom VJP for memory-efficient gradients.
            If False, uses standard JAX autodiff. Defaults to True.

    Returns:
        Sorted activations where output[i] = inputs[sort_indices[i]].
        Shape: same as `inputs`.

    Raises:
        AssertionError: If inputs.shape[0] != sort_indices.shape[0].

    Example:
        >>> x = jnp.array([[1, 2], [3, 4], [5, 6]])
        >>> indices = jnp.array([2, 0, 1])
        >>> sorted_x = sort_activations(x, indices)
        >>> # sorted_x = [[5, 6], [1, 2], [3, 4]]
    """
    assert inputs.shape[0] == sort_indices.shape[0], "Input and indices dimensions must match"

    if use_custom_vjp:
        return sort_activations_custom(inputs, sort_indices)
    else:
        return inputs[sort_indices, ...]


@jax.custom_vjp
def sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
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

        This function is decorated with @jax.custom_vjp, which means it has custom
        forward and backward implementations defined in _sort_activations_custom_fwd
        and _sort_activations_custom_bwd.
    """
    return inputs[sort_indices, ...]


def sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    sorted_output: jax.Array = inputs[sort_indices, ...]
    residuals: jax.Array = sort_indices
    return sorted_output, residuals


def sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
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
    input_grads: jax.Array = grads[inverse_indices, ...]
    return input_grads, None


sort_activations_custom.defvjp(sort_activations_custom_fwd, sort_activations_custom_bwd)


@dataclass(frozen=True)
class MoeFusedHooks:
    """Optional callbacks executed at key points of the fused MoE pipeline.

    Hooks allow fine-grained customization of the MoE execution flow without modifying
    core logic. Each hook is invoked at a specific stage of the pipeline and can
    inspect/modify tensors.

    **Routing & Selection Hooks:**
        before_gate: Invoked before gate/router layer. Can preprocess hidden states.
            Signature: (hidden_states: Array) -> Array

        after_gate: Invoked after gate/router logits are computed. Can postprocess logits.
            Signature: (gate_logits: Array) -> Array

        normalize_gate_logits: Invoked to compute gate weights from logits. If set, replaces
            the default softmax normalization. Useful for models using sigmoid-based routing
            (e.g., Llama4) instead of softmax.
            Signature: (gate_logits: Array) -> Array
            Default: softmax(gate_logits)

        before_topk: Invoked before top-k expert selection. Can modify logits before selection.
            Signature: (gate_logits: Array) -> Array

        select_hook: Invoked after top-k selection to refine expert weights/scores.
            Used for custom routing logic or weight normalization.
            Signature: (selected_weights: Array, selected_experts: Array) -> (weights: Array, experts: Array)

            Default for TOP_K routing: Normalizes weights by their sum (softmax-like normalization).

    **Expert Processing Hooks:**
        refine_weights_hook: Invoked before W_i and W_u (gate/up) projections.
            Can refine expert weights before linear transformations.
            Signature: (weights: Array) -> Array

        after_wiwu: Invoked after W_i and W_u (gate/up) projections.
            Can post-process expert intermediate activations.
            Signature: (intermediate: Array) -> Array

        before_wo: Invoked before W_o (output) projection.
            Can modify combined expert outputs before final projection.
            Signature: (combined_output: Array) -> Array

        after_wo: Invoked after W_o (output) projection.
            Can post-process final expert layer outputs.
            Signature: (output: Array) -> Array

    **Distributed Execution Hooks:**
        refine_inputs_hook: Invoked before expert-parallel all-to-all communication.
            Can refine token representations or route them to specific experts.
            Signature: (inputs: Array, weights: Array, shape: Tuple) -> Array

        after_ep_receive: Invoked after receiving tokens from other expert shards.
            Can refine received token representations before expert computation.
            Signature: (received_tokens: Array) -> Array

    **Output Combination Hooks:**
        before_combine: Invoked before combining outputs from multiple experts per token.
            Can adjust expert weights or outputs before weighted sum.
            Signature: (outputs: Array, weights: Array) -> (outputs: Array, weights: Array)

        finalize_output: Invoked at the very end of MoE computation.
            Can apply final normalization, residual connections, etc.
            Signature: (final_output: Array) -> Array

    Default behavior: All hooks are `None`, so the pipeline proceeds without intervention.
    """

    before_gate: Callable | None = None
    after_gate: Callable | None = None
    normalize_gate_logits: Callable | None = None  # If set, replaces default softmax normalization
    before_topk: Callable | None = None
    select_hook: Callable | None = None
    refine_inputs_hook: Callable | None = None
    scale_replicated_inputs: Callable | None = None  # Scale inputs after replication for input-scaling MoE
    after_ep_receive: Callable | None = None
    refine_weights_hook: Callable | None = None
    output_weights_hook: Callable | None = None  # Modify weights during output combination (unpermute)
    after_wiwu: Callable | None = None
    before_wo: Callable | None = None
    after_wo: Callable | None = None
    before_combine: Callable | None = None
    finalize_output: Callable | None = None

    def _hash__(self) -> int:
        """Makes the hooks dataclass hashable for NNX graph hashing.

        Returns:
            Hash value based on the identity of all hook callables.

            Uses id() of callables rather than hashing the functions themselves,
            which works correctly for functions, lambdas, and partials.
        """
        return hash(
            (
                id(self.before_gate),
                id(self.after_gate),
                id(self.normalize_gate_logits),
                id(self.before_topk),
                id(self.select_hook),
                id(self.refine_inputs_hook),
                id(self.scale_replicated_inputs),
                id(self.after_ep_receive),
                id(self.refine_weights_hook),
                id(self.output_weights_hook),
                id(self.after_wiwu),
                id(self.before_wo),
                id(self.after_wo),
                id(self.before_combine),
                id(self.finalize_output),
            )
        )

    def replace(self, **kws) -> MoeFusedHooks:
        return replace(self, **kws)


def canon_dim(ndim: int, dim: int) -> int:
    """Canonicalizes a dimension index to be non-negative.

    Converts negative dimension indices (counting from the end) to their
    equivalent positive indices, following NumPy/JAX conventions where
    -1 refers to the last dimension, -2 to second-to-last, etc.

    Args:
        ndim: Total number of dimensions in the array.
        dim: Dimension index to canonicalize. Can be negative.
            Must be in range [-ndim, ndim).

    Returns:
        Non-negative dimension index in range [0, ndim).

    Example:
        >>> canon_dim(3, -1)  # Last dimension of 3D array
        2
        >>> canon_dim(3, 1)   # Already positive
        1
        >>> canon_dim(4, -2)  # Second to last
        2
    """
    return dim if dim >= 0 else (ndim + dim)


def psum_maybe(
    x: jax.Array,
    axes: tuple[str, ...],
    mesh: jax.sharding.Mesh,
    dtype: jnp.dtype = jnp.float32,
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

    """
    names = tuple(a for a in axes if a in mesh.axis_names and mesh.shape.get(a, 1) > 1)
    return lax.psum(x.astype(dtype), names).astype(x.dtype) if names else x


def rsum_scatter_maybe(
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

    """
    if axis_name not in mesh.axis_names or mesh.shape.get(axis_name, 1) == 1:
        return x
    dim = canon_dim(x.ndim, dim)
    return lax.psum_scatter(x.astype(dtype), axis_name, scatter_dimension=dim, tiled=True).astype(x.dtype)


def slice_k_for_param_shards(
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

    """
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
    ragged all-to-all communication patterns during expert parallel execution. Each
    strategy corresponds to a different parameter needed for `jax.lax.ragged_all_to_all`.

    Ragged all-to-all is used when different devices need to exchange variable-sized
    chunks of data, which is common in MoE when tokens are redistributed across expert
    shards based on routing decisions.

    Attributes:
        INPUT_OFFSET: Strategy for computing input buffer offsets.
            Determines where to read from in the local input buffer when sending
            data to each destination device.

        SEND_SIZE: Strategy for computing send buffer sizes.
            Specifies how many elements to send to each destination device.

        OUTPUT_OFFSET: Strategy for computing output buffer offsets.
            Determines where to write received data in the local output buffer.

        RECV_SIZE: Strategy for computing receive buffer sizes.
            Specifies how many elements to receive from each source device.

    Note:
        This enum is used internally by `get_all_to_all_params` to compute
        the communication parameters for expert-parallel token redistribution.
    """

    INPUT_OFFSET = enum.auto()
    SEND_SIZE = enum.auto()
    OUTPUT_OFFSET = enum.auto()
    RECV_SIZE = enum.auto()


def get_all_to_all_params(
    all_shards_group_sizes: jax.Array,
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


def tp_global_topk(logits_shard: jax.Array, k: int, tp_axis: str) -> tuple[jax.Array, jax.Array]:
    """Computes global top-k selection across tensor-parallel shards.

    This function performs top-k selection across multiple tensor-parallel shards by:
    1. Computing local top-k on each shard's expert subset
    2. Gathering local top-k results from all shards
    3. Computing global top-k from the gathered results

    This is more communication-efficient than gathering all logits and then computing top-k,
    as it only communicates k*tp values instead of E values per token.

    Args:
        logits_shard: Local router logits for experts on this shard.
            Shape: (batch_size, E_local) where E_local = total_experts / tp_size.
        k: Number of top experts to select globally.
        tp_axis: Name of the tensor parallel mesh axis.

    Returns:
        A tuple of (values, indices) where:
            - values: Top-k router logit values. Shape: (batch_size, k).
            - indices: Global expert indices for top-k experts. Shape: (batch_size, k).

    """
    vals_l, idx_l = jax.lax.top_k(logits_shard, k)
    shard = jax.lax.axis_index(tp_axis)
    E_local = logits_shard.shape[-1]
    idx_l = idx_l + shard * E_local

    vals_all = jax.lax.all_gather(vals_l, tp_axis, axis=-1, tiled=True)
    idx_all = jax.lax.all_gather(idx_l, tp_axis, axis=-1, tiled=True)

    vals_g, pos = jax.lax.top_k(vals_all, k)
    idx_g = jnp.take_along_axis(idx_all, pos, axis=-1)
    return vals_g, idx_g


def get_experts_location(
    gate_logits,
    pre_bias_logits,
    select_hook: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None,
    refine_weights_hook: typing.Callable[[jax.Array], jax.Array] | None = None,
    *,
    num_experts_per_tok: int,
):
    """Compute top-k experts and weights with optional overrides.

    If a custom `select_hook` is provided, it is used to compute the top-k selection
    from the provided logits. Otherwise, `jax.lax.top_k` is used on
    `gate_logits`. After selection, an optional `refine_weights_hook` can adjust the raw
    weights (e.g., apply softmax, temperature scaling, or masking).

    Args:
        gate_logits: Router logits used for expert selection, typically after any
            preprocessing. Shape: (tokens, num_experts).
        pre_bias_logits: Optional pre-bias logits that some custom `select_hook`
            might consume. Shape should align with `gate_logits` if used.
        select_hook: Optional function `(gate_logits, pre_bias_logits, k) -> (vals, idx)`
            to compute the top-k scores and indices.
        refine_weights_hook: Optional function to modify the selected weights after top-k.
        num_experts_per_tok: Number of experts to select per token (k).

    Returns:
        A tuple `(top_k_weights, top_k_indices)` where:
        - `top_k_weights`: Selected weights per token. Shape: (tokens, k).
        - `top_k_indices`: Expert indices per token. Shape: (tokens, k).
    """
    if select_hook:
        top_k_weights, top_k_indices = select_hook(gate_logits, pre_bias_logits, num_experts_per_tok)
    else:
        top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, num_experts_per_tok)
    if refine_weights_hook:
        top_k_weights = refine_weights_hook(top_k_weights)

    return top_k_weights, top_k_indices


def permute(
    inputs: jax.Array,
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array | None = None,
    use_custom_sort_vjp: bool = True,
    roll_to_expert_id=None,
    *,
    num_experts_per_tok: int,
    num_experts: int,
    dtype: jnp.dtype,
    select_hook: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None,
    refine_weights_hook: typing.Callable[[jax.Array], jax.Array] | None = None,
    refine_inputs_hook: typing.Callable[[jax.Array, jax.Array, tuple[int]], jax.Array] | None = None,
    scale_replicated_inputs: typing.Callable[[jax.Array, jax.Array], jax.Array] | None = None,
):
    """Permute tokens by expert assignment for grouped matmul.

    This routine selects top-k experts per token, optionally refines the input or
    weights, and then repeats and sorts token representations so that tokens
    belonging to the same expert are contiguous. The output layout is suitable
    for ragged or grouped matrix multiplications used in expert FFNs.

    **Algorithm Overview:**

        1. **Expert Selection**: For each token, select top-k experts based on router logits
        2. **Token Replication**: Replicate each token k times (once per selected expert)
        3. **Sorting**: Sort all replicated tokens by their assigned expert ID
        4. **Group Computation**: Compute how many tokens are assigned to each expert

    This creates a layout where tokens for expert 0 are first, then expert 1, etc.,
    enabling efficient grouped/ragged matrix multiplication where each expert processes
    its assigned tokens as a contiguous batch.

    Args:
        inputs: Token representations. Shape: (batch, seq, hidden).
        gate_logits: Router logits for expert selection. Shape: (batch*seq, E)
            if flattened, or broadcastable to that shape.
        pre_bias_logits: Optional alternate logits for custom `select_hook`.
        use_custom_sort_vjp: Whether to use the custom VJP sorter for efficiency.
        roll_to_expert_id: Optional expert ID offset to rotate indices (e.g., when
            each shard processes a subset of experts in ring-of-experts setups).
            This is used in ring-of-experts mode where each device processes a
            contiguous subset of experts. The offset makes expert IDs local to each shard.
        num_experts_per_tok: Number of experts selected per token (k).
        num_experts: Total number of experts (E).
        dtype: Output dtype for the sorted inputs.
        select_hook: Optional function for top-k selection; otherwise `lax.top_k`.
        refine_weights_hook: Optional function to modify selected weights.
        refine_inputs_hook: Optional function to refine or modify token representations based
            on the selected weights and original input shape.

    Returns:
        A 5-tuple `(sorted_inputs, sorted_selected_experts, weights, group_size, sorted_experts)`
        where:
        - `sorted_inputs`: Tokens repeated k times and sorted by selected expert.
          Shape: (tokens*k, hidden).
        - `sorted_selected_experts`: Sorting indices (permutation of range(tokens*k)).
          Shape: (tokens*k,).
        - `weights`: Selected expert weights per repeated token. Shape: (tokens*k,).
        - `group_size`: Number of tokens per expert. Shape: (E,).
        - `sorted_experts`: Expert id per sorted row. Shape: (tokens*k,).

    Example:
        >>> # 2 tokens, 2 experts per token, 4 total experts
        >>> inputs = jnp.ones((1, 2, 128))  # (batch, seq, hidden)
        >>> gate_logits = jnp.array([[0.9, 0.1, 0.8, 0.2],  # token 0 -> experts 0, 2
        ...                           [0.3, 0.7, 0.4, 0.6]]) # token 1 -> experts 1, 3
        >>>
        >>> sorted_inputs, indices, weights, sizes, expert_ids = permute(
        ...     inputs, gate_logits, num_experts_per_tok=2, num_experts=4, dtype=jnp.bfloat16
        ... )
        >>>
        >>> # sorted_inputs: tokens grouped as [tok0_exp0, tok1_exp1, tok0_exp2, tok1_exp3]
        >>> # sizes: [1, 1, 1, 1] - one token per expert
        >>> # expert_ids: [0, 1, 2, 3] - expert ID for each sorted token
    """
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = get_experts_location(
        gate_logits=gate_logits,
        pre_bias_logits=pre_bias_logits,
        select_hook=select_hook,
        refine_weights_hook=refine_weights_hook,
        num_experts_per_tok=num_experts_per_tok,
    )
    if refine_inputs_hook:
        inputs_2d = refine_inputs_hook(inputs_2d, weights, inputs_shape)

    flatten_selected_experts = jnp.ravel(selected_experts)

    if roll_to_expert_id is not None:
        flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % num_experts

    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    replicated_inputs_2d = jnp.repeat(inputs_2d, num_experts_per_tok, axis=0)

    # Apply input scaling if hook is provided (for input-scaling MoE like Llama4)
    # weights shape: (tokens, k), flatten to (tokens*k,) to match replicated_inputs_2d
    if scale_replicated_inputs is not None:
        flattened_weights = jnp.ravel(weights)  # (tokens*k,)
        replicated_inputs_2d = scale_replicated_inputs(replicated_inputs_2d, flattened_weights)

    sorted_inputs = sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp).astype(dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=num_experts)

    expert_indices = jnp.arange(num_experts)
    sorted_experts = jnp.repeat(
        expert_indices,
        repeats=group_size,
        total_repeat_length=flatten_selected_experts.shape[0],
    )
    return (
        sorted_inputs,
        sorted_selected_experts,
        weights,
        group_size,
        sorted_experts,
    )


def unpermute(
    intermediate,
    sorted_selected_experts,
    weights,
    batch_size,
    sequence_length,
    use_custom_sort_vjp=True,
    weight_modif_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
    *,
    num_experts_per_tok: int,
    dtype: jnp.dtype,
):
    """Invert expert permutation and combine expert outputs per token.

    This function undoes the expert-grouped permutation applied by `permute`,
    sums across the per-expert dimension using the provided weights, and restores
    the original `(batch, seq, hidden)` layout.

    **Algorithm Overview:**

        1. **Unsort**: Apply inverse permutation to restore original token order
        2. **Reshape**: Reshape from (tokens*k, hidden) to (tokens, k, hidden)
        3. **Weighted Sum**: Combine k expert outputs per token using weights
        4. **Restore Shape**: Reshape from (tokens, hidden) to (batch, seq, hidden)

    The weighted combination computes: output[i] = sum_j(weights[i,j] * expert_outputs[i,j])
    where j ranges over the k selected experts for token i.

    Args:
        intermediate: Outputs computed in expert-grouped order. Shape:
            (tokens*k, out_dim).
        sorted_selected_experts: Indices used to sort tokens during `permute`.
            Shape: (tokens*k,).
        weights: Expert weights per token before unpermutation. Shape:
            (tokens*k,) or reshaped to (tokens, k).
        batch_size: Original batch size.
        sequence_length: Original sequence length.
        use_custom_sort_vjp: Whether to use the custom VJP sorter for efficiency.
        weight_modif_fn: Optional function to post-process weights prior to
            combining (e.g., re-normalization).
        num_experts_per_tok: Number of experts per token (k).
        dtype: Output dtype after combining.

    Returns:
        Output tensor with original layout. Shape: (batch_size, sequence_length, out_dim).

    Example:
        >>> # Following the permute example
        >>> # intermediate outputs from experts (4 tokens, 128 dim)
        >>> intermediate = expert_outputs  # Shape: (4, 128)
        >>> sorted_indices = jnp.array([0, 1, 2, 3])  # From permute
        >>> weights = jnp.array([0.6, 0.4, 0.7, 0.3])  # Expert weights
        >>>
        >>> output = unpermute(
        ...     intermediate,
        ...     sorted_indices,
        ...     weights,
        ...     batch_size=1,
        ...     sequence_length=2,
        ...     num_experts_per_tok=2,
        ...     dtype=jnp.bfloat16
        ... )
        >>>
        >>> # output.shape = (1, 2, 128)
        >>> # output[0,0] = 0.6*expert0_out + 0.7*expert2_out  # token 0's combined output
        >>> # output[0,1] = 0.4*expert1_out + 0.3*expert3_out  # token 1's combined output
    """

    unsort_intermediate = sort_activations(intermediate, jnp.argsort(sorted_selected_experts), use_custom_sort_vjp)
    reshaped_weights = jnp.reshape(weights, (-1, num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(unsort_intermediate, (reshaped_weights.shape[0], num_experts_per_tok, -1))
    with jax.named_scope("weight_sum"):
        if weight_modif_fn is not None:
            reshaped_weights = weight_modif_fn(reshaped_weights)
        output = jnp.einsum(
            "BKE,BK -> BE", reshaped_intermediate.astype(jnp.float32), reshaped_weights.astype(jnp.float32)
        )
    return output.reshape(batch_size, sequence_length, -1).astype(dtype)


def local_permute(
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
    sorted_inputs = sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids


class MoeRoutingStrategy(enum.Enum):
    """Defines the available strategies for routing tokens to experts in an MoE layer.

    Each strategy determines how tokens are assigned to experts and how expert weights
    are computed. When using fused MoE, hooks are automatically configured based on
    the routing strategy to ensure correct behavior.

    **Attributes:**

        TOP_K: Standard top-k routing with weight normalization (softmax).
            - Each token is routed to the k experts with the highest router logits.
            - Expert weights are normalized by their sum (softmax normalization).
            - Default hook: `select_hook` normalizes weights so they sum to 1.0.
            - Use case: Most common approach, works well for balanced expert utilization.

        TOP_K_NDIV: Top-k routing WITHOUT weight normalization.
            - Each token is routed to the k experts with the highest router logits.
            - Expert weights are NOT normalized (raw logit values used as weights).
            - Default hook: `select_hook` passes weights through unchanged.
            - Use case: When you want to use raw logits directly for expert combination.

        SWITCH: Switch Transformer-style routing (token -> 1 expert only).
            - Each token is routed to ONLY the single top-1 expert.
            - Expert weight is enforced as exactly 1.0 (hard assignment).
            - Default hook: `select_hook` sets weights to 1.0 (hard gating).
            - Use case: Sparse, efficient routing with reduced computational cost.

        EMPTY_CHOICE: Expert Choice routing (expert -> k tokens).
            - INVERTED routing: each expert selects its top-k tokens.
            - Better load balancing compared to top-k token routing.
            - Default hook: `select_hook` uses uniform weights (1/k per expert).
            - Use case: Scenarios requiring strict expert load balancing.

        HASH: Hash-based routing (token -> expert by token_id % num_experts).
            - Simple deterministic routing based on token IDs.
            - All experts receive equal number of tokens.
            - Default hook: `select_hook` uses uniform weights (1/k per expert).
            - Use case: Debugging, baseline comparisons, or fully deterministic execution.

    **Hook Auto-Configuration:**
        When using `MoeFusedHooks` with fused MoE execution, the `select_hook` is
        automatically configured based on the routing strategy if not explicitly set:

        - TOP_K: Normalize weights → sum to 1.0 (probability distribution)
        - TOP_K_NDIV: Passthrough → raw weights unchanged
        - SWITCH: Hard gating → weights = 1.0
        - EMPTY_CHOICE: Uniform → weights = 1/k
        - HASH: Uniform → weights = 1/k

        Custom hooks can override the defaults by setting them on `self.moe_hooks`
        before calling the MoE layer.
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

    expert_loads: Float[Array, "num_experts"]  # noqa
    router_probs: Float[Array, "batch_seq num_experts"]
    selected_experts: Int[Array, "batch_seq num_experts_per_tok"]
    selected_weights: Float[Array, "batch_seq num_experts_per_tok"]
    load_balancing_loss: float | None = None
    router_z_loss: float | None = None
    expert_utilization: float | None = None
    routing_entropy: float | None = None


def resolve_eformer_axis(axis: str | list[str], manager: PartitionManager):
    """Resolves logical axis name(s) to physical mesh axis names.

    This convenience wrapper resolves symbolic axis names (like "tp", "ep", "fsdp")
    to their actual names in the device mesh for training mode. This is necessary
    because EFormer's PartitionManager may map logical parallelism axes to different
    physical mesh axes depending on configuration.

    Args:
        axis: A single axis name or list of axis names to resolve. Common values:
            - "tp": Tensor parallel axis
            - "ep": Expert parallel axis
            - "dp": Data parallel axis
            - "fsdp": Fully sharded data parallel axis
            - "sp": Sequence parallel axis
        manager: The `PartitionManager` instance providing axis resolution configuration.

    Returns:
        If input was a string, returns a single resolved axis name (str).
        If input was a list/tuple, returns a list of resolved axis names preserving order.

    Example:
        >>> # Single axis
        >>> resolved = resolve_eformer_axis("tp", partition_manager)
        >>> # resolved might be "tensor" or "model" depending on config
        >>>
        >>> # Multiple axes
        >>> resolved = resolve_eformer_axis(["tp", "ep"], partition_manager)
        >>> # resolved might be ["tensor", "expert"]
    """
    was_list = isinstance(axis, list | tuple)
    if not was_list:
        axis = [axis]
    out = manager.paxis.resolve_axis(axes=axis, mode=MODE_TRAIN)
    if not was_list:
        return out[0]
    return out


def get_moe_partition_spec(
    partition_manager: PartitionManager,
    direction: typing.Literal["row", "column"],
    tensors_are_expert: bool,
    is_bias: bool = False,
    fsdp_is_ep_bound: bool = True,
    sp_is_ep_bound: bool = True,
    module_view: bool = False,
) -> jax.sharding.PartitionSpec:
    if direction not in ("row", "column"):
        raise ValueError(f"direction must be 'row' or 'column', got '{direction}'")

    expert_axis_name = resolve_eformer_axis(EP, partition_manager)
    fsdp_axis_name = resolve_eformer_axis(FSDP, partition_manager)
    sp_axis_name = resolve_eformer_axis(SP, partition_manager)
    tensor_axis_name = resolve_eformer_axis(TP, partition_manager)
    expert_place = (expert_axis_name,)
    if module_view:
        if sp_is_ep_bound:
            expert_place += (sp_axis_name,)
        if fsdp_is_ep_bound:
            expert_place += (fsdp_axis_name,)
    if len(expert_place) == 1:
        expert_place = expert_place[0]
    if tensors_are_expert:
        # Expert tensor mode: all experts sharded across TP axis
        if is_bias:
            return jax.sharding.PartitionSpec(tensor_axis_name, None)
        else:
            return jax.sharding.PartitionSpec(tensor_axis_name, None, None)
    else:
        # Standard mode: experts on EP, features on TP
        if is_bias:
            return jax.sharding.PartitionSpec(expert_place, tensor_axis_name)
        else:
            if direction == "column":
                # Column-wise: [expert, None, tp] for wi/wu
                return jax.sharding.PartitionSpec(expert_place, None, tensor_axis_name)
            else:  # direction == "row"
                # Row-wise: [expert, tp, None] for wd
                return jax.sharding.PartitionSpec(expert_place, tensor_axis_name, None)
