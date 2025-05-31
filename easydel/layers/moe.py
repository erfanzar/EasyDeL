# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp
from enum import Enum, auto
from functools import partial

import chex
import flax.nnx as nn
import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from jax import lax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Integer

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import block_wise_ffn
from easydel.kernels.tpu_ops import pallas_grouped_matmul

HiddenState3d = Float[Array, "batch sequence_length hidden_size"]
FlattenState2d = Float[Array, "flattened hidden_size"]
WiType = Float[Array, "dim0 dim1"]
WoType = Float[Array, "dim1 dim0"]
ExpertsWiType = Float[Array, "experts dim0 dim1"]
ExpertsWoType = Float[Array, "experts dim1 dim0"]
RouterLogits2d = Float[Array, "flattened num_experts"]
RouterIndcies2d = Integer[Array, "flattened num_experts"]
MoEConfig = tp.TypeVar("MoEConfig", bound=EasyDeLBaseConfig)

EXPERT_PARALLEL = common_types.EXPERT_PARALLEL
TENSOR_PARALLEL = common_types.TENSOR_PARALLEL
BATCH = common_types.BATCH
EMPTY = common_types.EMPTY

HEAD_MOE_SHARIND = [common_types.DATA_PARALLEL, common_types.FULLY_SHARDED_DATA_PARALLEL]


class MoELayer(nn.Module):
    """A Mixture of Experts (MoE) layer.

    This layer implements a Mixture of Experts (MoE) layer for use in large-scale models.
    The MoE layer consists of multiple experts, each of which is a neural network.
    The input to the MoE layer is routed to a subset of the experts, and the outputs of the selected experts are
    combined to produce the final output.

    Attributes:
        config: An instance of `EasyDeLBaseConfig` that holds the configuration parameters for the MoE layer.
        rngs: A dictionary of PRNG keys for stochastic operations.
        dtype: The data type used for computation (default: `jnp.bfloat16`).
        param_dtype: The data type used for storing parameters (default: `jnp.bfloat16`).
        precision: The precision used for matrix multiplications (default: `lax.Precision.DEFAULT`).
        num_local_experts: The number of experts available on each device.
        num_experts_per_tok: The number of experts to which each token is routed.
        tiling_size: The tiling size used for `pallas_grouped_matmul`.
        n_routing_groups: The number of routing groups used in `deepseek_routing`.
        topk_routing_group: The number of top routing groups to select in `deepseek_routing`.
        use_megablox: A boolean indicating whether to use the `pallas_grouped_matmul` kernel.
        _collection_name: The name of the collection used to store the experts.
        _w0_coll_name: The name of the collection used to store the weights for the first linear layer of the experts.
        _w1_coll_name: The name of the collection used to store the weights for the second linear layer of the experts.
        _wo_coll_name: The name of the collection used to store the weights for the output linear layer of the experts.
        _ep: The name of the expert parallel axis.
        _tp: The name of the tensor parallel axis.
    """

    config: EasyDeLBaseConfig
    rngs: nn.Rngs
    dtype: tp.Any = jnp.bfloat16
    param_dtype: tp.Any = jnp.bfloat16
    precision: lax.Precision = lax.Precision.DEFAULT

    # MoE Arguments
    num_local_experts: int
    num_experts_per_tok: int
    tiling_size: tuple[int, int, int] = (4, 128, 128)
    n_routing_groups: int = -1
    topk_routing_group: int = 1
    use_megablox: bool = True

    # Collection Arguments
    _collection_name: str = "experts"
    _w0_coll_name: str | None = None
    _w1_coll_name: str | None = None
    _wo_coll_name: str | None = None

    _ep: str = "ep"
    _tp: str = "tp"

    def __init__(
        self,
        config: MoEConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: lax.Precision,
        rngs: nn.Rngs,
        collection_name: str,
        num_local_experts: int,
        num_experts_per_tok: int,
        n_routing_groups: int = 1,
        topk_routing_group: int = 1,
        use_megablox: bool = True,
        w0_name: str | None = None,
        w1_name: str | None = None,
        wo_name: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.tiling_size = (config.moe_tiling_size_batch, config.moe_tiling_size_seqlen, config.moe_tiling_size_dim)

        self._collection_name = collection_name
        self._w0_coll_name = w0_name
        self._w1_coll_name = w1_name
        self._wo_coll_name = wo_name

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.n_routing_groups = n_routing_groups
        self.topk_routing_group = topk_routing_group
        self.use_megablox = use_megablox
        self._available_ep = config.mesh.shape[self._ep]
        self._available_tp = config.mesh.shape[self._tp]

    def _prepare_moe_inputs(self, hidden_states: HiddenState3d) -> FlattenState2d:
        """Prepares the input hidden states for the MoE layer by applying logical sharding.

        Args:
            hidden_states: The input hidden states.

        Returns:
            The flattened and sharded hidden states.
        """
        partition_manager = self.config.partition_manager
        return apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=partition_manager,
        )

    def _collect_weights(self) -> tuple[list[WiType], list[WiType], list[WoType]]:
        """Collects the weights from the experts.

        Returns:
            A tuple containing lists of weights for the first linear layer, the second linear layer,
            and the output linear layer of the experts.
        """
        _w0_collection = []
        _w1_collection = []
        _wo_collection = []
        for _module in getattr(self, self._collection_name):
            if self._w0_coll_name is not None:
                _w0_collection.append(getattr(_module, self._w0_coll_name).kernel.value)
            if self._w1_coll_name is not None:
                _w1_collection.append(getattr(_module, self._w1_coll_name).kernel.value)
            if self._wo_coll_name is not None:
                _wo_collection.append(getattr(_module, self._wo_coll_name).kernel.value)
        return _w0_collection, _w1_collection, _wo_collection

    def _collect_stacked_weights(self) -> list[ExpertsWiType, ExpertsWiType, ExpertsWoType]:
        """Collects the weights from the experts and stacks them into a single array.

        Returns:
            A list containing the stacked weights for the first linear layer, the second linear layer,
            and the output linear layer of the experts.
        """
        _w0_collection, _w1_collection, _wo_collection = self._collect_weights()
        return [
            x
            for x in [
                jnp.stack(_w0_collection, axis=0) if len(_w0_collection) != 0 else None,
                jnp.stack(_w1_collection, axis=0) if len(_w1_collection) != 0 else None,
                jnp.stack(_wo_collection, axis=0) if len(_wo_collection) != 0 else None,
            ]
            if x is not None
        ]

    def _get_topk(
        self,
        router_logits: RouterLogits2d,
        bias_logits: RouterLogits2d | None = None,
        use_softmax: bool = True,
        random_routing: bool = False,
        deepseek_routing: bool = False,
        scale_weights: bool = True,
        routed_scaling_factor: float = 1.0,
        score_function: str | None | tp.Literal["sigmoid"] = None,
    ) -> tuple[RouterLogits2d, RouterIndcies2d]:
        """Selects the top-k experts for each token based on the router logits.

        Args:
            router_logits: The logits from the router.
            bias_logits: Optional bias logits.
            use_softmax: Whether to apply softmax to the weights.
            random_routing: Whether to use random routing.
            deepseek_routing: Whether to use deepseek routing.
            scale_weights: Whether to scale the weights.
            routed_scaling_factor: The scaling factor for the weights.
            score_function: The score function to use.

        Returns:
            A tuple containing the weights and indices of the selected experts.
        """
        resolver = self.config.partition_manager.resolve
        mode = common_types.MODE_DECODE if router_logits.shape[1] == 1 else common_types.MODE_TRAIN

        @partial(
            shard_map,
            mesh=self.config.mesh,
            in_specs=(
                resolver(axes=[HEAD_MOE_SHARIND, EMPTY, EMPTY], mode=mode),
                resolver(axes=[HEAD_MOE_SHARIND, EMPTY, EMPTY], mode=mode) if bias_logits is not None else None,
            ),
            out_specs=(
                resolver(axes=[HEAD_MOE_SHARIND, EMPTY, EMPTY], mode=mode),
                resolver(axes=[HEAD_MOE_SHARIND, EMPTY, EMPTY], mode=mode),
            ),
            check_rep=False,
        )
        def compute(local_router_logits: chex.Array, local_bias_logits: chex.Array | None):
            b, s, e = local_router_logits.shape
            local_router_logits = local_router_logits.reshape(-1, local_router_logits.shape[-1])
            if local_bias_logits is not None:
                local_bias_logits = local_bias_logits.reshape(-1, local_bias_logits.shape[-1])
            fltten_dim, num_experts = local_router_logits.shape
            expt = self.num_experts_per_tok
            if random_routing:
                indices = jnp.arange(num_experts).repeat(fltten_dim)
                selected_num = fltten_dim * expt
                indices = jax.random.choice(self.rngs.param(), indices, shape=(selected_num,))
                indices = indices.reshape(fltten_dim, expt)
                weights = jnp.take_along_axis(local_router_logits, indices, axis=-1)
                return weights.reshape(b, s, -1), indices.reshape(b, s, -1)

            if deepseek_routing:
                weights, indices = self._deepseek_routing(local_router_logits, local_bias_logits)
            else:
                weights, indices = lax.top_k(local_router_logits, expt)
            if scale_weights:
                weights = self._scale_weights(
                    weights,
                    routed_scaling_factor=routed_scaling_factor,
                    score_function=score_function,
                )
            if use_softmax:
                weights = jax.nn.softmax(weights.astype("f4"), axis=-1).astype(weights.dtype)
            return weights.reshape(b, s, -1), indices.reshape(b, s, -1)

        weights, indices = compute(router_logits, bias_logits)
        return weights, indices

    def _deepseek_routing(self, router_logits: chex.Array, bias_logits: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Implements the DeepSeek routing strategy.

        Args:
            router_logits: The logits from the router.
            bias_logits: The bias logits.

        Returns:
            A tuple containing the weights and indices of the selected experts.
        """
        n = router_logits.shape[0]
        expt = self.num_experts_per_tok
        num_experts = self.num_local_experts

        if self.n_routing_groups != -1:
            experts_per_group = num_experts // self.n_routing_groups
            scores_grouped = jnp.reshape(router_logits, (n, self.n_routing_groups, experts_per_group))

            top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
            group_scores = jnp.sum(top2_in_group_vals.astype(jnp.float32), axis=-1)
            group_idx = jax.lax.top_k(group_scores, k=self.topk_routing_group)[1]
            group_mask = jax.nn.one_hot(group_idx, num_classes=self.n_routing_groups, dtype=jnp.float32)
            group_mask = jnp.sum(group_mask, axis=1)

            score_mask_grouped = jnp.expand_dims(group_mask, axis=-1)
            score_mask_expanded = jnp.broadcast_to(score_mask_grouped, (n, self.n_routing_groups, experts_per_group))
            score_mask = jnp.reshape(score_mask_expanded, (n, num_experts))
            negative_infinity = -jax.numpy.inf
            masked_scores = jnp.where(score_mask > 0, router_logits, negative_infinity)
            indices = jax.lax.top_k(masked_scores, k=expt)[1]
        else:
            indices = jax.lax.top_k(router_logits, k=expt)[1]

        weights = jnp.take_along_axis(bias_logits, indices, axis=-1)
        return weights, indices

    def _scale_weights(
        self,
        weights: chex.Array,
        routed_scaling_factor: float = 1.0,
        score_function: str | None | tp.Literal["sigmoid"] = None,
    ) -> chex.Array:
        if score_function is not None and score_function == "sigmoid":
            weights /= weights.sum(axis=-1, keepdims=True)
        return weights * routed_scaling_factor

    @staticmethod
    def _local_permute(
        inputs,
        global_group_sizes,
        local_expert_size,
        shard_index,
        is_offset=False,
        global_sorted_experts=None,
    ):
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes,
            shard_index * local_expert_size,
            local_expert_size,
            axis=1,
        )
        local_sizes = all_shard_local_sizes.reshape(-1)
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)
        if is_offset:
            divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
            expert_indices = jnp.where(
                divided_assignments == shard_index,
                jnp.mod(global_sorted_experts, local_expert_size),
                local_expert_size,
            )
        else:
            base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
            expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        return (
            sorted_inputs,
            sorted_indices,
            local_group_size,
            sorted_experts_ids,
        )

    @staticmethod
    def _all_to_all_params(all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True):
        class TransformStrategy(Enum):
            INPUT_OFFSET = auto()
            SEND_SIZE = auto()
            OUTPUT_OFFSET = auto()
            RECV_SIZE = auto()

        def transform_array(input_array, shard_id, strategy, is_batch_sharded):
            if is_batch_sharded:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    local_array = input_array[shard_id]
                    return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
                elif strategy == TransformStrategy.SEND_SIZE:
                    return input_array[shard_id]
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
                    array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
                    cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
                    return cumulated_array[shard_id]
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array[:, shard_id]
                else:
                    raise ValueError(f"Unknown tranform array strategy: {strategy}")
            else:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
                elif strategy == TransformStrategy.SEND_SIZE:
                    return jnp.repeat(input_array[shard_id], num_expert_parallelism)
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
                    return jnp.repeat(output_offset, num_expert_parallelism)
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array
                else:
                    raise ValueError(f"Unknown tranform array strategy: {strategy}")

        input_offsets = transform_array(
            all_shards_group_sizes,
            shard_id,
            TransformStrategy.INPUT_OFFSET,
            is_batch_sharded,
        )
        send_sizes = transform_array(
            all_shards_group_sizes,
            shard_id,
            TransformStrategy.SEND_SIZE,
            is_batch_sharded,
        )
        output_offsets = transform_array(
            all_shards_group_sizes,
            shard_id,
            TransformStrategy.OUTPUT_OFFSET,
            is_batch_sharded,
        )
        recv_sizes = transform_array(
            all_shards_group_sizes,
            shard_id,
            TransformStrategy.RECV_SIZE,
            is_batch_sharded,
        )
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def permute(
        self,
        inputs: chex.Array,
        router_logits: chex.Array,
        pre_bias_logits: chex.Array,
        oneplus_weights: bool = False,
    ) -> list[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        num_experts = self.num_local_experts
        expt = self.num_experts_per_tok
        inputs_2d = jnp.reshape(inputs, (-1, inputs.shape[2]))
        weights, selected_experts = self.get_topk(router_logits, pre_bias_logits)
        if oneplus_weights:
            router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))
            inputs_2d = inputs_2d * router_scores.reshape(inputs_2d.shape[0], -1)

        flatten_selected_experts = jnp.ravel(selected_experts)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // expt
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        group_size = jnp.bincount(flatten_selected_experts, length=num_experts)

        expert_indices = jnp.arange(num_experts)
        mr = flatten_selected_experts.shape[0]
        sorted_experts = jnp.repeat(expert_indices, repeats=group_size, total_repeat_length=mr)

        return sorted_inputs, sorted_selected_experts, weights, group_size, sorted_experts

    def unpermute(
        self,
        intermediate: chex.Array,
        sorted_selected_experts: chex.Array,
        weights: chex.Array,
        batch_size: int,
        sequence_length: int,
        oneplus_weights: bool = False,
    ) -> chex.Array:
        expt = self.num_experts_per_tok

        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        reshaped_weights = jnp.reshape(weights, (-1, expt))
        reshaped_intermediate = jnp.reshape(unsort_intermediate, (reshaped_weights.shape[0], expt, -1))

        with jax.named_scope("unpremute_weights"):
            matmul_precision = lax.Precision(self.precision)
            if oneplus_weights:
                reshaped_weights = jnp.ones_like(reshaped_weights)
            output = jnp.einsum(
                "bte,bt->be",
                reshaped_intermediate.astype(jnp.float32),
                reshaped_weights.astype(jnp.float32),
                precision=matmul_precision,
                optimize=True,
            )
        return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

    def _vanilla_compute(
        self,
        hidden_states: chex.Array,
        out: chex.Array,
        weights: chex.Array,
        indices: chex.Array,
    ) -> chex.Array:
        """Computes the output of the MoE layer without routing."""
        for index in range(self.num_local_experts):
            _mdl = getattr(self, self._collection_name)[index]

            if self.config.use_scan_mlp:
                output = block_wise_ffn(_mdl, hidden_states, self.config.scan_mlp_chunk_size)
            else:
                output = _mdl(hidden_states)
            output_exp = jnp.sum(jnp.multiply(indices == index, weights), axis=-1)[:, :, None] * output
            out += output_exp
        return out

    def _create_grouped_matmul(self):
        def _fn(
            inputs: chex.Array,
            kernel: chex.Array,
            group_sizes: chex.Array,
            expert_assignments: chex.Array,
        ) -> chex.Array:
            pad_length = self.tiling_size[0]
            inputs_shape = inputs.shape
            if inputs.shape[0] != expert_assignments.shape[0]:
                raise ValueError("The number of input tokens must match the number of expert assignments!")

            if inputs_shape[0] % pad_length:
                pad_length = pad_length - inputs_shape[0] % pad_length
                inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

            inputs = inputs.astype(self.dtype)
            kernel = kernel.astype(self.dtype)

            if self.use_megablox:
                m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]

                output = pallas_grouped_matmul(
                    lhs=inputs,
                    rhs=kernel,
                    group_sizes=group_sizes,
                    preferred_element_type=self.dtype,
                    tiling=(min(self.tiling_size[0], m), min(self.tiling_size[1], k), min(self.tiling_size[2], n)),
                )
            else:
                rhs_inputs = kernel

                output = jax.lax.ragged_dot(
                    lhs=inputs,
                    rhs=rhs_inputs,
                    group_sizes=group_sizes,
                    preferred_element_type=self.dtype,
                )

            if inputs_shape[0] % self.tiling_size[0]:
                output = output[: inputs_shape[0]]
            return output

        return _fn

    def _sparse_compute(
        self,
        hidden_states: chex.Array,
        activation_fn: tp.Callable[[chex.Array], chex.Array],
        router_logits: chex.Array = None,
        pre_bias_logits: chex.Array | None = None,
    ) -> chex.Array:
        grouped_matmul = self._create_grouped_matmul()
        manager = self.config.partition_manager

        @partial(
            shard_map,
            mesh=self.config.mesh,
            in_specs=(
                manager.resolve(axes=[BATCH, EMPTY, EMPTY]),
                manager.resolve(axes=[BATCH, EMPTY]),
                manager.resolve(axes=[BATCH, EMPTY]) if pre_bias_logits is not None else None,
                manager.resolve(axes=[EXPERT_PARALLEL, EMPTY, TENSOR_PARALLEL]),
                manager.resolve(axes=[EXPERT_PARALLEL, EMPTY, TENSOR_PARALLEL]),
                manager.resolve(axes=[EXPERT_PARALLEL, TENSOR_PARALLEL, EMPTY]),
            ),
            out_specs=(
                manager.resolve(axes=[BATCH, EMPTY, EMPTY]),
                None,
            ),
            check_rep=False,
        )
        def wrapper(x, logits, pre_bias_logits, w0, w1, wo):
            batch_size, sequence_length, _ = x.shape
            x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(x, logits, pre_bias_logits)

            expert_shard_id = jax.lax.axis_index(self._ep)
            if self._available_ep > 1:
                local_expert_size = self.num_local_experts // self._available_ep
                reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                global_group_sizes = group_sizes

                x, local_sorted_indices, group_sizes, selected_experts = MoELayer._local_permute(
                    x,
                    global_group_sizes[None, :],
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=True,
                    global_sorted_experts=selected_experts,
                )

            layer_w0 = grouped_matmul(x, w0, group_sizes, selected_experts)
            layer_w1 = grouped_matmul(x, w1, group_sizes, selected_experts)
            layer_act = activation_fn(layer_w0)
            intermediate_layer = jnp.multiply(layer_act, layer_w1)
            intermediate_output = grouped_matmul(intermediate_layer, wo, group_sizes, selected_experts)

            if self._available_tp > 1:
                intermediate_output = jax.lax.psum_scatter(
                    intermediate_output,
                    self._tp,
                    scatter_dimension=1,
                    tiled=True,
                )

            if self._available_ep > 1:
                original_inputs_first_dim = batch_size * sequence_length * self.num_local_experts_per_tok
                if sorted_selected_experts.shape[0] != original_inputs_first_dim:
                    raise ValueError("original_inputs_first_dim does not match the original tensor shape!")
                output_shape = jnp.zeros(
                    (original_inputs_first_dim, self.config.emb_dim // self._available_tp),
                    dtype=intermediate_output.dtype,
                )

                input_offsets, send_sizes, output_offsets, recv_sizes = MoELayer._all_to_all_params(
                    reshaped_group_sizes,
                    expert_shard_id,
                    self._available_ep,
                    is_batch_sharded=False,
                )
                intermediate_output = jax.lax.ragged_all_to_all(
                    intermediate_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self._ep,
                )

            output = self.unpermute(
                intermediate_output,
                sorted_selected_experts,
                weights,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

            return output, None

        return wrapper(hidden_states, router_logits, pre_bias_logits, *self._collect_stacked_weights())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n  "
            f"num_local_experts={self.num_local_experts},\n  "
            f"num_experts_per_tok={self.num_experts_per_tok},\n  "
            f"tiling_size={self.tiling_size},\n  "
            f"n_routing_groups={self.n_routing_groups},\n  "
            f"topk_routing_group={self.topk_routing_group},\n  "
            f"use_megablox={self.use_megablox},\n  "
            f"collection_name={self._collection_name},\n  "
            f"aviailable_ep={self._available_ep},\n  "
            f"aviailable_tp={self._available_tp}\n)"
        )

    __str__ = __repr__
