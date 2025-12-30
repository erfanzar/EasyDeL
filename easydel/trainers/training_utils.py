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
import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.utils.helpers import check_bool_flag

SCAN_TRAINER = check_bool_flag("SCAN_TRAINER")
FAST_COMPILE = check_bool_flag("FAST_COMPILE")

QuantizationMode = tp.Literal["fp8", "int8", "nf4"]
DEFAULT_NF4_BLOCK_SIZE = 64


def _ste(x: jax.Array, q: jax.Array) -> jax.Array:
    q = q.astype(x.dtype)
    return x + lax.stop_gradient(q - x)


def _quantize_dequantize_int8(x: jax.Array) -> jax.Array:
    from eformer.ops.quantization.quantization_functions import dequantize_int8, quantize_int8

    q, scale = quantize_int8(x)
    return dequantize_int8(q, scale)


def _quantize_dequantize_nf4(x: jax.Array, *, block_size: int) -> jax.Array:
    from eformer.ops.quantization.quantization_functions import dequantize_nf4, quantize_and_pack_nf4

    if block_size <= 0:
        raise ValueError(f"`quantization_block` must be > 0 for NF4, got {block_size}.")
    original_last_dim = x.shape[-1]
    if original_last_dim % block_size != 0:
        pad_amount = block_size - (original_last_dim % block_size)
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_amount)]
        x = jnp.pad(x, pad_width, mode="constant", constant_values=0)

    packed, absmax = quantize_and_pack_nf4(x, block_size)
    deq = dequantize_nf4(packed, absmax, block_size)
    if deq.shape[-1] != original_last_dim:
        deq = deq[..., :original_last_dim]
    return deq


def _quantize_dequantize_fp8(x: jax.Array) -> jax.Array:
    # FP8 simulation via cast; STE is applied by the caller.
    return x.astype(jnp.float8_e4m3fn).astype(x.dtype)


def make_default_tensor_straight_through(
    quantization_mode: QuantizationMode,
    quantization_block: int | None = None,
) -> tp.Callable[[jax.Array], jax.Array]:
    """Create a per-tensor STE quantization function.

    Forward path uses a quantize->dequantize simulation, while gradients flow as
    if the transform is identity (STE).

    Notes:
        - `quantization_block` is used for NF4 block-wise quantization.
    """
    try:
        from eformer.ops.quantization import straight_through as eformer_straight_through  # type: ignore
    except Exception:  # pragma: no cover
        eformer_straight_through = None

    nf4_block_size: int | None = None
    if quantization_mode == "nf4":
        nf4_block_size = DEFAULT_NF4_BLOCK_SIZE if quantization_block is None else int(quantization_block)

    def tensor_straight_through(x: jax.Array) -> jax.Array:
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        if eformer_straight_through is not None and (quantization_mode != "nf4" or quantization_block is None):
            try:
                q = eformer_straight_through(x, method=quantization_mode)
            except TypeError:
                q = eformer_straight_through(x, quantization_mode)
            return _ste(x, q)

        if quantization_mode == "int8":
            qdq = _quantize_dequantize_int8
        elif quantization_mode == "nf4":
            def qdq(y):
                return _quantize_dequantize_nf4(y, block_size=tp.cast(int, nf4_block_size))
        elif quantization_mode == "fp8":
            qdq = _quantize_dequantize_fp8
        else:  # pragma: no cover
            raise ValueError(f"Unsupported `quantization_mode`: {quantization_mode!r}.")
        return _ste(x, qdq(x))

    return tensor_straight_through


def resolve_straight_through_emulator(
    *,
    quantization_mode: QuantizationMode | None,
    quantization_block: int | None,
    tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None,
) -> tp.Callable[[tp.Any], tp.Any] | None:
    """Resolve the graphstate-level straight-through emulator callable.

    Priority:
      1) `straight_through_emulator` (user-provided)
      2) `tensor_straight_through` mapped over graphstate
      3) default tensor STE built from (`quantization_mode`, `quantization_block`) and mapped over graphstate
      4) None (disabled)
    """
    if straight_through_emulator is not None:
        return straight_through_emulator

    if tensor_straight_through is None and quantization_mode is None:
        return None

    if tensor_straight_through is None:
        tensor_straight_through = make_default_tensor_straight_through(
            quantization_mode,
            quantization_block=quantization_block,
        )

    def _default_emulator(graphstate: tp.Any) -> tp.Any:
        return tu.tree_map(tensor_straight_through, graphstate)

    return _default_emulator


def resolve_total_steps(
    *,
    forced_steps: int | None,
    total_data_len: int | None,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    is_train: bool,
) -> int:
    """Resolve total train/eval steps from config and dataset length.

    Notes:
        - `forced_steps` is interpreted as *optimizer update* steps for training (i.e., after gradient accumulation).
        - When `forced_steps` is not provided, training steps are derived from dataset length and then divided by
          `gradient_accumulation_steps` to convert micro-batches into optimizer updates.
    """
    if forced_steps is not None:
        return int(forced_steps)

    if total_data_len is None:
        raise ValueError("`total_data_len` must be provided when `forced_steps` is None.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if num_epochs <= 0:
        return 0

    steps_per_epoch = (total_data_len + batch_size - 1) // batch_size
    steps = steps_per_epoch * num_epochs

    if is_train:
        if gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` must be > 0.")
        steps //= gradient_accumulation_steps

    return int(steps)


def make_assertions_and_get_sizes(
    batch: dict,
    gradient_accumulation_steps: int,
    batch_partition_spec: PartitionSpec | None = None,
) -> tuple[int, int, PartitionSpec]:
    """
    Validates the input parameters and computes the batch size, minibatch size, and batch partition specification.
    Args:
        batch (tp.Dict): A dictionary containing the batch data. The batch size is inferred from the
            first element's shape.
        gradient_accumulation_steps (int): The number of gradient accumulation steps. Must be greater than 0.
        batch_partition_spec (tp.Optional[PartitionSpec], optional): The partition specification for the batch.
            Defaults to None.
    Returns:
        tp.Tuple[int, int, PartitionSpec]: A tuple containing:
            - batch_size (int): The size of the batch.
            - minibatch_size (int): The size of the minibatch.
            - batch_partition_spec (PartitionSpec): The partition specification for the batch.
    Raises:
            ValueError: If `gradient_accumulation_steps` is not greater than 0.
            ValueError: If the batch size is not divisible by the gradient accumulation steps.
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be greater than 0.")

    batch_size = None
    for leaf in tu.tree_leaves(batch):
        if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) >= 1:
            batch_size = leaf.shape[0]
            break
    if batch_size is None:
        raise ValueError(
            "Unable to infer batch size from `batch`; expected at least one array leaf with a leading batch dimension."
        )

    minibatch_size = batch_size // gradient_accumulation_steps
    if minibatch_size * gradient_accumulation_steps != batch_size:
        raise ValueError("Batch size must be divisible by gradient accumulation steps.")
    if batch_partition_spec is None:
        batch_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
    return batch_size, minibatch_size, batch_partition_spec


def update_metrics(
    metrics: LossMetrics,
    learning_rate_fn: tp.Callable,
    step: int | jax.Array,
    gradients: jax.Array | None,
) -> LossMetrics:
    """
    Updates the given metrics with the current learning rate and gradient norms.

    Args:
            metrics (LossMetrics): An instance of LossMetrics to be updated.
            learning_rate_fn (tp.Callable): A callable that returns the learning rate given the current step.
            step (int | jax.Array): The current training step.
            gradients (Optional(jax.Array)): The gradients to compute norms from.

    Returns:
            LossMetrics: The updated metrics with learning rate and gradient norms.
    """
    if learning_rate_fn is not None:
        metrics.learning_rate = learning_rate_fn(step)
    if gradients is not None:
        grad_norms = tu.tree_map(jnp.linalg.norm, gradients)
        metrics.max_grad_norm = tu.tree_reduce(jnp.maximum, grad_norms)
        grad_size = tu.tree_reduce(jnp.add, tu.tree_map(jnp.size, grad_norms))
        grad_sum = tu.tree_reduce(jnp.add, tu.tree_map(jnp.sum, grad_norms))
        metrics.mean_grad_norm = grad_sum / grad_size
        metrics.grad_norms = grad_norms
    return metrics


def update_state_respectfully(
    state: EasyDeLState,
    gradients: jax.Array,
    loss_config: LossConfig,
    metrics: LossMetrics,
) -> EasyDeLState:
    """
    Updates the state of the model respectfully based on the provided gradients, loss configuration, and metrics.

    Args:
            state (EasyDeLState): The current state of the model.
            gradients (jax.Array): The gradients to be applied to the model's parameters.
            loss_config (LossConfig): Configuration for the loss, including conditions for breaking on NaN values.
            metrics (LossMetrics): Metrics containing the loss value.

    Returns:
            EasyDeLState: The updated state of the model.
    """
    if FAST_COMPILE:
        return state.apply_gradients(grads=gradients)
    else:

        def update_fn(args):
            state, gradients = args
            return state.apply_gradients(grads=gradients)

        def skip_fn(args):
            state, _ = args
            return state

        should_update = True
        if loss_config is not None:
            should_update = lax.cond(
                loss_config.break_on_nan,
                lambda x: lax.cond(
                    jnp.isnan(metrics.loss),
                    lambda _: False,
                    lambda _: True,
                    None,
                ),
                lambda x: True,
                None,
            )
        state = lax.cond(should_update, update_fn, skip_fn, (state, gradients))
        return state


def minibatch_call(
    state: EasyDeLState,
    batch: dict,
    minibatch_size: int,
    grad_fn: tp.Callable[[jax.Array, dict], tuple[jax.Array, LossMetrics]],
) -> tuple[jax.Array, LossMetrics]:
    """
    Processes batch in smaller chunks for gradient accumulation using jax.lax.scan.
    Uses eval_shape to initialize accumulator structures efficiently.
    """
    batch_size = None
    for leaf in tu.tree_leaves(batch):
        if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) >= 1:
            batch_size = leaf.shape[0]
            break
    if batch_size is None:
        raise ValueError(
            "Unable to infer batch size from `batch`; expected at least one array leaf with a leading batch dimension."
        )
    if minibatch_size <= 0:
        raise ValueError(f"`minibatch_size` must be > 0, got {minibatch_size}.")

    num_accum_steps = batch_size // minibatch_size
    if num_accum_steps * minibatch_size != batch_size:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by minibatch_size ({minibatch_size}) for gradient accumulation."
        )
    if num_accum_steps > 1:

        def reshape_to_minibatches(arr):
            """Reshape the batch into minibatches for accumulation."""
            batch_shape = (num_accum_steps, minibatch_size, *arr.shape[1:])
            return jnp.reshape(arr, batch_shape)

        batch = jax.tree_util.tree_map(reshape_to_minibatches, batch)

        (_, metrics_shape), grads_shape = jax.eval_shape(
            grad_fn,
            state.graphstate,
            jax.tree_util.tree_map(lambda x: x[0], batch),
        )

        init_acc = {
            "grads": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shape),
            "metrics": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape),
        }

        def accumulate_gradients(acc, minibatch):
            """Accumulate gradients and metrics for each minibatch."""
            (_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
            new_acc = {
                "grads": jax.tree_util.tree_map(jnp.add, acc["grads"], step_grads),
                "metrics": jax.tree_util.tree_map(jnp.add, acc["metrics"], step_aux),
            }
            return new_acc, step_aux

        final_acc, _aux = jax.lax.scan(
            accumulate_gradients,
            init_acc,
            batch,
            length=num_accum_steps,
        )
        gradients = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["grads"])
        metrics = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["metrics"])

    else:
        (_, metrics), gradients = grad_fn(state.graphstate, batch)

    return gradients, metrics
