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
import collections
import collections.abc
import inspect
import typing as tp
import warnings

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

QuantizationMode = tp.Literal[
    "nf4",
    "affine",
    "mxfp8",
    "nvfp8",
    "mxfp4",
    "nvfp4",
]
AFFINE_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 7, 8})
FIXED_QUANTIZATION_BITS_BY_MODE: dict[QuantizationMode, int] = {
    "nf4": 4,
    "mxfp4": 4,
    "nvfp4": 4,
    "mxfp8": 8,
    "nvfp8": 8,
}


def filter_kwargs_for_callable(
    callable_obj: tp.Callable[..., tp.Any],
    kwargs: collections.abc.Mapping[str, tp.Any],
) -> dict[str, tp.Any]:
    """Filter kwargs so only parameters accepted by ``callable_obj`` are forwarded.

    This prevents runtime failures when dataset batches carry auxiliary metadata
    fields (for example preference scores) that a model forward signature does
    not accept.
    """
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)

    accepted_keys = set(parameters.keys())
    return {key: value for key, value in kwargs.items() if key in accepted_keys}


def sanitize_model_call_kwargs(kwargs: collections.abc.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Normalize model call kwargs to avoid known incompatible combinations.

    Causal LM forwards generally accept either ``input_ids`` or ``inputs_embeds``,
    but not both at the same time. Prefer token IDs when both are present.
    """
    normalized_kwargs = dict(kwargs)
    if normalized_kwargs.get("input_ids", None) is not None and normalized_kwargs.get("inputs_embeds", None) is not None:
        normalized_kwargs.pop("inputs_embeds", None)
    return normalized_kwargs


GENERATION_MODEL_INPUT_KEYS = (
    "inputs_embeds",
    "position_ids",
    "token_type_ids",
    "cache_position",
    "decoder_position_ids",
    "pixel_values",
    "pixel_attention_mask",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_grid_hws",
    "image_sizes",
    "image_max_grid_size",
    "video_max_grid_size",
    "visual_pos_masks",
    "deepstack_visual_embeds",
    "rope_deltas",
    "mm_token_type_ids",
    "image_embeds",
    "video_embeds",
    "visual_embeds",
    "image_hidden_states",
    "video_hidden_states",
    "image_features",
    "video_features",
)

SHARED_GENERATION_MODEL_INPUT_KEYS = frozenset(
    {
        "image_max_grid_size",
        "video_max_grid_size",
    }
)

GROUPED_MULTIMODAL_MODEL_INPUT_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_grid_hws",
        "image_sizes",
    }
)

PROMPT_SCORING_EXTENSION_KEYS = frozenset(
    {
        "token_type_ids",
        "mm_token_type_ids",
        "pixel_attention_mask",
        "visual_pos_masks",
    }
)

PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS = frozenset(
    {
        "position_ids",
        "cache_position",
        "decoder_position_ids",
        "visual_pos_masks",
        "rope_deltas",
    }
)


def normalize_generation_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    model_callable: tp.Callable[..., tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Normalize model-side generation kwargs to a stable key set.

    Generation JITs work best when auxiliary model inputs use a fixed pytree
    structure. This helper keeps only known model input keys, filters them
    against the model forward signature when available, and fills missing keys
    with ``None`` so callers can safely pass the result into cached compiled
    functions.

    Args:
        kwargs: Raw mapping of model keyword arguments. May be ``None``.
        model_callable: Optional model forward callable used to filter keys
            against its signature.

    Returns:
        dict: Normalized dictionary with all ``GENERATION_MODEL_INPUT_KEYS``
            present (missing ones set to ``None``).
    """

    normalized = {key: None for key in GENERATION_MODEL_INPUT_KEYS}
    if not kwargs:
        return normalized

    extracted = {key: value for key, value in kwargs.items() if key in normalized and value is not None}
    if model_callable is not None:
        extracted = filter_kwargs_for_callable(model_callable, extracted)
    normalized.update(extracted)
    return normalized


def compact_generation_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
) -> dict[str, tp.Any]:
    """Drop ``None`` leaves from normalized generation model kwargs.

    Args:
        kwargs: Normalized generation model kwargs mapping. May be ``None``.

    Returns:
        dict: Compact dictionary with only non-``None`` entries.
    """

    if not kwargs:
        return {}
    return {key: value for key, value in kwargs.items() if value is not None}


def _flatten_grouped_multimodal_model_value(
    key: str,
    value: tp.Any,
) -> tp.Any:
    """Flatten grouped multimodal leaves before the actual model call."""

    if key not in GROUPED_MULTIMODAL_MODEL_INPUT_KEYS or not hasattr(value, "shape"):
        return value

    if key == "pixel_values" and value.ndim >= 5:
        return jnp.reshape(value, (-1, *value.shape[2:]))
    if key == "pixel_values_videos" and value.ndim >= 6:
        return jnp.reshape(value, (-1, *value.shape[2:]))
    if key in {"image_grid_thw", "video_grid_thw"} and value.ndim >= 3 and value.shape[-1] == 3:
        return jnp.reshape(value, (-1, 3))
    if key in {"image_grid_hws", "image_sizes"} and value.ndim >= 3 and value.shape[-1] == 2:
        return jnp.reshape(value, (-1, 2))
    return value


def _extend_prompt_scoring_value_to_sequence_length(
    key: str,
    value: tp.Any,
    *,
    prompt_length: int | None,
    target_sequence_length: int | None,
) -> tp.Any:
    """Extend prompt-only token-type style tensors across generated text tokens."""

    if key not in PROMPT_SCORING_EXTENSION_KEYS or not hasattr(value, "shape"):
        return value
    if prompt_length is None or target_sequence_length is None or value.ndim == 0:
        return value
    current_length = value.shape[-1]
    if current_length != prompt_length or target_sequence_length <= current_length:
        return value

    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_sequence_length - current_length)
    pad_value = 0.0 if jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating) else 0
    return jnp.pad(jnp.asarray(value), pad_width, mode="constant", constant_values=pad_value)


def prepare_generation_model_kwargs_for_call(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    target_sequence_length: int | None = None,
    prompt_length: int | None = None,
    flatten_grouped_multimodal: bool = True,
) -> dict[str, tp.Any]:
    """Prepare generation kwargs for a model call without losing prompt grouping upstream.

    Args:
        kwargs: Compact generation model kwargs mapping. May be ``None``.
        target_sequence_length: Target sequence length for extending prompt-only
            tensors. Defaults to ``None``.
        prompt_length: Length of the prompt portion. Defaults to ``None``.
        flatten_grouped_multimodal: Whether grouped multimodal leaves should be
            flattened to model-call layout immediately. Generation should keep
            prompt grouping until after any batch expansion.

    Returns:
        dict: Prepared dictionary ready for a model forward call.
    """

    prepared: dict[str, tp.Any] = {}
    for key, value in compact_generation_model_kwargs(kwargs).items():
        if flatten_grouped_multimodal:
            value = _flatten_grouped_multimodal_model_value(key, value)
        value = _extend_prompt_scoring_value_to_sequence_length(
            key,
            value,
            prompt_length=prompt_length,
            target_sequence_length=target_sequence_length,
        )
        prepared[key] = value
    return prepared


def strip_prompt_only_scoring_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
) -> dict[str, tp.Any]:
    """Drop prompt-only sequence-control kwargs before full-sequence scoring.

    Args:
        kwargs: Generation model kwargs mapping. May be ``None``.

    Returns:
        dict: Kwargs with prompt-only scoring keys removed.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if (
        compact_kwargs.get("deepstack_visual_embeds", None) is not None
        and compact_kwargs.get("visual_pos_masks", None) is not None
    ):
        excluded_keys = PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS - {"visual_pos_masks"}
    else:
        excluded_keys = PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS
    return {key: value for key, value in compact_kwargs.items() if key not in excluded_keys}


def extract_generation_model_kwargs(
    batch: collections.abc.Mapping[str, tp.Any] | None,
    *,
    model_callable: tp.Callable[..., tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Extract generation-related model inputs from a larger batch mapping.

    Args:
        batch: Batch mapping potentially containing generation model inputs.
            May be ``None``.
        model_callable: Optional model forward callable used to filter keys
            against its signature.

    Returns:
        dict: Compact dictionary of generation-related model inputs.
    """

    return compact_generation_model_kwargs(
        normalize_generation_model_kwargs(batch, model_callable=model_callable),
    )


def validate_prompt_aligned_generation_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    prompt_batch_size: int | None,
) -> None:
    """Validate that generation kwargs preserve prompt boundaries for GRPO-style scoring.

    Args:
        kwargs: Generation model kwargs to validate. May be ``None``.
        prompt_batch_size: Expected prompt batch size that kwargs should align with.

    Raises:
        ValueError: If any non-shared kwarg doesn't expose the prompt batch size
            on any axis.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if prompt_batch_size is None or prompt_batch_size <= 0:
        return

    for key, value in compact_kwargs.items():
        if key in SHARED_GENERATION_MODEL_INPUT_KEYS:
            continue
        if infer_prompt_batch_axis(value, prompt_batch_size, key=key) is not None:
            continue

        shape = getattr(value, "shape", None)
        raise ValueError(
            "GRPO requires prompt-aligned generation kwargs for scoring. "
            f"Got `{key}` with shape {shape!r}, which does not expose the prompt batch size "
            f"{prompt_batch_size} on any axis. Raw ragged multimodal inputs are not supported "
            "here; use prompt-batch-aligned embeddings/features or a single aligned item per prompt."
        )


def infer_prompt_batch_axis(
    value: tp.Any,
    prompt_batch_size: int | None,
    *,
    key: str | None = None,
) -> int | None:
    """Infer which axis of ``value`` is aligned with the prompt batch.

    Args:
        value: Array, list, or tuple whose batch axis is to be inferred.
        prompt_batch_size: The expected prompt batch size.
        key: Optional key name for special-case handling (e.g. ``"position_ids"``).

    Returns:
        int | None: The axis index aligned with the prompt batch, or ``None``
            if no axis matches.
    """

    if prompt_batch_size is None or prompt_batch_size <= 0:
        return None

    if isinstance(value, (list, tuple)):
        return 0 if len(value) == prompt_batch_size else None

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None

    if key == "position_ids" and len(shape) >= 3 and shape[0] == 3 and shape[1] == prompt_batch_size:
        return 1
    if shape[0] == prompt_batch_size:
        return 0
    return None


def repeat_prompt_aligned_model_value(
    value: tp.Any,
    repeat_factor: int,
    *,
    prompt_batch_size: int | None = None,
    key: str | None = None,
) -> tp.Any:
    """Repeat a prompt-aligned model input along its actual batch axis.

    Args:
        value: Array, list, or tuple to repeat.
        repeat_factor: Number of times to repeat each element.
        prompt_batch_size: Expected prompt batch size for axis inference.
        key: Optional key name for special-case axis inference.

    Returns:
        The repeated value along its batch axis, or the original value if
        ``repeat_factor <= 1`` or no batch axis is found.
    """

    if repeat_factor <= 1:
        return value

    batch_axis = infer_prompt_batch_axis(value, prompt_batch_size, key=key)
    if batch_axis is None:
        return value

    if isinstance(value, list):
        return [item for item in value for _ in range(repeat_factor)]
    if isinstance(value, tuple):
        return tuple(item for item in value for _ in range(repeat_factor))

    return jnp.repeat(jnp.asarray(value), repeat_factor, axis=batch_axis)


def slice_prompt_aligned_model_value(
    value: tp.Any,
    start: int,
    end: int,
    *,
    prompt_batch_size: int | None = None,
    key: str | None = None,
) -> tp.Any:
    """Slice a prompt-aligned model input along its actual batch axis.

    Args:
        value: Array, list, or tuple to slice.
        start: Start index of the slice.
        end: End index of the slice.
        prompt_batch_size: Expected prompt batch size for axis inference.
        key: Optional key name for special-case axis inference.

    Returns:
        The sliced value along its batch axis, or the original value if
        no batch axis is found.
    """

    batch_axis = infer_prompt_batch_axis(value, prompt_batch_size, key=key)
    if batch_axis is None:
        return value

    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]

    index = [slice(None)] * value.ndim
    index[batch_axis] = slice(start, end)
    return value[tuple(index)]


def slice_prompt_aligned_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any],
    start: int,
    end: int,
    *,
    prompt_batch_size: int | None = None,
) -> dict[str, tp.Any]:
    """Slice prompt-aligned model kwargs while preserving shared leaves.

    Args:
        kwargs: Mapping of model kwargs to slice.
        start: Start index of the slice.
        end: End index of the slice.
        prompt_batch_size: Expected prompt batch size for axis inference.

    Returns:
        dict: Sliced kwargs dictionary.
    """

    sliced: dict[str, tp.Any] = {}
    for key, value in kwargs.items():
        if value is None:
            sliced[key] = None
            continue
        sliced[key] = slice_prompt_aligned_model_value(
            value,
            start,
            end,
            prompt_batch_size=prompt_batch_size,
            key=key,
        )
    return sliced


def repeat_prompt_aligned_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    repeat_factor: int,
    *,
    prompt_batch_size: int | None = None,
) -> dict[str, tp.Any]:
    """Repeat prompt-aligned model kwargs to match completion-aligned batches.

    Args:
        kwargs: Mapping of model kwargs to repeat. May be ``None``.
        repeat_factor: Number of times to repeat each element along its batch axis.
        prompt_batch_size: Expected prompt batch size for axis inference.

    Returns:
        dict: Repeated kwargs dictionary.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if repeat_factor <= 1 or not compact_kwargs:
        return dict(compact_kwargs)

    repeated: dict[str, tp.Any] = {}
    for key, value in compact_kwargs.items():
        repeated[key] = repeat_prompt_aligned_model_value(
            value,
            repeat_factor,
            prompt_batch_size=prompt_batch_size,
            key=key,
        )
    return repeated


def _ste(x: jax.Array, q: jax.Array) -> jax.Array:
    q = q.astype(x.dtype)
    return x + lax.stop_gradient(q - x)


def make_default_tensor_straight_through(
    quantization_mode: QuantizationMode,
    quantization_group_size: int | None = None,
    quantization_bits: int | None = None,
    *,
    quantization_block: int | None = None,
) -> tp.Callable[[jax.Array], jax.Array]:
    """Create a per-tensor STE quantization function.

    Forward path uses a quantize->dequantize simulation, while gradients flow as
    if the transform is identity (STE).

    Notes:
        - `quantization_group_size` controls group-wise quantization where relevant.
        - `quantization_bits` controls bit-width for configurable formats (for example `affine`).
    """
    if quantization_block is not None:
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if quantization_group_size is None:
            quantization_group_size = quantization_block
        elif quantization_group_size != quantization_block:
            warnings.warn(
                f"Both `quantization_group_size` ({quantization_group_size}) and "
                f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
                FutureWarning,
                stacklevel=2,
            )

    if quantization_bits is not None:
        quantization_bits = int(quantization_bits)
        if quantization_bits <= 0:
            raise ValueError(f"`quantization_bits` must be > 0 when specified, got {quantization_bits}.")
        if quantization_mode == "affine" and quantization_bits not in AFFINE_SUPPORTED_BITS:
            bits_values = ", ".join(str(v) for v in sorted(AFFINE_SUPPORTED_BITS))
            raise ValueError(
                f"`quantization_bits` for `affine` must be one of {{{bits_values}}}, got {quantization_bits}."
            )
        required_bits = FIXED_QUANTIZATION_BITS_BY_MODE.get(quantization_mode, None)
        if required_bits is not None and quantization_bits != required_bits:
            raise ValueError(
                f"`quantization_bits` for `{quantization_mode}` must be {required_bits}, got {quantization_bits}."
            )

    from ejkernel.quantization import dequantize as ej_dequantize  # pyright: ignore[reportMissingTypeStubs]
    from ejkernel.quantization import quantize as ej_quantize  # pyright: ignore[reportMissingTypeStubs]

    from easydel.layers.quantization import QuantizationConfig
    from easydel.layers.quantization._configs import resolve_ejkernel_quant_params

    quantization_config = QuantizationConfig(
        dtype=quantization_mode,
        group_size=quantization_group_size,
        bits=quantization_bits,
    )
    mode, group_size, bits, needs_biases = resolve_ejkernel_quant_params(quantization_config)

    def _quantize_dequantize(y: jax.Array) -> jax.Array:
        input_dtype = y.dtype
        if y.ndim == 0:
            # Scalar leaves can appear in graphstate pytrees; keep them unchanged.
            return y.astype(input_dtype)
        was_vector = y.ndim == 1
        if was_vector:
            # ejkernel quantize expects rank >= 2.
            y = y[None, :]
        original_last_dim = y.shape[-1]
        if original_last_dim % group_size != 0:
            pad_amount = group_size - (original_last_dim % group_size)
            pad_width = [(0, 0)] * (y.ndim - 1) + [(0, pad_amount)]
            y = jnp.pad(y, pad_width, mode="constant", constant_values=0)

        if needs_biases:
            wq, scales, biases = ej_quantize(y, group_size=group_size, bits=bits, mode=mode, axis="col")
        else:
            wq, scales = ej_quantize(y, group_size=group_size, bits=bits, mode=mode, axis="col")
            biases = None
        dequantized = ej_dequantize(
            wq,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis="col",
        )
        if dequantized.shape[-1] != original_last_dim:
            dequantized = dequantized[..., :original_last_dim]
        if was_vector:
            dequantized = jnp.squeeze(dequantized, axis=0)
        return dequantized.astype(input_dtype)

    def tensor_straight_through(x: jax.Array) -> jax.Array:
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        return _ste(x, _quantize_dequantize(x))

    return tensor_straight_through


def resolve_straight_through_emulator(
    *,
    quantization_mode: QuantizationMode | None,
    quantization_group_size: int | None = None,
    quantization_bits: int | None = None,
    tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None,
    quantization_block: int | None = None,
) -> tp.Callable[[tp.Any], tp.Any] | None:
    """Resolve the graphstate-level straight-through emulator callable.

    Priority:
      1) `straight_through_emulator` (user-provided)
      2) `tensor_straight_through` mapped over graphstate
      3) default tensor STE built from (`quantization_mode`, `quantization_group_size`, `quantization_bits`) and
         mapped over graphstate
      4) None (disabled)
    """
    if quantization_block is not None:
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if quantization_group_size is None:
            quantization_group_size = quantization_block
        elif quantization_group_size != quantization_block:
            warnings.warn(
                f"Both `quantization_group_size` ({quantization_group_size}) and "
                f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
                FutureWarning,
                stacklevel=2,
            )

    if straight_through_emulator is not None:
        return straight_through_emulator

    if tensor_straight_through is None and quantization_mode is None:
        return None

    if tensor_straight_through is None:
        tensor_straight_through = make_default_tensor_straight_through(
            quantization_mode,
            quantization_group_size=quantization_group_size,
            quantization_bits=quantization_bits,
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
            dominant leading dimension across array leaves.
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

    batch_size = _infer_batch_size(batch)

    minibatch_size = batch_size // gradient_accumulation_steps
    if minibatch_size * gradient_accumulation_steps != batch_size:
        raise ValueError("Batch size must be divisible by gradient accumulation steps.")
    if batch_partition_spec is None:
        batch_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
    return batch_size, minibatch_size, batch_partition_spec


def _infer_batch_size(batch: tp.Any) -> int:
    """Infer batch size from the most common leading dimension in the batch pytree."""

    leading_dims = [
        int(leaf.shape[0])
        for leaf in tu.tree_leaves(batch)
        if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) >= 1
    ]
    if not leading_dims:
        raise ValueError(
            "Unable to infer batch size from `batch`; expected at least one array leaf with a leading batch dimension."
        )
    return collections.Counter(leading_dims).most_common(1)[0][0]


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
    loss_config: LossConfig | None,
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

    Rather than reshaping the whole batch into
    ``(num_accum_steps, minibatch_size, ...)``, this slices minibatches from the
    original batch inside the scan body. That is friendlier to sharded arrays
    coming from model forwards (for example cached teacher hidden states in
    distillation), where introducing a new leading accumulation axis can confuse
    downstream partitioned computations.
    """
    batch_size = _infer_batch_size(batch)
    if minibatch_size <= 0:
        raise ValueError(f"`minibatch_size` must be > 0, got {minibatch_size}.")

    num_accum_steps = batch_size // minibatch_size
    if num_accum_steps * minibatch_size != batch_size:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by minibatch_size "
            f"({minibatch_size}) for gradient accumulation."
        )
    if num_accum_steps > 1:

        def slice_minibatch(tree, start_index):
            """Extract one minibatch while leaving shared/global leaves untouched."""

            def _slice_leaf(arr):
                if not hasattr(arr, "shape") or arr.ndim == 0:
                    return arr
                if arr.shape[0] == batch_size:
                    return lax.dynamic_slice_in_dim(arr, start_index, minibatch_size, axis=0)
                return arr

            return jax.tree_util.tree_map(_slice_leaf, tree)

        shape_minibatch = slice_minibatch(batch, 0)

        (_, metrics_shape), grads_shape = jax.eval_shape(
            grad_fn,
            state.graphstate,
            shape_minibatch,
        )

        init_acc = {
            "grads": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shape),
            "metrics": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape),
        }

        def accumulate_gradients(acc, start_index):
            """Accumulate gradients and metrics for each minibatch."""
            minibatch = slice_minibatch(batch, start_index)
            (_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
            new_acc = {
                "grads": jax.tree_util.tree_map(jnp.add, acc["grads"], step_grads),
                "metrics": jax.tree_util.tree_map(jnp.add, acc["metrics"], step_aux),
            }
            return new_acc, step_aux

        start_indices = jnp.arange(num_accum_steps, dtype=jnp.int32) * minibatch_size
        final_acc, _aux = jax.lax.scan(
            accumulate_gradients,
            init_acc,
            start_indices,
            length=num_accum_steps,
        )
        gradients = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["grads"])
        metrics = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["metrics"])

    else:
        (_, metrics), gradients = grad_fn(state.graphstate, batch)

    return gradients, metrics  # type: ignore[return-value]
