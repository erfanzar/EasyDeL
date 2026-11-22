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

"""eSurge generation helpers behind the trainers' rollout path.

:meth:`~easydel.trainers.base_trainer.BaseTrainer.rollout` is the trainers' one generation entry point. Its eSurge back
end is built from the pieces in this module, which read the training arguments but need no trainer instance:

* :func:`resolve_sampling` turns the generation fields of the training arguments, plus per-call overrides, into
  :class:`~easydel.inference.SamplingParams`.
* :func:`engine_kwargs` builds the keyword arguments of ``model.get_esurge``. Its ``max_model_len`` is
  ``max(B, prompt_len + max_new_tokens) + reserve``, where ``B`` (:func:`generation_length_bound`) is fixed per
  trainer, so every call whose prompt width and token budget fit in ``B`` reuses one engine.
* :func:`prompt_token_id_rows` selects the real tokens of each pretokenized prompt row; eSurge is asked to condition
  on exactly these ids.
* :func:`outputs_to_arrays` turns the engine's request-ordered outputs into padded token arrays: completion ``i`` of
  prompt ``j`` becomes row ``j * n + i``.

eSurge itself is imported inside the functions that need it, never at module import.
"""

from __future__ import annotations

import math
import typing as tp
from collections.abc import Mapping

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

if tp.TYPE_CHECKING:
    from easydel.inference.sampling_params import SamplingParams

logger = get_logger(__name__)

DEFAULT_MAX_NEW_TOKENS = 1024
"""New-token budget used when neither ``generation_max_new_tokens`` nor ``max_completion_length`` is set."""

_DEFAULT_ESURGE_PAGE_SIZE = 128
_DEFAULT_CACHE_CAPACITY_MARGIN = 0.92


def _normalize_stop_strings(value: tp.Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_int_list(value: tp.Any) -> list[int]:
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item) for item in value]


def _normalize_logprobs(value: tp.Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else None
    return int(value)


_SAMPLING_OVERRIDES: dict[str, tuple[str, tp.Callable[[tp.Any], tp.Any]]] = {
    "max_new_tokens": ("max_tokens", int),
    "max_tokens": ("max_tokens", int),
    "temperature": ("temperature", float),
    "top_p": ("top_p", float),
    "min_p": ("min_p", float),
    "top_k": ("top_k", int),
    "num_return_sequences": ("n", int),
    "n": ("n", int),
    "best_of": ("best_of", int),
    "presence_penalty": ("presence_penalty", float),
    "frequency_penalty": ("frequency_penalty", float),
    "repetition_penalty": ("repetition_penalty", float),
    "seed": ("seed", int),
    "stop": ("stop", _normalize_stop_strings),
    "stop_token_ids": ("stop_token_ids", _normalize_int_list),
    "eos_token_id": ("stop_token_ids", _normalize_int_list),
    "stop_pattern": ("stop_pattern", str),
    "bad_words": ("bad_words", _normalize_stop_strings),
    "ignore_eos": ("ignore_eos", bool),
    "min_tokens": ("min_tokens", int),
    "logprobs": ("logprobs", _normalize_logprobs),
    "prompt_logprobs": ("prompt_logprobs", _normalize_logprobs),
    "detokenize": ("detokenize", bool),
    "skip_special_tokens": ("skip_special_tokens", bool),
    "spaces_between_special_tokens": ("spaces_between_special_tokens", bool),
    "include_stop_str_in_output": ("include_stop_str_in_output", bool),
    "ignore_stop_strings_in_reasoning": ("ignore_stop_strings_in_reasoning", bool),
    "truncate_prompt_tokens": ("truncate_prompt_tokens", int),
    "guided_decoding": ("guided_decoding", lambda v: v),
    "logit_bias": ("logit_bias", lambda v: {int(k): float(val) for k, val in dict(v).items()}),
    "allowed_token_ids": ("allowed_token_ids", _normalize_int_list),
    "extra_args": ("extra_args", dict),
}


class EsurgeGenerationArrays(tp.NamedTuple):
    """Token arrays and per-completion records rebuilt from eSurge outputs.

    Attributes:
        prompt_ids: Prompt token IDs, one row per prompt: the caller's rows when it passed ``input_ids``,
            otherwise the engine's prompt tokens left-padded.
        prompt_mask: Attention mask aligned with ``prompt_ids``.
        sequences: Prompt row followed by completion, one row per completion, sliced to
            ``prompt_len + completion_len``.
        completion_ids: Right-padded completion token IDs, one row per completion.
        completion_mask: Attention mask aligned with ``completion_ids``.
        output_records: Visible completion text per completion.
        reasoning_records: Parsed reasoning text per completion.
        tool_call_records: Parsed tool-call payloads per completion.
        raw_output_records: Raw completion text per completion.
        finish_reason_records: Per-completion stop reason.
        completion_prompts: Prompt objects aligned one-to-one with completions.
        sampling_per_token_logps: Behaviour log-probability of each sampled completion token, zero-padded, or
            ``None`` when the engine returned no sampled-token scores.
        completion_weight_versions: ``int32`` engine weight version that produced each completion token, aligned
            with ``completion_ids`` (``-1`` on padding and on completions without version provenance), or ``None``
            when no completion carried ``weight_version_segments``.
    """

    prompt_ids: jax.Array
    prompt_mask: jax.Array
    sequences: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    output_records: list[str]
    reasoning_records: list[str | None]
    tool_call_records: list[list | None]
    raw_output_records: list[str]
    finish_reason_records: list[str | None]
    completion_prompts: list[str | list[dict[str, str]]]
    sampling_per_token_logps: jax.Array | None = None
    completion_weight_versions: jax.Array | None = None


def expand_weight_version_segments(segments: tp.Sequence[tp.Sequence[int]] | None, length: int) -> list[int] | None:
    """Expand ``(start_token_index, version)`` segments into one version per token.

    Args:
        segments: Run-length segments in token order, as on ``CompletionOutput.weight_version_segments``.
        length: Number of tokens to cover.

    Returns:
        ``length`` versions (``-1`` before the first segment), or ``None`` when ``segments`` is ``None``.
    """
    if segments is None:
        return None
    versions = [-1] * length
    ordered = sorted((int(start), int(version)) for start, version in segments)
    for position, (start, version) in enumerate(ordered):
        stop = ordered[position + 1][0] if position + 1 < len(ordered) else length
        for index in range(max(0, start), min(stop, length)):
            versions[index] = version
    return versions


def _collect_sampling_mapping(
    params: dict[str, tp.Any],
    mapping: tp.Mapping[str, tp.Any] | None,
) -> bool | None:
    """Fold a generation-kwarg mapping into ``params`` and return any explicit ``do_sample``."""
    if not isinstance(mapping, Mapping):
        return None
    do_sample: bool | None = None
    if "do_sample" in mapping and mapping["do_sample"] is not None:
        do_sample = bool(mapping["do_sample"])
    layer_stop_token_ids: list[int] | None = None
    for key, value in mapping.items():
        if value is None or key == "do_sample":
            continue
        spec = _SAMPLING_OVERRIDES.get(key)
        if spec is None:
            continue
        attr, cast = spec
        converted = cast(value)
        if attr == "stop_token_ids":
            if layer_stop_token_ids is None:
                layer_stop_token_ids = []
            for token_id in converted:
                if token_id not in layer_stop_token_ids:
                    layer_stop_token_ids.append(token_id)
        else:
            params[attr] = converted
    if layer_stop_token_ids is not None:
        params["stop_token_ids"] = layer_stop_token_ids
    return do_sample


def apply_sampling_overrides(
    sampling_params: SamplingParams,
    overrides: tp.Mapping[str, tp.Any] | None = None,
) -> SamplingParams:
    """Return a validated copy of ``sampling_params`` with ``overrides`` applied.

    Reconstructs :class:`~easydel.inference.SamplingParams` so ``__post_init__``
    validation (including temperature clamping and greedy normalization) always runs.
    """
    import dataclasses

    from easydel.inference.sampling_params import SamplingParams

    if not overrides:
        return sampling_params
    params = {
        field.name: getattr(sampling_params, field.name)
        for field in dataclasses.fields(SamplingParams)
        if field.init
    }
    do_sample = _collect_sampling_mapping(params, overrides)
    if do_sample is False:
        params["temperature"] = 0.0
    return SamplingParams(**params)


def resolve_sampling(
    args: tp.Any,
    overrides: tp.Mapping[str, tp.Any] | None = None,
    *,
    generate_kwargs: tp.Mapping[str, tp.Any] | None = None,
) -> SamplingParams:
    """Build the sampling parameters of one rollout from the training arguments and per-call overrides.

    ``max_tokens`` comes from ``generation_max_new_tokens``, else ``max_completion_length``, else
    :data:`DEFAULT_MAX_NEW_TOKENS`. Unset temperature, top-p and top-k fall back to 0.7, 0.95 and 64.
    Supported entries from ``args.generation_extra_kwargs``, ``overrides``, and ``generate_kwargs``
    are folded in that precedence order before constructing :class:`~easydel.inference.SamplingParams`,
    so ``__post_init__`` validation (such as the ``< 1e-2`` temperature clamp) always runs. A resolved
    ``do_sample`` of ``False`` forces greedy decoding with temperature 0.

    Args:
        args: Training arguments (any object exposing the ``generation_*`` fields).
        overrides: Optional per-call overrides keyed by generation-config or ``SamplingParams`` name.
        generate_kwargs: Optional extra per-call generation kwargs taking precedence over ``overrides``.

    Returns:
        SamplingParams: A new validated object; the arguments are not modified.
    """
    from easydel.inference.sampling_params import SamplingParams

    max_tokens = getattr(args, "generation_max_new_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(args, "max_completion_length", None)
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_NEW_TOKENS

    params: dict[str, tp.Any] = {
        "max_tokens": int(max_tokens),
        "temperature": (
            0.7 if getattr(args, "generation_temperature", None) is None else float(args.generation_temperature)
        ),
        "top_p": 0.95 if getattr(args, "generation_top_p", None) is None else float(args.generation_top_p),
        "top_k": 64 if getattr(args, "generation_top_k", None) is None else int(args.generation_top_k),
        "presence_penalty": float(getattr(args, "generation_presence_penalty", 0.0) or 0.0),
        "frequency_penalty": float(getattr(args, "generation_frequency_penalty", 0.0) or 0.0),
        "repetition_penalty": float(getattr(args, "generation_repetition_penalty", 1.0) or 1.0),
        "n": int(getattr(args, "generation_num_return_sequences", None) or 1),
    }

    do_sample: bool | None = getattr(args, "generation_do_sample", None)
    for layer in (
        getattr(args, "generation_extra_kwargs", None),
        overrides,
        generate_kwargs,
    ):
        layer_do_sample = _collect_sampling_mapping(params, layer)
        if layer_do_sample is not None:
            do_sample = layer_do_sample

    if do_sample is False:
        params["temperature"] = 0.0
    return SamplingParams(**params)


def generation_length_bound(args: tp.Any) -> int:
    """Return ``B``, the trainer's fixed bound on prompt plus generated tokens.

    ``B`` is at least ``max_length`` (when set) and the prompt bound
    (``max_prompt_length``, else ``max_length``) plus the new-token budget
    (``generation_max_new_tokens``, else ``max_completion_length``, else
    :data:`DEFAULT_MAX_NEW_TOKENS`).

    Args:
        args: Training arguments.

    Returns:
        int: The bound in tokens.
    """
    prompt_bound = getattr(args, "max_prompt_length", None)
    if prompt_bound is None:
        prompt_bound = getattr(args, "max_length", None)
    new_tokens = getattr(args, "generation_max_new_tokens", None)
    if new_tokens is None:
        new_tokens = getattr(args, "max_completion_length", None)
    if new_tokens is None:
        new_tokens = DEFAULT_MAX_NEW_TOKENS
    bound = int(prompt_bound or 0) + int(new_tokens)
    max_length = getattr(args, "max_length", None)
    if max_length is not None:
        bound = max(bound, int(max_length))
    return bound


def clamp_new_tokens_to_bound(
    args: tp.Any,
    *,
    prompt_len: int,
    max_new_tokens: int,
) -> int:
    """Clamp per-turn ``max_new_tokens`` so ``prompt_len + max_new_tokens`` stays within ``B``.

    Multi-turn rollout loops (agentic trajectories and GRPO tool-calling follow-ups)
    grow ``prompt_len`` across turns. Keeping ``prompt_len + max_new_tokens`` at or
    below :func:`generation_length_bound` guarantees :func:`engine_kwargs` computes a
    single constant ``max_model_len`` instead of allocating a fresh cached eSurge
    engine + KV cache for every distinct prompt width.

    Args:
        args: Training arguments.
        prompt_len: Padded prompt token width for this generation call.
        max_new_tokens: Requested new-token budget for this turn.

    Returns:
        int: Effective new-token budget, clamped to ``[1, max(1, B - prompt_len)]``
        when ``B > 0`` and ``prompt_len + max_new_tokens > B``.
    """
    requested = max(1, int(max_new_tokens))
    bound = generation_length_bound(args)
    if bound <= 0:
        return requested
    remaining = max(1, bound - int(prompt_len))
    return min(requested, remaining)


def _effective_max_num_seqs(args: tp.Any) -> int:
    """Compute default concurrent sequence count accounting for eval and agentic rollouts."""
    batch_size = max(
        int(getattr(args, "total_batch_size", 0) or 0),
        int(getattr(args, "eval_batch_size", 0) or 0),
    )
    num_generations = max(
        int(getattr(args, "generation_num_return_sequences", None) or 1),
        int(getattr(args, "num_generations", None) or 1),
        int(getattr(args, "num_generations_eval", None) or 1),
    )
    base_concurrency = batch_size * num_generations
    num_env_groups = int(getattr(args, "num_env_groups", 0) or 0)
    group_size = int(getattr(args, "group_size", 0) or 0)
    agentic_concurrency = num_env_groups * group_size
    return max(base_concurrency, agentic_concurrency)


def engine_kwargs(
    args: tp.Any,
    *,
    processor: tp.Any = None,
    prompt_len: int | None = None,
    max_new_tokens: int | None = None,
    max_num_seqs: int | None = None,
) -> dict[str, tp.Any]:
    """Build the keyword arguments of ``model.get_esurge`` for a trainer.

    The ``esurge_*`` training arguments map onto engine settings. When neither ``esurge_max_cache_tokens`` nor
    ``esurge_hbm_utilization`` is set, the KV pool is sized to hold ``max_num_seqs`` concurrent full-length
    sequences with page-alignment rounding and headroom for eSurge's ``cache_capacity_margin``.

    With ``prompt_len`` and ``max_new_tokens`` given, ``max_model_len`` is
    ``max(generation_length_bound(args), prompt_len + max_new_tokens) + reserve``, where ``reserve`` is
    ``max_num_seqs`` (eSurge's default reserve). The engine length is part of the engine cache key, so every call
    whose prompt width and budget fit in the bound reuses the same engine. Admission caps a request's new tokens
    only beyond ``max_model_len - prompt_len - reserve``, which the bound never makes smaller than
    ``prompt_len + max_new_tokens`` needs.

    Args:
        args: Training arguments.
        processor: Tokenizer or processor handed to the engine, if any.
        prompt_len: Padded prompt width of the call (``P``): the caller's ``input_ids`` width, else
            ``max_length``. Omit, with ``max_new_tokens``, to leave ``max_model_len`` to ``get_esurge``.
        max_new_tokens: Token budget of the call.
        max_num_seqs: Concurrency used when ``esurge_max_num_seqs`` is unset; defaults to the
            effective rollout concurrency across train, eval, and agentic groups.

    Returns:
        dict: Keyword arguments for ``get_esurge``.
    """
    esurge_kwargs: dict[str, tp.Any] = {}
    if args.esurge_hbm_utilization is not None:
        esurge_kwargs["hbm_utilization"] = args.esurge_hbm_utilization
    if args.esurge_max_num_seqs is not None:
        esurge_kwargs["max_num_seqs"] = args.esurge_max_num_seqs
    elif max_num_seqs is not None:
        esurge_kwargs["max_num_seqs"] = int(max_num_seqs)
    else:
        esurge_kwargs["max_num_seqs"] = _effective_max_num_seqs(args)
    if args.esurge_min_input_pad is not None:
        esurge_kwargs["min_input_pad"] = args.esurge_min_input_pad
    if args.esurge_page_size is not None:
        esurge_kwargs["page_size"] = args.esurge_page_size
    explicit_cache_tokens = getattr(args, "esurge_max_cache_tokens", None)
    if explicit_cache_tokens is not None:
        esurge_kwargs["max_cache_tokens"] = int(explicit_cache_tokens)
    elif args.esurge_hbm_utilization is None:
        seq_len = int(getattr(args, "max_length", 0) or 0)
        if seq_len <= 0:
            seq_len = generation_length_bound(args)
        concurrency = int(esurge_kwargs.get("max_num_seqs", 0) or 0)
        if seq_len > 0 and concurrency > 0:
            page_size = max(1, int(getattr(args, "esurge_page_size", None) or _DEFAULT_ESURGE_PAGE_SIZE))
            margin = float(getattr(args, "esurge_cache_capacity_margin", None) or _DEFAULT_CACHE_CAPACITY_MARGIN)
            margin = min(max(margin, 1e-3), 1.0)
            pages_per_seq = math.ceil(seq_len / page_size)
            needed_pages = pages_per_seq * concurrency
            pre_margin_pages = math.ceil(needed_pages / margin)
            while int(pre_margin_pages * margin) < needed_pages:
                pre_margin_pages += 1
            computed_cache_tokens = pre_margin_pages * page_size
            esurge_kwargs["max_cache_tokens"] = computed_cache_tokens
            logger.info(
                "eSurge KV-cache auto-sized (no hbm_utilization/max_cache_tokens set): "
                "max_cache_tokens=%s (seq_len=%s, max_num_seqs=%s, page_size=%s, margin=%.2f).",
                computed_cache_tokens,
                seq_len,
                concurrency,
                page_size,
                margin,
            )
    if hasattr(args, "esurge_silent_mode"):
        esurge_kwargs["silent_mode"] = args.esurge_silent_mode
    if hasattr(args, "esurge_runner_verbose"):
        esurge_kwargs["runner_verbose"] = args.esurge_runner_verbose
    if args.esurge_max_num_batched_tokens is not None:
        esurge_kwargs["max_num_batched_tokens"] = args.esurge_max_num_batched_tokens
    else:
        raw_max_len = getattr(args, "max_length", None)
        if raw_max_len is None:
            raw_max_len = generation_length_bound(args) or 4096
        resolved_max_len = int(raw_max_len)
        esurge_kwargs["max_num_batched_tokens"] = min(resolved_max_len, 4096) if resolved_max_len > 0 else 4096
    if args.esurge_enable_prefix_caching is not None:
        esurge_kwargs["enable_prefix_caching"] = args.esurge_enable_prefix_caching
    if args.esurge_data_parallelism_axis is not None:
        esurge_kwargs["data_parallelism_axis"] = args.esurge_data_parallelism_axis
    if getattr(args, "esurge_async_scheduling", None) is not None:
        esurge_kwargs["async_scheduling"] = args.esurge_async_scheduling
    if getattr(args, "esurge_overlap_execution", None) is not None:
        esurge_kwargs["overlap_execution"] = args.esurge_overlap_execution
    if args.esurge_max_num_seq_buckets is not None:
        esurge_kwargs["max_num_seq_buckets"] = [int(v) for v in args.esurge_max_num_seq_buckets]
    if processor is not None:
        esurge_kwargs["tokenizer"] = processor
    if prompt_len is not None and max_new_tokens is not None:
        # eSurge reserves max_num_seqs tokens of the context budget by default; without them in max_model_len, a
        # tightly packed prompt_len + max_new_tokens would have its new tokens capped by the reserve.
        reserve_tokens = int(esurge_kwargs.get("max_num_seqs", 0) or 0)
        needed = int(prompt_len) + int(max_new_tokens)
        esurge_kwargs["max_model_len"] = max(generation_length_bound(args), needed) + reserve_tokens
    return esurge_kwargs


def _flatten_prompt_token_ids(prompt_token_ids: tp.Any) -> list[int]:
    """Flatten eSurge ``prompt_token_ids`` supporting both flat ``list[int]`` and nested ``list[list[int]]``."""
    if prompt_token_ids is None:
        return []
    if isinstance(prompt_token_ids, np.ndarray):
        return [int(token) for token in prompt_token_ids.reshape(-1).tolist()]
    flattened: list[int] = []
    for item in prompt_token_ids:
        if isinstance(item, (int, np.integer)):
            flattened.append(int(item))
        elif item is not None:
            flattened.extend(int(token) for token in item)
    return flattened


def _materialize_rollout_array(arr: np.ndarray, mesh: tp.Any) -> jax.Array:
    """Convert a host rollout array into a JAX array, globally replicated on ``mesh`` when provided."""
    if mesh is None:
        return jnp.array(arr)
    from easydel.infra.sharding import replicated_named_sharding

    target = replicated_named_sharding(mesh)
    return jax.make_array_from_callback(arr.shape, target, lambda index: arr[index])


def prompt_token_id_rows(
    input_ids: tp.Any,
    attention_mask: tp.Any = None,
    pad_token_id: int | None = None,
) -> list[list[int]]:
    """Return the real token ids of each pretokenized prompt row.

    Args:
        input_ids: Prompt ids of shape ``(batch, seq_len)`` or ``(seq_len,)``, host or device array.
        attention_mask: Optional mask of the same shape; its nonzero positions select a row's tokens.
        pad_token_id: Read only without a mask: tokens equal to it are dropped.

    Returns:
        list[list[int]]: One list of Python ints per row, in row order.

    Raises:
        ValueError: If the mask's shape differs from the ids' shape.
    """
    ids = np.asarray(input_ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    if attention_mask is not None:
        mask = np.asarray(attention_mask).astype(bool)
        if mask.ndim == 1:
            mask = mask[None, :]
        if mask.shape != ids.shape:
            raise ValueError(f"attention_mask shape {mask.shape} does not match input_ids shape {ids.shape}.")
        return [[int(token) for token in row[keep]] for row, keep in zip(ids, mask, strict=True)]
    if pad_token_id is None:
        return [[int(token) for token in row] for row in ids]
    return [[int(token) for token in row if token != pad_token_id] for row in ids]


def outputs_to_arrays(
    outputs: tp.Sequence[tp.Any],
    *,
    input_ids: tp.Any,
    attention_mask: tp.Any,
    return_prompts: tp.Sequence[tp.Any] | None,
    pad_token_id: int,
    max_seq_len: int,
    max_new_tokens: int,
    mesh: tp.Any = None,
) -> EsurgeGenerationArrays:
    """Build padded prompt, completion and sequence arrays from eSurge outputs.

    ``outputs`` are in request order, one per prompt (eSurge's ``generate`` returns them so), and each holds its
    ``n`` completions in sample order, so completion ``i`` of prompt ``j`` becomes row ``j * n + i``. Nothing is
    matched or reordered.

    With caller ``input_ids``, ``prompt_ids``/``prompt_mask`` match the caller's rows unless eSurge's admission
    layer truncated ``output.prompt_token_ids``, in which case the admitted prompt tokens are left-padded to the
    caller's width so learner conditioning matches the sampled logprobs. Without caller ``input_ids``, prompt
    rows are the engine's prompt tokens left-padded to ``max_seq_len``.

    Args:
        outputs: eSurge ``RequestOutput`` objects in request order.
        input_ids: Caller prompt ids of shape ``(num_prompts, P)``, or ``None`` for text prompts.
        attention_mask: Mask aligned with ``input_ids``, or ``None`` (all ones).
        return_prompts: Prompts used as each completion's ``source_prompt`` when an output carries no ``prompt``.
        pad_token_id: Pad id for left/right padding.
        max_seq_len: Prompt width for text prompts; with ``input_ids`` the caller's width is used.
        max_new_tokens: Completion width; longer completions are truncated.
        mesh: Optional JAX/SpectraX mesh; when provided, output arrays are constructed directly as
            globally replicated arrays via ``jax.make_array_from_callback`` (multi-host safe).

    Returns:
        EsurgeGenerationArrays: The arrays and per-completion records.

    Raises:
        ValueError: If caller rows and outputs differ in number, or the sampled-token log-probability payload is
            incomplete or present for only some completions.
        RuntimeError: If the engine returned no completions.
    """
    prompt_id_rows: list[list[int]] = []
    prompt_mask_rows: list[list[int]] = []
    sequence_rows: list[list[int]] = []
    completion_id_rows: list[list[int]] = []
    completion_mask_rows: list[list[int]] = []
    sampling_logps_rows: list[list[float] | None] = []
    weight_version_rows: list[list[int] | None] = []
    output_records: list[str] = []
    reasoning_records: list[str | None] = []
    tool_call_records: list[list | None] = []
    raw_output_records: list[str] = []
    finish_reason_records: list[str | None] = []
    completion_prompts: list[str | list[dict[str, str]]] = []

    caller_rows = input_ids is not None
    has_caller_attention_mask = attention_mask is not None
    if caller_rows:
        base_prompt_ids = np.asarray(input_ids, dtype=np.int32)
        if base_prompt_ids.ndim == 1:
            base_prompt_ids = base_prompt_ids[None, :]
        if attention_mask is None:
            base_prompt_mask = np.ones_like(base_prompt_ids, dtype=np.int32)
        else:
            base_prompt_mask = np.asarray(attention_mask, dtype=np.int32)
            if base_prompt_mask.ndim == 1:
                base_prompt_mask = base_prompt_mask[None, :]
        prompt_id_rows = [[int(token) for token in row] for row in base_prompt_ids]
        prompt_mask_rows = [[int(keep) for keep in row] for row in base_prompt_mask]
        if len(outputs) != len(prompt_id_rows):
            raise ValueError(
                f"eSurge returned {len(outputs)} request outputs for {len(prompt_id_rows)} prompt rows; "
                "outputs must be one per prompt, in request order."
            )
        max_seq_len = int(base_prompt_ids.shape[-1])
    max_total_len = max_seq_len + max_new_tokens

    for output_idx, output in enumerate(outputs):
        admitted_prompt_tokens = _flatten_prompt_token_ids(getattr(output, "prompt_token_ids", None))
        if caller_rows:
            if has_caller_attention_mask:
                expected_tokens = [
                    token
                    for token, keep in zip(prompt_id_rows[output_idx], prompt_mask_rows[output_idx], strict=True)
                    if keep
                ]
            else:
                expected_tokens = [token for token in prompt_id_rows[output_idx] if token != pad_token_id]
            if admitted_prompt_tokens and admitted_prompt_tokens != expected_tokens:
                logger.warning(
                    "eSurge admitted prompt tokens (len=%d) differ from caller prompt tokens (len=%d) on row %d; "
                    "reconciling prompt_ids/prompt_mask to match engine conditioning.",
                    len(admitted_prompt_tokens),
                    len(expected_tokens),
                    output_idx,
                )
                reconciled_tokens = admitted_prompt_tokens[-max_seq_len:]
                prompt_padding = max_seq_len - len(reconciled_tokens)
                prompt_id_rows[output_idx] = [pad_token_id] * prompt_padding + reconciled_tokens
                prompt_mask_rows[output_idx] = [0] * prompt_padding + [1] * len(reconciled_tokens)
            sequence_prefix = prompt_id_rows[output_idx]
        else:
            base_prompt_tokens = admitted_prompt_tokens[:max_seq_len]
            prompt_padding = max_seq_len - len(base_prompt_tokens)
            # Left-pad prompts for RL training.
            sequence_prefix = [pad_token_id] * prompt_padding + base_prompt_tokens
            prompt_id_rows.append(sequence_prefix)
            prompt_mask_rows.append([0] * prompt_padding + [1] * len(base_prompt_tokens))

        source_prompt = getattr(output, "prompt", return_prompts[output_idx] if return_prompts else None)
        num_output_completions = len(output.outputs)

        # Each request output holds its n completions in sample order.
        for completion in output.outputs:
            completion_prompts.append(source_prompt)
            completion_text = getattr(completion, "text", None)
            if not isinstance(completion_text, str):
                if num_output_completions == 1:
                    output_get_text = getattr(output, "get_text", None)
                    output_fallback_text = output_get_text() if callable(output_get_text) else ""
                    completion_text = getattr(output, "accumulated_text", "") or output_fallback_text
                else:
                    completion_text = ""
            output_records.append(completion_text)

            completion_reasoning = getattr(completion, "reasoning_content", None)
            reasoning_records.append(completion_reasoning if isinstance(completion_reasoning, str) else None)

            completion_tool_calls = getattr(completion, "tool_calls", None)
            tool_call_records.append(completion_tool_calls if completion_tool_calls is not None else None)

            completion_raw_text = getattr(completion, "raw_text", None)
            if not isinstance(completion_raw_text, str):
                if num_output_completions == 1:
                    output_get_text = getattr(output, "get_text", None)
                    output_fallback_text = output_get_text() if callable(output_get_text) else ""
                    completion_raw_text = (
                        getattr(output, "raw_accumulated_text", "")
                        or getattr(output, "accumulated_text", "")
                        or output_fallback_text
                    )
                else:
                    completion_raw_text = completion_text
            raw_output_records.append(completion_raw_text)

            completion_finish_reason = getattr(completion, "finish_reason", None)
            finish_reason_records.append(completion_finish_reason if isinstance(completion_finish_reason, str) else None)

            completion_tokens: list[int] = []
            if completion:
                completion_tokens = list(getattr(completion, "token_ids", []) or [])

            if len(completion_tokens) > max_new_tokens:
                completion_tokens = completion_tokens[:max_new_tokens]

            completion_len = len(completion_tokens)
            completion_padding = max_new_tokens - completion_len
            completion_id_rows.append(completion_tokens + [pad_token_id] * completion_padding)
            completion_mask_rows.append([1] * completion_len + [0] * completion_padding)
            token_versions = expand_weight_version_segments(
                getattr(completion, "weight_version_segments", None), completion_len
            )
            weight_version_rows.append(None if token_versions is None else token_versions + [-1] * completion_padding)
            token_logprobs = getattr(completion, "logprobs", None)
            if token_logprobs is None:
                sampling_logps_rows.append(None)
            else:
                if len(token_logprobs) < completion_len:
                    raise ValueError("eSurge returned fewer sampled-token log probabilities than completion tokens.")
                sampled_logps = []
                for token, candidates in zip(completion_tokens, token_logprobs[:completion_len], strict=True):
                    if token not in candidates:
                        raise ValueError(f"eSurge log probabilities omit sampled token {token}.")
                    value = candidates[token]
                    sampled_logps.append(float(getattr(value, "logprob", value)))
                sampling_logps_rows.append(sampled_logps + [0.0] * completion_padding)

            sequence_tokens = sequence_prefix + completion_tokens
            sequence_rows.append(sequence_tokens + [pad_token_id] * (max_total_len - len(sequence_tokens)))

    if not sequence_rows:
        raise RuntimeError("eSurge generation returned no completions")
    if not prompt_id_rows or not prompt_mask_rows:
        raise RuntimeError("Could not determine prompt token metadata for eSurge generation")

    prompt_ids = _materialize_rollout_array(np.asarray(prompt_id_rows, dtype=np.int32), mesh)
    prompt_mask = _materialize_rollout_array(np.asarray(prompt_mask_rows, dtype=np.int32), mesh)
    sequences = _materialize_rollout_array(np.asarray(sequence_rows, dtype=np.int32), mesh)
    completion_ids = _materialize_rollout_array(np.asarray(completion_id_rows, dtype=np.int32), mesh)
    completion_mask = _materialize_rollout_array(np.asarray(completion_mask_rows, dtype=np.int32), mesh)
    total_seq_len = prompt_ids.shape[-1] + completion_ids.shape[-1]
    sequences = sequences[:, :total_seq_len]
    sampling_per_token_logps = None
    if any(row is not None for row in sampling_logps_rows):
        if any(row is None for row in sampling_logps_rows):
            raise ValueError("eSurge returned sampled-token log probabilities for only some completions.")
        sampling_per_token_logps = _materialize_rollout_array(
            np.asarray(sampling_logps_rows, dtype=np.float32),
            mesh,
        )
    completion_weight_versions = None
    if any(row is not None for row in weight_version_rows):
        filled = [row if row is not None else [-1] * max_new_tokens for row in weight_version_rows]
        completion_weight_versions = _materialize_rollout_array(
            np.asarray(filled, dtype=np.int32),
            mesh,
        )
    return EsurgeGenerationArrays(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=sequences,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        output_records=output_records,
        reasoning_records=reasoning_records,
        tool_call_records=tool_call_records,
        raw_output_records=raw_output_records,
        finish_reason_records=finish_reason_records,
        completion_prompts=completion_prompts,
        sampling_per_token_logps=sampling_per_token_logps,
        completion_weight_versions=completion_weight_versions,
    )


__all__ = (
    "DEFAULT_MAX_NEW_TOKENS",
    "EsurgeGenerationArrays",
    "apply_sampling_overrides",
    "clamp_new_tokens_to_bound",
    "engine_kwargs",
    "expand_weight_version_segments",
    "generation_length_bound",
    "outputs_to_arrays",
    "prompt_token_id_rows",
    "resolve_sampling",
)
