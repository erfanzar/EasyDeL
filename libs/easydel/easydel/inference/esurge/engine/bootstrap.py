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

"""Engine-asset construction for eSurge.

``build_engine_assets`` performs every model-facing resolution step the
engine needs before any runtime component exists: processor/tokenizer
resolution, model loading (with backend attention-mechanism preferences and
the MLA compatibility graphdef swap), tool/reasoning parser auto-detection
and construction probing, drafter construction, and EOS-token-id
combination. It is a pure function of its inputs — no engine instance, no
threads, no device state — which makes the whole bootstrap unit-testable in
isolation.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any

import jax
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.inference.speculative import DrafterProtocol
from easydel.utils import set_inference_mode

from ..logger import logger

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.modules.auto.auto_modeling import PreTrainedLoading

    from ..config import eSurgeDrafterConfig, eSurgeParsingConfig, eSurgeRuntimeConfig

MLA_RAGGED_ATTN_MECHANISM = "multi_latent_ragged_page_attention_v2"
_MLA_RAGGED_ATTN_MECHANISMS = {
    "multi_latent_ragged_page_attention_v1",
    "multi_latent_ragged_page_attention_v2",
}


def normalize_attn_mechanism_value(attn_mechanism: Any) -> str | None:
    """Normalize an attention mechanism identifier to a plain string.

    Handles enum-like objects (with a ``.value`` attribute) as well as
    raw strings.

    Args:
        attn_mechanism: An attention mechanism identifier. May be ``None``,
            a string, or an enum with a ``value`` attribute.

    Returns:
        The string representation of the mechanism, or ``None`` if the
        input is ``None``.
    """
    if attn_mechanism is None:
        return None
    if hasattr(attn_mechanism, "value"):
        attn_mechanism = attn_mechanism.value
    return str(attn_mechanism)


def _text_config_uses_mla(text_config: Any) -> bool:
    """Detect whether a text config indicates Multi-Latent Attention (MLA).

    Uses several heuristics in order of priority:
    1. Explicit ``attn_mechanism`` or ``mla_attn_mechanism`` matching the
       MLA ragged attention constant.
    2. An ``attention_type`` attribute equal to ``"mla"``.
    3. A callable or boolean ``is_mla`` attribute.
    4. Presence of ``kv_lora_rank`` (or ``kv_lora_dim``) together with
       ``qk_rope_head_dim`` or ``qk_nope_head_dim``.

    Args:
        text_config: A model text configuration object (e.g.,
            ``PretrainedConfig``). May be ``None``.

    Returns:
        ``True`` if the config signals MLA usage, ``False`` otherwise.
    """
    if text_config is None:
        return False

    attn_mechanism = normalize_attn_mechanism_value(getattr(text_config, "attn_mechanism", None))
    if attn_mechanism in _MLA_RAGGED_ATTN_MECHANISMS:
        return True
    mla_attn_mechanism = normalize_attn_mechanism_value(getattr(text_config, "mla_attn_mechanism", None))
    if mla_attn_mechanism in _MLA_RAGGED_ATTN_MECHANISMS:
        return True

    attention_type = getattr(text_config, "attention_type", None)
    if attention_type is not None and str(attention_type).lower() == "mla":
        return True

    is_mla_attr = getattr(text_config, "is_mla", None)
    try:
        if callable(is_mla_attr):
            if bool(is_mla_attr()):
                return True
        elif is_mla_attr is not None and bool(is_mla_attr):
            return True
    except Exception:
        pass

    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    if kv_lora_rank is None:
        kv_lora_rank = getattr(text_config, "kv_lora_dim", None)

    qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", None)
    qk_nope_head_dim = getattr(text_config, "qk_nope_head_dim", None)

    if kv_lora_rank is None or (qk_rope_head_dim is None and qk_nope_head_dim is None):
        return False

    try:
        return int(kv_lora_rank) > 0
    except Exception:
        return True


def detect_mla_attention_mix(model: Any, text_config: Any = None) -> tuple[bool, bool]:
    """Detect whether a model contains MLA and/or non-MLA attention blocks.

    Traverses all ``UnifiedAttention`` sub-modules in *model* and inspects
    each one's ``attention_type``. Falls back to config-level heuristics
    via ``_text_config_uses_mla`` when no attention modules are found.

    Args:
        model: An EasyDeL model instance to inspect.
        text_config: Optional text configuration used as a fallback when
            module traversal yields no results.

    Returns:
        A ``(has_mla, has_non_mla)`` tuple of booleans indicating
        whether MLA and/or standard attention blocks were detected.
    """
    has_mla_attention = False
    has_non_mla_attention = False

    try:
        from easydel.layers.attention import UnifiedAttention
        from easydel.utils.traversals import iter_module_search

        for _, module in iter_module_search(model, UnifiedAttention):
            attention_type = str(getattr(module, "attention_type", "standard")).lower()
            if attention_type == "mla":
                has_mla_attention = True
            else:
                has_non_mla_attention = True
            if has_mla_attention and has_non_mla_attention:
                break
    except Exception:
        pass

    if not has_mla_attention and not has_non_mla_attention and _text_config_uses_mla(text_config):
        has_mla_attention = True

    return has_mla_attention, has_non_mla_attention


def auto_detect_tool_parser(*, tokenizer: PreTrainedTokenizerBase | None, model_type: str | None):
    """Infer the tool parser from tokenizer/template hints and model type.

    Args:
        tokenizer: Tokenizer whose chat template and special tokens may
            indicate a tool-call format.
        model_type: Model architecture identifier used as a fallback hint
            (e.g. ``"llama"``, ``"qwen2"``).

    Returns:
        The detected tool-parser name or ``None`` if nothing matched.
    """
    from easydel.inference.tools.auto_detect import detect_tool_parser

    detected = detect_tool_parser(model_type=model_type, tokenizer=tokenizer)
    return detected or None


def auto_detect_reasoning_parser_name(*, tokenizer: PreTrainedTokenizerBase | None, model_type: str | None):
    """Infer the reasoning parser from tokenizer/template hints and model type.

    Args:
        tokenizer: Tokenizer whose chat template and special tokens may
            signal a reasoning format (``<think>`` tags, etc.).
        model_type: Model architecture identifier used as a fallback hint.

    Returns:
        The detected reasoning-parser name or ``None`` if nothing matched.
    """
    from easydel.inference.reasoning.auto_detect import detect_reasoning_parser

    detected = detect_reasoning_parser(model_type=model_type, tokenizer=tokenizer)
    return detected or None


def calculate_model_size(graphstate) -> str:
    """Calculate the model size in billions of parameters.

    Args:
        graphstate: The model's graph state containing parameter arrays.

    Returns:
        String representation of model size in billions (e.g., "7.00").
        Returns "unknown" if calculation fails.
    """
    try:
        num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
        return f"{num_params / 1e9:.2f}"
    except Exception:
        return "unknown"


def get_model_name(model) -> str:
    """Generate a human-readable model name.

    Args:
        model: The EasyDeL model instance.

    Returns:
        String in format "{model_type}-{size}b" (e.g., "llama-7.00b").
    """
    model_type = getattr(model.config, "model_type", "unknown").lower()
    model_size = calculate_model_size(model.graphstate)
    return f"{model_type}-{model_size}b"


def _coerce_token_ids(raw: Any) -> list[int]:
    """Coerce a scalar/iterable of EOS-token candidates into ``list[int]``."""
    if raw is None:
        return []
    candidates = raw if isinstance(raw, (list, tuple, set)) else [raw]
    token_ids: list[int] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            token_id = int(candidate)
        except (TypeError, ValueError):
            logger.debug("Ignoring non-integer EOS token candidate: %r", candidate)
            continue
        token_ids.append(token_id)
    return token_ids


@dataclass
class EngineAssets:
    """Everything the engine needs that derives from the model + tokenizer.

    Produced once by :func:`build_engine_assets` and consumed by the eSurge
    composition root; no field is re-derived elsewhere.

    Attributes:
        model: The loaded (and MLA-compat-adjusted) EasyDeL module.
        processor: Unified text/multimodal processor object.
        tokenizer: The resolved tokenizer.
        tokenizer_source: Id/path used to spawn tokenizer/detokenizer workers.
        drafter: Constructed speculative-decoding drafter, or ``None``.
        tool_parser_name: Resolved tool-parser name (possibly auto-detected),
            or ``None`` when function calling is disabled.
        tool_parser_class: Probed tool-parser class, or ``None``.
        reasoning_parser_name: Resolved reasoning-parser name, or ``None``.
        reasoning_parser_class: Reasoning parser class, or ``None``.
        generation_config_dict: The model's generation config as a plain dict.
        generation_config_eos_ids: EOS ids contributed by the generation config.
        eos_token_ids: Combined, de-duplicated EOS ids (tokenizer +
            generation config + engine extras), in priority order.
        model_name: Auto-generated ``{type}-{size}b`` display name.
    """

    model: EasyDeLBaseModule
    processor: Any
    tokenizer: PreTrainedTokenizerBase
    tokenizer_source: str
    drafter: DrafterProtocol | None
    tool_parser_name: Any | None
    tool_parser_class: type | None
    reasoning_parser_name: Any | None
    reasoning_parser_class: type | None
    generation_config_dict: dict[str, Any] = field(default_factory=dict)
    generation_config_eos_ids: list[int] = field(default_factory=list)
    eos_token_ids: list[int] = field(default_factory=list)
    model_name: str = "unknown"


def _resolve_processor_and_tokenizer(
    model: str | EasyDeLBaseModule,
    processor: Any | None,
) -> tuple[Any, PreTrainedTokenizerBase, str]:
    """Resolve the processor, tokenizer, and worker tokenizer source.

    Args:
        model: Model id/path or preloaded module.
        processor: Processor/tokenizer object, id/path, or ``None`` to
            auto-load from a string ``model``.

    Returns:
        ``(processor, tokenizer, tokenizer_source)``.

    Raises:
        ValueError: If neither a processor nor a tokenizer source can be
            resolved for a preloaded model.
    """
    processor_obj: Any | None = processor
    processor_source: str | None = None

    if processor_obj is None:
        if isinstance(model, str):
            processor_source = model
            processor_obj = AutoTokenizer.from_pretrained(model)
        else:
            raise ValueError("Processor must be provided when using a preloaded model.")
    elif isinstance(processor_obj, str):
        processor_source = processor_obj
        processor_obj = AutoTokenizer.from_pretrained(processor_obj)
    else:
        processor_source = getattr(processor_obj, "name_or_path", None)

    tokenizer_obj: PreTrainedTokenizerBase | None = None
    if isinstance(processor_obj, PreTrainedTokenizerBase):
        tokenizer_obj = processor_obj
    else:
        maybe_tok = getattr(processor_obj, "tokenizer", None)
        if isinstance(maybe_tok, PreTrainedTokenizerBase):
            tokenizer_obj = maybe_tok

    if tokenizer_obj is None:
        source = processor_source or (model if isinstance(model, str) else None)
        if source is None:
            raise ValueError("Tokenizer must be provided (or inferable from processor) when using a preloaded model.")
        tokenizer_obj = AutoTokenizer.from_pretrained(source)

    tokenizer_source = (
        getattr(tokenizer_obj, "name_or_path", None) or processor_source or (model if isinstance(model, str) else None)
    )
    if tokenizer_source is None:
        raise ValueError("Could not infer a tokenizer source for tokenizer/detokenizer workers.")

    return processor_obj, tokenizer_obj, str(tokenizer_source)


def _load_model(
    model: str,
    *,
    loading_kwargs: PreTrainedLoading,
    runtime_config: eSurgeRuntimeConfig,
    data_parallelism_axis: str,
    dtype,
) -> EasyDeLBaseModule:
    """Load a model from an id/path with eSurge attention preferences.

    Applies the backend-preferred attention mechanism (unless the caller
    pinned one), warns on known-bad backend/mechanism pairs, and performs
    the MLA compatibility graphdef swap for MLA architectures whose
    configured mechanism cannot serve the paged inference path.

    Args:
        model: Model id/path.
        loading_kwargs: Coerced ``PreTrainedLoading`` config.
        runtime_config: Runtime section (max_model_len feeds rope/mask caps).
        data_parallelism_axis: KV-page DP axis name (mesh membership check).
        dtype: Compute/storage dtype for the loaded weights.

    Returns:
        The loaded (and possibly graphdef-swapped) module.
    """
    from easydel.infra import EasyDeLBaseConfigDict
    from easydel.layers.attention import AttentionMechanisms
    from easydel.modules.auto import AutoEasyDeLModelForCausalLM

    backend = jax.default_backend()
    preferred_attn_mechanism = (
        AttentionMechanisms.UNIFIED_ATTENTION if backend == "gpu" else AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3
    )
    user_config_kwargs = dict(loading_kwargs.config_kwargs or {})
    user_provided_attn = "attn_mechanism" in user_config_kwargs
    requested_attn = user_config_kwargs.get("attn_mechanism") if user_provided_attn else None
    if requested_attn is None:
        user_provided_attn = False

    if user_provided_attn:
        attn_value = (
            requested_attn.value
            if isinstance(requested_attn, AttentionMechanisms)
            else str(requested_attn)
            if requested_attn is not None
            else None
        )
        if backend == "gpu" and attn_value in (
            AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2.value,
            AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3.value,
        ):
            logger.warning(
                "GPU backend detected: `unified_attention` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_value!r}."
            )
        elif backend != "gpu" and attn_value == AttentionMechanisms.PAGED_FLASH_ATTENTION.value:
            logger.warning(
                "Paged flash attention is CUDA-only; non-GPU backends are not supported. "
                f"got attn_mechanism={attn_value!r}."
            )
        elif backend == "tpu" and attn_value == AttentionMechanisms.UNIFIED_ATTENTION.value:
            logger.warning(
                "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_value!r}."
            )
        elif backend == "tpu" and attn_value == AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2.value:
            logger.warning(
                "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                f"got attn_mechanism={attn_value!r}."
            )

    sharding_axis_names_resolved = tuple(loading_kwargs.sharding_axis_names)
    if data_parallelism_axis not in sharding_axis_names_resolved:
        logger.warning(
            "`data_parallelism_axis=%r` not found in `sharding_axis_names=%r`; "
            "KV page sharding will behave as unsharded for that axis.",
            data_parallelism_axis,
            sharding_axis_names_resolved,
        )

    user_config_kwargs.pop("attn_mechanism", None)

    loading = loading_kwargs.to_dict()
    loading["dtype"] = dtype
    loading["param_dtype"] = dtype
    loading["precision"] = jax.lax.Precision.DEFAULT
    loading["sharding_axis_dims"] = tuple(loading_kwargs.sharding_axis_dims)
    loading["sharding_axis_names"] = sharding_axis_names_resolved
    loading["config_kwargs"] = EasyDeLBaseConfigDict(
        attn_mechanism=requested_attn if user_provided_attn else preferred_attn_mechanism,
        attn_dtype=dtype,
        kvdtype=dtype,
        freq_max_position_embeddings=runtime_config.max_model_len,
        mask_max_position_embeddings=runtime_config.max_model_len,
        **user_config_kwargs,
    )

    with set_inference_mode():
        loaded = AutoEasyDeLModelForCausalLM.from_pretrained(**loading)
    text_config = loaded.config.get_text_config()
    has_mla_attention, has_non_mla_attention = detect_mla_attention_mix(loaded, text_config)

    _num_heads = getattr(text_config, "num_attention_heads", 0) or 0
    _mla_kernel_compatible = int(_num_heads) > 0

    if has_mla_attention and _mla_kernel_compatible:
        attn_value = normalize_attn_mechanism_value(getattr(text_config, "attn_mechanism", None))
        mla_compatible = attn_value in _MLA_RAGGED_ATTN_MECHANISMS
        if jax.default_backend() == "gpu":
            mla_compatible = mla_compatible or attn_value in {
                AttentionMechanisms.UNIFIED_ATTENTION.value,
                AttentionMechanisms.PAGED_FLASH_ATTENTION.value,
            }
        if not mla_compatible:
            if has_non_mla_attention:
                logger.warning(
                    "Mixed MLA and non-MLA full-attention layers detected, "
                    "but forcing all inference layers to "
                    f"{MLA_RAGGED_ATTN_MECHANISM!r}."
                )
            logger.info(
                f"MLA architecture detected; forcing inference attention mechanism to {MLA_RAGGED_ATTN_MECHANISM!r}."
            )
            compat_graphdef = loaded.new_graphdef(
                recursive_update=True,
                attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
                decode_attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
                mla_attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
            )
            loaded = loaded.merge_module(compat_graphdef, loaded.graphstate, loaded.graphother)
    elif has_mla_attention and not _mla_kernel_compatible:
        fallback_attn = (
            AttentionMechanisms.UNIFIED_ATTENTION
            if jax.default_backend() == "gpu"
            else AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3
        )
        logger.info(f"MLA architecture detected but num_attention_heads <= 0; falling back to {fallback_attn.value!r}.")
        compat_graphdef = loaded.new_graphdef(
            recursive_update=True,
            attn_mechanism=fallback_attn,
            decode_attn_mechanism=fallback_attn,
        )
        loaded = loaded.merge_module(compat_graphdef, loaded.graphstate, loaded.graphother)

    return loaded


def _resolve_parsers(
    *,
    model: EasyDeLBaseModule,
    tokenizer: PreTrainedTokenizerBase,
    parsing_config: eSurgeParsingConfig,
) -> tuple[Any | None, type | None, Any | None, type | None]:
    """Resolve (and probe) tool/reasoning parsers for this model + tokenizer.

    Auto-detects parser names when the parsing section leaves them ``None``,
    then validates the tool parser can construct against this tokenizer —
    some parsers require special vocabulary tokens and raise in ``__init__``;
    those degrade gracefully to disabled function calling instead of
    crashing at request-add time.

    Args:
        model: Loaded module (its ``config.model_type`` is the detection hint).
        tokenizer: Resolved tokenizer.
        parsing_config: Parsing section carrying explicit parser names.

    Returns:
        ``(tool_parser_name, tool_parser_class, reasoning_parser_name,
        reasoning_parser_class)``.
    """
    detected_model_type = getattr(getattr(model, "config", None), "model_type", None)
    tool_parser = parsing_config.tool_parser
    if tool_parser is None:
        tool_parser = auto_detect_tool_parser(tokenizer=tokenizer, model_type=detected_model_type)
    reasoning_parser = parsing_config.reasoning_parser
    if reasoning_parser is None:
        reasoning_parser = auto_detect_reasoning_parser_name(tokenizer=tokenizer, model_type=detected_model_type)

    tool_parser_class: type | None = None
    reasoning_parser_class: type | None = None

    if tool_parser:
        try:
            from easydel.inference.tools import ToolParserManager
            from easydel.inference.tools.tool_calling_mixin import build_tool_parser_or_none

            parser_class = ToolParserManager.get_tool_parser(tool_parser)
            probe = build_tool_parser_or_none(parser_class, tokenizer, context=f"tool parser '{tool_parser}'")
            if probe is None:
                tool_parser = None
            else:
                tool_parser_class = parser_class
                if not parsing_config.silent_mode:
                    logger.info("Initialized tool parser: %s", tool_parser)
        except KeyError:
            logger.warning("Tool parser '%s' not found, function calling disabled", tool_parser)
            tool_parser = None

    if reasoning_parser:
        try:
            from easydel.inference.reasoning import ReasoningParserManager

            reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
            if not parsing_config.silent_mode:
                logger.info("Initialized reasoning parser: %s", reasoning_parser)
        except KeyError:
            logger.warning("Reasoning parser '%s' not found, reasoning disabled", reasoning_parser)

    return tool_parser, tool_parser_class, reasoning_parser, reasoning_parser_class


def _build_drafter(
    *,
    model: EasyDeLBaseModule,
    drafter: DrafterProtocol | bool | None,
    drafter_config: eSurgeDrafterConfig,
    loading_kwargs: PreTrainedLoading,
    dtype,
) -> DrafterProtocol | None:
    """Construct the speculative-decoding drafter, if any.

    Args:
        model: Loaded target module (owns ``model.drafter(...)``).
        drafter: Explicit drafter object, ``True`` for auto-construction,
            ``False``/``None`` for config-driven behavior.
        drafter_config: Drafter section (method, token count, assistant model).
        loading_kwargs: Loader config reused to load a string assistant model.
        dtype: dtype for a string-loaded assistant model.

    Returns:
        The drafter object, or ``None`` when speculation is disabled.

    Raises:
        ValueError: If both an explicit drafter object and an enabled
            ``drafter_config`` are provided.

    Notes:
        Dynamic speculative-token scheduling: config-built drafters carry
        ``drafter_config.num_draft_tokens_per_batch_size`` verbatim (``None``
        = static K, today's behavior). The plug-and-play ``drafter=True``
        auto path is different — unless the user provided a schedule it
        installs :func:`~easydel.inference.esurge.config.default_auto_draft_schedule`
        (``[(1, 4, K), (5, 2**30, 0)]``) so
        ``drafter=True`` is never a regression: a latency win at low
        concurrency and speculation auto-throttled OFF at saturation. Those
        thresholds are measured for the current inline-MTP head (see the
        default's docstring) and are overridable through the drafter config.
        Explicit drafter *objects* are never touched — set
        ``num_draft_tokens_per_batch_size`` on the object to opt in.
    """
    from easydel.modules.auto import AutoEasyDeLModelForCausalLM

    from ..config import default_auto_draft_schedule

    auto_drafter_requested = drafter is True
    if drafter is True or drafter is False:
        drafter = None

    if drafter_config.enabled:
        if drafter is not None:
            raise ValueError("Pass either an explicit `drafter` object or `drafter_config`, not both.")
        drafter_kwargs = dict(drafter_config.kwargs or {})
        if drafter_config.target_embed_module is not None:
            drafter_kwargs.setdefault("target_embed_module", drafter_config.target_embed_module)
        if drafter_config.layer_mapping is not None:
            drafter_kwargs.setdefault("layer_mapping", list(drafter_config.layer_mapping))
        assistant_model = drafter_config.assistant_model
        if assistant_model is not None and isinstance(assistant_model, str):
            assistant_loading = loading_kwargs.to_dict()
            assistant_loading["pretrained_model_name_or_path"] = assistant_model
            assistant_loading["dtype"] = dtype
            assistant_loading["param_dtype"] = dtype
            assistant_loading.pop("processor", None)
            assistant_loading.pop("tokenizer", None)
            assistant_loading.pop("config_kwargs", None)
            assistant_model = AutoEasyDeLModelForCausalLM.from_pretrained(**assistant_loading)
        drafter = model.drafter(
            method=drafter_config.method or "auto",
            num_draft_tokens=int(drafter_config.num_draft_tokens),
            assistant_model=assistant_model,
            **drafter_kwargs,
        )
        # Explicit static configs keep static behavior: the schedule rides the
        # drafter only when the user asked for one (None = static K).
        drafter.num_draft_tokens_per_batch_size = drafter_config.num_draft_tokens_per_batch_size

    if auto_drafter_requested and drafter is None:
        drafter = model.drafter(
            method="auto",
            num_draft_tokens=int(drafter_config.num_draft_tokens),
        )
        # Plug-and-play path: default to the measured dynamic schedule so
        # `drafter=True` never regresses throughput at high concurrency (see
        # the Notes section above). A user-provided schedule wins.
        schedule = drafter_config.num_draft_tokens_per_batch_size
        if schedule is None:
            schedule = default_auto_draft_schedule(int(drafter_config.num_draft_tokens))
        drafter.num_draft_tokens_per_batch_size = schedule
    return drafter


def _combine_eos_token_ids(
    *,
    model: EasyDeLBaseModule,
    tokenizer: PreTrainedTokenizerBase,
    extra_eos_token_ids: list[int] | None,
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Combine tokenizer, generation-config, and engine-extra EOS ids.

    Args:
        model: Loaded module whose ``generation_config`` may carry EOS ids.
        tokenizer: Resolved tokenizer.
        extra_eos_token_ids: Engine-level extra EOS ids (parsing section).

    Returns:
        ``(generation_config_dict, generation_config_eos_ids,
        combined_eos_ids)`` with the combined list de-duplicated in
        priority order (tokenizer, generation config, extras).
    """
    generation_config = getattr(model, "generation_config", None)
    if isinstance(generation_config, dict):
        generation_config_dict = dict(generation_config)
    elif generation_config is not None and hasattr(generation_config, "to_dict"):
        try:
            generation_config_dict = generation_config.to_dict()
        except Exception:
            generation_config_dict = {}
            logger.debug("Failed to serialize model generation_config; continuing without it", exc_info=True)
    else:
        generation_config_dict = {}

    tokenizer_eos_ids = _coerce_token_ids(getattr(tokenizer, "eos_token_id", None))
    generation_config_eos_ids = _coerce_token_ids(generation_config_dict.get("eos_token_id"))
    engine_extra_eos_ids = _coerce_token_ids(extra_eos_token_ids)

    combined_eos_ids: list[int] = []
    seen_eos_ids: set[int] = set()
    for token_id in [*tokenizer_eos_ids, *generation_config_eos_ids, *engine_extra_eos_ids]:
        if token_id in seen_eos_ids:
            continue
        seen_eos_ids.add(token_id)
        combined_eos_ids.append(token_id)

    return generation_config_dict, generation_config_eos_ids, combined_eos_ids


def build_engine_assets(
    *,
    model: str | EasyDeLBaseModule,
    processor: Any | None,
    loading_kwargs: PreTrainedLoading,
    runtime_config: eSurgeRuntimeConfig,
    parsing_config: eSurgeParsingConfig,
    drafter_config: eSurgeDrafterConfig,
    drafter: DrafterProtocol | bool | None,
    data_parallelism_axis: str,
    dtype,
) -> EngineAssets:
    """Resolve every model-facing asset the eSurge engine composes over.

    Args:
        model: Model id/path (loaded here) or a preloaded EasyDeL module.
        processor: Processor/tokenizer object, id/path, or ``None``.
        loading_kwargs: Coerced ``PreTrainedLoading`` section.
        runtime_config: Runtime section.
        parsing_config: Parsing section (parser names, extra EOS ids).
        drafter_config: Drafter section.
        drafter: Explicit drafter object / ``True`` for auto / ``None``.
        data_parallelism_axis: Normalized KV-page DP axis name.
        dtype: Compute/storage dtype for model loading.

    Returns:
        A fully-populated :class:`EngineAssets`.
    """
    processor_obj, tokenizer_obj, tokenizer_source = _resolve_processor_and_tokenizer(model, processor)

    if isinstance(model, str):
        model = _load_model(
            model,
            loading_kwargs=loading_kwargs,
            runtime_config=runtime_config,
            data_parallelism_axis=data_parallelism_axis,
            dtype=dtype,
        )

    tool_parser_name, tool_parser_class, reasoning_parser_name, reasoning_parser_class = _resolve_parsers(
        model=model,
        tokenizer=tokenizer_obj,
        parsing_config=parsing_config,
    )

    drafter_obj = _build_drafter(
        model=model,
        drafter=drafter,
        drafter_config=drafter_config,
        loading_kwargs=loading_kwargs,
        dtype=dtype,
    )

    generation_config_dict, generation_config_eos_ids, combined_eos_ids = _combine_eos_token_ids(
        model=model,
        tokenizer=tokenizer_obj,
        extra_eos_token_ids=parsing_config.extra_eos_token_ids,
    )

    return EngineAssets(
        model=model,
        processor=processor_obj,
        tokenizer=tokenizer_obj,
        tokenizer_source=tokenizer_source,
        drafter=drafter_obj,
        tool_parser_name=tool_parser_name,
        tool_parser_class=tool_parser_class,
        reasoning_parser_name=reasoning_parser_name,
        reasoning_parser_class=reasoning_parser_class,
        generation_config_dict=generation_config_dict,
        generation_config_eos_ids=generation_config_eos_ids,
        eos_token_ids=combined_eos_ids,
        model_name=get_model_name(model),
    )
