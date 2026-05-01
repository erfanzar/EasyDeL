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
"""Utility helpers for compiling and running EasyDeL training/eval steps.

This module hosts the lower-level building blocks used by trainers:

* Quantization helpers and straight-through-estimator emulators for
  low-precision training (``mxfp8``, ``nvfp8``, ``nf4``, ...).
* :func:`compile_trainer_step`, the thin wrapper around ``jax.jit`` /
  ``jax.pjit`` that produces sharded and (optionally) scan-friendly step
  functions.
* Pipeline-parallel scheduling utilities (``scheduled_training_step`` and
  friends) that drive MPMD schedulers.
* Generation kwarg normalization helpers used by every trainer that calls
  into the model's ``generate`` / eSurge entry points.
* Misc utilities such as ``filter_kwargs_for_callable`` for safely
  dispatching to user-supplied reward callables.
"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing as tp
import warnings

import jax
import spectrax as spx
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.infra.sharding import MeshLike
from easydel.utils.helpers import check_bool_flag

if tp.TYPE_CHECKING:
    from easydel.infra.etils import MpMdSchedulers

SCAN_TRAINER = check_bool_flag("SCAN_TRAINER")
FAST_COMPILE = check_bool_flag("FAST_COMPILE")
_UNSPECIFIED = object()

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

_ScheduledLossFn = tp.Callable[[tp.Any, collections.abc.Mapping[str, jax.Array]], jax.Array]
_ScheduledValueAndGradFn = tp.Callable[[tp.Any, dict], tuple[jax.Array, tp.Any]]
_SCHEDULED_LOSS_ADAPTERS: dict[tuple[str, str], ScheduledLossAdapter] = {}
_SCHEDULED_AUXILIARY_CACHE: dict[tuple[int, int], tp.Callable[..., tp.Any]] = {}


@dataclasses.dataclass(frozen=True)
class ScheduledStepCall:
    """Frozen snapshot of one trainer step invocation, passed to scheduled-loss adapters.

    When a training step decorated with :func:`compile_trainer_step` is invoked
    under an MPMD pipeline schedule, the wrapper packages the live arguments
    of *that* call into a ``ScheduledStepCall`` and forwards it to the adapter
    registered for the underlying step function (see
    :func:`register_scheduled_loss_adapter`). The adapter uses the snapshot
    to (a) build a *trainer-specific* scalar loss closure consumed by
    ``spx.jit(..., schedule=...)`` / :func:`spx.sxvalue_and_grad`, (b) compute
    a cache key so that repeated calls with the same shape/dtype signature
    reuse the compiled scheduled loss, and (c) optionally rewrite the batch
    that flows into the scheduled loss.

    Instances are frozen and hashable-by-identity; do not mutate the captured
    mappings — the contained pytrees are still live state from the caller.

    Attributes:
        step_fn (Callable[..., Any]): The original (undecorated) trainer step
            function whose adapter is being looked up. Used purely for
            adapter-registry lookups and naming; never invoked from inside
            the adapter.
        state (EasyDeLState): The current trainer state pytree (model graphdef
            + optimizer state + step counter). This is the differentiation
            target the resulting value-and-grad runs against.
        batch (Mapping[str, jax.Array]): The mini-batch dict as passed to the
            step function (may be modified by ``ScheduledLossAdapter.prepare_batch``
            before reaching the compiled loss).
        args (tuple[Any, ...]): The positional arguments the wrapper received,
            preserved verbatim so adapters can access trailer args (e.g.
            optional reference logps).
        kwargs (Mapping[str, Any]): The keyword arguments the wrapper received,
            preserved verbatim alongside ``args``.
        bound_arguments (Mapping[str, Any]): A flat ``name -> value`` mapping
            produced by binding ``args`` / ``kwargs`` against the wrapped
            step function's signature. Use :meth:`get` for safe lookup of
            optional parameters.
        schedule (Any): The active MPMD schedule object (typically an
            ``MpMdSchedulers`` instance) under which the scheduled loss will
            be compiled. Adapters use this to specialize compilation (e.g.
            change pipeline microbatch handling).
    """

    step_fn: tp.Callable[..., tp.Any]
    state: EasyDeLState
    batch: collections.abc.Mapping[str, jax.Array]
    args: tuple[tp.Any, ...]
    kwargs: collections.abc.Mapping[str, tp.Any]
    bound_arguments: collections.abc.Mapping[str, tp.Any]
    schedule: tp.Any

    def get(self, name: str, default: tp.Any = None) -> tp.Any:
        """Look up a bound argument by name.

        Args:
            name: The argument name as it appears in the original step
                function signature.
            default: Returned when ``name`` is not present in
                ``bound_arguments``.

        Returns:
            The bound argument value or ``default``.
        """
        return self.bound_arguments.get(name, default)


@dataclasses.dataclass(frozen=True)
class ScheduledLossAdapter:
    """Trainer-specific glue between a step function and ``spx.jit(schedule=...)``.

    A ``ScheduledLossAdapter`` is registered once per *step function flavor*
    (SFT, DPO, KTO, GRPO, …) via :func:`register_scheduled_loss_adapter`, and
    is consulted by :func:`_compile_scheduled_training_step` whenever that
    step function is compiled under a non-trivial MPMD schedule.

    The adapter must satisfy three responsibilities, modelled as the three
    callable fields below:

    1. **Build a scalar loss** the SpectraX scheduler can differentiate. The
       loss is what will be compiled with ``spx.jit(loss, schedule=...)`` and
       passed to :func:`spx.sxvalue_and_grad`, so it must take exactly
       ``(state_tree, batch_dict) -> scalar`` regardless of the underlying
       trainer's richer step signature.
    2. **Produce a cache key**. The compiled scheduled value-and-grad is
       expensive; the key returned here decides when it is safe to reuse the
       cached compilation versus retrace. Include any tensor shapes/dtypes
       and trainer flags that change the loss closure.
    3. **Optionally rewrite the batch**. If the trainer needs to inject extra
       tensors (reference logps, scheduling masks, …) before the compiled
       loss sees the batch, ``prepare_batch`` returns the modified mapping;
       the original ``batch`` on :class:`ScheduledStepCall` is left untouched.

    Adapters are stored in the module-level ``_SCHEDULED_LOSS_ADAPTERS``
    registry keyed by ``(module, qualname)`` of the step function.

    Attributes:
        name (str): Short, human-readable adapter tag (``"sft"``, ``"dpo"``,
            ``"grpo"``, …). Embedded into the generated step function's
            ``__name__`` for traceability in profiles and logs.
        make_loss (Callable[[ScheduledStepCall], _ScheduledLossFn]): Factory
            that, given the live call context, returns the scalar
            ``(tree, batch) -> jax.Array`` loss closure to be JIT-compiled
            under the schedule.
        make_cache_key (Callable[[ScheduledStepCall], tuple[Any, ...]]):
            Factory that returns a hashable cache key derived from the call
            context. Two calls returning equal keys must be safe to share
            the same compiled loss.
        prepare_batch (Callable[[ScheduledStepCall], Mapping[str, jax.Array]] | None):
            Optional pre-processor invoked on every call to produce the
            mapping that flows into the compiled loss. ``None`` means the
            untouched ``ScheduledStepCall.batch`` is forwarded as-is.
    """

    name: str
    make_loss: tp.Callable[[ScheduledStepCall], _ScheduledLossFn]
    make_cache_key: tp.Callable[[ScheduledStepCall], tuple[tp.Any, ...]]
    prepare_batch: tp.Callable[[ScheduledStepCall], collections.abc.Mapping[str, jax.Array]] | None = None


@dataclasses.dataclass
class _ScheduledValueAndGradCompiler:
    """Per-step lazy compiler/cache for ``spx.jit(schedule=...)`` + ``sxvalue_and_grad``.

    One instance is created inside the closure of each scheduled training step
    produced by :func:`_compile_scheduled_training_step`. On every call the
    instance asks its adapter for a cache key built from the current
    :class:`ScheduledStepCall`; if that key matches the previously seen one,
    the cached compiled value-and-grad function is reused directly. Otherwise
    the compiler:

    1. Asks the adapter to materialise the scalar loss closure.
    2. Compiles it through ``spx.jit`` with the step's mesh, MPMD schedule,
       and ``batch_argnums`` (so the scheduler knows which positional argument
       carries the per-microbatch tensors).
    3. Wraps the compiled function with :func:`spx.sxvalue_and_grad` against
       the state tree (``argnums=0``) and stores it for future reuse.

    The class is mutable on purpose — mutation is the cache update. There is
    no thread safety here because each compiler lives behind a single
    ``scheduled_training_step`` closure that is invoked sequentially by the
    trainer loop.

    Attributes:
        mesh (MeshLike): Spectrax/JAX mesh the compiled loss runs on.
            Forwarded to ``spx.jit``.
        schedule (Any): MPMD schedule object passed to ``spx.jit(..., schedule=...)``.
        batch_argnums (int | Sequence[int] | None): Positional indices of the
            scheduled-loss arguments that carry per-microbatch data. ``None``
            disables microbatch slicing.
        adapter (ScheduledLossAdapter): Trainer-specific adapter used to build
            both the loss closure and the cache key.
        cached_key (tuple[Any, ...] | None): Last cache key returned by
            ``adapter.make_cache_key``. ``None`` until the first compilation.
        cached_value_and_grad (_ScheduledValueAndGradFn | None): Last
            compiled ``(tree, batch) -> (loss, gradients)`` function. ``None``
            until the first compilation; reused while ``cached_key`` matches.
    """

    mesh: MeshLike
    schedule: tp.Any
    batch_argnums: int | tp.Sequence[int] | None
    adapter: ScheduledLossAdapter
    cached_key: tuple[tp.Any, ...] | None = None
    cached_value_and_grad: _ScheduledValueAndGradFn | None = None

    def get(self, call: ScheduledStepCall) -> _ScheduledValueAndGradFn:
        """Return a cached scheduled value-and-grad callable for ``call``.

        On a cache miss, builds a fresh ``spx.jit``-compiled loss with the
        configured schedule and wraps it in :func:`spx.sxvalue_and_grad`.
        Subsequent calls with the same adapter cache key reuse the existing
        compiled callable.

        Args:
            call: The current scheduled step call context.

        Returns:
            A function ``(tree, batch) -> (loss, gradients)`` ready to be
            applied by the trainer.
        """
        key = self.adapter.make_cache_key(call)
        if self.cached_value_and_grad is not None and self.cached_key == key:
            return self.cached_value_and_grad

        loss_fn = self.adapter.make_loss(call)
        scheduled_loss = spx.jit(
            loss_fn,
            mesh=self.mesh,
            schedule=self.schedule,
            static_argnums=(),
            batch_argnums=self.batch_argnums,
        )
        scheduled_value_and_grad = spx.sxvalue_and_grad(scheduled_loss, argnums=0)

        def value_and_grad(tree, batch):
            """Run the scheduled value-and-grad and unwrap the gradient tuple.

            Args:
                tree: The state pytree to differentiate against.
                batch: The minibatch dictionary forwarded to the loss.

            Returns:
                A ``(loss, gradients)`` tuple where ``gradients`` matches
                the structure of ``tree``.
            """
            loss, (gradients,) = scheduled_value_and_grad(tree, batch)
            return loss, gradients

        self.cached_key = key
        self.cached_value_and_grad = value_and_grad
        return value_and_grad


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


def register_scheduled_loss_adapter(
    step_fn: tp.Callable[..., tp.Any],
    adapter: ScheduledLossAdapter,
) -> tp.Callable[..., tp.Any]:
    """Bind a :class:`ScheduledLossAdapter` to a trainer step function.

    The registry maintained here lets :func:`_compile_scheduled_training_step`
    discover, given a raw step function, how to build the scalar loss that
    SpectraX's scheduled VJP needs. Concretely the adapter is the
    *trainer-specific* piece (DPO/KTO/PPO/SFT/...) that knows how to:

    * project the live ``state``/``batch`` into a
      ``(tree, batch) -> jax.Array`` scalar loss closure,
    * build a hashable cache key so equivalent calls reuse the compiled
      ``spx.jit(..., schedule=...)`` artifact, and
    * optionally pre-process the batch before it reaches the compiled loss.

    Everything else — gradient accumulation, scheduled VJP, stage-local
    gradient placement, and the optimizer update — stays in the shared
    pipeline-parallel path and is *not* the adapter's concern.

    Registration writes to two locations so lookups via
    :func:`get_scheduled_loss_adapter` succeed regardless of how the caller
    holds the function:

    1. The module-level ``_SCHEDULED_LOSS_ADAPTERS`` dict, keyed by
       ``(module, qualname)`` of ``step_fn``. This survives even when
       ``step_fn`` is wrapped or copied later.
    2. The function attribute ``step_fn.__easydel_scheduled_loss_adapter__``,
       a fast-path direct pointer used when the caller still has the
       original object.

    The function is intentionally usable as a decorator factory (returns
    ``step_fn`` unchanged) so trainers can write
    ``register_scheduled_loss_adapter(step_fn, adapter)`` at module import
    time.

    Args:
        step_fn (Callable[..., Any]): The trainer's raw, unwrapped step
            function (e.g. the SFT ``training_step``, DPO ``training_step``,
            …) that will later be compiled under an MPMD schedule. Used only
            for registry-key derivation and attribute attachment; never
            invoked by this function.
        adapter (ScheduledLossAdapter): The trainer-specific adapter that
            knows how to materialise a scalar loss / cache key / batch
            override from a live :class:`ScheduledStepCall`. Stored by
            reference, so the caller must not mutate it after registration.

    Returns:
        Callable[..., Any]: The same ``step_fn`` object passed in,
        unmodified except for the freshly-attached
        ``__easydel_scheduled_loss_adapter__`` attribute. Returning
        ``step_fn`` lets this function be chained or used as a decorator.
    """

    _SCHEDULED_LOSS_ADAPTERS[_scheduled_step_key(step_fn)] = adapter
    step_fn.__easydel_scheduled_loss_adapter__ = adapter
    return step_fn


def get_scheduled_loss_adapter(fn: tp.Callable[..., tp.Any]) -> ScheduledLossAdapter | None:
    """Return the :class:`ScheduledLossAdapter` registered for ``fn``, if any.

    Adapters can be attached either directly via the
    ``__easydel_scheduled_loss_adapter__`` attribute or through
    :func:`register_scheduled_loss_adapter`.  Both lookup paths are
    consulted.

    Args:
        fn: A trainer step callable.

    Returns:
        The associated adapter, or ``None`` when none has been registered.
    """
    adapter = getattr(fn, "__easydel_scheduled_loss_adapter__", None)
    if adapter is not None:
        return adapter
    return _SCHEDULED_LOSS_ADAPTERS.get(_scheduled_step_key(fn))


def scheduled_cache_token(value: tp.Any) -> tp.Hashable:
    """Return a stable token for scheduled-loss cache keys."""

    try:
        hash(value)
    except TypeError:
        return id(value)
    return value


def scheduled_loss_cache_key(
    call: ScheduledStepCall,
    *,
    value_fields: collections.abc.Iterable[str] = (),
    object_fields: collections.abc.Iterable[str] = (),
    include_graph: bool = True,
) -> tuple[tp.Any, ...]:
    """Build a cache key for scheduled scalar-loss compilation.

    ``value_fields`` are hashed by value when possible, while
    ``object_fields`` are keyed by identity.  This keeps trainer adapters from
    open-coding the same graph/config/function identity tuple.
    """

    pieces: list[tp.Any] = []
    if include_graph:
        pieces.extend((id(call.state.graphdef), id(call.state.graphother)))
    pieces.extend(scheduled_cache_token(call.get(name)) for name in value_fields)
    pieces.extend(id(call.get(name)) for name in object_fields)
    return tuple(pieces)


def _sync_schedule_config(config: tp.Any, schedule: tp.Any, seen: set[int]) -> None:
    """Keep a config and its nested sub-configs aligned with the runtime schedule."""
    if config is None:
        return
    config_id = id(config)
    if config_id in seen:
        return
    seen.add(config_id)

    virtual_stages = getattr(schedule, "virtual_stages_per_rank", None)
    if callable(virtual_stages) and hasattr(config, "pipeline_virtual_stages"):
        config.pipeline_virtual_stages = int(virtual_stages())
    stage_layout = getattr(schedule, "stage_layout", None)
    if stage_layout is not None and hasattr(config, "pipeline_stage_layout"):
        config.pipeline_stage_layout = stage_layout

    for attr_name in ("text_config", "vision_config", "encoder_config", "decoder_config"):
        _sync_schedule_config(getattr(config, attr_name, None), schedule, seen)


def sync_module_schedule_config(module: tp.Any, schedule: tp.Any) -> None:
    """Keep model-side PP marker generation in sync with the runtime schedule."""
    _sync_schedule_config(getattr(module, "config", None), schedule, set())

    for attr_name in ("model", "base_model", "language_model", "visual"):
        child = getattr(module, attr_name, None)
        _sync_schedule_config(getattr(child, "config", None), schedule, set())


def bind_scheduled_module(
    call: ScheduledStepCall,
    tree: tp.Any,
    *,
    straight_through_field: str = "straight_through_emulator",
) -> tp.Any:
    """Merge a scheduled trainable tree and sync its PP marker config."""

    straight_through = call.get(straight_through_field)
    if straight_through is not None:
        tree = straight_through(tree)
    module = call.state.merge(tree)
    sync_module_schedule_config(module, call.schedule)
    return module


def constrain_scheduled_batch(
    module: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    partition_spec: tp.Any,
) -> dict[str, tp.Any]:
    """Apply the standard EasyDeL scheduled batch sharding constraint."""

    return spx.with_sharding_constraint(
        dict(batch),
        partition_spec,
        mesh=module.mesh,
        ignore_mpmd=True,
    )


def cached_scheduled_auxiliary(fn: tp.Callable[..., tp.Any], mesh: MeshLike) -> tp.Callable[..., tp.Any]:
    """Return a cached regular ``spx.jit`` for non-gradient auxiliary forwards."""

    key = (id(fn), id(mesh))
    cached = _SCHEDULED_AUXILIARY_CACHE.get(key)
    if cached is None:
        cached = spx.jit(fn, mesh=mesh)
        _SCHEDULED_AUXILIARY_CACHE[key] = cached
    return cached


def stop_gradient_tree(value: tp.Any) -> tp.Any:
    """Stop gradients for array leaves while preserving non-array metadata."""

    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if hasattr(leaf, "shape") else leaf,
        value,
    )


def prepare_scheduled_reference_outputs(
    call: ScheduledStepCall,
    *,
    reference_state_field: str,
    forward_field: str,
    output_to_batch: collections.abc.Mapping[str, str],
    partition_spec_field: str = "partition_spec",
    skip_field: str | None = None,
    missing_error: str | None = None,
) -> dict[str, tp.Any]:
    """Precompute reference-model outputs before the scheduled VJP.

    Preference/RL trainers often need frozen reference log-probs.  Computing
    those inside the policy scheduled loss would trace two model forwards into a
    single PP graph, so this helper runs the reference forward once as a regular
    auxiliary JIT and appends its requested outputs to the batch.
    """

    batch = dict(call.batch)
    if skip_field is not None and bool(call.get(skip_field, False)):
        return batch
    if all(batch_key in batch for batch_key in output_to_batch.values()):
        return batch

    reference_state = call.get(reference_state_field)
    forward_fn = call.get(forward_field)
    if reference_state is None or forward_fn is None:
        raise RuntimeError(
            missing_error or f"scheduled MPMD training requires {reference_state_field!r} and {forward_field!r}."
        )

    ref_model = reference_state.model
    ref_model.eval()
    sync_module_schedule_config(ref_model, call.schedule)
    constrained_batch = spx.with_sharding_constraint(
        batch,
        call.get(partition_spec_field),
        mesh=ref_model.mesh,
        ignore_mpmd=True,
    )
    ref_forward = cached_scheduled_auxiliary(forward_fn, ref_model.mesh)
    ref_out = stop_gradient_tree(ref_forward(ref_model, constrained_batch))
    for output_key, batch_key in output_to_batch.items():
        batch[batch_key] = ref_out[output_key]
    return batch


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


def compact_generation_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None) -> dict[str, tp.Any]:
    """Drop ``None`` leaves from normalized generation model kwargs.

    Args:
        kwargs: Normalized generation model kwargs mapping. May be ``None``.

    Returns:
        dict: Compact dictionary with only non-``None`` entries.
    """

    if not kwargs:
        return {}
    return {key: value for key, value in kwargs.items() if value is not None}


def _flatten_grouped_multimodal_model_value(key: str, value: tp.Any) -> tp.Any:
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
    """Straight-through estimator: ``q`` on the forward pass, identity on the backward pass.

    Args:
        x: The original (full-precision) tensor.
        q: A quantized approximation of ``x``.

    Returns:
        ``x + stop_gradient(q - x)``, equal to ``q`` numerically while
        passing gradients through unchanged.
    """
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
        """Simulate the quantize/dequantize round-trip for a single tensor leaf.

        Handles 0-d and 1-d edge cases, pads the last dim to a multiple of
        ``group_size``, and dispatches to the appropriate ejkernel quantizer
        (with or without zero-points / biases).

        Args:
            y: Float tensor to round-trip through the quantization scheme.

        Returns:
            A tensor of the same shape and dtype as ``y`` whose values are
            the quantize-then-dequantize image of ``y``.
        """
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
        """Apply STE quantization to a single tensor leaf.

        Non-floating tensors are returned unchanged so that integer
        bookkeeping leaves (e.g. step counters) are not perturbed.

        Args:
            x: Tensor leaf to quantize on the forward pass only.

        Returns:
            The straight-through quantized tensor.
        """
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
        """Apply ``tensor_straight_through`` over every leaf of ``graphstate``.

        Args:
            graphstate: Pytree of tensor leaves (typically the model's
                graph-state).

        Returns:
            A pytree of identical shape with each float leaf passed through
            the per-tensor STE.
        """
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


def _normalize_static_argnums(static_argnums: int | tp.Sequence[int] | None) -> tuple[int, ...]:
    """Coerce a static_argnums spec into a tuple of ints.

    Args:
        static_argnums: Either a single int, an iterable of ints, or
            ``None``.

    Returns:
        A (possibly empty) tuple of ints.
    """
    if static_argnums is None:
        return ()
    if isinstance(static_argnums, int):
        return (static_argnums,)
    return tuple(static_argnums)


def _normalize_static_argnames(static_argnames: str | tp.Iterable[str] | None) -> tuple[str, ...]:
    """Coerce a static_argnames spec into a tuple of names.

    Args:
        static_argnames: Either a single name, an iterable of names, or
            ``None``.

    Returns:
        A (possibly empty) tuple of names.
    """
    if static_argnames is None:
        return ()
    if isinstance(static_argnames, str):
        return (static_argnames,)
    return tuple(static_argnames)


def compile_trainer_step(
    fn: tp.Callable[..., tp.Any],
    *,
    mutable: tp.Any = (),
    mesh: MeshLike | None = None,
    schedule: MpMdSchedulers | None = None,
    arguments: tp.Any | None = None,
    in_shardings: tp.Any = _UNSPECIFIED,
    out_shardings: tp.Any = _UNSPECIFIED,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    batch_argnums: int | tp.Sequence[int] | None = None,
    keep_unused: bool = False,
    **jit_kwargs,
) -> tp.Callable[..., tp.Any]:
    """Compile a trainer step with the SpectraX MPMD path when the mesh requires it.

    When ``schedule`` is None, falls back to ``arguments.mpmd_scheduler`` if
    ``arguments`` is supplied -- so trainers can opt in to 1F1B / GPipe
    microbatching by setting ``TrainingArguments.mpmd_scheduler`` once,
    without touching every trainer's call site. Trainer steps that register a
    :class:`ScheduledLossAdapter` are run through the shared scheduled-VJP path;
    unregistered full trainer steps keep the regular marker-based JIT path.
    """

    if schedule is None and arguments is not None:
        schedule = getattr(arguments, "mpmd_scheduler", None)
    scheduled_adapter = get_scheduled_loss_adapter(fn) if schedule is not None else None
    if scheduled_adapter is not None:
        return _compile_scheduled_training_step(
            step_fn=fn,
            mesh=mesh,
            schedule=schedule,
            batch_argnums=(1,) if batch_argnums is None else batch_argnums,
            static_argnums=static_argnums,
            adapter=scheduled_adapter,
        )
    # SpectraX's schedule= path is a scalar-loss custom-VJP runtime. Whole
    # EasyDeL trainer steps usually return (state, metrics) or metrics, and
    # auxiliary forwards return logits/log-probs. Keep all unregistered
    # callables on the regular marker-based MPMD path so custom trainers keep
    # their full metrics and is_training behavior.

    static_nums = _normalize_static_argnums(static_argnums)
    kwargs = {
        "mutable": mutable,
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "donate_argnums": donate_argnums,
        "donate_argnames": donate_argnames,
        "keep_unused": keep_unused,
        **jit_kwargs,
    }
    if mesh is not None:
        kwargs["mesh"] = mesh
    if in_shardings is not _UNSPECIFIED:
        kwargs["in_shardings"] = in_shardings
    if out_shardings is not _UNSPECIFIED:
        kwargs["out_shardings"] = out_shardings
    compiled = spx.jit(fn, **kwargs)
    compiled.static_argnums_ = static_nums
    return compiled


def _slice_batch_for_scheduled_step(batch: dict, batch_size: int, start_index: int, minibatch_size: int) -> dict:
    """Slice leading-batch leaves while passing shared leaves through."""

    def _slice_leaf(arr):
        """Slice a single leaf along axis 0 when it carries the full batch dim.

        Args:
            arr: A pytree leaf (typically a JAX array).

        Returns:
            The dynamically sliced minibatch view, or ``arr`` unchanged
            when it does not carry the leading batch dimension.
        """
        if not hasattr(arr, "shape") or arr.ndim == 0:
            return arr
        if arr.shape[0] == batch_size:
            return lax.dynamic_slice_in_dim(arr, start_index, minibatch_size, axis=0)
        return arr

    return jax.tree_util.tree_map(_slice_leaf, batch)


def _scheduled_step_key(fn: tp.Callable[..., tp.Any]) -> tuple[str, str]:
    """Return a stable ``(module, qualname)`` key for an unwrapped step function.

    Unwraps :class:`functools.partial` so registrations made on the
    original function are still discoverable through partial wrappers.

    Args:
        fn: The trainer step callable.

    Returns:
        A tuple of strings suitable for use as a dictionary key.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    return getattr(fn, "__module__", ""), getattr(fn, "__name__", "")


def _scheduled_step_name(fn: tp.Callable[..., tp.Any]) -> str:
    """Return a human-readable name for a (possibly partial) step function.

    Args:
        fn: The trainer step callable.

    Returns:
        The function name when available, otherwise its module name or
        the class name of the wrapper.
    """
    module, name = _scheduled_step_key(fn)
    return name or module or type(fn).__name__


def _apply_stage_local_gradients(
    *,
    state: EasyDeLState,
    gradients: tp.Any,
    loss: jax.Array,
    loss_config: LossConfig | None,
    learning_rate_fn: tp.Any,
) -> tuple[EasyDeLState, LossMetrics]:
    """Apply stage-local gradients via the optimizer's PP-aware update path.

    Used by the scheduled training-step path so each pipeline stage updates
    only its local shard of parameters and optimizer state.  Honors a
    ``break_on_nan`` LossConfig by returning the unchanged state when the
    loss is ``NaN``.

    Args:
        state: Current model/optimizer state.
        gradients: Stage-local gradient pytree.
        loss: Scalar loss value used for metrics and NaN detection.
        loss_config: Optional :class:`LossConfig`; ``break_on_nan`` is
            consulted.
        learning_rate_fn: Schedule function for the optimizer.

    Returns:
        ``(new_state, metrics)`` after applying the optimizer update; or
        ``(state, metrics)`` unchanged on a NaN loss when ``break_on_nan``
        is set.

    Raises:
        RuntimeError: If the state's optimizer is missing or does not
            implement :meth:`apply_gradients_stage_local`.
    """
    metrics = update_metrics(
        metrics=LossMetrics(loss=loss),
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=None,
    )
    if loss_config is not None and bool(getattr(loss_config, "break_on_nan", False)):
        if bool(jax.device_get(jnp.isnan(loss))):
            return state, metrics

    if state.tx is None:
        raise RuntimeError("mpmd_scheduler requires an initialized optimizer transformation.")
    if state.opt_state is None:
        raise RuntimeError("mpmd_scheduler requires initialized optimizer state.")

    apply_stage_local = getattr(state.tx, "apply_gradients_stage_local", None)
    if not callable(apply_stage_local):
        raise RuntimeError(
            "mpmd_scheduler produced stage-local gradients, but the optimizer does not expose "
            "`apply_gradients_stage_local`. Use an eFormer optimizer with PP stage-local support."
        )

    try:
        graphstate, opt_state = apply_stage_local(
            params=state.graphstate,
            grads=gradients,
            opt_state=state.opt_state,
            learning_rate_fn=learning_rate_fn,
            delete_grads=True,
        )
    except NotImplementedError as exc:
        raise RuntimeError(f"Optimizer does not support PP stage-local updates: {exc}") from exc

    new_state = state.replace(step=state.step + 1, graphstate=graphstate, opt_state=opt_state)
    return new_state, metrics


def _run_scheduled_value_and_grad(
    *,
    value_and_grad: _ScheduledValueAndGradFn,
    graphstate: tp.Any,
    batch: dict,
    batch_size: int,
    minibatch_size: int,
) -> tuple[jax.Array, tp.Any]:
    """Run the scheduled value-and-grad with optional gradient accumulation.

    When ``batch_size > minibatch_size``, the input batch is split into
    equal-sized minibatches and the gradients are averaged.

    Args:
        value_and_grad: The compiled scheduled value-and-grad callable.
        graphstate: The state pytree to differentiate against.
        batch: The full minibatch dictionary.
        batch_size: Total batch size (leading dimension of ``batch``).
        minibatch_size: Size of each accumulation step.

    Returns:
        ``(loss, gradients)`` aggregated across all accumulation steps.
    """
    num_accum_steps = batch_size // minibatch_size
    if num_accum_steps == 1:
        return value_and_grad(graphstate, batch)

    loss_acc = None
    grad_acc = None
    for accum_idx in range(num_accum_steps):
        minibatch = _slice_batch_for_scheduled_step(
            batch,
            batch_size,
            accum_idx * minibatch_size,
            minibatch_size,
        )
        loss_i, gradients_i = value_and_grad(graphstate, minibatch)
        loss_acc = loss_i if loss_acc is None else loss_acc + loss_i
        grad_acc = gradients_i if grad_acc is None else jax.tree_util.tree_map(jnp.add, grad_acc, gradients_i)

    inv_steps = jnp.asarray(1.0 / num_accum_steps, dtype=jnp.float32)
    return loss_acc * inv_steps, jax.tree_util.tree_map(lambda x: x * inv_steps, grad_acc)


def _compile_scheduled_training_step(
    *,
    step_fn: tp.Callable[..., tp.Any],
    mesh: MeshLike | None,
    schedule: tp.Any,
    batch_argnums: int | tp.Sequence[int] | None,
    static_argnums: int | tp.Sequence[int] | None,
    adapter: ScheduledLossAdapter,
) -> tp.Callable[..., tp.Any]:
    """Build the scheduled-VJP version of a registered trainer step.

    Wraps the trainer-supplied ``step_fn`` with a SpectraX-scheduled
    value-and-grad path: gradient accumulation is run via
    :func:`_run_scheduled_value_and_grad` and the resulting stage-local
    gradients are applied through :func:`_apply_stage_local_gradients`.

    Args:
        step_fn: The original trainer step function.
        mesh: Device mesh used for the scheduled compilation.
        schedule: The MPMD pipeline schedule (e.g. 1F1B, GPipe).
        batch_argnums: Position(s) of batch-typed arguments inside the
            trainer step signature.
        static_argnums: Position(s) of static (compile-time constant)
            arguments.
        adapter: The :class:`ScheduledLossAdapter` describing how to
            extract the scalar loss from ``step_fn``.

    Returns:
        A callable with the same ``(state, batch, ...)`` signature as
        ``step_fn`` that runs through the scheduled VJP path.

    Raises:
        ValueError: If ``mesh`` is ``None``.
    """
    if mesh is None:
        raise ValueError("mpmd_scheduler requires compile_trainer_step(..., mesh=...).")

    step_signature = inspect.signature(step_fn)
    scheduled_vag = _ScheduledValueAndGradCompiler(
        mesh=mesh,
        schedule=schedule,
        batch_argnums=batch_argnums,
        adapter=adapter,
    )

    def scheduled_training_step(
        state: EasyDeLState,
        batch: collections.abc.Mapping[str, jax.Array],
        *step_args: tp.Any,
        **step_kwargs: tp.Any,
    ) -> tuple[EasyDeLState, LossMetrics]:
        """Run one scheduled-VJP training step with optional gradient accumulation.

        Args:
            state: Current model/optimizer state.
            batch: Minibatch dictionary of sharded JAX arrays.
            *step_args: Forwarded positional arguments matching
                ``step_fn``'s signature.
            **step_kwargs: Forwarded keyword arguments.

        Returns:
            ``(new_state, metrics)`` after applying the optimizer update.
        """
        batch = dict(batch)
        bound = step_signature.bind(state, batch, *step_args, **step_kwargs)
        bound.apply_defaults()
        bound_arguments = dict(bound.arguments)
        loss_config = bound_arguments.get("loss_config")
        learning_rate_fn = bound_arguments.get("learning_rate_fn")
        gradient_accumulation_steps = bound_arguments.get("gradient_accumulation_steps", 1)
        batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=bound_arguments.get("partition_spec"),
        )
        bound_arguments["batch"] = batch
        bound_arguments["partition_spec"] = partition_spec
        call = ScheduledStepCall(
            step_fn=step_fn,
            state=state,
            batch=batch,
            args=step_args,
            kwargs=step_kwargs,
            bound_arguments=bound_arguments,
            schedule=schedule,
        )
        if adapter.prepare_batch is not None:
            batch = dict(adapter.prepare_batch(call))
            bound_arguments["batch"] = batch
            call = ScheduledStepCall(
                step_fn=step_fn,
                state=state,
                batch=batch,
                args=step_args,
                kwargs=step_kwargs,
                bound_arguments=bound_arguments,
                schedule=schedule,
            )
        value_and_grad = scheduled_vag.get(call)
        loss, gradients = _run_scheduled_value_and_grad(
            value_and_grad=value_and_grad,
            graphstate=state.graphstate,
            batch=batch,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
        )
        return _apply_stage_local_gradients(
            state=state,
            gradients=gradients,
            loss=loss,
            loss_config=loss_config,
            learning_rate_fn=learning_rate_fn,
        )

    scheduled_training_step.__name__ = f"{type(schedule).__name__}_{adapter.name}_{_scheduled_step_name(step_fn)}"
    scheduled_training_step.static_argnums_ = _normalize_static_argnums(static_argnums)
    return scheduled_training_step


def compile_trainer_auxiliary(
    fn: tp.Callable[..., tp.Any],
    *,
    mesh: MeshLike | None = None,
    arguments: tp.Any | None = None,
    in_shardings: tp.Any = _UNSPECIFIED,
    out_shardings: tp.Any = _UNSPECIFIED,
    **jit_kwargs,
) -> tp.Callable[..., tp.Any]:
    """Compile nested trainer helpers through the same SpectraX jit surface."""

    return compile_trainer_step(
        fn,
        mesh=mesh,
        arguments=arguments,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        **jit_kwargs,
    )


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
            """Apply ``gradients`` to ``state`` via ``apply_gradients``.

            Args:
                args: ``(state, gradients)`` tuple.

            Returns:
                The updated state.
            """
            state, gradients = args
            return state.apply_gradients(grads=gradients)

        def skip_fn(args):
            """Return ``state`` unchanged (used when the gradient step is skipped).

            Args:
                args: ``(state, gradients)`` tuple; gradients are ignored.

            Returns:
                The original state.
            """
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
                """Slice a single leaf along axis 0 if it carries the full batch dim.

                Args:
                    arr: Pytree leaf (typically a JAX array).

                Returns:
                    The minibatch slice (when ``arr`` has the full batch
                    leading dim) or ``arr`` unchanged otherwise.
                """
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
