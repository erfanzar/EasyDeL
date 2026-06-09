# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic-axis sharding manager owned by SpectraX.

This module provides the runtime-side resolver that turns the
*symbolic* axis tokens declared in :mod:`spectrax.common_types`
(``BATCH``, ``EMBED``, ``HEAD``, ``DP``, ``FSDP``, …) into concrete
:class:`jax.sharding.PartitionSpec` objects bound to a specific
physical mesh layout.

The flow is:

1. A user instantiates :class:`PartitionAxis` to describe their
   physical mesh — which mesh-axis name corresponds to "data parallel",
   which to "tensor parallel", and so on. They can override per-tensor
   defaults (``batch_axis``, ``hidden_state_axis``, …) for non-standard
   layouts.
2. The user wraps a code region in ``with PartitionManager(paxis):``
   to publish the active mapping in a :class:`contextvars.ContextVar`.
3. Layer code calls :func:`apply_logical_sharding` (or, equivalently,
   ``manager.shard(x, …)``) with a list of *symbolic* axis names.
4. :meth:`PartitionAxis.resolve_axis` walks the symbolic list, looks
   up the configured mesh-axis name(s), and returns a list ready to
   feed into :class:`~jax.sharding.PartitionSpec`.

A small registry (:meth:`PartitionAxis.register` /
:meth:`~PartitionAxis.unregister`) lets downstream libraries add new
semantic axis names without subclassing.

Generation modes (:data:`spectrax.common_types.MODE_DECODE`,
``MODE_INSERT``) get their own ``decode_*`` overrides on
:class:`PartitionAxis`, so the same code path can shard differently
during autoregressive sampling than during training (e.g. for a KV
cache that is too small to shard along its sequence axis).
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import threading
import typing as tp
from types import TracebackType

import jax
from jax.sharding import PartitionSpec

from .. import common_types as ct
from ..core._typing import Array
from .partition import get_corrected_named_sharding, with_sharding_constraint

AxisRule: tp.TypeAlias = ct.AxisType | list[ct.AxisType] | tuple[ct.AxisType, ...]
RegisteredAxes: tp.TypeAlias = dict[str, dict[str, AxisRule]]

_CURRENT_PARTITION_MANAGER: contextvars.ContextVar[PartitionManager | None] = contextvars.ContextVar(
    "_CURRENT_PARTITION_MANAGER",
    default=None,
)
_LAST_PARTITION_MANAGER: PartitionManager | None = None


def _to_hashable(value: object) -> object:
    """Convert an arbitrary value into a stable, hashable representation.

    Used by :func:`hash_fn` to give SpectraX dataclasses a deterministic
    hash even when their fields contain mutable structures (dicts,
    lists, sets, sub-dataclasses, ad-hoc objects). The conversion is
    recursive and order-stable: dicts and sets are sorted, lists/tuples
    keep their order, and objects with a ``__dict__`` are turned into
    a ``(qualname, sorted-attr-tuple)`` pair. Anything that's already
    hashable is returned unchanged; anything that isn't and that has
    no ``__dict__`` falls back to ``repr(value)``.

    Args:
        value: The object to make hashable.

    Returns:
        A nested tuple/primitive structure suitable as a key into a
        ``dict`` or :func:`hash`.
    """
    if dataclasses.is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return tuple(sorted((str(k), _to_hashable(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_to_hashable(v) for v in value))
    if hasattr(value, "__dict__"):
        return (value.__class__.__qualname__, _to_hashable(vars(value)))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def hash_fn(self: object) -> int:
    """Generic ``__hash__`` implementation for SpectraX dataclasses.

    Builds the hash from ``self.__class__.__qualname__`` plus a stable,
    hashable view of the instance's fields (for dataclasses, only
    fields with ``compare=True`` are included; for other objects, all
    of ``vars(self)`` is used). The recursive flattening is delegated
    to :func:`_to_hashable`, so containers and nested objects all hash
    consistently.

    The function is meant to be assigned at class scope as
    ``__hash__ = hash_fn`` on dataclasses that want hashability without
    paying for ``frozen=True``. It is a free function (rather than a
    method) so the same implementation can be reused across multiple
    classes without subclassing.

    Args:
        self: The instance whose hash is being computed.

    Returns:
        A Python ``int`` suitable as ``__hash__`` output.
    """
    if dataclasses.is_dataclass(self):
        payload = tuple(
            (field.name, _to_hashable(getattr(self, field.name))) for field in dataclasses.fields(self) if field.compare
        )
        return hash((self.__class__.__qualname__, payload))
    return hash((self.__class__.__qualname__, _to_hashable(vars(self))))


def get_safe_hash_int(text: object, algorithm: str = "md5") -> int:
    """Hash any object to a deterministic, large positive integer.

    Useful for deriving stable seeds, cache keys, or shard indexes from
    arbitrary inputs (strings, configs, paths). Unlike Python's built-in
    :func:`hash`, the output is reproducible across processes (it does
    not depend on ``PYTHONHASHSEED``) and across machines, because it
    is grounded in :mod:`hashlib`.

    Args:
        text: Any object — it is stringified via :func:`str` before
            hashing, so callers can pass dicts, tuples, paths, etc.
        algorithm: The :mod:`hashlib` algorithm name (e.g. ``"md5"``,
            ``"sha1"``, ``"sha256"``). Defaults to ``"md5"`` because
            this is *not* a cryptographic primitive — speed and
            stability are what matter.

    Returns:
        The full hash digest interpreted as a big-endian unsigned
        integer.

    Raises:
        ValueError: If ``algorithm`` is not a name recognized by
            :mod:`hashlib`.
    """
    try:
        hash_object = getattr(hashlib, algorithm)(str(text).encode())
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    return int.from_bytes(hash_object.digest(), byteorder="big")


@dataclasses.dataclass(slots=True)
class PartitionAxis:
    """Map symbolic tensor-axis tokens to physical mesh-axis names.

    A ``PartitionAxis`` is the central configuration object that turns
    symbolic axis tokens defined in :mod:`spectrax.common_types`
    (``BATCH``, ``EMBED``, ``HEAD``, …) into concrete mesh-axis names
    (``"dp"``, ``"tp"``, …) that JAX understands.

    The dataclass has two layers of fields:

    * **Role fields** — ``data_parallel_axis``, ``fully_sharded_data_parallel_axis``,
      ``tensor_parallel_axis``, ``sequence_parallel_axis``,
      ``expert_parallel_axis``. These name the physical mesh axes
      that play each parallelism role on the active mesh. Defaults
      match SpectraX's standard ``("dp", "fsdp", "tp", "sp", "ep")``.
    * **Per-tensor-axis fields** — ``batch_axis``, ``sequence_axis``,
      ``query_sequence_axis``, ``key_sequence_axis``, ``head_axis``,
      ``kv_head_axis``, ``hidden_state_axis``, ``mlp_intermediate_axis``,
      ``vocab_axis``, ``expert_axis``, … and a parallel set of
      ``decode_*`` fields that override the choice when the runtime is
      in :data:`~spectrax.common_types.MODE_DECODE` /
      :data:`~spectrax.common_types.MODE_INSERT`.

    Most per-tensor-axis fields default to
    :data:`~spectrax.common_types.NOT_GIVEN`, a sentinel that
    ``__post_init__`` resolves to a sensible mapping derived from the
    role fields (e.g. ``hidden_state_axis`` defaults to
    ``tensor_parallel_axis``). A few fields that have no useful
    role-based default — ``expert_gate_axis``, ``attention_dim_axis``,
    ``attention_kv_dim_axis``, ``bias_head_sequence_axis``,
    ``bias_key_sequence_axis`` and the corresponding ``decode_*``
    overrides — default to ``None`` (replicated) instead. After
    resolution, every field has a concrete value —
    :meth:`_safety_check` will raise if any remain ``NOT_GIVEN``.

    The class is the SpectraX-owned equivalent of the EasyDeL/eFormer
    logical-sharding configuration, with the same field names so
    existing configs can be loaded directly.

    A small thread-safe class registry (:meth:`register`,
    :meth:`unregister`, :meth:`get_registered_axes`) lets downstream
    libraries add new symbolic axes without subclassing.
    """

    pipeline_parallel_axis: str = "pp"
    data_parallel_axis: str = "dp"
    fully_sharded_data_parallel_axis: str = "fsdp"
    tensor_parallel_axis: str = "tp"
    sequence_parallel_axis: str = "sp"
    expert_parallel_axis: str = "ep"

    batch_axis: ct.AxisType = ct.NOT_GIVEN
    sequence_axis: ct.AxisType = ct.NOT_GIVEN
    query_sequence_axis: ct.AxisType = ct.NOT_GIVEN
    head_axis: ct.AxisType = ct.NOT_GIVEN
    kv_head_axis: ct.AxisType = ct.NOT_GIVEN
    key_sequence_axis: ct.AxisType = ct.NOT_GIVEN
    hidden_state_axis: ct.AxisType = ct.NOT_GIVEN
    mlp_intermediate_axis: ct.AxisType = ct.NOT_GIVEN
    vocab_axis: ct.AxisType = ct.NOT_GIVEN
    expert_axis: ct.AxisType = ct.NOT_GIVEN
    expert_gate_axis: ct.AxisType = None

    attention_dim_axis: ct.AxisType = None
    attention_kv_dim_axis: ct.AxisType = None
    bias_head_sequence_axis: ct.AxisType = None
    bias_key_sequence_axis: ct.AxisType = None

    decode_batch_axis: ct.AxisType = ct.NOT_GIVEN
    decode_query_sequence_axis: ct.AxisType = None
    decode_head_axis: ct.AxisType = ct.NOT_GIVEN
    decode_kv_head_axis: ct.AxisType = ct.NOT_GIVEN
    decode_key_sequence_axis: ct.AxisType = ct.NOT_GIVEN
    decode_attention_dim_axis: ct.AxisType = None
    decode_attention_kv_dim_axis: ct.AxisType = None

    _SEMANTIC_MAP: tp.ClassVar[dict[str, str | None]] = {
        ct.BATCH: "batch_axis",
        ct.LENGTH: "sequence_axis",
        ct.QUERY_LENGTH: "query_sequence_axis",
        ct.KV_LENGTH: "key_sequence_axis",
        ct.EMBED: "hidden_state_axis",
        ct.HEAD: "head_axis",
        ct.KV_HEAD: "kv_head_axis",
        ct.MLP_INTERMEDIATE: "mlp_intermediate_axis",
        ct.VOCAB: "vocab_axis",
        ct.EXPERT: "expert_axis",
        ct.EXPERT_GATE: "expert_gate_axis",
        ct.HEAD_DIM: "attention_dim_axis",
        ct.KV_HEAD_DIM: "attention_kv_dim_axis",
        ct.BIAS_HEAD_SEQ: "bias_head_sequence_axis",
        ct.BIAS_KV_SEQ: "bias_key_sequence_axis",
        ct.EMPTY: None,
        ct.DATA_PARALLEL: "data_parallel_axis",
        ct.PIPELINE_PARALLEL: "pipeline_parallel_axis",
        ct.FULLY_SHARDED_DATA_PARALLEL: "fully_sharded_data_parallel_axis",
        ct.TENSOR_PARALLEL: "tensor_parallel_axis",
        ct.SEQUENCE_PARALLEL: "sequence_parallel_axis",
        ct.EXPERT_PARALLEL: "expert_parallel_axis",
    }
    _STANDARD_TO_GENERATION_ATTR_MAP: tp.ClassVar[dict[str, str]] = {
        "batch_axis": "decode_batch_axis",
        "query_sequence_axis": "decode_query_sequence_axis",
        "key_sequence_axis": "decode_key_sequence_axis",
        "head_axis": "decode_head_axis",
        "kv_head_axis": "decode_kv_head_axis",
        "attention_dim_axis": "decode_attention_dim_axis",
        "attention_kv_dim_axis": "decode_attention_kv_dim_axis",
    }
    _REGISTRY_LOCK: tp.ClassVar[threading.RLock] = threading.RLock()
    _REGISTERED_SEMANTIC_MAP: tp.ClassVar[dict[str, AxisRule]] = {}
    _REGISTERED_GENERATION_MAP: tp.ClassVar[dict[str, AxisRule]] = {}

    @classmethod
    def register(
        cls,
        semantic_axis: str,
        axis_rule: AxisRule,
        *,
        generation_axis_rule: AxisRule = ct.NOT_GIVEN,
        override: bool = False,
    ) -> None:
        """Register a new symbolic-axis name for all ``PartitionAxis`` instances.

        Use this when a downstream library or model needs an axis
        token that is not in the built-in ``_SEMANTIC_MAP``. The
        registration is process-wide and protected by a class-level
        :class:`threading.RLock`.

        Args:
            semantic_axis: The symbolic name to register, e.g.
                ``"__MY_CUSTOM_AXIS__"``. Whitespace is stripped. An
                empty string raises ``ValueError``.
            axis_rule: What the symbolic name resolves to. Either a
                :class:`PartitionAxis` field name (string), in which
                case lookups recurse to that field, or an arbitrary
                value that will be returned verbatim from
                :meth:`resolve_axis`.
            generation_axis_rule: Optional override used when the
                runtime is in a generation mode
                (:data:`~spectrax.common_types.GENERATION_MODES`).
                Default :data:`~spectrax.common_types.NOT_GIVEN` means
                "infer from ``axis_rule``": if ``axis_rule`` is a
                standard field name with a ``decode_*`` companion,
                that companion is used; otherwise generation falls
                back to ``axis_rule`` itself.
            override: If ``False`` (default), registering a name that
                already exists (built-in or custom) raises
                :class:`ValueError`. If ``True``, the existing entry
                is replaced.

        Raises:
            ValueError: If ``semantic_axis`` is empty after stripping
                or if it is already registered and ``override`` is
                ``False``.
        """
        name = str(semantic_axis).strip()
        if not name:
            raise ValueError("`semantic_axis` must be a non-empty string.")
        if axis_rule is ct.NOT_GIVEN:
            raise ValueError("`axis_rule` must be provided; got NOT_GIVEN.")
        with cls._REGISTRY_LOCK:
            if not override and (name in cls._SEMANTIC_MAP or name in cls._REGISTERED_SEMANTIC_MAP):
                raise ValueError(f"Semantic axis '{name}' already exists. Use override=True to replace it.")
            cls._REGISTERED_SEMANTIC_MAP[name] = axis_rule
            if generation_axis_rule is ct.NOT_GIVEN:
                inferred = cls._STANDARD_TO_GENERATION_ATTR_MAP.get(axis_rule) if isinstance(axis_rule, str) else None
                if inferred is None:
                    cls._REGISTERED_GENERATION_MAP.pop(name, None)
                else:
                    cls._REGISTERED_GENERATION_MAP[name] = inferred
            else:
                cls._REGISTERED_GENERATION_MAP[name] = generation_axis_rule

    @classmethod
    def unregister(cls, semantic_axis: str, *, missing_ok: bool = True) -> None:
        """Remove a previously :meth:`register`-ed symbolic axis.

        Built-in axes from ``_SEMANTIC_MAP`` cannot be removed; only
        names registered at runtime through :meth:`register` are
        affected.

        Args:
            semantic_axis: The symbolic name to remove. Whitespace
                is stripped. An empty string raises ``ValueError``.
            missing_ok: If ``True`` (default), silently ignore the
                request when ``semantic_axis`` is not registered. If
                ``False``, raise :class:`KeyError` instead.

        Raises:
            ValueError: If ``semantic_axis`` is empty.
            KeyError: If the axis is not registered and
                ``missing_ok`` is ``False``.
        """
        name = str(semantic_axis).strip()
        if not name:
            raise ValueError("`semantic_axis` must be a non-empty string.")
        with cls._REGISTRY_LOCK:
            removed = name in cls._REGISTERED_SEMANTIC_MAP or name in cls._REGISTERED_GENERATION_MAP
            cls._REGISTERED_SEMANTIC_MAP.pop(name, None)
            cls._REGISTERED_GENERATION_MAP.pop(name, None)
            if not removed and not missing_ok:
                raise KeyError(f"Semantic axis '{name}' is not registered.")

    @classmethod
    def clear_registered_axes(cls) -> None:
        """Drop *all* runtime-registered symbolic axes.

        Restores the registry to its initial state — only the
        built-in axes from ``_SEMANTIC_MAP`` remain available. Useful
        in tests that need to isolate state between cases.
        """
        with cls._REGISTRY_LOCK:
            cls._REGISTERED_SEMANTIC_MAP.clear()
            cls._REGISTERED_GENERATION_MAP.clear()

    @classmethod
    def get_registered_axes(cls) -> RegisteredAxes:
        """Return a snapshot of all runtime-registered symbolic axes.

        Returns:
            A new ``dict`` mapping each registered semantic-axis name
            to a sub-dict with two keys:

            * ``"axis_rule"`` — the rule passed to :meth:`register`.
            * ``"generation_axis_rule"`` — either the rule passed
              under that argument or
              :data:`~spectrax.common_types.NOT_GIVEN` if the caller
              did not provide one.

            The returned dict is a copy; mutating it does not affect
            the registry.
        """
        with cls._REGISTRY_LOCK:
            return {
                name: {
                    "axis_rule": cls._REGISTERED_SEMANTIC_MAP[name],
                    "generation_axis_rule": cls._REGISTERED_GENERATION_MAP.get(name, ct.NOT_GIVEN),
                }
                for name in cls._REGISTERED_SEMANTIC_MAP
            }

    @classmethod
    def _lookup_semantic_mapping(cls, semantic_axis: str) -> AxisRule | None:
        """Look up a symbolic axis name in the registry, then in the built-ins.

        Runtime-registered axes shadow built-in ones with the same
        name. Returns ``None`` if the axis is not known at all (the
        caller is expected to treat this as an unknown-axis error).

        Args:
            semantic_axis: Semantic axis value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if semantic_axis in cls._REGISTERED_SEMANTIC_MAP:
            return cls._REGISTERED_SEMANTIC_MAP[semantic_axis]
        return cls._SEMANTIC_MAP.get(semantic_axis)

    @classmethod
    def _lookup_generation_mapping(cls, semantic_axis: str) -> AxisRule:
        """Look up the generation-mode override for a registered axis.

        Returns :data:`~spectrax.common_types.NOT_GIVEN` if the axis
        is unregistered or has no explicit generation override (in
        which case :meth:`resolve_axis` falls back to the standard
        rule, possibly via the ``decode_*`` field).

        Args:
            semantic_axis: Semantic axis value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return cls._REGISTERED_GENERATION_MAP.get(semantic_axis, ct.NOT_GIVEN)

    def _resolve_axis_rule(self, axis_rule: AxisRule, _visited: set[str] | None = None) -> AxisRule:
        """Recursively resolve an axis rule down to a concrete mesh-axis name.

        Handles the four shapes a rule can take:

        * **list / tuple** — element-wise resolution; the result keeps
          the same container type (list stays list, tuple stays tuple)
          which lets callers express *fused* mesh axes.
        * **string that names a field** — replace with the value of
          that field (e.g. ``"data_parallel_axis"`` becomes ``"dp"``).
        * **string that names another semantic axis** — recurse, with
          a ``_visited`` guard against cycles in the registry.
        * **anything else** — return verbatim.

        Args:
            axis_rule: The rule to resolve.
            _visited: Internal cycle-detection set; callers should
                leave this at its ``None`` default.

        Returns:
            A fully resolved mesh-axis spec (string, list, tuple, or
            arbitrary value).

        Raises:
            ValueError: If a cycle is detected in the registered
                semantic-axis graph.
        """
        if isinstance(axis_rule, list):
            return tp.cast(
                AxisRule,
                [self._resolve_axis_rule(item, set(_visited) if _visited is not None else None) for item in axis_rule],
            )
        if isinstance(axis_rule, tuple):
            return tp.cast(
                AxisRule,
                tuple(
                    self._resolve_axis_rule(item, set(_visited) if _visited is not None else None) for item in axis_rule
                ),
            )
        if isinstance(axis_rule, str):
            if hasattr(self, axis_rule):
                return getattr(self, axis_rule)
            mapped = self._lookup_semantic_mapping(axis_rule)
            if mapped is not None:
                visited = set() if _visited is None else set(_visited)
                if axis_rule in visited:
                    raise ValueError(f"Cyclic semantic axis registration detected at '{axis_rule}'.")
                visited.add(axis_rule)
                return self._resolve_axis_rule(mapped, _visited=visited)
        return axis_rule

    def __post_init__(self) -> None:
        """Resolve every ``NOT_GIVEN`` field to a concrete default.

        Per-tensor-axis fields default to
        :data:`~spectrax.common_types.NOT_GIVEN`, which is a marker
        meaning "compute me from the role fields". This hook walks
        through each such field, fills in the standard mapping (e.g.
        ``hidden_state_axis = tensor_parallel_axis``), and then runs
        :meth:`_safety_check` to make sure nothing is left unresolved.

        The defaults follow conventional transformer sharding:
        batches go on ``(fsdp, dp)``, sequences on ``sp``, model
        weights on ``tp``, and experts on ``ep``. ``decode_*`` fields
        mirror their non-decode counterparts unless explicitly set.
        """
        resolved_values: dict[str, ct.AxisType] = {}

        def resolve_field(name: str, default_logic: tp.Callable[[], ct.AxisType]) -> None:
            """Helper: store ``default_logic()`` for ``name`` if currently ``NOT_GIVEN``.

            Otherwise keep the user-provided value. Idempotent within
            a single ``__post_init__`` call.

            Args:
                name: Name used for lookup, logging, or registration.
                default_logic: Default logic value consumed by this operation.
            """
            current_value = getattr(self, name)
            if current_value is ct.NOT_GIVEN:
                resolved_values[name] = default_logic()
            elif name not in resolved_values:
                resolved_values[name] = current_value

        def get_resolved(name: str) -> ct.AxisType:
            """Return the just-resolved value for ``name`` (or the live attr).

            Args:
                name: Name used for lookup, logging, or registration.

            Returns:
                Return the just-resolved value for ``name`` (or the live attr).
            """
            return resolved_values.get(name, getattr(self, name))

        resolve_field("batch_axis", lambda: (self.fully_sharded_data_parallel_axis, self.data_parallel_axis))
        resolve_field("sequence_axis", lambda: self.sequence_parallel_axis)
        resolve_field("query_sequence_axis", lambda: self.sequence_parallel_axis)
        resolve_field("head_axis", lambda: self.tensor_parallel_axis)
        resolve_field("kv_head_axis", lambda: self.tensor_parallel_axis)
        resolve_field("key_sequence_axis", lambda: self.sequence_parallel_axis)
        resolve_field("hidden_state_axis", lambda: self.tensor_parallel_axis)
        resolve_field("mlp_intermediate_axis", lambda: self.tensor_parallel_axis)
        resolve_field("vocab_axis", lambda: self.tensor_parallel_axis)
        resolve_field("expert_axis", lambda: self.expert_parallel_axis)
        resolve_field("decode_batch_axis", lambda: get_resolved("batch_axis"))
        resolve_field("decode_head_axis", lambda: get_resolved("head_axis"))
        resolve_field("decode_kv_head_axis", lambda: get_resolved("kv_head_axis"))
        resolve_field("decode_key_sequence_axis", lambda: get_resolved("key_sequence_axis"))

        for field in dataclasses.fields(self):
            if field.name not in resolved_values:
                resolved_values[field.name] = getattr(self, field.name)
        for name, value in resolved_values.items():
            object.__setattr__(self, name, value)
        self._safety_check()

    def _safety_check(self) -> None:
        """Assert that no field is still ``NOT_GIVEN`` after ``__post_init__``.

        Raises:
            ValueError: If any field is still the
                :data:`~spectrax.common_types.NOT_GIVEN` sentinel,
                naming the field for easier debugging.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is ct.NOT_GIVEN:
                raise ValueError(f"Partitioning rule `{field.name}` was not resolved.")

    def resolve_axis(
        self,
        axes: tp.Sequence[str | None],
        mode: ct.RUNTIME_MODE_TYPES | str,
    ) -> list[ct.AxisType]:
        """Resolve a list of symbolic axes to concrete mesh-axis specs.

        This is the heart of the manager: given a per-dimension list
        of symbolic axis tokens (and the active runtime mode), return
        a list ready to feed into :class:`jax.sharding.PartitionSpec`.

        Each entry of ``axes`` is processed independently:

        * ``None`` or :data:`~spectrax.common_types.EMPTY` → ``None``
          in the result (replicated).
        * A list/tuple of symbolic names → fused mesh axis. Each
          sub-name is mapped via :meth:`_lookup_semantic_mapping`.
        * A bare symbolic name → looked up in the registry then in
          ``_SEMANTIC_MAP``.

        When ``mode`` is in
        :data:`~spectrax.common_types.GENERATION_MODES`, the resolver
        consults the ``decode_*`` overrides and any
        ``generation_axis_rule`` registered on the class — letting
        callers specialize sharding for autoregressive paths.

        After symbolic-to-field resolution, :meth:`_resolve_axis_rule`
        recursively expands the rule into final mesh-axis names.

        Args:
            axes: Per-tensor-dimension symbolic axes. Length must
                match the rank of the tensor that will receive the
                spec.
            mode: Runtime mode token (one of
                :data:`~spectrax.common_types.MODE_TRAIN`,
                ``MODE_PREFILL``, ``MODE_DECODE``, ``MODE_INSERT``).

        Returns:
            A list of mesh-axis specs, one per entry in ``axes``,
            ready to be unpacked into a ``PartitionSpec``.

        Raises:
            ValueError: If a symbolic axis is unknown, or a resolved
                rule is still ``NOT_GIVEN``.
        """
        resolved_rules: list[ct.AxisType] = []
        for axis_name in axes:
            if axis_name is None or axis_name == ct.EMPTY:
                resolved_rules.append(None)
                continue

            if isinstance(axis_name, list | tuple):
                standard_rule: list[ct.AxisType] = []
                for sub_axis in axis_name:
                    if sub_axis is None or sub_axis == ct.EMPTY:
                        standard_rule.append(None)
                        continue
                    sub_mapped = self._lookup_semantic_mapping(sub_axis)
                    if sub_mapped is None:
                        raise ValueError(f"Unknown semantic axis name: '{sub_axis}'")
                    standard_rule.append(sub_mapped)
            else:
                standard_rule = self._lookup_semantic_mapping(axis_name)
                if standard_rule is None:
                    raise ValueError(f"Unknown semantic axis name: '{axis_name}'")

            target_rule = standard_rule
            if mode in ct.GENERATION_MODES:
                if isinstance(axis_name, list | tuple):
                    gen_composite: list[ct.AxisType] = []
                    any_changed = False
                    for idx, sub_axis in enumerate(axis_name):
                        sub_standard = standard_rule[idx]
                        sub_target = sub_standard
                        sub_gen = self._lookup_generation_mapping(sub_axis)
                        if sub_gen is not ct.NOT_GIVEN:
                            sub_target = sub_gen
                            any_changed = True
                        elif isinstance(sub_standard, str):
                            sub_gen_attr = self._STANDARD_TO_GENERATION_ATTR_MAP.get(sub_standard)
                            if sub_gen_attr and hasattr(self, sub_gen_attr):
                                sub_gen_val = getattr(self, sub_gen_attr)
                                if sub_gen_val is not None and sub_gen_val is not ct.NOT_GIVEN:
                                    sub_target = sub_gen_attr
                                    any_changed = True
                        gen_composite.append(sub_target)
                    if any_changed:
                        target_rule = gen_composite
                else:
                    custom_gen_rule = self._lookup_generation_mapping(axis_name)
                    if custom_gen_rule is not ct.NOT_GIVEN:
                        target_rule = custom_gen_rule
                    elif isinstance(standard_rule, str):
                        gen_attr_name = self._STANDARD_TO_GENERATION_ATTR_MAP.get(standard_rule)
                        if gen_attr_name and hasattr(self, gen_attr_name):
                            gen_val = getattr(self, gen_attr_name)
                            if gen_val is not None and gen_val is not ct.NOT_GIVEN:
                                target_rule = gen_attr_name

            mesh_axis_rule = self._resolve_axis_rule(target_rule)
            if mesh_axis_rule is ct.NOT_GIVEN:
                raise ValueError(f"Resolved axis rule for '{axis_name}' is still NOT_GIVEN.")
            resolved_rules.append(mesh_axis_rule)
        return resolved_rules

    def resolve_spec(
        self,
        axes: tp.Sequence[str | None],
        mode: ct.RUNTIME_MODE_TYPES | str,
    ) -> PartitionSpec:
        """Resolve symbolic axes directly to a :class:`jax.sharding.PartitionSpec`.

        Thin wrapper around :meth:`resolve_axis` that wraps the result
        in a ``PartitionSpec``. Use this when you want a JAX-ready
        spec (e.g. for ``with_sharding_constraint``); use
        :meth:`resolve_axis` if you need the intermediate list.

        Args:
            axes: Per-tensor-dimension symbolic axes.
            mode: Runtime mode token.

        Returns:
            A :class:`jax.sharding.PartitionSpec` whose entries are
            the resolved mesh-axis specs.
        """
        return PartitionSpec(*self.resolve_axis(axes=axes, mode=mode))

    def to_dict(self) -> dict[str, ct.AxisType]:
        """Return a plain ``dict`` of every dataclass field.

        Useful for serialization (config dump, JSON export) and for
        comparing two ``PartitionAxis`` instances by value. Includes
        every field — both role and per-tensor-axis — but not the
        class-level registry.

        Returns:
            Return a plain ``dict`` of every dataclass field.
        """
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    __hash__ = hash_fn


@dataclasses.dataclass(slots=True)
class PartitionManager:
    """Context-managed handle that applies sharding through a ``PartitionAxis``.

    A ``PartitionManager`` couples a :class:`PartitionAxis`
    (the symbolic-to-physical mapping) with a thread-safe context-var
    so layer code can find the active manager without explicit
    plumbing. Used as::

        with PartitionManager(my_paxis):
            apply_logical_sharding(x, axes=[BATCH, EMBED], mode=MODE_TRAIN)

    On enter, ``self`` becomes the current manager (visible to
    :func:`get_current_partition_manager`); on exit, the previous
    state is restored. The most-recently-instantiated manager is
    *also* recorded in a process-wide slot retrievable via
    :func:`get_partition_manager`, so code that runs outside any
    ``with`` block (e.g. in tests, or top-level scripts) still finds
    a sensible default.

    Attributes:
        paxis: The :class:`PartitionAxis` to delegate to.
        _context_token: Internal token returned by the context-var
            ``set`` call; held only between ``__enter__`` and
            ``__exit__``. Not part of the public API.
    """

    paxis: PartitionAxis
    _context_token: contextvars.Token[PartitionManager | None] | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate ``paxis`` and record this instance as the most-recent manager."""
        global _LAST_PARTITION_MANAGER
        if not isinstance(self.paxis, PartitionAxis):
            raise TypeError(f"Expected PartitionAxis, got {type(self.paxis)}")
        _LAST_PARTITION_MANAGER = self

    def __enter__(self) -> PartitionManager:
        """Push ``self`` onto the active-manager :class:`contextvars.ContextVar`.

        Returns:
            Result described by this helper.
        """
        global _LAST_PARTITION_MANAGER
        token = _CURRENT_PARTITION_MANAGER.set(self)
        object.__setattr__(self, "_context_token", token)
        _LAST_PARTITION_MANAGER = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Restore the previous active manager. Always re-raises (returns ``False``).

        Args:
            exc_type: Exc type value consumed by this operation.
            exc_val: Exc val value consumed by this operation.
            exc_tb: Exc tb value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        token = getattr(self, "_context_token", None)
        if token is not None:
            _CURRENT_PARTITION_MANAGER.reset(token)
            object.__setattr__(self, "_context_token", None)
        return False

    def resolve(
        self,
        axes: tp.Sequence[str | None] | ct.DynamicShardingAxes = ct.NOT_GIVEN,
        mode: ct.RUNTIME_MODE_TYPES | int | str = ct.NOT_GIVEN,
        dynamic_axes: ct.DynamicShardingAxes | None = ct.NOT_GIVEN,
        shape: tp.Sequence[int] = ct.NOT_GIVEN,
    ) -> PartitionSpec:
        """Resolve ``axes``/``mode`` (or a ``DynamicShardingAxes``) to a ``PartitionSpec``.

        This is the most flexible entry point on the manager. It
        accepts three calling conventions:

        1. Explicit ``axes`` + ``mode`` — direct delegation to
           :meth:`PartitionAxis.resolve_spec`.
        2. A :class:`~spectrax.common_types.DynamicShardingAxes`
           subclass or instance passed via ``dynamic_axes`` (or as
           the ``axes`` positional arg, which is auto-detected).
           ``axes`` and ``mode`` are read off it.
        3. As above, but ``mode`` is an *integer* axis index. The
           manager dispatches to ``MODE_DECODE`` if
           ``shape[mode] == 1`` and ``MODE_TRAIN`` otherwise — useful
           for code that should auto-pick its sharding based on the
           tensor's batch shape.

        Args:
            axes: Either an axis-spec list/tuple or a ``DynamicShardingAxes``
                that bundles axes+mode. If left as
                :data:`~spectrax.common_types.NOT_GIVEN`, ``dynamic_axes``
                must be provided.
            mode: Runtime mode token, an integer (see above), or
                ``NOT_GIVEN`` to fall back to ``dynamic_axes.mode``.
            dynamic_axes: Optional pre-built ``DynamicShardingAxes``.
            shape: The tensor's shape — required only when ``mode``
                is an integer index.

        Returns:
            A :class:`jax.sharding.PartitionSpec`.

        Raises:
            ValueError: If neither ``axes``/``mode`` nor
                ``dynamic_axes`` was provided, or if integer-mode
                dispatch is used without ``shape``.
        """
        if dynamic_axes is ct.NOT_GIVEN and axes is not ct.NOT_GIVEN:
            if isinstance(axes, tuple) and hasattr(axes, "_fields"):
                dynamic_axes = axes
                axes = ct.NOT_GIVEN
            elif isinstance(axes, type) and issubclass(axes, tuple) and hasattr(axes, "_fields"):
                dynamic_axes = axes
                axes = ct.NOT_GIVEN
            elif hasattr(axes, "axes") and hasattr(axes, "mode"):
                dynamic_axes = axes
                axes = ct.NOT_GIVEN

        if axes is ct.NOT_GIVEN or mode is ct.NOT_GIVEN:
            if dynamic_axes is ct.NOT_GIVEN:
                raise ValueError("if axes or mode is empty you should provide dynamic axes")
            axes = dynamic_axes.axes
            mode = dynamic_axes.mode
        if isinstance(mode, int):
            if shape is ct.NOT_GIVEN:
                raise ValueError("when using dynamic mode detection shape should be provided")
            mode = ct.MODE_DECODE if shape[mode] == 1 else ct.MODE_TRAIN
        return self.paxis.resolve_spec(axes, mode)

    def shard(
        self,
        x: jax.Array,
        axes: tp.Sequence[str | None] = ct.NOT_GIVEN,
        mode: ct.RUNTIME_MODE_TYPES | int | str = ct.NOT_GIVEN,
        dynamic_axes: ct.DynamicShardingAxes | None = ct.NOT_GIVEN,
        auto_correct: bool = True,
    ) -> jax.Array:
        """Apply a sharding constraint to ``x`` using the manager's ``PartitionAxis``.

        Equivalent to::

            spec = self.resolve(axes, mode, dynamic_axes, x.shape)
            return with_sharding_constraint(x, spec)

        with one extra step: when ``auto_correct`` is true (default),
        the resolved spec is passed through
        :func:`~spectrax.sharding.partition.get_corrected_named_sharding`
        to drop axes that don't divide the tensor's shape evenly,
        making the call safe under irregular shapes (e.g. attention
        head counts that are coprime with the TP mesh size).

        Args:
            x: The tensor to shard.
            axes: Same as :meth:`resolve`.
            mode: Same as :meth:`resolve`.
            dynamic_axes: Same as :meth:`resolve`.
            auto_correct: If true, drop mesh axes that don't divide
                the tensor's corresponding dimension; if false, the
                raw ``PartitionSpec`` is used and a downstream JAX
                error will fire on a bad fit.

        Returns:
            ``x`` with the resulting sharding constraint attached.
        """
        spec = self.resolve(axes=axes, mode=mode, dynamic_axes=dynamic_axes, shape=x.shape)
        if auto_correct:
            spec = get_corrected_named_sharding(x.shape, spec, raise_mesh_error=False).spec
        return tp.cast(Array, with_sharding_constraint(x, spec))

    def __str__(self) -> str:
        """Return a short, hash-stable string repr (the manager is opaque).

        Returns:
            Return a short, hash-stable string repr (the manager is opaque).
        """
        return "PartitionManager(...)"

    def __repr__(self) -> str:
        """Identical to :meth:`__str__` — the manager is intentionally opaque.

        Returns:
            Result described by this helper.
        """
        return "PartitionManager(...)"

    __hash__ = hash_fn


def get_current_partition_manager() -> PartitionManager | None:
    """Return the manager currently active on this thread, or ``None``.

    Reads from the :class:`contextvars.ContextVar` set by
    ``with PartitionManager(...):``. Use this in layer code that
    needs the *active* manager (the one for the enclosing ``with``).

    Returns:
        Return the manager currently active on this thread, or ``None``.
    """
    return _CURRENT_PARTITION_MANAGER.get()


def get_partition_manager() -> PartitionManager | None:
    """Return the most-recently-instantiated manager process-wide, or ``None``.

    Falls back when no ``with`` block is active. Useful for scripts
    that build a single manager at startup and never re-enter it.

    Returns:
        Return the most-recently-instantiated manager process-wide, or ``None``.
    """
    return _LAST_PARTITION_MANAGER


def apply_logical_sharding(
    x: jax.Array,
    partition_manager: PartitionManager | None = ct.NOT_GIVEN,
    axes: tp.Sequence[str | None] = ct.NOT_GIVEN,
    mode: ct.RUNTIME_MODE_TYPES | int | str = ct.NOT_GIVEN,
    dynamic_axes: ct.DynamicShardingAxes | None = ct.NOT_GIVEN,
    auto_correct: bool = True,
) -> jax.Array:
    """Top-level helper: shard ``x`` using whichever manager is in scope.

    This is the function layer code should call. It looks up the
    active :class:`PartitionManager` in this order:

    1. The explicit ``partition_manager`` argument, if provided.
    2. The thread-local active manager from ``with PartitionManager(...)``.
    3. The most-recently-instantiated manager process-wide.

    If none of those is available, a :class:`ValueError` is raised.

    Args:
        x: The tensor to shard.
        partition_manager: Optional explicit manager.
        axes: Per-dimension symbolic axes (forwarded to
            :meth:`PartitionManager.shard`).
        mode: Runtime mode token (forwarded).
        dynamic_axes: Optional :class:`~spectrax.common_types.DynamicShardingAxes`.
        auto_correct: Whether to drop mesh axes that don't divide
            the tensor's shape (forwarded).

    Returns:
        The sharded tensor.

    Raises:
        ValueError: If no manager is available anywhere.
    """
    resolved_manager = partition_manager
    if resolved_manager is ct.NOT_GIVEN or resolved_manager is None:
        resolved_manager = get_current_partition_manager() or get_partition_manager()
    if resolved_manager is None:
        raise ValueError(
            "No PartitionManager is available. Provide `partition_manager` or use `with PartitionManager(...)`."
        )
    return resolved_manager.shard(
        x=x,
        axes=axes,
        mode=mode,
        dynamic_axes=dynamic_axes,
        auto_correct=auto_correct,
    )


__all__ = [
    "PartitionAxis",
    "PartitionManager",
    "apply_logical_sharding",
    "get_current_partition_manager",
    "get_partition_manager",
    "get_safe_hash_int",
]
