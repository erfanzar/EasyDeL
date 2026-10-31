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
This module provides classes and functions for managing JAX sharding configurations
and applying sharding constraints within a context.

It includes the `PartitionAxis` class for defining logical-to-physical axis mappings
and the `PartitionManager` context manager for applying these rules.
"""

import contextvars
import dataclasses
import hashlib
import threading
import typing as tp

import jax
from jax.sharding import PartitionSpec

from eformer.common_types import (
    BATCH,
    BIAS_HEAD_SEQ,
    BIAS_KV_SEQ,
    DATA_PARALLEL,
    EMBED,
    EMPTY,
    EXPERT,
    EXPERT_GATE,
    EXPERT_PARALLEL,
    FULLY_SHARDED_DATA_PARALLEL,
    GENERATION_MODES,
    HEAD,
    HEAD_DIM,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_LENGTH,
    LENGTH,
    MLP_INTERMEDIATE,
    MODE_DECODE,
    MODE_TRAIN,
    NOT_GIVEN,
    QUERY_LENGTH,
    RUNTIME_MODE_TYPES,
    SEQUENCE_PARALLEL,
    TENSOR_PARALLEL,
    VOCAB,
    AxisType,
    DynamicShardingAxes,
)
from eformer.pytree import PyTree, xTree

from .constraints import get_corrected_named_sharding, with_sharding_constraint

_CURRENT_PARTITION_MANAGER = contextvars.ContextVar("_CURRENT_PARTITION_MANAGER", default=None)
_LAST_PARTITION_MANAGER: tp.Any = None


def _to_hashable(value: tp.Any) -> tp.Any:
    """Convert nested structures and dataclass-like objects to hashable tuples."""
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


def hash_fn(self) -> int:
    """Compute a hash value using dataclass fields (or object dict fallback)."""
    if dataclasses.is_dataclass(self):
        payload = tuple(
            (field.name, _to_hashable(getattr(self, field.name))) for field in dataclasses.fields(self) if field.compare
        )
        return hash((self.__class__.__qualname__, payload))

    return hash((self.__class__.__qualname__, _to_hashable(vars(self))))


def get_safe_hash_int(text, algorithm="md5"):
    """Convert text to an integer hash using the specified algorithm.

    Provides a safe way to generate integer hashes from strings, useful
    for creating hashable keys from complex objects.

    Args:
        text: The text to hash. Will be converted to string if not already.
        algorithm: Hash algorithm to use. Defaults to "md5". Supports any
            algorithm available in hashlib (e.g., "sha256", "sha1").

    Returns:
        An integer representation of the hash digest.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
        Exception: If any other error occurs during hash generation.

    Example:
        >>> get_safe_hash_int("hello world")
        309817674445039181685702831361671
        >>> get_safe_hash_int("hello world", algorithm="sha256")
        ...  # Different integer value
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


class PartitionAxis(xTree):
    """
    Configuration for partitioning model axes across a device mesh.

    Defines the mesh dimension names for standard parallelism strategies and maps
    logical model axes to these dimensions. Allows overriding defaults.

    Mesh Dimensions Attributes:
        data_parallel_axis: Name for data parallel mesh dim. Default: "dp".
        fully_sharded_data_parallel_axis: Name for FSDP mesh dim. Default: "fsdp".
        tensor_parallel_axis: Name for tensor parallel mesh dim. Default: "tp".
        sequence_parallel_axis: Name for sequence parallel mesh dim. Default: "sp".
        expert_parallel_axis: Name for expert parallel mesh dim (MoE). Default: "ep".

    Logical Model Axes Attributes:
        Maps logical tensor axes (like batch, sequence, hidden) to one or more
        mesh dimension names defined above, or None if not partitioned.
        Defaults are derived from the standard mesh dimension names but can be
        overridden during instantiation. For example, `head_axis` defaults to
        the value of `tensor_parallel_axis` ('tp').

        batch_axis: Mesh axis for the batch dimension.
        sequence_axis: Mesh axis for the general sequence length dimension.
        query_sequence_axis: Mesh axis for the query sequence length dimension.
        head_axis: Mesh axis for the attention head dimension.
        key_sequence_axis: Mesh axis for the key/value sequence length dimension.
        hidden_state_axis: Mesh axis for the embedding or hidden state dimension.
        mlp_intermediate_axis: Mesh axis for the intermediate dimension in MLP layers.
        vocab_axis: Mesh axis for the vocabulary dimension.
        expert_axis: Mesh axis for the expert dimension.
        expert_gate_axis: Mesh axis for the expert gate dimension.
        attention_dim_axis: Mesh axis for the dimension within each attention head.
        bias_head_sequence_axis: Mesh axis for bias related to head and sequence dimensions.
        bias_key_sequence_axis: Mesh axis for bias related to key/value sequence dimensions.

        decode_batch_axis: Mesh axis for the batch dimension during decoding.
        decode_query_sequence_axis: Mesh axis for the query sequence length during decoding.
        decode_head_axis: Mesh axis for the attention head dimension during decoding.
        decode_key_sequence_axis: Mesh axis for the key/value sequence length during decoding.
        decode_attention_dim_axis: Mesh axis for the dimension within each attention head during decoding.
    """

    data_parallel_axis: str = "dp"
    fully_sharded_data_parallel_axis: str = "fsdp"
    tensor_parallel_axis: str = "tp"
    sequence_parallel_axis: str = "sp"
    expert_parallel_axis: str = "ep"

    batch_axis: AxisType = NOT_GIVEN
    sequence_axis: AxisType = NOT_GIVEN
    query_sequence_axis: AxisType = NOT_GIVEN
    head_axis: AxisType = NOT_GIVEN
    kv_head_axis: AxisType = NOT_GIVEN
    key_sequence_axis: AxisType = NOT_GIVEN
    hidden_state_axis: AxisType = NOT_GIVEN
    mlp_intermediate_axis: AxisType = NOT_GIVEN
    vocab_axis: AxisType = NOT_GIVEN
    expert_axis: AxisType = NOT_GIVEN
    expert_gate_axis: AxisType = None

    attention_dim_axis: AxisType = None
    attention_kv_dim_axis: AxisType = None
    bias_head_sequence_axis: AxisType = None
    bias_key_sequence_axis: AxisType = None

    decode_batch_axis: AxisType = NOT_GIVEN
    decode_query_sequence_axis: AxisType = None
    decode_head_axis: AxisType = NOT_GIVEN
    decode_kv_head_axis: AxisType = NOT_GIVEN
    decode_key_sequence_axis: AxisType = NOT_GIVEN
    decode_attention_dim_axis: AxisType = None
    decode_attention_kv_dim_axis: AxisType = None

    _SEMANTIC_MAP: tp.ClassVar[dict[str, str]] = {
        BATCH: "batch_axis",
        LENGTH: "sequence_axis",
        QUERY_LENGTH: "query_sequence_axis",
        KV_LENGTH: "key_sequence_axis",
        EMBED: "hidden_state_axis",
        HEAD: "head_axis",
        KV_HEAD: "kv_head_axis",
        MLP_INTERMEDIATE: "mlp_intermediate_axis",
        VOCAB: "vocab_axis",
        EXPERT: "expert_axis",
        EXPERT_GATE: "expert_gate_axis",
        HEAD_DIM: "attention_dim_axis",
        KV_HEAD_DIM: "attention_kv_dim_axis",
        BIAS_HEAD_SEQ: "bias_head_sequence_axis",
        BIAS_KV_SEQ: "bias_key_sequence_axis",
        EMPTY: None,
        DATA_PARALLEL: "data_parallel_axis",
        FULLY_SHARDED_DATA_PARALLEL: "fully_sharded_data_parallel_axis",
        TENSOR_PARALLEL: "tensor_parallel_axis",
        SEQUENCE_PARALLEL: "sequence_parallel_axis",
        EXPERT_PARALLEL: "expert_parallel_axis",
    }

    """
	Maps semantic axis name constants (e.g., BATCH) to their corresponding
	attribute names in the PartitionAxis class (e.g., "batch_axis").
	"""

    _STANDARD_TO_GENERATION_ATTR_MAP: tp.ClassVar[dict[str, str]] = {
        "batch_axis": "decode_batch_axis",
        "query_sequence_axis": "decode_query_sequence_axis",
        "key_sequence_axis": "decode_key_sequence_axis",
        "head_axis": "decode_head_axis",
        "kv_head_axis": "decode_kv_head_axis",
        "attention_dim_axis": "decode_attention_dim_axis",
        "attention_kv_dim_axis": "decode_attention_kv_dim_axis",
    }
    """
	Maps standard axis attribute names to their corresponding generation-specific
	attribute names. Used to apply different sharding rules during generation modes.
	"""

    _REGISTRY_LOCK: tp.ClassVar[threading.RLock] = threading.RLock()
    _REGISTERED_SEMANTIC_MAP: tp.ClassVar[dict[str, tp.Any]] = {}
    _REGISTERED_GENERATION_MAP: tp.ClassVar[dict[str, tp.Any]] = {}

    @classmethod
    def register(
        cls,
        semantic_axis: str,
        axis_rule: tp.Any,
        *,
        generation_axis_rule: tp.Any = NOT_GIVEN,
        override: bool = False,
    ) -> None:
        """Register a semantic axis mapping globally.

        This updates a process-global registry used by all current/future
        ``PartitionAxis`` instances. Mapping values may be:
        - an attribute name on ``PartitionAxis`` (for example ``"head_axis"``),
        - a literal mesh axis name (for example ``"tp"``),
        - a tuple/list of either of the above,
        - or ``None``.

        Args:
            semantic_axis: Semantic axis token to register.
            axis_rule: Resolution rule for standard/train mode.
            generation_axis_rule: Optional explicit rule for generation modes.
                If omitted and ``axis_rule`` maps to a known standard attribute,
                the corresponding decode attribute mapping is inferred.
            override: If ``False``, raises when ``semantic_axis`` already exists
                in built-in or custom maps. Set ``True`` to replace existing rules.
        """
        name = str(semantic_axis).strip()
        if not name:
            raise ValueError("`semantic_axis` must be a non-empty string.")

        with cls._REGISTRY_LOCK:
            is_existing_builtin = name in cls._SEMANTIC_MAP
            is_existing_custom = name in cls._REGISTERED_SEMANTIC_MAP
            if not override and (is_existing_builtin or is_existing_custom):
                raise ValueError(f"Semantic axis '{name}' already exists. Use override=True to replace it.")

            cls._REGISTERED_SEMANTIC_MAP[name] = axis_rule
            if generation_axis_rule is NOT_GIVEN:
                inferred = None
                if isinstance(axis_rule, str):
                    inferred = cls._STANDARD_TO_GENERATION_ATTR_MAP.get(axis_rule)
                if inferred is not None:
                    cls._REGISTERED_GENERATION_MAP[name] = inferred
                else:
                    cls._REGISTERED_GENERATION_MAP.pop(name, None)
            else:
                cls._REGISTERED_GENERATION_MAP[name] = generation_axis_rule

    @classmethod
    def unregister(cls, semantic_axis: str, *, missing_ok: bool = True) -> None:
        """Remove a previously registered semantic axis mapping."""
        name = str(semantic_axis).strip()
        if not name:
            raise ValueError("`semantic_axis` must be a non-empty string.")
        with cls._REGISTRY_LOCK:
            removed = False
            if name in cls._REGISTERED_SEMANTIC_MAP:
                cls._REGISTERED_SEMANTIC_MAP.pop(name, None)
                removed = True
            if name in cls._REGISTERED_GENERATION_MAP:
                cls._REGISTERED_GENERATION_MAP.pop(name, None)
                removed = True
            if not removed and not missing_ok:
                raise KeyError(f"Semantic axis '{name}' is not registered.")

    @classmethod
    def clear_registered_axes(cls) -> None:
        """Clear all custom semantic axis registrations."""
        with cls._REGISTRY_LOCK:
            cls._REGISTERED_SEMANTIC_MAP.clear()
            cls._REGISTERED_GENERATION_MAP.clear()

    @classmethod
    def get_registered_axes(cls) -> dict[str, dict[str, tp.Any]]:
        """Return a snapshot of globally registered custom axis mappings."""
        with cls._REGISTRY_LOCK:
            return {
                name: {
                    "axis_rule": cls._REGISTERED_SEMANTIC_MAP[name],
                    "generation_axis_rule": cls._REGISTERED_GENERATION_MAP.get(name, NOT_GIVEN),
                }
                for name in cls._REGISTERED_SEMANTIC_MAP
            }

    @classmethod
    def _lookup_semantic_mapping(cls, semantic_axis: str) -> tp.Any:
        """Lookup semantic mapping from custom registry, then built-ins."""
        if semantic_axis in cls._REGISTERED_SEMANTIC_MAP:
            return cls._REGISTERED_SEMANTIC_MAP[semantic_axis]
        return cls._SEMANTIC_MAP.get(semantic_axis)

    @classmethod
    def _lookup_generation_mapping(cls, semantic_axis: str) -> tp.Any:
        """Lookup generation mapping from custom registry."""
        return cls._REGISTERED_GENERATION_MAP.get(semantic_axis, NOT_GIVEN)

    def _resolve_axis_rule(self, axis_rule: tp.Any, _visited: set[str] | None = None) -> tp.Any:
        """Resolve rule references (attribute names or semantic aliases) to concrete axis rules."""
        if isinstance(axis_rule, list):
            return [
                self._resolve_axis_rule(
                    item,
                    _visited=set(_visited) if _visited is not None else None,
                )
                for item in axis_rule
            ]
        if isinstance(axis_rule, tuple):
            return tuple(
                self._resolve_axis_rule(
                    item,
                    _visited=set(_visited) if _visited is not None else None,
                )
                for item in axis_rule
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

    def __post_init__(self):
        """
        Post-initialization hook to resolve default axis values.

        If an axis attribute is set to NOT_GIVEN, its value is resolved based
        on default logic, typically using the standard mesh dimension names.
        """
        resolved_values = {}

        def resolve_field(name, default_logic):
            """Helper to resolve a single field's value if it's NOT_GIVEN."""
            current_value = getattr(self, name)
            if current_value is NOT_GIVEN:
                resolved_values[name] = default_logic()
            elif name not in resolved_values:
                resolved_values[name] = current_value

        def get_resolved(name):
            """Helper to get a field's value, prioritizing resolved values."""
            return resolved_values.get(name, getattr(self, name))

        resolve_field(
            "batch_axis",
            lambda: (self.fully_sharded_data_parallel_axis, self.data_parallel_axis),
        )
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

        for fld in dataclasses.fields(self):
            if fld.name not in resolved_values and fld.name not in [
                "_SEMANTIC_MAP",
                "_STANDARD_TO_GENERATION_ATTR_MAP",
            ]:
                resolved_values[fld.name] = getattr(self, fld.name)

        for name, value in resolved_values.items():
            object.__setattr__(self, name, value)

        self._safety_check()

    def _safety_check(self):
        """
        Checks if any axis attribute still has the NOT_GIVEN value after resolution.

        Raises:
            ValueError: If any attribute is still NOT_GIVEN, indicating a
                        configuration error.
        """
        for fld in dataclasses.fields(self):
            if fld.name not in ["_SEMANTIC_MAP", "_STANDARD_TO_GENERATION_ATTR_MAP"]:
                val = getattr(self, fld.name)
                if val == NOT_GIVEN:
                    raise ValueError(f"Partitioning rule `{fld.name}` was not resolved.")

    def resolve_axis(
        self,
        axes: tp.Sequence[str | None],
        mode: RUNTIME_MODE_TYPES,  # type:ignore
    ) -> list[str | None]:
        """
        Generates a Axis from a sequence of semantic axis names and a mode.

        Maps a sequence of semantic axis name strings (like BATCH, LENGTH) to the
        actual mesh axis names defined in this `PartitionAxis` instance, considering
        the current runtime mode (e.g., training vs. generation).

        Args:
            axes: A sequence of semantic axis name strings (e.g., [BATCH, LENGTH, HEAD])
                or None (or "_") for axes that shouldn't be sharded.
            mode: The current operational mode (e.g., MODE_TRAIN,
                MODE_DECODE) which determines if generation-specific
                rules should be applied.

        Returns:
            A instance representing the sharding for the given sequence of axes.

        Raises:
            ValueError: If an unknown semantic axis name is encountered or if
                a resolved axis rule is still NOT_GIVEN (should be caught
                by `_safety_check` but included for robustness).
            LookupError: If an internal attribute name derived from the semantic
                map isn't found in the instance (shouldn't happen with
                correct class definition).
        """
        resolved_rules: list[AxisType] = []

        for axis_name in axes:
            if axis_name is None or axis_name == "_":
                resolved_rules.append(None)
                continue

            # Composite axis rules can include direct mesh names and/or semantic names.
            if isinstance(axis_name, (list, tuple)):
                standard_rule = []
                for sub_axis in axis_name:
                    sub_mapped = self._lookup_semantic_mapping(sub_axis)
                    standard_rule.append(sub_axis if sub_mapped is None else sub_mapped)
            else:
                standard_rule = self._lookup_semantic_mapping(axis_name)
                if standard_rule is None:
                    raise ValueError(f"Unknown semantic axis name: '{axis_name}'")

            target_rule = standard_rule
            if mode in GENERATION_MODES:
                if isinstance(axis_name, (list, tuple)):
                    gen_composite = []
                    any_changed = False
                    for idx, sub_axis in enumerate(axis_name):
                        sub_standard = standard_rule[idx]
                        sub_target = sub_standard

                        sub_gen = self._lookup_generation_mapping(sub_axis)
                        if sub_gen is not NOT_GIVEN:
                            sub_target = sub_gen
                            any_changed = True
                        elif isinstance(sub_standard, str):
                            sub_gen_attr = self._STANDARD_TO_GENERATION_ATTR_MAP.get(sub_standard)
                            if sub_gen_attr and hasattr(self, sub_gen_attr):
                                sub_gen_val = getattr(self, sub_gen_attr)
                                if sub_gen_val is not None and sub_gen_val is not NOT_GIVEN:
                                    sub_target = sub_gen_attr
                                    any_changed = True

                        gen_composite.append(sub_target)
                    if any_changed:
                        target_rule = gen_composite
                else:
                    custom_gen_rule = self._lookup_generation_mapping(axis_name)
                    if custom_gen_rule is not NOT_GIVEN:
                        target_rule = custom_gen_rule
                    elif isinstance(standard_rule, str):
                        gen_attr_name = self._STANDARD_TO_GENERATION_ATTR_MAP.get(standard_rule)
                        if gen_attr_name and hasattr(self, gen_attr_name):
                            gen_val = getattr(self, gen_attr_name)
                            if gen_val is not None and gen_val is not NOT_GIVEN:
                                target_rule = gen_attr_name

            mesh_axis_rule = self._resolve_axis_rule(target_rule)

            if mesh_axis_rule is NOT_GIVEN:
                raise ValueError(f"Resolved axis rule for '{axis_name}' is still NOT_GIVEN.")

            resolved_rules.append(mesh_axis_rule)
        return resolved_rules

    def resolve_spec(
        self,
        axes: tp.Sequence[str | None],
        mode: RUNTIME_MODE_TYPES,  # type:ignore
    ) -> PartitionSpec:
        """
        Generates a PartitionSpec from a sequence of semantic axis names and a mode.

        Maps a sequence of semantic axis name strings (like BATCH, LENGTH) to the
        actual mesh axis names defined in this `PartitionAxis` instance, considering
        the current runtime mode (e.g., training vs. generation).

        Args:
            axes: A sequence of semantic axis name strings (e.g., [BATCH, LENGTH, HEAD])
                or None (or "_") for axes that shouldn't be sharded.
            mode: The current operational mode (e.g., MODE_TRAIN,
                MODE_DECODE) which determines if generation-specific
                rules should be applied.

        Returns:
            A jax.sharding.PartitionSpec instance representing the sharding
            for the given sequence of axes.

        Raises:
            ValueError: If an unknown semantic axis name is encountered or if
                a resolved axis rule is still NOT_GIVEN (should be caught
                by `_safety_check` but included for robustness).
            LookupError: If an internal attribute name derived from the semantic
                map isn't found in the instance (shouldn't happen with
                correct class definition).
        """
        return PartitionSpec(*self.resolve_axis(axes=axes, mode=mode))

    __hash__ = hash_fn


class PartitionManager(PyTree):
    """
    Context manager for applying sharding constraints using PartitionAxis.

    This class acts as a context manager (`with PartitionManager(...)`) to
    set a context-local variable (`_CURRENT_PARTITION_MANAGER`) that makes
    the current manager implicitly available via functions like
    `get_current_partition_manager()` or the static `shard()` method.

    Args:
        paxis: The PartitionAxis instance defining the sharding strategy
               to be used within this context.
    """

    paxis: PartitionAxis

    def __post_init__(self):
        global _LAST_PARTITION_MANAGER
        if not isinstance(self.paxis, PartitionAxis):
            raise TypeError(f"Expected PartitionAxis, got {type(self.paxis)}")
        _LAST_PARTITION_MANAGER = self

    def __enter__(self):
        global _LAST_PARTITION_MANAGER
        token = _CURRENT_PARTITION_MANAGER.set(self)
        object.__setattr__(self, "_context_token", token)
        _LAST_PARTITION_MANAGER = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        token = getattr(self, "_context_token", None)
        if token is not None:
            _CURRENT_PARTITION_MANAGER.reset(token)
            object.__setattr__(self, "_context_token", None)
        return False

    def shard(
        self,
        x: jax.Array,
        axes: tp.Sequence[str | None] = NOT_GIVEN,
        mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
        dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
        auto_correct: bool = True,
    ) -> jax.Array:
        """
        Applies sharding constraint to a JAX array using this manager's PartitionAxis.

        Uses this `PartitionManager` instance to resolve semantic axis names (`axes`)
        into a `PartitionSpec`, then applies the sharding constraint to `x`.

        Supports specifying axes and mode directly, or providing a `DynamicShardingAxes`
        named tuple. Can also infer the mode based on a dimension size if an integer
        `mode` is provided.

        Args:
            x: The JAX array to apply the sharding constraint to.
            axes: A sequence of semantic axis name strings or None. Required if
                `dynamic_axes` is NOT_GIVEN.
            mode: The runtime mode (string constant) or an integer representing
                the dimension index to check for mode inference. Required if
                `dynamic_axes` is NOT_GIVEN.
            dynamic_axes: An optional `DynamicShardingAxes` named tuple that
                provides both `axes` and `mode`. If provided, `axes` and
                `mode` arguments are ignored.
            auto_correct: If True, automatically corrects the resolved `PartitionSpec`
                based on array shape and mesh compatibility using
                `get_corrected_named_sharding`. Defaults to True.

        Returns:
            The array `x` with the sharding constraint applied.

        Raises:
            ValueError: If neither `axes`/`mode` nor `dynamic_axes` are provided.
            ValueError: Propagated from `PartitionAxis.resolve_spec` or if resolved
                axis rule is NOT_GIVEN.
        """

        spec = self.resolve(
            axes=axes,
            mode=mode,
            dynamic_axes=dynamic_axes,
            shape=x.shape,
        )

        if auto_correct:
            spec = get_corrected_named_sharding(x.shape, spec).spec

        return with_sharding_constraint(x, spec)

    def resolve(
        self,
        axes: tp.Sequence[str | None] | DynamicShardingAxes = NOT_GIVEN,
        mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
        dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
        shape: tp.Sequence[int] = NOT_GIVEN,
    ) -> PartitionSpec:
        """Resolve semantic axis names to a PartitionSpec.

        Converts semantic axis names (like BATCH, LENGTH, HEAD) into a
        concrete PartitionSpec using the configured PartitionAxis mapping.
        Supports dynamic mode detection based on array shape.

        Args:
            axes: Sequence of semantic axis names, or a DynamicShardingAxes
                tuple containing both axes and mode.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE) or an integer
                dimension index for dynamic mode detection. When an integer
                is provided, mode is inferred based on whether shape[mode] == 1.
            dynamic_axes: Alternative way to provide axes and mode together
                as a DynamicShardingAxes named tuple.
            shape: Array shape, required when mode is an integer for
                dynamic mode detection.

        Returns:
            A PartitionSpec mapping semantic axes to mesh dimensions.

        Raises:
            ValueError: If axes/mode are not provided and dynamic_axes is
                also not provided, or if shape is missing for dynamic mode.

        Example:
            >>> manager = PartitionManager(paxis=paxis)
            >>> # Direct specification
            >>> spec = manager.resolve([BATCH, LENGTH, HEAD], mode=MODE_TRAIN)
            >>> # Dynamic mode detection (decode if dim 1 has size 1)
            >>> spec = manager.resolve([BATCH, LENGTH, HEAD], mode=1, shape=x.shape)
        """
        if dynamic_axes is NOT_GIVEN and axes is not NOT_GIVEN:
            if isinstance(axes, tuple) and hasattr(axes, "_fields"):
                dynamic_axes = axes
                axes = NOT_GIVEN
            elif isinstance(axes, type) and issubclass(axes, tuple) and hasattr(axes, "_fields"):
                dynamic_axes = DynamicShardingAxes(axes=axes.axes, mode=axes.mode)
                axes = NOT_GIVEN

        if axes is NOT_GIVEN or mode is NOT_GIVEN:
            if dynamic_axes is NOT_GIVEN:
                raise ValueError("if axes or mode is empty you should provide dynamic axes")
            axes = dynamic_axes.axes
            mode = dynamic_axes.mode
        if isinstance(mode, int):
            if shape is NOT_GIVEN:
                raise ValueError("when using dynamic mode detection shape should be provided")
            mode = MODE_DECODE if shape[mode] == 1 else MODE_TRAIN
        return self.paxis.resolve_spec(axes, mode)

    def __str__(self):
        """String representation of the PartitionManager."""
        return "PartitionManager(...)"

    def __repr__(self):
        """Representation of the PartitionManager."""
        return "PartitionManager(...)"

    __hash__ = hash_fn


def get_current_partition_manager() -> PartitionManager | None:
    """Get the current context-local partition manager, if set."""
    return _CURRENT_PARTITION_MANAGER.get()


def get_partition_manager() -> PartitionManager | None:
    """Get the last created partition manager instance."""
    return _LAST_PARTITION_MANAGER


def apply_logical_sharding(
    x: jax.Array,
    partition_manager: PartitionManager | None = NOT_GIVEN,
    axes: tp.Sequence[str | None] = NOT_GIVEN,
    mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
    dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
    auto_correct: bool = True,
):
    """
    Applies logical sharding to a JAX array using an available PartitionManager.

    This function is a convenience wrapper around `PartitionManager.shard`.
    It attempts to find a `PartitionManager` from the current context first
    (`get_current_partition_manager`), and if none is found, it falls back
    to the last created manager (`get_partition_manager`).

    Args:
        x: The JAX array to apply sharding to.
        partition_manager: An explicit `PartitionManager` instance to use.
            If not provided, the function tries the current context manager
            first, then the last created manager.
        axes: A sequence of semantic axis name strings or None. Required if
              `dynamic_axes` is NOT_GIVEN and `partition_manager` is NOT_GIVEN.
        mode: The runtime mode or dimension index for inference. Required if
              `dynamic_axes` is NOT_GIVEN and `partition_manager` is NOT_GIVEN.
        dynamic_axes: An optional `DynamicShardingAxes` tuple. If provided,
                      `axes` and `mode` are ignored.
        auto_correct: If True, automatically corrects the resolved PartitionSpec.
                      Defaults to True.

    Returns:
        The JAX array with sharding constraints applied.

    Raises:
        ValueError: If neither `axes`/`mode` nor `dynamic_axes` are provided
                        when a manager is found or provided.
    """

    resolved_manager = partition_manager
    if resolved_manager is NOT_GIVEN or resolved_manager is None:
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
