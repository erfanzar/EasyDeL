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

"""Operation cache requirements mixin for EasyDeL modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from easydel.layers.operations import OperationRegistry
from easydel.layers.operations.requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "LayerOperationInfo",
    "OperationCacheMixin",
    "OperationsCacheInfo",
]


@dataclass
class LayerOperationInfo:
    """Information about a single layer's operation and cache requirements.

    Attributes:
        layer_index: Index of the layer in the model.
        slot: Unique slot identifier for this operation within the layer.
            Most models have a single cache-bearing operation per layer (slot=None).
            Some architectures can have multiple cache-bearing operations in the same
            decoder layer (e.g., parallel attention + SSM), in which case different
            slots distinguish them.
        layer_type: Type of layer (e.g., "full_attention", "linear_attention").
        operation_name: Name of the prefill operation implementation.
        decode_operation_name: Name of the decode operation (if different from prefill).
        requirements: Operation requirements (metadata and cache).
        supported_cache_types: Cache types supported by this operation.
        requires_kv_cache: Whether this layer requires KV cache.
        requires_state_cache: Whether this layer requires recurrent state cache.
    """

    layer_index: int
    slot: str | None
    layer_type: str
    operation_name: str
    requirements: OperationRequirements
    supported_cache_types: CacheType
    decode_operation_name: str | None = None
    requires_kv_cache: bool = True
    requires_state_cache: bool = False

    @property
    def is_attention_layer(self) -> bool:
        """Check if this is an attention-based layer.

        Determined dynamically from the operation's supported cache types.
        An operation is primarily attention-based if it supports TRANSFORMER or RAGGED_PAGES cache.
        """
        return (
            CacheType.TRANSFORMER in self.supported_cache_types or CacheType.RAGGED_PAGES in self.supported_cache_types
        )

    @property
    def is_recurrent_layer(self) -> bool:
        """Check if this is a recurrent/linear attention layer.

        Determined dynamically from the operation's supported cache types.
        An operation is primarily recurrent if it supports RECURRENT cache but NOT standard
        attention caches (TRANSFORMER/RAGGED_PAGES). This distinguishes true recurrent
        operations from cache-agnostic operations like vanilla attention.
        """
        supports_recurrent = CacheType.RECURRENT in self.supported_cache_types
        supports_attention = (
            CacheType.TRANSFORMER in self.supported_cache_types or CacheType.RAGGED_PAGES in self.supported_cache_types
        )
        # True recurrent operations support RECURRENT but not standard attention caches
        return supports_recurrent and not supports_attention

    @property
    def has_separate_decode(self) -> bool:
        """Whether this layer uses different operations for prefill and decode."""
        return self.decode_operation_name is not None and self.decode_operation_name != self.operation_name


@dataclass
class OperationsCacheInfo:
    """Complete information about all operations' cache requirements.

    Attributes:
        layers: List of layer operation info.
        prefill_operation: Operation used for prefill phase.
        decode_operation: Operation used for decode phase.
        combined_cache_types: Cache types that work for all operations (intersection).
        combined_metadata: Combined metadata requirements.
        is_hybrid_model: Whether the model has mixed layer types (attention + recurrent).
        supports_ragged_pages: Whether all operations support RaggedPagesCache.
        supports_transformer_cache: Whether all operations support TransformerCache.
        requires_hybrid_cache: Whether the model needs HybridCache (has recurrent layers).
        requires_state_management: Whether any layer needs recurrent state.
        has_separate_decode_ops: Whether any layer uses different decode operation.
    """

    layers: list[LayerOperationInfo] = field(default_factory=list)
    prefill_operation: str = ""
    decode_operation: str = ""
    combined_cache_types: CacheType = field(default_factory=CacheType.any)
    combined_metadata: MetadataField = field(default_factory=MetadataField.basic)
    is_hybrid_model: bool = False
    supports_ragged_pages: bool = False
    supports_transformer_cache: bool = False
    requires_hybrid_cache: bool = False
    requires_state_management: bool = False
    has_separate_decode_ops: bool = False

    @property
    def requires_ragged_pages(self) -> bool:
        """Deprecated: Use supports_ragged_pages instead."""
        return self.supports_ragged_pages

    @property
    def requires_transformer_cache(self) -> bool:
        """Deprecated: Use supports_transformer_cache instead."""
        return self.supports_transformer_cache

    @property
    def num_attention_layers(self) -> int:
        """Count of attention-based layers."""
        return sum(1 for layer in self.layers if layer.is_attention_layer)

    @property
    def num_recurrent_layers(self) -> int:
        """Count of recurrent/linear attention layers."""
        return sum(1 for layer in self.layers if layer.is_recurrent_layer)

    @property
    def attention_ratio(self) -> float:
        """Ratio of attention layers to total layers."""
        if not self.layers:
            return 1.0
        return self.num_attention_layers / len(self.layers)

    def get_recommended_cache_type(self) -> str:
        """Get recommended cache type based on model requirements.

        Priority order:
        1. "hybrid" - if model has recurrent layers requiring state management
        2. "transformer" - if supported (simplest, most compatible)
        3. "ragged" - if transformer not supported but ragged is

        Returns:
            "hybrid", "transformer", or "ragged" based on model requirements.
        """
        if self.requires_hybrid_cache or self.requires_state_management:
            return "hybrid"

        if self.supports_transformer_cache:
            return "transformer"

        if self.supports_ragged_pages:
            return "ragged"

        return "transformer"

    def get_layer_by_index(self, index: int) -> LayerOperationInfo | None:
        """Get layer info by index."""
        for layer in self.layers:
            if layer.layer_index == index:
                return layer
        return None

    def get_layers_by_index(self, index: int) -> list[LayerOperationInfo]:
        """Get all layer infos for an index (slot-aware)."""
        return [layer for layer in self.layers if layer.layer_index == index]


class OperationCacheMixin:
    """Mixin that provides operation cache requirements discovery.

    This mixin adds methods to EasyDeLBaseModule for discovering
    what operations are used and what cache types they require.

    The recommended approach is to use dynamic discovery (default),
    which traverses the actual module graph to find operations.
    """

    def _get_operation_requirements(
        self,
        name: str | None,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements | None:
        """Get requirements for an operation by name from the registry.

        Args:
            name (str | None): Operation name to look up. If None, returns None immediately.
            mode (ExecutionMode): The execution mode to get requirements for. Can be
                PREFILL, DECODE, or MIXED. Defaults to ExecutionMode.MIXED.

        Returns:
            OperationRequirements | None: Requirements for the operation if found in
                the registry, None otherwise.
        """
        if name is None:
            return None
        try:
            op_class = OperationRegistry.get(name.lower())
            if op_class is not None:
                return op_class.get_requirements(mode)
        except ValueError:
            pass
        return None

    def get_operations_cache_info(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
        dynamic: bool = True,
    ) -> OperationsCacheInfo:
        """Get complete information about operations and their cache requirements.

        Args:
            mode: Execution mode for requirements lookup.
            dynamic: If True, discover from actual module graph (recommended).
                     If False, read from config (faster but less accurate).

        Returns:
            OperationsCacheInfo with complete cache requirements information.

        Example:
            >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(...)
            >>> cache_info = model.get_operations_cache_info()
            >>> print(f"Recommended cache: {cache_info.get_recommended_cache_type()}")
            >>> for layer in cache_info.layers:
            ...     print(f"Layer {layer.layer_index}: {layer.operation_name}")
        """
        if dynamic:
            info = self.get_operations_cache_info_dynamic(mode)
            if info.layers:
                return info
            # Fallback: some models (e.g., pure recurrent/state-space) may not expose
            # operations in the module graph, so we attempt a config-based inference.
            return self._get_operations_cache_info_from_config(mode)
        return self._get_operations_cache_info_from_config(mode)

    def _get_operations_cache_info_from_config(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationsCacheInfo:
        """Get cache info by reading from model configuration (static fallback).

        This is a fallback method that reads from config attributes.
        Use dynamic=True (default) for more accurate discovery.

        Note: This method queries the OperationRegistry to get requirements,
        so it will work with any registered operation without hardcoding.

        Args:
            mode: Execution mode for requirements lookup.

        Returns:
            OperationsCacheInfo with cache requirements information.
        """
        config = getattr(self, "config", None)
        if config is None:
            return OperationsCacheInfo()

        attn_mechanism = getattr(config, "attn_mechanism", None)
        decode_attn_mechanism = getattr(config, "decode_attn_mechanism", None)

        if attn_mechanism is not None and hasattr(attn_mechanism, "value"):
            attn_mechanism = str(attn_mechanism.value)
        if decode_attn_mechanism is not None and hasattr(decode_attn_mechanism, "value"):
            decode_attn_mechanism = str(decode_attn_mechanism.value)

        prefill_op = attn_mechanism or "vanilla"
        decode_op = decode_attn_mechanism or prefill_op

        layer_types = getattr(config, "layer_types", None)
        num_hidden_layers = getattr(config, "num_hidden_layers", 1)

        layers: list[LayerOperationInfo] = []
        combined_cache = CacheType.any()
        combined_metadata = MetadataField.NONE
        has_attention = False
        has_recurrent = False

        if layer_types is not None:
            for idx, layer_type in enumerate(layer_types):
                prefill_reqs = self._get_operation_requirements(layer_type, ExecutionMode.PREFILL)
                op_name = layer_type if prefill_reqs is not None else prefill_op

                if prefill_reqs is None:
                    prefill_reqs = self._get_operation_requirements(prefill_op, ExecutionMode.PREFILL)
                if prefill_reqs is None:
                    prefill_reqs = OperationRequirements.default(op_name)

                layer_decode_op = decode_op if decode_op != prefill_op else op_name
                decode_reqs = self._get_operation_requirements(layer_decode_op, ExecutionMode.DECODE)
                if decode_reqs is None:
                    decode_reqs = OperationRequirements.default(layer_decode_op)

                reqs = prefill_reqs | decode_reqs

                prefill_supports_attention = (
                    CacheType.TRANSFORMER in prefill_reqs.cache.supported
                    or CacheType.RAGGED_PAGES in prefill_reqs.cache.supported
                )
                decode_supports_attention = (
                    CacheType.TRANSFORMER in decode_reqs.cache.supported
                    or CacheType.RAGGED_PAGES in decode_reqs.cache.supported
                )
                prefill_supports_recurrent = CacheType.RECURRENT in prefill_reqs.cache.supported
                decode_supports_recurrent = CacheType.RECURRENT in decode_reqs.cache.supported
                prefill_is_recurrent = prefill_supports_recurrent and not prefill_supports_attention
                decode_is_recurrent = decode_supports_recurrent and not decode_supports_attention

                layer_info = LayerOperationInfo(
                    layer_index=idx,
                    slot=None,
                    layer_type=layer_type,
                    operation_name=op_name,
                    decode_operation_name=layer_decode_op if layer_decode_op != op_name else None,
                    requirements=reqs,
                    supported_cache_types=reqs.cache.supported,
                    requires_kv_cache=(prefill_reqs.cache.requires_cache and prefill_supports_attention)
                    or (decode_reqs.cache.requires_cache and decode_supports_attention),
                    requires_state_cache=prefill_is_recurrent or decode_is_recurrent,
                )
                layers.append(layer_info)

                combined_cache &= reqs.cache.supported
                combined_metadata |= reqs.metadata.required

                if prefill_supports_attention or decode_supports_attention:
                    has_attention = True
                if prefill_is_recurrent or decode_is_recurrent:
                    has_recurrent = True
        else:
            prefill_reqs = self._get_operation_requirements(prefill_op, ExecutionMode.PREFILL)
            decode_reqs = self._get_operation_requirements(decode_op, ExecutionMode.DECODE)

            if prefill_reqs is None:
                prefill_reqs = OperationRequirements.default(prefill_op)
            if decode_reqs is None:
                decode_reqs = OperationRequirements.default(decode_op)

            reqs = prefill_reqs | decode_reqs

            prefill_supports_attention = (
                CacheType.TRANSFORMER in prefill_reqs.cache.supported
                or CacheType.RAGGED_PAGES in prefill_reqs.cache.supported
            )
            decode_supports_attention = (
                CacheType.TRANSFORMER in decode_reqs.cache.supported
                or CacheType.RAGGED_PAGES in decode_reqs.cache.supported
            )
            prefill_supports_recurrent = CacheType.RECURRENT in prefill_reqs.cache.supported
            decode_supports_recurrent = CacheType.RECURRENT in decode_reqs.cache.supported
            prefill_is_recurrent = prefill_supports_recurrent and not prefill_supports_attention
            decode_is_recurrent = decode_supports_recurrent and not decode_supports_attention

            for idx in range(num_hidden_layers):
                layer_info = LayerOperationInfo(
                    layer_index=idx,
                    slot=None,
                    layer_type="attention" if (prefill_supports_attention or decode_supports_attention) else "recurrent",
                    operation_name=prefill_op,
                    decode_operation_name=decode_op if decode_op != prefill_op else None,
                    requirements=reqs,
                    supported_cache_types=reqs.cache.supported,
                    requires_kv_cache=(prefill_reqs.cache.requires_cache and prefill_supports_attention)
                    or (decode_reqs.cache.requires_cache and decode_supports_attention),
                    requires_state_cache=prefill_is_recurrent or decode_is_recurrent,
                )
                layers.append(layer_info)

            combined_cache = reqs.cache.supported
            combined_metadata = reqs.metadata.required
            has_attention = prefill_supports_attention or decode_supports_attention
            has_recurrent = prefill_is_recurrent or decode_is_recurrent

        supports_ragged = CacheType.RAGGED_PAGES in combined_cache
        supports_transformer = CacheType.TRANSFORMER in combined_cache
        needs_hybrid = has_attention and has_recurrent

        return OperationsCacheInfo(
            layers=layers,
            prefill_operation=prefill_op,
            decode_operation=decode_op,
            combined_cache_types=combined_cache,
            combined_metadata=combined_metadata,
            is_hybrid_model=has_attention and has_recurrent,
            supports_ragged_pages=supports_ragged,
            supports_transformer_cache=supports_transformer,
            requires_hybrid_cache=needs_hybrid,
            requires_state_management=has_recurrent,
            has_separate_decode_ops=decode_op != prefill_op,
        )

    def get_layer_cache_requirements(self, layer_index: int) -> LayerOperationInfo | None:
        """Get cache requirements for a specific layer.

        Args:
            layer_index: Index of the layer.

        Returns:
            LayerOperationInfo for the layer, or None if not found.
        """
        cache_info = self.get_operations_cache_info()
        return cache_info.get_layer_by_index(layer_index)

    def get_operations_cache_info_dynamic(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationsCacheInfo:
        """Discover operations dynamically from the module graph.

        This method traverses the actual model structure to find all
        operation instances, including:
        1. FlexibleAttentionModule instances (which contain impl and impl_decode)
        2. BaseOperation instances stored as attributes on any nn.Module

        This comprehensive approach works because:
        - Standard attention uses FlexibleAttentionModule via attention_performer
        - Some models (like Qwen3Next) use operations directly (e.g., gdr_op)
        - Searching both patterns catches all cases for hybrid models

        Args:
            mode: Execution mode for requirements lookup.

        Returns:
            OperationsCacheInfo with complete cache requirements information.
        """
        from flax import nnx as nn

        from easydel.layers.attention import FlexibleAttentionModule
        from easydel.layers.operations._base_operation import BaseOperation
        from easydel.utils.traversals import iter_module_search

        layers: list[LayerOperationInfo] = []
        # Deduplicate by "slot" rather than by layer index so a single layer
        # can expose multiple cache-bearing operations (e.g., FalconH1 parallel hybrid).
        seen_slots: set[tuple[int, tuple]] = set()
        seen_op_ids: set[int] = set()
        has_separate_decode = False

        # Phase 1: Find all FlexibleAttentionModule instances
        for path, flex_attn in iter_module_search(self, FlexibleAttentionModule):
            layer_idx = self._extract_layer_index_from_path(path)
            slot_key = tuple(path)

            if (layer_idx, slot_key) in seen_slots:
                continue
            seen_slots.add((layer_idx, slot_key))

            executor = flex_attn.operation_executor

            if executor.has_separate_decode:
                has_separate_decode = True

            op = executor.get_operation(mode)
            if op is not None:
                seen_op_ids.add(id(op))
                reqs = executor.get_combined_requirements()
                op_name = op.get_impl_name()
                if isinstance(op_name, tuple):
                    op_name = op_name[0]

                decode_op = executor.decode_operation
                decode_op_name = None
                if decode_op is not None and executor.has_separate_decode:
                    decode_name = decode_op.get_impl_name()
                    decode_op_name = decode_name[0] if isinstance(decode_name, tuple) else decode_name

                supports_attention = (
                    CacheType.TRANSFORMER in reqs.cache.supported or CacheType.RAGGED_PAGES in reqs.cache.supported
                )
                supports_recurrent = CacheType.RECURRENT in reqs.cache.supported
                is_recurrent = supports_recurrent and not supports_attention

                layer_info = LayerOperationInfo(
                    layer_index=layer_idx,
                    slot=".".join(map(str, path)),
                    layer_type=self._infer_layer_type_from_path(path),
                    operation_name=op_name,
                    decode_operation_name=decode_op_name,
                    requirements=reqs,
                    supported_cache_types=reqs.cache.supported,
                    requires_kv_cache=reqs.cache.requires_cache and supports_attention,
                    requires_state_cache=is_recurrent,
                )
                layers.append(layer_info)

        # Phase 2: Find BaseOperation instances stored directly on modules
        # This catches cases like Qwen3NextLinearAttention which stores GatedDeltaRuleOp as gdr_op
        for path, module in iter_module_search(self, nn.Module):
            # Skip FlexibleAttentionModule - already handled in Phase 1
            if isinstance(module, FlexibleAttentionModule):
                continue

            layer_idx = self._extract_layer_index_from_path(path)

            for _attr_name, attr_value in vars(module).items():
                if attr_value is None:
                    continue

                if isinstance(attr_value, BaseOperation):
                    if id(attr_value) in seen_op_ids:
                        continue
                    slot_key = (*tuple(path), _attr_name)
                    if (layer_idx, slot_key) in seen_slots:
                        continue
                    seen_slots.add((layer_idx, slot_key))
                    seen_op_ids.add(id(attr_value))

                    op = attr_value
                    reqs = op.get_requirements(mode)
                    op_name = op.get_impl_name()
                    if isinstance(op_name, tuple):
                        op_name = op_name[0]

                    supports_attention = (
                        CacheType.TRANSFORMER in reqs.cache.supported or CacheType.RAGGED_PAGES in reqs.cache.supported
                    )
                    supports_recurrent = CacheType.RECURRENT in reqs.cache.supported
                    is_recurrent = supports_recurrent and not supports_attention

                    layer_type = self._infer_layer_type_from_path(path)
                    if is_recurrent:
                        layer_type = "linear_attention"

                    layer_info = LayerOperationInfo(
                        layer_index=layer_idx,
                        slot=".".join(map(str, slot_key)),
                        layer_type=layer_type,
                        operation_name=op_name,
                        decode_operation_name=None,
                        requirements=reqs,
                        supported_cache_types=reqs.cache.supported,
                        requires_kv_cache=reqs.cache.requires_cache and supports_attention,
                        requires_state_cache=is_recurrent,
                    )
                    layers.append(layer_info)
                    break

        # Normalize slot labeling: only keep explicit slot identifiers for layers that
        # expose multiple cache-bearing operations. For the common case (one cache-bearing
        # op per layer), keep slot=None for backward compatibility and readability.
        cache_slot_counts: dict[int, int] = {}
        for layer in layers:
            if layer.requirements.cache.cache_view_class is None:
                continue
            cache_slot_counts[layer.layer_index] = cache_slot_counts.get(layer.layer_index, 0) + 1

        for layer in layers:
            if cache_slot_counts.get(layer.layer_index, 0) <= 1:
                layer.slot = None

        layers.sort(key=lambda x: (x.layer_index, x.slot or "", x.operation_name))

        return self._build_cache_info_from_layers(layers, has_separate_decode)

    def _extract_layer_index_from_path(self, path: tuple) -> int:
        """Extract layer index from module path.

        Path format from nn.graph.iter_graph is a tuple of mixed str/int elements.
        Examples:
        - ("model", "layers", 5, "self_attn") -> 5
        - ("transformer", "h", 3, "attn") -> 3
        - ("decoder", "block", 0, "layer", 0, "SelfAttention") -> 0

        Args:
            path: Module path tuple.

        Returns:
            Layer index, or 0 if not found.
        """
        layer_containers = {
            "layers",
            "h",
            "block",
            "blocks",
            "decoder_layers",
        }

        for i, elem in enumerate(path):
            if isinstance(elem, str) and elem.lower() in layer_containers:
                if i + 1 < len(path) and isinstance(path[i + 1], int):
                    return path[i + 1]

        for elem in path:
            if isinstance(elem, int):
                return elem

        return 0

    def _infer_layer_type_from_path(self, path: tuple) -> str:
        """Infer layer type from the module path.

        Args:
            path: Module path tuple.

        Returns:
            Layer type string.
        """
        path_str = ".".join(map(str, path)).lower()

        # Check for specific attention types in path
        if "cross_attn" in path_str:
            return "cross_attention"
        if "vision" in path_str:
            return "vision_attention"

        # Default to full attention (most common)
        return "full_attention"

    def _build_cache_info_from_layers(
        self,
        layers: list[LayerOperationInfo],
        has_separate_decode: bool = False,
    ) -> OperationsCacheInfo:
        """Build OperationsCacheInfo from discovered layers.

        Args:
            layers: List of LayerOperationInfo.
            has_separate_decode: Whether any layer has separate decode operation.

        Returns:
            OperationsCacheInfo with combined requirements.
        """
        if not layers:
            return OperationsCacheInfo()

        # Compute combined requirements (intersection of all layers' supported cache types)
        combined_cache = CacheType.any()
        combined_metadata = MetadataField.NONE
        has_attention = False
        has_recurrent = False

        for layer in layers:
            combined_cache &= layer.supported_cache_types
            combined_metadata |= layer.requirements.metadata.required

            if layer.is_attention_layer:
                has_attention = True
            if layer.is_recurrent_layer:
                has_recurrent = True

        # Get prefill/decode ops from first layer
        prefill_op = layers[0].operation_name if layers else ""
        decode_op = layers[0].decode_operation_name or prefill_op if layers else ""

        # Check which cache types are supported by ALL operations (intersection)
        supports_ragged = CacheType.RAGGED_PAGES in combined_cache
        supports_transformer = CacheType.TRANSFORMER in combined_cache

        return OperationsCacheInfo(
            layers=layers,
            prefill_operation=prefill_op,
            decode_operation=decode_op,
            combined_cache_types=combined_cache,
            combined_metadata=combined_metadata,
            is_hybrid_model=has_attention and has_recurrent,
            supports_ragged_pages=supports_ragged,
            supports_transformer_cache=supports_transformer,
            requires_hybrid_cache=has_attention and has_recurrent,
            requires_state_management=has_recurrent,
            has_separate_decode_ops=has_separate_decode,
        )

    def get_required_cache_class(self) -> type:
        """Get the required cache class based on operation requirements.

        Dynamically discovers the cache view classes required by the model's
        operations and returns the appropriate cache class.

        For hybrid models (mixing attention and recurrent), returns HybridCache.
        For pure attention models, returns TransformerCache.
        For pure recurrent models, returns RecurrentCache.

        Returns:
            The cache class appropriate for the model's operations.

        Example:
            >>> model = LlamaForCausalLM(config)
            >>> cache_class = model.get_required_cache_class()
            >>> # cache_class is TransformerCache for standard attention models
        """
        cache_info = self.get_operations_cache_info()
        cache_view_classes: set[type] = set()

        for layer in cache_info.layers:
            cache_view_class = layer.requirements.cache.cache_view_class
            if cache_view_class is not None:
                cache_view_classes.add(cache_view_class)

        from easydel.layers.caching import (
            HybridCache,
            RecurrentCache,
            RecurrentCacheView,
            TransformerCache,
            TransformerCacheView,
        )

        has_transformer = any(issubclass(cls, TransformerCacheView) for cls in cache_view_classes if cls is not None)
        has_recurrent = any(issubclass(cls, RecurrentCacheView) for cls in cache_view_classes if cls is not None)

        if not has_transformer and has_recurrent:
            return RecurrentCache
        elif has_transformer and not has_recurrent:
            return TransformerCache
        else:
            return HybridCache

    def get_operations_cache_view(self) -> dict[int, type]:
        """Get the cache view class required for each layer.

        Returns a mapping from layer index to the cache view class that
        layer's operation requires.

        Notes:
            Most models have exactly one cache-bearing operation per layer.
            Some models can have multiple cache-bearing operations in the same
            layer (e.g., parallel attention + SSM). In that case, this method
            returns a composite cache view class for that layer.

        Returns:
            Dict mapping layer_index -> cache_view_class.

        Example:
            >>> model = Qwen3NextForCausalLM(config)  # Hybrid model
            >>> cache_views = model.get_operations_cache_view()
            >>> # {0: TransformerCacheView, 1: RecurrentCacheView, 2: TransformerCacheView, ...}
        """
        cache_info = self.get_operations_cache_info()
        per_layer: dict[int, set[type]] = {}

        for op_info in cache_info.layers:
            cache_view_class = op_info.requirements.cache.cache_view_class
            if cache_view_class is None:
                continue
            per_layer.setdefault(op_info.layer_index, set()).add(cache_view_class)

        # Resolve to a single view class per layer (backward compatible).
        result: dict[int, type] = {}
        if not per_layer:
            return result

        from easydel.layers.caching import ParallelHybridCacheView, RecurrentCacheView, TransformerCacheView

        for layer_idx, view_classes in per_layer.items():
            if len(view_classes) == 1:
                result[layer_idx] = next(iter(view_classes))
                continue

            # Parallel hybrid (e.g., FalconH1): attention KV + recurrent/SSM state
            if view_classes == {TransformerCacheView, RecurrentCacheView}:
                result[layer_idx] = ParallelHybridCacheView
                continue

            raise ValueError(
                f"Layer {layer_idx} requires multiple cache view classes: {view_classes}. "
                "Use a composite cache view or extend resolution logic."
            )

        return result

    def get_operations_cache_info_by_slot(self) -> dict[int, dict[int, LayerOperationInfo]]:
        """Get per-layer cache requirements grouped by slot.

        Most models have exactly one cache-bearing operation per layer, in which case
        each layer maps to a single slot (slot 0). Models that run multiple cache-bearing
        blocks inside a single decoder layer (e.g., attention + SSM in parallel) will
        expose multiple slots for that layer.

        Returns:
            Dict mapping layer_index -> {slot_index -> LayerOperationInfo}.
        """
        cache_info = self.get_operations_cache_info()
        per_layer: dict[int, list[LayerOperationInfo]] = {}
        for layer in cache_info.layers:
            if layer.requirements.cache.cache_view_class is None:
                continue
            per_layer.setdefault(layer.layer_index, []).append(layer)

        result: dict[int, dict[int, LayerOperationInfo]] = {}
        for layer_idx, layer_infos in per_layer.items():
            layer_infos_sorted = sorted(layer_infos, key=lambda info: (info.slot or "", info.operation_name))
            result[layer_idx] = {slot_idx: info for slot_idx, info in enumerate(layer_infos_sorted)}

        return result

    def get_operations_cache_view_by_slot(self) -> dict[int, dict[int, type]]:
        """Get the cache view classes per layer, grouped by slot.

        This is the slot-aware form of :meth:`get_operations_cache_view`.
        It is useful for architectures where a single decoder layer contains multiple
        cache-bearing blocks and you want to reason about them separately.

        Returns:
            Dict mapping layer_index -> {slot_index -> cache_view_class}.
        """
        info_by_slot = self.get_operations_cache_info_by_slot()
        result: dict[int, dict[int, type]] = {}

        for layer_idx, slots in info_by_slot.items():
            slot_views: dict[int, type] = {}
            for slot_idx, op_info in slots.items():
                view_cls = op_info.requirements.cache.cache_view_class
                if view_cls is None:
                    continue
                slot_views[slot_idx] = view_cls
            if slot_views:
                result[layer_idx] = slot_views

        return result

    def get_unique_cache_view_classes(self) -> set[type]:
        """Get all unique cache view classes used by this model.

        Returns the set of all cache view classes declared by the model's operations.
        For pure attention models, this will be {TransformerCacheView}.
        For hybrid models, this could be {TransformerCacheView, RecurrentCacheView}.

        Returns:
            Set of cache view classes used by the model.
        """
        cache_info = self.get_operations_cache_info()
        cache_view_classes: set[type] = set()

        for layer in cache_info.layers:
            cache_view_class = layer.requirements.cache.cache_view_class
            if cache_view_class is not None:
                cache_view_classes.add(cache_view_class)

        return cache_view_classes
