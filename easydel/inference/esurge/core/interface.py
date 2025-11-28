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
import copy
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Self

from jax import numpy as jnp

from ..utils import cdiv, get_dtype_size

if TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig


@dataclass
class CacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    page_size: int

    @property
    def type_id(self) -> str:
        """
        The type identifier of this KV cache.
        Return different strings for layers with different KV cache type (e.g.,
        different number of tokens like full attention vs sliding window
        attention, different KV cache size per token like layers with different
        number of heads)

        Returns:
            The type identifier of this KV cache.
        """
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `page_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    def max_memory_usage_bytes(self, *args, **kwargs) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of CacheSpec objects into a single CacheSpec object.
        """
        assert all(spec.type_id == specs[0].type_id for spec in specs[1:]), (
            "All layers in the same KV cache group must share the same type_id."
        )
        return copy.deepcopy(specs[0])


@dataclass
class AttentionSpec(CacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: jnp.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        coef = 1 if self.use_mla else 2
        return coef * self.page_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)


@dataclass
class FullAttentionSpec(AttentionSpec):
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (pages are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    Default to None for not using sliding window attention.
    """

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len, **kwargs) -> int:
        return cdiv(max_model_len, self.page_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError("All attention layers in the same KV cache group must have the same window size.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        merged_spec = super().merge(specs)
        sliding_window = set(spec.sliding_window for spec in specs if spec.sliding_window is not None)
        attention_chunk_size = set(spec.attention_chunk_size for spec in specs if spec.attention_chunk_size is not None)

        merged_spec.sliding_window = cls.merge_window_sizes(sliding_window)
        merged_spec.attention_chunk_size = cls.merge_window_sizes(attention_chunk_size)
        assert (merged_spec.sliding_window is not None) + (merged_spec.attention_chunk_size is not None) <= 1, (
            "Model with both sliding window layers and chunked local attention layers is not supported."
        )
        return merged_spec


@dataclass
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    @property
    def type_id(self) -> str:
        return f"local_attention_{self.attention_chunk_size}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len, max_num_batched_tokens, **kwargs) -> int:
        num_tokens = min(self.attention_chunk_size + max_num_batched_tokens, max_model_len)

        return cdiv(num_tokens, self.page_size) * self.page_size_bytes


@dataclass
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def __post_init__(self):
        assert not self.use_mla, "MLA is not supported for sliding window"

    @property
    def type_id(self) -> str:
        return f"sliding_window_{self.sliding_window}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len, max_num_batched_tokens, **kwargs) -> int:
        num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens, max_model_len)

        return (cdiv(num_tokens, self.page_size) + 1) * self.page_size_bytes


@dataclass
class MambaSpec(CacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtype: jnp.dtype
    page_size_padded: int | None = None

    def __post_init__(self):
        self.num_elements = sum(prod(shape) for shape in self.shapes)

    @property
    def type_id(self) -> str:
        return f"mamba_{self.shapes}_{self.dtype}"

    @property
    def page_size_bytes(self) -> int:
        page_size = self.num_elements * get_dtype_size(self.dtype)
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, **kwargs) -> int:
        return self.page_size_bytes


@dataclass
class CacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache page table.
    These layers are regarded as one layer in the KV cache manager.
    """

    kv_cache_spec: CacheSpec

    layer_names: list[str] | None = None


@dataclass
class CacheGroupsConfig:
    """
    The KV cache configuration of a model.
    """

    num_pages: int
    kv_cache_groups: list[CacheGroupSpec]


def create_kv_cache_specs_from_config(
    config: "EasyDeLBaseConfig",
    page_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: jnp.dtype,
    use_mla: bool = False,
) -> list[CacheGroupSpec]:
    """Convert model config's get_mask_details() to CacheGroupSpec list.

    This function reads the attention mask details from the model configuration
    and creates appropriate cache specifications for each attention type.
    Layers with the same attention type are grouped together.

    Args:
        config: Model configuration with get_mask_details() method.
        page_size: Number of tokens per cache page.
        num_kv_heads: Number of key-value attention heads.
        head_size: Dimension of each attention head.
        dtype: Data type for cache tensors.
        use_mla: Whether to use Multi-head Latent Attention.

    Returns:
        List of CacheGroupSpec, one per attention type found in the config.
        Falls back to a single FullAttentionSpec if no mask details available.
    """
    from easydel.infra.utils import AttnMaskType

    mask_details = config.get_mask_details() if hasattr(config, "get_mask_details") else None

    if not mask_details:
        return [
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                    use_mla=use_mla,
                ),
                layer_names=None,
            )
        ]

    groups: dict[AttnMaskType, list[tuple[int, int | None, int | None]]] = defaultdict(list)
    for layer_idx, detail in mask_details.items():
        groups[detail.mask_type].append((layer_idx, detail.size, detail.chunks))

    specs: list[CacheGroupSpec] = []

    for mask_type, layers in groups.items():
        layer_names = [f"layer.{idx}" for idx, _, _ in layers]

        if mask_type == AttnMaskType.FULL:
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=FullAttentionSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=use_mla,
                    ),
                    layer_names=layer_names,
                )
            )
        elif mask_type == AttnMaskType.SLIDING:
            sliding_window = layers[0][1]
            if sliding_window is None:
                raise ValueError(f"Sliding window size is required for sliding attention layers: {layer_names}")
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=SlidingWindowSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=False,  # MLA is not supported for sliding window
                        sliding_window=sliding_window,
                    ),
                    layer_names=layer_names,
                )
            )
        elif mask_type == AttnMaskType.CHUNK:
            chunk_size = layers[0][2]
            if chunk_size is None:
                raise ValueError(f"Chunk size is required for chunked attention layers: {layer_names}")
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=ChunkedLocalAttentionSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=use_mla,
                        attention_chunk_size=chunk_size,
                    ),
                    layer_names=layer_names,
                )
            )
        else:
            raise ValueError(f"Unknown attention mask type: {mask_type}")

    return (
        specs
        if specs
        else [
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                    use_mla=use_mla,
                ),
                layer_names=None,
            )
        ]
    )
