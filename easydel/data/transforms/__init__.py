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

"""Transform DSL for data manipulation.

This module provides a comprehensive transform system for data pipelines:
- Chainable transforms with >> operator
- Map and filter operations
- Field manipulation (rename, select, drop, extract, combine)
- Chat template application for conversational data
"""

from .base import ChainedTransform, ExpandTransform, Transform
from .chat_template import (
    DEFAULT_ROLE_MAPPING,
    ChatTemplateTransform,
    ConvertInputOutputToChatML,
    ConvertToChatML,
    MaybeApplyChatTemplate,
)
from .field_ops import (
    AddField,
    CombineFields,
    DropFields,
    ExtractField,
    RenameFields,
    SelectFields,
)
from .filter_ops import FilterByField, FilterNonEmpty, FilterTransform
from .map_ops import MapField, MapTransform
from .mixture import (
    MixedShardedSource,
    MixedShardState,
    MixStage,
    WeightScheduler,
    block_mixture_interleave,
)
from .pack import (
    FirstFitPacker,
    GreedyPacker,
    PackedSequence,
    PackedShardedSource,
    PackStage,
    PoolPacker,
    pack_constant_length,
    pack_pre_tokenized,
)
from .source import TransformedShardedSource
from .tokenize import (
    TokenizedShardedSource,
    TokenizerManager,
    TokenizeStage,
    batched_tokenize_iterator,
    compute_tokenizer_hash,
    tokenize_dataset_config,
)

# Trainer-specific transforms are now in easydel.trainers.transforms
# Use lazy imports for backwards compatibility to avoid circular imports
_TRAINER_TRANSFORMS = {
    "BCOPreprocessTransform",
    "CPOPreprocessTransform",
    "DPOPreprocessTransform",
    "GRPOPreprocessTransform",
    "KTOPreprocessTransform",
    "ORPOPreprocessTransform",
    "RewardPreprocessTransform",
    "SFTPreprocessTransform",
}


def __getattr__(name: str):
    """Lazy import trainer transforms for backwards compatibility."""
    if name in _TRAINER_TRANSFORMS:
        from easydel.trainers import prompt_transforms as prompt_transforms

        return getattr(prompt_transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_ROLE_MAPPING",
    "AddField",
    "BCOPreprocessTransform",
    "CPOPreprocessTransform",
    "ChainedTransform",
    "ChatTemplateTransform",
    "CombineFields",
    "ConvertInputOutputToChatML",
    "ConvertToChatML",
    "DPOPreprocessTransform",
    "DropFields",
    "ExpandTransform",
    "ExtractField",
    "FilterByField",
    "FilterNonEmpty",
    "FilterTransform",
    "FirstFitPacker",
    "GRPOPreprocessTransform",
    "GreedyPacker",
    "KTOPreprocessTransform",
    "MapField",
    "MapTransform",
    "MaybeApplyChatTemplate",
    "MixStage",
    "MixedShardState",
    "MixedShardedSource",
    "ORPOPreprocessTransform",
    "PackStage",
    "PackedSequence",
    "PackedShardedSource",
    "PoolPacker",
    "RenameFields",
    "RewardPreprocessTransform",
    "SFTPreprocessTransform",
    "SelectFields",
    "TokenizeStage",
    "TokenizedShardedSource",
    "TokenizerManager",
    "Transform",
    "TransformedShardedSource",
    "WeightScheduler",
    "batched_tokenize_iterator",
    "block_mixture_interleave",
    "compute_tokenizer_hash",
    "pack_constant_length",
    "pack_pre_tokenized",
    "tokenize_dataset_config",
]
