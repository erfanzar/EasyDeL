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

"""ELM type definitions package.

Re-exports all TypedDicts, type aliases, and type-related utilities
from the sub-modules so that external code can continue importing from
``easydel.infra.elarge.types``.

Sub-modules:
    aliases: Fundamental type aliases (DTypeLike, PrecisionLike, etc.).
    model: Model identification and loading configs (ModelCfg, LoaderCfg).
    infra: Sharding and platform configs (ShardingCfg, PlatformCfg).
    quantization: Quantization configs (QuantizationCfg).
    engine: Inference engine configs (BaseCfg, eSurgeCfg).
    data: Dataset and mixture configs (DataMixtureCfg, TokenizationCfg).
    eval: Evaluation and benchmark configs (BenchmarkConfig, EvalKwargs).
    training: Trainer configs and defaults (TrainerConfig, LossConfig).
    root: Top-level eLMConfig combining all sections.
"""

# Re-export TaskType for backward compatibility (originally imported via types.py)
from easydel.infra.factory import TaskType

from .aliases import (
    DatasetTypeLike,
    DTypeLike,
    OperationImplName,
    PartitionRules,
    PrecisionLike,
)
from .data import (
    DataMixtureCfg,
    DatasetMixtureCfg,
    DatasetSaveCfg,
    TextDatasetInformCfg,
    TokenizationCfg,
    VisualDatasetInformCfg,
)
from .engine import BaseCfg, eSurgeCfg
from .eval import (
    BenchmarkConfig,
    BenchmarkTask,
    BenchmarkTasks,
    EvalKwargs,
    ResolvedBenchmarkConfig,
)
from .infra import PlatformCfg, ShardingCfg
from .model import LoaderCfg, ModelCfg, OperationConfigsDict
from .quantization import EasyDeLQuantizationCfg, QuantizationCfg
from .root import eLMConfig
from .training import (
    BASE_TRAINER_DEFAULTS,
    TRAINER_SPECIFIC_DEFAULTS,
    AgenticMoshPitTrainerCfg,
    AsyncGRPOTrainerCfg,
    BaseTrainerCfg,
    BCOTrainerCfg,
    CPOTrainerCfg,
    DistillationTrainerCfg,
    DPOTrainerCfg,
    DPPOTrainerCfg,
    GFPOTrainerCfg,
    GKDTrainerCfg,
    GOLDTrainerCfg,
    GRPOTrainerCfg,
    GRPOWithReplayBufferTrainerCfg,
    GSPOTokenTrainerCfg,
    GSPOTrainerCfg,
    KTOTrainerCfg,
    LossConfig,
    MiniLLMTrainerCfg,
    NashMDTrainerCfg,
    NeMoGymTrainerCfg,
    OnlineDPOTrainerCfg,
    OnPolicyDistillationTrainerCfg,
    ORPOTrainerCfg,
    PAPOTrainerCfg,
    PPOTrainerCfg,
    PRMTrainerCfg,
    RewardTrainerCfg,
    RLOOTrainerCfg,
    RLVRTrainerCfg,
    SDFTTrainerCfg,
    SDPOTrainerCfg,
    SeqKDTrainerCfg,
    SFTTrainerCfg,
    SparseDistillationTrainerCfg,
    SSDTrainerCfg,
    TPOTrainerCfg,
    TrainerConfig,
    XPOTrainerCfg,
    get_trainer_class,
    get_trainer_defaults,
    get_training_arguments_class,
    normalize_trainer_config,
    register_trainer_defaults,
)

__all__ = (
    "BASE_TRAINER_DEFAULTS",
    "TRAINER_SPECIFIC_DEFAULTS",
    "AgenticMoshPitTrainerCfg",
    "AsyncGRPOTrainerCfg",
    "BCOTrainerCfg",
    "BaseCfg",
    "BaseTrainerCfg",
    "BenchmarkConfig",
    "BenchmarkTask",
    "BenchmarkTasks",
    "CPOTrainerCfg",
    "DPOTrainerCfg",
    "DPPOTrainerCfg",
    "DTypeLike",
    "DataMixtureCfg",
    "DatasetMixtureCfg",
    "DatasetSaveCfg",
    "DatasetTypeLike",
    "DistillationTrainerCfg",
    "EasyDeLQuantizationCfg",
    "EvalKwargs",
    "GFPOTrainerCfg",
    "GKDTrainerCfg",
    "GOLDTrainerCfg",
    "GRPOTrainerCfg",
    "GRPOWithReplayBufferTrainerCfg",
    "GSPOTokenTrainerCfg",
    "GSPOTrainerCfg",
    "KTOTrainerCfg",
    "LoaderCfg",
    "LossConfig",
    "MiniLLMTrainerCfg",
    "ModelCfg",
    "NashMDTrainerCfg",
    "NeMoGymTrainerCfg",
    "ORPOTrainerCfg",
    "OnPolicyDistillationTrainerCfg",
    "OnlineDPOTrainerCfg",
    "OperationConfigsDict",
    "OperationImplName",
    "PAPOTrainerCfg",
    "PPOTrainerCfg",
    "PRMTrainerCfg",
    "PartitionRules",
    "PlatformCfg",
    "PrecisionLike",
    "QuantizationCfg",
    "RLOOTrainerCfg",
    "RLVRTrainerCfg",
    "ResolvedBenchmarkConfig",
    "RewardTrainerCfg",
    "SDFTTrainerCfg",
    "SDPOTrainerCfg",
    "SFTTrainerCfg",
    "SSDTrainerCfg",
    "SeqKDTrainerCfg",
    "ShardingCfg",
    "SparseDistillationTrainerCfg",
    "TPOTrainerCfg",
    "TaskType",
    "TextDatasetInformCfg",
    "TokenizationCfg",
    "TrainerConfig",
    "VisualDatasetInformCfg",
    "XPOTrainerCfg",
    "eLMConfig",
    "eSurgeCfg",
    "get_trainer_class",
    "get_trainer_defaults",
    "get_training_arguments_class",
    "normalize_trainer_config",
    "register_trainer_defaults",
)
