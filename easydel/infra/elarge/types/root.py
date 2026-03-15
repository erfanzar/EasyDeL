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

"""Top-level ELMConfig TypedDict.

Defines the root configuration structure that combines all ELM subsystem
configurations into a single declarative specification.
"""

from __future__ import annotations

from typing import NotRequired, Required, TypedDict

from .data import DataMixtureCfg
from .engine import BaseCfg, eSurgeCfg
from .eval import EvalKwargs
from .infra import PlatformCfg, ShardingCfg
from .model import LoaderCfg, ModelCfg
from .quantization import QuantizationCfg
from .training import TrainerConfig


class ELMConfig(TypedDict, total=False):
    """Complete ELM (EasyDeL Large Model) configuration structure.

    This is the top-level configuration type that combines all configuration
    sections for model loading, sharding, quantization, inference, training,
    and data pipelines into a single declarative specification.

    Attributes:
        model: Model identification and source configuration (required).
        teacher_model: Teacher model config for knowledge distillation
            workflows.
        reference_model: Reference model config for preference optimization
            (DPO, ORPO, etc.).
        loader: Data type, precision, and device settings for model loading.
        sharding: Distributed sharding and mesh configuration.
        platform: Hardware backend and platform selection.
        quantization: Weight and KV cache quantization settings.
        base_config: Base model configuration values and operation overrides.
        mixture: Dataset mixture and data pipeline configuration.
        esurge: eSurge inference engine configuration.
        trainer: Training configuration (trainer type, hyperparameters,
            loss settings).
        eval: Default evaluation keyword arguments for lm-evaluation-harness.
    """

    model: Required[ModelCfg]
    teacher_model: NotRequired[ModelCfg]
    reference_model: NotRequired[ModelCfg]
    loader: NotRequired[LoaderCfg]
    sharding: NotRequired[ShardingCfg]
    platform: NotRequired[PlatformCfg]
    quantization: NotRequired[QuantizationCfg]
    base_config: NotRequired[BaseCfg]
    mixture: NotRequired[DataMixtureCfg]
    esurge: NotRequired[eSurgeCfg]
    trainer: NotRequired[TrainerConfig]
    eval: NotRequired[EvalKwargs]
