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

"""Ray-oriented distributed trainer launcher utilities for EasyDeL.

This module provides :class:`RayDistributedTrainer` (and its JSON-persistable
:class:`RayDistributedConfig`), a lightweight launcher wrapper intended to be
invoked from Ray workers or distributed scaling scripts.

The wrapper handles pre-training setup and delegates the training loop, mesh
sharding, and checkpoint resumption to an inner EasyDeL trainer class:
- Model class resolution from EasyDeL's module registry via ``model_type`` /
  ``model_task`` or an explicit ``model_class``
- Linear model-width scaling through ``config_scaling_variables`` and
  ``scaling_index``
- Offloaded model initialization on ``offload_backend`` (default ``"cpu"``)
- Optional checkpoint loading from an explicit ``bucket_path``
- JSON configuration persistence via ``save_config`` / ``from_config``
- Signature-aware construction of ``Trainer`` and specialized subclasses
  (``SFTTrainer``, ``DPOTrainer``, ``RewardTrainer``, ``EmbeddingTrainer``,
  ``PRMTrainer``, etc.)

Example:
    >>> from easydel.modules.llama import LlamaForCausalLM
    >>> from easydel.trainers import TrainingArguments
    >>> from easydel.trainers.ray_scaler import RayDistributedTrainer
    >>> launcher = RayDistributedTrainer(
    ...     pretrained_model_name_or_path="meta-llama/Llama-3.2-1B",
    ...     model_class=LlamaForCausalLM,
    ... )
    >>> launcher.train(
    ...     scaling_index=1,
    ...     arguments=TrainingArguments(save_directory="./checkpoints"),
    ...     dataset_train=dataset,
    ... )
"""

from .distributed_trainer import RayDistributedConfig, RayDistributedTrainer

__all__ = ("RayDistributedConfig", "RayDistributedTrainer")
