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

"""Ray-based distributed training module for EasyDeL.

This module provides distributed training capabilities using Ray, enabling
efficient multi-node and multi-GPU training for large language models.
Ray provides a unified framework for scaling Python workloads from a single
machine to large clusters.

The module includes:
- RayDistributedTrainer: Main class for distributed training with Ray
- Support for data parallelism and model parallelism
- Automatic resource management and fault tolerance
- Integration with Ray's distributed computing primitives

Key Features:
- Seamless scaling from single GPU to multi-node clusters
- Automatic gradient synchronization across workers
- Efficient data loading with Ray datasets
- Built-in fault tolerance and checkpointing
- Support for heterogeneous hardware configurations

Example:
    >>> from easydel.trainers.ray_scaler import RayDistributedTrainer
    >>> trainer = RayDistributedTrainer(
    ...     num_workers=4,
    ...     use_gpu=True,
    ...     model=model,
    ...     train_dataset=dataset,
    ...     arguments=training_args
    ... )
    >>> trainer.train()

Requirements:
    - Ray must be installed: pip install ray[default]
    - For multi-node setups, Ray cluster must be configured

References:
    - Ray documentation: https://docs.ray.io/
    - Ray Train: https://docs.ray.io/en/latest/train/train.html
"""

from .distributed_trainer import RayDistributedTrainer

__all__ = ("RayDistributedTrainer",)
