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

"""Stable Diffusion trainer for EasyDeL.

This module provides a comprehensive trainer for Stable Diffusion models,
supporting text-to-image training with:

- VAE encoding to latent space
- Text conditioning with CLIP
- UNet noise prediction training
- DDPM/DDIM scheduling
- SNR-based loss weighting
- Timestep bias sampling
- Mixed precision training
- Gradient accumulation
- Distributed training with sharding

Example:
	>>> from easydel.trainers.stable_diffusion_trainer import (
	...     StableDiffusionTrainer,
	...     StableDiffusionConfig,
	... )
	>>>
	>>> config = StableDiffusionConfig(
	...     pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
	...     resolution=512,
	...     learning_rate=1e-5,
	...     num_train_epochs=100,
	...     total_batch_size=4,
	...     gradient_accumulation_steps=4,
	...     snr_gamma=5.0,
	...     prediction_type="epsilon",
	... )
	>>>
	>>> trainer = StableDiffusionTrainer(
	...     arguments=config,
	...     dataset_train=train_dataset,
	...     dataset_eval=eval_dataset,
	... )
	>>>
	>>> output = trainer.train()
"""

from .stable_diffusion_config import StableDiffusionConfig
from .stable_diffusion_trainer import StableDiffusionTrainer

__all__ = [
	"StableDiffusionConfig",
	"StableDiffusionTrainer",
]
