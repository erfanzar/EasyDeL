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

from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn
from easydel.trainers.training_configurations import TrainingArguments


@auto_pytree
class ImageDiffusionConfig(TrainingArguments):
	"""
	Configuration for image diffusion training with rectified flow.

	This configuration extends TrainingArguments with image diffusion specific parameters
	for training DiT (Diffusion Transformer) models.

	Args:
		trainer_prefix: Prefix name for trainer checkpoints
		dataset_image_field: Name of the field containing images in the dataset
		dataset_label_field: Name of the field containing class labels (optional)
		image_size: Size of input images (assumes square images)
		num_train_timesteps: Number of diffusion timesteps during training
		prediction_type: Type of prediction ('velocity', 'epsilon', or 'sample')
		use_vae: Whether to train in latent space using a VAE
		vae_model_name_or_path: Path/name of pretrained VAE for latent diffusion
		class_conditional: Whether to use class-conditional generation
		class_dropout_prob: Probability of dropping class labels (for CFG)
		loss_aggregation: How to aggregate loss ('mean' or 'sum')
		loss_scale: Scaling factor for the loss
		min_snr_gamma: Minimum SNR gamma for loss weighting (set to -1 to disable)
	"""

	trainer_prefix: str | None = field(
		default="imagediffusiontrainer",
		metadata={"help": "default prefix name for trainer."},
	)
	dataset_image_field: str = field(
		default="image",
		metadata={"help": "Name of the field in the dataset that contains images."},
	)
	dataset_label_field: str | None = field(
		default="label",
		metadata={"help": "Name of the field in the dataset that contains class labels."},
	)
	image_size: int = field(
		default=32,
		metadata={"help": "Size of input images (assumes square images)."},
	)
	num_train_timesteps: int = field(
		default=1000,
		metadata={"help": "Number of diffusion timesteps during training."},
	)
	prediction_type: str = field(
		default="velocity",
		metadata={"help": "Type of prediction: 'velocity', 'epsilon', or 'sample'."},
	)
	use_vae: bool = field(
		default=False,
		metadata={"help": "Whether to train in latent space using a VAE."},
	)
	vae_model_name_or_path: str | None = field(
		default=None,
		metadata={"help": "Path or name of pretrained VAE for latent diffusion."},
	)
	class_conditional: bool = field(
		default=True,
		metadata={"help": "Whether to use class-conditional generation."},
	)
	class_dropout_prob: float = field(
		default=0.1,
		metadata={"help": "Probability of dropping class labels for classifier-free guidance."},
	)
	loss_aggregation: str = field(
		default="mean",
		metadata={"help": "Loss aggregation method: 'mean' or 'sum'."},
	)
	loss_scale: float = field(
		default=1.0,
		metadata={"help": "Scaling factor for the loss."},
	)
	min_snr_gamma: float = field(
		default=-1.0,
		metadata={"help": "Minimum SNR gamma for loss weighting. Set to -1 to disable."},
	)

	__hash__ = hash_fn
