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

"""Configuration for Stable Diffusion training."""

from __future__ import annotations

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils import Registry

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "stable-diffusion")
@auto_pytree
class StableDiffusionConfig(TrainingArguments):
	"""
	Training configuration for Stable Diffusion models.

	This configuration extends the base TrainingArguments with specific settings
	for diffusion model training, including VAE encoding, text conditioning,
	noise scheduling, and loss weighting strategies.

	Attributes:
		# Image and Resolution Settings
		resolution: int = 512
			Image resolution for training (both height and width)
		center_crop: bool = True
			Whether to center crop images to resolution
		random_flip: bool = False
			Whether to randomly flip images horizontally

		# Model Component Settings
		pretrained_model_name_or_path: str
			Path or name of pretrained Stable Diffusion model
		revision: str | None = None
			Model revision to use (e.g., "main", "fp16")
		variant: str | None = None
			Model variant (e.g., "fp16")

		# VAE Settings
		vae_encode_batch_size: int | None = None
			Batch size for VAE encoding (if None, uses total_batch_size)
		cache_latents: bool = False
			Whether to cache VAE latents to save computation
		scaling_factor: float = 0.18215
			VAE latent scaling factor (SD 1.x/2.x: 0.18215, SDXL: 0.13025)

		# Text Encoder Settings
		train_text_encoder: bool = False
			Whether to train the text encoder along with UNet
		text_encoder_learning_rate: float | None = None
			Separate learning rate for text encoder (if None, uses main learning_rate)
		max_sequence_length: int = 77
			Maximum text sequence length for CLIP encoder

		# Noise Scheduler Settings
		prediction_type: str = "epsilon"
			Noise prediction type: "epsilon", "v_prediction", or "sample"
		num_train_timesteps: int = 1000
			Number of diffusion timesteps for training
		beta_start: float = 0.00085
			Starting beta value for noise schedule
		beta_end: float = 0.012
			Ending beta value for noise schedule
		beta_schedule: str = "scaled_linear"
			Beta schedule type: "linear", "scaled_linear", "squaredcos_cap_v2"

		# Loss Weighting
		snr_gamma: float | None = None
			SNR (Signal-to-Noise Ratio) gamma for loss weighting
			If None, no SNR weighting is applied
			Common values: 5.0 (Min-SNR weighting)
		loss_type: str = "mse"
			Loss function type: "mse", "l1", "huber"

		# Timestep Sampling
		timestep_bias_strategy: str = "none"
			Strategy for biased timestep sampling:
			- "none": uniform sampling
			- "earlier": bias towards earlier timesteps
			- "later": bias towards later timesteps
			- "range": sample from specific range
		timestep_bias_multiplier: float = 1.0
			Multiplier for timestep bias
		timestep_bias_begin: int = 0
			Begin timestep for range bias
		timestep_bias_end: int = 1000
			End timestep for range bias
		timestep_bias_portion: float = 0.25
			Portion of timesteps to apply bias to

		# SDXL-specific Settings
		use_sdxl: bool = False
			Whether training SDXL model (affects embeddings)
		proportion_empty_prompts: float = 0.0
			Proportion of prompts to replace with empty strings (for classifier-free guidance)

		# Dataset Settings
		image_column: str = "image"
			Name of image column in dataset
		caption_column: str = "text"
			Name of caption column in dataset
		conditioning_dropout_prob: float = 0.0
			Probability of dropping text conditioning (unconditional training)

		# Memory Optimization
		enable_xformers_memory_efficient_attention: bool = False
			Use memory-efficient attention (requires xformers)
		gradient_checkpointing: bool = False
			Use gradient checkpointing to save memory
		allow_tf32: bool = True
			Allow TF32 on Ampere GPUs for faster training

	Example:
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
	"""

	# Image and Resolution Settings
	resolution: int = field(
		default=512,
		metadata={"help": "Image resolution for training (both height and width)."},
	)
	center_crop: bool = field(
		default=True,
		metadata={"help": "Whether to center crop images to resolution."},
	)
	random_flip: bool = field(
		default=False,
		metadata={"help": "Whether to randomly flip images horizontally."},
	)

	# Model Component Settings
	pretrained_model_name_or_path: str = field(
		default="runwayml/stable-diffusion-v1-5",
		metadata={"help": "Path or name of pretrained Stable Diffusion model."},
	)
	revision: str | None = field(
		default=None,
		metadata={"help": "Model revision to use (e.g., 'main', 'fp16')."},
	)
	variant: str | None = field(
		default=None,
		metadata={"help": "Model variant (e.g., 'fp16')."},
	)

	# VAE Settings
	vae_encode_batch_size: int | None = field(
		default=None,
		metadata={"help": "Batch size for VAE encoding (if None, uses total_batch_size)."},
	)
	cache_latents: bool = field(
		default=False,
		metadata={"help": "Whether to cache VAE latents to save computation."},
	)
	scaling_factor: float = field(
		default=0.18215,
		metadata={"help": "VAE latent scaling factor (SD 1.x/2.x: 0.18215, SDXL: 0.13025)."},
	)

	# Text Encoder Settings
	train_text_encoder: bool = field(
		default=False,
		metadata={"help": "Whether to train the text encoder along with UNet."},
	)
	text_encoder_learning_rate: float | None = field(
		default=None,
		metadata={"help": "Separate learning rate for text encoder (if None, uses main learning_rate)."},
	)
	text_encoder_max_length: int = field(
		default=77,
		metadata={"help": "Maximum text sequence length for CLIP encoder."},
	)

	# Noise Scheduler Settings
	prediction_type: str = field(
		default="epsilon",
		metadata={"help": "Noise prediction type: 'epsilon', 'v_prediction', or 'sample'."},
	)
	num_train_timesteps: int = field(
		default=1000,
		metadata={"help": "Number of diffusion timesteps for training."},
	)
	beta_start: float = field(
		default=0.00085,
		metadata={"help": "Starting beta value for noise schedule."},
	)
	beta_end: float = field(
		default=0.012,
		metadata={"help": "Ending beta value for noise schedule."},
	)
	beta_schedule: str = field(
		default="scaled_linear",
		metadata={"help": "Beta schedule type: 'linear', 'scaled_linear', 'squaredcos_cap_v2'."},
	)

	# Loss Weighting
	snr_gamma: float | None = field(
		default=None,
		metadata={"help": "SNR gamma for Min-SNR loss weighting (common: 5.0). None disables."},
	)
	loss_type: str = field(
		default="mse",
		metadata={"help": "Loss function type: 'mse', 'l1', 'huber'."},
	)

	# Timestep Sampling
	timestep_bias_strategy: str = field(
		default="none",
		metadata={"help": "Timestep sampling strategy: 'none', 'earlier', 'later', 'range'."},
	)
	timestep_bias_multiplier: float = field(
		default=1.0,
		metadata={"help": "Multiplier for timestep bias."},
	)
	timestep_bias_begin: int = field(
		default=0,
		metadata={"help": "Begin timestep for range bias."},
	)
	timestep_bias_end: int = field(
		default=1000,
		metadata={"help": "End timestep for range bias."},
	)
	timestep_bias_portion: float = field(
		default=0.25,
		metadata={"help": "Portion of timesteps to apply bias to."},
	)

	# SDXL-specific Settings
	use_sdxl: bool = field(
		default=False,
		metadata={"help": "Whether training SDXL model (affects embeddings)."},
	)
	proportion_empty_prompts: float = field(
		default=0.0,
		metadata={"help": "Proportion of prompts to replace with empty strings."},
	)

	# Dataset Settings
	image_column: str = field(
		default="image",
		metadata={"help": "Name of image column in dataset."},
	)
	caption_column: str = field(
		default="text",
		metadata={"help": "Name of caption column in dataset."},
	)
	conditioning_dropout_prob: float = field(
		default=0.0,
		metadata={"help": "Probability of dropping text conditioning."},
	)

	# Memory Optimization
	enable_xformers_memory_efficient_attention: bool = field(
		default=False,
		metadata={"help": "Use memory-efficient attention (requires xformers)."},
	)
	gradient_checkpointing: bool = field(
		default=False,
		metadata={"help": "Use gradient checkpointing to save memory."},
	)
	allow_tf32: bool = field(
		default=True,
		metadata={"help": "Allow TF32 on Ampere GPUs for faster training."},
	)

	def __post_init__(self):
		"""Validate and set default values."""
		super().__post_init__()

		# Set VAE batch size if not specified
		if self.vae_encode_batch_size is None:
			self.vae_encode_batch_size = self.total_batch_size

		# Set text encoder learning rate if not specified
		if self.text_encoder_learning_rate is None:
			self.text_encoder_learning_rate = self.learning_rate

		# Validate prediction type
		valid_prediction_types = ["epsilon", "v_prediction", "sample"]
		if self.prediction_type not in valid_prediction_types:
			raise ValueError(f"prediction_type must be one of {valid_prediction_types}, got {self.prediction_type}")

		# Validate timestep bias strategy
		valid_strategies = ["none", "earlier", "later", "range"]
		if self.timestep_bias_strategy not in valid_strategies:
			raise ValueError(
				f"timestep_bias_strategy must be one of {valid_strategies}, got {self.timestep_bias_strategy}"
			)

		# Validate loss type
		valid_loss_types = ["mse", "l1", "huber"]
		if self.loss_type not in valid_loss_types:
			raise ValueError(f"loss_type must be one of {valid_loss_types}, got {self.loss_type}")
