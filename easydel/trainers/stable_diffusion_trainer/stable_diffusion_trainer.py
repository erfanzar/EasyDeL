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

"""Stable Diffusion Trainer for EasyDeL.

This module provides a comprehensive trainer for Stable Diffusion models,
supporting text-to-image training with various diffusion schedulers and
loss weighting strategies.
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer.loggings import get_logger
from eformer.escale import match_partition_rules
from jax.sharding import NamedSharding, PartitionSpec
from transformers import CLIPTextModel, CLIPTokenizer

from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.modules.unet2d import UNet2DConditionModel
from easydel.modules.vae import AutoencoderKL
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time
from easydel.utils.traversals import specs_to_name_sharding

from ..base_trainer import BaseTrainer, TrainerConfigureFunctionOutput
from ..trainer_protocol import BaseProgressBar, MetricsTracker, StepMetrics, TrainerOutput
from ._fn import stable_diffusion_evaluation_step, stable_diffusion_training_step
from .stable_diffusion_config import StableDiffusionConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset

logger = get_logger(__name__)


@Registry.register("trainer", "stable-diffusion")
class StableDiffusionTrainer(BaseTrainer):
	"""
	Stable Diffusion trainer for text-to-image generation.

	This trainer handles the complete training pipeline for Stable Diffusion models:
	- VAE encoding of images to latent space
	- Text encoding with CLIP
	- UNet noise prediction training
	- Support for DDPM/DDIM schedulers
	- SNR-based loss weighting
	- Timestep bias sampling
	- Optional text encoder fine-tuning

	The trainer integrates with EasyDeL's infrastructure for:
	- Distributed training with sharding
	- Mixed precision training
	- Gradient accumulation
	- Comprehensive checkpointing
	- Metrics logging to WandB/TensorBoard

	Example:
		>>> from easydel.trainers import StableDiffusionTrainer, StableDiffusionConfig
		>>>
		>>> config = StableDiffusionConfig(
		...     pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
		...     resolution=512,
		...     learning_rate=1e-5,
		...     num_train_epochs=100,
		...     total_batch_size=4,
		...     snr_gamma=5.0,
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

	arguments: StableDiffusionConfig

	def __init__(
		self,
		arguments: StableDiffusionConfig,
		dataset_train: Dataset | None = None,
		dataset_eval: Dataset | None = None,
		data_collator: tp.Callable | None = None,
		**kwargs,
	):
		"""
		Initialize Stable Diffusion trainer.

		Args:
			arguments: Training configuration
			dataset_train: Training dataset (should contain images and captions)
			dataset_eval: Evaluation dataset
			data_collator: Optional data collator function
			**kwargs: Additional arguments passed to BaseTrainer
		"""
		# We'll initialize the models in _initialize_models
		self.unet_state = None
		self.vae_state = None
		self.text_encoder_state = None
		self.noise_scheduler = None
		self.tokenizer = None

		# Initialize models before calling super().__init__
		self._initialize_models(arguments)

		# Now initialize base trainer with the UNet state
		super().__init__(
			arguments=arguments,
			model_state=self.unet_state,
			dataset_train=dataset_train,
			dataset_eval=dataset_eval,
			data_collator=data_collator,
			**kwargs,
		)

	def _initialize_models(self, arguments: StableDiffusionConfig):
		"""Initialize UNet, VAE, text encoder, and noise scheduler."""
		logger.info(f"Loading Stable Diffusion models from {arguments.pretrained_model_name_or_path}")

		# Import here to avoid circular imports
		from maxdiffusion import FlaxDDPMScheduler

		# Initialize noise scheduler
		self.noise_scheduler = FlaxDDPMScheduler.from_pretrained(
			arguments.pretrained_model_name_or_path,
			subfolder="scheduler",
			revision=arguments.revision,
		)

		# Update scheduler config with training parameters
		self.noise_scheduler.config.num_train_timesteps = arguments.num_train_timesteps
		self.noise_scheduler.config.beta_start = arguments.beta_start
		self.noise_scheduler.config.beta_end = arguments.beta_end
		self.noise_scheduler.config.beta_schedule = arguments.beta_schedule
		self.noise_scheduler.config.prediction_type = arguments.prediction_type

		# Initialize tokenizer
		self.tokenizer = CLIPTokenizer.from_pretrained(
			arguments.pretrained_model_name_or_path,
			subfolder="tokenizer",
			revision=arguments.revision,
		)

		# Initialize text encoder (HuggingFace CLIP for now)
		# TODO: Port to EasyDeL once CLIP is available
		text_encoder = CLIPTextModel.from_pretrained(
			arguments.pretrained_model_name_or_path,
			subfolder="text_encoder",
			revision=arguments.revision,
		)

		# For now, we'll wrap the HuggingFace model
		# In a full implementation, this should be converted to JAX/Flax
		logger.warning(
			"Using HuggingFace CLIP text encoder. For full JAX implementation, "
			"CLIP should be ported to EasyDeL."
		)

		# Initialize UNet from EasyDeL
		# TODO: Load from pretrained once conversion is available
		# For now, create from config
		logger.info("Initializing UNet2DConditionModel")
		from easydel.modules.unet2d import UNet2DConfig

		unet_config = UNet2DConfig(
			sample_size=arguments.resolution // 8,  # Latent size
			in_channels=4,
			out_channels=4,
			center_input_sample=False,
			flip_sin_to_cos=True,
			freq_shift=0,
			down_block_types=[
				"CrossAttnDownBlock2D",
				"CrossAttnDownBlock2D",
				"CrossAttnDownBlock2D",
				"DownBlock2D",
			],
			up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
			block_out_channels=[320, 640, 1280, 1280],
			layers_per_block=2,
			attention_head_dim=8,
			cross_attention_dim=768,  # CLIP embedding dim
			use_linear_projection=False,
		)

		# Create UNet model and state
		unet = UNet2DConditionModel(
			config=unet_config,
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			rngs=jax.random.PRNGKey(0),
		)
		self.unet_state = unet.to_state()

		# Initialize VAE from EasyDeL
		logger.info("Initializing AutoencoderKL (VAE)")
		from easydel.modules.vae import VAEConfig

		vae_config = VAEConfig(
			in_channels=3,
			out_channels=3,
			down_block_types=["DownEncoderBlock2D"] * 4,
			up_block_types=["UpDecoderBlock2D"] * 4,
			block_out_channels=[128, 256, 512, 512],
			layers_per_block=2,
			latent_channels=4,
			scaling_factor=arguments.scaling_factor,
		)

		vae = AutoencoderKL(
			config=vae_config,
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			rngs=jax.random.PRNGKey(1),
		)
		self.vae_state = vae.to_state()

		# Create text encoder state (placeholder for now)
		# In full implementation, convert HF model to JAX
		self.text_encoder_state = None  # Will be properly initialized later

		logger.info("Models initialized successfully")

	def create_grain_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		"""
		Create data collation function for Grain dataloaders.

		Args:
			max_sequence_length: Maximum text sequence length
			truncation_mode: How to truncate sequences

		Returns:
			Collation function
		"""

		def collate_fn(batch):
			"""Collate batch of images and captions."""
			# Extract images and captions
			images = []
			captions = []

			for example in batch:
				images.append(example[self.arguments.image_column])
				captions.append(example[self.arguments.caption_column])

			# Tokenize captions
			tokenized = self.tokenizer(
				captions,
				padding="max_length",
				max_length=self.arguments.text_encoder_max_length,
				truncation=True,
				return_tensors="np",
			)

			# Process images to tensors
			# Assuming images are PIL Images or similar
			import numpy as np
			from PIL import Image

			processed_images = []
			for img in images:
				if isinstance(img, Image.Image):
					# Resize and center crop
					if self.arguments.center_crop:
						img = img.resize(
							(self.arguments.resolution, self.arguments.resolution),
							Image.LANCZOS,
						)
					# Convert to array and normalize
					img_array = np.array(img).astype(np.float32) / 255.0
					# Convert to CHW format
					img_array = np.transpose(img_array, (2, 0, 1))
					# Normalize to [-1, 1]
					img_array = 2.0 * img_array - 1.0
					processed_images.append(img_array)

			return {
				"pixel_values": jnp.array(processed_images),
				"input_ids": jnp.array(tokenized["input_ids"]),
			}

		return collate_fn

	def create_tfds_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		"""
		Create data collation function for TensorFlow dataloaders.

		Args:
			max_sequence_length: Maximum text sequence length
			truncation_mode: How to truncate sequences

		Returns:
			Collation function
		"""
		# Similar to grain version but adapted for TF datasets
		return self.create_grain_collect_function(max_sequence_length, truncation_mode)

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"],
	) -> tp.Callable:
		"""Create appropriate collect function based on dataloader type."""
		return (
			self.create_grain_collect_function(max_sequence_length, truncation_mode)
			if self.arguments.use_grain
			else self.create_tfds_collect_function(max_sequence_length, truncation_mode)
		)

	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configure and JIT-compile training and evaluation functions.

		Returns:
			TrainerConfigureFunctionOutput with compiled functions
		"""
		# Get mesh from model
		mesh = self.model.mesh

		# Create sharding specs for states
		unet_sharding = self._create_state_sharding(self.unet_state)
		vae_sharding = self._create_state_sharding(self.vae_state)
		text_encoder_sharding = None  # Will be set when text encoder is properly initialized

		# Create data sharding
		data_sharding = NamedSharding(mesh, self.arguments.step_partition_spec)

		# Create training step function
		training_step_fn = partial(
			stable_diffusion_training_step,
			noise_scheduler=self.noise_scheduler,
			config=self.arguments,
			learning_rate_fn=self.scheduler,
			partition_spec=self.arguments.step_partition_spec,
			gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
		)

		# Create evaluation step function
		evaluation_step_fn = partial(
			stable_diffusion_evaluation_step,
			noise_scheduler=self.noise_scheduler,
			config=self.arguments,
			partition_spec=self.arguments.step_partition_spec,
		)

		# JIT compile functions
		logger.info("JIT compiling training and evaluation functions...")

		# Training function signature
		sharded_training_step_function = ejit(
			training_step_fn,
			donate_argnums=(0, 1, 2),  # Donate state objects
		)

		# Evaluation function signature
		sharded_evaluation_step_function = ejit(
			evaluation_step_fn,
		)

		return TrainerConfigureFunctionOutput(
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=self.arguments.get_streaming_checkpointer(),
		)

	def _create_state_sharding(self, state: EasyDeLState):
		"""Create sharding specification for a model state."""
		if state is None:
			return None

		# Get partition rules from model config
		rules = state.model.config.get_partition_rules()

		# Create sharding specs
		import flax.nnx as nn

		shape = nn.eval_shape(lambda: state)
		state_shardings = specs_to_name_sharding(match_partition_rules(rules, shape))

		return state_shardings

	def train(self) -> TrainerOutput:
		"""
		Execute the main training loop.

		Returns:
			TrainerOutput with trained state and metrics
		"""
		logger.info("Starting Stable Diffusion training...")

		# Ensure functions are compiled
		self._ensure_functions_compiled()

		# Setup initial metrics and hooks
		self.start_training_hook()
		self._setup_initial_metrics(self.unet_state)

		# Get dataloaders
		train_iter = iter(self.dataloader_train)

		# Progress bar
		pbar = self.create_progress_bar(
			total=self.max_training_steps,
			desc="Training",
			disabled=not self.is_enable,
		)

		# Training state
		unet_state = self.unet_state
		vae_state = self.vae_state
		text_encoder_state = self.text_encoder_state

		# RNG for training
		rng = jax.random.PRNGKey(self.arguments.shuffle_seed_train)

		# Metrics tracker
		metrics_tracker = MetricsTracker()

		try:
			for step in range(self.arguments.step_start_point, self.max_training_steps):
				# Get next batch
				try:
					batch, train_iter = self._get_next_batch(train_iter, self.dataloader_train)
				except StopIteration:
					break

				# Preprocess batch
				batch, _ = self._preprocess_batch_input(unet_state, batch, is_train=True)

				# Training step
				with capture_time() as step_time:
					unet_state, text_encoder_state, metrics, rng = self.sharded_training_step_function(
						unet_state,
						vae_state,
						text_encoder_state,
						batch,
						rng=rng,
					)

				# Apply training hooks
				metrics = self.apply_training_hooks(metrics)

				# Update metrics
				step_metrics = StepMetrics.create_step_metrics(
					loss_metrics=metrics,
					step_time=step_time.elapsed,
					learning_rate=self.scheduler(step) if self.scheduler else self.arguments.learning_rate,
					step=step,
				)
				metrics_tracker.update(step_metrics)

				# Log metrics
				self.log_metrics(
					metrics=step_metrics.to_dict(),
					pbar=pbar,
					step=step,
					mode="train",
				)

				# Save checkpoint
				if self._should_save_checkpoint(step):
					self.unet_state = unet_state
					self.text_encoder_state = text_encoder_state
					self._save_state(unet_state)

				# Evaluation
				if self._should_run_evaluation(step):
					eval_metrics = self._run_evaluation(unet_state, vae_state, text_encoder_state, step)
					self.log_metrics(metrics=eval_metrics, pbar=pbar, step=step, mode="eval")

		except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest) as e:
			logger.warning(f"Training interrupted: {e}")
			self.unet_state = unet_state
			self.text_encoder_state = text_encoder_state
			return self._prepare_training_output(unet_state, run_exception=e)

		# Final save
		self.unet_state = unet_state
		self.text_encoder_state = text_encoder_state

		pbar.close()
		return self._prepare_training_output(unet_state)

	def _run_evaluation(self, unet_state, vae_state, text_encoder_state, step: int):
		"""Run evaluation loop."""
		if self.dataloader_eval is None:
			return {}

		logger.info(f"Running evaluation at step {step}...")

		eval_metrics_list = []
		eval_iter = iter(self.dataloader_eval)
		rng = jax.random.PRNGKey(0)

		for eval_step in range(min(self.max_evaluation_steps, 100)):  # Limit eval steps
			try:
				batch, eval_iter = self._get_next_batch(eval_iter, self.dataloader_eval)
			except StopIteration:
				break

			batch, _ = self._preprocess_batch_input(unet_state, batch, is_train=False)

			# Evaluation step
			_, metrics, rng = self.sharded_evaluation_step_function(
				unet_state,
				vae_state,
				text_encoder_state,
				batch,
				rng=rng,
			)

			eval_metrics_list.append(metrics)

		# Average metrics
		if eval_metrics_list:
			avg_loss = jnp.mean(jnp.array([m.loss for m in eval_metrics_list]))
			return {"eval/loss": float(avg_loss)}

		return {}
