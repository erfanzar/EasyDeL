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

"""Training step functions for Stable Diffusion trainer."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import optax
from eformer.escale import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.infra.loss_utils import LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully

if tp.TYPE_CHECKING:
	from .stable_diffusion_config import StableDiffusionConfig


def compute_snr(timesteps: jax.Array, noise_scheduler) -> jax.Array:
	"""
	Compute the signal-to-noise ratio for the given timesteps.

	Args:
		timesteps: Timesteps for which to compute SNR
		noise_scheduler: Noise scheduler with alphas_cumprod

	Returns:
		SNR values for each timestep
	"""
	alphas_cumprod = noise_scheduler.alphas_cumprod
	sqrt_alphas_cumprod = alphas_cumprod**0.5
	sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

	sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps]
	sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps]

	# SNR = (signal^2) / (noise^2)
	snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
	return snr


def generate_timestep_weights(config: StableDiffusionConfig, num_train_timesteps: int) -> jax.Array:
	"""
	Generate timestep sampling weights based on bias strategy.

	Args:
		config: Training configuration with timestep bias settings
		num_train_timesteps: Total number of training timesteps

	Returns:
		Array of weights for timestep sampling
	"""
	weights = jnp.ones(num_train_timesteps)

	if config.timestep_bias_strategy == "none":
		return weights

	if config.timestep_bias_strategy == "earlier":
		# Bias towards earlier timesteps (more noise)
		weights = jnp.exp(-jnp.arange(num_train_timesteps) * config.timestep_bias_multiplier / num_train_timesteps)
	elif config.timestep_bias_strategy == "later":
		# Bias towards later timesteps (less noise)
		weights = jnp.exp(jnp.arange(num_train_timesteps) * config.timestep_bias_multiplier / num_train_timesteps)
	elif config.timestep_bias_strategy == "range":
		# Bias towards specific range
		begin = config.timestep_bias_begin
		end = config.timestep_bias_end
		portion = config.timestep_bias_portion

		# Create mask for the range
		range_mask = (jnp.arange(num_train_timesteps) >= begin) & (jnp.arange(num_train_timesteps) < end)
		weights = jnp.where(range_mask, portion * num_train_timesteps, 1.0)

	# Normalize weights
	weights = weights / weights.sum()
	return weights


def stable_diffusion_training_step(
	unet_state,
	vae_state,
	text_encoder_state,
	batch: tp.Mapping[str, jax.Array],
	noise_scheduler,
	config: StableDiffusionConfig,
	learning_rate_fn: optax.Schedule = None,
	partition_spec: PartitionSpec | None = None,
	gradient_accumulation_steps: int = 1,
	rng: jax.Array = None,
) -> tuple:
	"""
	Stable Diffusion training step with noise prediction.

	This function implements the core training loop for Stable Diffusion:
	1. Encode images to latent space using VAE
	2. Sample noise and random timesteps
	3. Add noise to latents (forward diffusion process)
	4. Get text embeddings from text encoder
	5. Predict noise with UNet
	6. Compute loss (with optional SNR weighting)
	7. Update UNet (and optionally text encoder) parameters

	Args:
		unet_state: EasyDeLState for UNet model
		vae_state: EasyDeLState for VAE model (frozen)
		text_encoder_state: EasyDeLState for text encoder (may be frozen)
		batch: Dict with 'pixel_values' (images) and 'input_ids' (text tokens)
		noise_scheduler: Noise scheduler for adding noise and computing SNR
		config: Training configuration
		learning_rate_fn: Learning rate schedule function
		partition_spec: Partition specification for sharding
		gradient_accumulation_steps: Number of gradient accumulation steps
		rng: Random number generator key

	Returns:
		Tuple of (updated_unet_state, updated_text_encoder_state, metrics, new_rng)
	"""
	# Determine batch and minibatch sizes
	batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	# Split RNG for different random operations
	rng, sample_rng, noise_rng, timestep_rng, dropout_rng = jax.random.split(rng, 5)

	# Get pixel values and input_ids from batch
	pixel_values = batch["pixel_values"]
	input_ids = batch["input_ids"]

	def compute_loss(unet_params, text_encoder_params, minibatch):
		"""Compute diffusion loss for a minibatch."""
		minibatch_pixel_values = minibatch["pixel_values"]
		minibatch_input_ids = minibatch["input_ids"]

		# 1. Encode images to latent space using VAE
		# VAE expects NCHW format, minibatch_pixel_values should already be in that format
		vae_outputs = vae_state.model.encode(
			minibatch_pixel_values,
			deterministic=True,
			return_dict=True,
		)
		# Sample from the latent distribution
		latents = vae_outputs.latent_dist.sample(sample_rng)
		# Scale latents according to the VAE's scaling factor
		latents = latents * config.scaling_factor

		# 2. Sample noise to add to latents
		noise = jax.random.normal(noise_rng, latents.shape)

		# 3. Sample random timesteps for each image
		bsz = latents.shape[0]

		if config.timestep_bias_strategy == "none":
			# Uniform sampling
			timesteps = jax.random.randint(
				timestep_rng,
				(bsz,),
				0,
				noise_scheduler.config.num_train_timesteps,
			)
		else:
			# Biased sampling
			weights = generate_timestep_weights(config, noise_scheduler.config.num_train_timesteps)
			# Sample using categorical distribution with log probabilities
			timesteps = jax.random.categorical(timestep_rng, logits=jnp.log(weights), shape=(bsz,))

		# 4. Add noise to latents according to noise magnitude at each timestep
		# This is the forward diffusion process
		noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

		# 5. Get text embeddings for conditioning
		if config.train_text_encoder:
			# Use trainable text encoder parameters
			# Assuming text encoder is a simple model that processes input_ids
			# This would need to be adapted based on actual text encoder implementation
			encoder_hidden_states = text_encoder_state.model(minibatch_input_ids, deterministic=True)
		else:
			# Use frozen text encoder
			encoder_hidden_states = text_encoder_state.model(minibatch_input_ids, deterministic=True)

		# Optional: Apply conditioning dropout for classifier-free guidance training
		if config.conditioning_dropout_prob > 0.0:
			dropout_mask = jax.random.uniform(dropout_rng, (bsz, 1, 1)) > config.conditioning_dropout_prob
			# Create unconditional embeddings (zeros or special token)
			unconditional_embeds = jnp.zeros_like(encoder_hidden_states)
			encoder_hidden_states = jnp.where(dropout_mask, encoder_hidden_states, unconditional_embeds)

		# 6. Predict noise residual with UNet
		# UNet expects NCHW format for noisy_latents
		model_pred = unet_state.model(
			sample=noisy_latents,
			timesteps=timesteps,
			encoder_hidden_states=encoder_hidden_states,
			return_dict=True,
		).sample

		# 7. Get the target for loss depending on prediction type
		if noise_scheduler.config.prediction_type == "epsilon":
			target = noise
		elif noise_scheduler.config.prediction_type == "v_prediction":
			# v = alpha_t * noise - sigma_t * latents
			target = noise_scheduler.get_velocity(latents, noise, timesteps)
		elif noise_scheduler.config.prediction_type == "sample":
			target = latents
		else:
			raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

		# 8. Compute loss
		if config.loss_type == "mse":
			loss = jnp.mean((target - model_pred) ** 2)
		elif config.loss_type == "l1":
			loss = jnp.mean(jnp.abs(target - model_pred))
		elif config.loss_type == "huber":
			loss = jnp.mean(optax.huber_loss(model_pred, target, delta=1.0))
		else:
			raise ValueError(f"Unknown loss type {config.loss_type}")

		# 9. Apply SNR weighting if configured
		if config.snr_gamma is not None and config.snr_gamma > 0:
			snr = compute_snr(timesteps, noise_scheduler)
			# Min-SNR weighting
			snr_weights = jnp.minimum(snr, config.snr_gamma)

			if noise_scheduler.config.prediction_type == "epsilon":
				snr_weights = snr_weights / snr
			elif noise_scheduler.config.prediction_type == "v_prediction":
				snr_weights = snr_weights / (snr + 1)

			# Reshape for broadcasting
			snr_weights = snr_weights.reshape(-1, 1, 1, 1)
			loss = loss * snr_weights
			loss = loss.mean()

		# Create metrics
		metrics = LossMetrics(
			loss=loss,
			aux_loss=jnp.array(0.0),
			perplexity=jnp.array(0.0),
			learned_perplexity=jnp.array(0.0),
		)

		return loss, metrics

	# Prepare parameters for gradient computation
	if config.train_text_encoder:
		# Train both UNet and text encoder
		params = (unet_state.graphstate, text_encoder_state.graphstate)

		def loss_fn(params_tuple, minibatch):
			unet_params, text_encoder_params = params_tuple
			return compute_loss(unet_params, text_encoder_params, minibatch)

	else:
		# Train only UNet, text encoder is frozen
		params = unet_state.graphstate

		def loss_fn(unet_params, minibatch):
			return compute_loss(unet_params, text_encoder_state.graphstate, minibatch)

	# Compute gradients across minibatches
	gradients, metrics = minibatch_call(
		state=unet_state if not config.train_text_encoder else None,  # Will handle manually
		batch=batch,
		minibatch_size=minibatch_size,
		grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
	)

	# Update states
	if config.train_text_encoder:
		unet_grads, text_encoder_grads = gradients

		# Update UNet
		unet_state = update_state_respectfully(
			state=unet_state,
			gradients=unet_grads,
			loss_config=None,
			metrics=update_metrics(
				metrics=metrics,
				learning_rate_fn=learning_rate_fn,
				step=unet_state.step,
				gradients=unet_grads,
			),
		)

		# Update text encoder
		text_encoder_state = update_state_respectfully(
			state=text_encoder_state,
			gradients=text_encoder_grads,
			loss_config=None,
			metrics=LossMetrics(
				loss=metrics.loss,  # Same loss
				aux_loss=jnp.array(0.0),
				perplexity=jnp.array(0.0),
				learned_perplexity=jnp.array(0.0),
			),
		)
	else:
		# Update only UNet
		unet_state = update_state_respectfully(
			state=unet_state,
			gradients=gradients,
			loss_config=None,
			metrics=update_metrics(
				metrics=metrics,
				learning_rate_fn=learning_rate_fn,
				step=unet_state.step,
				gradients=gradients,
			),
		)

	return unet_state, text_encoder_state, metrics, rng


def stable_diffusion_evaluation_step(
	unet_state,
	vae_state,
	text_encoder_state,
	batch: tp.Mapping[str, jax.Array],
	noise_scheduler,
	config: StableDiffusionConfig,
	partition_spec: PartitionSpec | None = None,
	rng: jax.Array = None,
) -> tuple:
	"""
	Stable Diffusion evaluation step.

	Similar to training step but without gradient computation.

	Args:
		unet_state: EasyDeLState for UNet model
		vae_state: EasyDeLState for VAE model
		text_encoder_state: EasyDeLState for text encoder
		batch: Dict with 'pixel_values' and 'input_ids'
		noise_scheduler: Noise scheduler
		config: Training configuration
		partition_spec: Partition specification for sharding
		rng: Random number generator key

	Returns:
		Tuple of (predictions, metrics, new_rng)
	"""
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec or PartitionSpec())

	# Split RNG
	rng, sample_rng, noise_rng, timestep_rng = jax.random.split(rng, 4)

	pixel_values = batch["pixel_values"]
	input_ids = batch["input_ids"]

	# Encode images to latents
	vae_outputs = vae_state.model.encode(pixel_values, deterministic=True, return_dict=True)
	latents = vae_outputs.latent_dist.sample(sample_rng)
	latents = latents * config.scaling_factor

	# Sample noise and timesteps
	noise = jax.random.normal(noise_rng, latents.shape)
	bsz = latents.shape[0]
	timesteps = jax.random.randint(timestep_rng, (bsz,), 0, noise_scheduler.config.num_train_timesteps)

	# Add noise to latents
	noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

	# Get text embeddings
	encoder_hidden_states = text_encoder_state.model(input_ids, deterministic=True)

	# Predict noise
	model_pred = unet_state.model(
		sample=noisy_latents,
		timesteps=timesteps,
		encoder_hidden_states=encoder_hidden_states,
		return_dict=True,
	).sample

	# Compute loss
	if noise_scheduler.config.prediction_type == "epsilon":
		target = noise
	elif noise_scheduler.config.prediction_type == "v_prediction":
		target = noise_scheduler.get_velocity(latents, noise, timesteps)
	else:
		target = latents

	if config.loss_type == "mse":
		loss = jnp.mean((target - model_pred) ** 2)
	elif config.loss_type == "l1":
		loss = jnp.mean(jnp.abs(target - model_pred))
	else:
		loss = jnp.mean(optax.huber_loss(model_pred, target, delta=1.0))

	metrics = LossMetrics(
		loss=loss,
		aux_loss=jnp.array(0.0),
		perplexity=jnp.array(0.0),
		learned_perplexity=jnp.array(0.0),
	)

	return model_pred, metrics, rng
