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

"""Training step functions for image diffusion."""

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.trainers.trainer_protocol import StepMetrics


def compute_rectified_flow_loss(
	model_pred: jnp.ndarray,
	target: jnp.ndarray,
	timesteps: jnp.ndarray,
	num_train_timesteps: int = 1000,
	prediction_type: str = "velocity",
	min_snr_gamma: float = -1.0,
	loss_aggregation: str = "mean",
) -> jnp.ndarray:
	"""
	Compute rectified flow loss for image diffusion.

	Args:
		model_pred: Model prediction [batch_size, height, width, channels]
		target: Target (velocity, epsilon, or sample) [batch_size, height, width, channels]
		timesteps: Current timesteps [batch_size]
		num_train_timesteps: Total number of training timesteps
		prediction_type: Type of prediction ('velocity', 'epsilon', or 'sample')
		min_snr_gamma: Minimum SNR gamma for loss weighting (-1 to disable)
		loss_aggregation: How to aggregate loss ('mean' or 'sum')

	Returns:
		Loss value
	"""
	# Compute per-sample MSE loss
	mse_loss = (model_pred - target) ** 2
	mse_loss = mse_loss.reshape(mse_loss.shape[0], -1).mean(axis=1)  # [batch_size]

	# Apply SNR weighting if enabled
	if min_snr_gamma > 0:
		# Convert timesteps to SNR
		# For rectified flow: t ∈ [0, 1], SNR = (1-t)^2 / t^2
		t = timesteps.astype(jnp.float32) / num_train_timesteps
		snr = ((1 - t) / t.clip(1e-8)) ** 2

		# Compute Min-SNR gamma weighting
		snr_weight = jnp.minimum(snr, min_snr_gamma) / snr
		mse_loss = mse_loss * snr_weight

	# Aggregate loss
	if loss_aggregation == "mean":
		return mse_loss.mean()
	elif loss_aggregation == "sum":
		return mse_loss.sum()
	else:
		raise ValueError(f"Unknown loss aggregation: {loss_aggregation}")


def training_step(
	state: EasyDeLState,
	batch: dict,
	num_train_timesteps: int = 1000,
	prediction_type: str = "velocity",
	min_snr_gamma: float = -1.0,
	loss_config: dict | None = None,
	scheduler: optax.Schedule | None = None,
	step_partition_spec: PartitionSpec | None = None,
	gradient_accumulation_steps: int = 1,
	loss_aggregation: str = "mean",
	loss_scale: float = 1.0,
	is_train: bool = True,
) -> tuple[EasyDeLState, StepMetrics] | StepMetrics:
	"""
	Single training/evaluation step for image diffusion.

	Args:
		state: Current training state
		batch: Batch of data with 'pixel_values' and optionally 'labels'
		num_train_timesteps: Number of training timesteps
		prediction_type: Type of prediction ('velocity', 'epsilon', or 'sample')
		min_snr_gamma: Minimum SNR gamma for loss weighting
		loss_config: Loss configuration
		scheduler: Learning rate scheduler
		step_partition_spec: Partition spec for step
		gradient_accumulation_steps: Number of gradient accumulation steps
		loss_aggregation: How to aggregate loss
		loss_scale: Scaling factor for loss
		is_train: Whether this is a training step

	Returns:
		Updated state and metrics (if training) or just metrics (if eval)
	"""
	pixel_values = batch["pixel_values"]
	labels = batch.get("labels", None)
	batch_size = pixel_values.shape[0]

	def loss_fn(params):
		"""Compute loss for the current batch."""
		# Sample random timesteps
		timesteps = jax.random.randint(
			state.rng,
			(batch_size,),
			minval=0,
			maxval=num_train_timesteps,
		)

		# Sample noise
		noise_rng, model_rng = jax.random.split(state.rng)
		noise = jax.random.normal(noise_rng, pixel_values.shape)

		# Compute noisy samples using rectified flow interpolation
		# x_t = (1 - t) * noise + t * x_1
		t = timesteps[:, None, None, None].astype(jnp.float32) / num_train_timesteps
		noisy_samples = (1 - t) * noise + t * pixel_values

		# Compute target based on prediction type
		if prediction_type == "velocity":
			# Velocity: v = x_1 - noise (for rectified flow with epsilon=0)
			# More generally: v = x_1 - (1-epsilon)*noise
			target = pixel_values - noise
		elif prediction_type == "epsilon":
			target = noise
		elif prediction_type == "sample":
			target = pixel_values
		else:
			raise ValueError(f"Unknown prediction type: {prediction_type}")

		# Forward pass
		model_pred = state.call_model(
			pixel_values=noisy_samples,
			timesteps=timesteps,
			labels=labels,
			params=params,
		)

		# Compute loss
		loss = compute_rectified_flow_loss(
			model_pred=model_pred,
			target=target,
			timesteps=timesteps,
			num_train_timesteps=num_train_timesteps,
			prediction_type=prediction_type,
			min_snr_gamma=min_snr_gamma,
			loss_aggregation=loss_aggregation,
		)

		# Scale loss
		loss = loss * loss_scale

		# Metrics
		metrics = {
			"loss": loss,
			"pred_norm": jnp.sqrt(jnp.mean(model_pred**2)),
			"target_norm": jnp.sqrt(jnp.mean(target**2)),
		}

		return loss, metrics

	if is_train:
		# Compute gradients
		(loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

		# Update state
		updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
		new_params = optax.apply_updates(state.params, updates)

		# Additional metrics
		metrics.update(
			{
				"grad_norm": optax.global_norm(grads),
				"update_norm": optax.global_norm(updates),
				"param_norm": optax.global_norm(new_params),
			}
		)

		# Add learning rate if scheduler exists
		if scheduler is not None:
			metrics["learning_rate"] = scheduler(state.step)

		# Update state
		new_rng, _ = jax.random.split(state.rng)
		new_state = state.replace(
			step=state.step + 1,
			params=new_params,
			opt_state=new_opt_state,
			rng=new_rng,
		)

		return new_state, metrics
	else:
		# Evaluation mode
		loss, metrics = loss_fn(state.params)
		return metrics
