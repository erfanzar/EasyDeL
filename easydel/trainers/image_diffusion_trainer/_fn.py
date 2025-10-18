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
from eformer.escale import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics

from ..training_utils import (
	make_assertions_and_get_sizes,
	minibatch_call,
	update_metrics,
	update_state_respectfully,
)


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
	loss_config=None,
	scheduler=None,
	step_partition_spec: PartitionSpec | None = None,
	gradient_accumulation_steps: int = 1,
	loss_aggregation: str = "mean",
	loss_scale: float = 1.0,
	is_train: bool = True,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
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
	if "rng_keys" not in batch:
		raise ValueError("Batch passed to image diffusion step must contain 'rng_keys'.")

	_batch_without_rng = {k: v for k, v in batch.items() if k != "rng_keys"}
	_, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=_batch_without_rng,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=step_partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	def loss_fn(params, minibatch):
		pixel_values = minibatch["pixel_values"].astype(jnp.float32)
		labels = minibatch.get("labels", None)
		rng_keys = minibatch["rng_keys"]

		# Split RNG keys per-example for sampling noise and timesteps.
		def split_example(key):
			return jax.random.split(key, 3)

		subkeys = jax.vmap(split_example)(rng_keys)
		sample_keys = subkeys[:, 0]
		noise_keys = subkeys[:, 1]
		timestep_keys = subkeys[:, 2]

		latents_shape = pixel_values.shape[1:]

		def sample_noise(key):
			return jax.random.normal(key, latents_shape)

		noise = jax.vmap(sample_noise)(noise_keys).astype(pixel_values.dtype)
		timesteps = jax.vmap(lambda key: jax.random.randint(key, (), 0, num_train_timesteps))(timestep_keys)
		t = timesteps.astype(jnp.float32)[:, None, None, None] / num_train_timesteps
		noisy_samples = (1.0 - t) * noise + t * pixel_values

		if prediction_type == "velocity":
			target = pixel_values - noise
		elif prediction_type == "epsilon":
			target = noise
		elif prediction_type == "sample":
			target = pixel_values
		else:
			raise ValueError(f"Unknown prediction type: {prediction_type}")

		module = state.merge(params)
		predictions = module(
			pixel_values=noisy_samples,
			timesteps=timesteps,
			labels=labels,
			return_dict=True,
		).last_hidden_state.astype(pixel_values.dtype)

		loss = compute_rectified_flow_loss(
			model_pred=predictions,
			target=target,
			timesteps=timesteps,
			num_train_timesteps=num_train_timesteps,
			prediction_type=prediction_type,
			min_snr_gamma=min_snr_gamma,
			loss_aggregation=loss_aggregation,
		)
		loss = loss * loss_scale

		other_metrics = {
			"pred_norm": jnp.sqrt(jnp.mean(predictions**2)),
			"target_norm": jnp.sqrt(jnp.mean(target**2)),
		}
		return loss, LossMetrics(loss=loss, other_metrics=other_metrics)

	if is_train:
		gradients, metrics = minibatch_call(
			state=state,
			batch=batch,
			minibatch_size=minibatch_size,
			grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
		)
		metrics = update_metrics(
			metrics=metrics,
			learning_rate_fn=scheduler,
			step=state.step,
			gradients=gradients,
		)
		state = update_state_respectfully(
			state=state,
			gradients=gradients,
			loss_config=loss_config,
			metrics=metrics,
		)
		return state, metrics

	_, metrics = loss_fn(state.graphstate, batch)
	return metrics
