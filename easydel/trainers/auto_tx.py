# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp

import fjformer.optimizers
import jax

from easydel.infra.etils import (
	AVAILABLE_OPTIMIZERS,
	AVAILABLE_SCHEDULERS,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
)


def get_optimizer_and_scheduler(
	optimizer: AVAILABLE_OPTIMIZERS,
	scheduler: AVAILABLE_SCHEDULERS,
	steps: int,
	learning_rate: float = 1e-5,
	learning_rate_end: float = 1e-5,
	gradient_accumulation_steps: int = 1,
	weight_decay: float = 0.02,
	warmup_steps: int = 0,
	clip_grad: tp.Optional[float] = None,
	mu_dtype: tp.Optional[jax.numpy.dtype] = None,
	**kwargs,
):
	"""The get_optimizer_and_scheduler function is a helper function that returns an optimizer and scheduler
		based on the parameters passed to it.

	Args:
		optimizer: AVAILABLE_OPTIMIZERS: Choose the optimizer
		scheduler: AVAILABLE_SCHEDULERS: Determine the learning rate scheduler
		steps: int: Specify the number of steps in the training process
		learning_rate: float: Set the learning rate for the optimizer
		learning_rate_end: float: Set the final learning rate
		gradient_accumulation_steps: int: Accumulate the gradients before updating the weights
		weight_decay: float: Set the weight decay for adamw optimizer
		warmup_steps: int: Specify the number of steps to warm up the learning rate
		clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.
		mu_dtype (Optional[jax.numpy.dtype]): The dtype for the optimizer.
		**kwargs: Pass extra arguments to the optimizer

	Returns:
		A tuple of two objects: (Optimizer and scheduler)
	"""
	optimizer_kwargs = {
		"learning_rate": learning_rate,
		"learning_rate_start": learning_rate,
		"learning_rate_end": learning_rate
		if scheduler == EasyDeLSchedulers.NONE
		else learning_rate_end,
		"steps": steps,
		"gradient_accumulation_steps": gradient_accumulation_steps,
		"weight_decay": weight_decay,
		"warmup_steps": warmup_steps,
		"clip_grad": clip_grad,
		"mu_dtype": mu_dtype,
		**kwargs,
	}

	def get_optimizer_fn(optimizer, scheduler):
		if optimizer == EasyDeLOptimizers.ADAFACTOR:
			if scheduler == EasyDeLSchedulers.LINEAR:
				return fjformer.optimizers.get_adafactor_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.COSINE:
				return fjformer.optimizers.get_adafactor_with_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.NONE:
				return fjformer.optimizers.get_adafactor_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
				return fjformer.optimizers.get_adafactor_with_warmup_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
				return fjformer.optimizers.get_adafactor_with_warmup_linear_scheduler
		elif optimizer == EasyDeLOptimizers.LION:
			if scheduler == EasyDeLSchedulers.LINEAR:
				return fjformer.optimizers.get_lion_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.COSINE:
				return fjformer.optimizers.get_lion_with_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.NONE:
				return fjformer.optimizers.get_lion_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
				return fjformer.optimizers.get_lion_with_warmup_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
				return fjformer.optimizers.get_lion_with_with_warmup_linear_scheduler
		elif optimizer == EasyDeLOptimizers.ADAMW:
			if scheduler == EasyDeLSchedulers.LINEAR:
				return fjformer.optimizers.get_adamw_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.COSINE:
				return fjformer.optimizers.get_adamw_with_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.NONE:
				return fjformer.optimizers.get_adamw_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
				return fjformer.optimizers.get_adamw_with_warmup_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
				return fjformer.optimizers.get_adamw_with_warmup_linear_scheduler
		elif optimizer == EasyDeLOptimizers.RMSPROP:
			if scheduler == EasyDeLSchedulers.LINEAR:
				return fjformer.optimizers.get_rmsprop_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.COSINE:
				return fjformer.optimizers.get_rmsprop_with_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.NONE:
				return fjformer.optimizers.get_rmsprop_with_linear_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
				return fjformer.optimizers.get_rmsprop_with_warmup_cosine_scheduler
			elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
				return fjformer.optimizers.get_rmsprop_with_warmup_linear_scheduler
		return None

	optimizer_fn = get_optimizer_fn(optimizer, scheduler)
	if optimizer_fn is None:
		raise ValueError(f"Invalid optimizer {optimizer} or scheduler {scheduler}")

	tx, sc = optimizer_fn(**optimizer_kwargs)
	return tx, sc
