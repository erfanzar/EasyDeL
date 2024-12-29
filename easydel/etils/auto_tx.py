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

from .etils import (
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
	    **kwargs: Pass extra arguments to the optimizer

	Returns:
	    A tuple of two objects: (Optimizer and scheduler)
	"""
	if optimizer == EasyDeLOptimizers.ADAFACTOR:
		if scheduler == EasyDeLSchedulers.LINEAR:
			tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.COSINE:
			tx, sc = fjformer.optimizers.get_adafactor_with_cosine_scheduler(
				learning_rate=learning_rate,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.NONE:
			tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
			tx, sc = fjformer.optimizers.get_adafactor_with_warmup_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				weight_decay=weight_decay,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
			tx, sc = fjformer.optimizers.get_adafactor_with_warmup_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)

		else:
			raise ValueError("seems like you have choose wrong type or unavailable scheduler")
	elif optimizer == EasyDeLOptimizers.LION:
		if scheduler == EasyDeLSchedulers.LINEAR:
			tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.COSINE:
			tx, sc = fjformer.optimizers.get_lion_with_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.NONE:
			tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
			tx, sc = fjformer.optimizers.get_lion_with_warmup_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)

		elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
			tx, sc = fjformer.optimizers.get_lion_with_with_warmup_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		else:
			raise ValueError("seems like you have choose wrong type or unavailable scheduler")
	elif optimizer == EasyDeLOptimizers.ADAMW:
		if scheduler == EasyDeLSchedulers.LINEAR:
			tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				clip_grad=clip_grad,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.COSINE:
			tx, sc = fjformer.optimizers.get_adamw_with_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				weight_decay=weight_decay,
				clip_grad=clip_grad,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.NONE:
			tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				clip_grad=clip_grad,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
			tx, sc = fjformer.optimizers.get_adamw_with_warmup_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				weight_decay=weight_decay,
				gradient_accumulation_steps=gradient_accumulation_steps,
				clip_grad=clip_grad,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
			tx, sc = fjformer.optimizers.get_adamw_with_warmup_linear_scheduler(
				learning_rate_start=learning_rate,
				steps=steps,
				weight_decay=weight_decay,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				clip_grad=clip_grad,
				**kwargs,
			)
		else:
			raise ValueError("seems like you have choose wrong type or unavailable scheduler")
	elif optimizer == EasyDeLOptimizers.RMSPROP:
		if scheduler == EasyDeLSchedulers.LINEAR:
			tx, sc = fjformer.optimizers.get_rmsprop_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				gradient_accumulation_steps=gradient_accumulation_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.COSINE:
			tx, sc = fjformer.optimizers.get_rmsprop_with_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				weight_decay=weight_decay,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.NONE:
			tx, sc = fjformer.optimizers.get_rmsprop_with_linear_scheduler(
				learning_rate_start=learning_rate,
				learning_rate_end=learning_rate,
				gradient_accumulation_steps=gradient_accumulation_steps,
				steps=steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_COSINE:
			tx, sc = fjformer.optimizers.get_rmsprop_with_warmup_cosine_scheduler(
				learning_rate=learning_rate,
				learning_rate_end=learning_rate_end,
				steps=steps,
				weight_decay=weight_decay,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		elif scheduler == EasyDeLSchedulers.WARM_UP_LINEAR:
			tx, sc = fjformer.optimizers.get_rmsprop_with_warmup_linear_scheduler(
				learning_rate_start=learning_rate,
				steps=steps,
				weight_decay=weight_decay,
				learning_rate_end=learning_rate_end,
				gradient_accumulation_steps=gradient_accumulation_steps,
				warmup_steps=warmup_steps,
				**kwargs,
			)
		else:
			raise ValueError("seems like you have choose wrong type or unavailable scheduler")
	else:
		raise ValueError(
			f"seems like you have choose wrong type or unavailable optimizer {optimizer} and scheduler {scheduler}"
		)
	return tx, sc
