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

"""Image Diffusion Trainer for EasyDeL."""

import random
import typing as tp

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils.helpers import get_logger
from easydel.trainers.trainer import Trainer
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput

from ._fn import training_step
from .image_diffusion_config import ImageDiffusionConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset
else:
	Dataset = tp.Any

logger = get_logger(__name__)


class ImageDiffusionTrainer(Trainer):
	"""
	Trainer for image diffusion models using rectified flow.

	This trainer implements training for DiT (Diffusion Transformer) and other
	transformer-based image diffusion models using the rectified flow framework.

	Args:
		arguments: Training configuration
		model_state: Optional pre-initialized model state
		model: Optional EasyDeL model instance
		train_dataset: Training dataset
		eval_dataset: Evaluation dataset(s)
		seed: Random seed
		dtype: Data type for computations
	"""

	arguments: ImageDiffusionConfig

	def __init__(
		self,
		arguments: ImageDiffusionConfig,
		model_state: EasyDeLState | None = None,
		model: EasyDeLBaseModule | None = None,
		train_dataset: Dataset | None = None,
		eval_dataset: Dataset | dict[str, Dataset] | None = None,
		seed: int | None = None,
		dtype: jnp.dtype = None,
	):
		assert isinstance(arguments, ImageDiffusionConfig), "passed argument must be `ImageDiffusionConfig`."
		assert model is not None or model_state is not None, "You must pass a `model` to the ImageDiffusionTrainer."

		_model = model
		if _model is None:
			_model = model_state.model

		if seed is None:
			seed = random.randint(0, 2**31 - 1)
		self.key = jax.random.PRNGKey(seed)

		self.arguments = arguments

		super().__init__(
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			model_state=model_state,
			model=model,
			data_collator=self.prepare_batch,
		)
		logger.info("Initialized ImageDiffusionTrainer")

	def prepare_batch(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
		"""
		Prepare a batch for training by adding random key.

		Args:
			batch: Input batch dictionary

		Returns:
			Batch with additional fields
		"""
		self.key, curr_key = jax.random.split(self.key, 2)
		batch["rng_key"] = curr_key
		return batch

	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		Returns:
			TrainerConfigureFunctionOutput: Configured functions and settings
		"""
		logger.info("Configuring functions for ImageDiffusionTrainer...")
		mesh = self.model.mesh

		empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

		self._train_shared_fn_static_args = (
			self.arguments.num_train_timesteps,
			self.arguments.prediction_type,
			self.arguments.min_snr_gamma,
			self.arguments.loss_config,
			self.scheduler,
			self.arguments.step_partition_spec,
			self.arguments.gradient_accumulation_steps,
			self.arguments.loss_aggregation,
			self.arguments.loss_scale,
			True,  # is_train
		)
		static_argnames = tuple(range(2, 12))  # Arguments 2-11 are static

		sharded_training_step_function = jax.jit(
			training_step,
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=(self.state_shardings, empty_sharding),
			donate_argnums=(0,),
			static_argnums=static_argnames,
		)

		self._eval_shared_fn_static_args = self._train_shared_fn_static_args[:-1] + (False,)  # is_train=False

		sharded_evaluation_step_function = jax.jit(
			training_step,
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=empty_sharding,
			static_argnums=static_argnames,
		)

		self.arguments.ensure_checkpoint_path()

		logger.info("Functions configured successfully.")
		return TrainerConfigureFunctionOutput(
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=self.arguments.get_streaming_checkpointer(),
		)
