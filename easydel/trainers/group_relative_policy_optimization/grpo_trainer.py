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
from functools import partial

import eformer
import eformer.escale
import flax
import flax.nnx
import jax
from transformers import AutoTokenizer
from jax import numpy as jnp
from easydel.inference.vinference.vinference import vInference
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.group_relative_policy_optimization._fn import (
	get_per_token_logps,
	grpo_step,
)
from easydel.trainers.prompt_utils import (
	maybe_apply_chat_template,
	maybe_extract_prompt,
)
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput
from easydel.trainers.training_configurations import MetricsType
from easydel.utils.helpers import get_logger
from easydel.utils.traversals import deepcopy_model

from ..trainer.trainer import Trainer
from .grpo_config import GRPOConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset, IterableDataset
	from tensorflow import data

	TFDataset = data.Dataset
else:
	IterableDataset = tp.Any
	Dataset = tp.Any
	TFDataset = tp.Any

logger = get_logger(__name__)
RewardFunc = tp.Union[
	EasyDeLBaseModule,
	EasyDeLState,
	tp.Callable[[list, list], list[float]],
]


class GRPOTrainer(Trainer):
	arguments: GRPOConfig

	def __init__(
		self,
		arguments: GRPOConfig,
		vinference: vInference,
		model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]],
		reward_funcs: tp.Union[RewardFunc, list[RewardFunc]],
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]] = None,
		processing_class: tp.Optional[ProcessingClassType] = None,
		reward_processing_classes: tp.Optional[ProcessingClassType] = None,
	):
		# fmt:off
		# caused by OCD
		assert arguments is not None, "You Have to pass arguments that will be used for training but you have passed `arguments=None`"
		assert isinstance(arguments, GRPOConfig), f"arguments type must be `GRPOConfig` but got {type(arguments)}"
		assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."
		# fmt:on
		self.arguments = arguments
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class

		if not isinstance(model, EasyDeLState):
			model = model.to_state()
		self.ref_state = deepcopy_model(model=model)
		if processing_class is None:
			processing_class = AutoTokenizer.from_pretrained(
				model.model.config._name_or_path,
				padding_side="left",
			)
		if not isinstance(reward_funcs, list):
			reward_funcs = [reward_funcs]
		self.reward_funcs = reward_funcs
		if reward_processing_classes is None:
			reward_processing_classes = [None] * len(reward_funcs)
		elif not isinstance(reward_processing_classes, list):
			reward_processing_classes = [reward_processing_classes]
		else:
			if len(reward_processing_classes) != len(reward_funcs):
				raise ValueError(
					"The number of reward processing classes must match the number of reward functions."
				)

		for i, (reward_processing_class, reward_func) in enumerate(
			zip(reward_processing_classes, reward_funcs)
		):
			if isinstance(reward_func, (EasyDeLBaseModule, EasyDeLState)):
				if isinstance(reward_func, EasyDeLBaseModule):
					reward_func = reward_func.to_state()
				if reward_processing_class is None:
					reward_processing_class = AutoTokenizer.from_pretrained(
						reward_func.model.config._name_or_path
					)
				if reward_processing_class.pad_token_id is None:
					reward_processing_class.pad_token = reward_processing_class.eos_token

				reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
				reward_processing_classes[i] = reward_processing_class
				reward_funcs[i] = reward_func

		self.vinference = vinference
		self.num_return_sequences = vinference.generation_config.num_return_sequences
		self.eos_token_id = vinference.generation_config.eos_token_id
		self.pad_token_id = vinference.generation_config.pad_token_id
		self.reward_processing_classes = reward_processing_classes
		self.reward_funcs = reward_funcs
		self.arguments = arguments
		self.processing_class = processing_class

		if train_dataset is not None:
			train_dataset = self._prepare_dataset(
				dataset=train_dataset,
				processing_class=processing_class,
				arguments=arguments,
				dataset_name="train",
			)
		if eval_dataset is not None:
			eval_dataset = self._prepare_dataset(
				dataset=eval_dataset,
				processing_class=processing_class,
				arguments=arguments,
				dataset_name="eval",
			)
		super().__init__(
			model_state=model,
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			data_collator=None,
		)

	def _prepare_dataset(
		self,
		dataset: tp.Union[Dataset, IterableDataset],
		processing_class: ProcessingClassType,
		arguments: GRPOConfig,
		dataset_name: str,
	) -> tp.Union[Dataset, IterableDataset]:
		map_kwargs = {"writer_batch_size": 10}
		from datasets import Dataset

		if isinstance(dataset, Dataset):
			map_kwargs["num_proc"] = arguments.dataset_num_proc

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
		dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
		dataset = dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={
				"tokenizer": processing_class,
				"tools": arguments.tools,
			},
			**map_kwargs,
		)

		def _tokenize(example):
			return processing_class(
				example["prompt"],
				return_tensors="np",
				padding="max_length",
				padding_side="left",
				max_length=arguments.max_prompt_length,
				truncation=True,
				add_special_tokens=False,
			)

		dataset = dataset.map(
			_tokenize,
			batched=True,
			num_proc=arguments.dataset_num_proc,
		)
		return dataset

	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method sets up the necessary functions for training and evaluation, including:
		    - Initialization of the model state.
		    - Sharding of the model parameters and optimizer state.
		    - JIT-compilation of the training and evaluation step functions.

		Returns:
		    TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
		"""
		mesh = self.model.mesh

		empty_sharding = jax.sharding.NamedSharding(
			spec=jax.sharding.PartitionSpec(),
			mesh=mesh,
		)
		sharded_training_step_function = jax.jit(
			partial(
				grpo_step,
				eos_token_id=self.eos_token_id,
				num_generations=self.num_return_sequences,
				beta=self.arguments.beta,
				learning_rate_fn=self.scheduler,
				partition_spec=self.arguments.step_partition_spec,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
				loss_config=self.arguments.loss_config,
				reward_funcs=self.reward_funcs,
				is_training=True,
			),
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=(self.state_shardings, empty_sharding),
			donate_argnums=(0,),
			static_argnames=[
				"eos_token_id",
				"num_generations",
				"beta",
				"learning_rate_fn",
				"partition_spec",
				"gradient_accumulation_steps",
				"loss_config",
				"reward_funcs",
				"is_training",
			],
		)

		sharded_evaluation_step_function = jax.jit(
			partial(
				grpo_step,
				eos_token_id=self.eos_token_id,
				num_generations=self.num_return_sequences,
				beta=self.arguments.beta,
				learning_rate_fn=self.scheduler,
				partition_spec=self.arguments.step_partition_spec,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
				loss_config=self.arguments.loss_config,
				reward_funcs=self.reward_funcs,
				is_training=False,
			),
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=(empty_sharding),
			static_argnames=[
				"eos_token_id",
				"num_generations",
				"beta",
				"learning_rate_fn",
				"partition_spec",
				"gradient_accumulation_steps",
				"loss_config",
				"reward_funcs",
				"is_training",
			],
		)

		def _compute_refmodel_logps(graphtree, graphother, batch, graphdef):
			batch = eformer.escale.with_sharding_constraint(
				arr=batch,
				sharding=self.arguments.step_partition_spec,
			)
			apply = flax.nnx.merge(graphdef, graphtree, graphother)
			return get_per_token_logps(
				apply,
				batch["prompt_completion_ids"],
				batch["input_ids"].shape[-1],
			)

		self.compute_refmodel_logps = jax.jit(
			partial(
				_compute_refmodel_logps,
				graphdef=self.model_state.graphdef,
			),
			static_argnames=["graphdef"],
			in_shardings=(
				self.model_state.shardings.graphstate,
				self.model_state.shardings.graphother,
				empty_sharding,
			),
			out_shardings=empty_sharding,
		)

		self.arguments.ensure_checkpoint_path()
		checkpoint_manager = self.arguments.get_streaming_checkpointer()

		return TrainerConfigureFunctionOutput(
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
		)

	def _preprocess_batch_input(
		self,
		state: EasyDeLState,
		batch: tp.Dict[str, jax.Array],
		is_train: bool,
	) -> tp.Dict[str, jax.Array]:
		for output in self.vinference.generate(
			input_ids=batch["input_ids"],
			attention_mask=batch["attention_mask"],
			graphother=state.graphother,
			graphstate=state.graphstate,
		):
			...
		sequences = output.sequences
		completion_ids = sequences[..., output.padded_length :]
		is_eos = completion_ids == self.vinference.generation_config.eos_token_id
		eos_idx = jnp.full((is_eos.shape[0],), is_eos.shape[1])
		has_eos = is_eos.any(axis=1)
		completion_mask = (
			(jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
			<= jnp.where(
				has_eos,
				jnp.argmax(is_eos.astype(jnp.int32), axis=1),
				eos_idx,
			)[:, None]
		).astype(jnp.int32)
		del output  # free kv memory
		batch["prompt_completion_ids"] = sequences
		batch["prompt_completion_mask"] = completion_mask

		token_logps = self.compute_refmodel_logps(
			self.ref_state.graphstate,
			self.ref_state.graphother,
			batch,
		)
		batch["ref_per_token_logps"] = token_logps

		return batch

	def on_step_end(
		self,
		state: EasyDeLState,
		metrics: MetricsType,
		step: int,
	) -> tp.Tuple[EasyDeLState, MetricsType]:
		"""hook process to call in start of the step."""

		if (
			self.arguments.sync_ref_model
			and self.ref_state is not None
			and (step % self.arguments.ref_model_sync_steps == 0)
		):
			self.ref_state = self.ref_state.replace(
				graphstate=deepcopy_model(state.graphstate)
			)
		return state, metrics
