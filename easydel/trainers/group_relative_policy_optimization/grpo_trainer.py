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

import flax
import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers import AutoTokenizer

from easydel.inference.vinference.vinference import vInference
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.group_relative_policy_optimization._fn import (
	get_per_token_logps,
	grpo_step,
)
from easydel.trainers.prompt_utils import (
	apply_chat_template,
	is_conversational,
	maybe_apply_chat_template,
	maybe_extract_prompt,
)
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput
from easydel.trainers.training_configurations import MetricsType
from easydel.utils.compiling_utils import cjit
from easydel.utils.helpers import capture_time, get_logger
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


def _fileaf(x):
	return isinstance(x, jax.Array)


def delete_tree(pytree):
	return jax.tree_util.tree_map(
		lambda x: x.delete() if isinstance(x, jax.Array) else None,
		pytree,
		is_leaf=_fileaf,
	)


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
		data_tokenize_fn: tp.Optional[tp.Callable] = None,
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
		empty_sharding = jax.sharding.NamedSharding(
			spec=jax.sharding.PartitionSpec(),
			mesh=model.model.mesh,
		)
		for i, (reward_processing_class, reward_func) in enumerate(
			zip(reward_processing_classes, reward_funcs)
		):
			if isinstance(reward_func, (EasyDeLBaseModule, EasyDeLState)):
				if isinstance(reward_func, EasyDeLBaseModule):
					reward_func = reward_func.to_state()
					sharding = reward_func.shardings

					@partial(cjit, static_argnums=(0,))
					@partial(
						jax.jit,
						static_argnums=(0,),
						in_shardings=(
							sharding.graphstate,
							sharding.graphother,
							empty_sharding,
						),
						out_shardings=empty_sharding,
					)
					def apply_fn(gd, gs, gt, batch):
						batch = with_sharding_constraint(
							arr=batch,
							sharding=self.arguments.step_partition_spec,
						)
						return nn.merge(gd, gs, gt)(**batch)

					reward_func = reward_func.replace(apply_fn=apply_fn)

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
		self.num_generations = vinference.generation_config.num_return_sequences
		self.train_is_conversational = False
		self.eval_is_conversational = False
		self.data_tokenize_fn = data_tokenize_fn
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
		if dataset_name == "train":
			self.train_is_conversational = is_conversational(dataset[0])
		else:
			self.eval_is_conversational = is_conversational(dataset[0])

		dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
		if not self.arguments.skip_apply_chat_template:
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
				return_attention_mask=True,
			)

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"tokenizing {dataset_name} dataset"
		if self.data_tokenize_fn is not None:
			dataset = dataset.map(
				self.data_tokenize_fn,
				batched=True,
				fn_kwargs={
					"tokenizer": processing_class,
					"tools": arguments.tools,
				},
				**map_kwargs,
			)
		else:
			dataset = dataset.map(
				_tokenize,
				batched=True,
				**map_kwargs,
			)
		return dataset

	def _all_gather(self, arr: jax.Array):
		return with_sharding_constraint(arr, PartitionSpec())

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
				prompt_length=self.arguments.max_prompt_length,
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
				"prompt_length",
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
				prompt_length=self.arguments.max_prompt_length,
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
				"prompt_length",
				"is_training",
			],
		)

		def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
			ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
			mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
			apply = flax.nnx.merge(graphdef, graphtree, graphother)
			return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

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

	def _make_attn_mask(self, arr):
		is_eos = arr == self.vinference.generation_config.eos_token_id
		return (
			(jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
			<= jnp.where(
				is_eos.any(axis=1),
				jnp.argmax(is_eos.astype(jnp.int32), axis=1),
				jnp.full((is_eos.shape[0],), is_eos.shape[1]),
			)[:, None]
		).astype(jnp.int32)

	def _preprocess_batch_input(
		self,
		state: EasyDeLState,
		batch: tp.Dict[str, jax.Array],
		is_train: bool,
	) -> tp.Tuple[tp.Dict[str, jax.Array], tp.Dict[str, tp.Union[float, int, str]]]:
		with capture_time() as preprocessing_time_fn:
			prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

			with capture_time() as vinference_time_fn:
				for output in self.vinference.generate(
					input_ids=prompt_ids,
					attention_mask=prompt_mask,
					graphother=state.graphother,
					graphstate=state.graphstate,
				):
					...
			vinference_time = vinference_time_fn()
			prompt_completion_ids = jnp.copy(self._all_gather(output.sequences))
			completion_ids = prompt_completion_ids[..., output.padded_length :]
			completion_mask = self._make_attn_mask(completion_ids)
			ridmask = prompt_mask.repeat(self.num_generations, 0)
			output = delete_tree(output)  # free kv memory
			del output
			
			with capture_time() as token_logps_time_fn:
				ref_per_token_logps = self.compute_refmodel_logps(
					self.ref_state.graphstate,
					self.ref_state.graphother,
					prompt_completion_ids,
					self._all_gather(jnp.concatenate([ridmask, completion_mask], -1)),
				)
			token_logps_time = token_logps_time_fn()
			prompts = self.processing_class.batch_decode(
				batch["input_ids"],
				skip_special_tokens=True,
			)
			completions_text = self.processing_class.batch_decode(
				completion_ids,
				skip_special_tokens=True,
			)

			is_conversational = (
				self.train_is_conversational if is_train else self.eval_is_conversational
			)
			if is_conversational:
				completions = [
					[{"role": "assistant", "content": completion}]
					for completion in completions_text
				]
			else:
				completions = completions_text

			rewards_per_func = jnp.zeros(
				(prompt_ids.shape[0] * self.num_generations, len(self.reward_funcs)),
				dtype="f4",
			)
			with capture_time() as rewarding_time_fn:
				for i, (reward_func, reward_processing_class) in enumerate(
					zip(
						self.reward_funcs,
						self.reward_processing_classes,
					)
				):
					if isinstance(reward_func, EasyDeLState):
						if is_conversational:
							messages = [
								{"messages": p + c}
								for p, c in zip(prompts * self.num_generations, completions)
							]
							texts = [
								apply_chat_template(x, reward_processing_class)["text"]
								for x in messages
							]
						else:
							texts = [
								p + c for p, c in zip(prompts * self.num_generations, completions)
							]

						rew = reward_func.apply_fn(
							reward_func.graphdef,
							reward_func.graphstate,
							reward_func.graphother,
							dict(
								reward_processing_class(
									texts,
									return_tensors="jax",
									padding="max_length",
									padding_side="right",
									add_special_tokens=False,
									truncation=True,
									return_attention_mask=True,
									max_length=self.arguments.max_sequence_length,
								)
							),
						).logits[:, 0]
					else:
						in_prompts = prompts * self.num_generations
						output_reward_func = reward_func(
							prompts=in_prompts,
							completions=completions,
							max_length=self.arguments.max_sequence_length,
							batch=batch,
						)
						rew = jnp.array(output_reward_func, dtype="f4")
					rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
			rewarding_time = rewarding_time_fn()
			with capture_time() as grouped_comp_time_fn:
				rewards = rewards_per_func.sum(axis=1)
				advantages = (
					rewards
					- jnp.mean(
						rewards.reshape(-1, self.num_generations),
						axis=-1,
					).repeat(self.num_generations, axis=0)
				) / (
					jnp.std(
						rewards.reshape(-1, self.num_generations),
						axis=-1,
					).repeat(self.num_generations, axis=0)
					+ 1e-4
				)
			grouped_comp_time = grouped_comp_time_fn()
		preprocessing_time = preprocessing_time_fn()
		metrics_dict = {
			"rewards": jnp.mean(rewards, -1),
			"completion_length": jnp.sum(completion_mask.sum(-1), -1),
			"grouped_comp_time": grouped_comp_time,
			"rewarding_time": rewarding_time,
			"token_logps_time": token_logps_time,
			"vinference_time": vinference_time,
			"preprocessing_time": preprocessing_time,
		}
		for i, reward_func in enumerate(self.reward_funcs):
			metrics_dict[
				getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
			] = jnp.mean(rewards_per_func[:, i])
		return (
			{
				"prompt_ids": prompt_ids,
				"prompt_mask": prompt_mask,
				"completion_ids": completion_ids,
				"completion_mask": completion_mask,
				"ref_per_token_logps": self._all_gather(ref_per_token_logps),
				"advantages": advantages,
			},
			metrics_dict,
		)

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
