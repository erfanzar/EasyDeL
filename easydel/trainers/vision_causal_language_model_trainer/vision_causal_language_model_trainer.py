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

import copy
import os
import time
import typing
from typing import Callable, Mapping, Optional

import chex
import flax
import jax
import termcolor
from fjformer import (
	make_shard_and_gather_fns,
	match_partition_rules,
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.trainers.base_trainer import TrainerConfigureFunctionOutput
from easydel.trainers.causal_language_model_trainer import CausalLanguageModelTrainer
from easydel.trainers.vision_causal_language_model_trainer.functions import (
	VisionCausalLanguageModelStepOutput,
	create_vision_casual_language_model_evaluation_step,
	create_vision_casual_language_model_train_step,
)
from easydel.trainers.vision_causal_language_model_trainer.modelling_output import (
	VisionCausalLMTrainerOutput,
)

logger = get_logger(__name__)


class VisionCausalLanguageModelTrainer(CausalLanguageModelTrainer):
	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
	) -> Callable:
		def collate_fn(batch):
			results = {}
			corrected_sequence = None
			for key in batch[0].keys():
				if truncation_mode == "keep_end":
					corrected_sequence = [
						jnp.array(f[key])[..., -max_sequence_length:] for f in batch
					]
				else:
					corrected_sequence = [
						jnp.array(f[key])[..., :max_sequence_length] for f in batch
					]
				results[key] = jnp.stack(corrected_sequence).reshape(
					-1, corrected_sequence[0].shape[-1]
				)
			return results

		return collate_fn

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

		def initialize_state_function():
			initialized_parameters = self.model.init_weights(
				jax.random.PRNGKey(0), self.arguments.init_input_shape
			)

			if self.arguments.dtype == jnp.bfloat16:
				initialized_parameters = self.model.to_bf16(initialized_parameters)
			elif self.arguments.dtype == jnp.float16:
				initialized_parameters = self.model.to_fp16(initialized_parameters)

			tx = self.tx
			parameters = flax.core.freeze({"params": initialized_parameters})
			tx_init = copy.deepcopy(self.arguments.optimizer_kwargs)

			if self.rapture is not None:
				lora_parameters = self.lora_parameters
				if self.arguments.dtype == jnp.bfloat16:
					lora_parameters = self.model.to_bf16(lora_parameters)
				elif self.arguments.dtype == jnp.float16:
					lora_parameters = self.model.to_fp16(lora_parameters)

				return EasyDeLState(
					step=0,
					apply_fn=self.lora_apply_fn,
					params=lora_parameters,
					tx=self.lora_tx,
					opt_state=self.lora_opt_state,
					tx_init=EasyDeLState.safe_dict(tx_init),
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.lora_model,
					module_config=self.model.config,
					module_config_args=None,
				)
			else:
				return EasyDeLState.create(
					tx=tx,
					params=parameters,
					apply_fn=self.model.__call__,
					module_config=copy.deepcopy(self.model.config),
					tx_init=tx_init,
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.model,
					module_config_args=None,
				)

		def create_state_from_params_function(parameters):
			if self.rapture is None:
				return EasyDeLState.create(
					tx=self.tx,
					params=parameters,
					apply_fn=self.model.__call__,
					module_config=copy.deepcopy(self.model.config),
					tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.model,
					module_config_args=None,
				)
			else:
				return EasyDeLState(
					step=0,
					apply_fn=self.lora_apply_fn,
					params=parameters,
					tx=self.lora_tx,
					opt_state=self.lora_opt_state,
					tx_init=EasyDeLState.safe_dict(
						copy.deepcopy(self.arguments.optimizer_kwargs)
					),
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.lora_model,
					module_config=self.model.config,
					module_config_args=None,
				)

		state_shape = jax.eval_shape(initialize_state_function)

		state_partition_spec = match_partition_rules(
			(
				self.config.get_partition_rules(
					fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
				)
				if self.arguments.custom_rule is None
				else self.arguments.custom_rule
			),
			state_shape,
		)
		spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
		empty_sharding = jax.sharding.NamedSharding(
			spec=PartitionSpec(), mesh=self.arguments.get_mesh()
		)
		create_sharded_state_from_params_function = jax.jit(
			create_state_from_params_function,
			in_shardings=(spec_named_sharding.params,),
			out_shardings=spec_named_sharding,
			donate_argnums=(0,),
		)
		sharded_train_step_function = jax.jit(
			create_vision_casual_language_model_train_step(
				self.arguments.step_partition_spec
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(spec_named_sharding, empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		sharded_eval_step_function = jax.jit(
			create_vision_casual_language_model_evaluation_step(
				self.arguments.step_partition_spec
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		mesh = self.arguments.get_mesh()
		self.arguments.ensure_checkpoint_path()
		checkpoint_manager = self.arguments.get_streaming_checkpointer()
		self.state_partition_spec = state_partition_spec
		self.state_named_sharding = spec_named_sharding
		self.state_shape = state_shape

		return TrainerConfigureFunctionOutput(
			create_sharded_state_from_params_function=create_sharded_state_from_params_function,
			sharded_train_step_function=sharded_train_step_function,
			sharded_eval_step_function=sharded_eval_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
			initialize_state_function=initialize_state_function,
		)

	def initialize_state(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> typing.Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
		if (
			model_parameters is None
			and state is None
			and self.rapture is None
			and self.checkpoint_path is None
		):
			raise RuntimeError(
				"You are passing `model_parameters=None`, `state=None`, and `checkpoint_path=None` and also you are not"
				" using LoRA, if you are "
				"Using LoRA make sure to pass parameters and Rapture Config correctly otherwise pass the "
				"model_parameters or state."
			)
		if model_parameters is None and state is None:
			model_parameters = self.lora_parameters
		with self.mesh:
			shard_fns, gather_fns = make_shard_and_gather_fns(
				self.state_partition_spec, mesh=self.mesh
			)
			if state is not None:
				sharded_state = state
				if sharded_state.opt_state is None:
					logger.info("Optimizer State is not Found!, initializing one.")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = sharded_state.init_opt_state()
			elif self.finetune:
				if model_parameters is None and self.checkpoint_path is not None:
					logger.info(f"Loading Model From {self.checkpoint_path}")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = EasyDeLState.load_state(
							verbose=self.arguments.verbose,
							state_shard_fns=shard_fns,
							init_optimizer_state=True,
							checkpoint_path=self.checkpoint_path,
							input_shape=self.arguments.init_input_shape,
						)
						# sharded_state = sharded_state.replace(
						#     tx=self.tx,
						# )
						state_shape = jax.eval_shape(lambda: sharded_state)
						state_partition_spec = match_partition_rules(
							(
								self.config.get_partition_rules(
									fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
								)
								if self.arguments.custom_rule is None
								else self.arguments.custom_rule
							),
							state_shape,
						)
						spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
						empty_sharding = jax.sharding.NamedSharding(
							spec=PartitionSpec(), mesh=self.arguments.get_mesh()
						)
						sharded_train_step_function = jax.jit(
							create_vision_casual_language_model_train_step(
								partition_spec=self.arguments.step_partition_spec,
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								spec_named_sharding,
								empty_sharding,
								empty_sharding,
							),
							donate_argnums=(0, 0),
						)

						sharded_eval_step_function = jax.jit(
							create_vision_casual_language_model_evaluation_step(
								self.arguments.step_partition_spec
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(empty_sharding, empty_sharding),
							donate_argnums=(0, 0),
						)

						self.state_partition_spec = state_partition_spec
						self.state_named_sharding = spec_named_sharding
						self.state_shape = state_shape
						self.sharded_train_step_function = sharded_train_step_function
						self.sharded_eval_step_function = sharded_eval_step_function

					if self.arguments.remove_ckpt_after_load:
						os.remove(self.checkpoint_path)
				elif model_parameters is not None and self.checkpoint_path is None:
					if not isinstance(model_parameters, flax.core.FrozenDict):
						logger.warn(
							"Model Parameters should be like FrozenDict({'params': params}) make sure to "
							"pass as type FrozenDict in case of not getting UnExcepted Errors ",
						)
					sharded_state = self.create_sharded_state_from_params_function(
						model_parameters
					)
				elif model_parameters is not None and self.checkpoint_path is not None:
					raise EasyDeLTimerError(
						"You can't pass `model_parameters` and `checkpoint_path` at same time"
					)
				else:
					raise EasyDeLTimerError(
						"You should pass `model_parameters` or `checkpoint_path` to trainer in order to load model"
					)
			else:
				sharded_state = self.initialize_state_function()
			self.sharded_state = sharded_state
			return sharded_state, shard_fns, gather_fns

	def train(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> VisionCausalLMTrainerOutput:
		"""The train function is the main function of this module.
		It takes a model_parameters argument which can be used to load a pretrained model and finetune it.
		The train function returns an TrainerOutput object that contains the last saved file name, predict func,
		train state, mesh and checkpoint streamer.

		Args:
		    self: Make the class methods aware of other methods and
		        attributes within the class
		    model_parameters: flax.core.FrozenDict: Load a pre-trained
		        model
		    state: Optional[EasyDeLState]: Ready to Use State

		Returns:
		    An object of type "TrainerOutput"
		"""

		def count_model_parameters(_p):
			termcolor.cprint(
				f"Model Contain {sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0]) / 1e9} "
				f"Billion Parameters",
				color="red",
				force_color=True,
			)

		checkpoint_path = "SAVING_SKIPPED"
		start_time = time.time()
		sharded_state, shard_fns, gather_fns = self.initialize_state(
			model_parameters=model_parameters, state=state
		)

		count_model_parameters(sharded_state.params)
		flops_per_device = (
			self.calculate_number_total_flops_per_device(params=sharded_state.params) / 1e12
		)
		with self.mesh:
			pbar = tqdm(total=self.max_training_steps)
			current_step = int(jax.device_get(sharded_state.step))

			loss_sum = None
			filename = None
			vision_loss_sum = None
			vision_accuracy_sum = None
			text_loss_sum = None
			text_accuracy_sum = None
			pbar.update(sharded_state.step.tolist())  # type: ignore
			learning_rates = []
			model_parameters_number = (
				sum(
					n.size
					for n in jax.tree_util.tree_flatten(flax.core.unfreeze(sharded_state.params))[
						0
					]
				)
				/ 1e9
			)
			self.arguments.log_metrics()(
				{"Number of Model Parameters (Billion)": model_parameters_number}, 0
			)
			try:
				for epoch in range(self.arguments.num_train_epochs):
					for batch in self.dataloader_train:
						if (
							self.arguments.step_start_point is not None
							and self.arguments.step_start_point > current_step
						):
							pbar.update(1)
						elif current_step < self.max_training_steps:
							for ssb in self.arguments.ids_to_pop_from_dataset:
								_ = batch.pop(ssb, None)
							time_start = time.time()

							outputs_and_metrics: tuple[
								EasyDeLState,
								chex.Array,
								VisionCausalLanguageModelStepOutput,
							] = self.sharded_train_step_function(sharded_state, batch)

							sharded_state, loss, information_and_accuracies = outputs_and_metrics

							loss.block_until_ready()
							total_time = time.time() - time_start
							flops = flops_per_device / total_time
							loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
							vision_loss = information_and_accuracies.vision_loss
							vision_accuracy = information_and_accuracies.vision_accuracy
							text_loss = information_and_accuracies.text_loss
							text_accuracy = information_and_accuracies.text_accuracy

							loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
							vision_accuracy_sum = (
								vision_accuracy.tolist()
								if vision_accuracy_sum is None
								else (vision_accuracy_sum + vision_accuracy)
							)
							vision_loss_sum = (
								vision_loss.tolist()
								if vision_loss_sum is None
								else (vision_loss_sum + vision_loss)
							)
							text_loss_sum = (
								text_loss.tolist()
								if text_loss_sum is None
								else text_loss_sum + text_loss
							)
							text_accuracy_sum = (
								text_accuracy.tolist()
								if text_accuracy_sum is None
								else (text_accuracy_sum + text_accuracy)
							)
							learning_rates.append(self.scheduler(current_step).tolist())
							pbar.update(1)

							trained_tokens = jnp.multiply(
								self.arguments.max_sequence_length,
								jnp.multiply(current_step, self.arguments.total_batch_size),
							)

							total_roved_steps = (current_step + 1) - self.arguments.step_start_point

							with jax.spmd_mode("allow_all"):
								train_metrics = {
									"train/loss": loss.tolist(),
									"train/mean_loss": loss_sum / total_roved_steps,
									"train/vision_accuracy": vision_accuracy,
									"train/vision_loss": vision_loss,
									"train/text_loss": text_loss,
									"train/text_accuracy": text_accuracy,
									"train/mean_vision_accuracy": vision_accuracy_sum / total_roved_steps,
									"train/mean_vision_loss": vision_loss_sum / total_roved_steps,
									"train/mean_text_loss": text_loss_sum / total_roved_steps,
									"train/mean_text_accuracy": text_accuracy_sum / total_roved_steps,
									"train/learning_rate": self.scheduler(current_step).tolist(),
									"train/step": current_step,
									"train/step_time": total_time,
									"train/perplexity": jnp.exp(loss).tolist(),
									"train/trained_tokens": trained_tokens,
									"train/epoch": epoch,
									"train/TFLOPs": flops,
								}

								log_metrics = copy.deepcopy(train_metrics)
								train_metrics.update(**self.arguments._captured_memory)
								self.arguments.log_metrics(
									metrics=train_metrics,
									step=current_step,
								)

							pbar.set_postfix(
								**{k.replace("train/", ""): v for k, v in log_metrics.items()}
							)
							self.arguments.ensure_training_time(time.time() - start_time)
						else:
							break

						current_step += 1
						if (
							self.arguments.save_steps is not None
							and current_step % self.arguments.save_steps == 0
						):
							if self.rapture is None:
								filename = self._save_state(
									state=sharded_state,
									gather_fns=gather_fns,
									milestone=True,
								)
								checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"
							else:
								print(
									termcolor.colored("Info : ", color="red", force_color=True),
									termcolor.colored(
										"You can not use `save_steps` while using LoRA "
										"right now. this action will be skipped",
										color="white",
										force_color=True,
									),
								)
			except KeyboardInterrupt:
				termcolor.cprint(
					"KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
					color="cyan",
					force_color=True,
				)

			except EasyDeLTimerError:
				termcolor.cprint(
					"Training reached out maximum training Time Killing training Process "
					"and Will return Current State of the Model with Parameters.",
					color="cyan",
					force_color=True,
				)
			if self.arguments.merge_lora_rapture_parameters and self.rapture is not None:
				print(
					termcolor.colored("Info : ", color="red", force_color=True),
					termcolor.colored(
						"Merging LoRA Parameters.", color="white", force_color=True
					),
				)
				sharded_state = sharded_state.replace(
					params=self.rapture.merge_parameters(sharded_state.params)
				)
			output = VisionCausalLMTrainerOutput(
				state=sharded_state,
				mesh=self.mesh,
				shard_fns=shard_fns,
				gather_fns=gather_fns,
				checkpoint_manager=self.checkpoint_manager,
			)
			if self.arguments.save_steps is None and self.arguments.do_last_save:
				shard_fns, gather_fns = make_shard_and_gather_fns(
					match_partition_rules(
						(
							self.config.get_partition_rules(
								fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
							)
							if self.arguments.custom_rule is None
							else self.arguments.custom_rule
						),
						jax.eval_shape(lambda: sharded_state),
					),
					mesh=self.mesh,
				)  # You have to re-init the new shard and gather functions in order to be able to skip LoRA weight
				# crashing errors and saving errors
				filename = self._save_state(state=sharded_state, gather_fns=gather_fns)
				checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

			if self.arguments.do_eval:
				for _ in self.eval(sharded_state):
					...

			output.checkpoint_path = checkpoint_path
			output.last_save_file_name = filename
			self.finish()

			return output

	def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
		"""
		Evaluates the VCLM model using the provided model state.

		This method iterates over the evaluation dataset, performs evaluation steps,
		calculates metrics, logs metrics, and yields a dictionary of metrics for each step.

		Args:
		    model_state (EasyDeLState): The EasyDeLState object containing the model parameters
		                                and other relevant information.

		Yields:
		    Iterator[dict]: An iterator that yields a dictionary of evaluation metrics for each step.

		Raises:
		    AssertionError: If the evaluation dataset is not set.
		"""
		assert (
			self.dataloader_eval is not None
		), "`dataloader_eval` is required by evaluator function."
		with self.mesh:
			pbar = tqdm(total=self.max_evaluation_steps)
			pbar.set_description("Evaluating")
			current_step = 0
			loss_sum = None
			vision_loss_sum = None
			vision_accuracy_sum = None
			text_loss_sum = None
			text_accuracy_sum = None

			flops_per_device = (
				self.calculate_number_total_flops_per_device(params=model_state.params) / 1e12
			)
			try:
				for batch in self.dataloader_eval:
					time_start = time.time()
					for key in self.arguments.ids_to_pop_from_dataset:
						_ = batch.pop(key, None)

					metrics: tuple[chex.Array, VisionCausalLanguageModelStepOutput] = (
						self.sharded_eval_step_function(model_state, batch)
					)
					total_time = time.time() - time_start
					flops = flops_per_device / total_time
					loss, information_and_accuracies = metrics

					vision_loss = information_and_accuracies.vision_loss
					vision_accuracy = information_and_accuracies.vision_accuracy
					text_loss = information_and_accuracies.text_loss
					text_accuracy = information_and_accuracies.text_accuracy

					loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
					vision_accuracy_sum = (
						vision_accuracy.tolist()
						if vision_accuracy_sum is None
						else (vision_accuracy_sum + vision_accuracy)
					)
					vision_loss_sum = (
						vision_loss.tolist()
						if vision_loss_sum is None
						else vision_loss_sum + vision_loss
					)
					text_loss_sum = (
						text_loss.tolist() if text_loss_sum is None else text_loss_sum + text_loss
					)
					text_accuracy_sum = (
						text_accuracy.tolist()
						if text_accuracy_sum is None
						else (text_accuracy_sum + text_accuracy)
					)

					total_roved_steps = (current_step + 1) - self.arguments.step_start_point

					eval_metrics = {
						"eval/loss": loss.tolist(),
						"eval/mean_loss": loss_sum / total_roved_steps,
						"eval/vision_accuracy": vision_accuracy,
						"eval/vision_loss": vision_loss,
						"eval/text_loss": text_loss,
						"eval/text_accuracy": text_accuracy,
						"eval/mean_vision_accuracy": vision_accuracy_sum / total_roved_steps,
						"eval/mean_vision_loss": vision_loss_sum / total_roved_steps,
						"eval/mean_text_loss": text_loss_sum / total_roved_steps,
						"eval/mean_text_accuracy": text_accuracy_sum / total_roved_steps,
						"eval/step": current_step,
						"eval/step_time": total_time,
						"eval/perplexity": jnp.exp(loss).tolist(),
						"eval/TFLOPs": flops,
					}
					log_metrics = copy.deepcopy(eval_metrics)
					eval_metrics.update(**self.arguments._captured_memory)

					self.arguments.log_metrics(metrics=eval_metrics, step=current_step)
					pbar.update(1)
					pbar.set_postfix(
						**{k.replace("eval/", ""): v for k, v in log_metrics.items()}
					)
					yield eval_metrics
					current_step += 1
			except KeyboardInterrupt:
				termcolor.cprint(
					"KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
					color="cyan",
					force_color=True,
				)
