
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
from typing import Callable, Mapping, Optional, Tuple

import flax
import jax
import termcolor
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from flax.core import FrozenDict
from jax import numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.trainers.base_trainer import (
	BaseTrainer,
	TrainerConfigureFunctionOutput,
)
from easydel.trainers.causal_language_model_trainer.fwd_bwd_functions import (
	create_casual_language_model_evaluation_step,
	create_casual_language_model_train_step,
)
from easydel.trainers.causal_language_model_trainer.modeling_output import (
	CausalLMTrainerOutput,
)

logger = get_logger(__name__)


class CausalLanguageModelTrainer(BaseTrainer):
	"""
	Trainer for Causal Language Models (CLMs).

	This trainer handles training, evaluation, and checkpointing of CLMs
	using JAX and EasyDeL. It supports features like sharding, gradient
	accumulation, mixed precision training, and LoRA.

	Attributes:
			(Inherited from BaseTrainer)

	Methods:
			create_collect_function(self, max_sequence_length: int, truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end") -> Callable:
					Creates a function to collect and pad/truncate batches of data.
			configure_functions(self) -> TrainerConfigureFunctionOutput:
					Configures and JIT-compiles the training and evaluation step functions.
			initialize_state(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
					Initializes the training state, either from scratch, pretrained parameters, or a checkpoint.
			train(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> CausalLMTrainerOutput:
					Trains the CLM and returns the training output.
			eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
					Evaluates the CLM and yields evaluation metrics.



	>>> import jax.lax
	>>> from easydel import (
	...   TrainArguments,
	...   CausalLanguageModelTrainer,
	...   AutoEasyDeLModelForCausalLM,
	...   EasyDeLOptimizers,
	...   EasyDeLSchedulers,
	...   EasyDeLGradientCheckPointers,
	...   PartitionAxis,
	... )
	>>> from datasets import load_dataset
	>>> import flax
	>>> from jax import numpy as jnp
	>>> from transformers import AutoTokenizer

	>>> huggingface_repo_id_or_path = (
	...   "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
	... )

	>>> max_length = 4096
	>>> dtype = jnp.bfloat16
	>>> input_shape = (1, 1)
	>>> partition_axis = PartitionAxis()
	>>> sharding_axis_dims = (1, -1, 1, 1)  # Change to 1,1,1,-1 for Sequence Sharding
	>>> model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	...   huggingface_repo_id_or_path,
	...   dtype=dtype,
	...   param_dtype=dtype,
	...   precision=jax.lax.Precision("fastest"),
	...   auto_shard_params=True,
	...   sharding_axis_dims=sharding_axis_dims,
	...   verbose_params=True,
	...   config_kwargs=dict(use_scan_mlp=False, partition_axis=partition_axis),
	...   partition_axis=partition_axis,
	... )

	>>> tokenizer = AutoTokenizer.from_pretrained(
	...   huggingface_repo_id_or_path, trust_remote_code=True
	... )

	>>> tokenizer.pad_token = tokenizer.eos_token
	>>> configs_to_initialize_model_class = {
	...   "config": model.config,
	...   "dtype": dtype,
	...   "param_dtype": dtype,
	...   "input_shape": input_shape,
	... }

	>>> train_arguments = TrainArguments(
	...   model_class=type(model),
	...   model_name="my_first_model_to_train_using_easydel",
	...   num_train_epochs=3,
	...   configs_to_initialize_model_class=configs_to_initialize_model_class,
	...   learning_rate=5e-5,
	...   learning_rate_end=1e-6,
	...   optimizer=EasyDeLOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
	...   scheduler=EasyDeLSchedulers.LINEAR,
	...   # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
	...   weight_decay=0.01,
	...   total_batch_size=64,
	...   max_training_steps=None,  # None to let trainer Decide
	...   do_train=True,
	...   do_eval=False,  # it's optional but supported
	...   backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
	...   max_sequence_length=max_length,  # Note that you have to change this in the model config too
	...   gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
	...   sharding_array=sharding_axis_dims,
	...   # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
	...   # everything training will be in sequence and model parallel automatic and share data between devices
	...   remove_ckpt_after_load=True,
	...   gradient_accumulation_steps=8,
	...   loss_re_mat="",
	...   dtype=dtype,
	...   param_dtype=dtype,
	...   init_input_shape=input_shape,
	... )


	>>> def ultra_chat_prompting_process(data_chunk):
	...   user_part = [
	...     chunk["content"]
	...     for chunk in data_chunk["messages"]
	...     if chunk["role"] == "user"
	...   ]
	...   assistant_part = [
	...     chunk["content"]
	...     for chunk in data_chunk["messages"]
	...     if chunk["role"] == "assistant"
	...   ]
	...
	...   prompt = ""
	...
	...   for uc, ac in zip(user_part, assistant_part):
	...     prompt += f"<|user|>\n{uc}</s>\n<|assistant|>\n{ac}</s>\n"
	...
	...   return {"prompt": prompt}


	>>> tokenization_process = lambda data_chunk: tokenizer(
	...   data_chunk["prompt"],
	...   add_special_tokens=False,
	...   max_length=max_length,
	...   padding="max_length",
	... )

	>>> dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
	>>> dataset_train = dataset["train_gen"].map(
	...   ultra_chat_prompting_process, num_proc=12
	... )
	>>> dataset_train = dataset_train.map(
	...   tokenization_process, num_proc=12, remove_columns=dataset_train.column_names
	... )

	>>> # you can do the same for evaluation process dataset

	>>> trainer = CausalLanguageModelTrainer(
	...   train_arguments, dataset_train, checkpoint_path=None
	... )

	>>> output = trainer.train(flax.core.FrozenDict({"params": params}))
	>>> print(f"Hey ! , here's where your model saved {output.checkpoint_path}")


	### With Using LoRA and XRapture



	>>> from flax.core import FrozenDict
	>>> from easydel import (
	...   TrainArguments,
	...   CausalLanguageModelTrainer,
	...   AutoEasyDeLModelForCausalLM,
	...   EasyDeLOptimizers,
	...   EasyDeLSchedulers,
	...   EasyDeLGradientCheckPointers,
	...   LoraRaptureConfig,
	... )
	>>> from datasets import load_dataset
	>>> import flax
	>>> from jax import numpy as jnp
	>>> from transformers import AutoTokenizer

	>>> huggingface_repo_id_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

	>>> model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	...   huggingface_repo_id_or_path,
	... )

	>>> max_length = 8196
	>>> model_parameters = FrozenDict({"params": params})

	>>> dtype = jnp.bfloat16
	>>> param_dtype = jnp.bfloat16  # you can change that if you want

	>>> tokenizer = AutoTokenizer.from_pretrained(
	...   huggingface_repo_id_or_path, trust_remote_code=True
	... )

	>>> model.config.add_basic_configurations(
	...   attn_mechanism="flash",  # Using FlashAttention
	...   block_b=1,
	...   block_q=1024,
	...   block_k=1024,
	...   block_k_major=1024,
	... )

	>>> tokenizer.pad_token = tokenizer.eos_token
	>>> configs_to_initialize_model_class = {
	...   "config": model.config,
	...   "dtype": dtype,
	...   "param_dtype": param_dtype,
	...   "input_shape": (1, 1),
	... }

	>>> rapture = LoraRaptureConfig(
	...   parameters=model_parameters,
	...   lora_dim=64,
	...   fully_fine_tune_parameters=[
	...     "embed_tokens"
	...   ],  # Model layer to be fully fine tuned
	...   lora_fine_tune_parameters=[
	...     "q_proj",
	...     "v_proj",
	...     "k_proj",
	...     "o_proj",
	...   ],  # LoRA Layer Targets you can pass this to none
	...   # For only Layer Tuning or transfer learning
	...   verbose=True,
	... )

	>>> train_arguments = TrainArguments(
	...   model_class=type(model),
	...   model_name="EasyDeL-Lora-Example",
	...   num_train_epochs=3,
	...   configs_to_initialize_model_class=configs_to_initialize_model_class,
	...   learning_rate=1e-4,  # Using higher learning rate is recommended
	...   learning_rate_end=8e-5,
	...   optimizer=EasyDeLOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
	...   scheduler=EasyDeLSchedulers.LINEAR,
	...   # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
	...   weight_decay=0.01,
	...   total_batch_size=512,
	...   max_training_steps=None,  # None to let trainer Decide
	...   do_train=True,
	...   do_eval=False,  # it's optional but supported
	...   backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
	...   max_sequence_length=max_length,  # Note that you have to change this in the model config too
	...   gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
	...   sharding_array=(
	...     1,
	...     -1,
	...     1,
	...     1,
	...   ),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
	...   # everything training will be in sequence and model parallel automatic and share data between devices
	...   remove_ckpt_after_load=True,
	...   gradient_accumulation_steps=1,
	...   loss_re_mat="",
	...   dtype=dtype,
	...   param_dtype=param_dtype,
	...   rapture_config=rapture,
	...   merge_lora_rapture_parameters=True,  # turning this off is still not supported and not recommended to do so
	...   # What this does ? this will merge the lora parameters with the original model parameters and the end of training
	... )


	>>> def ultra_chat_prompting_process(data_chunk):
	...   user_part = [
	...     chunk["content"]
	...     for chunk in data_chunk["messages"]
	...     if chunk["role"] == "user"
	...   ]
	...   assistant_part = [
	...     chunk["content"]
	...     for chunk in data_chunk["messages"]
	...     if chunk["role"] == "assistant"
	...   ]
	...
	...   prompt = ""
	...
	...   for uc, ac in zip(user_part, assistant_part):
	...     prompt += f"<|user|>\n{uc}</s>\n<|assistant|>\n{ac}</s>\n"
	...
	...   return {"prompt": prompt}


	>>> tokenization_process = lambda data_chunk: tokenizer(
	...   data_chunk["prompt"],
	...   add_special_tokens=False,
	...   max_length=max_length,
	...   padding="max_length",
	... )

	>>> dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
	>>> dataset_train = dataset["train_gen"].map(
	...   ultra_chat_prompting_process, num_proc=12
	... )
	>>> dataset_train = dataset_train.map(
	...   tokenization_process, num_proc=12, remove_columns=dataset_train.column_names
	... )

	>>> # you can do the same for evaluation process dataset

	>>> trainer = CausalLanguageModelTrainer(
	...   train_arguments, dataset_train, checkpoint_path=None
	... )

	>>> output = (
	...   trainer.train()
	... )  # you should not pass the parameters in Trainer.train anymore when
	>>> # you are using LoRA or transfer Learning
	>>> print(f"Hey ! , here's where your model saved {output.checkpoint_path}")
	"""

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
	) -> Callable:
		"""
		Creates a function to collect and process batches of data for training or evaluation.

		This function handles padding or truncating sequences to the specified `max_sequence_length`
		based on the chosen `truncation_mode`.

		Args:
		    max_sequence_length (int): The maximum allowed sequence length.
		    truncation_mode (typing.Literal["keep_end", "keep_start"], optional):
		        The truncation mode. Defaults to "keep_end".

		Returns:
		    Callable: A function that takes a batch of data and returns a processed batch.
		"""

		def collate_fn(batch):
			results = {}
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
		if self.arguments.sparsify_module:
			self.model.__call__ = sparse.sparsify(self.model.__call__)

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
			"""
			Creates an EasyDeLState object from given parameters.

			This function is used when loading a model from pretrained parameters
			or a checkpoint.

			Args:
			    parameters (FrozenDict): The model parameters.

			Returns:
			    EasyDeLState: The EasyDeLState object initialized with the provided parameters.
			"""
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
			spec=PartitionSpec(),
			mesh=self.arguments.get_mesh(),
		)
		create_sharded_state_from_params_function = jax.jit(
			create_state_from_params_function,
			in_shardings=(spec_named_sharding.params,),
			out_shardings=spec_named_sharding,
			donate_argnums=(0,),
		)
		sharded_train_step_function = jax.jit(
			create_casual_language_model_train_step(
				partition_spec=self.arguments.step_partition_spec,
				label_smoothing_factor=self.arguments.label_smoothing_factor,
				z_loss=self.arguments.z_loss,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(spec_named_sharding, empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		sharded_eval_step_function = jax.jit(
			create_casual_language_model_evaluation_step(self.arguments.step_partition_spec),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(empty_sharding, empty_sharding, empty_sharding),
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
	) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
		"""
		Initializes the training state, either from scratch, pretrained parameters, or a checkpoint.

		This method handles different initialization scenarios:
		1.  **No parameters, no state, no checkpoint, no LoRA:** Raises an error, as there's nothing to initialize from.
		2.  **Using LoRA, no parameters, no state:** Initializes from `self.lora_parameters`.
		3.  **State provided:** Uses the provided `state` directly.
		4.  **Finetuning and checkpoint path provided:** Loads the state from the checkpoint.
		5.  **Finetuning and model parameters provided:** Initializes the state from the provided `model_parameters`.
		6.  **No finetuning:** Initializes the state from scratch.

		Args:
		    model_parameters (Optional[flax.core.FrozenDict], optional):
		        Pretrained model parameters to initialize from. Defaults to None.
		    state (Optional[EasyDeLState], optional):
		        An existing EasyDeLState object to use. Defaults to None.

		Returns:
		    Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
		        A tuple containing the initialized or loaded EasyDeLState, shard functions, and gather functions.

		Raises:
		    RuntimeError: If no initialization source is provided and LoRA is not being used.
		    EasyDeLTimerError: If both `model_parameters` and `checkpoint_path` are provided,
		                        or if neither is provided when finetuning.
		"""
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
				params = sharded_state.params
				sharded_state.params = params
				if sharded_state.opt_state is None:
					logger.info("Optimizer State is not Found!, initializing one.")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = sharded_state.init_opt_state()
						opt_state = sharded_state.opt_state
						sharded_state = sharded_state.replace(opt_state=opt_state)
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
							config_kwargs=self.arguments.loaded_model_config_kwargs,
						)
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
							create_casual_language_model_train_step(
								partition_spec=self.arguments.step_partition_spec,
								label_smoothing_factor=self.arguments.label_smoothing_factor,
								z_loss=self.arguments.z_loss,
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
							create_casual_language_model_evaluation_step(
								self.arguments.step_partition_spec
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								empty_sharding,
								empty_sharding,
								empty_sharding,
							),
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

					model_parameters = model_parameters
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
			if self.arguments.sparsify_module:
				...  # disabled at the moment to fix shardings..
				# sharded_state = sharded_state.replace(
				# 	params=apply_sparsity_to_params(
				# 		params=sharded_state.params,
				# 		sparsify_module=self.arguments.sparse_module_type,
				# 		verbose=True,
				# 	)
				# )
			self.sharded_state = sharded_state
			return sharded_state, shard_fns, gather_fns

	def train(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> CausalLMTrainerOutput:
		"""
		Trains the Causal Language Model (CLM).

		This method orchestrates the entire training process, including:
		- Initializing the training state.
		- Iterating over epochs and batches of data.
		- Performing forward and backward passes.
		- Updating model parameters.
		- Logging training metrics.
		- Saving checkpoints.
		- Handling KeyboardInterrupts and timeouts.
		- Evaluating the model (optional).
		- Merging LoRA parameters (optional).

		Args:
		    model_parameters (Optional[flax.core.FrozenDict], optional):
		        Pretrained model parameters for initialization. Defaults to None.
		    state (Optional[EasyDeLState], optional):
		        An existing EasyDeLState to resume training from. Defaults to None.

		Returns:
		    CausalLMTrainerOutput: An object containing the trained state, mesh, shard/gather functions, checkpoint manager,
		                            checkpoint path, and last saved file name.
		"""

		def get_layer_names(frozen_dict, prefix=""):
			"""
			Recursively retrieves layer names and their corresponding parameter arrays from a FrozenDict.

			Args:
			    frozen_dict (FrozenDict): The FrozenDict containing the model parameters.
			    prefix (str, optional): A prefix to add to the layer names. Defaults to "".

			Returns:
			    dict[str, jnp.ndarray]: A dictionary mapping layer names to their parameter arrays.
			"""
			layer_names = {}
			for key, value in frozen_dict.items():
				if isinstance(value, FrozenDict):
					layer_names.update(get_layer_names(value, prefix=f"{prefix}_{key}"))
				else:
					layer_name = f"{prefix}_{key}".lstrip("/")
					layer_names[layer_name] = value
			return layer_names

		def count_model_parameters(_p):
			"""Prints the number of model parameters in billions."""
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
		flops_per_device = (
			self.calculate_number_total_flops_per_device(params=model_parameters) / 1e12
		)
		count_model_parameters(sharded_state.params)
		with self.mesh:
			pbar = tqdm(total=self.max_training_steps)
			current_step = int(jax.device_get(sharded_state.step))
			loss_sum = None
			accuracy_sum = None
			filename = None
			pbar.update(sharded_state.step.tolist())  # type: ignore

			model_parameters_number = (
				sum(
					n.size
					for n in jax.tree_util.tree_flatten(flax.core.unfreeze(sharded_state.params))[
						0
					]
				)
				/ 1e9
			)
			self.arguments.log_metrics(
				{"Number of Model Parameters (Billion)": model_parameters_number},
				step=0,
			)
			try:
				train_iter = iter(self.dataloader_train)
				for epoch in range(self.arguments.num_train_epochs):
					time_start = time.time()
					for _ in range(self.max_training_steps // self.arguments.num_train_epochs):
						try:
							batch = next(train_iter)
						except StopIteration:
							train_iter = iter(self.dataloader_train)
							batch = next(train_iter)
						if (
							self.arguments.step_start_point is not None
							and self.arguments.step_start_point > current_step
						):
							pbar.update(1)
						elif current_step < self.max_training_steps:
							time_prev = time_start
							time_start = time.time()
							step_time = time_start - time_prev

							for ssb in self.arguments.ids_to_pop_from_dataset:
								_ = batch.pop(ssb, None)

							if self.pruning_module is not None:
								sharded_state = sharded_state.replace(
									params=self.pruning_module.pre_forward_update(
										sharded_state.params,
										sharded_state.opt_state,
									)
								)

							(
								sharded_state,
								loss,
								metrics,
							) = self.sharded_train_step_function(sharded_state, batch)
							if self.pruning_module is not None:
								sharded_state = sharded_state.replace(
									params=self.pruning_module.post_gradient_update(
										sharded_state.params,
										sharded_state.opt_state,
									)
								)
							loss.block_until_ready()
							total_time = time.time() - time_start
							flops = flops_per_device / total_time
							trained_tokens = jnp.multiply(
								self.arguments.max_sequence_length,
								jnp.multiply(current_step, self.arguments.total_batch_size),
							)  # It's faster

							with jax.spmd_mode("allow_all"):
								calculating_metrics_start = time.time()
								loss_sum = loss if loss_sum is None else loss_sum + loss
								accuracy = metrics["accuracy"]
								accuracy_sum = (
									accuracy if accuracy_sum is None else accuracy_sum + accuracy
								)
								mean_loss = loss_sum / (
									(current_step + 1) - self.arguments.step_start_point
								)
								mean_accuracy = accuracy_sum / (
									(current_step + 1) - self.arguments.step_start_point
								)
								perplexity = jnp.exp(loss)
								calculating_metrics_end = time.time()
								train_metrics = {
									"train/loss": loss.tolist(),
									"train/mean_loss": mean_loss.tolist(),
									"train/accuracy": accuracy,
									"train/mean_accuracy": mean_accuracy.tolist(),
									"train/learning_rate": self.scheduler(current_step).tolist(),
									"train/step": current_step,
									"train/step_time": step_time,
									"train/perplexity": perplexity.tolist(),
									"train/trained_tokens": trained_tokens,
									"train/regularization_z_loss": metrics[
										"regularization_z_loss"
									].tolist(),
									"train/epoch": epoch,
									"train/TFLOPs": flops,
								}
							if self.arguments.log_grad_norms:
								train_metrics.update(
									{
										"train/max_grad_norm": metrics["max_grad_norm"].tolist(),
										"train/mean_grad_norm": metrics["mean_grad_norm"].tolist(),
									}
								)
							aux_loss = metrics.get("aux_loss", None)
							if aux_loss is not None:
								train_metrics.update({"train/aux_loss": aux_loss.tolist()})
							pbar.update(1)
							pbar.set_postfix(
								**{k.replace("train/", ""): v for k, v in train_metrics.items()}
							)
							if not self.arguments.performance_mode:
								if self.arguments.log_grad_norms:
									train_metrics.update(
										{
											f"grad_norm/{layer_name}": grad_norm.tolist()
											for layer_name, grad_norm in get_layer_names(
												metrics["grad_norms"]
											).items()
										}
									)
								train_metrics.update(
									{
										"time_cal/calculating_metrics_step_time": (
											calculating_metrics_end - calculating_metrics_start
										)
									}
								)
								train_metrics.update(self.arguments._captured_memory)
							self.arguments.log_metrics(
								metrics=train_metrics,
								step=current_step,
							)
							self.arguments.ensure_training_time(time_passed=time.time() - start_time)
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
			output = CausalLMTrainerOutput(
				state=sharded_state,
				mesh=self.mesh,
				shard_fns=shard_fns,
				gather_fns=gather_fns,
				checkpoint_manager=self.checkpoint_manager,
			)
			if self.arguments.save_steps is None or self.arguments.do_last_save:
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
			self.arguments._stop_capturing_memory = True
			self.finish()
			return output

	def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
		"""
		Evaluates the Causal Language Model (CLM) using the provided model state.

		This method iterates over the evaluation dataset, performs forward passes,
		calculates evaluation metrics, logs the metrics, and yields the metrics for
		each evaluation step.

		Args:
		    model_state (EasyDeLState): The EasyDeLState object containing the model parameters
		                                and other relevant information.

		Yields:
		    Iterator[dict]: An iterator yielding a dictionary of evaluation metrics for each step.

		Raises:
		    AssertionError: If `self.dataloader_eval` is not set (meaning the evaluation dataloader is missing).
		"""
		assert (
			self.dataloader_eval is not None
		), "`dataloader_eval` is required by evaluator function."
		with self.mesh:
			pbar = tqdm(total=self.max_evaluation_steps)
			pbar.set_description("Evaluating")
			current_step = 0
			loss_sum = None
			accuracy_sum = None

			flops_per_device = (
				self.calculate_number_total_flops_per_device(params=model_state.params) / 1e12
			)
			try:
				eval_iter = iter(self.dataloader_eval)
				for _ in range(self.max_evaluation_steps):
					try:
						batch = next(eval_iter)
					except StopIteration:
						eval_iter = iter(self.dataloader_eval)
						batch = next(eval_iter)
					time_start = time.time()
					for key in self.arguments.ids_to_pop_from_dataset:
						_ = batch.pop(key, None)
					metrics = self.sharded_eval_step_function(model_state, batch)
					total_time = time.time() - time_start
					flops = flops_per_device / total_time
					(loss, accuracy, aux_loss) = metrics

					loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
					accuracy_sum = (
						accuracy.tolist() if (accuracy_sum is None) else accuracy_sum + accuracy
					)

					eval_metrics = {
						"eval/loss": loss.tolist(),
						"eval/mean_loss": loss_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/mean_accuracy_sum": accuracy_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/step": current_step,
						"eval/step_time": total_time,
						"eval/perplexity": jnp.exp(loss).tolist(),
						"eval/TFLOPs": flops,
					}
					if aux_loss is not None:
						eval_metrics.update({"eval/aux_loss": aux_loss})
					log_metrics = copy.deepcopy(eval_metrics)
					eval_metrics.update(self.arguments._captured_memory)
					self.arguments.log_metrics(metrics=eval_metrics, step=current_step)

					pbar.update(1)
					pbar.set_postfix(
						**{k.replace("eval/", ""): v for k, v in log_metrics.items()}
					)
					yield log_metrics
					current_step += 1
			except KeyboardInterrupt:
				termcolor.cprint(
					"KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
					color="cyan",
					force_color=True,
				)
