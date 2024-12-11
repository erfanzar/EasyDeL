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
from __future__ import annotations

import inspect
import os
import re
import typing as tp
import warnings
from copy import deepcopy
from functools import cached_property

import chex
import fjformer
import fjformer.sharding
import jax
import jax.extend
import jax.tree_util
from fjformer.checkpoint import CheckpointManager
from fjformer.dtypes import Array8Bit
from fjformer.sharding import match_partition_rules
from flax import nnx as nn
from flax.core import FrozenDict, unfreeze
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.generation.flax_utils import FlaxSampleOutput
from transformers.utils.generic import working_or_temp_dir

from easydel.etils.easystate import EasyDeLState
from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLPlatforms,
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.inference.logits_process import FlaxLogitsProcessorList
from easydel.layers.caching import (
	TransformerCache,
	TransformerCacheMetaData,
)
from easydel.modules._base.base_config import EasyDeLBaseConfig
from easydel.utils.quantizers import DEFAULT_QUANTIZATION_PATTERN, EasyQuantizer
from easydel.utils.traversals import flatten_dict, unflatten_dict
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
)

logger = get_logger(__name__)

FLAX_WEIGHTS_NAME = "easydel-model.parameters"
AVAILALBE_DEVICES = jax.device_count()
PartitionLike = tp.Optional[
	tp.Union[tp.Mapping[str, tp.Callable], tp.Mapping[tuple, tp.Callable]]
]


class EasyDeLBaseModule(nn.Module):
	config_class: EasyDeLBaseConfig
	base_model_prefix: str
	_model_task: tp.Optional[str] = None
	_model_type: tp.Optional[str] = None

	def __init__(
		self,
		config: EasyDeLBaseConfig,
		dtype: jnp.dtype,
		param_dtype: jnp.dtype,
		precision: lax.PrecisionLike,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

	@cached_property
	def graphtree_params_shape(self):
		graphtree = nn.eval_shape(lambda: nn.split(self, nn.Param, ...)[1])
		graphtree = flatten_dict(graphtree)
		for key in list(graphtree.keys()):
			graphtree[key] = graphtree[key].value
		graphtree = unflatten_dict(graphtree)
		return graphtree

	@cached_property
	def mesh(self):
		return self.config.mesh

	@property
	def model_task(self):
		return self._model_task

	@property
	def model_type(self):
		return self._model_type

	@cached_property
	def causal_mask(self):
		return self.config.get_basic_causal_mask()

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies()

	def get_named_sharding(self, partition_rules=None, partition_specs=None):
		if partition_rules is None:
			partition_rules = self.config.get_partition_rules(True)
		if partition_specs is None:
			partition_specs = match_partition_rules(partition_rules, self.params_shape_tree)
		return jax.tree_util.tree_map(
			lambda spec: jax.sharding.NamedSharding(
				spec=spec,
				mesh=self.mesh,
			),
			partition_specs,
		)

	def get_input_embeddings(self):
		"""The get_input_embeddings function returns the embedding layer of the model.

		Args:
		    self: Refer to the current object

		Returns:
		    The embedding layer of the model
		"""
		raise NotImplementedError()

	def set_input_embeddings(self, value):
		"""The set_input_embeddings function is used to set the embedding module of the model.

		Args:
		    self: Represent the instance of the class
		    value: Set the embeddings of the model
		"""
		raise NotImplementedError()

	def get_output_embeddings(self):
		"""The get_output_embeddings function returns the output embeddings of a model.

		Args:
		    self: Represent the instance of the class

		Returns:
		    The output embeddings of the model
		"""
		raise NotImplementedError()

	def set_output_embeddings(self, new_embeddings):
		"""The set_output_embeddings function is used to set the output embeddings of a model.
		This function can be used to change the output embedding layer of a pretrained model in order to finetune it
		to some downstream task. Changing this layer has an effect only if the model has already been fine-tuned on some
		task (e.g., for classification). If you are training your own language models, you should call this function before
		you start training.

		Args:
		    self: Represent the instance of the class
		    new_embeddings: Set the embeddings of the output layer

		Returns:
		    A new embedding layer
		"""
		raise NotImplementedError()

	def set_decoder(self, decoder):
		"""The set_decoder function is used to set the decoder for a given encoder.

		Args:
		    self: Refer to the object itself
		    decoder: Set the decoder for a given encoder

		Returns:
		    A decoder
		"""
		raise NotImplementedError()

	def get_decoder(self):
		"""The get_decoder function is used to create a decoder object.

		Args:
		    self: Represent the instance of the class

		Returns:
		    A decoder object
		"""
		raise NotImplementedError()

	def init_cache(self, batch_size: int, max_length: int):
		return TransformerCache.init_layers_cache(
			num_hidden_layers=self.config.num_hidden_layers,
			dtype=self.dtype,
			key_values_partition_specs=PartitionSpec(
				self.config.partition_axis.batch_axis,
				self.config.partition_axis.key_sequence_axis,
				self.config.partition_axis.head_axis,
				self.config.partition_axis.attention_dim_axis,
			),
			metadata=TransformerCacheMetaData.create(
				batch_size=batch_size,
				sequence_length=max_length,
				num_heads=self.config.num_key_value_heads,
				head_dim=self.config.head_dim,
			),
			quantizer=EasyQuantizer(
				quantization_method=self.config.kv_cache_quantization_method,
				block_size=self.config.kv_cache_quantization_blocksize,
				quantization_platform=self.config.platform,
			),
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		attention_mask: tp.Optional[chex.Array] = None,
	):
		"""The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

		Args:
		    self: Access variables that belong to the class
		    input_ids: Pass in the input tokens
		    max_length: Set the length of the sequence to be generated
		    attention_mask: tp.Optional[chex.Array]: Mask the attention
		        weights

		Returns:
		    A dictionary of the past_key_values, attention_mask and
		    position ids
		"""
		batch_size, seq_length = input_ids.shape
		past_key_values = self.init_cache(batch_size, max_length)

		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			position_ids = attention_mask.cumsum(axis=-1) - 1
			extended_attention_mask = jax.lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
			)

		return {
			"past_key_values": past_key_values,
			"attention_mask": extended_attention_mask,
			"position_ids": position_ids,
		}

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs["past_key_values"] = model_outputs.past_key_values
		model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
		return model_kwargs

	def _validate_signature(
		self,
		method,
		args: tuple,
		kwargs: tp.Dict[str, tp.Any],
	) -> tp.Dict[str, tp.Any]:
		"""
		Validates and filters arguments based on the method's signature.

		Args:
				method: The method to check signature against
				args: Positional arguments
				kwargs: Keyword arguments

		Returns:
				tp.Dict[str, tp.Any]: Filtered kwargs containing only valid parameters
		"""
		# Get the signature of the child class's __call__ method
		sig = inspect.signature(method)
		valid_params = sig.parameters

		# Convert args to kwargs based on parameter names
		args_as_kwargs = {}
		positional_params = [
			param
			for param in valid_params.values()
			if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
		]

		for i, arg in enumerate(args):
			if i < len(positional_params):
				args_as_kwargs[positional_params[i].name] = arg

		# Combine converted args and original kwargs
		all_kwargs = {**args_as_kwargs, **kwargs}

		# Filter out invalid kwargs
		filtered_kwargs = {}
		for name, value in all_kwargs.items():
			if name in valid_params:
				# Check if the parameter accepts the value's type
				param = valid_params[name]
				if param.annotation != inspect.Parameter.empty:
					try:
						# Handle tp.Optional types
						if (
							getattr(param.annotation, "__origin__", None) is tp.Optional
							and value is not None
						):
							expected_type = param.annotation.__args__[0]
							if not isinstance(value, expected_type):
								print(
									f"Warning: Parameter '{name}' expected type {expected_type}, "
									f"got {type(value)}. Skipping parameter."
								)
								continue
					except Exception:
						# If type checking fails, still include the parameter
						pass
				filtered_kwargs[name] = value
			else:
				warnings.warn(
					f"  Parameter '{name}' not found in child class signature. Skipping.",
					stacklevel=1,
				)

		return filtered_kwargs

	def to_easydel_state(
		self,
		params: FrozenDict,
		auto_check_params: bool = True,
	):
		"""
		Convert the Model to EasyDeLState
		"""
		if auto_check_params:
			gp = params.get("params", None)
			params = FrozenDict({"params": params} if gp is None else {"params": gp})
		return EasyDeLState.load(
			apply_fn=self.__call__,
			params=params,
			opt_state=None,
			module_config=self.config,
			module=self,
			step=0,
		)

	def to_pytorch(
		self,
		params: FrozenDict,
		base_hf_auto_class=None,
		easystate_to_huggingface_model_kwargs: tp.Optional[dict] = None,
	):
		"""
		Return the Huggingface / Pytorch implementation of the model with same weights  (if model is available in HF)
		"""
		if base_hf_auto_class is None:
			from transformers import AutoModelForCausalLM as base_hf_auto_class
		from easydel.transform.parameters_transformation import (
			easystate_to_huggingface_model,
		)

		state = self.to_easydel_state(params=params)
		if easystate_to_huggingface_model_kwargs is None:
			easystate_to_huggingface_model_kwargs = {}

		model_config = state.module_config
		if model_config is None:
			model_config = state.module.config_class
		# model_type = model_config.model_type
		model_class = base_hf_auto_class._model_mapping[type(model_config)]  # noqa
		hf_model = easystate_to_huggingface_model(
			state=state,
			base_huggingface_module=model_class,
			config=model_config,
			**easystate_to_huggingface_model_kwargs,
		)
		return hf_model

	@staticmethod
	def to_8bit(params, quantization_fields=None):
		if quantization_fields is None:
			quantization_fields = ["kernel", "embedding"]

		def quantize_params(params: dict) -> dict:
			"""Quantizes model parameters using Array8Bit.

			Args:
			    params: A dictionary of model parameters.

			Returns:
			    A dictionary of quantized model parameters.
			"""

			def q(path: str, array: tp.Any) -> Array8Bit:
				"""Quantizes a single parameter array."""
				path = [p for p in path[0].key]
				for field in quantization_fields:
					if field in path:
						return Array8Bit.quantize(array, qk=64)
				return array

			return unflatten_dict(
				jax.tree_util.tree_map_with_path(
					q,
					flatten_dict(params),
				)
			)

		return quantize_params(params)

	def _model_card(self, name, repo_id):
		from easydel import __version__
		from easydel.utils.readme_generator import ModelInfo, ReadmeGenerator

		return ReadmeGenerator().generate_readme(
			ModelInfo(
				name=name,
				type=self.__class__.__name__,
				repo_id=repo_id,
				model_class=self.config_class.model_type,
				version=__version__,
			)
		)

	def save_pretrained(  # noqa
		self,
		save_directory: tp.Union[str, os.PathLike],
		params,
		push_to_hub=False,
		token: tp.Optional[tp.Union[str, bool]] = None,
		gather_fns: dict[tp.Callable] = None,
		float_dtype=None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		safe=True,
		**kwargs,
	):
		if token is not None:
			kwargs["token"] = token
		if os.path.isfile(save_directory):
			logger.error(
				f"Provided path ({save_directory}) should be a directory, not a file"
			)
			return
		os.makedirs(save_directory, exist_ok=True)
		repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
		if push_to_hub:
			commit_message = kwargs.pop("commit_message", None)
			repo_id = self._create_repo(repo_id, **kwargs)
			files_timestamps = self._get_files_timestamps(save_directory)
		save_directory = os.path.abspath(save_directory)
		self.config.architectures = [self.__class__.__name__[4:]]
		config = deepcopy(self.config)
		config.__dict__.pop("attn_dtype", None)  # make sure dtypes are not included
		config.save_pretrained(save_directory)
		if self.can_generate():
			self.generation_config.save_pretrained(save_directory)
		output_model_file = os.path.join(save_directory, "easydel-model.parameters")
		readme_path = os.path.join(save_directory, "README.md")
		if not os.path.exists(readme_path):
			open(readme_path, "w").write(self._model_card(repo_id, repo_id))
		func = (
			CheckpointManager.save_checkpoint_safe
			if (safe)
			else CheckpointManager.save_state_to_file
		)

		func(
			path=output_model_file,
			gather_fns=gather_fns,
			mismatch_allowed=mismatch_allowed,
			state=params,
			float_dtype=float_dtype,
			verbose=verbose,
		)

		logger.info(f"Model weights saved in {output_model_file}")

		if push_to_hub:
			self._upload_modified_files(
				save_directory,
				repo_id,
				files_timestamps,
				commit_message=commit_message,
				token=token,
			)

	def push_to_hub(
		self,
		repo_id: str,
		params,
		use_temp_dir: tp.Optional[bool] = None,
		commit_message: tp.Optional[str] = None,
		private: tp.Optional[bool] = None,
		token: tp.Optional[tp.Union[bool, str]] = None,
		create_pr: bool = False,
		safe_serialization: bool = True,
		gather_fns: dict[tp.Callable] = None,
		float_dtype=None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		revision: str = None,
		commit_description: str = None,
		tags: tp.Optional[tp.List[str]] = None,
	) -> str:
		working_dir = repo_id.split("/")[-1]

		repo_id = self._create_repo(
			repo_id,
			private=private,
			token=token,
			repo_url=None,
			organization=None,
		)

		if use_temp_dir is None:
			use_temp_dir = not os.path.isdir(working_dir)

		with working_or_temp_dir(
			working_dir=working_dir, use_temp_dir=use_temp_dir
		) as work_dir:
			files_timestamps = self._get_files_timestamps(work_dir)

			# Save all files.
			self.save_pretrained(
				work_dir,
				params=params,
				mismatch_allowed=mismatch_allowed,
				safe=safe_serialization,
				gather_fns=gather_fns,
				float_dtype=float_dtype,
				verbose=verbose,
				repo_id=repo_id,
			)

			return self._upload_modified_files(
				work_dir,
				repo_id,
				files_timestamps,
				commit_message=commit_message,
				token=token,
				create_pr=create_pr,
				revision=revision,
				commit_description=commit_description,
			)

	@classmethod
	def can_generate(cls) -> bool:
		"""
		Returns whether this model can generate sequences with `.generate()`. Returns:
		    `bool`: Whether this model can generate sequences with `.generate()`.
		"""
		# Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
		# Alternativelly, the model can also have a custom `generate` function.
		if "GenerationMixin" in str(
			cls.prepare_inputs_for_generation
		) and "GenerationMixin" in str(cls.generate):
			return False
		return True

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: tp.Union[str, os.PathLike],
		sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: PartitionAxis = PartitionAxis(),  # noqa
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		safe: bool = True,
		precision: jax.lax.PrecisionLike = jax.lax.Precision("fastest"),  # noqa
		config_kwargs: tp.Optional[dict[str, tp.Any]] = None,
		partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]] = None,
		quantization_method: tp.Optional[EasyDeLQuantizationMethods] = None,
		quantization_platform: tp.Optional[EasyDeLPlatforms] = "jax",
		backend: tp.Optional[EasyDeLBackends] = None,
		platform: tp.Optional[EasyDeLPlatforms] = "jax",
		bit_targeted_params: tp.Optional[tp.List[str]] = None,
		model_task: str = "base-module",
		quantization_block_size: int = 128,
		shard_fns: dict[tp.Callable] = None,
		auto_shard_model: bool = False,
		remove_dict_prefix=None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		*model_args,
		config: tp.Optional[tp.Union[EasyDeLBaseConfig, str, os.PathLike]] = None,
		cache_dir: tp.Optional[tp.Union[str, os.PathLike]] = None,
		ignore_mismatched_sizes: bool = False,
		force_download: bool = False,
		local_files_only: bool = False,
		token: tp.Optional[tp.Union[str, bool]] = None,
		revision: str = "main",
		**kwargs,
	):
		"""
		loads EasyDeL Models
		"""

		from huggingface_hub import HfApi
		from transformers import GenerationConfig
		from transformers.utils import download_url as _download_url
		from transformers.utils import is_offline_mode as _is_offline_mode
		from transformers.utils import is_remote_url as _is_remote_url

		api = HfApi(token=token)

		proxies = kwargs.pop("proxies", None)
		trust_remote_code = kwargs.pop("trust_remote_code", None)
		from_pipeline = kwargs.pop("_from_pipeline", None)
		from_auto_class = kwargs.pop("_from_auto", False)
		subfolder = kwargs.pop("subfolder", "")
		commit_hash = kwargs.pop("_commit_hash", None)

		# Not relevant for Flax Models
		_ = kwargs.pop("adapter_kwargs", None)

		if trust_remote_code is True:
			logger.warning(
				"The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
				" ignored."
			)

		if _is_offline_mode() and not local_files_only:
			logger.info("Offline mode: forcing local_files_only=True")
			local_files_only = True

		config_path = config if config is not None else pretrained_model_name_or_path
		from easydel.modules.auto_configuration import (
			AutoEasyDeLConfig,
			AutoShardAndGatherFunctions,
			get_modules_by_type,
		)

		config = AutoEasyDeLConfig.from_pretrained(
			config_path,
			sharding_axis_dims=sharding_axis_dims,
			sharding_axis_names=sharding_axis_names,
			partition_axis=partition_axis,
			from_torch=False,
			backend=backend,
			platform=platform,
		)

		if config_kwargs is not None:
			for k, v in config_kwargs.items():
				setattr(config, k, v)

		if commit_hash is None:
			commit_hash = getattr(config, "_commit_hash", None)
		if auto_shard_model and shard_fns is None:
			shard_fns, _ = AutoShardAndGatherFunctions.from_config(
				config=config,
				flatten=False,
				partition_rules=partition_rules,
			)
			fns = {"params": shard_fns}
			fns.update(shard_fns)
			shard_fns = fns
		elif auto_shard_model and shard_fns is not None:
			logger.warning(
				"`auto_shard_model` will be ignored since `shard_fns` is provided."
			)
		if pretrained_model_name_or_path is not None:
			pretrained_model_name_or_path = str(pretrained_model_name_or_path)
			is_local = os.path.isdir(pretrained_model_name_or_path)
			if os.path.isdir(pretrained_model_name_or_path):
				if os.path.isfile(
					os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
				):
					archive_file = os.path.join(
						pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
					)
				else:
					raise EnvironmentError(
						f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}"
					)
			elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
				archive_file = pretrained_model_name_or_path
				is_local = True
			elif _is_remote_url(pretrained_model_name_or_path):
				filename = pretrained_model_name_or_path
				resolved_archive_file = _download_url(pretrained_model_name_or_path)
			else:
				filename = FLAX_WEIGHTS_NAME
				try:
					resolved_archive_file = api.hf_hub_download(
						repo_id=pretrained_model_name_or_path,
						filename=filename,
						subfolder=subfolder,
						revision=revision,
						cache_dir=cache_dir,
						force_download=force_download,
						proxies=proxies,
						token=token,
						local_files_only=local_files_only,
					)

					if resolved_archive_file is None:
						raise EnvironmentError("no model parameters found!")
				except EnvironmentError:
					raise
				except Exception:
					raise EnvironmentError(
						f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
						" from 'https://huggingface.co/models', make sure you don't have a local directory with the"
						f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
						f" directory containing a file named {FLAX_WEIGHTS_NAME}."
					) from None

			if is_local:
				logger.debug(f"loading weights file {archive_file}")
				resolved_archive_file = archive_file
				filename = resolved_archive_file.split(os.path.sep)[-1]
			else:
				logger.debug(
					f"loading weights file {filename} from cache at {resolved_archive_file}"
				)
		else:
			resolved_archive_file = None

		if cls.__name__ == "EasyDeLBaseModule":
			# if they are using EasyDeLBaseModule.from_pretrained
			# they will get error AssertionError: `module` must be provided.` so we autoset this to make sure user don't
			# experience this error.
			_, cls, _ = get_modules_by_type(config.model_type, model_task)
		model = cls(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=nn.Rngs(0),
		)
		if bit_targeted_params is None:
			params_pattern_selection = re.compile(DEFAULT_QUANTIZATION_PATTERN)
		else:
			params_pattern_selection = bit_targeted_params
		if quantization_method is not None:
			quantizer = EasyQuantizer(
				quantization_method=quantization_method,
				block_size=quantization_block_size,
				quantization_platform=quantization_platform,
			)

		def maybe_quantize(tensor, key):
			if isinstance(key, str):
				key = key.split(".")
			if quantization_method is not None:
				if (
					quantizer is not None
					and key[-1] != "embedding"
					and params_pattern_selection.search("/".join(key))
				):
					tensor = quantizer(array=tensor)
			return tensor

		if safe:
			state, _ = CheckpointManager.load_checkpoint_safe(
				path=resolved_archive_file,
				mismatch_allowed=mismatch_allowed,
				verbose=verbose,
				shard_fns=shard_fns,
				callback=maybe_quantize,
			)
		else:
			state = CheckpointManager.load_checkpoint(
				path=resolved_archive_file,
				mismatch_allowed=mismatch_allowed,
				verbose=verbose,
				shard_fns=shard_fns,
				remove_dict_prefix=remove_dict_prefix,
				callback=maybe_quantize,
			)

		params = state.get("params", None)
		if params is not None:
			state = params

		state = flatten_dict(state)
		random_state = flatten_dict(unfreeze(model.params_shape_tree))

		missing_keys = model.required_params - set(state.keys())
		unexpected_keys = set(state.keys()) - model.required_params

		# Disabling warning when porting pytorch weights to flax, flax does not uses num_batches_tracked
		for unexpected_key in unexpected_keys.copy():
			if "num_batches_tracked" in unexpected_key[-1]:
				unexpected_keys.remove(unexpected_key)

		if missing_keys:
			logger.warning(
				f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
				"Make sure to call model.init_weights to initialize the missing weights."
			)
			cls._missing_keys = missing_keys

		mismatched_keys = []
		for key in state.keys():
			if key in random_state and state[key].shape != random_state[key].shape:
				if ignore_mismatched_sizes:
					mismatched_keys.append((key, state[key].shape, random_state[key].shape))
					state[key] = random_state[key]
				else:
					raise ValueError(
						f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
						f"{state[key].shape} which is incompatible with the model shape {random_state[key].shape}. "
						"Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
						"model."
					)

		if missing_keys:
			for missing_key in missing_keys:
				state[missing_key] = random_state[missing_key]

		# remove unexpected keys to not be saved again
		for unexpected_key in unexpected_keys:
			del state[unexpected_key]

		if len(unexpected_keys) > 0:
			logger.warning(
				f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
				f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
				f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
				" with another architecture (e.g. initializing a BertForSequenceClassification model from a"
				" BertForPreTraining model).\n- This IS NOT expected if you are initializing"
				f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
				" (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
			)

		if len(missing_keys) > 0:
			logger.warning(
				f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
				f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
				" TRAIN this model on a down-stream task to be able to use it for predictions and inference."
			)
		if len(mismatched_keys) > 0:
			mismatched_warning = "\n".join(
				[
					f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
					for key, shape1, shape2 in mismatched_keys
				]
			)
			logger.warning(
				f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
				f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
				f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
				" to use it for predictions and inference."
			)

		if model.can_generate():
			try:
				model.generation_config = GenerationConfig.from_pretrained(
					pretrained_model_name_or_path,
					cache_dir=cache_dir,
					force_download=force_download,
					proxies=proxies,
					local_files_only=local_files_only,
					token=token,
					revision=revision,
					subfolder=subfolder,
					_from_auto=from_auto_class,
					_from_pipeline=from_pipeline,
					**kwargs,
				)
			except OSError:
				logger.info(
					"Generation config file not found, using a generation config created from the model config."
				)
				pass
		return model, unflatten_dict(state)

	def shard_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[jax.sharding.Mesh] = None,
	) -> EasyDeLBaseModule:
		"""
		Shards model parameters according to the provided partition rules.

		Args:
		    partition_rules: A dictionary mapping parameter names or a tuple of parameter names to
		        partitioning functions. The partitioning functions should take the shape and dtype of
		        the parameter as input and return a `jax.sharding.PartitionSpec`. If `None`, defaults to
		        the partition rules specified in the model configuration for fully sharded data parallelism.
		    mesh: The `jax.sharding.Mesh` object specifying the device mesh. If `None`, defaults to the mesh
		        defined in the model configuration.

		Returns:
		    A sharded version of the input parameters, where each parameter is partitioned across devices
		    according to the specified rules and mesh.
		"""
		if mesh is None:
			mesh = self.config.mesh
		if partition_rules is None:
			partition_rules = self.config.get_partition_rules(
				fully_sharded_data_parallel=True
			)
		shard_fns = fjformer.sharding.make_shard_and_gather_fns(
			partition_specs=fjformer.sharding.match_partition_rules(
				rules=partition_rules,
				params=self.graphtree_params_shape,
			),
			mesh=mesh,
		)[0]

		return self.apply_shardings(sharding_fns=shard_fns)

	def gather_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[jax.sharding.Mesh] = None,
	):
		"""
		Gathers sharded model parameters to the host device.

		This method reverses the sharding process performed by `shard_model`, collecting the parameter shards
		from different devices and aggregating them into a single PyTree on the host device.

		Args:
		    partition_rules: A dictionary mapping parameter names or a tuple of parameter names to
		        partitioning functions. The partitioning functions should take the shape and dtype of
		        the parameter as input and return a `jax.sharding.PartitionSpec`. If `None`, defaults to
		        the partition rules specified in the model configuration for fully sharded data parallelism.
		    mesh: The `jax.sharding.Mesh` object specifying the device mesh. If `None`, defaults to the mesh
		        defined in the model configuration.

		Returns:
		    A non-sharded version of the input parameters, where all parameters are gathered onto the host device.
		"""
		if mesh is None:
			mesh = self.config.mesh
		if partition_rules is None:
			partition_rules = self.config.get_partition_rules(
				fully_sharded_data_parallel=True
			)
		gather_fns = fjformer.sharding.make_shard_and_gather_fns(
			partition_specs=fjformer.sharding.match_partition_rules(
				rules=partition_rules,
				params=self.graphtree_params_shape,
			),
			mesh=mesh,
		)[1]
		return self.apply_shardings(self, sharding_fns=gather_fns)

	def apply_shardings(self, sharding_fns):
		gdef, state, other = nn.split(self, nn.Param, ...)
		sharding_fns = flatten_dict(sharding_fns)
		_shard_keys = list(sharding_fns.keys())

		def _map(path, val: nn.VariableState):
			if val.value is not None and path in _shard_keys:
				val.value = sharding_fns[path](val.value)
			return val

		state = state.map(_map)
		self = nn.merge(gdef, state, other)
		return self

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]: ...
	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxSequenceClassifierOutput, tp.Tuple]: ...


class EasyDeLBaseVisionModule(EasyDeLBaseModule):
	def init_cache(self, batch_size, max_length):
		input_ids = jnp.ones((batch_size, max_length))
		attention_mask = jnp.ones((batch_size, max_length), "i4")
		position_ids = jnp.broadcast_to(
			jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
		)
		vision_mask = jnp.ones((batch_size, max_length), dtype=bool)

		init_variables = self.module.init(
			jax.random.PRNGKey(0),
			input_ids=input_ids,
			vision_mask=vision_mask,
			attention_mask=attention_mask,
			position_ids=position_ids,
			return_dict=False,
			init_cache=True,
		)
		return init_variables["cache"]

	def __call__(
		self,
		input_ids: chex.Array,
		vision_mask: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		params: dict = None,
		past_key_values: tp.Optional[dict] = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		extra_embedding: tp.Optional[tp.Union[jnp.ndarray, None]] = None,
		add_params_field: bool = False,
		**kwargs,
	):
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		batch_size, sequence_length = input_ids.shape

		if position_ids is None:
			if past_key_values is not None:
				raise ValueError(
					"Make sure to provide `position_ids` when passing `past_key_values`."
				)

			position_ids = jnp.broadcast_to(
				jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
			)

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length))

		rngs = {}
		if dropout_rng is not None:
			rngs["dropout"] = dropout_rng

		inputs = (
			{
				"params": params or self.params,
			}
			if add_params_field
			else params or self.params
		)

		if past_key_values is not None:
			inputs["cache"] = past_key_values
			mutable = ["cache"]
		else:
			mutable = False
		kwargs.pop("deterministic", None)
		kwargs.pop("init_cache", None)
		child_call_args = dict(
			input_ids=jnp.array(input_ids, dtype="i4"),
			vision_mask=jnp.array(vision_mask, dtype="f4"),
			attention_mask=jnp.array(attention_mask, dtype="i4"),
			position_ids=jnp.array(position_ids, dtype="i4"),
			deterministic=not train,
			init_cache=False,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			extra_embedding=extra_embedding,
			**kwargs,
		)
		all_kwargs = {k: v for k, v in child_call_args.items()}
		filtered_kwargs = self._validate_signature(self.module.__call__, (), all_kwargs)
		outputs = self.module.apply(inputs, rngs=rngs, mutable=mutable, **filtered_kwargs)

		if past_key_values is not None and return_dict:
			outputs, past_key_values = outputs
			outputs["past_key_values"] = unfreeze(past_key_values["cache"])
			return outputs
		elif past_key_values is not None and not return_dict:
			outputs, past_key_values = outputs
			outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

		return outputs

	def prepare_inputs_for_generation(
		self,
		input_ids: jax.Array,
		max_length: int,
		attention_mask: tp.Optional[jax.Array] = None,
		vision_mask: tp.Optional[jax.Array] = None,
	):
		# initializing the cache
		batch_size, seq_length = input_ids.shape

		past_key_values = self.init_cache(batch_size, max_length)
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			position_ids = attention_mask.cumsum(axis=-1) - 1
			extended_attention_mask = lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
			)

		return {
			"past_key_values": past_key_values,
			"attention_mask": extended_attention_mask,
			"position_ids": position_ids,
			"vision_mask": vision_mask,
		}

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		return {
			"past_key_values": model_outputs.past_key_values,
			"position_ids": model_kwargs["position_ids"][:, -1:] + 1,
			"attention_mask": model_kwargs["attention_mask"],
			"vision_mask": model_kwargs["vision_mask"],
		}

	def _sample_vision(
		self,
		input_ids: None,
		max_length: tp.Optional[int] = None,
		pad_token_id: tp.Optional[int] = None,
		eos_token_id: tp.Optional[int] = None,
		prng_key: tp.Optional[jnp.ndarray] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		logits_warper: tp.Optional[FlaxLogitsProcessorList] = None,
		cfg_scales: jnp.ndarray = 1.0,
		trace: bool = True,
		params: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
		model_kwargs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
	):
		from easydel.inference.utils import SampleState

		# init values
		max_length = (
			max_length if max_length is not None else self.generation_config.max_length
		)
		pad_token_id = (
			pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
		)
		eos_token_id = (
			eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
		)
		prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

		batch_size, cur_len = input_ids.shape
		initial_len = cur_len

		eos_token_id = jnp.array(
			eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None
		)
		pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
		cur_len = jnp.array(cur_len)

		# per batch-item holding current token in loop.
		sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
		sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

		# per batch-item state bit indicating if sentence has finished.
		is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

		# For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
		# and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
		model = self.decode if self.config.is_encoder_decoder else self

		# initialize model specific kwargs
		model_kwargs = self.prepare_inputs_for_generation(
			input_ids, max_length, **model_kwargs
		)

		# initialize state
		state = SampleState(
			cur_len=cur_len,
			sequences=sequences,
			running_token=input_ids,
			is_sent_finished=is_sent_finished,
			prng_key=prng_key,
			model_kwargs=model_kwargs,
		)

		def sample_search_cond_fn(state):
			"""state termination condition fn."""
			has_reached_max_length = state.cur_len == max_length
			all_sequence_finished = jnp.all(state.is_sent_finished)
			finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
			return ~finish_generation

		def sample_search_body_fn(state):
			"""state update fn."""
			prng_key, prng_key_next = jax.random.split(state.prng_key)
			model_outputs = model(state.running_token, params=params, **state.model_kwargs)

			logits = model_outputs.logits[:, -1]
			cond_logits, uncond_logits = jnp.split(logits, 2, axis=0)
			logits = uncond_logits + cfg_scales[:, None] * (cond_logits - uncond_logits)

			# apply min_length, ...
			logits = logits_processor(state.sequences, logits, state.cur_len)
			# apply top_p, top_k, temperature
			logits = logits_warper(logits, logits, state.cur_len)

			next_token = jax.random.categorical(prng_key, logits, axis=-1)
			next_token = jax.lax.cond(
				(state.cur_len - initial_len + 1) % 257 == 0,
				lambda: jnp.full_like(next_token, 8192),
				lambda: next_token,
			)
			next_token = jnp.concatenate([next_token, next_token], axis=0)

			# next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
			next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
			next_token = next_token[:, None]

			next_sequences = lax.dynamic_update_slice(
				state.sequences, next_token, (0, state.cur_len)
			)
			next_model_kwargs = self.update_inputs_for_generation(
				model_outputs, state.model_kwargs
			)

			return SampleState(
				cur_len=state.cur_len + 1,
				sequences=next_sequences,
				running_token=next_token,
				is_sent_finished=next_is_sent_finished,
				model_kwargs=next_model_kwargs,
				prng_key=prng_key_next,
			)

		if input_ids.shape[1] > 1:
			state = sample_search_body_fn(state)

		if not trace:
			state = self._run_loop_in_debug(
				sample_search_cond_fn, sample_search_body_fn, state
			)
		else:
			state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

		return FlaxSampleOutput(sequences=state.sequences)

	def generate_vision(
		self,
		input_ids: jnp.ndarray,
		cfg_scales: jnp.ndarray,
		generation_config: tp.Optional["transformers.GenerationConfig"] = None,  # noqa #type:ignore
		prng_key: tp.Optional[jnp.ndarray] = None,
		trace: bool = True,
		params: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		**kwargs,
	):
		self._validate_model_class()

		if generation_config is None:
			if (
				self.generation_config._from_model_config
				and self.generation_config._original_object_hash == hash(self.generation_config)
			):
				from transformers import GenerationConfig

				new_generation_config = GenerationConfig.from_model_config(self.config)
				if new_generation_config != self.generation_config:
					logger.warn(
						"You have modified the pretrained model configuration to control generation. This is a"
						" deprecated strategy to control generation and will be removed soon, in a future version."
						" Please use and modify the model generation configuration (see"
						" https://huggingface.co/docs/transformers/generation_strategies#"
						"default-text-generation-configuration )"
					)
					self.generation_config = new_generation_config
			generation_config = self.generation_config
		import copy

		generation_config = copy.deepcopy(generation_config)
		model_kwargs = generation_config.update(
			**kwargs
		)  # All unused kwargs must be model kwargs
		generation_config.validate()
		self._validate_model_kwargs(model_kwargs.copy())

		logits_processor = (
			logits_processor if logits_processor is not None else FlaxLogitsProcessorList()
		)

		# set init values
		prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

		if (
			generation_config.pad_token_id is None
			and generation_config.eos_token_id is not None
		):
			if model_kwargs.get("attention_mask") is None:
				logger.warn(
					"The attention mask and the pad token id were not set. As a consequence, you may observe "
					"unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
				)
			eos_token_id = generation_config.eos_token_id
			if isinstance(eos_token_id, list):
				eos_token_id = eos_token_id[0]
			logger.warn(
				f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
			)
			generation_config.pad_token_id = eos_token_id

		if (
			generation_config.decoder_start_token_id is None
			and self.config.is_encoder_decoder
		):
			raise ValueError(
				"`decoder_start_token_id` has to be defined for encoder-decoder generation."
			)

		# decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
		if not self.config.is_encoder_decoder and not trace:
			if (
				generation_config.pad_token_id is not None
				and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
			):
				logger.warn(
					"A decoder-only architecture is being used, but right-padding was detected! For correct "
					"generation results, please set `padding_side='left'` when initializing the tokenizer."
				)

		batch_size = input_ids.shape[0]

		if self.config.is_encoder_decoder:
			# add encoder_outputs to model_kwargs
			if model_kwargs.get("encoder_outputs") is None:
				model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
					input_ids, params, model_kwargs
				)
			# prepare decoder_input_ids for generation
			input_ids = self._prepare_decoder_input_ids_for_generation(
				batch_size,
				decoder_start_token_id=generation_config.decoder_start_token_id,
				bos_token_id=generation_config.bos_token_id,
				model_kwargs=model_kwargs,
			)

		# Prepare `max_length` depending on other stopping criteria.
		input_ids_seq_length = input_ids.shape[-1]
		has_default_max_length = (
			kwargs.get("max_length") is None and generation_config.max_length is not None
		)
		if (
			has_default_max_length
			and generation_config.max_new_tokens is None
			and generation_config.max_length == 20
		):
			# 20 is the default max_length of the generation config
			logger.warn(
				f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
				"to control the generation length.  recommend setting `max_new_tokens` to control"
				" the maximum length of the generation.",
				UserWarning,
			)
		elif generation_config.max_new_tokens is not None:
			if not has_default_max_length and generation_config.max_length is not None:
				logger.warn(
					f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
					f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
					"Please refer to the documentation for more information. "
					"(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
				)
			generation_config.max_length = (
				generation_config.max_new_tokens + input_ids_seq_length
			)

		if (
			generation_config.min_length is not None
			and generation_config.min_length > generation_config.max_length
		):
			raise ValueError(
				f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
				f" the maximum length ({generation_config.max_length})"
			)
		if input_ids_seq_length >= generation_config.max_length:
			input_ids_string = (
				"decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
			)
			logger.warn(
				f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
				f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
				" increasing`max_new_tokens`."
			)

		logits_processor = self._get_logits_processor(
			generation_config=generation_config,
			input_ids_seq_length=input_ids_seq_length,
			logits_processor=logits_processor,
		)

		if not generation_config.do_sample and generation_config.num_beams == 1:
			raise NotImplementedError
		elif generation_config.do_sample and generation_config.num_beams == 1:
			logits_warper = self._get_logits_warper(generation_config=generation_config)
			return self._sample_vision(
				input_ids,
				generation_config.max_length,
				generation_config.pad_token_id,
				generation_config.eos_token_id,
				prng_key,
				logits_warper=logits_warper,
				logits_processor=logits_processor,
				cfg_scales=cfg_scales,
				trace=trace,
				params=params,
				model_kwargs=model_kwargs,
			)
		elif not generation_config.do_sample and generation_config.num_beams > 1:
			raise NotImplementedError
		else:
			raise NotImplementedError("`Beam sampling is currently not implemented.")


M = tp.TypeVar("M", bound=nn.Module)


def wrap_easydel_module(
	config_class: tp.Type[EasyDeLBaseConfig],
	base_model_prefix: str = "model",
):
	def wrapper(mdl: tp.Type[M]) -> tp.Type[EasyDeLBaseModule]:
		class_dict = {
			"config_class": config_class,
			"base_model_prefix": base_model_prefix,
			"__annotations__": {
				"config_class": tp.Type[EasyDeLBaseConfig],
				"base_model_prefix": str,
				"flax_module": tp.Type[M],
				"module_class": tp.Union[nn.Module, tp.Type[M]],
			},
		}

		for name, attr in mdl.__dict__.items():
			if not name.startswith("__"):
				class_dict[name] = attr

		WrappedModule = type(mdl.__name__, (EasyDeLBaseModule,), class_dict)
		WrappedModule.__module__ = mdl.__module__
		WrappedModule.__qualname__ = mdl.__qualname__
		WrappedModule.__doc__ = mdl.__doc__

		return WrappedModule

	return wrapper


def wrap_custom_easydel_module(
	base,
	config_class: tp.Type[EasyDeLBaseConfig],
	base_model_prefix: str = "model",
):
	def wrapper(mdl: tp.Type[M]) -> tp.Type[EasyDeLBaseModule]:
		class_dict = {
			"config_class": config_class,
			"base_model_prefix": base_model_prefix,
			"module_class": mdl,
			"flax_module": mdl,
			"__annotations__": {
				"config_class": tp.Type[EasyDeLBaseConfig],
				"base_model_prefix": str,
				"flax_module": tp.Type[M],
				"module_class": tp.Union[nn.Module, tp.Type[M]],
			},
		}

		for name, attr in mdl.__dict__.items():
			if not name.startswith("__"):
				class_dict[name] = attr

		WrappedModule = type(mdl.__name__, (base,), class_dict)
		WrappedModule.__module__ = mdl.__module__
		WrappedModule.__qualname__ = mdl.__qualname__
		WrappedModule.__doc__ = mdl.__doc__

		return WrappedModule

	return wrapper
