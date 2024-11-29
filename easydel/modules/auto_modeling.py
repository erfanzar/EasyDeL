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

import os
from typing import (
	Any,
	Callable,
	List,
	Mapping,
	Optional,
	Sequence,
	Tuple,
	Union,
)

import jax.numpy
from jax.sharding import PartitionSpec

from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLPlatforms,
	EasyDeLQuantizationMethods,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.modules.factory import TaskType
from easydel.modules.modeling_utils import (
	EasyDeLBaseModule,
)


class BaseAutoEasyModel:
	model_task: TaskType

	@classmethod
	def _from_easydel_params(
		cls,
		pretrained_model_name_or_path,
		dtype: jax.numpy.dtype,
		param_dtype: jax.numpy.dtype,
		precision: Optional[jax.lax.Precision],
		sharding_axis_dims: Sequence[int],
		sharding_axis_names: Sequence[str],
		partition_axis: PartitionAxis,
		input_shape: Tuple[int, int],
		shard_fns: Optional[Mapping[tuple, Callable] | dict],
		quantization_method: Optional[EasyDeLQuantizationMethods],
		quantization_platform: Optional[EasyDeLPlatforms],
		backend: Optional[EasyDeLBackends],
		platform: Optional[EasyDeLPlatforms],
		bit_targeted_params: Optional[List[str]],
		quantization_block_size: int,
		config_kwargs: Optional[Mapping[str, Any]],
		auto_shard_params: bool,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]],
		safe: bool,
		**kwargs,
	):
		return EasyDeLBaseModule.from_pretrained(
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			input_shape=input_shape,
			dtype=dtype,
			precision=precision,
			param_dtype=param_dtype,
			partition_axis=partition_axis,
			auto_shard_params=auto_shard_params,
			shard_fns=shard_fns,
			sharding_axis_dims=sharding_axis_dims,
			sharding_axis_names=sharding_axis_names,
			backend=backend,
			platform=platform,
			config_kwargs=config_kwargs,
			partition_rules=partition_rules,
			quantization_method=quantization_method,
			quantization_platform=quantization_platform,
			bit_targeted_params=bit_targeted_params,
			quantization_block_size=quantization_block_size,
			safe=safe,
			model_task=cls.model_task,
			**kwargs,
		)

	@classmethod
	def _is_easydel(
		cls,
		pretrained_model_name_or_path,
		FLAX_WEIGHTS_NAME="easydel-model.parameters",
		cache_dir: Optional[Union[str, os.PathLike]] = None,
		force_download: bool = False,
		local_files_only: bool = False,
		token: Optional[Union[str, bool]] = None,
		revision: str = "main",
	):
		from transformers.utils import cached_file as _cached_file
		from transformers.utils import download_url as _download_url
		from transformers.utils import is_remote_url as _is_remote_url

		proxies = None
		subfolder = ""
		commit_hash = None
		pretrained_model_name_or_path = str(pretrained_model_name_or_path)
		if os.path.isdir(pretrained_model_name_or_path):
			if os.path.isfile(
				os.path.join(
					pretrained_model_name_or_path,
					subfolder,
					FLAX_WEIGHTS_NAME,
				)
			):
				archive_file = os.path.join(  # noqa
					pretrained_model_name_or_path,
					subfolder,
					FLAX_WEIGHTS_NAME,
				)
			else:
				raise EnvironmentError(
					f"Error no file named {FLAX_WEIGHTS_NAME} found in"
					f" directory {pretrained_model_name_or_path}"
				)
		elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
			...
		elif _is_remote_url(pretrained_model_name_or_path):
			filename = pretrained_model_name_or_path
			resolved_archive_file = _download_url(pretrained_model_name_or_path)
		else:
			filename = FLAX_WEIGHTS_NAME
			try:
				cached_file_kwargs = {
					"cache_dir": cache_dir,
					"force_download": force_download,
					"proxies": proxies,
					"local_files_only": local_files_only,
					"token": token,
					"user_agent": {
						"file_type": "model",
						"framework": "flax",
						"from_auto_class": False,
					},
					"revision": revision,
					"subfolder": subfolder,
					"_raise_exceptions_for_gated_repo": False,
					"_raise_exceptions_for_missing_entries": False,
					"_commit_hash": commit_hash,
				}
				resolved_archive_file = _cached_file(
					pretrained_model_name_or_path,
					filename,
					**cached_file_kwargs,
				)

				if resolved_archive_file is None:
					return False
			except EnvironmentError:
				raise
			except Exception:
				return False
		return True
