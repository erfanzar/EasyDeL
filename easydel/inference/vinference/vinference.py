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

"""Module for text generation pipeline using JAX/Flax."""

import asyncio
import os
import pathlib
import pickle
import random
import time
import typing as tp
import warnings
from datetime import datetime
from functools import cached_property
from uuid import uuid4

import jax
import numpy as np
from chex import PRNGKey
from fjformer import GenerateRNG
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from pydantic import BaseModel

from easydel.etils.etils import get_logger
from easydel.infra import EasyDeLBaseModule
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import (
	load_compiled_fn,
	save_compiled_fn,
	smart_compile,
)

from ..utils import (
	SampleState,
	vInferenceConfig,
)
from ._fn import (
	causal_lm_first_iter_fn,
	causal_lm_iter_fn,
	get_compiled_funcs,
	measure_flops,
	put_compiled_funcs,
)
from .metrics import vInferenceMetrics

logger = get_logger(__name__)
TIME = str(datetime.fromtimestamp(time.time())).split(" ")[0]


class vInferenceMetaData(BaseModel):
	inference_name: str
	generation_config: vInferenceConfig
	precompiled_configs: set
	in_compiling_process: set
	input_partition_spec: jax.sharding.PartitionSpec
	uuid4: str
	model_config = dict(arbitrary_types_allowed=True)


class vInference:
	"""
	Class for performing text generation using a pre-trained language graphdef in EasyDeL.

	This class handles the generation process, including initialization, precompilation,
	and generating text in streaming chunks.
	"""

	def __init__(
		self,
		model: EasyDeLBaseModule,
		processor_class: ProcessingClassType,
		generation_config: tp.Optional[vInferenceConfig] = None,
		seed: tp.Optional[int] = None,
		input_partition_spec: tp.Optional[PartitionSpec] = None,
		max_new_tokens: int = 512,
		inference_name: tp.Optional[str] = None,
	):
		"""
		Initializes the vInference class.

		Args:
			model: The pre-trained language model.
			processor_class: The processor_class for the model.
			generation_config: The generation configuration.
			seed: The random seed for generation.
			input_partition_spec: The partitioning specification for input data.
			max_new_tokens: The maximum number of new tokens to generate.
		"""
		# fmt:off
		graphdef, graphstate = nn.split(model)
		self.graphdef = graphdef 
		self.graphstate = graphstate
		self.model=model 
		self.processor_class = processor_class
		self.generation_config = self._init_generation_config(generation_config, max_new_tokens)
		if seed is None:
			seed = random.randint(0, int(1e6))
		self._rng_generator = GenerateRNG(seed)
		self.input_partition_spec = input_partition_spec or PartitionSpec(("dp", "fsdp"), "sp")
		self.mesh = self.model.config.mesh
		self._precompile_lock = asyncio.Lock()
		self._precompiled_configs = set()
		self._in_compiling_process = set()
		self._init_variables()
		self._validate_token_ids()
		self._uuid4 = uuid4().hex 
		self._inference_name = inference_name or self._generate_inference_name(model)
		self.metrics = vInferenceMetrics(self._inference_name)
		# fmt:on

	@cached_property
	def tokenizer(self):
		from transformers import PreTrainedTokenizerBase

		if isinstance(self.processor_class, PreTrainedTokenizerBase):
			return self.processor_class

		from transformers import ProcessorMixin

		if isinstance(self.processor_class, ProcessorMixin):
			return self.processor_class.tokenizer
		raise ValueError("Unknown `processor_class` to extract `tokenizer` from.")

	@cached_property
	def _logits_warper(self):
		return self.generation_config.get_logits_warper()

	@cached_property
	def _logits_processor(self):
		return self.generation_config.get_logits_processor()

	def _generate_inference_name(self, model) -> str:
		"""
		Generate a standardized inference name combining model type, size, and timestamp.

		Format: {model_type}-{size_in_B}B-{timestamp}
		Example: llama-7.00B-20240311
		"""
		model_type = self._get_model_type(model)
		model_size = self._calculate_model_size(self.graphstate)
		timestamp = datetime.now().strftime("%Y%m%d")

		return f"{model_type}-{model_size}B-{timestamp}"

	def _get_model_type(self, model) -> str:
		"""Get the model type, with fallback to 'unknown' if not found."""
		return getattr(model.config, "model_type", "unknown").lower()

	def _calculate_model_size(self, graphstate) -> str:
		"""
		Calculate model size in billions of parameters.
		Returns formatted string with 2 decimal places.
		"""
		try:
			num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
			size_in_billions = num_params / 1e9
			return f"{size_in_billions:.2f}"
		except Exception as e:
			logger.warning(f"Failed to calculate model size: {e}")
			return "unknown"

	@property
	def inference_name(self):
		return self._inference_name

	@property
	def model_prefill_length(self) -> int:
		"""
		Calculate the maximum length available for input prefill by subtracting
		the maximum new tokens from the model's maximum sequence length.

		Returns:
				int: The maximum length available for input prefill

		Raises:
				ValueError: If no maximum sequence length configuration is found
		"""
		possible_length_attributes = [
			"granted_mask_max_position_embedding",
			"max_position_embedding",
			"max_sequence_length",
		]

		max_length = self._get_model_max_length(possible_length_attributes)

		if max_length is None:
			raise ValueError(
				"Could not determine model's maximum sequence length. "
				f"Looked for attributes: {', '.join(possible_length_attributes)}"
			)

		return max_length - self.generation_config.max_new_tokens

	def _get_model_max_length(self, attributes: list[str]) -> tp.Optional[int]:
		"""
		Find the first available maximum length configuration from a list of possible attributes.

		Args:
				attributes: tp.List of attribute names to check in order of preference

		Returns:
				tp.Optional[int]: The maximum length if found, None otherwise
		"""
		for attr in attributes:
			max_length = getattr(self.model.config, attr, None)
			if max_length is not None:
				return max_length
		return None

	def _init_generation_config(
		self, generation_config: tp.Optional[vInferenceConfig], max_new_tokens: int
	) -> vInferenceConfig:
		"""
		Initializes the generation configuration.

		Args:
			generation_config: The generation configuration.
			max_new_tokens: The maximum number of new tokens to generate.

		Returns:
			vInferenceConfig: The initialized generation configuration.
		"""
		if generation_config is None:
			if self.model.generation_config is not None:
				return vInferenceConfig(
					bos_token_id=self.model.generation_config.bos_token_id,
					eos_token_id=self.model.generation_config.eos_token_id,
					pad_token_id=self.model.generation_config.pad_token_id,
					top_k=self.model.generation_config.top_k,
					top_p=self.model.generation_config.top_p,
					temperature=self.model.generation_config.temperature,
					max_new_tokens=self.model.generation_config.max_new_tokens or max_new_tokens,
				)
			return vInferenceConfig(max_new_tokens=max_new_tokens)
		return generation_config

	def _init_variables(self):
		"""
		Initializes the shardings for input data.
		"""
		self.input_sharding = NamedSharding(
			spec=self.input_partition_spec,
			mesh=self.model.mesh,
		)
		self.empty_sharding = NamedSharding(
			spec=PartitionSpec(),
			mesh=self.model.mesh,
		)
		self.gen_input_sharding = NamedSharding(
			spec=PartitionSpec(self.input_partition_spec[0], None),
			mesh=self.model.mesh,
		)
		self._init_state = jax.jit(self._init_state_non_jit)

	def _init_state_non_jit(
		self,
		input_ids: jax.Array = None,
		rng: tp.Optional[PRNGKey] = None,
		**model_kwargs,
	):
		if rng is None:
			rng = self._rng_generator.rng
		pad_token_id = jnp.array(self.generation_config.pad_token_id, dtype=jnp.int32)
		batch_size, current_length = input_ids.shape
		max_length = current_length + self.generation_config.max_new_tokens
		current_length = jnp.array(current_length)
		sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
		sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
		is_sequence_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

		return SampleState(
			current_length=current_length,
			sequences=sequences,
			running_token=input_ids,
			is_sequence_finished=is_sequence_finished,
			prng_key=rng,
			model_kwargs=self.model.prepare_inputs_for_generation(
				input_ids=input_ids,
				max_length=max_length,
				**model_kwargs,
			),
			generated_tokens=0,
		)

	def _validate_token_ids(self):
		"""
		Validates the token IDs for padding, end-of-sequence, and beginning-of-sequence.
		"""

		if self.generation_config.pad_token_id is None:
			self.generation_config.pad_token_id = self.tokenizer.pad_token_id
		if self.generation_config.eos_token_id is None:
			self.generation_config.eos_token_id = self.tokenizer.eos_token_id
		if self.generation_config.bos_token_id is None:
			self.generation_config.bos_token_id = self.tokenizer.bos_token_id

		assert self.generation_config.pad_token_id is not None, (
			"`pad_token_id` cannot be None. "
			"(Set `tokenizer.pad_token_id = tokenizer.eos_token_id` if undefined"
			" or (`processing_class.tokenizer.pad_token_id = processing_class.tokenizer.eos_token_id`))"
		)
		assert (
			self.generation_config.eos_token_id is not None
		), "`eos_token_id` cannot be None."

	def generate(
		self,
		input_ids: jax.Array,
		attention_mask: tp.Optional[jax.Array] = None,
		**model_kwargs,
	) -> tp.Union[tp.Generator[SampleState, tp.Any, tp.Any], SampleState]:
		"""
		Generates text in streaming chunks.

		This function takes the input IDs, attention mask, and position IDs as input,
		precompiles the generation functions if necessary, and yields the generated text
		in streaming chunks.

		Args:
			input_ids: The input token IDs.
			attention_mask: The attention mask.

		Yields:
			SampleState: The generated text in streaming chunks.
		"""
		self.metrics.queue_size.labels(model_name=self.metrics.model_name).inc()

		try:
			with self.metrics.inference_latency.labels(
				model_name=self.metrics.model_name,
				stage="preprocessing",
			).time():
				input_ids = jnp.array(input_ids, dtype="i4", device=self.input_sharding)
				batch_size, sequence_length = input_ids.shape

				generate_func, interval_func = get_compiled_funcs(
					batch_size=batch_size,
					input_tokens_length=sequence_length,
					id=self._uuid4,
				)
				if attention_mask is None:
					warnings.warn(
						"`attention_mask` is not provided, it's recommended to "
						"pass an attention mask for better results.",
						stacklevel=1,
					)
					attention_mask = jnp.ones((batch_size, sequence_length), "i4")

				attention_mask = jnp.array(
					attention_mask,
					dtype="i4",
					device=self.input_sharding,
				)

				model_kwargs.update(dict(input_ids=input_ids, attention_mask=attention_mask))

				state = self._init_state(**model_kwargs)
			with self.metrics.inference_latency.labels(
				model_name=self.metrics.model_name,
				stage="inference",
			).time():
				(
					state,
					flop,
					generate_func_flops,
					generate_func_elapsed_time,
				) = measure_flops(
					generate_func,
					graphstate=self.graphstate,
					state=state,
				)
				interval_time = 0
				state.generate_func_flops = generate_func_flops
				if not state.is_sequence_finished:
					all_interval_func_flops = []
					for _ in range(self.generation_config._loop_rows):
						(
							state,
							flop,
							interval_func_flops,
							interval_func_elapsed_time,
						) = measure_flops(
							interval_func,
							graphstate=self.graphstate,
							state=state,
							loop_max_tokens=self.generation_config.streaming_chunks,
						)
						interval_time += interval_func_elapsed_time

						all_interval_func_flops.append(interval_func_flops)
						interval_func_flops = np.mean(all_interval_func_flops)
						state.generate_func_flops = generate_func_flops
						state.interval_func_flops = interval_func_flops
						state.tokens_pre_second = state.generated_tokens / interval_time
						yield state
						if state.is_sequence_finished:
							break
				else:
					yield state

			self.metrics.token_throughput.labels(
				model_name=self.metrics.model_name,
				operation="output",
			).inc(state.generated_tokens)
			self.metrics.generation_length.labels(
				model_name=self.metrics.model_name,
			).observe(state.generated_tokens)
			self.metrics.inference_requests.labels(
				model_name=self.metrics.model_name,
				status="success",
			).inc()
		except Exception as e:
			self.metrics.inference_requests.labels(
				model_name=self.metrics.model_name,
				status="error",
			).inc()
			raise e
		finally:
			self.metrics.queue_size.labels(model_name=self.metrics.model_name).dec()

	def _compile_and_lower_funs(self, batch_size: int, input_tokens_length: int):
		compiled_generate_func, compiled_interval_func = get_compiled_funcs(
			batch_size=batch_size,
			input_tokens_length=input_tokens_length,
			id=self._uuid4,
			safe=False,
		)
		do_compile = compiled_generate_func is None or compiled_interval_func is None
		if do_compile:
			model_kwargs = dict(
				input_ids=jnp.ones((batch_size, input_tokens_length), dtype="i4"),
				attention_mask=jnp.ones((batch_size, input_tokens_length), dtype="i4"),
			)
			state = self._init_state(**model_kwargs)
			compiled_generate_func = smart_compile(
				causal_lm_first_iter_fn.lower(
					graphdef=self.graphdef,
					graphstate=self.graphstate,
					state=state,
					generation_config=self.generation_config,
				),
				tag="vinference",
			)
			compiled_interval_func = smart_compile(
				causal_lm_iter_fn.lower(
					graphdef=self.graphdef,
					graphstate=self.graphstate,
					state=compiled_generate_func(
						graphstate=self.graphstate,
						state=state,
					),
					generation_config=self.generation_config,
					loop_max_tokens=self.generation_config.streaming_chunks,
				),
				tag="vinference",
			)

			del state
			put_compiled_funcs(
				compiled_generate_func=compiled_generate_func,
				compiled_interval_func=compiled_interval_func,
				batch_size=batch_size,
				input_tokens_length=input_tokens_length,
				id=self._uuid4,
			)

	def precompile(
		self,
		batch_size: int = 1,
		input_tokens_length: tp.Optional[int] = None,
	):
		"""
		Precompiles the generation functions for a given batch size and input length.

		This function checks if the generation functions have already been compiled for
		the given configuration. If not, it compiles them asynchronously and stores them
		in a cache.

		Args:
			batch_size: The batch size.
			input_tokens_length: The length of the input tokens.

		Returns:
			bool: True if precompilation was successful, False otherwise.
		"""
		if input_tokens_length is None:
			input_tokens_length = self.model_prefill_length
		config_key = (batch_size, input_tokens_length)

		if config_key in self._precompiled_configs:
			return True
		if config_key in self._in_compiling_process:
			time.sleep(5)
			return self.precompile(
				batch_size=batch_size,
				input_tokens_length=input_tokens_length,
			)
		else:
			with self.metrics.compilation_time.labels(
				model_name=self.metrics.model_name,
				function_name="_compile_and_lower_funs",
			).time():
				self._in_compiling_process.add(config_key)
				self._compile_and_lower_funs(
					batch_size=batch_size,
					input_tokens_length=input_tokens_length,
				)
				self._precompiled_configs.add(config_key)
		return True

	@tp.overload
	def count_tokens(self, messages: tp.List[tp.Dict[str, str]]): ...
	@tp.overload
	def count_tokens(self, text: str): ...

	def count_tokens(self, conv: tp.Union[str, tp.List[tp.Dict[str, str]]]) -> int:
		if isinstance(conv, list) and all(isinstance(item, dict) for item in conv):
			tokens = self.processor_class.apply_chat_template(
				conv,
				tokenize=True,
				apply_chat_template=True,
			)
			return len(tokens)
		else:
			tokens = self.tokenizer.encode(conv)
			return len(tokens)

	def save_inference(self, path: tp.Union[os.PathLike, str]):
		path = pathlib.Path(path)
		path.mkdir(exist_ok=True, parents=True)
		metadata = vInferenceMetaData(
			inference_name=self.inference_name,
			generation_config=self.generation_config,
			precompiled_configs=self._precompiled_configs,
			in_compiling_process=self._in_compiling_process,
			input_partition_spec=self.input_partition_spec,
			uuid4=self._uuid4,
		)
		for config_key in self._precompiled_configs:
			batch_size, input_tokens_length = config_key
			metafile = f"{metadata.uuid4}-{batch_size}-{input_tokens_length}"
			compiled_generation_fn, compiled_interval_fn = get_compiled_funcs(
				batch_size=batch_size,
				input_tokens_length=input_tokens_length,
				id=metadata.uuid4,
			)
			save_compiled_fn(path=path, fn=compiled_generation_fn, prefix=f"cgf-{metafile}")
			save_compiled_fn(path=path, fn=compiled_interval_fn, prefix=f"cif-{metafile}")

		metadata = pickle.dump(metadata, open(path / "config", "wb"))

	@classmethod
	def load_inference(
		cls,
		path: tp.Union[os.PathLike, str],
		model: EasyDeLBaseModule,
		processor_class: ProcessingClassType,
	):
		path = pathlib.Path(path)
		assert path.exists(), "provided path to vInference doesn't exists."
		metadata = pickle.load(open(path / "config", "rb"))
		for config_key in metadata.precompiled_configs:
			batch_size, input_tokens_length = config_key
			metafile = f"{metadata.uuid4}-{batch_size}-{input_tokens_length}"
			compiled_generation_fn = load_compiled_fn(path=path, prefix=f"cgf-{metafile}")
			compiled_interval_fn = load_compiled_fn(path=path, prefix=f"cif-{metafile}")
			put_compiled_funcs(
				compiled_generate_func=compiled_generation_fn,
				compiled_interval_func=compiled_interval_fn,
				batch_size=batch_size,
				input_tokens_length=input_tokens_length,
				id=metadata.uuid4,
			)
		self = cls(
			model=model,
			processor_class=processor_class,
			generation_config=metadata.generation_config,
			input_partition_spec=metadata.input_partition_spec,
			inference_name=metadata.inference_name,
		)
		self._uuid4 = metadata.uuid4
		return self
