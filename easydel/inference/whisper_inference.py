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
import math
import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
import requests
from flax import nnx as nn
from jax import numpy as jnp
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from easydel.modules.whisper import WhisperForConditionalGeneration
from easydel.utils.compiling_utils import get_safe_hash_int

if tp.TYPE_CHECKING:
	from transformers import GenerationConfig, WhisperProcessor, WhisperTokenizer
else:
	GenerationConfig, WhisperProcessor, WhisperTokenizer = [tp.Any] * 3


@partial(
	jax.jit,
	static_argnames=[
		"graphdef",
		"inference_config",
		"return_timestamps",
	],
)
def _compiled_generate(
	graphdef,
	graphstate,
	inference_config,
	input_features,
	decoder_input_ids,
	return_timestamps,
):
	model = nn.merge(graphdef, graphstate)
	return model._force_generate(
		input_features=input_features,
		forced_decoder_ids=decoder_input_ids,
		return_timestamps=return_timestamps,
		generation_config=inference_config.generation_config,
	)


@jax.tree_util.register_pytree_node_class
@dataclass
class vWhisperInferenceConfig:
	"""
	Configuration class for Whisper inference.

	Args:
	    batch_size (`int`, *optional*, defaults to 1):
	        Batch size used for inference.
	    max_length (`int`, *optional*):
	        Maximum sequence length for generation.
	    generation_config (`transformers.GenerationConfig`, *optional*):
	        Generation configuration object.
	    logits_processor (*optional*): Not used.
	    return_timestamps (`bool`, *optional*):
	        Whether to return timestamps with the transcribed text.
	    task (`str`, *optional*):
	        Task for the model (e.g., "transcribe", "translate").
	    language (`str`, *optional*):
	        Language of the input audio.
	    is_multilingual (`bool`, *optional*):
	        Whether the model is multilingual.
	"""

	batch_size: tp.Optional[int] = 1
	max_length: tp.Optional[int] = None
	generation_config: tp.Optional[GenerationConfig] = None
	logits_processor = None
	return_timestamps = None
	task = None
	language = None
	is_multilingual = None

	def tree_flatten(self):
		return (
			self.batch_size,
			self.max_length,
			self.generation_config,
			self.logits_processor,
			self.return_timestamps,
			self.task,
			self.language,
			self.is_multilingual,
		), {}

	@classmethod
	def tree_unflatten(cls, aux, children):
		return cls(*children)

	def __hash__(self):
		return get_safe_hash_int("".join(str(k) + str(v) for k, v in self.__dict__.items()))


class vWhisperInference:
	"""
	Whisper inference pipeline for performing speech-to-text transcription or translation.

	Args:
	    model (`WhisperForConditionalGeneration`):
	        The fine-tuned Whisper model to use for inference.
	    tokenizer (`WhisperTokenizer`):
	        Tokenizer for Whisper.
	    processor (`WhisperProcessor`):
	        Processor for Whisper.
	    inference_config (`vWhisperInferenceConfig`, *optional*):
	        Inference configuration.
	    dtype (`jax.typing.DTypeLike`, *optional*, defaults to `jnp.float32`):
	        Data type for computations.

	Example usage:

		>>> import easydel as ed
		>>> from transformers import WhisperTokenizer, WhisperProcessor

		>>> REPO_ID = "openai/whisper-small"  # Replace with your desired model

		>>> model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
		...		REPO_ID,
		...		# ... (config_kwargs as needed)
		>>> )
		>>> tokenizer = WhisperTokenizer.from_pretrained(REPO_ID)
		>>> processor = WhisperProcessor.from_pretrained(REPO_ID)

		>>> inference = vWhisperInference(
		...		model=model,
		...		tokenizer=tokenizer,
		...		processor=processor,
		...		dtype=jnp.float16,  # Or jnp.float32
		>>> )

		>>> result = inference("sample1.flac", return_timestamps=True)
		>>> print(result)


		>>> # Example using a URL:
		>>> result_url = inference(
		...		"https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/raw/main/common_voice_en_100038.mp3",
		...		return_timestamps=True
		>>> )
		>>> print(result_url)


		>>> # Example specifying language and task:
		>>> result_lang_task = inference(
		...		"sample1.flac", language="en", task="transcribe", return_timestamps=True
		>>> )
		>>> print(result_lang_task)

	"""

	def __init__(
		self,
		model: WhisperForConditionalGeneration,
		tokenizer: WhisperTokenizer,
		processor: WhisperProcessor,
		inference_config: tp.Optional[vWhisperInferenceConfig] = None,
		dtype: jax.typing.DTypeLike = jnp.float32,
	):
		# fmt:off
		if inference_config is None:
			inference_config = vWhisperInferenceConfig() 
		self.dtype = dtype
		self.processor = processor
		self.feature_extractor = self.processor.feature_extractor
		self.tokenizer = tokenizer
		self.model = model
		graphdef, graphstate = nn.split(model)
		self.graphdef = graphdef
		self.graphstate = graphstate
		generation_config = inference_config.generation_config or self.model.generation_config
		inference_config.generation_config = generation_config
		self.generation_config = generation_config
		self.max_length = inference_config.max_length or self.generation_config.max_length
		self.inference_config = inference_config
		self.generate_function = _compiled_generate
		# fmt:on

	def _generate(
		self,
		input_features: jax.Array,
		language: tp.Optional[str] = None,
		task: tp.Optional[str] = None,
		return_timestamps: bool = False,
	) -> jax.Array:
		forced_decoder_ids = self.get_decoder_input_ids(
			language=language,
			task=task,
			return_timestamps=return_timestamps,
		)
		output_sequences = self.generate_function(
			graphdef=self.graphdef,
			graphstate=self.graphstate,
			inference_config=self.inference_config,
			input_features=input_features,
			decoder_input_ids=forced_decoder_ids,
			return_timestamps=return_timestamps,
		).sequences
		return output_sequences

	def get_decoder_input_ids(
		self,
		generation_config: tp.Optional[GenerationConfig] = None,
		task: tp.Optional[str] = None,
		language: tp.Optional[str] = None,
		return_timestamps: bool = False,
	) -> list[tp.Tuple[int, int]]:
		generation_config = generation_config or self.model.generation_config
		is_multilingual = getattr(generation_config, "is_multilingual", None)
		decoder_input_ids = []
		if is_multilingual:
			if language is not None:
				language = language.lower()
				if language in generation_config.lang_to_id:
					language_token = language
				elif language in TO_LANGUAGE_CODE.values():
					language_token = f"<|{language}|>"
				elif language in TO_LANGUAGE_CODE:
					language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
				else:
					acceptable_languages = (
						list(TO_LANGUAGE_CODE.values())
						if len(language) == 2
						else list(generation_config.lang_to_id)
						if "<" in language or "|" in language or ">" in language
						else list(TO_LANGUAGE_CODE)
					)
					raise ValueError(
						f"Unsupported language: {language}. Language should be one of: {acceptable_languages}."
					)

				decoder_input_ids.append((1, generation_config.lang_to_id[language_token]))

			if task is not None:
				decoder_input_ids.append((2, generation_config.task_to_id[task]))
			else:
				decoder_input_ids.append((2, generation_config.task_to_id["transcribe"]))

		if (
			not return_timestamps
			and decoder_input_ids
			and decoder_input_ids[-1][0] != generation_config.no_timestamps_token_id
		):
			next_idx = (decoder_input_ids[-1][0] + 1) if decoder_input_ids else 1
			decoder_input_ids.append((next_idx, generation_config.no_timestamps_token_id))

		return decoder_input_ids

	def chunk_iter_with_batch(
		self,
		audio_array: jnp.ndarray,
		chunk_length: int,
		stride_left: int,
		stride_right: int,
		batch_size: int,
	):
		inputs_len = audio_array.shape[0]
		step = chunk_length - stride_left - stride_right
		all_chunk_start_idx = np.arange(0, inputs_len, step)
		num_samples = len(all_chunk_start_idx)
		num_batches = math.ceil(num_samples / batch_size)
		batch_idx = np.array_split(np.arange(num_samples), num_batches)

		for idx in batch_idx:
			chunk_start_idx = all_chunk_start_idx[idx]
			chunk_end_idx = chunk_start_idx + chunk_length
			chunks = [
				audio_array[chunk_start:chunk_end]
				for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)
			]
			processed = self.feature_extractor(
				chunks,
				sampling_rate=self.feature_extractor.sampling_rate,
				return_tensors="np",
			)

			yield {
				"stride": [
					(chunk_l, _stride_l, _stride_r)
					for chunk_l, _stride_l, _stride_r in zip(
						[chunk.shape[0] for chunk in chunks],
						np.where(chunk_start_idx == 0, 0, stride_left),
						np.where(
							np.where(
								stride_right > 0,
								chunk_end_idx > inputs_len,
								chunk_end_idx >= inputs_len,
							),
							0,
							stride_right,
						),
					)
				],
				**processed,
			}

	def _process_model_inputs(
		self,
		audio_input: tp.Union[
			str, bytes, np.ndarray, tp.Dict[str, tp.Union[np.ndarray, int]]
		],
		chunk_length_s: float = 30.0,
		stride_length_s: tp.Optional[tp.Union[float, list[float]]] = None,
		batch_size: tp.Optional[int] = None,
	):
		if isinstance(audio_input, str):
			if audio_input.startswith("http://") or audio_input.startswith("https://"):
				audio_input = requests.get(audio_input).content
			else:
				with open(audio_input, "rb") as f:
					audio_input = f.read()

		if isinstance(audio_input, bytes):
			audio_input = ffmpeg_read(audio_input, self.feature_extractor.sampling_rate)

		stride = None
		if isinstance(audio_input, dict):
			stride = audio_input.get("stride", None)
			if not ("sampling_rate" in audio_input and "array" in audio_input):
				raise ValueError(
					"When passing a dictionary to FlaxWhisperPipline, the dict needs to contain an 'array' key "
					"containing the numpy array representing the audio, and a 'sampling_rate' key "
					"containing the sampling rate associated with the audio array."
				)

			in_sampling_rate = audio_input.get("sampling_rate")
			audio_input = audio_input.get("array", None)

			if in_sampling_rate != self.feature_extractor.sampling_rate:
				try:
					import librosa
				except ImportError as err:
					raise ImportError(
						"To support resampling audio files, please install 'librosa' and 'soundfile'."
					) from err

				audio_input = librosa.resample(
					audio_input,
					orig_sr=in_sampling_rate,
					target_sr=self.feature_extractor.sampling_rate,
				)
				ratio = self.feature_extractor.sampling_rate / in_sampling_rate
			else:
				ratio = 1

		if not isinstance(audio_input, np.ndarray):
			raise ValueError(f"We expect a numpy ndarray as input, got `{type(audio_input)}`")
		if len(audio_input.shape) != 1:
			raise ValueError(
				"We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
			)

		if stride is not None:
			if stride[0] + stride[1] > audio_input.shape[0]:
				raise ValueError("Stride is too large for input")
			stride = (
				audio_input.shape[0],
				int(round(stride[0] * ratio)),
				int(round(stride[1] * ratio)),
			)

		if chunk_length_s:
			if stride_length_s is None:
				stride_length_s = chunk_length_s / 6

			if isinstance(stride_length_s, (int, float)):
				stride_length_s = [stride_length_s, stride_length_s]

			chunk_length = round(chunk_length_s * self.feature_extractor.sampling_rate)
			stride_left = round(stride_length_s[0] * self.feature_extractor.sampling_rate)
			stride_right = round(stride_length_s[1] * self.feature_extractor.sampling_rate)

			if chunk_length < stride_left + stride_right:
				raise ValueError("Chunk length must be superior to stride length")

			for item in self.chunk_iter_with_batch(
				audio_array=audio_input,
				chunk_length=chunk_length,
				stride_left=stride_left,
				stride_right=stride_right,
				batch_size=batch_size,
			):
				yield item
		else:
			processed = self.feature_extractor(
				audio_input,
				sampling_rate=self.feature_extractor.sampling_rate,
				return_tensors="np",
			)
			if stride is not None:
				processed["stride"] = stride
			yield processed

	def _process_model_outputs(
		self,
		model_outputs,
		return_timestamps: tp.Optional[bool] = None,
		return_language: tp.Optional[str] = None,
	):
		# fmt:off
		model_outputs = [
			dict(zip(output, t)) for output in model_outputs for t in zip(*output.values())
		]
		time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
		sampling_rate = self.feature_extractor.sampling_rate
		for output in model_outputs:
			if "stride" in output:
				chunk_length, stride_left, stride_right = output["stride"]
				output["stride"] = (
					chunk_length / sampling_rate,
					stride_left / sampling_rate,
					stride_right / sampling_rate,
				)

		text, optional = self.tokenizer._decode_asr(
			model_outputs,
			return_timestamps=return_timestamps,
			return_language=return_language,
			time_precision=time_precision,
		)
		return {"text": text, **optional}
		# fmt:on

	def _single_batch_process(
		self,
		model_inputs: tp.Dict[str, tp.Any],
		batch_size: int,
		language: tp.Optional[str] = None,
		task: tp.Optional[str] = None,
		return_timestamps: bool = False,
	):
		input_features = model_inputs.pop("input_features")
		input_batch_size = input_features.shape[0]
		if input_batch_size != batch_size:
			padding = np.zeros(
				[batch_size - input_batch_size, *input_features.shape[1:]], input_features.dtype
			)
			input_features = np.concatenate([input_features, padding])
		output_tokens = self._generate(
			input_features=input_features,
			language=language,
			task=task,
			return_timestamps=return_timestamps,
		)[:input_batch_size]

		output = {"tokens": output_tokens[:, None, :]}
		stride = model_inputs.pop("stride", None)
		if stride is not None:
			output["stride"] = stride
		return output

	def generate(
		self,
		audio_input: tp.Union[
			str, bytes, np.ndarray, tp.Dict[str, tp.Union[np.ndarray, int]]
		],
		chunk_length_s: float = 30.0,
		stride_length_s: tp.Optional[tp.Union[float, list[float]]] = None,
		batch_size: tp.Optional[int] = None,
		language: tp.Optional[str] = None,
		task: tp.Optional[str] = None,
		return_timestamps: tp.Optional[bool] = None,
	):
		"""
		Transcribe or translate audio input.

		Args:
		    audio_input (`tp.Union[str, bytes, np.ndarray, tp.Dict[str, tp.Union[np.ndarray, int]]]`):
		        Input audio. Can be a local file path, URL, bytes, numpy array, or a dictionary
		        containing the array and sampling rate.
		    chunk_length_s (`float`, *optional*, defaults to 30.0):
		        Length of audio chunks in seconds.
		    stride_length_s (`float` or `list[float]`, *optional*):
		        Stride length for chunking audio, in seconds.  Defaults to `chunk_length_s / 6`.
		    batch_size (`int`, *optional*):
		        Batch size for processing. Defaults to the `batch_size` in `inference_config`.
		    language (`str`, *optional*):
		        Language of the input audio. Defaults to the `language` in `inference_config`.
		    task (`str`, *optional*):
		        Task to perform (e.g., "transcribe", "translate"). Defaults to the `task` in `inference_config`.
		    return_timestamps (`bool`, *optional*):
		        Whether to return timestamps with the transcription.  Defaults to the `return_timestamps` in `inference_config`.

		Returns:
		    `dict`: A dictionary containing the transcribed text ("text") and optionally other information
		    like timestamps or detected language.
		"""
		batch_size = (
			batch_size if batch_size is not None else self.inference_config.batch_size
		)
		language = language if language is not None else self.inference_config.language
		task = task if task is not None else self.inference_config.task
		return_timestamps = (
			return_timestamps
			if return_timestamps is not None
			else self.inference_config.return_timestamps
		)

		dataloader = self._process_model_inputs(
			audio_input=audio_input,
			chunk_length_s=chunk_length_s,
			stride_length_s=stride_length_s,
			batch_size=batch_size,
		)
		model_outputs = []
		for model_inputs in dataloader:
			model_outputs.append(
				self._single_batch_process(
					model_inputs=model_inputs,
					batch_size=batch_size,
					language=language,
					task=task,
					return_timestamps=return_timestamps,
				)
			)
		return self._process_model_outputs(
			model_outputs,
			return_timestamps=return_timestamps,
		)

	__call__ = generate
