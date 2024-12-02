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

__version__ = "0.0.80"

import os as _os

if _os.environ.get("EASYDEL_AUTO", "true") in ["true", "1", "on", "yes"]:
	# Taking care of some optional GPU FLAGs
	_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	_os.environ["KMP_AFFINITY"] = "noverbose"
	_os.environ["GRPC_VERBOSITY"] = "3"
	_os.environ["GLOG_minloglevel"] = "3"
	_os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
	_os.environ["CACHE_TRITON_KERNELS"] = "1"
	_os.environ["XLA_FLAGS"] = (
		_os.environ.get("XLA_FLAGS", "") + " "
		"--xla_gpu_triton_gemm_any=True \ "
		"--xla_gpu_enable_while_loop_double_buffering=true \ "
		"--xla_gpu_enable_pipelined_all_gather=true \ "
		"--xla_gpu_enable_pipelined_reduce_scatter=true \ "
		"--xla_gpu_enable_pipelined_all_reduce=true \ "
		"--xla_gpu_enable_pipelined_collectives=false  \ "
		"--xla_gpu_enable_reduce_scatter_combine_by_dim=false \ "
		"--xla_gpu_enable_all_gather_combine_by_dim=false \ "
		"--xla_gpu_enable_reduce_scatter_combine_by_dim=false \ "
		"--xla_gpu_all_gather_combine_threshold_bytes=8589934592 \ "
		"--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \ "
		"--xla_gpu_all_reduce_combine_threshold_bytes=8589934592 \ "
		"--xla_gpu_multi_streamed_windowed_einsum=true \ "
		"--xla_gpu_threshold_for_windowed_einsum_mib=0 \ "
		"--xla_gpu_enable_latency_hiding_scheduler=true \ "
		"--xla_gpu_enable_command_buffer= \ "
	)
	_os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if _os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", None) is None:
		_os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
	if _os.environ.get("JAX_TRACEBACK_FILTERING", None) is None:
		_os.environ["JAX_TRACEBACK_FILTERING"] = "off"
del _os

# EasyDel Imports
from packaging.version import Version as _Version

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import (
	EasyDeLRuntimeError,
	EasyDeLSyntaxRuntimeError,
	EasyDeLTimerError,
)
from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLGradientCheckPointers,
	EasyDeLOptimizers,
	EasyDeLPlatforms,
	EasyDeLQuantizationMethods,
	EasyDeLSchedulers,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.inference.vinference import (
	vInference,
	vInferenceApiServer,
	vInferenceConfig,
)
from easydel.inference.whisper_inference import (
	vWhisperInference,
	vWhisperInferenceConfig,
)
from easydel.layers.attention import (
	AttentionBenchmarker,
	AttentionMechanisms,
	FlexibleAttentionModule,
)
from easydel.modules.arctic import (
	ArcticConfig,
	FlaxArcticForCausalLM,
	FlaxArcticModel,
)
from easydel.modules.auto_causal_language_model import (
	AutoEasyDeLModelForCausalLM,
	AutoStateForCausalLM,
)
from easydel.modules.auto_configuration import (
	AutoEasyDeLConfig,
	AutoShardAndGatherFunctions,
	get_modules_by_type,
)
from easydel.modules.auto_speech_seq2seq_model import (
	AutoEasyDeLModelForSpeechSeq2Seq,
	AutoStateForSpeechSeq2Seq,
)
from easydel.modules.cohere import (
	CohereConfig,
	FlaxCohereForCausalLM,
	FlaxCohereModel,
)
from easydel.modules.dbrx import (
	DbrxAttentionConfig,
	DbrxConfig,
	DbrxFFNConfig,
	FlaxDbrxForCausalLM,
	FlaxDbrxModel,
)
from easydel.modules.deepseek_v2 import (
	DeepseekV2Config,
	FlaxDeepseekV2ForCausalLM,
	FlaxDeepseekV2Model,
)
from easydel.modules.exaone import (
	ExaoneConfig,
	FlaxExaoneForCausalLM,
	FlaxExaoneModel,
)
from easydel.modules.falcon import (
	FalconConfig,
	FlaxFalconForCausalLM,
	FlaxFalconModel,
)
from easydel.modules.gemma import (
	FlaxGemmaForCausalLM,
	FlaxGemmaModel,
	GemmaConfig,
)
from easydel.modules.gemma2 import (
	FlaxGemma2ForCausalLM,
	FlaxGemma2Model,
	Gemma2Config,
)
from easydel.modules.gpt2 import (
	FlaxGPT2LMHeadModel,
	FlaxGPT2Model,
	GPT2Config,
)
from easydel.modules.gpt_j import (
	FlaxGPTJForCausalLM,
	FlaxGPTJModel,
	GPTJConfig,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXForCausalLM,
	FlaxGPTNeoXModel,
	GPTNeoXConfig,
)
from easydel.modules.grok_1 import (
	FlaxGrok1ForCausalLM,
	FlaxGrok1Model,
	Grok1Config,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForCausalLM,
	FlaxInternLM2ForSequenceClassification,
	FlaxInternLM2Model,
	InternLM2Config,
)
from easydel.modules.llama import (
	FlaxLlamaForCausalLM,
	FlaxLlamaForSequenceClassification,
	FlaxLlamaModel,
	FlaxVisionLlamaForCausalLM,
	LlamaConfig,
	VisionLlamaConfig,
)
from easydel.modules.mamba import (
	FlaxMambaCache,
	FlaxMambaForCausalLM,
	FlaxMambaModel,
	MambaConfig,
)
from easydel.modules.mamba2 import (
	FlaxMamba2Cache,
	FlaxMamba2ForCausalLM,
	FlaxMamba2Model,
	Mamba2Config,
)
from easydel.modules.mistral import (
	FlaxMistralForCausalLM,
	FlaxMistralModel,
	FlaxVisionMistralForCausalLM,
	MistralConfig,
	VisionMistralConfig,
)
from easydel.modules.mixtral import (
	FlaxMixtralForCausalLM,
	FlaxMixtralModel,
	MixtralConfig,
)
from easydel.modules.modeling_utils import (
	EasyDeLBaseConfig,
	EasyDeLBaseConfigDict,
	EasyDeLBaseModule,
	EasyDeLBaseVisionModule,
)
from easydel.modules.mosaic_mpt import (
	FlaxMptForCausalLM,
	FlaxMptModel,
	MptAttentionConfig,
	MptConfig,
)
from easydel.modules.olmo import (
	FlaxOlmoForCausalLM,
	FlaxOlmoModel,
	OlmoConfig,
)
from easydel.modules.olmo2 import (
	FlaxOlmo2ForCausalLM,
	FlaxOlmo2Model,
	Olmo2Config,
)
from easydel.modules.openelm import (
	FlaxOpenELMForCausalLM,
	FlaxOpenELMModel,
	OpenELMConfig,
)
from easydel.modules.opt import (
	FlaxOPTForCausalLM,
	FlaxOPTModel,
	OPTConfig,
)
from easydel.modules.palm import (
	FlaxPalmForCausalLM,
	FlaxPalmModel,
	PalmConfig,
)
from easydel.modules.phi import (
	FlaxPhiForCausalLM,
	FlaxPhiModel,
	PhiConfig,
)
from easydel.modules.phi3 import (
	FlaxPhi3ForCausalLM,
	FlaxPhi3Model,
	Phi3Config,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeForCausalLM,
	FlaxPhiMoeModel,
	PhiMoeConfig,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForCausalLM,
	FlaxQwen2ForSequenceClassification,
	FlaxQwen2Model,
	Qwen2Config,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeForCausalLM,
	FlaxQwen2MoeModel,
	Qwen2MoeConfig,
)
from easydel.modules.stablelm import (
	FlaxStableLmForCausalLM,
	FlaxStableLmModel,
	StableLmConfig,
)
from easydel.modules.t5 import (
	FlaxT5ForConditionalGeneration,
	FlaxT5Model,
	T5Config,
)
from easydel.modules.whisper import (
	FlaxWhisperForAudioClassification,
	FlaxWhisperForConditionalGeneration,
	FlaxWhisperTimeStampLogitsProcessor,
	WhisperConfig,
)
from easydel.modules.xerxes import (
	FlaxXerxesForCausalLM,
	FlaxXerxesModel,
	XerxesConfig,
)
from easydel.modules.factory import (
	register_config,
	register_module,
	ConfigType,
	TaskType,
)
from easydel.smi import get_mem, initialise_tracking, run
from easydel.trainers import (
	BaseTrainer,
	CausalLanguageModelTrainer,
	CausalLMTrainerOutput,
	DPOConfig,
	DPOTrainer,
	DPOTrainerOutput,
	JaxDistributedConfig,
	LoraRaptureConfig,
	ORPOConfig,
	ORPOTrainer,
	ORPOTrainerOutput,
	Seq2SeqTrainer,
	SequenceClassificationTrainer,
	SequenceClassificationTrainerOutput,
	SFTTrainer,
	TrainingArguments,
	VisionCausalLanguageModelTrainer,
	VisionCausalLMTrainerOutput,
	pack_sequences,
)
from easydel.transform import (
	easystate_to_huggingface_model,
	easystate_to_torch,
	torch_dict_to_easydel_params,
)

_targeted_versions = ["0.0.91"]

from fjformer import __version__ as _fjformer_version

assert _Version(_fjformer_version) in [
	_Version(_targeted_version) for _targeted_version in _targeted_versions
], (
	f"this version of EasyDeL is only compatible with fjformer {', '.join(_targeted_versions)},"
	f" but found fjformer {_fjformer_version}"
)
import jax as _jax

if _jax.default_backend() == "gpu":
	try:
		import torch  # noqa #type:ignore

		del torch
	except ModuleNotFoundError:
		print(
			"UserWarning: please install `torch-cpu` since `easydel` "
			"uses `triton` and `triton` uses `torch` for autotuning.",
		)
del _jax
del _Version
del _fjformer_version
