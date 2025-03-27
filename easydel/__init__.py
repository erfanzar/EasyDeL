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

__version__ = "0.1.2"

import os as _os
import sys as _sys
import typing as _tp
from logging import getLogger as _getlogger

from .utils import (
	check_bool_flag as _check_bool_flag,
	LazyModule as _LazyModule,
	is_package_available as _is_package_available,
	get_logger as _get_logger,
)

from packaging.version import Version as _version

if _check_bool_flag("EASYDEL_AUTO", True):
	_sys.setrecursionlimit(10000)

	# Tell jax xla bridge to stay quiet and only yied warnings or errors.
	_getlogger("jax._src.xla_bridge").setLevel(30)
	_getlogger("jax._src.mesh_utils").setLevel(30)
	_getlogger("datasets").setLevel(30)

	# Taking care of some optional GPU FLAGs
	_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	_os.environ["KMP_AFFINITY"] = "noverbose"
	_os.environ["GRPC_VERBOSITY"] = "3"
	_os.environ["GLOG_minloglevel"] = "3"
	_os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
	_os.environ["CACHE_TRITON_KERNELS"] = "1"
	_os.environ["XLA_FLAGS"] = (
		_os.getenv("XLA_FLAGS", "") + " "
		"--xla_gpu_triton_gemm_any=True  "
		"--xla_gpu_enable_while_loop_double_buffering=true  "
		"--xla_gpu_enable_pipelined_all_gather=true  "
		"--xla_gpu_enable_pipelined_reduce_scatter=true  "
		"--xla_gpu_enable_pipelined_all_reduce=true  "
		"--xla_gpu_enable_pipelined_collectives=false   "
		"--xla_gpu_enable_reduce_scatter_combine_by_dim=false  "
		"--xla_gpu_enable_all_gather_combine_by_dim=false  "
		"--xla_gpu_enable_reduce_scatter_combine_by_dim=false  "
		"--xla_gpu_all_gather_combine_threshold_bytes=8589934592  "
		"--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592  "
		"--xla_gpu_all_reduce_combine_threshold_bytes=8589934592  "
		"--xla_gpu_multi_streamed_windowed_einsum=true  "
		"--xla_gpu_threshold_for_windowed_einsum_mib=0  "
		"--xla_gpu_enable_latency_hiding_scheduler=true  "
		"--xla_gpu_enable_command_buffer=  "
	)
	_os.environ["LIBTPU_INIT_ARGS"] = (
		_os.getenv("LIBTPU_INIT_ARGS", "") + " "
		"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 "
		"--xla_tpu_spmd_threshold_for_allgather_cse=10000  "
		"--xla_tpu_enable_latency_hiding_scheduler=true "
		"--xla_tpu_megacore_fusion_allow_ags=false "
		"--xla_enable_async_collective_permute=true "
		"--xla_tpu_enable_ag_backward_pipelining=true "
		"--xla_tpu_enable_data_parallel_all_reduce_opt=true "
		"--xla_tpu_data_parallel_opt_different_sized_ops=true "
		"--xla_tpu_enable_async_collective_fusion=true "
		"--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
		"--xla_tpu_overlap_compute_collective_tc=true "
		"--xla_enable_async_all_gather=true "
		"--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
		"TPU_MEGACORE=MEGACORE_DENSE "
	)
	_os.environ.update(
		{
			"NCCL_LL128_BUFFSIZE": "-2",
			"NCCL_LL_BUFFSIZE": "-2",
			"NCCL_PROTO": "SIMPLE,LL,LL128",
		}
	)
	_os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if _os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", None) is None:
		_os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
	if _os.getenv("JAX_TRACEBACK_FILTERING", None) is None:
		_os.environ["JAX_TRACEBACK_FILTERING"] = "off"


_import_structure = {
	"utils": ["traversals"],
	"inference": [
		"vInference",
		"vInferenceApiServer",
		"vInferenceConfig",
		"vInferencePreCompileConfig",
		"vWhisperInference",
		"vWhisperInferenceConfig",
	],
	"infra": [
		"escale",
		"PartitionAxis",
		"EasyDeLBaseConfig",
		"EasyDeLBaseConfigDict",
		"EasyDeLBaseModule",
		"LossConfig",
	],
	"infra.base_state": ["EasyDeLState"],
	"infra.errors": [
		"EasyDeLRuntimeError",
		"EasyDeLSyntaxRuntimeError",
		"EasyDeLTimerError",
	],
	"infra.etils": [
		"EasyDeLBackends",
		"EasyDeLGradientCheckPointers",
		"EasyDeLOptimizers",
		"EasyDeLPlatforms",
		"EasyDeLQuantizationMethods",
		"EasyDeLSchedulers",
	],
	"infra.factory": [
		"ConfigType",
		"TaskType",
		"register_config",
		"register_module",
	],
	"layers": [],
	"layers.attention": [
		"AttentionMechanisms",
		"AttentionMetadata",
		"AttentionRegistry",
		"FlexibleAttentionModule",
	],
	"modules": [],
	"modules.arctic": [
		"ArcticConfig",
		"ArcticForCausalLM",
		"ArcticModel",
	],
	"modules.auto": [
		"AutoEasyDeLConfig",
		"AutoEasyDeLModel",
		"AutoEasyDeLModelForCausalLM",
		"AutoEasyDeLModelForImageTextToText",
		"AutoEasyDeLModelForSeq2SeqLM",
		"AutoEasyDeLModelForSequenceClassification",
		"AutoEasyDeLModelForSpeechSeq2Seq",
		"AutoEasyDeLModelForZeroShotImageClassification",
		"AutoEasyDeLVisionModel",
		"AutoShardAndGatherFunctions",
		"AutoState",
		"AutoStateForCausalLM",
		"AutoStateForImageSequenceClassification",
		"AutoStateForImageTextToText",
		"AutoStateForSeq2SeqLM",
		"AutoStateForSpeechSeq2Seq",
		"AutoStateForZeroShotImageClassification",
		"AutoStateVisionModel",
		"get_modules_by_type",
	],
	"modules.aya_vision": [
		"AyaVisionConfig",
		"AyaVisionForConditionalGeneration",
	],
	"modules.clip": [
		"CLIPConfig",
		"CLIPForImageClassification",
		"CLIPModel",
		"CLIPTextConfig",
		"CLIPTextModel",
		"CLIPTextModelWithProjection",
		"CLIPVisionConfig",
		"CLIPVisionModel",
	],
	"modules.cohere": [
		"CohereConfig",
		"CohereForCausalLM",
		"CohereForSequenceClassification",
		"CohereModel",
	],
	"modules.cohere2": [
		"Cohere2Config",
		"Cohere2ForCausalLM",
		"Cohere2ForSequenceClassification",
		"Cohere2Model",
	],
	"modules.dbrx": [
		"DbrxAttentionConfig",
		"DbrxConfig",
		"DbrxFFNConfig",
		"DbrxForCausalLM",
		"DbrxForSequenceClassification",
		"DbrxModel",
	],
	"modules.deepseek_v2": [
		"DeepseekV2Config",
		"DeepseekV2ForCausalLM",
		"DeepseekV2Model",
	],
	"modules.deepseek_v3": [
		"DeepseekV3Config",
		"DeepseekV3ForCausalLM",
		"DeepseekV3Model",
	],
	"modules.exaone": [
		"ExaoneConfig",
		"ExaoneForCausalLM",
		"ExaoneForSequenceClassification",
		"ExaoneModel",
	],
	"modules.falcon": [
		"FalconConfig",
		"FalconForCausalLM",
		"FalconModel",
	],
	"modules.gemma": [
		"GemmaConfig",
		"GemmaForCausalLM",
		"GemmaForSequenceClassification",
		"GemmaModel",
	],
	"modules.gemma2": [
		"Gemma2Config",
		"Gemma2ForCausalLM",
		"Gemma2ForSequenceClassification",
		"Gemma2Model",
	],
	"modules.gemma3": [
		"Gemma3Config",
		"Gemma3ForCausalLM",
		"Gemma3ForConditionalGeneration",
		"Gemma3ForSequenceClassification",
		"Gemma3MultiModalProjector",
		"Gemma3TextConfig",
		"Gemma3TextModel",
	],
	"modules.gpt2": [
		"GPT2Config",
		"GPT2LMHeadModel",
		"GPT2Model",
	],
	"modules.gpt_j": [
		"GPTJConfig",
		"GPTJForCausalLM",
		"GPTJModel",
	],
	"modules.gpt_neox": [
		"GPTNeoXConfig",
		"GPTNeoXForCausalLM",
		"GPTNeoXModel",
	],
	"modules.grok_1": [
		"Grok1Config",
		"Grok1ForCausalLM",
		"Grok1Model",
	],
	"modules.internlm2": [
		"InternLM2Config",
		"InternLM2ForCausalLM",
		"InternLM2ForSequenceClassification",
		"InternLM2Model",
	],
	"modules.llama": [
		"LlamaConfig",
		"LlamaForCausalLM",
		"LlamaForSequenceClassification",
		"LlamaModel",
	],
	"modules.llava": [
		"LlavaConfig",
		"LlavaForConditionalGeneration",
	],
	"modules.mamba": [
		"MambaConfig",
		"MambaForCausalLM",
		"MambaModel",
	],
	"modules.mamba2": [
		"Mamba2Config",
		"Mamba2ForCausalLM",
		"Mamba2Model",
	],
	"modules.mistral": [
		"MistralConfig",
		"MistralForCausalLM",
		"MistralForSequenceClassification",
		"MistralModel",
	],
	"modules.mixtral": [
		"MixtralConfig",
		"MixtralForCausalLM",
		"MixtralForSequenceClassification",
		"MixtralModel",
	],
	"modules.mosaic_mpt": [
		"MptAttentionConfig",
		"MptConfig",
		"MptForCausalLM",
		"MptModel",
	],
	"modules.olmo": [
		"OlmoConfig",
		"OlmoForCausalLM",
		"OlmoModel",
	],
	"modules.olmo2": [
		"Olmo2Config",
		"Olmo2ForCausalLM",
		"Olmo2ForSequenceClassification",
		"Olmo2Model",
	],
	"modules.openelm": [
		"OpenELMConfig",
		"OpenELMForCausalLM",
		"OpenELMModel",
	],
	"modules.opt": [
		"OPTConfig",
		"OPTForCausalLM",
		"OPTModel",
	],
	"modules.phi": [
		"PhiConfig",
		"PhiForCausalLM",
		"PhiModel",
	],
	"modules.phi3": [
		"Phi3Config",
		"Phi3ForCausalLM",
		"Phi3Model",
	],
	"modules.phimoe": [
		"PhiMoeConfig",
		"PhiMoeForCausalLM",
		"PhiMoeModel",
	],
	"modules.pixtral": [
		"PixtralVisionConfig",
		"PixtralVisionModel",
	],
	"modules.qwen2": [
		"Qwen2Config",
		"Qwen2ForCausalLM",
		"Qwen2ForSequenceClassification",
		"Qwen2Model",
	],
	"modules.qwen2_moe": [
		"Qwen2MoeConfig",
		"Qwen2MoeForCausalLM",
		"Qwen2MoeForSequenceClassification",
		"Qwen2MoeModel",
	],
	"modules.qwen2_vl": [
		"Qwen2VLConfig",
		"Qwen2VLForConditionalGeneration",
		"Qwen2VLModel",
	],
	"modules.roberta": [
		"RobertaConfig",
		"RobertaForCausalLM",
		"RobertaForMultipleChoice",
		"RobertaForQuestionAnswering",
		"RobertaForSequenceClassification",
		"RobertaForTokenClassification",
	],
	"modules.siglip": [
		"SiglipConfig",
		"SiglipForImageClassification",
		"SiglipModel",
		"SiglipTextConfig",
		"SiglipTextModel",
		"SiglipVisionConfig",
		"SiglipVisionModel",
	],
	"modules.stablelm": [
		"StableLmConfig",
		"StableLmForCausalLM",
		"StableLmModel",
	],
	"modules.whisper": [
		"WhisperConfig",
		"WhisperForAudioClassification",
		"WhisperForConditionalGeneration",
		"WhisperTimeStampLogitsProcessor",
	],
	"modules.xerxes": [
		"XerxesConfig",
		"XerxesForCausalLM",
		"XerxesModel",
	],
	"modules.xerxes2": [
		"Xerxes2Config",
		"Xerxes2ForCausalLM",
		"Xerxes2Model",
	],
	"trainers": [
		"BaseTrainer",
		"DPOConfig",
		"DPOTrainer",
		"GRPOConfig",
		"GRPOTrainer",
		"JaxDistributedConfig",
		"ORPOConfig",
		"ORPOTrainer",
		"RewardConfig",
		"RewardTrainer",
		"SFTConfig",
		"SFTTrainer",
		"Trainer",
		"TrainingArguments",
		"pack_sequences",
	],
	"utils.parameters_transformation": [
		"module_to_huggingface_model",
		"module_to_torch",
		"torch_dict_to_easydel_params",
	],
}


if _tp.TYPE_CHECKING:
	from . import utils
	from .inference.vinference import (
		vInference,
		vInferenceApiServer,
		vInferenceConfig,
		vInferencePreCompileConfig,
		vWhisperInference,
		vWhisperInferenceConfig,
	)
	from .infra import (
		EasyDeLBaseConfig,
		EasyDeLBaseConfigDict,
		EasyDeLBaseModule,
		LossConfig,
	)
	from .infra.base_state import EasyDeLState
	from .infra.errors import (
		EasyDeLRuntimeError,
		EasyDeLSyntaxRuntimeError,
		EasyDeLTimerError,
	)
	from .infra.etils import (
		EasyDeLBackends,
		EasyDeLGradientCheckPointers,
		EasyDeLOptimizers,
		EasyDeLPlatforms,
		EasyDeLQuantizationMethods,
		EasyDeLSchedulers,
	)
	from .infra.factory import (
		ConfigType,
		TaskType,
		register_config,
		register_module,
	)
	from .layers.attention import (
		AttentionMechanisms,
		AttentionMetadata,
		AttentionRegistry,
		FlexibleAttentionModule,
	)
	from .modules.arctic import (
		ArcticConfig,
		ArcticForCausalLM,
		ArcticModel,
	)
	from .modules.auto import (
		AutoEasyDeLConfig,
		AutoEasyDeLModel,
		AutoEasyDeLModelForCausalLM,
		AutoEasyDeLModelForImageTextToText,
		AutoEasyDeLModelForSeq2SeqLM,
		AutoEasyDeLModelForSequenceClassification,
		AutoEasyDeLModelForSpeechSeq2Seq,
		AutoEasyDeLModelForZeroShotImageClassification,
		AutoEasyDeLVisionModel,
		AutoShardAndGatherFunctions,
		AutoState,
		AutoStateForCausalLM,
		AutoStateForImageSequenceClassification,
		AutoStateForImageTextToText,
		AutoStateForSeq2SeqLM,
		AutoStateForSpeechSeq2Seq,
		AutoStateForZeroShotImageClassification,
		AutoStateVisionModel,
		get_modules_by_type,
	)
	from .modules.aya_vision import (
		AyaVisionConfig,
		AyaVisionForConditionalGeneration,
	)
	from .modules.clip import (
		CLIPConfig,
		CLIPForImageClassification,
		CLIPModel,
		CLIPTextConfig,
		CLIPTextModel,
		CLIPTextModelWithProjection,
		CLIPVisionConfig,
		CLIPVisionModel,
	)
	from .modules.cohere import (
		CohereConfig,
		CohereForCausalLM,
		CohereForSequenceClassification,
		CohereModel,
	)
	from .modules.cohere2 import (
		Cohere2Config,
		Cohere2ForCausalLM,
		Cohere2ForSequenceClassification,
		Cohere2Model,
	)
	from .modules.dbrx import (
		DbrxAttentionConfig,
		DbrxConfig,
		DbrxFFNConfig,
		DbrxForCausalLM,
		DbrxForSequenceClassification,
		DbrxModel,
	)
	from .modules.deepseek_v2 import (
		DeepseekV2Config,
		DeepseekV2ForCausalLM,
		DeepseekV2Model,
	)
	from .modules.deepseek_v3 import (
		DeepseekV3Config,
		DeepseekV3ForCausalLM,
		DeepseekV3Model,
	)
	from .modules.exaone import (
		ExaoneConfig,
		ExaoneForCausalLM,
		ExaoneForSequenceClassification,
		ExaoneModel,
	)
	from .modules.falcon import (
		FalconConfig,
		FalconForCausalLM,
		FalconModel,
	)
	from .modules.gemma import (
		GemmaConfig,
		GemmaForCausalLM,
		GemmaForSequenceClassification,
		GemmaModel,
	)
	from .modules.gemma2 import (
		Gemma2Config,
		Gemma2ForCausalLM,
		Gemma2ForSequenceClassification,
		Gemma2Model,
	)
	from .modules.gemma3 import (
		Gemma3Config,
		Gemma3ForCausalLM,
		Gemma3ForConditionalGeneration,
		Gemma3ForSequenceClassification,
		Gemma3MultiModalProjector,
		Gemma3TextConfig,
		Gemma3TextModel,
	)
	from .modules.gpt2 import (
		GPT2Config,
		GPT2LMHeadModel,
		GPT2Model,
	)
	from .modules.gpt_j import (
		GPTJConfig,
		GPTJForCausalLM,
		GPTJModel,
	)
	from .modules.gpt_neox import (
		GPTNeoXConfig,
		GPTNeoXForCausalLM,
		GPTNeoXModel,
	)
	from .modules.grok_1 import (
		Grok1Config,
		Grok1ForCausalLM,
		Grok1Model,
	)
	from .modules.internlm2 import (
		InternLM2Config,
		InternLM2ForCausalLM,
		InternLM2ForSequenceClassification,
		InternLM2Model,
	)
	from .modules.llama import (
		LlamaConfig,
		LlamaForCausalLM,
		LlamaForSequenceClassification,
		LlamaModel,
	)
	from .modules.llava import (
		LlavaConfig,
		LlavaForConditionalGeneration,
	)
	from .modules.mamba import (
		MambaConfig,
		MambaForCausalLM,
		MambaModel,
	)
	from .modules.mamba2 import (
		Mamba2Config,
		Mamba2ForCausalLM,
		Mamba2Model,
	)
	from .modules.mistral import (
		MistralConfig,
		MistralForCausalLM,
		MistralForSequenceClassification,
		MistralModel,
	)
	from .modules.mixtral import (
		MixtralConfig,
		MixtralForCausalLM,
		MixtralForSequenceClassification,
		MixtralModel,
	)
	from .modules.mosaic_mpt import (
		MptAttentionConfig,
		MptConfig,
		MptForCausalLM,
		MptModel,
	)
	from .modules.olmo import (
		OlmoConfig,
		OlmoForCausalLM,
		OlmoModel,
	)
	from .modules.olmo2 import (
		Olmo2Config,
		Olmo2ForCausalLM,
		Olmo2ForSequenceClassification,
		Olmo2Model,
	)
	from .modules.openelm import (
		OpenELMConfig,
		OpenELMForCausalLM,
		OpenELMModel,
	)
	from .modules.opt import (
		OPTConfig,
		OPTForCausalLM,
		OPTModel,
	)
	from .modules.phi import (
		PhiConfig,
		PhiForCausalLM,
		PhiModel,
	)
	from .modules.phi3 import (
		Phi3Config,
		Phi3ForCausalLM,
		Phi3Model,
	)
	from .modules.phimoe import (
		PhiMoeConfig,
		PhiMoeForCausalLM,
		PhiMoeModel,
	)
	from .modules.pixtral import (
		PixtralVisionConfig,
		PixtralVisionModel,
	)
	from .modules.qwen2 import (
		Qwen2Config,
		Qwen2ForCausalLM,
		Qwen2ForSequenceClassification,
		Qwen2Model,
	)
	from .modules.qwen2_moe import (
		Qwen2MoeConfig,
		Qwen2MoeForCausalLM,
		Qwen2MoeForSequenceClassification,
		Qwen2MoeModel,
	)
	from .modules.qwen2_vl import (
		Qwen2VLConfig,
		Qwen2VLForConditionalGeneration,
		Qwen2VLModel,
	)
	from .modules.roberta import (
		RobertaConfig,
		RobertaForCausalLM,
		RobertaForMultipleChoice,
		RobertaForQuestionAnswering,
		RobertaForSequenceClassification,
		RobertaForTokenClassification,
	)
	from .modules.siglip import (
		SiglipConfig,
		SiglipForImageClassification,
		SiglipModel,
		SiglipTextConfig,
		SiglipTextModel,
		SiglipVisionConfig,
		SiglipVisionModel,
	)
	from .modules.stablelm import (
		StableLmConfig,
		StableLmForCausalLM,
		StableLmModel,
	)
	from .modules.whisper import (
		WhisperConfig,
		WhisperForAudioClassification,
		WhisperForConditionalGeneration,
		WhisperTimeStampLogitsProcessor,
	)
	from .modules.xerxes import (
		XerxesConfig,
		XerxesForCausalLM,
		XerxesModel,
	)
	from .modules.xerxes2 import (
		Xerxes2Config,
		Xerxes2ForCausalLM,
		Xerxes2Model,
	)
	from .trainers import (
		BaseTrainer,
		DPOConfig,
		DPOTrainer,
		GRPOConfig,
		GRPOTrainer,
		JaxDistributedConfig,
		ORPOConfig,
		ORPOTrainer,
		RewardConfig,
		RewardTrainer,
		SFTConfig,
		SFTTrainer,
		Trainer,
		TrainingArguments,
		pack_sequences,
	)
	from .utils import traversals
	from .utils.parameters_transformation import (
		module_to_huggingface_model,
		module_to_torch,
		torch_dict_to_easydel_params,
	)
else:
	_logger = _get_logger("EasyDeL")
	_sys.modules[__name__] = _LazyModule(
		__name__,
		globals()["__file__"],
		_import_structure,
		module_spec=__spec__,
		extra_objects={"__version__": __version__},
	)

	_targeted_versions = ["0.0.18.2"]

	from eformer import __version__ as _eform_version

	assert _version(_eform_version) in [
		_version(_targeted_version) for _targeted_version in _targeted_versions
	], (
		f"this version of EasyDeL is only compatible with eformer {', '.join(_targeted_versions)},"
		f" but found eformer {_eform_version}"
	)

	if not _is_package_available("torch"):
		_logger.warning(
			"please install `torch` (cpu) if you want to use "
			"`AutoEasyDeLModel*.from_torch_pretrained`"
		)
	if not _is_package_available("tensorflow") or not _is_package_available(
		"tensorflow_datasets"
	):
		_logger.warning(
			"please install `tensorflow` and `tensorflow-datasets` "
			"if you want to use `EasyDeL` Trainers"
		)
	del _logger
	del _version
	del _eform_version

del _check_bool_flag
del _LazyModule
del _is_package_available
