__version__ = "0.0.80"

import os as _os

if bool(
	_os.environ.get("EASYDEL_AUTO", "true")
):  # Taking care of some optional GPU FLAGs
	_os.environ["XLA_FLAGS"] = (
		_os.environ.get("XLA_FLAGS", "") + " "
		"--xla_gpu_enable_triton_softmax_fusion=true \ "
		"--xla_gpu_triton_gemm_any=True \ "
		"--xla_gpu_enable_async_collectives=true \ "
		"--xla_gpu_enable_latency_hiding_scheduler=true \ "
		"--xla_gpu_enable_highest_priority_async_stream=true \ "
	)
	_os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	_os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
	_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# EasyDel Imports
from easydel.etils.easystate import EasyDeLState as EasyDeLState
from easydel.etils.errors import (
	EasyDeLRuntimeError as EasyDeLRuntimeError,
	EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
	EasyDeLTimerError as EasyDeLTimerError,
)
from easydel.etils.etils import (
	EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
	EasyDeLOptimizers as EasyDeLOptimizers,
	EasyDeLSchedulers as EasyDeLSchedulers,
)
from easydel.etils.partition_module import PartitionAxis as PartitionAxis
from easydel.inference.generation_pipeline import (
	ChatPipeline as ChatPipeline,
	GenerationPipeline as GenerationPipeline,
	GenerationPipelineConfig as GenerationPipelineConfig,
)
from easydel.inference.serve_engine import (
	ApiEngine as ApiEngine,
	engine_client as engine_client,
)
from easydel.modules.arctic import (
	ArcticConfig as ArcticConfig,
	FlaxArcticForCausalLM as FlaxArcticForCausalLM,
	FlaxArcticForCausalLMModule as FlaxArcticForCausalLMModule,
	FlaxArcticModel as FlaxArcticModel,
	FlaxArcticModule as FlaxArcticModule,
)
from easydel.modules.attention_module import (
	AttentionMechanisms as AttentionMechanisms,
	FlexibleAttentionModule as FlexibleAttentionModule,
)
from easydel.modules.auto_models import (
	AutoEasyDeLConfig as AutoEasyDeLConfig,
	AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
	AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
	AutoStateForCausalLM as AutoStateForCausalLM,
	get_modules_by_type as get_modules_by_type,
)
from easydel.modules.cohere import (
	CohereConfig as CohereConfig,
	FlaxCohereModel as FlaxCohereModel,
	FlaxCohereModule as FlaxCohereModule,
	FlaxCohereForCausalLMModule as FlaxCohereForCausalLMModule,
	FlaxCohereForCausalLM as FlaxCohereForCausalLM,
)
from easydel.modules.dbrx import (
	DbrxConfig as DbrxConfig,
	DbrxFFNConfig as DbrxFFNConfig,
	DbrxAttentionConfig as DbrxAttentionConfig,
	FlaxDbrxModel as FlaxDbrxModel,
	FlaxDbrxModule as FlaxDbrxModule,
	FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
	FlaxDbrxForCausalLMModule as FlaxDbrxForCausalLMModule,
)
from easydel.modules.deepseek_v2 import (
	DeepseekV2Config as DeepseekV2Config,
	FlaxDeepseekV2Model as FlaxDeepseekV2Model,
	FlaxDeepseekV2Module as FlaxDeepseekV2Module,
	FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
	FlaxDeepseekV2ForCausalLMModule as FlaxDeepseekV2ForCausalLMModule,
)
from easydel.modules.falcon import (
	FalconConfig as FalconConfig,
	FlaxFalconForCausalLM as FlaxFalconForCausalLM,
	FlaxFalconForCausalLMModule as FlaxFalconForCausalLMModule,
	FlaxFalconModel as FlaxFalconModel,
	FlaxFalconModule as FlaxFalconModule,
)
from easydel.modules.gemma import (
	GemmaConfig as GemmaConfig,
	FlaxGemmaModel as FlaxGemmaModel,
	FlaxGemmaModule as FlaxGemmaModule,
	FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
	FlaxGemmaForCausalLMModule as FlaxGemmaForCausalLMModule,
)
from easydel.modules.gemma2 import (
	Gemma2Config as Gemma2Config,
	FlaxGemma2Model as FlaxGemma2Model,
	FlaxGemma2Module as FlaxGemma2Module,
	FlaxGemma2ForCausalLM as FlaxGemma2ForCausalLM,
	FlaxGemma2ForCausalLMModule as FlaxGemma2ForCausalLMModule,
)
from easydel.modules.gpt2 import (
	GPT2Config as GPT2Config,
	FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
	FlaxGPT2LMHeadModule as FlaxGPT2LMHeadModule,
	FlaxGPT2Model as FlaxGPT2Model,
	FlaxGPT2Module as FlaxGPT2Module,
)
from easydel.modules.gpt_j import (
	GPTJConfig as GPTJConfig,
	FlaxGPTJModel as FlaxGPTJModel,
	FlaxGPTJModule as FlaxGPTJModule,
	FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
	FlaxGPTJForCausalLMModule as FlaxGPTJForCausalLMModule,
)
from easydel.modules.gpt_neo_x import (
	GPTNeoXConfig as GPTNeoXConfig,
	FlaxGPTNeoXModel as FlaxGPTNeoXModel,
	FlaxGPTNeoXModule as FlaxGPTNeoXModule,
	FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
	FlaxGPTNeoXForCausalLMModule as FlaxGPTNeoXForCausalLMModule,
)
from easydel.modules.grok_1 import (
	Grok1Config as Grok1Config,
	FlaxGrok1Model as FlaxGrok1Model,
	FlaxGrok1Module as FlaxGrok1Module,
	FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
	FlaxGrok1ForCausalLMModule as FlaxGrok1ForCausalLMModule,
)

from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForSequenceClassification as FlaxInternLM2ForSequenceClassification,
	FlaxInternLM2ForSequenceClassificationModule as FlaxInternLM2ForSequenceClassificationModule,
	FlaxInternLM2ForCausalLM as FlaxInternLM2ForCausalLM,
	FlaxInternLM2ForCausalLMModule as FlaxInternLM2ForCausalLMModule,
	FlaxInternLM2Model as FlaxInternLM2Model,
	FlaxInternLM2Module as FlaxInternLM2Module,
	InternLM2Config as InternLM2Config,
)
from easydel.modules.llama import (
	LlamaConfig as LlamaConfig,
	VisionLlamaConfig as VisionLlamaConfig,
	FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
	FlaxLlamaForSequenceClassificationModule as FlaxLlamaForSequenceClassificationModule,
	FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
	FlaxLlamaForCausalLMModule as FlaxLlamaForCausalLMModule,
	FlaxLlamaModel as FlaxLlamaModel,
	FlaxLlamaModule as FlaxLlamaModule,
	FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
)
from easydel.modules.lucid_transformer import (
	FlaxLTConfig as FlaxLTConfig,
	FlaxLTModel as FlaxLTModel,
	FlaxLTModule as FlaxLTModule,
	FlaxLTForCausalLM as FlaxLTForCausalLM,
	FlaxLTForCausalLMModule as FlaxLTForCausalLMModule,
)
from easydel.modules.mamba import (
	MambaConfig as MambaConfig,
	FlaxMambaModel as FlaxMambaModel,
	FlaxMambaModule as FlaxMambaModule,
	FlaxMambaForCausalLM as FlaxMambaForCausalLM,
	FlaxMambaForCausalLMModule as FlaxMambaForCausalLMModule,
	FlaxMambaCache as FlaxMambaCache,
)
from easydel.modules.mistral import (
	MistralConfig as MistralConfig,
	FlaxMistralModel as FlaxMistralModel,
	FlaxMistralModule as FlaxMistralModule,
	FlaxMistralForCausalLM as FlaxMistralForCausalLM,
	FlaxMistralForCausalLMModule as FlaxMistralForCausalLMModule,
	FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
	VisionMistralConfig as VisionMistralConfig,
)
from easydel.modules.mixtral import (
	MixtralConfig as MixtralConfig,
	FlaxMixtralModel as FlaxMixtralModel,
	FlaxMixtralModule as FlaxMixtralModule,
	FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
	FlaxMixtralForCausalLMModule as FlaxMixtralForCausalLMModule,
)
from easydel.modules.modeling_utils import (
	EDPretrainedConfig as EDPretrainedConfig,
	EDPretrainedModel as EDPretrainedModel,
)
from easydel.modules.mosaic_mpt import (
	MptConfig as MptConfig,
	MptAttentionConfig as MptAttentionConfig,
	FlaxMptModel as FlaxMptModel,
	FlaxMptModule as FlaxMptModule,
	FlaxMptForCausalLM as FlaxMptForCausalLM,
	FlaxMptForCausalLMModule as FlaxMptForCausalLMModule,
)
from easydel.modules.olmo import (
	OlmoConfig as OlmoConfig,
	FlaxOlmoForCausalLM as FlaxOlmoForCausalLM,
	FlaxOlmoModel as FlaxOlmoModel,
	FlaxOlmoModule as FlaxOlmoModule,
	FlaxOlmoForCausalLMModule as FlaxOlmoForCausalLMModule,
)
from easydel.modules.openelm import (
	OpenELMConfig as OpenELMConfig,
	FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
	FlaxOpenELMForCausalLMModule as FlaxOpenELMForCausalLMModule,
	FlaxOpenELMModel as FlaxOpenELMModel,
	FlaxOpenELMModule as FlaxOpenELMModule,
)
from easydel.modules.opt import (
	OPTConfig as OPTConfig,
	FlaxOPTModel as FlaxOPTModel,
	FlaxOPTModule as FlaxOPTModule,
	FlaxOPTForCausalLM as FlaxOPTForCausalLM,
	FlaxOPTForCausalLMModule as FlaxOPTForCausalLMModule,
)
from easydel.modules.palm import (
	PalmConfig as PalmConfig,
	FlaxPalmModel as FlaxPalmModel,
	FlaxPalmModule as FlaxPalmModule,
	FlaxPalmForCausalLM as FlaxPalmForCausalLM,
	FlaxPalmForCausalLMModule as FlaxPalmForCausalLMModule,
)
from easydel.modules.phi import (
	PhiConfig as PhiConfig,
	FlaxPhiModel as FlaxPhiModel,
	FlaxPhiModule as FlaxPhiModule,
	FlaxPhiForCausalLM as FlaxPhiForCausalLM,
	FlaxPhiForCausalLMModule as FlaxPhiForCausalLMModule,
)
from easydel.modules.phi3 import (
	Phi3Config as Phi3Config,
	FlaxPhi3Model as FlaxPhi3Model,
	FlaxPhi3Module as FlaxPhi3Module,
	FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
	FlaxPhi3ForCausalLMModule as FlaxPhi3ForCausalLMModule,
)
from easydel.modules.phimoe import (
	PhiMoeConfig as PhiMoeConfig,
	FlaxPhiMoeModel as FlaxPhiMoeModel,
	FlaxPhiMoeModule as FlaxPhiMoeModule,
	FlaxPhiMoeForCausalLM as FlaxPhiMoeForCausalLM,
	FlaxPhiMoeForCausalLMModule as FlaxPhiMoeForCausalLMModule,
)

from easydel.modules.qwen1 import (
	Qwen1Config as Qwen1Config,
	FlaxQwen1Model as FlaxQwen1Model,
	FlaxQwen1Module as FlaxQwen1Module,
	FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
	FlaxQwen1ForCausalLMModule as FlaxQwen1ForCausalLMModule,
	FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
	FlaxQwen1ForSequenceClassificationModule as FlaxQwen1ForSequenceClassificationModule,
)
from easydel.modules.qwen2 import (
	Qwen2Config as Qwen2Config,
	FlaxQwen2Model as FlaxQwen2Model,
	FlaxQwen2Module as FlaxQwen2Module,
	FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
	FlaxQwen2ForCausalLMModule as FlaxQwen2ForCausalLMModule,
	FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
	FlaxQwen2ForSequenceClassificationModule as FlaxQwen2ForSequenceClassificationModule,
)
from easydel.modules.qwen2_moe import (
	Qwen2MoeConfig as Qwen2MoeConfig,
	FlaxQwen2MoeModel as FlaxQwen2MoeModel,
	FlaxQwen2MoeModule as FlaxQwen2MoeModule,
	FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
	FlaxQwen2MoeForCausalLMModule as FlaxQwen2MoeForCausalLMModule,
)
from easydel.modules.stablelm import (
	StableLmConfig as StableLmConfig,
	FlaxStableLmModel as FlaxStableLmModel,
	FlaxStableLmModule as FlaxStableLmModule,
	FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
	FlaxStableLmForCausalLMModule as FlaxStableLmForCausalLMModule,
)
from easydel.modules.t5 import (
	T5Config as T5Config,
	FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
	FlaxT5ForConditionalGenerationModule as FlaxT5ForConditionalGenerationModule,
	FlaxT5Model as FlaxT5Model,
	FlaxT5Module as FlaxT5Module,
)
from easydel.modules.whisper import (
	WhisperConfig as WhisperConfig,
	FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
	FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
	FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
	FlaxWhisperForConditionalGenerationModule as FlaxWhisperForConditionalGenerationModule,
	FlaxWhisperForAudioClassificationModule as FlaxWhisperForAudioClassificationModule,
)

from easydel.modules.xerxes import (
	XerxesConfig as XerxesConfig,
	FlaxXerxesModel as FlaxXerxesModel,
	FlaxXerxesModule as FlaxXerxesModule,
	FlaxXerxesForCausalLM as FlaxXerxesForCausalLM,
	FlaxXerxesForCausalLMModule as FlaxXerxesForCausalLMModule,
)

from easydel.modules.exaone import (
	ExaoneConfig as ExaoneConfig,
	FlaxExaoneForCausalLM as FlaxExaoneForCausalLM,
	FlaxExaoneForCausalLMModule as FlaxExaoneForCausalLMModule,
	FlaxExaoneModel as FlaxExaoneModel,
	FlaxExaoneModule as FlaxExaoneModule,
)

from easydel.smi import (
	get_mem as get_mem,
	initialise_tracking as initialise_tracking,
	run as run,
)
from easydel.trainers import (
	BaseTrainer as BaseTrainer,
	CausalLanguageModelTrainer as CausalLanguageModelTrainer,
	CausalLMTrainerOutput as CausalLMTrainerOutput,
	DPOTrainer as DPOTrainer,
	DPOTrainerOutput as DPOTrainerOutput,
	LoraRaptureConfig as LoraRaptureConfig,
	JaxDistributedConfig as JaxDistributedConfig,
	ORPOTrainer as ORPOTrainer,
	ORPOTrainerOutput as ORPOTrainerOutput,
	SFTTrainer as SFTTrainer,
	TrainArguments as TrainArguments,
	VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
	VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
	conversations_formatting_function as conversations_formatting_function,
	create_constant_length_dataset as create_constant_length_dataset,
	get_formatting_func_from_dataset as get_formatting_func_from_dataset,
	instructions_formatting_function as instructions_formatting_function,
)
from easydel.transform import (
	easystate_to_huggingface_model as easystate_to_huggingface_model,
	easystate_to_torch as easystate_to_torch,
	torch_dict_to_easydel_params as torch_dict_to_easydel_params,
)

from easydel import modules as modules
from easydel import etils as etils

from fjformer import __version__ as _fjv
from packaging.version import Version as _Version

_targeted_versions = ["0.0.78"]
assert _Version(_fjv) in [
	_Version(_targeted_version) for _targeted_version in _targeted_versions
], (
	f"this version of EasyDeL is only compatible with fjformer {', '.join(_targeted_versions)},"
	f" but found fjformer {_fjv}"
)
