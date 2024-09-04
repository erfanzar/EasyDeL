
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

from easydel.modules.arctic import (
	ArcticConfig as ArcticConfig,
)
from easydel.modules.arctic import (
	FlaxArcticForCausalLM as FlaxArcticForCausalLM,
)
from easydel.modules.arctic import (
	FlaxArcticForCausalLMModule as FlaxArcticForCausalLMModule,
)
from easydel.modules.arctic import (
	FlaxArcticModel as FlaxArcticModel,
)
from easydel.modules.arctic import (
	FlaxArcticModule as FlaxArcticModule,
)
from easydel.modules.attention_module import (
	AttentionMechanisms as AttentionMechanisms,
)
from easydel.modules.attention_module import (
	FlexibleAttentionModule as FlexibleAttentionModule,
)
from easydel.modules.auto_models import (
	AutoEasyDeLConfig as AutoEasyDeLConfig,
)
from easydel.modules.auto_models import (
	AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
)
from easydel.modules.auto_models import (
	AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
)
from easydel.modules.auto_models import (
	AutoStateForCausalLM as AutoStateForCausalLM,
)
from easydel.modules.auto_models import (
	get_modules_by_type as get_modules_by_type,
)
from easydel.modules.cohere import (
	CohereConfig as CohereConfig,
)
from easydel.modules.cohere import (
	FlaxCohereForCausalLM as FlaxCohereForCausalLM,
)
from easydel.modules.cohere import (
	FlaxCohereForCausalLMModule as FlaxCohereForCausalLMModule,
)
from easydel.modules.cohere import (
	FlaxCohereModel as FlaxCohereModel,
)
from easydel.modules.cohere import (
	FlaxCohereModule as FlaxCohereModule,
)
from easydel.modules.dbrx import (
	DbrxAttentionConfig as DbrxAttentionConfig,
)
from easydel.modules.dbrx import (
	DbrxConfig as DbrxConfig,
)
from easydel.modules.dbrx import (
	DbrxFFNConfig as DbrxFFNConfig,
)
from easydel.modules.dbrx import (
	FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
)
from easydel.modules.dbrx import (
	FlaxDbrxForCausalLMModule as FlaxDbrxForCausalLMModule,
)
from easydel.modules.dbrx import (
	FlaxDbrxModel as FlaxDbrxModel,
)
from easydel.modules.dbrx import (
	FlaxDbrxModule as FlaxDbrxModule,
)
from easydel.modules.deepseek_v2 import (
	DeepseekV2Config as DeepseekV2Config,
)
from easydel.modules.deepseek_v2 import (
	FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
)
from easydel.modules.deepseek_v2 import (
	FlaxDeepseekV2ForCausalLMModule as FlaxDeepseekV2ForCausalLMModule,
)
from easydel.modules.deepseek_v2 import (
	FlaxDeepseekV2Model as FlaxDeepseekV2Model,
)
from easydel.modules.deepseek_v2 import (
	FlaxDeepseekV2Module as FlaxDeepseekV2Module,
)
from easydel.modules.exaone import (
	ExaoneConfig as ExaoneConfig,
)
from easydel.modules.exaone import (
	FlaxExaoneForCausalLM as FlaxExaoneForCausalLM,
)
from easydel.modules.exaone import (
	FlaxExaoneForCausalLMModule as FlaxExaoneForCausalLMModule,
)
from easydel.modules.exaone import (
	FlaxExaoneModel as FlaxExaoneModel,
)
from easydel.modules.exaone import (
	FlaxExaoneModule as FlaxExaoneModule,
)
from easydel.modules.falcon import (
	FalconConfig as FalconConfig,
)
from easydel.modules.falcon import (
	FlaxFalconForCausalLM as FlaxFalconForCausalLM,
)
from easydel.modules.falcon import (
	FlaxFalconForCausalLMModule as FlaxFalconForCausalLMModule,
)
from easydel.modules.falcon import (
	FlaxFalconModel as FlaxFalconModel,
)
from easydel.modules.falcon import (
	FlaxFalconModule as FlaxFalconModule,
)
from easydel.modules.gemma import (
	FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
)
from easydel.modules.gemma import (
	FlaxGemmaForCausalLMModule as FlaxGemmaForCausalLMModule,
)
from easydel.modules.gemma import (
	FlaxGemmaModel as FlaxGemmaModel,
)
from easydel.modules.gemma import (
	FlaxGemmaModule as FlaxGemmaModule,
)
from easydel.modules.gemma import (
	GemmaConfig as GemmaConfig,
)
from easydel.modules.gemma2 import (
	FlaxGemma2ForCausalLM as FlaxGemma2ForCausalLM,
)
from easydel.modules.gemma2 import (
	FlaxGemma2ForCausalLMModule as FlaxGemma2ForCausalLMModule,
)
from easydel.modules.gemma2 import (
	FlaxGemma2Model as FlaxGemma2Model,
)
from easydel.modules.gemma2 import (
	FlaxGemma2Module as FlaxGemma2Module,
)
from easydel.modules.gemma2 import (
	Gemma2Config as Gemma2Config,
)
from easydel.modules.gpt2 import (
	FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
)
from easydel.modules.gpt2 import (
	FlaxGPT2LMHeadModule as FlaxGPT2LMHeadModule,
)
from easydel.modules.gpt2 import (
	FlaxGPT2Model as FlaxGPT2Model,
)
from easydel.modules.gpt2 import (
	FlaxGPT2Module as FlaxGPT2Module,
)
from easydel.modules.gpt2 import (
	GPT2Config as GPT2Config,
)
from easydel.modules.gpt_j import (
	FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
)
from easydel.modules.gpt_j import (
	FlaxGPTJForCausalLMModule as FlaxGPTJForCausalLMModule,
)
from easydel.modules.gpt_j import (
	FlaxGPTJModel as FlaxGPTJModel,
)
from easydel.modules.gpt_j import (
	FlaxGPTJModule as FlaxGPTJModule,
)
from easydel.modules.gpt_j import (
	GPTJConfig as GPTJConfig,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXForCausalLMModule as FlaxGPTNeoXForCausalLMModule,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXModel as FlaxGPTNeoXModel,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXModule as FlaxGPTNeoXModule,
)
from easydel.modules.gpt_neo_x import (
	GPTNeoXConfig as GPTNeoXConfig,
)
from easydel.modules.grok_1 import (
	FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
)
from easydel.modules.grok_1 import (
	FlaxGrok1ForCausalLMModule as FlaxGrok1ForCausalLMModule,
)
from easydel.modules.grok_1 import (
	FlaxGrok1Model as FlaxGrok1Model,
)
from easydel.modules.grok_1 import (
	FlaxGrok1Module as FlaxGrok1Module,
)
from easydel.modules.grok_1 import (
	Grok1Config as Grok1Config,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForCausalLM as FlaxInternLM2ForCausalLM,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForCausalLMModule as FlaxInternLM2ForCausalLMModule,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForSequenceClassification as FlaxInternLM2ForSequenceClassification,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForSequenceClassificationModule as FlaxInternLM2ForSequenceClassificationModule,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2Model as FlaxInternLM2Model,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2Module as FlaxInternLM2Module,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	InternLM2Config as InternLM2Config,
)
from easydel.modules.llama import (
	FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
)
from easydel.modules.llama import (
	FlaxLlamaForCausalLMModule as FlaxLlamaForCausalLMModule,
)
from easydel.modules.llama import (
	FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
)
from easydel.modules.llama import (
	FlaxLlamaForSequenceClassificationModule as FlaxLlamaForSequenceClassificationModule,
)
from easydel.modules.llama import (
	FlaxLlamaModel as FlaxLlamaModel,
)
from easydel.modules.llama import (
	FlaxLlamaModule as FlaxLlamaModule,
)
from easydel.modules.llama import (
	FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
)
from easydel.modules.llama import (
	LlamaConfig as LlamaConfig,
)
from easydel.modules.llama import (
	VisionLlamaConfig as VisionLlamaConfig,
)
from easydel.modules.lucid_transformer import (
	FlaxLTConfig as FlaxLTConfig,
)
from easydel.modules.lucid_transformer import (
	FlaxLTForCausalLM as FlaxLTForCausalLM,
)
from easydel.modules.lucid_transformer import (
	FlaxLTForCausalLMModule as FlaxLTForCausalLMModule,
)
from easydel.modules.lucid_transformer import (
	FlaxLTModel as FlaxLTModel,
)
from easydel.modules.lucid_transformer import (
	FlaxLTModule as FlaxLTModule,
)
from easydel.modules.mamba import (
	FlaxMambaCache as FlaxMambaCache,
)
from easydel.modules.mamba import (
	FlaxMambaForCausalLM as FlaxMambaForCausalLM,
)
from easydel.modules.mamba import (
	FlaxMambaForCausalLMModule as FlaxMambaForCausalLMModule,
)
from easydel.modules.mamba import (
	FlaxMambaModel as FlaxMambaModel,
)
from easydel.modules.mamba import (
	FlaxMambaModule as FlaxMambaModule,
)
from easydel.modules.mamba import (
	MambaConfig as MambaConfig,
)
from easydel.modules.mistral import (
	FlaxMistralForCausalLM as FlaxMistralForCausalLM,
)
from easydel.modules.mistral import (
	FlaxMistralForCausalLMModule as FlaxMistralForCausalLMModule,
)
from easydel.modules.mistral import (
	FlaxMistralModel as FlaxMistralModel,
)
from easydel.modules.mistral import (
	FlaxMistralModule as FlaxMistralModule,
)
from easydel.modules.mistral import (
	FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
)
from easydel.modules.mistral import (
	MistralConfig as MistralConfig,
)
from easydel.modules.mistral import (
	VisionMistralConfig as VisionMistralConfig,
)
from easydel.modules.mixtral import (
	FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
)
from easydel.modules.mixtral import (
	FlaxMixtralForCausalLMModule as FlaxMixtralForCausalLMModule,
)
from easydel.modules.mixtral import (
	FlaxMixtralModel as FlaxMixtralModel,
)
from easydel.modules.mixtral import (
	FlaxMixtralModule as FlaxMixtralModule,
)
from easydel.modules.mixtral import (
	MixtralConfig as MixtralConfig,
)
from easydel.modules.modeling_utils import (
	EDPretrainedConfig as EDPretrainedConfig,
)
from easydel.modules.modeling_utils import (
	EDPretrainedModel as EDPretrainedModel,
)
from easydel.modules.mosaic_mpt import (
	FlaxMptForCausalLM as FlaxMptForCausalLM,
)
from easydel.modules.mosaic_mpt import (
	FlaxMptForCausalLMModule as FlaxMptForCausalLMModule,
)
from easydel.modules.mosaic_mpt import (
	FlaxMptModel as FlaxMptModel,
)
from easydel.modules.mosaic_mpt import (
	FlaxMptModule as FlaxMptModule,
)
from easydel.modules.mosaic_mpt import (
	MptAttentionConfig as MptAttentionConfig,
)
from easydel.modules.mosaic_mpt import (
	MptConfig as MptConfig,
)
from easydel.modules.olmo import (
	FlaxOlmoForCausalLM as FlaxOlmoForCausalLM,
)
from easydel.modules.olmo import (
	FlaxOlmoForCausalLMModule as FlaxOlmoForCausalLMModule,
)
from easydel.modules.olmo import (
	FlaxOlmoModel as FlaxOlmoModel,
)
from easydel.modules.olmo import (
	FlaxOlmoModule as FlaxOlmoModule,
)
from easydel.modules.olmo import (
	OlmoConfig as OlmoConfig,
)
from easydel.modules.openelm import (
	FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
)
from easydel.modules.openelm import (
	FlaxOpenELMForCausalLMModule as FlaxOpenELMForCausalLMModule,
)
from easydel.modules.openelm import (
	FlaxOpenELMModel as FlaxOpenELMModel,
)
from easydel.modules.openelm import (
	FlaxOpenELMModule as FlaxOpenELMModule,
)
from easydel.modules.openelm import (
	OpenELMConfig as OpenELMConfig,
)
from easydel.modules.opt import (
	FlaxOPTForCausalLM as FlaxOPTForCausalLM,
)
from easydel.modules.opt import (
	FlaxOPTForCausalLMModule as FlaxOPTForCausalLMModule,
)
from easydel.modules.opt import (
	FlaxOPTModel as FlaxOPTModel,
)
from easydel.modules.opt import (
	FlaxOPTModule as FlaxOPTModule,
)
from easydel.modules.opt import (
	OPTConfig as OPTConfig,
)
from easydel.modules.palm import (
	FlaxPalmForCausalLM as FlaxPalmForCausalLM,
)
from easydel.modules.palm import (
	FlaxPalmForCausalLMModule as FlaxPalmForCausalLMModule,
)
from easydel.modules.palm import (
	FlaxPalmModel as FlaxPalmModel,
)
from easydel.modules.palm import (
	FlaxPalmModule as FlaxPalmModule,
)
from easydel.modules.palm import (
	PalmConfig as PalmConfig,
)
from easydel.modules.phi import (
	FlaxPhiForCausalLM as FlaxPhiForCausalLM,
)
from easydel.modules.phi import (
	FlaxPhiForCausalLMModule as FlaxPhiForCausalLMModule,
)
from easydel.modules.phi import (
	FlaxPhiModel as FlaxPhiModel,
)
from easydel.modules.phi import (
	FlaxPhiModule as FlaxPhiModule,
)
from easydel.modules.phi import (
	PhiConfig as PhiConfig,
)
from easydel.modules.phi3 import (
	FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
)
from easydel.modules.phi3 import (
	FlaxPhi3ForCausalLMModule as FlaxPhi3ForCausalLMModule,
)
from easydel.modules.phi3 import (
	FlaxPhi3Model as FlaxPhi3Model,
)
from easydel.modules.phi3 import (
	FlaxPhi3Module as FlaxPhi3Module,
)
from easydel.modules.phi3 import (
	Phi3Config as Phi3Config,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeForCausalLM as FlaxPhiMoeForCausalLM,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeForCausalLMModule as FlaxPhiMoeForCausalLMModule,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeModel as FlaxPhiMoeModel,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeModule as FlaxPhiMoeModule,
)
from easydel.modules.phimoe import (
	PhiMoeConfig as PhiMoeConfig,
)
from easydel.modules.qwen1 import (
	FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
)
from easydel.modules.qwen1 import (
	FlaxQwen1ForCausalLMModule as FlaxQwen1ForCausalLMModule,
)
from easydel.modules.qwen1 import (
	FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
)
from easydel.modules.qwen1 import (
	FlaxQwen1ForSequenceClassificationModule as FlaxQwen1ForSequenceClassificationModule,
)
from easydel.modules.qwen1 import (
	FlaxQwen1Model as FlaxQwen1Model,
)
from easydel.modules.qwen1 import (
	FlaxQwen1Module as FlaxQwen1Module,
)
from easydel.modules.qwen1 import (
	Qwen1Config as Qwen1Config,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForCausalLMModule as FlaxQwen2ForCausalLMModule,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForSequenceClassificationModule as FlaxQwen2ForSequenceClassificationModule,
)
from easydel.modules.qwen2 import (
	FlaxQwen2Model as FlaxQwen2Model,
)
from easydel.modules.qwen2 import (
	FlaxQwen2Module as FlaxQwen2Module,
)
from easydel.modules.qwen2 import (
	Qwen2Config as Qwen2Config,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeForCausalLMModule as FlaxQwen2MoeForCausalLMModule,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeModel as FlaxQwen2MoeModel,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeModule as FlaxQwen2MoeModule,
)
from easydel.modules.qwen2_moe import (
	Qwen2MoeConfig as Qwen2MoeConfig,
)
from easydel.modules.stablelm import (
	FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
)
from easydel.modules.stablelm import (
	FlaxStableLmForCausalLMModule as FlaxStableLmForCausalLMModule,
)
from easydel.modules.stablelm import (
	FlaxStableLmModel as FlaxStableLmModel,
)
from easydel.modules.stablelm import (
	FlaxStableLmModule as FlaxStableLmModule,
)
from easydel.modules.stablelm import (
	StableLmConfig as StableLmConfig,
)
from easydel.modules.t5 import (
	FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
)
from easydel.modules.t5 import (
	FlaxT5ForConditionalGenerationModule as FlaxT5ForConditionalGenerationModule,
)
from easydel.modules.t5 import (
	FlaxT5Model as FlaxT5Model,
)
from easydel.modules.t5 import (
	FlaxT5Module as FlaxT5Module,
)
from easydel.modules.t5 import (
	T5Config as T5Config,
)
from easydel.modules.whisper import (
	FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
)
from easydel.modules.whisper import (
	FlaxWhisperForAudioClassificationModule as FlaxWhisperForAudioClassificationModule,
)
from easydel.modules.whisper import (
	FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
)
from easydel.modules.whisper import (
	FlaxWhisperForConditionalGenerationModule as FlaxWhisperForConditionalGenerationModule,
)
from easydel.modules.whisper import (
	FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
)
from easydel.modules.whisper import (
	WhisperConfig as WhisperConfig,
)
from easydel.modules.xerxes import (
	FlaxXerxesForCausalLM as FlaxXerxesForCausalLM,
)
from easydel.modules.xerxes import (
	FlaxXerxesForCausalLMModule as FlaxXerxesForCausalLMModule,
)
from easydel.modules.xerxes import (
	FlaxXerxesModel as FlaxXerxesModel,
)
from easydel.modules.xerxes import (
	FlaxXerxesModule as FlaxXerxesModule,
)
from easydel.modules.xerxes import (
	XerxesConfig as XerxesConfig,
)
