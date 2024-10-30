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
	ArcticConfig,
	FlaxArcticForCausalLM,
	FlaxArcticForCausalLMModule,
	FlaxArcticModel,
	FlaxArcticModule,
)
from easydel.modules.attention_module import (
	AttentionMechanisms,
	FlexibleAttentionModule,
	AttentionBenchmarker,
)
from easydel.modules.auto_models import (
	AutoEasyDeLConfig,
	AutoEasyDeLModelForCausalLM,
	AutoShardAndGatherFunctions,
	AutoStateForCausalLM,
	get_modules_by_type,
)
from easydel.modules.cohere import (
	CohereConfig,
	FlaxCohereForCausalLM,
	FlaxCohereForCausalLMModule,
	FlaxCohereModel,
	FlaxCohereModule,
)
from easydel.modules.dbrx import (
	DbrxAttentionConfig,
	DbrxConfig,
	DbrxFFNConfig,
	FlaxDbrxForCausalLM,
	FlaxDbrxForCausalLMModule,
	FlaxDbrxModel,
	FlaxDbrxModule,
)
from easydel.modules.deepseek_v2 import (
	DeepseekV2Config,
	FlaxDeepseekV2ForCausalLM,
	FlaxDeepseekV2ForCausalLMModule,
	FlaxDeepseekV2Model,
	FlaxDeepseekV2Module,
)
from easydel.modules.exaone import (
	ExaoneConfig,
	FlaxExaoneForCausalLM,
	FlaxExaoneForCausalLMModule,
	FlaxExaoneModel,
	FlaxExaoneModule,
)
from easydel.modules.falcon import (
	FalconConfig,
	FlaxFalconForCausalLM,
	FlaxFalconForCausalLMModule,
	FlaxFalconModel,
	FlaxFalconModule,
)
from easydel.modules.gemma import (
	FlaxGemmaForCausalLM,
	FlaxGemmaForCausalLMModule,
	FlaxGemmaModel,
	FlaxGemmaModule,
	GemmaConfig,
)
from easydel.modules.gemma2 import (
	FlaxGemma2ForCausalLM,
	FlaxGemma2ForCausalLMModule,
	FlaxGemma2Model,
	FlaxGemma2Module,
	Gemma2Config,
)
from easydel.modules.gpt2 import (
	FlaxGPT2LMHeadModel,
	FlaxGPT2LMHeadModule,
	FlaxGPT2Model,
	FlaxGPT2Module,
	GPT2Config,
)
from easydel.modules.gpt_j import (
	FlaxGPTJForCausalLM,
	FlaxGPTJForCausalLMModule,
	FlaxGPTJModel,
	FlaxGPTJModule,
	GPTJConfig,
)
from easydel.modules.gpt_neo_x import (
	FlaxGPTNeoXForCausalLM,
	FlaxGPTNeoXForCausalLMModule,
	FlaxGPTNeoXModel,
	FlaxGPTNeoXModule,
	GPTNeoXConfig,
)
from easydel.modules.grok_1 import (
	FlaxGrok1ForCausalLM,
	FlaxGrok1ForCausalLMModule,
	FlaxGrok1Model,
	FlaxGrok1Module,
	Grok1Config,
)
from easydel.modules.internlm2.modeling_internlm2_flax import (
	FlaxInternLM2ForCausalLM,
	FlaxInternLM2ForCausalLMModule,
	FlaxInternLM2ForSequenceClassification,
	FlaxInternLM2ForSequenceClassificationModule,
	FlaxInternLM2Model,
	FlaxInternLM2Module,
	InternLM2Config,
)
from easydel.modules.llama import (
	FlaxLlamaForCausalLM,
	FlaxLlamaForCausalLMModule,
	FlaxLlamaForSequenceClassification,
	FlaxLlamaForSequenceClassificationModule,
	FlaxLlamaModel,
	FlaxLlamaModule,
	FlaxVisionLlamaForCausalLM,
	LlamaConfig,
	VisionLlamaConfig,
)
 
from easydel.modules.mamba import (
	FlaxMambaCache,
	FlaxMambaForCausalLM,
	FlaxMambaForCausalLMModule,
	FlaxMambaModel,
	FlaxMambaModule,
	MambaConfig,
)
from easydel.modules.mamba2 import (
	FlaxMamba2Cache,
	FlaxMamba2ForCausalLM,
	FlaxMamba2ForCausalLMModule,
	FlaxMamba2Model,
	FlaxMamba2Module,
	Mamba2Config,
)
from easydel.modules.mistral import (
	FlaxMistralForCausalLM,
	FlaxMistralForCausalLMModule,
	FlaxMistralModel,
	FlaxMistralModule,
	FlaxVisionMistralForCausalLM,
	MistralConfig,
	VisionMistralConfig,
)
from easydel.modules.mixtral import (
	FlaxMixtralForCausalLM,
	FlaxMixtralForCausalLMModule,
	FlaxMixtralModel,
	FlaxMixtralModule,
	MixtralConfig,
)
from easydel.modules.modeling_utils import EDPretrainedConfig, EDPretrainedModel
from easydel.modules.mosaic_mpt import (
	FlaxMptForCausalLM,
	FlaxMptForCausalLMModule,
	FlaxMptModel,
	FlaxMptModule,
	MptAttentionConfig,
	MptConfig,
)
from easydel.modules.olmo import (
	FlaxOlmoForCausalLM,
	FlaxOlmoForCausalLMModule,
	FlaxOlmoModel,
	FlaxOlmoModule,
	OlmoConfig,
)
from easydel.modules.openelm import (
	FlaxOpenELMForCausalLM,
	FlaxOpenELMForCausalLMModule,
	FlaxOpenELMModel,
	FlaxOpenELMModule,
	OpenELMConfig,
)
from easydel.modules.opt import (
	FlaxOPTForCausalLM,
	FlaxOPTForCausalLMModule,
	FlaxOPTModel,
	FlaxOPTModule,
	OPTConfig,
)
from easydel.modules.palm import (
	FlaxPalmForCausalLM,
	FlaxPalmForCausalLMModule,
	FlaxPalmModel,
	FlaxPalmModule,
	PalmConfig,
)
from easydel.modules.phi import (
	FlaxPhiForCausalLM,
	FlaxPhiForCausalLMModule,
	FlaxPhiModel,
	FlaxPhiModule,
	PhiConfig,
)
from easydel.modules.phi3 import (
	FlaxPhi3ForCausalLM,
	FlaxPhi3ForCausalLMModule,
	FlaxPhi3Model,
	FlaxPhi3Module,
	Phi3Config,
)
from easydel.modules.phimoe import (
	FlaxPhiMoeForCausalLM,
	FlaxPhiMoeForCausalLMModule,
	FlaxPhiMoeModel,
	FlaxPhiMoeModule,
	PhiMoeConfig,
)
from easydel.modules.qwen1 import (
	FlaxQwen1ForCausalLM,
	FlaxQwen1ForCausalLMModule,
	FlaxQwen1ForSequenceClassification,
	FlaxQwen1ForSequenceClassificationModule,
	FlaxQwen1Model,
	FlaxQwen1Module,
	Qwen1Config,
)
from easydel.modules.qwen2 import (
	FlaxQwen2ForCausalLM,
	FlaxQwen2ForCausalLMModule,
	FlaxQwen2ForSequenceClassification,
	FlaxQwen2ForSequenceClassificationModule,
	FlaxQwen2Model,
	FlaxQwen2Module,
	Qwen2Config,
)
from easydel.modules.qwen2_moe import (
	FlaxQwen2MoeForCausalLM,
	FlaxQwen2MoeForCausalLMModule,
	FlaxQwen2MoeModel,
	FlaxQwen2MoeModule,
	Qwen2MoeConfig,
)
from easydel.modules.stablelm import (
	FlaxStableLmForCausalLM,
	FlaxStableLmForCausalLMModule,
	FlaxStableLmModel,
	FlaxStableLmModule,
	StableLmConfig,
)
from easydel.modules.t5 import (
	FlaxT5ForConditionalGeneration,
	FlaxT5ForConditionalGenerationModule,
	FlaxT5Model,
	FlaxT5Module,
	T5Config,
)
from easydel.modules.whisper import (
	FlaxWhisperForAudioClassification,
	FlaxWhisperForAudioClassificationModule,
	FlaxWhisperForConditionalGeneration,
	FlaxWhisperForConditionalGenerationModule,
	FlaxWhisperTimeStampLogitsProcessor,
	WhisperConfig,
)
from easydel.modules.xerxes import (
	FlaxXerxesForCausalLM,
	FlaxXerxesForCausalLMModule,
	FlaxXerxesModel,
	FlaxXerxesModule,
	XerxesConfig,
)
