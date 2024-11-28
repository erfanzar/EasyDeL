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
	FlaxArcticModel,
)
from easydel.modules.auto_causal_language_model import (
	AutoEasyDeLModelForCausalLM,
	AutoStateForCausalLM,
)
from easydel.modules.auto_speech_seq2seq_model import (
	AutoEasyDeLModelForSpeechSeq2Seq,
	AutoStateForSpeechSeq2Seq,
)
from easydel.modules.auto_configuration import (
	AutoEasyDeLConfig,
	AutoShardAndGatherFunctions,
	get_modules_by_type,
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
	EasyDeLBaseModule,
	EasyDeLBaseVisionModule,
	EasyDeLBaseConfigDict,
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
