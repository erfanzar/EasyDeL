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

from easydel.infra.base_config import (
	EasyDeLBaseConfig,
	EasyDeLBaseConfigDict,
)
from easydel.infra.base_module import (
	EasyDeLBaseModule,
)
from easydel.modules.arctic import (
	ArcticConfig,
	ArcticForCausalLM,
	ArcticModel,
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
	CohereForCausalLM,
	CohereModel,
)
from easydel.modules.dbrx import (
	DbrxAttentionConfig,
	DbrxConfig,
	DbrxFFNConfig,
	DbrxForCausalLM,
	DbrxModel,
)
from easydel.modules.deepseek_v2 import (
	DeepseekV2Config,
	DeepseekV2ForCausalLM,
	DeepseekV2Model,
)
from easydel.modules.exaone import (
	ExaoneConfig,
	ExaoneForCausalLM,
	ExaoneModel,
)
from easydel.modules.falcon import (
	FalconConfig,
	FalconForCausalLM,
	FalconModel,
)
from easydel.modules.gemma import (
	GemmaConfig,
	GemmaForCausalLM,
	GemmaModel,
)
from easydel.modules.gemma2 import (
	Gemma2Config,
	Gemma2ForCausalLM,
	Gemma2Model,
)
from easydel.modules.gpt2 import (
	GPT2Config,
	GPT2LMHeadModel,
	GPT2Model,
)
from easydel.modules.gpt_j import (
	GPTJConfig,
	GPTJForCausalLM,
	GPTJModel,
)
from easydel.modules.gpt_neo_x import (
	GPTNeoXConfig,
	GPTNeoXForCausalLM,
	GPTNeoXModel,
)
from easydel.modules.grok_1 import (
	Grok1Config,
	Grok1ForCausalLM,
	Grok1Model,
)
from easydel.modules.internlm2 import (
	InternLM2Config,
	InternLM2ForCausalLM,
	InternLM2ForSequenceClassification,
	InternLM2Model,
)
from easydel.modules.llama import (
	LlamaConfig,
	LlamaForCausalLM,
	LlamaForSequenceClassification,
	LlamaModel,
)
from easydel.modules.mamba import (
	MambaConfig,
	MambaForCausalLM,
	MambaModel,
)
from easydel.modules.mamba2 import (
	Mamba2Config,
	Mamba2ForCausalLM,
	Mamba2Model,
)
from easydel.modules.mistral import (
	MistralConfig,
	MistralForCausalLM,
	MistralModel,
)
from easydel.modules.mixtral import (
	MixtralConfig,
	MixtralForCausalLM,
	MixtralModel,
)
from easydel.modules.mosaic_mpt import (
	MptAttentionConfig,
	MptConfig,
	MptForCausalLM,
	MptModel,
)
from easydel.modules.olmo import (
	OlmoConfig,
	OlmoForCausalLM,
	OlmoModel,
)
from easydel.modules.olmo2 import (
	Olmo2Config,
	Olmo2ForCausalLM,
	Olmo2Model,
)
from easydel.modules.openelm import (
	OpenELMConfig,
	OpenELMForCausalLM,
	OpenELMModel,
)
from easydel.modules.opt import (
	OPTConfig,
	OPTForCausalLM,
	OPTModel,
)
from easydel.modules.phi import (
	PhiConfig,
	PhiForCausalLM,
	PhiModel,
)
from easydel.modules.phi3 import (
	Phi3Config,
	Phi3ForCausalLM,
	Phi3Model,
)
from easydel.modules.phimoe import (
	PhiMoeConfig,
	PhiMoeForCausalLM,
	PhiMoeModel,
)
from easydel.modules.qwen2 import (
	Qwen2Config,
	Qwen2ForCausalLM,
	Qwen2ForSequenceClassification,
	Qwen2Model,
)
from easydel.modules.qwen2_moe import (
	Qwen2MoeConfig,
	Qwen2MoeForCausalLM,
	Qwen2MoeModel,
)

from easydel.modules.roberta import (
	RobertaForCausalLM,
	RobertaForMultipleChoice,
	RobertaForQuestionAnswering,
	RobertaForSequenceClassification,
	RobertaForTokenClassification,
	RobertaConfig,
)

from easydel.modules.stablelm import (
	StableLmConfig,
	StableLmForCausalLM,
	StableLmModel,
)
from easydel.modules.whisper import (
	WhisperConfig,
	WhisperForAudioClassification,
	WhisperForConditionalGeneration,
	WhisperTimeStampLogitsProcessor,
)
from easydel.modules.xerxes import (
	XerxesConfig,
	XerxesForCausalLM,
	XerxesModel,
)
