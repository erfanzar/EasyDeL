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
from easydel.infra.base_module import EasyDeLBaseModule

from .arctic import (
	ArcticConfig,
	ArcticForCausalLM,
	ArcticModel,
)
from .auto import (
	AutoEasyDeLConfig,
	AutoEasyDeLModelForCausalLM,
	AutoEasyDeLModelForImageTextToText,
	AutoEasyDeLModelForSeq2SeqLM,
	AutoEasyDeLModelForSpeechSeq2Seq,
	AutoEasyDeLModelForZeroShotImageClassification,
	AutoShardAndGatherFunctions,
	AutoStateForCausalLM,
	AutoStateForImageTextToText,
	AutoStateForSeq2SeqLM,
	AutoStateForSpeechSeq2Seq,
	AutoStateForZeroShotImageClassification,
	get_modules_by_type,
)
from .clip import (
	CLIPConfig,
	CLIPForImageClassification,
	CLIPModel,
	CLIPTextConfig,
	CLIPTextModel,
	CLIPTextModelWithProjection,
	CLIPVisionConfig,
	CLIPVisionModel,
)
from .cohere import (
	CohereConfig,
	CohereForCausalLM,
	CohereModel,
)
from .dbrx import (
	DbrxAttentionConfig,
	DbrxConfig,
	DbrxFFNConfig,
	DbrxForCausalLM,
	DbrxModel,
)
from .deepseek_v2 import (
	DeepseekV2Config,
	DeepseekV2ForCausalLM,
	DeepseekV2Model,
)
from .exaone import (
	ExaoneConfig,
	ExaoneForCausalLM,
	ExaoneModel,
)
from .falcon import (
	FalconConfig,
	FalconForCausalLM,
	FalconModel,
)
from .gemma import (
	GemmaConfig,
	GemmaForCausalLM,
	GemmaModel,
)
from .gemma2 import (
	Gemma2Config,
	Gemma2ForCausalLM,
	Gemma2Model,
)
from .gpt2 import (
	GPT2Config,
	GPT2LMHeadModel,
	GPT2Model,
)
from .gpt_j import (
	GPTJConfig,
	GPTJForCausalLM,
	GPTJModel,
)
from .gpt_neox import (
	GPTNeoXConfig,
	GPTNeoXForCausalLM,
	GPTNeoXModel,
)
from .grok_1 import (
	Grok1Config,
	Grok1ForCausalLM,
	Grok1Model,
)
from .internlm2 import (
	InternLM2Config,
	InternLM2ForCausalLM,
	InternLM2ForSequenceClassification,
	InternLM2Model,
)
from .llama import (
	LlamaConfig,
	LlamaForCausalLM,
	LlamaForSequenceClassification,
	LlamaModel,
)
from .mamba import (
	MambaConfig,
	MambaForCausalLM,
	MambaModel,
)
from .mamba2 import (
	Mamba2Config,
	Mamba2ForCausalLM,
	Mamba2Model,
)
from .mistral import (
	MistralConfig,
	MistralForCausalLM,
	MistralModel,
)
from .mixtral import (
	MixtralConfig,
	MixtralForCausalLM,
	MixtralModel,
)
from .mosaic_mpt import (
	MptAttentionConfig,
	MptConfig,
	MptForCausalLM,
	MptModel,
)
from .olmo import (
	OlmoConfig,
	OlmoForCausalLM,
	OlmoModel,
)
from .olmo2 import (
	Olmo2Config,
	Olmo2ForCausalLM,
	Olmo2Model,
)
from .openelm import (
	OpenELMConfig,
	OpenELMForCausalLM,
	OpenELMModel,
)
from .opt import (
	OPTConfig,
	OPTForCausalLM,
	OPTModel,
)
from .phi import (
	PhiConfig,
	PhiForCausalLM,
	PhiModel,
)
from .phi3 import (
	Phi3Config,
	Phi3ForCausalLM,
	Phi3Model,
)
from .phimoe import (
	PhiMoeConfig,
	PhiMoeForCausalLM,
	PhiMoeModel,
)
from .pixtral import (
	PixtralVisionConfig,
	PixtralVisionModel,
)
from .qwen2 import (
	Qwen2Config,
	Qwen2ForCausalLM,
	Qwen2ForSequenceClassification,
	Qwen2Model,
)
from .qwen2_moe import (
	Qwen2MoeConfig,
	Qwen2MoeForCausalLM,
	Qwen2MoeModel,
)
from .qwen2_vl import (
	Qwen2VLConfig,
	Qwen2VLForConditionalGeneration,
	Qwen2VLModel,
)
from .roberta import (
	RobertaConfig,
	RobertaForCausalLM,
	RobertaForMultipleChoice,
	RobertaForQuestionAnswering,
	RobertaForSequenceClassification,
	RobertaForTokenClassification,
)
from .stablelm import (
	StableLmConfig,
	StableLmForCausalLM,
	StableLmModel,
)
from .whisper import (
	WhisperConfig,
	WhisperForAudioClassification,
	WhisperForConditionalGeneration,
	WhisperTimeStampLogitsProcessor,
)
from .xerxes import (
	XerxesConfig,
	XerxesForCausalLM,
	XerxesModel,
)
