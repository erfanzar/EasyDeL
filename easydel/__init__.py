# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

# pyright:reportUnusedImport=none
# pyright:reportImportCycles=none

"""EasyDeL: JAX/spectrax library for training and serving large language models.

Provides lazy-loaded access to all EasyDeL modules, model implementations,
trainers, inference engines, and utilities. Configures default environment
variables and JAX/XLA flags for optimal performance on TPU and GPU
backends when ``EASYDEL_AUTO`` is enabled (the default).

At import time this module also applies compatibility patches for
HuggingFace Transformers remote-code models that would otherwise fail
to load on newer Transformers versions or in environments lacking
optional dependencies (e.g. ``deepspeed``).
"""

__version__ = "0.4.0"

import sys as _sys
import typing as _tp

try:
    from eformer.loggings import get_logger as _get_logger
except ModuleNotFoundError:  # pragma: no cover
    from logging import getLogger as _getlogger

    def _get_logger(name: str | None = None):
        """Fallback logger factory used when ``eformer.loggings`` is unavailable.

        Args:
            name: Optional logger name. Defaults to the current module's name.

        Returns:
            A ``logging.Logger`` configured with the resolved name.
        """
        return _getlogger(name or __name__)


from packaging.version import Version as _version

from ._linkup import apply_linkup as _apply_linkup
from ._linkup import initialize_distributed as _initialize_distributed
from .utils import LazyModule as _LazyModule
from .utils import is_package_available as _is_package_available

_logger = _get_logger("EasyDeL")
_apply_linkup()

# If distributed initialization is explicitly enabled, do it before any
# compatibility/version imports can pull JAX-heavy modules into the process.
_distributed_init_enabled = _initialize_distributed(_logger)

_import_structure = {
    "utils": [
        "ModelConverter",
        "Registry",
        "StateDictConverter",
        "TensorConverter",
        "ejit",
        "ePath",
        "ePathLike",
        "is_inference_mode",
        "set_inference_mode",
        "traversals",
    ],
    "inference": [
        "EngineRequest",
        "EngineRequestStatus",
        "InferenceApiRouter",
        "JitableSamplingParams",
        "SamplingParams",
        "ToolParser",
        "ToolParserManager",
        "eSurge",
        "eSurgeApiServer",
        "eSurgeCacheRuntimeConfig",
        "eSurgeContextConfig",
        "eSurgeDistributedConfig",
        "eSurgeParsingConfig",
        "eSurgeRunner",
        "eSurgeRuntimeConfig",
        "eSurgeVisionConfig",
        "eSurgeWorkerConfig",
        "vWhisperInference",
        "vWhisperInferenceConfig",
    ],
    "inference.evaluations": ["eSurgeLMEvalAdapter"],
    "infra": [
        "EasyDeLBaseConfig",
        "EasyDeLBaseConfigDict",
        "EasyDeLBaseModule",
        "EasyDeLState",
        "LossConfig",
        "PartitionAxis",
        "PyTree",
        "Rngs",
        "auto_pytree",
        "eLargeModel",
        "BenchmarkConfig",
        "init_cluster",
        "sharding",
    ],
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
        "EasyDeLSchedulers",
    ],
    "infra.factory": [
        "ConfigType",
        "TaskType",
        "register_config",
        "register_module",
    ],
    "layers": [],
    "layers.quantization": [
        "EasyDeLQuantizationConfig",
        "EasyQuantizer",
        "QuantizationType",
        "TurboQuantConfig",
    ],
    "operations": [
        "AttentionConfig",
        "AttentionOutput",
        "AutoRegressiveDecodeAttn",
        "BaseOperationConfig",
        "BlockSparseAttn",
        "BlockSparseAttentionConfig",
        "FlashAttn",
        "PagedFlashAttn",
        "FlashAttentionConfig",
        "OperationImpl",
        "OperationMetadata",
        "OperationRegistry",
        "RaggedPageAttnV2",
        "RaggedPageAttnV3",
        "RaggedPageAttentionv2Config",
        "RaggedPageAttentionv3Config",
        "RingAttn",
        "RingAttentionConfig",
        "ScaledDotProductAttn",
        "ScaledDotProductAttentionConfig",
        "VanillaAttn",
    ],
    "layers.attention": [
        "AttentionMechanisms",
        "AttentionModule",
        "FlexibleAttentionModule",
    ],
    "layers.moe": ["MoEMethods"],
    "modules": [],
    "modules.arctic": [
        "ArcticConfig",
        "ArcticForCausalLM",
        "ArcticModel",
    ],
    "modules.auto": [
        "AutoEasyDeLAnyToAnyModel",
        "AutoEasyDeLConfig",
        "AutoEasyDeLModel",
        "AutoEasyDeLModelForCausalLM",
        "AutoEasyDeLModelForDiffusionLM",
        "AutoEasyDeLModelForEmbedding",
        "AutoEasyDeLModelForImageTextToText",
        "AutoEasyDeLModelForSeq2SeqLM",
        "AutoEasyDeLModelForEmbedding",
        "AutoEasyDeLModelForSequenceClassification",
        "AutoEasyDeLModelForSpeechSeq2Seq",
        "AutoEasyDeLModelForZeroShotImageClassification",
        "AutoEasyDeLVisionModel",
        "AutoShardAndGatherFunctions",
        "AutoState",
        "AutoStateAnyToAnyModel",
        "AutoStateForCausalLM",
        "AutoStateForDiffusionLM",
        "AutoStateForEmbedding",
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
        "AyaVisionModel",
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
    "modules.exaone4": [
        "Exaone4Config",
        "Exaone4ForCausalLM",
        "Exaone4ForSequenceClassification",
        "Exaone4Model",
    ],
    "modules.falcon": [
        "FalconConfig",
        "FalconForCausalLM",
        "FalconModel",
    ],
    "modules.falcon_h1": [
        "FalconH1Config",
        "FalconH1ForCausalLM",
        "FalconH1Model",
    ],
    "modules.falcon_mamba": [
        "FalconMambaConfig",
        "FalconMambaForCausalLM",
        "FalconMambaModel",
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
    "modules.gemma4": [
        "Gemma4Config",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4Model",
        "Gemma4MultimodalEmbedder",
        "Gemma4RMSNorm",
        "Gemma4TextConfig",
        "Gemma4TextModel",
        "Gemma4VisionConfig",
        "Gemma4VisionModel",
    ],
    "modules.gidd": [
        "GiddConfig",
        "GiddModel",
        "GiddForDiffusionLM",
    ],
    "modules.glm": [
        "GlmConfig",
        "GlmForCausalLM",
        "GlmForSequenceClassification",
        "GlmModel",
    ],
    "modules.glm4": [
        "Glm4Config",
        "Glm4ForCausalLM",
        "Glm4ForSequenceClassification",
        "Glm4Model",
    ],
    "modules.glm4_moe": [
        "Glm4MoeConfig",
        "Glm4MoeForCausalLM",
        "Glm4MoeForSequenceClassification",
        "Glm4MoeModel",
    ],
    "modules.glm4_moe_lite": [
        "Glm4MoeLiteConfig",
        "Glm4MoeLiteForCausalLM",
        "Glm4MoeLiteModel",
    ],
    "modules.glm_moe_dsa": [
        "GlmMoeDsaConfig",
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaModel",
    ],
    "modules.glm4v": [
        "Glm4vConfig",
        "Glm4vForConditionalGeneration",
        "Glm4vModel",
        "Glm4vTextConfig",
        "Glm4vTextModel",
        "Glm4vVisionConfig",
        "Glm4vVisionModel",
    ],
    "modules.glm4v_moe": [
        "Glm4vMoeConfig",
        "Glm4vMoeForConditionalGeneration",
        "Glm4vMoeModel",
        "Glm4vMoeTextConfig",
        "Glm4vMoeTextModel",
        "Glm4vMoeVisionConfig",
        "Glm4vMoeVisionModel",
    ],
    "modules.glm46v": [
        "Glm46VConfig",
        "Glm46VForConditionalGeneration",
        "Glm46VModel",
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
    "modules.gpt_oss": [
        "GptOssConfig",
        "GptOssForCausalLM",
        "GptOssForSequenceClassification",
        "GptOssModel",
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
    "modules.kimi_linear": [
        "KimiLinearConfig",
        "KimiLinearForCausalLM",
        "KimiLinearModel",
    ],
    "modules.kimi_vl": [
        "KimiVLConfig",
        "KimiVLForConditionalGeneration",
        "MoonViTConfig",
    ],
    "modules.llama": [
        "LlamaConfig",
        "LlamaForCausalLM",
        "LlamaForSequenceClassification",
        "LlamaModel",
    ],
    "modules.llama4": [
        "Llama4Config",
        "Llama4ForCausalLM",
        "Llama4ForConditionalGeneration",
        "Llama4ForSequenceClassification",
        "Llama4TextConfig",
        "Llama4TextModel",
        "Llama4VisionConfig",
        "Llama4VisionModel",
    ],
    "modules.llava": [
        "LlavaConfig",
        "LlavaForConditionalGeneration",
        "LlavaModel",
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
    "modules.minimax": [
        "MiniMaxConfig",
        "MiniMaxForCausalLM",
        "MiniMaxModel",
    ],
    "modules.mistral": [
        "MistralConfig",
        "MistralForCausalLM",
        "MistralForSequenceClassification",
        "MistralModel",
    ],
    "modules.mistral3": [
        "Mistral3Config",
        "Mistral3ForConditionalGeneration",
        "Mistral3Model",
        "Mistral3Tokenizer",
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
    "modules.olmo3": [
        "Olmo3Config",
        "Olmo3ForCausalLM",
        "Olmo3ForSequenceClassification",
        "Olmo3Model",
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
        "Qwen2ForEmbedding",
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
    "modules.qwen3_5": [
        "Qwen3_5Config",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MTPHead",
        "Qwen3_5MTPLayer",
        "Qwen3_5MTPOutput",
        "Qwen3_5Model",
        "Qwen3_5TextConfig",
        "Qwen3_5TextModel",
        "Qwen3_5VisionConfig",
    ],
    "modules.gemma4_assistant": [
        "Gemma4AssistantCentroidHead",
        "Gemma4AssistantConfig",
        "Gemma4AssistantForCausalLM",
        "Gemma4AssistantModel",
        "Gemma4AssistantOutput",
        "Gemma4AssistantTextConfig",
    ],
    "modules.qwen3_5_moe": [
        "Qwen3_5MoeConfig",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeModel",
        "Qwen3_5MoeTextConfig",
        "Qwen3_5MoeTextModel",
        "Qwen3_5MoeVisionConfig",
    ],
    "modules.qwen3": [
        "Qwen3Config",
        "Qwen3ForCausalLM",
        "Qwen3ForEmbedding",
        "Qwen3ForSequenceClassification",
        "Qwen3Model",
    ],
    "modules.qwen3_moe": [
        "Qwen3MoeConfig",
        "Qwen3MoeForCausalLM",
        "Qwen3MoeForSequenceClassification",
        "Qwen3MoeModel",
    ],
    "modules.qwen3_next": [
        "Qwen3NextConfig",
        "Qwen3NextForCausalLM",
        "Qwen3NextModel",
    ],
    "modules.qwen3_omni_moe": [
        "Qwen3OmniMoeAudioConfig",
        "Qwen3OmniMoeAudioEncoder",
        "Qwen3OmniMoeAudioEncoderConfig",
        "Qwen3OmniMoeCode2Wav",
        "Qwen3OmniMoeCode2WavConfig",
        "Qwen3OmniMoeConfig",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeModel",
        "Qwen3OmniMoeTalkerCodePredictorConfig",
        "Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration",
        "Qwen3OmniMoeTalkerCodePredictorModel",
        "Qwen3OmniMoeTalkerConfig",
        "Qwen3OmniMoeTalkerForConditionalGeneration",
        "Qwen3OmniMoeTalkerTextConfig",
        "Qwen3OmniMoeTextConfig",
        "Qwen3OmniMoeThinkerConfig",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "Qwen3OmniMoeThinkerModel",
        "Qwen3OmniMoeVisionConfig",
        "Qwen3OmniMoeVisionEncoder",
        "Qwen3OmniMoeVisionEncoderConfig",
    ],
    "modules.qwen3_vl": [
        "Qwen3VLConfig",
        "Qwen3VLTextConfig",
        "Qwen3VLVisionConfig",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLModel",
        "Qwen3VLTextModel",
        "Qwen3VisionTransformerPretrainedModel",
    ],
    "modules.qwen3_vl_moe": [
        "Qwen3VLMoeConfig",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLMoeModel",
        "Qwen3VLMoeTextConfig",
        "Qwen3VLMoeTextModel",
        "Qwen3VLMoeVisionConfig",
        "Qwen3VLMoeVisionTransformerPretrainedModel",
    ],
    "modules.roberta": [
        "RobertaConfig",
        "RobertaForCausalLM",
        "RobertaForMultipleChoice",
        "RobertaForQuestionAnswering",
        "RobertaForSequenceClassification",
        "RobertaForTokenClassification",
    ],
    "modules.seed_oss": [
        "SeedOssConfig",
        "SeedOssForCausalLM",
        "SeedOssForSequenceClassification",
        "SeedOssModel",
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
    "modules.smollm3": [
        "SmolLM3Config",
        "SmolLM3ForCausalLM",
        "SmolLM3ForSequenceClassification",
        "SmolLM3Model",
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
        "AgenticMoshPitConfig",
        "AgenticMoshPitTrainer",
        "AsyncGRPOConfig",
        "AsyncGRPOTrainer",
        "BEMAConfig",
        "BEMACallback",
        "BEMADPOTrainer",
        "BCOConfig",
        "BCOPreprocessTransform",
        "BCOTrainer",
        "BaseTrainer",
        "CPOConfig",
        "CPOPreprocessTransform",
        "CPOTrainer",
        "DPOConfig",
        "DPOPreprocessTransform",
        "DPOTrainer",
        "DPPOConfig",
        "DPPOTrainer",
        "DistillationConfig",
        "DistillationTrainer",
        "EmbeddingConfig",
        "EmbeddingTrainer",
        "GFPOConfig",
        "GFPOTrainer",
        "GKDConfig",
        "GKDTrainer",
        "GOLDConfig",
        "GOLDTrainer",
        "GRPOConfig",
        "GRPOPreprocessTransform",
        "GRPOTrainer",
        "GRPOReplayBufferConfig",
        "GRPOReplayBufferTrainer",
        "GRPOWithReplayBufferConfig",
        "GRPOWithReplayBufferTrainer",
        "GSPOConfig",
        "GSPOTokenConfig",
        "GSPOTokenTrainer",
        "GSPOTrainer",
        "KTOConfig",
        "KTOPreprocessTransform",
        "KTOTrainer",
        "LogWatcher",
        "MergeConfig",
        "MergeModelCallback",
        "MiniLLMConfig",
        "MiniLLMTrainer",
        "NashMDConfig",
        "NashMDTrainer",
        "NeMoGymConfig",
        "NeMoGymTrainer",
        "OpenRewardSpec",
        "OnlineDPOConfig",
        "OnlineDPOTrainer",
        "ORPOConfig",
        "ORPOPreprocessTransform",
        "ORPOTrainer",
        "OnPolicyDistillationConfig",
        "OnPolicyDistillationTrainer",
        "PPOConfig",
        "PPOPreprocessTransform",
        "PPOTrainer",
        "PAPOConfig",
        "PAPOTrainer",
        "PRMConfig",
        "PRMPreprocessTransform",
        "PRMTrainer",
        "RLVRConfig",
        "RLVRTrainer",
        "RLOOConfig",
        "RLOOTrainer",
        "RayDistributedTrainer",
        "RewardConfig",
        "RewardPreprocessTransform",
        "RewardTrainer",
        "SDFTConfig",
        "SDFTTrainer",
        "SDPOConfig",
        "SDPOTrainer",
        "SFTConfig",
        "SFTPreprocessTransform",
        "SFTTrainer",
        "SSDConfig",
        "SSDTrainer",
        "SelfDistillationConfig",
        "SelfDistillationTrainer",
        "SeqKDConfig",
        "SeqKDTrainer",
        "SparseDistillationConfig",
        "SparseDistillationTrainer",
        "Trainer",
        "TrainingArguments",
        "TPOConfig",
        "TPOTrainer",
        "XPOConfig",
        "XPOTrainer",
        "eSurgeRolloutConfig",
        "eSurgeRolloutGenerator",
        "generate_rollout_completions",
        "load_nemo_gym_jsonl",
        "pack_sequences",
        "prompt_transforms",
    ],
}


if _tp.TYPE_CHECKING:
    from . import data, utils
    from .inference import (
        EngineRequest,
        EngineRequestStatus,
        InferenceApiRouter,
        JitableSamplingParams,
        SamplingParams,
        ToolParser,
        ToolParserManager,
        eSurge,
        eSurgeApiServer,
        eSurgeCacheRuntimeConfig,
        eSurgeContextConfig,
        eSurgeDistributedConfig,
        eSurgeParsingConfig,
        eSurgeRunner,
        eSurgeRuntimeConfig,
        eSurgeVisionConfig,
        eSurgeWorkerConfig,
        vWhisperInference,
        vWhisperInferenceConfig,
    )
    from .inference.evaluations import eSurgeLMEvalAdapter
    from .infra import (
        BenchmarkConfig,
        EasyDeLBaseConfig,
        EasyDeLBaseConfigDict,
        EasyDeLBaseModule,
        EasyDeLState,
        LossConfig,
        PartitionAxis,
        PyTree,
        Rngs,
        auto_pytree,
        eLargeModel,
        init_cluster,
        sharding,
    )
    from .infra.errors import EasyDeLRuntimeError, EasyDeLSyntaxRuntimeError, EasyDeLTimerError
    from .infra.etils import (
        EasyDeLBackends,
        EasyDeLGradientCheckPointers,
        EasyDeLOptimizers,
        EasyDeLPlatforms,
        EasyDeLSchedulers,
    )
    from .infra.factory import ConfigType, TaskType, register_config, register_module
    from .layers.attention import AttentionMechanisms, AttentionModule, FlexibleAttentionModule
    from .layers.moe import MoEMethods
    from .layers.quantization import EasyDeLQuantizationConfig, EasyQuantizer, QuantizationType, TurboQuantConfig
    from .modules.arctic import ArcticConfig, ArcticForCausalLM, ArcticModel
    from .modules.auto import (
        AutoEasyDeLAnyToAnyModel,
        AutoEasyDeLConfig,
        AutoEasyDeLModel,
        AutoEasyDeLModelForCausalLM,
        AutoEasyDeLModelForDiffusionLM,
        AutoEasyDeLModelForEmbedding,
        AutoEasyDeLModelForImageTextToText,
        AutoEasyDeLModelForSeq2SeqLM,
        AutoEasyDeLModelForSequenceClassification,
        AutoEasyDeLModelForSpeechSeq2Seq,
        AutoEasyDeLModelForZeroShotImageClassification,
        AutoEasyDeLVisionModel,
        AutoShardAndGatherFunctions,
        AutoState,
        AutoStateAnyToAnyModel,
        AutoStateForCausalLM,
        AutoStateForDiffusionLM,
        AutoStateForEmbedding,
        AutoStateForImageSequenceClassification,
        AutoStateForImageTextToText,
        AutoStateForSeq2SeqLM,
        AutoStateForSpeechSeq2Seq,
        AutoStateForZeroShotImageClassification,
        AutoStateVisionModel,
        get_modules_by_type,
    )
    from .modules.aya_vision import AyaVisionConfig, AyaVisionForConditionalGeneration, AyaVisionModel
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
    from .modules.cohere import CohereConfig, CohereForCausalLM, CohereForSequenceClassification, CohereModel
    from .modules.cohere2 import Cohere2Config, Cohere2ForCausalLM, Cohere2ForSequenceClassification, Cohere2Model
    from .modules.dbrx import (
        DbrxAttentionConfig,
        DbrxConfig,
        DbrxFFNConfig,
        DbrxForCausalLM,
        DbrxForSequenceClassification,
        DbrxModel,
    )
    from .modules.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV2Model
    from .modules.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM, DeepseekV3Model
    from .modules.exaone import ExaoneConfig, ExaoneForCausalLM, ExaoneForSequenceClassification, ExaoneModel
    from .modules.exaone4 import Exaone4Config, Exaone4ForCausalLM, Exaone4ForSequenceClassification, Exaone4Model
    from .modules.falcon import FalconConfig, FalconForCausalLM, FalconModel
    from .modules.falcon_h1 import FalconH1Config, FalconH1ForCausalLM, FalconH1Model
    from .modules.falcon_mamba import FalconMambaConfig, FalconMambaForCausalLM, FalconMambaModel
    from .modules.gemma import GemmaConfig, GemmaForCausalLM, GemmaForSequenceClassification, GemmaModel
    from .modules.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2Model
    from .modules.gemma3 import (
        Gemma3Config,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        Gemma3ForSequenceClassification,
        Gemma3MultiModalProjector,
        Gemma3TextConfig,
        Gemma3TextModel,
    )
    from .modules.gemma4 import (
        Gemma4Config,
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4MultimodalEmbedder,
        Gemma4RMSNorm,
        Gemma4TextConfig,
        Gemma4TextModel,
        Gemma4VisionConfig,
        Gemma4VisionModel,
    )
    from .modules.gemma4_assistant import (
        Gemma4AssistantCentroidHead,
        Gemma4AssistantConfig,
        Gemma4AssistantForCausalLM,
        Gemma4AssistantModel,
        Gemma4AssistantOutput,
        Gemma4AssistantTextConfig,
    )
    from .modules.gidd import GiddConfig, GiddForDiffusionLM, GiddModel
    from .modules.glm import GlmConfig, GlmForCausalLM, GlmForSequenceClassification, GlmModel
    from .modules.glm4 import Glm4Config, Glm4ForCausalLM, Glm4ForSequenceClassification, Glm4Model
    from .modules.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM, Glm4MoeForSequenceClassification, Glm4MoeModel
    from .modules.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, Glm4MoeLiteModel
    from .modules.glm4v import (
        Glm4vConfig,
        Glm4vForConditionalGeneration,
        Glm4vModel,
        Glm4vTextConfig,
        Glm4vTextModel,
        Glm4vVisionConfig,
        Glm4vVisionModel,
    )
    from .modules.glm4v_moe import (
        Glm4vMoeConfig,
        Glm4vMoeForConditionalGeneration,
        Glm4vMoeModel,
        Glm4vMoeTextConfig,
        Glm4vMoeTextModel,
        Glm4vMoeVisionConfig,
        Glm4vMoeVisionModel,
    )
    from .modules.glm46v import Glm46VConfig, Glm46VForConditionalGeneration, Glm46VModel
    from .modules.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM, GlmMoeDsaModel
    from .modules.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Model
    from .modules.gpt_j import GPTJConfig, GPTJForCausalLM, GPTJModel
    from .modules.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXModel
    from .modules.gpt_oss import GptOssConfig, GptOssForCausalLM, GptOssForSequenceClassification, GptOssModel
    from .modules.grok_1 import Grok1Config, Grok1ForCausalLM, Grok1Model
    from .modules.internlm2 import (
        InternLM2Config,
        InternLM2ForCausalLM,
        InternLM2ForSequenceClassification,
        InternLM2Model,
    )
    from .modules.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM, KimiLinearModel
    from .modules.kimi_vl import KimiVLConfig, KimiVLForConditionalGeneration, MoonViTConfig
    from .modules.llama import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel
    from .modules.llama4 import (
        Llama4Config,
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4ForSequenceClassification,
        Llama4TextConfig,
        Llama4TextModel,
        Llama4VisionConfig,
        Llama4VisionModel,
    )
    from .modules.llava import LlavaConfig, LlavaForConditionalGeneration, LlavaModel
    from .modules.mamba import MambaConfig, MambaForCausalLM, MambaModel
    from .modules.mamba2 import Mamba2Config, Mamba2ForCausalLM, Mamba2Model
    from .modules.minimax import MiniMaxConfig, MiniMaxForCausalLM, MiniMaxModel
    from .modules.mistral import MistralConfig, MistralForCausalLM, MistralForSequenceClassification, MistralModel
    from .modules.mistral3 import Mistral3Config, Mistral3ForConditionalGeneration, Mistral3Model, Mistral3Tokenizer
    from .modules.mixtral import MixtralConfig, MixtralForCausalLM, MixtralForSequenceClassification, MixtralModel
    from .modules.mosaic_mpt import MptAttentionConfig, MptConfig, MptForCausalLM, MptModel
    from .modules.olmo import OlmoConfig, OlmoForCausalLM, OlmoModel
    from .modules.olmo2 import Olmo2Config, Olmo2ForCausalLM, Olmo2ForSequenceClassification, Olmo2Model
    from .modules.olmo3 import Olmo3Config, Olmo3ForCausalLM, Olmo3ForSequenceClassification, Olmo3Model
    from .modules.openelm import OpenELMConfig, OpenELMForCausalLM, OpenELMModel
    from .modules.opt import OPTConfig, OPTForCausalLM, OPTModel
    from .modules.phi import PhiConfig, PhiForCausalLM, PhiModel
    from .modules.phi3 import Phi3Config, Phi3ForCausalLM, Phi3Model
    from .modules.phimoe import PhiMoeConfig, PhiMoeForCausalLM, PhiMoeModel
    from .modules.pixtral import PixtralVisionConfig, PixtralVisionModel
    from .modules.qwen2 import (
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen2ForEmbedding,
        Qwen2ForSequenceClassification,
        Qwen2Model,
    )
    from .modules.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen2MoeForSequenceClassification, Qwen2MoeModel
    from .modules.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLModel
    from .modules.qwen3 import (
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3ForEmbedding,
        Qwen3ForSequenceClassification,
        Qwen3Model,
    )
    from .modules.qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MTPHead,
        Qwen3_5MTPLayer,
        Qwen3_5MTPOutput,
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
        Qwen3_5VisionConfig,
    )
    from .modules.qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
        Qwen3_5MoeVisionConfig,
    )
    from .modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeForSequenceClassification, Qwen3MoeModel
    from .modules.qwen3_next import (
        Qwen3NextConfig,
        Qwen3NextForCausalLM,
        Qwen3NextModel,
    )
    from .modules.qwen3_omni_moe import (
        Qwen3OmniMoeAudioConfig,
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeCode2Wav,
        Qwen3OmniMoeCode2WavConfig,
        Qwen3OmniMoeConfig,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeModel,
        Qwen3OmniMoeTalkerCodePredictorConfig,
        Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration,
        Qwen3OmniMoeTalkerCodePredictorModel,
        Qwen3OmniMoeTalkerConfig,
        Qwen3OmniMoeTalkerForConditionalGeneration,
        Qwen3OmniMoeTalkerTextConfig,
        Qwen3OmniMoeTextConfig,
        Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerModel,
        Qwen3OmniMoeVisionConfig,
        Qwen3OmniMoeVisionEncoder,
        Qwen3OmniMoeVisionEncoderConfig,
    )
    from .modules.qwen3_vl import (
        Qwen3VisionTransformerPretrainedModel,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLTextConfig,
        Qwen3VLTextModel,
        Qwen3VLVisionConfig,
    )
    from .modules.qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLMoeModel,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeTextModel,
        Qwen3VLMoeVisionConfig,
        Qwen3VLMoeVisionTransformerPretrainedModel,
    )
    from .modules.roberta import (
        RobertaConfig,
        RobertaForCausalLM,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
    )
    from .modules.seed_oss import SeedOssConfig, SeedOssForCausalLM, SeedOssForSequenceClassification, SeedOssModel
    from .modules.siglip import (
        SiglipConfig,
        SiglipForImageClassification,
        SiglipModel,
        SiglipTextConfig,
        SiglipTextModel,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from .modules.smollm3 import SmolLM3Config, SmolLM3ForCausalLM, SmolLM3ForSequenceClassification, SmolLM3Model
    from .modules.stablelm import StableLmConfig, StableLmForCausalLM, StableLmModel
    from .modules.whisper import (
        WhisperConfig,
        WhisperForAudioClassification,
        WhisperForConditionalGeneration,
        WhisperTimeStampLogitsProcessor,
    )
    from .modules.xerxes import XerxesConfig, XerxesForCausalLM, XerxesModel
    from .modules.xerxes2 import Xerxes2Config, Xerxes2ForCausalLM, Xerxes2Model
    from .operations import (
        AttentionConfig,
        AttentionOutput,
        AutoRegressiveDecodeAttn,
        BaseOperationConfig,
        BlockSparseAttentionConfig,
        BlockSparseAttn,
        FlashAttentionConfig,
        FlashAttn,
        OperationImpl,
        OperationMetadata,
        OperationRegistry,
        PagedFlashAttn,
        RaggedPageAttentionv2Config,
        RaggedPageAttentionv3Config,
        RaggedPageAttnV2,
        RaggedPageAttnV3,
        RingAttentionConfig,
        RingAttn,
        ScaledDotProductAttentionConfig,
        ScaledDotProductAttn,
        VanillaAttn,
    )
    from .trainers import (
        AgenticMoshPitConfig,
        AgenticMoshPitTrainer,
        AsyncGRPOConfig,
        AsyncGRPOTrainer,
        BaseTrainer,
        BCOConfig,
        BCOPreprocessTransform,
        BCOTrainer,
        BEMACallback,
        BEMAConfig,
        BEMADPOTrainer,
        CPOConfig,
        CPOPreprocessTransform,
        CPOTrainer,
        DistillationConfig,
        DistillationTrainer,
        DPOConfig,
        DPOPreprocessTransform,
        DPOTrainer,
        DPPOConfig,
        DPPOTrainer,
        EmbeddingConfig,
        EmbeddingTrainer,
        GFPOConfig,
        GFPOTrainer,
        GKDConfig,
        GKDTrainer,
        GOLDConfig,
        GOLDTrainer,
        GRPOConfig,
        GRPOPreprocessTransform,
        GRPOReplayBufferConfig,
        GRPOReplayBufferTrainer,
        GRPOTrainer,
        GRPOWithReplayBufferConfig,
        GRPOWithReplayBufferTrainer,
        GSPOConfig,
        GSPOTokenConfig,
        GSPOTokenTrainer,
        GSPOTrainer,
        KTOConfig,
        KTOPreprocessTransform,
        KTOTrainer,
        LogWatcher,
        MergeConfig,
        MergeModelCallback,
        MiniLLMConfig,
        MiniLLMTrainer,
        NashMDConfig,
        NashMDTrainer,
        NeMoGymConfig,
        NeMoGymTrainer,
        OnlineDPOConfig,
        OnlineDPOTrainer,
        OnPolicyDistillationConfig,
        OnPolicyDistillationTrainer,
        OpenRewardSpec,
        ORPOConfig,
        ORPOPreprocessTransform,
        ORPOTrainer,
        PAPOConfig,
        PAPOTrainer,
        PPOConfig,
        PPOPreprocessTransform,
        PPOTrainer,
        PRMConfig,
        PRMPreprocessTransform,
        PRMTrainer,
        RayDistributedTrainer,
        RewardConfig,
        RewardPreprocessTransform,
        RewardTrainer,
        RLOOConfig,
        RLOOTrainer,
        RLVRConfig,
        RLVRTrainer,
        SDFTConfig,
        SDFTTrainer,
        SDPOConfig,
        SDPOTrainer,
        SelfDistillationConfig,
        SelfDistillationTrainer,
        SeqKDConfig,
        SeqKDTrainer,
        SFTConfig,
        SFTPreprocessTransform,
        SFTTrainer,
        SparseDistillationConfig,
        SparseDistillationTrainer,
        SSDConfig,
        SSDTrainer,
        TPOConfig,
        TPOTrainer,
        Trainer,
        TrainingArguments,
        XPOConfig,
        XPOTrainer,
        eSurgeRolloutConfig,
        eSurgeRolloutGenerator,
        generate_rollout_completions,
        load_nemo_gym_jsonl,
        pack_sequences,
        prompt_transforms,
    )
    from .utils import (
        ModelConverter,
        Registry,
        StateDictConverter,
        TensorConverter,
        ejit,
        ePath,
        ePathLike,
        is_inference_mode,
        set_inference_mode,
        traversals,
    )
else:
    _sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )

    _targeted_eformer_versions = ["0.0.101"]
    _targeted_ejkernel_versions = ["0.0.79", "0.0.80"]

    from eformer import __version__ as _eform_version
    from ejkernel import __version__ as _ejker_version

    assert _version(_eform_version) in [
        _version(_targeted_version) for _targeted_version in _targeted_eformer_versions
    ], (
        f"this version of EasyDeL is only compatible with eformer {', '.join(_targeted_eformer_versions)},"
        f" but found eformer {_eform_version}"
    )
    assert _version(_ejker_version) in [
        _version(_targeted_version) for _targeted_version in _targeted_ejkernel_versions
    ], (
        f"this version of EasyDeL is only compatible with ejkernel {', '.join(_targeted_ejkernel_versions)},"
        f" but found ejkernel {_ejker_version}"
    )

    if not _is_package_available("torch"):
        _logger.warning("please install `torch` (cpu) if you want to use `AutoEasyDeLModel*.from_torch_pretrained`")

    del _version
    del _eform_version


del _logger
del _LazyModule
del _is_package_available
del _apply_linkup
del _initialize_distributed
del _distributed_init_enabled
