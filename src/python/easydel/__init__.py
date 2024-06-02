import os as _os

if bool(_os.environ.get("EASYDEL_AUTO", "true")):
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

from .utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "inference.serve_engine": [
        "EasyDeLServeEngine",
        "EasyDeLServeEngineConfig",
        "EngineClient"
    ],
    "inference.generation_pipeline": [
        "GenerationPipelineConfig",
        "GenerationPipeline"
    ],
    "modules.llama": [
        "FlaxLlamaModel",
        "FlaxLlamaForCausalLM",
        "FlaxLlamaForSequenceClassification",
        "LlamaConfig",
        "FlaxVisionLlamaForCausalLM",
        "VisionLlamaConfig"
    ],
    "modules.gpt_j": [
        "GPTJConfig",
        "FlaxGPTJForCausalLM",
        "FlaxGPTJModel"
    ],
    "modules.t5": [
        "T5Config",
        "FlaxT5ForConditionalGeneration",
        "FlaxT5Model"
    ],
    "modules.falcon": [
        "FalconConfig",
        "FlaxFalconModel",
        "FlaxFalconForCausalLM"
    ],
    "modules.opt": [
        "OPTConfig",
        "FlaxOPTForCausalLM",
        "FlaxOPTModel"
    ],
    "modules.mistral": [
        "MistralConfig",
        "FlaxMistralForCausalLM",
        "FlaxMistralModel",
        "FlaxVisionMistralForCausalLM",
        "VisionMistralConfig"
    ],
    "modules.palm": [
        "FlaxPalmModel",
        "PalmConfig",
        "FlaxPalmForCausalLM"
    ],
    "modules.mosaic_mpt": [
        "MptConfig",
        "MptAttentionConfig",
        "FlaxMptForCausalLM",
        "FlaxMptModel"
    ],
    "modules.gpt_neo_x": [
        "GPTNeoXConfig",
        "FlaxGPTNeoXModel",
        "FlaxGPTNeoXForCausalLM"
    ],
    "modules.lucid_transformer": [
        "FlaxLTModel",
        "FlaxLTConfig",
        "FlaxLTForCausalLM"
    ],
    "modules.gpt2": [
        "GPT2Config",
        "FlaxGPT2LMHeadModel",
        "FlaxGPT2Model"
    ],
    "modules.mixtral": [
        "FlaxMixtralForCausalLM",
        "FlaxMixtralModel",
        "MixtralConfig"
    ],
    "modules.phi": [
        "FlaxPhiForCausalLM",
        "PhiConfig",
        "FlaxPhiModel"
    ],
    "modules.qwen1": [
        "FlaxQwen1Model",
        "FlaxQwen1ForCausalLM",
        "FlaxQwen1ForSequenceClassification",
        "Qwen1Config"
    ],
    "modules.qwen2": [
        "FlaxQwen2Model",
        "FlaxQwen2ForCausalLM",
        "FlaxQwen2ForSequenceClassification",
        "Qwen2Config"
    ],
    "modules.gemma": [
        "FlaxGemmaModel",
        "GemmaConfig",
        "FlaxGemmaForCausalLM"
    ],
    "modules.stablelm": [
        "StableLmConfig",
        "FlaxStableLmForCausalLM",
        "FlaxStableLmModel"
    ],
    "modules.mamba": [
        "FlaxMambaModel",
        "FlaxMambaForCausalLM",
        "MambaConfig"
    ],
    "modules.grok_1": [
        "Grok1Config",
        "FlaxGrok1Model",
        "FlaxGrok1ForCausalLM"
    ],
    "modules.qwen2_moe": [
        "Qwen2MoeConfig",
        "FlaxQwen2MoeModel",
        "FlaxQwen2MoeForCausalLM"
    ],
    "modules.whisper": [
        "FlaxWhisperForConditionalGeneration",
        "FlaxWhisperForAudioClassification",
        "FlaxWhisperTimeStampLogitsProcessor",
        "WhisperConfig"
    ],
    "modules.cohere": [
        "FlaxCohereModel",
        "CohereConfig",
        "FlaxCohereForCausalLM"
    ],
    "modules.dbrx": [
        "DbrxConfig",
        "DbrxFFNConfig",
        "DbrxAttentionConfig",
        "FlaxDbrxModel",
        "FlaxDbrxForCausalLM"
    ],
    "modules.phi3": [
        "Phi3Config",
        "FlaxPhi3ForCausalLM",
        "FlaxPhi3Model"
    ],
    "modules.arctic": [
        "FlaxArcticForCausalLM",
        "FlaxArcticModel",
        "ArcticConfig"
    ],
    "modules.openelm": [
        "FlaxOpenELMModel",
        "FlaxOpenELMForCausalLM",
        "OpenELMConfig"
    ],
    "modules.deepseek_v2": [
        "FlaxDeepseekV2Model",
        "FlaxDeepseekV2ForCausalLM",
        "DeepseekV2Config"
    ],
    "modules.auto_easydel_model": [
        "AutoEasyDeLModelForCausalLM",
        "AutoEasyDeLConfig",
        "AutoShardAndGatherFunctions",
        "get_modules_by_type"
    ],
    "modules.attention_module": [
        "AttentionModule"
    ],
    "utils.utils": [
        "get_mesh",
        "RNG"
    ],
    "trainer": [
        "TrainArguments",
        "EasyDeLXRapTureConfig",
        "create_casual_language_model_evaluation_step",
        "create_casual_language_model_train_step",
        "CausalLanguageModelTrainer",
        "VisionCausalLanguageModelStepOutput",
        "VisionCausalLanguageModelTrainer",
        "create_vision_casual_language_model_evaluation_step",
        "create_vision_casual_language_model_train_step",
        "DPOTrainer",
        "create_dpo_eval_function",
        "create_concatenated_forward",
        "create_dpo_train_function",
        "SFTTrainer",
        "create_constant_length_dataset",
        "get_formatting_func_from_dataset",
        "conversations_formatting_function",
        "instructions_formatting_function",
        "ORPOTrainerOutput",
        "odds_ratio_loss",
        "ORPOTrainer",
        "create_orpo_step_function"
    ],
    "smi": [
        "run",
        "initialise_tracking",
        "get_mem"
    ],
    "transform": [
        "huggingface_to_easydel",
        "easystate_to_huggingface_model",
        "easystate_to_torch"
    ],
    "etils": [
        "EasyDeLOptimizers",
        "EasyDeLSchedulers",
        "EasyDeLGradientCheckPointers",
        "EasyDeLState",
        "EasyDeLTimerError",
        "EasyDeLRuntimeError",
        "EasyDeLSyntaxRuntimeError",
        "PartitionAxis"
    ],
}

if TYPE_CHECKING:
    from .inference.serve_engine import EasyDeLServeEngine, EasyDeLServeEngineConfig, EngineClient
    from .inference.generation_pipeline import GenerationPipelineConfig, GenerationPipeline

    from .modules.llama import (
        FlaxLlamaModel,
        FlaxLlamaForCausalLM,
        FlaxLlamaForSequenceClassification,
        LlamaConfig,
        FlaxVisionLlamaForCausalLM,
        VisionLlamaConfig
    )
    from .modules.gpt_j import GPTJConfig, FlaxGPTJForCausalLM, FlaxGPTJModel
    from .modules.t5 import T5Config, FlaxT5ForConditionalGeneration, FlaxT5Model
    from .modules.falcon import FalconConfig, FlaxFalconModel, FlaxFalconForCausalLM
    from .modules.opt import OPTConfig, FlaxOPTForCausalLM, FlaxOPTModel
    from .modules.mistral import (
        MistralConfig,
        FlaxMistralForCausalLM,
        FlaxMistralModel,
        FlaxVisionMistralForCausalLM,
        VisionMistralConfig
    )
    from .modules.palm import FlaxPalmModel, PalmConfig, FlaxPalmForCausalLM
    from .modules.mosaic_mpt import (
        MptConfig,
        MptAttentionConfig,
        FlaxMptForCausalLM,
        FlaxMptModel
    )
    from .modules.gpt_neo_x import GPTNeoXConfig, FlaxGPTNeoXModel, FlaxGPTNeoXForCausalLM
    from .modules.lucid_transformer import FlaxLTModel, FlaxLTConfig, FlaxLTForCausalLM
    from .modules.gpt2 import GPT2Config, FlaxGPT2LMHeadModel, FlaxGPT2Model
    from .modules.mixtral import FlaxMixtralForCausalLM, FlaxMixtralModel, MixtralConfig
    from .modules.phi import FlaxPhiForCausalLM, PhiConfig, FlaxPhiModel
    from .modules.qwen1 import FlaxQwen1Model, FlaxQwen1ForCausalLM, FlaxQwen1ForSequenceClassification, Qwen1Config
    from .modules.qwen2 import FlaxQwen2Model, FlaxQwen2ForCausalLM, FlaxQwen2ForSequenceClassification, Qwen2Config
    from .modules.gemma import FlaxGemmaModel, GemmaConfig, FlaxGemmaForCausalLM
    from .modules.stablelm import StableLmConfig, FlaxStableLmForCausalLM, FlaxStableLmModel
    from .modules.mamba import FlaxMambaModel, FlaxMambaForCausalLM, MambaConfig
    from .modules.grok_1 import Grok1Config, FlaxGrok1Model, FlaxGrok1ForCausalLM
    from .modules.qwen2_moe import Qwen2MoeConfig, FlaxQwen2MoeModel, FlaxQwen2MoeForCausalLM
    from .modules.whisper import (
        FlaxWhisperForConditionalGeneration,
        FlaxWhisperForAudioClassification,
        FlaxWhisperTimeStampLogitsProcessor,
        WhisperConfig
    )
    from .modules.cohere import FlaxCohereModel, CohereConfig, FlaxCohereForCausalLM
    from .modules.dbrx import (
        DbrxConfig,
        DbrxFFNConfig,
        DbrxAttentionConfig,
        FlaxDbrxModel,
        FlaxDbrxForCausalLM
    )
    from .modules.phi3 import Phi3Config, FlaxPhi3ForCausalLM, FlaxPhi3Model
    from .modules.arctic import FlaxArcticForCausalLM, FlaxArcticModel, ArcticConfig
    from .modules.openelm import FlaxOpenELMModel, FlaxOpenELMForCausalLM, OpenELMConfig
    from .modules.deepseek_v2 import FlaxDeepseekV2Model, FlaxDeepseekV2ForCausalLM, DeepseekV2Config
    from .modules.auto_easydel_model import (
        AutoEasyDeLModelForCausalLM,
        AutoEasyDeLConfig,
        AutoShardAndGatherFunctions,
        get_modules_by_type
    )
    from .modules.attention_module import AttentionModule

    from .utils.utils import get_mesh, RNG
    from .trainer import (
        TrainArguments,
        EasyDeLXRapTureConfig,
        create_casual_language_model_evaluation_step,
        create_casual_language_model_train_step,
        CausalLanguageModelTrainer,
        VisionCausalLanguageModelStepOutput,
        VisionCausalLanguageModelTrainer,
        create_vision_casual_language_model_evaluation_step,
        create_vision_casual_language_model_train_step,
        DPOTrainer,
        create_dpo_eval_function,
        create_concatenated_forward,
        create_dpo_train_function,
        SFTTrainer,
        create_constant_length_dataset,
        get_formatting_func_from_dataset,
        conversations_formatting_function,
        instructions_formatting_function,
        ORPOTrainerOutput,
        odds_ratio_loss,
        ORPOTrainer,
        create_orpo_step_function
    )
    from .smi import run, initialise_tracking, get_mem
    from .transform import huggingface_to_easydel, easystate_to_huggingface_model, easystate_to_torch
    from .etils import (
        EasyDeLOptimizers,
        EasyDeLSchedulers,
        EasyDeLGradientCheckPointers,
        EasyDeLState,
        EasyDeLTimerError,
        EasyDeLRuntimeError,
        EasyDeLSyntaxRuntimeError,
        PartitionAxis
    )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

__version__ = "0.0.66"
