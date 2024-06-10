import os as _os

if bool(_os.environ.get("EASYDEL_AUTO", "true")):
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

from .utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_modules_import_structure = {
    "modules.llama": [
        "FlaxLlamaModel",
        "FlaxLlamaForCausalLM",
        "FlaxLlamaForSequenceClassification",
        "LlamaConfig",
        "FlaxVisionLlamaForCausalLM",
        "VisionLlamaConfig",
    ],
    "modules.gpt_j": ["GPTJConfig", "FlaxGPTJForCausalLM", "FlaxGPTJModel"],
    "modules.t5": ["T5Config", "FlaxT5ForConditionalGeneration", "FlaxT5Model"],
    "modules.falcon": ["FalconConfig", "FlaxFalconModel", "FlaxFalconForCausalLM"],
    "modules.opt": ["OPTConfig", "FlaxOPTForCausalLM", "FlaxOPTModel"],
    "modules.mistral": [
        "MistralConfig",
        "FlaxMistralForCausalLM",
        "FlaxMistralModel",
        "FlaxVisionMistralForCausalLM",
        "VisionMistralConfig",
    ],
    "modules.palm": ["FlaxPalmModel", "PalmConfig", "FlaxPalmForCausalLM"],
    "modules.mosaic_mpt": [
        "MptConfig",
        "MptAttentionConfig",
        "FlaxMptForCausalLM",
        "FlaxMptModel",
    ],
    "modules.gpt_neo_x": [
        "GPTNeoXConfig",
        "FlaxGPTNeoXModel",
        "FlaxGPTNeoXForCausalLM",
    ],
    "modules.lucid_transformer": ["FlaxLTModel", "FlaxLTConfig", "FlaxLTForCausalLM"],
    "modules.gpt2": ["GPT2Config", "FlaxGPT2LMHeadModel", "FlaxGPT2Model"],
    "modules.mixtral": ["FlaxMixtralForCausalLM", "FlaxMixtralModel", "MixtralConfig"],
    "modules.phi": ["FlaxPhiForCausalLM", "PhiConfig", "FlaxPhiModel"],
    "modules.qwen1": [
        "FlaxQwen1Model",
        "FlaxQwen1ForCausalLM",
        "FlaxQwen1ForSequenceClassification",
        "Qwen1Config",
    ],
    "modules.qwen2": [
        "FlaxQwen2Model",
        "FlaxQwen2ForCausalLM",
        "FlaxQwen2ForSequenceClassification",
        "Qwen2Config",
    ],
    "modules.gemma": ["FlaxGemmaModel", "GemmaConfig", "FlaxGemmaForCausalLM"],
    "modules.stablelm": [
        "StableLmConfig",
        "FlaxStableLmForCausalLM",
        "FlaxStableLmModel",
    ],
    "modules.mamba": ["FlaxMambaModel", "FlaxMambaForCausalLM", "MambaConfig"],
    "modules.grok_1": ["Grok1Config", "FlaxGrok1Model", "FlaxGrok1ForCausalLM"],
    "modules.qwen2_moe": [
        "Qwen2MoeConfig",
        "FlaxQwen2MoeModel",
        "FlaxQwen2MoeForCausalLM",
    ],
    "modules.whisper": [
        "FlaxWhisperForConditionalGeneration",
        "FlaxWhisperForAudioClassification",
        "FlaxWhisperTimeStampLogitsProcessor",
        "WhisperConfig",
    ],
    "modules.cohere": ["FlaxCohereModel", "CohereConfig", "FlaxCohereForCausalLM"],
    "modules.dbrx": [
        "DbrxConfig",
        "DbrxFFNConfig",
        "DbrxAttentionConfig",
        "FlaxDbrxModel",
        "FlaxDbrxForCausalLM",
    ],
    "modules.phi3": ["Phi3Config", "FlaxPhi3ForCausalLM", "FlaxPhi3Model"],
    "modules.arctic": ["FlaxArcticForCausalLM", "FlaxArcticModel", "ArcticConfig"],
    "modules.openelm": ["FlaxOpenELMModel", "FlaxOpenELMForCausalLM", "OpenELMConfig"],
    "modules.deepseek_v2": [
        "FlaxDeepseekV2Model",
        "FlaxDeepseekV2ForCausalLM",
        "DeepseekV2Config",
    ],
    "modules.auto_easydel_model": [
        "AutoEasyDeLModelForCausalLM",
        "AutoEasyDeLConfig",
        "AutoShardAndGatherFunctions",
        "get_modules_by_type",
    ],
    "modules.easydel_modelling_utils": [
        "EasyDeLPretrainedConfig",
        "EasyDeLFlaxPretrainedModel",
    ],
    "modules.attention_module": ["AttentionModule"],
}

_trainer_import_structure = {
    "trainer.causal_language_model_trainer.causal_language_model_trainer": [
        "CausalLanguageModelTrainer",
        "CausalLMTrainerOutput",
    ],
    "trainer.supervised_fine_tuning_trainer.stf_trainer": ["SFTTrainer"],
    "trainer.vision_causal_language_model_trainer.vision_causal_language_model_trainer": [
        "VisionCausalLanguageModelStepOutput",
        "VisionCausalLanguageModelTrainer",
        "VisionCausalLMTrainerOutput",
    ],
    "trainer.odds_ratio_preference_optimization_trainer.orpo_trainer": [
        "ORPOTrainer",
        "ORPOTrainerOutput",
    ],
    "trainer.direct_preference_optimization_trainer.direct_preference_optimization_trainer": [
        "DPOTrainer",
        "DPOTrainerOutput",
    ],
    "trainer.training_configurations": ["TrainArguments", "EasyDeLXRapTureConfig"],
    "trainer.utils": [
        "create_constant_length_dataset",
        "get_formatting_func_from_dataset",
        "conversations_formatting_function",
        "instructions_formatting_function",
    ],
}

_inference_import_structure = {
    "inference.serve_engine": [
        "EasyDeLServeEngine",
        "EasyDeLServeEngineConfig",
        "EngineClient",
    ],
    "inference.generation_pipeline": ["GenerationPipelineConfig", "GenerationPipeline"],
}

_transform_import_structure = {
    "transform": [
        "huggingface_to_easydel",
        "easystate_to_huggingface_model",
        "easystate_to_torch",
    ],
}

_etils_import_structure = {
    "etils": [
        "EasyDeLOptimizers",
        "EasyDeLSchedulers",
        "EasyDeLGradientCheckPointers",
        "EasyDeLState",
        "EasyDeLTimerError",
        "EasyDeLRuntimeError",
        "EasyDeLSyntaxRuntimeError",
        "PartitionAxis",
    ],
}

_utils_import_structure = {
    "utils.utils": ["get_mesh", "RNG"],
}

_smi_import_structure = {
    "smi": ["run", "initialise_tracking", "get_mem"],
}
_import_structure = {}
_import_structure.update(_modules_import_structure)
_import_structure.update(_trainer_import_structure)
_import_structure.update(_inference_import_structure)
_import_structure.update(_transform_import_structure)
_import_structure.update(_etils_import_structure)
_import_structure.update(_utils_import_structure)
_import_structure.update(_smi_import_structure)

if TYPE_CHECKING:
    # INFERENCE IMPORT START HERE
    from .inference.serve_engine import (
        EasyDeLServeEngine as EasyDeLServeEngine,
        EasyDeLServeEngineConfig as EasyDeLServeEngineConfig,
        EngineClient as EngineClient,
    )
    from .inference.generation_pipeline import (
        GenerationPipelineConfig as GenerationPipelineConfig,
        GenerationPipeline as GenerationPipeline,
    )

    # INFERENCE IMPORT END HERE

    # MODULES IMPORT START HERE
    from .modules.llama import (
        FlaxLlamaModel as FlaxLlamaModel,
        FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
        FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
        LlamaConfig as LlamaConfig,
        FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
        VisionLlamaConfig as VisionLlamaConfig,
    )
    from .modules.gpt_j import (
        GPTJConfig as GPTJConfig,
        FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
        FlaxGPTJModel as FlaxGPTJModel,
    )
    from .modules.t5 import (
        T5Config as T5Config,
        FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
        FlaxT5Model as FlaxT5Model,
    )
    from .modules.falcon import (
        FalconConfig as FalconConfig,
        FlaxFalconModel as FlaxFalconModel,
        FlaxFalconForCausalLM as FlaxFalconForCausalLM,
    )
    from .modules.opt import (
        OPTConfig as OPTConfig,
        FlaxOPTForCausalLM as FlaxOPTForCausalLM,
        FlaxOPTModel as FlaxOPTModel,
    )
    from .modules.mistral import (
        MistralConfig as MistralConfig,
        FlaxMistralForCausalLM as FlaxMistralForCausalLM,
        FlaxMistralModel as FlaxMistralModel,
        FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
        VisionMistralConfig as VisionMistralConfig,
    )
    from .modules.palm import (
        FlaxPalmModel as FlaxPalmModel,
        PalmConfig as PalmConfig,
        FlaxPalmForCausalLM as FlaxPalmForCausalLM,
    )
    from .modules.mosaic_mpt import (
        MptConfig as MptConfig,
        MptAttentionConfig as MptAttentionConfig,
        FlaxMptForCausalLM as FlaxMptForCausalLM,
        FlaxMptModel as FlaxMptModel,
    )
    from .modules.gpt_neo_x import (
        GPTNeoXConfig as GPTNeoXConfig,
        FlaxGPTNeoXModel as FlaxGPTNeoXModel,
        FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
    )
    from .modules.lucid_transformer import (
        FlaxLTModel as FlaxLTModel,
        FlaxLTConfig as FlaxLTConfig,
        FlaxLTForCausalLM as FlaxLTForCausalLM,
    )
    from .modules.gpt2 import (
        GPT2Config as GPT2Config,
        FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
        FlaxGPT2Model as FlaxGPT2Model,
    )
    from .modules.mixtral import (
        FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
        FlaxMixtralModel as FlaxMixtralModel,
        MixtralConfig as MixtralConfig,
    )
    from .modules.phi import (
        FlaxPhiForCausalLM as FlaxPhiForCausalLM,
        PhiConfig as PhiConfig,
        FlaxPhiModel as FlaxPhiModel,
    )
    from .modules.qwen1 import (
        FlaxQwen1Model as FlaxQwen1Model,
        FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
        FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
        Qwen1Config as Qwen1Config,
    )
    from .modules.qwen2 import (
        FlaxQwen2Model as FlaxQwen2Model,
        FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
        FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
        Qwen2Config as Qwen2Config,
    )
    from .modules.gemma import (
        FlaxGemmaModel as FlaxGemmaModel,
        GemmaConfig as GemmaConfig,
        FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
    )
    from .modules.stablelm import (
        StableLmConfig as StableLmConfig,
        FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
        FlaxStableLmModel as FlaxStableLmModel,
    )
    from .modules.mamba import (
        FlaxMambaModel as FlaxMambaModel,
        FlaxMambaForCausalLM as FlaxMambaForCausalLM,
        MambaConfig as MambaConfig,
    )
    from .modules.grok_1 import (
        Grok1Config as Grok1Config,
        FlaxGrok1Model as FlaxGrok1Model,
        FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
    )
    from .modules.qwen2_moe import (
        Qwen2MoeConfig as Qwen2MoeConfig,
        FlaxQwen2MoeModel as FlaxQwen2MoeModel,
        FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
    )
    from .modules.whisper import (
        FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
        FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
        FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
        WhisperConfig as WhisperConfig,
    )
    from .modules.cohere import (
        FlaxCohereModel as FlaxCohereModel,
        CohereConfig as CohereConfig,
        FlaxCohereForCausalLM as FlaxCohereForCausalLM,
    )
    from .modules.dbrx import (
        DbrxConfig as DbrxConfig,
        DbrxFFNConfig as DbrxFFNConfig,
        DbrxAttentionConfig as DbrxAttentionConfig,
        FlaxDbrxModel as FlaxDbrxModel,
        FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
    )
    from .modules.phi3 import (
        Phi3Config as Phi3Config,
        FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
        FlaxPhi3Model as FlaxPhi3Model,
    )
    from .modules.arctic import (
        FlaxArcticForCausalLM as FlaxArcticForCausalLM,
        FlaxArcticModel as FlaxArcticModel,
        ArcticConfig as ArcticConfig,
    )
    from .modules.openelm import (
        FlaxOpenELMModel as FlaxOpenELMModel,
        FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
        OpenELMConfig as OpenELMConfig,
    )
    from .modules.deepseek_v2 import (
        FlaxDeepseekV2Model as FlaxDeepseekV2Model,
        FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
        DeepseekV2Config as DeepseekV2Config,
    )
    from .modules.auto_easydel_model import (
        AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
        AutoEasyDeLConfig as AutoEasyDeLConfig,
        AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
        get_modules_by_type as get_modules_by_type,
    )
    from .modules.easydel_modelling_utils import (
        EasyDeLPretrainedConfig as EasyDeLPretrainedConfig,
        EasyDeLFlaxPretrainedModel as EasyDeLFlaxPretrainedModel,
    )
    from .modules.attention_module import AttentionModule as AttentionModule

    # MODULES IMPORT END HERE

    # TRAINER IMPORT START HERE
    from .trainer.training_configurations import (
        TrainArguments as TrainArguments,
        EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
    )
    from .trainer.causal_language_model_trainer import (
        CausalLanguageModelTrainer as CausalLanguageModelTrainer,
        CausalLMTrainerOutput as CausalLMTrainerOutput,
    )
    from .trainer.vision_causal_language_model_trainer import (
        VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
        VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    )
    from .trainer.direct_preference_optimization_trainer import (
        DPOTrainer as DPOTrainer,
    )
    from .trainer.supervised_fine_tuning_trainer import SFTTrainer as SFTTrainer
    from .trainer.odds_ratio_preference_optimization_trainer import (
        ORPOTrainer as ORPOTrainer,
        ORPOTrainerOutput as ORPOTrainerOutput,
    )
    from .trainer.utils import (
        create_constant_length_dataset as create_constant_length_dataset,
        get_formatting_func_from_dataset as get_formatting_func_from_dataset,
        conversations_formatting_function as conversations_formatting_function,
        instructions_formatting_function as instructions_formatting_function,
    )

    # TRAINER IMPORT ENDS HERE

    # UTILS IMPORT START HERE
    from .utils.helpers import get_mesh as get_mesh, RNG as RNG

    # UTILS IMPORT ENDS HERE

    # SMI IMPORT START HERE
    from .smi import (
        run as run,
        initialise_tracking as initialise_tracking,
        get_mem as get_mem,
    )

    # SMI IMPORT ENDS HERE

    # TRANSFORM IMPORT START HERE
    from .transform import (
        huggingface_to_easydel as huggingface_to_easydel,
        easystate_to_huggingface_model as easystate_to_huggingface_model,
        easystate_to_torch as easystate_to_torch,
    )

    # TRANSFORM IMPORT ENDS HERE

    # ETILS IMPORT START HERE
    from .etils import (
        EasyDeLOptimizers as EasyDeLOptimizers,
        EasyDeLSchedulers as EasyDeLSchedulers,
        EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
        EasyDeLState as EasyDeLState,
        EasyDeLTimerError as EasyDeLTimerError,
        EasyDeLRuntimeError as EasyDeLRuntimeError,
        EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
        PartitionAxis as PartitionAxis,
    )

    # ETILS IMPORT ENDS HERE

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )

__version__ = "0.0.68"
