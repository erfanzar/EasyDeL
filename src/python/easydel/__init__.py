from .serve import (
    EasyServe as EasyServe,
    EasyServeConfig as EasyServeConfig,
    LLMBaseReq as LLMBaseReq,
    EasyClient as EasyClient,
    GradioUserInference as GradioUserInference,
    ChatRequest as ChatRequest,
    InstructRequest as InstructRequest,
    PyTorchServer as PyTorchServer,
    PyTorchServerConfig as PyTorchServerConfig,
    JAXServer as JAXServer,
    JAXServerConfig as JAXServerConfig,
    create_generate_function as create_generate_function
)

from .modules.llama import (
    FlaxLlamaModel as FlaxLlamaModel,
    FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
    FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
    LlamaConfig as LlamaConfig,
    FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
    VisionLlamaConfig as VisionLlamaConfig
)
from .modules.gpt_j import (
    GPTJConfig as GPTJConfig,
    FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
    FlaxGPTJModel as FlaxGPTJModel,
)
from .modules.t5 import (
    T5Config as T5Config,
    FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
    FlaxT5Model as FlaxT5Model
)
from .modules.falcon import (
    FalconConfig as FalconConfig,
    FlaxFalconModel as FlaxFalconModel,
    FlaxFalconForCausalLM as FlaxFalconForCausalLM
)
from .modules.opt import (
    OPTConfig as OPTConfig,
    FlaxOPTForCausalLM as FlaxOPTForCausalLM,
    FlaxOPTModel as FlaxOPTModel
)
from .modules.mistral import (
    MistralConfig as MistralConfig,
    FlaxMistralForCausalLM as FlaxMistralForCausalLM,
    FlaxMistralModel as FlaxMistralModel,
    FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
    VisionMistralConfig as VisionMistralConfig
)
from .modules.palm import (
    FlaxPalmModel as FlaxPalmModel,
    PalmConfig as PalmConfig,
    FlaxPalmForCausalLM as FlaxPalmForCausalLM
)

from .modules.mosaic_mpt import (
    MptConfig as MptConfig,
    FlaxMptForCausalLM as FlaxMptForCausalLM,
    FlaxMptModel as FlaxMptModel
)

from .modules.gpt_neo_x import (
    GPTNeoXConfig as GPTNeoXConfig,
    FlaxGPTNeoXModel as FlaxGPTNeoXModel,
    FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM
)

from .modules.lucid_transformer import (
    FlaxLTModel as FlaxLTModel,
    FlaxLTConfig as FlaxLTConfig,
    FlaxLTForCausalLM as FlaxLTForCausalLM
)

from .modules.gpt2 import (
    # GPT2 code is from huggingface but in the version of huggingface they don't support gradient checkpointing
    # and pjit attention force
    GPT2Config as GPT2Config,
    FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
    FlaxGPT2Model as FlaxGPT2Model
)

from .modules.mixtral import (
    FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
    FlaxMixtralModel as FlaxMixtralModel,
    MixtralConfig as MixtralConfig
)

from .modules.phi import (
    FlaxPhiForCausalLM as FlaxPhiForCausalLM,
    PhiConfig as PhiConfig,
    FlaxPhiModel as FlaxPhiModel
)
from .modules.qwen1 import (
    FlaxQwen1Model as FlaxQwen1Model,
    FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
    FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
    Qwen1Config as Qwen1Config
)

from .modules.qwen2 import (
    FlaxQwen2Model as FlaxQwen2Model,
    FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
    FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
    Qwen2Config as Qwen2Config
)

from .modules.gemma import (
    FlaxGemmaModel as FlaxGemmaModel,
    GemmaConfig as GemmaConfig,
    FlaxGemmaForCausalLM as FlaxGemmaForCausalLM
)
from .modules.stablelm import (

    StableLmConfig as StableLmConfig,
    FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
    FlaxStableLmModel as FlaxStableLmModel
)

from .modules.mamba import (

    FlaxMambaModel as FlaxMambaModel,
    FlaxMambaForCausalLM as FlaxMambaForCausalLM,
    MambaConfig as MambaConfig
)

from .modules.grok_1 import (
    Grok1Config as Grok1Config,
    FlaxGrok1Model as FlaxGrok1Model,
    FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM
)

from .modules.qwen2_moe import (
    Qwen2MoeConfig as Qwen2MoeConfig,
    FlaxQwen2MoeModel as FlaxQwen2MoeModel,
    FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM
)
from .modules.whisper import (
    FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
    FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
    FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
    WhisperConfig as WhisperConfig
)

from .modules.cohere import (
    FlaxCohereModel as FlaxCohereModel,
    CohereConfig as CohereConfig,
    FlaxCohereForCausalLM as FlaxCohereForCausalLM
)

from .modules.dbrx import (
    DbrxConfig as DbrxConfig,
    DbrxFFNConfig as DbrxFFNConfig,
    DbrxAttentionConfig as DbrxAttentionConfig,
    FlaxDbrxModel as FlaxDbrxModel,
    FlaxDbrxForCausalLM as FlaxDbrxForCausalLM
)

from .modules.phi3 import (
    Phi3Config as Phi3Config,
    FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
    FlaxPhi3Model as FlaxPhi3Model,
)

from .modules.arctic import (
    FlaxArcticForCausalLM as FlaxArcticForCausalLM,
    FlaxArcticModel as FlaxArcticModel,
    ArcticConfig as ArcticConfig
)

from .modules.openelm import (
    FlaxOpenELMModel as FlaxOpenELMModel,
    FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
    OpenELMConfig as OpenELMConfig
)

from .modules.deepseek_v2 import (
    FlaxDeepseekV2Model as FlaxDeepseekV2Model,
    FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
    DeepseekV2Config as DeepseekV2Config
)

from .modules.auto_easydel_model import (
    AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
    AutoEasyDeLConfig as AutoEasyDeLConfig,
    AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
    get_modules_by_type as get_modules_by_type
)

from .modules.attention_module import AttentionModule

from .utils.utils import (
    get_mesh as get_mesh,
    RNG as RNG
)
from .trainer import (
    TrainArguments as TrainArguments,
    EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
    create_casual_language_model_evaluation_step as create_casual_language_model_evaluation_step,
    create_casual_language_model_train_step as create_casual_language_model_train_step,
    CausalLanguageModelTrainer as CausalLanguageModelTrainer,
    VisionCausalLanguageModelStepOutput as VisionCausalLanguageModelStepOutput,
    VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
    create_vision_casual_language_model_evaluation_step as create_vision_casual_language_model_evaluation_step,
    create_vision_casual_language_model_train_step as create_vision_casual_language_model_train_step,
    DPOTrainer as DPOTrainer,
    create_dpo_eval_function as create_dpo_eval_function,
    create_concatenated_forward as create_concatenated_forward,
    create_dpo_train_function as create_dpo_train_function,
    concatenated_dpo_inputs as concatenated_dpo_inputs,
    SFTTrainer as SFTTrainer,
    create_constant_length_dataset as create_constant_length_dataset,
    get_formatting_func_from_dataset as get_formatting_func_from_dataset,
    conversations_formatting_function as conversations_formatting_function,
    instructions_formatting_function as instructions_formatting_function,
    ORPOTrainerOutput as ORPOTrainerOutput,
    odds_ratio_loss as odds_ratio_loss,
    ORPOTrainer as ORPOTrainer,
    create_orpo_step_function as create_orpo_step_function,
    create_orpo_concatenated_forward as create_orpo_concatenated_forward
)

from .smi import (
    run as smi_run,
    initialise_tracking as initialise_tracking,
    get_mem as get_mem
)

from .transform import (
    huggingface_to_easydel as huggingface_to_easydel,
    easystate_to_huggingface_model as easystate_to_huggingface_model,
    easystate_to_torch as easystate_to_torch,
    falcon_convert_flax_to_pt_7b as falcon_convert_flax_to_pt_7b,
    falcon_from_pretrained as falcon_from_pretrained,
    falcon_convert_hf_to_flax as falcon_convert_hf_to_flax,
    mpt_convert_pt_to_flax_1b as mpt_convert_pt_to_flax_1b,
    mpt_convert_pt_to_flax_7b as mpt_convert_pt_to_flax_7b,
    mpt_convert_flax_to_pt_7b as mpt_convert_flax_to_pt_7b,
    mpt_from_pretrained as mpt_from_pretrained,
    mistral_convert_hf_to_flax_load as mistral_convert_hf_to_flax_load,
    mistral_convert_flax_to_pt as mistral_convert_flax_to_pt,
    mistral_from_pretrained as mistral_from_pretrained,
    falcon_convert_pt_to_flax_7b as falcon_convert_pt_to_flax_7b,
    mistral_convert_hf_to_flax as mistral_convert_hf_to_flax,
    mpt_convert_flax_to_pt_1b as mpt_convert_flax_to_pt_1b,
    llama_convert_flax_to_pt as llama_convert_flax_to_pt,
    llama_convert_hf_to_flax_load as llama_convert_hf_to_flax_load,
    llama_convert_hf_to_flax as llama_convert_hf_to_flax,
    llama_from_pretrained as llama_from_pretrained
)
from .etils import (
    EasyDeLOptimizers as EasyDeLOptimizers,
    EasyDeLSchedulers as EasyDeLSchedulers,
    EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
    EasyDeLState as EasyDeLState,
    EasyDeLTimerError as EasyDeLTimerError,
    EasyDeLRuntimeError as EasyDeLRuntimeError,
    EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError
)

__all__ = (
    # API Serving Modules

    "EasyServe",
    "EasyServeConfig",
    "LLMBaseReq",
    "EasyClient",
    "GradioUserInference",
    "ChatRequest",
    "InstructRequest",
    "PyTorchServer",
    "PyTorchServerConfig",
    "JAXServer",
    "JAXServerConfig",
    "create_generate_function",

    # Models

    # LLama Models
    "LlamaConfig",
    "VisionLlamaConfig",
    "FlaxLlamaForCausalLM",
    "FlaxLlamaForSequenceClassification",
    "FlaxLlamaModel",
    "FlaxVisionMistralForCausalLM",

    # GPT-J Models
    "GPTJConfig",
    "FlaxGPTJForCausalLM",
    "FlaxGPTJModel",

    # T5 Models
    "T5Config",
    "FlaxT5ForConditionalGeneration",
    "FlaxT5Model",

    # Falcon Models
    "FalconConfig",
    "FlaxFalconModel",
    "FlaxFalconForCausalLM",

    # OPT Models
    "OPTConfig",
    "FlaxOPTForCausalLM",
    "FlaxOPTModel",

    # Mistral Models
    "MistralConfig",
    "VisionMistralConfig",
    "FlaxMistralForCausalLM",
    "FlaxMistralModel",
    "FlaxVisionLlamaForCausalLM",

    # Palm Models
    "FlaxPalmModel",
    "PalmConfig",
    "FlaxPalmForCausalLM",

    # Mpt Models
    "MptConfig",
    "FlaxMptForCausalLM",
    "FlaxMptModel",

    # GPTNeoX Models
    "GPTNeoXConfig",
    "FlaxGPTNeoXModel",
    "FlaxGPTNeoXForCausalLM",

    # LucidTransformer Models
    "FlaxLTModel",
    "FlaxLTConfig",
    "FlaxLTForCausalLM",

    # GPT2 Models
    "GPT2Config",
    "FlaxGPT2LMHeadModel",
    "FlaxGPT2Model",

    # Mixtral Models
    "FlaxMixtralForCausalLM",
    "FlaxMixtralModel",
    "MixtralConfig",

    # PHI-2 Models
    "FlaxPhiForCausalLM",
    "PhiConfig",
    "FlaxPhiModel",

    # Qwen1 Models
    "FlaxQwen1Model",
    "FlaxQwen1ForCausalLM",
    "FlaxQwen1ForSequenceClassification",
    "Qwen1Config",

    # Qwen2 Models
    "FlaxQwen2Model",
    "FlaxQwen2ForCausalLM",
    "FlaxQwen2ForSequenceClassification",
    "Qwen2Config",

    # Gemma Models
    "FlaxGemmaModel",
    "GemmaConfig",
    "FlaxGemmaForCausalLM",

    # StableLM Models
    "StableLmConfig",
    "FlaxStableLmForCausalLM",
    "FlaxStableLmModel",

    # Mamba Models
    "FlaxMambaModel",
    "FlaxMambaForCausalLM",
    "MambaConfig",

    # Grok-1 Models
    "Grok1Config",
    "FlaxGrok1Model",
    "FlaxGrok1ForCausalLM",

    # Qwen2Moe
    "Qwen2MoeConfig",
    "FlaxQwen2MoeModel",
    "FlaxQwen2MoeForCausalLM",

    # Whisper
    "WhisperConfig",
    "FlaxWhisperTimeStampLogitsProcessor",
    "FlaxWhisperForAudioClassification",
    "FlaxWhisperForConditionalGeneration",

    # Cohere
    "FlaxCohereModel",
    "CohereConfig",
    "FlaxCohereForCausalLM",

    # Dbrx

    "FlaxDbrxModel",
    "FlaxDbrxForCausalLM",
    "DbrxConfig",
    "DbrxFFNConfig",
    "DbrxAttentionConfig",

    # Phi3

    "Phi3Config",
    "FlaxPhi3ForCausalLM",
    "FlaxPhi3Model",

    # Arctic

    "FlaxArcticForCausalLM",
    "FlaxArcticModel",
    "ArcticConfig",

    # OpenELM

    "FlaxOpenELMForCausalLM",
    "FlaxOpenELMModel",
    "OpenELMConfig",

    # DeepseekV2

    "DeepseekV2Config",
    "FlaxDeepseekV2Model",
    "FlaxDeepseekV2ForCausalLM",

    # AutoModels Models
    "AutoEasyDeLModelForCausalLM",
    "AutoEasyDeLConfig",
    "AutoShardAndGatherFunctions",
    "get_modules_by_type",

    # Attention Module

    "AttentionModule",

    # Utils
    "get_mesh",
    "RNG",

    # Trainers
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
    "concatenated_dpo_inputs",

    "SFTTrainer",
    "create_constant_length_dataset",
    "get_formatting_func_from_dataset",
    "conversations_formatting_function",
    "instructions_formatting_function",

    "ORPOTrainer",
    "create_orpo_step_function",
    "create_orpo_concatenated_forward",
    "odds_ratio_loss",
    "ORPOTrainerOutput",

    # SMI Modules
    "smi_run",
    "initialise_tracking",
    "get_mem",

    # Converters
    "huggingface_to_easydel",
    "easystate_to_huggingface_model",
    "easystate_to_torch",
    "falcon_convert_flax_to_pt_7b",
    "falcon_from_pretrained",
    "falcon_convert_hf_to_flax",
    "mpt_convert_pt_to_flax_1b",
    "mpt_convert_pt_to_flax_7b",
    "mpt_convert_flax_to_pt_7b",
    "mpt_from_pretrained",
    "mistral_convert_hf_to_flax_load",
    "mistral_convert_flax_to_pt",
    'mistral_from_pretrained',
    "falcon_convert_pt_to_flax_7b",
    "mistral_convert_hf_to_flax",
    "mpt_convert_flax_to_pt_1b",
    "llama_convert_flax_to_pt",
    "llama_convert_hf_to_flax_load",
    "llama_convert_hf_to_flax",
    "llama_from_pretrained",

    # ETils Modules / ETils Errors
    "EasyDeLOptimizers",
    "EasyDeLSchedulers",
    "EasyDeLGradientCheckPointers",
    "EasyDeLState",
    "EasyDeLTimerError",
    "EasyDeLRuntimeError",
    "EasyDeLSyntaxRuntimeError"
)

__version__ = "0.0.65"
