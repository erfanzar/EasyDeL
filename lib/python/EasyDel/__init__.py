from .serve import (
    EasyServe as EasyServe,
    EasyServeConfig as EasyServeConfig,
    LLMBaseReq as LLMBaseReq,
    GenerateAPIRequest as GenerateAPIRequest,
    ConversationItem as ConversationItem,
    ModelOutput as ModelOutput,
    BaseModel as BaseModel,
    EasyClient as EasyClient,
    GradioUserInference as GradioUserInference,
    ChatRequest as ChatRequest,
    InstructRequest as InstructRequest,
    PyTorchServer as PyTorchServer,
    PyTorchServerConfig as PyTorchServerConfig,
    JAXServer as JAXServer,
    JAXServerConfig as JAXServerConfig
)

from .modules.llama import (
    LlamaConfig as LlamaConfig,
    FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
    FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
    FlaxLlamaModel as FlaxLlamaModel
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
    FlaxMistralModel as FlaxMistralModel
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
from .modules.auto_easydel_model import (
    AutoEasyDelModelForCausalLM as AutoEasyDelModelForCausalLM,
    AutoEasyDelConfig as AutoEasyDelConfig,
    AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
    get_modules_by_type as get_modules_by_type
)

from .utils.utils import (
    get_mesh as get_mesh,
    RNG as RNG
)

from .trainer import (
    CausalLanguageModelTrainer as CausalLanguageModelTrainer,
    EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
    TrainArguments as TrainArguments,
    create_casual_language_model_evaluation_step as create_casual_language_model_evaluation_step,
    create_casual_language_model_train_step as create_casual_language_model_train_step,
    create_vision_casual_language_model_train_step as create_vision_casual_language_model_train_step,
    create_vision_casual_language_model_evaluation_step as create_vision_casual_language_model_evaluation_step,
    VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
    VisionCausalLanguageModelStepOutput as VisionCausalLanguageModelStepOutput
)

from .reinforcement_learning import (
    create_dpo_eval_function as create_dpo_eval_function,
    create_dpo_train_function as create_dpo_train_function,
    DPOTrainer as DPOTrainer,
    create_concatenated_forward as create_concatenated_forward,
    AutoRLModelForCasualLMWithValueHead as AutoRLModelForCasualLMWithValueHead
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
    EasyDelOptimizers as EasyDelOptimizers,
    EasyDelSchedulers as EasyDelSchedulers,
    EasyDelGradientCheckPointers as EasyDelGradientCheckPointers,
    EasyDelState as EasyDelState,
    EasyDelTimerError as EasyDelTimerError,
    EasyDelRuntimeError as EasyDelRuntimeError,
    EasyDelSyntaxRuntimeError as EasyDelSyntaxRuntimeError
)

__all__ = (
    # API Serving Modules

    "EasyServe",
    "EasyServeConfig",
    "LLMBaseReq",
    "GenerateAPIRequest",
    "ConversationItem",
    "ModelOutput",
    "BaseModel",
    "EasyClient",
    "GradioUserInference",
    "ChatRequest",
    "InstructRequest",
    "PyTorchServer",
    "PyTorchServerConfig",
    "JAXServer",
    "JAXServerConfig",

    # Models

    # LLama Models
    "LlamaConfig",
    "FlaxLlamaForCausalLM",
    "FlaxLlamaForSequenceClassification",
    "FlaxLlamaModel",

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
    "FlaxMistralForCausalLM",
    "FlaxMistralModel",

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

    # AutoModels Models
    "AutoEasyDelModelForCausalLM",
    "AutoEasyDelConfig",
    "AutoShardAndGatherFunctions",
    "get_modules_by_type",

    # Utils
    "get_mesh",
    "RNG",

    # Trainers
    "CausalLanguageModelTrainer",
    "EasyDeLXRapTureConfig",
    "TrainArguments",
    "create_casual_language_model_evaluation_step",
    "create_casual_language_model_train_step",

    "VisionCausalLanguageModelStepOutput",
    "VisionCausalLanguageModelTrainer",
    "create_vision_casual_language_model_evaluation_step",
    "create_vision_casual_language_model_train_step",

    "create_dpo_eval_function",
    "create_dpo_train_function",
    "DPOTrainer",
    "create_concatenated_forward",
    "AutoRLModelForCasualLMWithValueHead",

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
    "EasyDelOptimizers",
    "EasyDelSchedulers",
    "EasyDelGradientCheckPointers",
    "EasyDelState",
    "EasyDelTimerError",
    "EasyDelRuntimeError",
    "EasyDelSyntaxRuntimeError"
)

__version__ = "0.0.55"
