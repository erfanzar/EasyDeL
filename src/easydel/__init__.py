import os as _os

if bool(_os.environ.get("EASYDEL_AUTO", "true")):
    _os.environ["XLA_FLAGS"] = (
        _os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    )
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# INFERENCE IMPORT START HERE
from easydel.inference.serve_engine import (
    EasyDeLServeEngine as EasyDeLServeEngine,
    EasyDeLServeEngineConfig as EasyDeLServeEngineConfig,
    EngineClient as EngineClient,
)
from easydel.inference.generation_pipeline import (
    GenerationPipelineConfig as GenerationPipelineConfig,
    GenerationPipeline as GenerationPipeline,
)

# INFERENCE IMPORT END HERE

# MODULES IMPORT START HERE
from easydel.modules.llama import (
    FlaxLlamaModel as FlaxLlamaModel,
    FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
    FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
    LlamaConfig as LlamaConfig,
    FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
    VisionLlamaConfig as VisionLlamaConfig,
)
from easydel.modules.gpt_j import (
    GPTJConfig as GPTJConfig,
    FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
    FlaxGPTJModel as FlaxGPTJModel,
)
from easydel.modules.t5 import (
    T5Config as T5Config,
    FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
    FlaxT5Model as FlaxT5Model,
)
from easydel.modules.falcon import (
    FalconConfig as FalconConfig,
    FlaxFalconModel as FlaxFalconModel,
    FlaxFalconForCausalLM as FlaxFalconForCausalLM,
)
from easydel.modules.opt import (
    OPTConfig as OPTConfig,
    FlaxOPTForCausalLM as FlaxOPTForCausalLM,
    FlaxOPTModel as FlaxOPTModel,
)
from easydel.modules.mistral import (
    MistralConfig as MistralConfig,
    FlaxMistralForCausalLM as FlaxMistralForCausalLM,
    FlaxMistralModel as FlaxMistralModel,
    FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
    VisionMistralConfig as VisionMistralConfig,
)
from easydel.modules.palm import (
    FlaxPalmModel as FlaxPalmModel,
    PalmConfig as PalmConfig,
    FlaxPalmForCausalLM as FlaxPalmForCausalLM,
)
from easydel.modules.mosaic_mpt import (
    MptConfig as MptConfig,
    MptAttentionConfig as MptAttentionConfig,
    FlaxMptForCausalLM as FlaxMptForCausalLM,
    FlaxMptModel as FlaxMptModel,
)
from easydel.modules.gpt_neo_x import (
    GPTNeoXConfig as GPTNeoXConfig,
    FlaxGPTNeoXModel as FlaxGPTNeoXModel,
    FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
)
from easydel.modules.lucid_transformer import (
    FlaxLTModel as FlaxLTModel,
    FlaxLTConfig as FlaxLTConfig,
    FlaxLTForCausalLM as FlaxLTForCausalLM,
)
from easydel.modules.gpt2 import (
    GPT2Config as GPT2Config,
    FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
    FlaxGPT2Model as FlaxGPT2Model,
)
from easydel.modules.mixtral import (
    FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
    FlaxMixtralModel as FlaxMixtralModel,
    MixtralConfig as MixtralConfig,
)
from easydel.modules.phi import (
    FlaxPhiForCausalLM as FlaxPhiForCausalLM,
    PhiConfig as PhiConfig,
    FlaxPhiModel as FlaxPhiModel,
)
from easydel.modules.qwen1 import (
    FlaxQwen1Model as FlaxQwen1Model,
    FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
    FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
    Qwen1Config as Qwen1Config,
)
from easydel.modules.qwen2 import (
    FlaxQwen2Model as FlaxQwen2Model,
    FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
    FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
    Qwen2Config as Qwen2Config,
)
from easydel.modules.olmo import FlaxOlmoModel, FlaxOlmoForCausalLM, OlmoConfig
from easydel.modules.gemma import (
    FlaxGemmaModel as FlaxGemmaModel,
    GemmaConfig as GemmaConfig,
    FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
)
from easydel.modules.stablelm import (
    StableLmConfig as StableLmConfig,
    FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
    FlaxStableLmModel as FlaxStableLmModel,
)
from easydel.modules.mamba import (
    FlaxMambaModel as FlaxMambaModel,
    FlaxMambaForCausalLM as FlaxMambaForCausalLM,
    MambaConfig as MambaConfig,
)
from easydel.modules.grok_1 import (
    Grok1Config as Grok1Config,
    FlaxGrok1Model as FlaxGrok1Model,
    FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
)
from easydel.modules.qwen2_moe import (
    Qwen2MoeConfig as Qwen2MoeConfig,
    FlaxQwen2MoeModel as FlaxQwen2MoeModel,
    FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
)
from easydel.modules.whisper import (
    FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
    FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
    FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
    WhisperConfig as WhisperConfig,
)
from easydel.modules.cohere import (
    FlaxCohereModel as FlaxCohereModel,
    CohereConfig as CohereConfig,
    FlaxCohereForCausalLM as FlaxCohereForCausalLM,
)
from easydel.modules.dbrx import (
    DbrxConfig as DbrxConfig,
    DbrxFFNConfig as DbrxFFNConfig,
    DbrxAttentionConfig as DbrxAttentionConfig,
    FlaxDbrxModel as FlaxDbrxModel,
    FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
)
from easydel.modules.phi3 import (
    Phi3Config as Phi3Config,
    FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
    FlaxPhi3Model as FlaxPhi3Model,
)
from easydel.modules.arctic import (
    FlaxArcticForCausalLM as FlaxArcticForCausalLM,
    FlaxArcticModel as FlaxArcticModel,
    ArcticConfig as ArcticConfig,
)
from easydel.modules.openelm import (
    FlaxOpenELMModel as FlaxOpenELMModel,
    FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
    OpenELMConfig as OpenELMConfig,
)
from easydel.modules.deepseek_v2 import (
    FlaxDeepseekV2Model as FlaxDeepseekV2Model,
    FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
    DeepseekV2Config as DeepseekV2Config,
)
from easydel.modules.auto_easydel_model import (
    AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
    AutoEasyDeLConfig as AutoEasyDeLConfig,
    AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
    get_modules_by_type as get_modules_by_type,
)
from easydel.modules.easydel_modelling_utils import (
    EasyDeLPretrainedConfig as EasyDeLPretrainedConfig,
    EasyDeLFlaxPretrainedModel as EasyDeLFlaxPretrainedModel,
)
from easydel.modules.attention_module import (
    AttentionModule as AttentionModule,
    AttentionMechanisms as AttentionMechanisms,
)

# MODULES IMPORT END HERE

# TRAINER IMPORT START HERE
from easydel.trainer import (
    VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
    VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    DPOTrainer as DPOTrainer,
    SFTTrainer as SFTTrainer,
    ORPOTrainer as ORPOTrainer,
    ORPOTrainerOutput as ORPOTrainerOutput,
    create_constant_length_dataset as create_constant_length_dataset,
    get_formatting_func_from_dataset as get_formatting_func_from_dataset,
    conversations_formatting_function as conversations_formatting_function,
    instructions_formatting_function as instructions_formatting_function,
    JaxDistributedConfig as JaxDistributedConfig,
    CausalLanguageModelTrainer as CausalLanguageModelTrainer,
    CausalLMTrainerOutput as CausalLMTrainerOutput,
    TrainArguments as TrainArguments,
    EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
    BaseTrainer as BaseTrainer,
    DPOTrainerOutput as DPOTrainerOutput,
)

# TRAINER IMPORT ENDS HERE

# UTILS IMPORT START HERE
from easydel.utils.helpers import get_mesh as get_mesh, RNG as RNG

# UTILS IMPORT ENDS HERE

# SMI IMPORT START HERE
from easydel.smi import (
    run as run,
    initialise_tracking as initialise_tracking,
    get_mem as get_mem,
)

# SMI IMPORT ENDS HERE

# TRANSFORM IMPORT START HERE
from easydel.transform import (
    huggingface_to_easydel as huggingface_to_easydel,
    easystate_to_huggingface_model as easystate_to_huggingface_model,
    easystate_to_torch as easystate_to_torch,
)

# TRANSFORM IMPORT ENDS HERE

# ETILS IMPORT START HERE
from easydel.etils.errors import (
    EasyDeLTimerError as EasyDeLTimerError,
    EasyDeLRuntimeError as EasyDeLRuntimeError,
    EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
)
from easydel.etils.easystate import EasyDeLState as EasyDeLState
from easydel.etils.etils import (
    EasyDeLOptimizers as EasyDeLOptimizers,
    EasyDeLSchedulers as EasyDeLSchedulers,
    EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
)
from easydel.etils.partition_module import PartitionAxis as PartitionAxis

# ETILS IMPORT ENDS HERE
__version__ = "0.0.69"
