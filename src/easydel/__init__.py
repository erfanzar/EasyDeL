__version__ = "0.0.80"

import os as _os

if bool(
    _os.environ.get("EASYDEL_AUTO", "true")
):  # Taking care of some optional GPU FLAGs
    _os.environ["XLA_FLAGS"] = (
        _os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    )
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# EasyDel Imports
from easydel.etils.easystate import EasyDeLState as EasyDeLState
from easydel.etils.errors import (
    EasyDeLRuntimeError as EasyDeLRuntimeError,
    EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
    EasyDeLTimerError as EasyDeLTimerError,
)
from easydel.etils.etils import (
    EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
    EasyDeLOptimizers as EasyDeLOptimizers,
    EasyDeLSchedulers as EasyDeLSchedulers,
)
from easydel.etils.partition_module import PartitionAxis as PartitionAxis
from easydel.inference.generation_pipeline import (
    ChatPipeline as ChatPipeline,
    GenerationPipeline as GenerationPipeline,
    GenerationPipelineConfig as GenerationPipelineConfig,
)
from easydel.inference.serve_engine import (
    ApiEngine as ApiEngine,
    engine_client as engine_client,
)
from easydel.modules.arctic import (
    ArcticConfig as ArcticConfig,
    FlaxArcticForCausalLM as FlaxArcticForCausalLM,
    FlaxArcticModel as FlaxArcticModel,
)
from easydel.modules.attention_module import (
    AttentionMechanisms as AttentionMechanisms,
    FlexibleAttentionModule as FlexibleAttentionModule,
)
from easydel.modules.auto_models import (
    AutoEasyDeLConfig as AutoEasyDeLConfig,
    AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
    AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
    AutoStateForCausalLM as AutoStateForCausalLM,
    get_modules_by_type as get_modules_by_type,
)
from easydel.modules.cohere import (
    CohereConfig as CohereConfig,
    FlaxCohereForCausalLM as FlaxCohereForCausalLM,
    FlaxCohereModel as FlaxCohereModel,
)
from easydel.modules.dbrx import (
    DbrxAttentionConfig as DbrxAttentionConfig,
    DbrxConfig as DbrxConfig,
    DbrxFFNConfig as DbrxFFNConfig,
    FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
    FlaxDbrxModel as FlaxDbrxModel,
)
from easydel.modules.deepseek_v2 import (
    DeepseekV2Config as DeepseekV2Config,
    FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
    FlaxDeepseekV2Model as FlaxDeepseekV2Model,
)
from easydel.modules.falcon import (
    FalconConfig as FalconConfig,
    FlaxFalconForCausalLM as FlaxFalconForCausalLM,
    FlaxFalconModel as FlaxFalconModel,
)
from easydel.modules.gemma import (
    FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
    FlaxGemmaModel as FlaxGemmaModel,
    GemmaConfig as GemmaConfig,
)
from easydel.modules.gemma2 import (
    FlaxGemma2ForCausalLM as FlaxGemma2ForCausalLM,
    FlaxGemma2Model as FlaxGemma2Model,
    Gemma2Config as Gemma2Config,
)
from easydel.modules.gpt2 import (
    FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
    FlaxGPT2Model as FlaxGPT2Model,
    GPT2Config as GPT2Config,
)
from easydel.modules.gpt_j import (
    FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
    FlaxGPTJModel as FlaxGPTJModel,
    GPTJConfig as GPTJConfig,
)
from easydel.modules.gpt_neo_x import (
    FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
    FlaxGPTNeoXModel as FlaxGPTNeoXModel,
    GPTNeoXConfig as GPTNeoXConfig,
)
from easydel.modules.grok_1 import (
    FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
    FlaxGrok1Model as FlaxGrok1Model,
    Grok1Config as Grok1Config,
)
from easydel.modules.llama import (
    FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
    FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
    FlaxLlamaModel as FlaxLlamaModel,
    FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
    LlamaConfig as LlamaConfig,
    VisionLlamaConfig as VisionLlamaConfig,
)
from easydel.modules.lucid_transformer import (
    FlaxLTConfig as FlaxLTConfig,
    FlaxLTForCausalLM as FlaxLTForCausalLM,
    FlaxLTModel as FlaxLTModel,
)
from easydel.modules.mamba import (
    FlaxMambaForCausalLM as FlaxMambaForCausalLM,
    FlaxMambaModel as FlaxMambaModel,
    MambaConfig as MambaConfig,
)
from easydel.modules.mistral import (
    FlaxMistralForCausalLM as FlaxMistralForCausalLM,
    FlaxMistralModel as FlaxMistralModel,
    FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
    MistralConfig as MistralConfig,
    VisionMistralConfig as VisionMistralConfig,
)
from easydel.modules.mixtral import (
    FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
    FlaxMixtralModel as FlaxMixtralModel,
    MixtralConfig as MixtralConfig,
)
from easydel.modules.modeling_utils import (
    EDPretrainedConfig as EDPretrainedConfig,
    EDPretrainedModel as EDPretrainedModel,
)
from easydel.modules.mosaic_mpt import (
    FlaxMptForCausalLM as FlaxMptForCausalLM,
    FlaxMptModel as FlaxMptModel,
    MptAttentionConfig as MptAttentionConfig,
    MptConfig as MptConfig,
)
from easydel.modules.olmo import (
    FlaxOlmoForCausalLM as FlaxOlmoForCausalLM,
    FlaxOlmoModel as FlaxOlmoModel,
    OlmoConfig as OlmoConfig,
)
from easydel.modules.openelm import (
    FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
    FlaxOpenELMModel as FlaxOpenELMModel,
    OpenELMConfig as OpenELMConfig,
)
from easydel.modules.opt import (
    FlaxOPTForCausalLM as FlaxOPTForCausalLM,
    FlaxOPTModel as FlaxOPTModel,
    OPTConfig as OPTConfig,
)
from easydel.modules.palm import (
    FlaxPalmForCausalLM as FlaxPalmForCausalLM,
    FlaxPalmModel as FlaxPalmModel,
    PalmConfig as PalmConfig,
)
from easydel.modules.phi import (
    FlaxPhiForCausalLM as FlaxPhiForCausalLM,
    FlaxPhiModel as FlaxPhiModel,
    PhiConfig as PhiConfig,
)
from easydel.modules.phi3 import (
    FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
    FlaxPhi3Model as FlaxPhi3Model,
    Phi3Config as Phi3Config,
)
from easydel.modules.qwen1 import (
    FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
    FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
    FlaxQwen1Model as FlaxQwen1Model,
    Qwen1Config as Qwen1Config,
)
from easydel.modules.qwen2 import (
    FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
    FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
    FlaxQwen2Model as FlaxQwen2Model,
    Qwen2Config as Qwen2Config,
)
from easydel.modules.qwen2_moe import (
    FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
    FlaxQwen2MoeModel as FlaxQwen2MoeModel,
    Qwen2MoeConfig as Qwen2MoeConfig,
)
from easydel.modules.stablelm import (
    FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
    FlaxStableLmModel as FlaxStableLmModel,
    StableLmConfig as StableLmConfig,
)
from easydel.modules.t5 import (
    FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
    FlaxT5Model as FlaxT5Model,
    T5Config as T5Config,
)
from easydel.modules.whisper import (
    FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
    FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
    FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
    WhisperConfig as WhisperConfig,
)

from easydel.modules.xerxes import (
    XerxesConfig as XerxesConfig,
    FlaxXerxesForCausalLM as FlaxXerxesForCausalLM,
    FlaxXerxesModel as FlaxXerxesModel,
)

from easydel.smi import (
    get_mem as get_mem,
    initialise_tracking as initialise_tracking,
    run as run,
)
from easydel.trainers import (
    BaseTrainer as BaseTrainer,
    CausalLanguageModelTrainer as CausalLanguageModelTrainer,
    CausalLMTrainerOutput as CausalLMTrainerOutput,
    DPOTrainer as DPOTrainer,
    DPOTrainerOutput as DPOTrainerOutput,
    LoraRaptureConfig as LoraRaptureConfig,
    JaxDistributedConfig as JaxDistributedConfig,
    ORPOTrainer as ORPOTrainer,
    ORPOTrainerOutput as ORPOTrainerOutput,
    SFTTrainer as SFTTrainer,
    TrainArguments as TrainArguments,
    VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
    VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    conversations_formatting_function as conversations_formatting_function,
    create_constant_length_dataset as create_constant_length_dataset,
    get_formatting_func_from_dataset as get_formatting_func_from_dataset,
    instructions_formatting_function as instructions_formatting_function,
)
from easydel.transform import (
    easystate_to_huggingface_model as easystate_to_huggingface_model,
    easystate_to_torch as easystate_to_torch,
    torch_dict_to_easydel_params as torch_dict_to_easydel_params,
)

from easydel import modules as modules
from easydel import etils as etils

import fjformer as _fj

_targeted_version = "0.0.73"
assert _fj.__version__ == _targeted_version, (
    f"this version os EasyDeL is only compatible with fjformer=={_targeted_version},"
    f" but found fjformer {_fj.__version__}"
)
