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

# INFERENCE IMPORT START HERE
from easydel.etils.easystate import EasyDeLState as EasyDeLState
from easydel.etils.errors import (
    EasyDeLRuntimeError as EasyDeLRuntimeError,
    EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
)
from easydel.inference.generation_pipeline import (
    ChatPipeline as ChatPipeline,
    GenerationPipeline as GenerationPipeline,
    GenerationPipelineConfig as GenerationPipelineConfig,
)
from easydel.inference.serve_engine import (
    ApiEngine as ApiEngine,
    engine_client as engine_client,
)

# TRANSFORM IMPORT ENDS HERE

# ETILS IMPORT START HERE
from easydel.etils.errors import EasyDeLTimerError as EasyDeLTimerError
from easydel.etils.etils import (
    EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
    EasyDeLOptimizers as EasyDeLOptimizers,
    EasyDeLSchedulers as EasyDeLSchedulers,
)
from easydel.etils.partition_module import PartitionAxis as PartitionAxis

# ETILS IMPORT ENDS HERE

# MODEL IMPORT START HERE
from easydel.models.arctic import (
    ArcticConfig as ArcticConfig,
    ArcticForCausalLM as ArcticForCausalLM,
    ArcticModel as ArcticModel,
)
from easydel.models.attention_module import (
    AttentionMechanisms as AttentionMechanisms,
    FlexibleAttentionModule as FlexibleAttentionModule,
)
from easydel.models.auto_easydel_model import (
    AutoEasyDeLConfig as AutoEasyDeLConfig,
    AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
    AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
    get_models_by_type as get_models_by_type,
)
from easydel.models.cohere import (
    CohereConfig as CohereConfig,
    CohereForCausalLM as CohereForCausalLM,
    CohereModel as CohereModel,
)
from easydel.models.dbrx import (
    DbrxAttentionConfig as DbrxAttentionConfig,
    DbrxConfig as DbrxConfig,
    DbrxFFNConfig as DbrxFFNConfig,
    DbrxForCausalLM as DbrxForCausalLM,
    DbrxModel as DbrxModel,
)
from easydel.models.deepseek_v2 import (
    DeepseekV2Config as DeepseekV2Config,
    DeepseekV2ForCausalLM as DeepseekV2ForCausalLM,
    DeepseekV2Model as DeepseekV2Model,
)
from easydel.models.falcon import (
    FalconConfig as FalconConfig,
    FalconForCausalLM as FalconForCausalLM,
    FalconModel as FalconModel,
)
from easydel.models.gemma import (
    GemmaConfig as GemmaConfig,
    GemmaForCausalLM as GemmaForCausalLM,
    GemmaModel as GemmaModel,
)
from easydel.models.gemma2 import (
    Gemma2Config as Gemma2Config,
    Gemma2ForCausalLM as Gemma2ForCausalLM,
    Gemma2Model as Gemma2Model,
)
from easydel.models.gpt2 import (
    GPT2Config as GPT2Config,
    GPT2LMHeadModel as GPT2LMHeadModel,
    GPT2Model as GPT2Model,
)
from easydel.models.gpt_j import (
    GPTJConfig as GPTJConfig,
    GPTJForCausalLM as GPTJForCausalLM,
    GPTJModel as GPTJModel,
)
from easydel.models.gpt_neo_x import (
    GPTNeoXConfig as GPTNeoXConfig,
    GPTNeoXForCausalLM as GPTNeoXForCausalLM,
    GPTNeoXModel as GPTNeoXModel,
)
from easydel.models.grok_1 import (
    Grok1Config as Grok1Config,
    Grok1ForCausalLM as Grok1ForCausalLM,
    Grok1Model as Grok1Model,
)
from easydel.models.llama import (
    LlamaConfig as LlamaConfig,
    LlamaForCausalLM as LlamaForCausalLM,
    LlamaForSequenceClassification as LlamaForSequenceClassification,
    LlamaModel as LlamaModel,
    VisionLlamaConfig as VisionLlamaConfig,
    VisionLlamaForCausalLM as VisionLlamaForCausalLM,
)
from easydel.models.mamba import (
    MambaConfig as MambaConfig,
    MambaForCausalLM as MambaForCausalLM,
    MambaModel as MambaModel,
)
from easydel.models.mistral import (
    MistralConfig as MistralConfig,
    MistralForCausalLM as MistralForCausalLM,
    MistralModel as MistralModel,
    VisionMistralConfig as VisionMistralConfig,
    VisionMistralForCausalLM as VisionMistralForCausalLM,
)
from easydel.models.mixtral import (
    MixtralConfig as MixtralConfig,
    MixtralForCausalLM as MixtralForCausalLM,
    MixtralModel as MixtralModel,
)
from easydel.models.modelling_utils import (
    BaseNNXModule as BaseNNXModule,
    EDPretrainedConfig as EDPretrainedConfig,
)
from easydel.models.mosaic_mpt import (
    MptAttentionConfig as MptAttentionConfig,
    MptConfig as MptConfig,
    MptForCausalLM as MptForCausalLM,
    MptModel as MptModel,
)
from easydel.models.olmo import (
    OlmoConfig as OlmoConfig,
    OlmoForCausalLM as OlmoForCausalLM,
    OlmoModel as OlmoModel,
)
from easydel.models.openelm import (
    OpenELMConfig as OpenELMConfig,
    OpenELMForCausalLM as OpenELMForCausalLM,
    OpenELMModel as OpenELMModel,
)
from easydel.models.opt import (
    OPTConfig as OPTConfig,
    OPTForCausalLM as OPTForCausalLM,
    OPTModel as OPTModel,
)
from easydel.models.phi import (
    PhiConfig as PhiConfig,
    PhiForCausalLM as PhiForCausalLM,
    PhiModel as PhiModel,
)
from easydel.models.phi3 import (
    Phi3Config as Phi3Config,
    Phi3ForCausalLM as Phi3ForCausalLM,
    Phi3Model as Phi3Model,
)
from easydel.models.qwen1 import (
    Qwen1Config as Qwen1Config,
    Qwen1ForCausalLM as Qwen1ForCausalLM,
    Qwen1ForSequenceClassification as Qwen1ForSequenceClassification,
    Qwen1Model as Qwen1Model,
)
from easydel.models.qwen2 import (
    Qwen2Config as Qwen2Config,
    Qwen2ForCausalLM as Qwen2ForCausalLM,
    Qwen2ForSequenceClassification as Qwen2ForSequenceClassification,
    Qwen2Model as Qwen2Model,
)
from easydel.models.qwen2_moe import (
    Qwen2MoeConfig as Qwen2MoeConfig,
    Qwen2MoeForCausalLM as Qwen2MoeForCausalLM,
    Qwen2MoeModel as Qwen2MoeModel,
)
from easydel.models.stablelm import (
    StableLmConfig as StableLmConfig,
    StableLmForCausalLM as StableLmForCausalLM,
    StableLmModel as StableLmModel,
)

# MODEL IMPORT ENDS HERE

# SMI IMPORT START HERE
from easydel.smi import (
    get_mem as get_mem,
    initialise_tracking as initialise_tracking,
    run as run,
)

# SMI IMPORT ENDS HERE

# TRAINER IMPORT START HERE
from easydel.trainers import (
    BaseTrainer as BaseTrainer,
    CausalLanguageModelTrainer as CausalLanguageModelTrainer,
    CausalLMTrainerOutput as CausalLMTrainerOutput,
    DPOTrainer as DPOTrainer,
    DPOTrainerOutput as DPOTrainerOutput,
    EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
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

# TRAINER IMPORT ENDS HERE

# TRANSFORM IMPORT START HERE
from easydel.transform import (
    easystate_to_huggingface_model as easystate_to_huggingface_model,
    easystate_to_torch as easystate_to_torch,
    torch_dict_to_flatten_dict as torch_dict_to_flatten_dict,
)

# TRANSFORM IMPORT ENDS HERE

# UTILS IMPORT START HERE
from easydel.utils import traversal as traversal

# UTILS IMPORT ENDS HERE

__version__ = "0.1.0"
