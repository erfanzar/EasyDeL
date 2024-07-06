# import os as _os

# if bool(
#     _os.environ.get("EASYDEL_AUTO", "true")
# ):  # Taking care of some optional GPU FLAGs
#     _os.environ["XLA_FLAGS"] = (
#         _os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
#     )
#     _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#     _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
#     _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# # INFERENCE IMPORT START HERE

# from easydel.etils.easystate import EasyDeLState as EasyDeLState
# from easydel.etils.errors import (
#     EasyDeLRuntimeError as EasyDeLRuntimeError,
# )
# from easydel.etils.errors import (
#     EasyDeLSyntaxRuntimeError as EasyDeLSyntaxRuntimeError,
# )

# # TRANSFORM IMPORT ENDS HERE
# # ETILS IMPORT START HERE
# from easydel.etils.errors import (
#     EasyDeLTimerError as EasyDeLTimerError,
# )
# from easydel.etils.etils import (
#     EasyDeLGradientCheckPointers as EasyDeLGradientCheckPointers,
# )
# from easydel.etils.etils import (
#     EasyDeLOptimizers as EasyDeLOptimizers,
# )
# from easydel.etils.etils import (
#     EasyDeLSchedulers as EasyDeLSchedulers,
# )
# from easydel.etils.partition_module import PartitionAxis as PartitionAxis
# from easydel.inference.generation_pipeline import (
#     ChatPipeline as ChatPipeline,
# )
# from easydel.inference.generation_pipeline import (
#     GenerationPipeline as GenerationPipeline,
# )
# from easydel.inference.generation_pipeline import (
#     GenerationPipelineConfig as GenerationPipelineConfig,
# )
# from easydel.inference.serve_engine import (
#     ApiEngine as ApiEngine,
# )
# from easydel.inference.serve_engine import (
#     engine_client as engine_client,
# )
# from easydel.modules.arctic import (
#     ArcticConfig as ArcticConfig,
# )
# from easydel.modules.arctic import (
#     FlaxArcticForCausalLM as FlaxArcticForCausalLM,
# )
# from easydel.modules.arctic import (
#     FlaxArcticModel as FlaxArcticModel,
# )
# from easydel.modules.attention_module import (
#     AttentionMechanisms as AttentionMechanisms,
# )
# from easydel.modules.attention_module import (
#     FlexibleAttentionModule as FlexibleAttentionModule,
# )
# from easydel.modules.auto_easydel_model import (
#     AutoEasyDeLConfig as AutoEasyDeLConfig,
# )
# from easydel.modules.auto_easydel_model import (
#     AutoEasyDeLModelForCausalLM as AutoEasyDeLModelForCausalLM,
# )
# from easydel.modules.auto_easydel_model import (
#     AutoShardAndGatherFunctions as AutoShardAndGatherFunctions,
# )
# from easydel.modules.auto_easydel_model import (
#     get_modules_by_type as get_modules_by_type,
# )
# from easydel.modules.cohere import (
#     CohereConfig as CohereConfig,
# )
# from easydel.modules.cohere import (
#     FlaxCohereForCausalLM as FlaxCohereForCausalLM,
# )
# from easydel.modules.cohere import (
#     FlaxCohereModel as FlaxCohereModel,
# )
# from easydel.modules.dbrx import (
#     DbrxAttentionConfig as DbrxAttentionConfig,
# )
# from easydel.modules.dbrx import (
#     DbrxConfig as DbrxConfig,
# )
# from easydel.modules.dbrx import (
#     DbrxFFNConfig as DbrxFFNConfig,
# )
# from easydel.modules.dbrx import (
#     FlaxDbrxForCausalLM as FlaxDbrxForCausalLM,
# )
# from easydel.modules.dbrx import (
#     FlaxDbrxModel as FlaxDbrxModel,
# )
# from easydel.modules.deepseek_v2 import (
#     DeepseekV2Config as DeepseekV2Config,
# )
# from easydel.modules.deepseek_v2 import (
#     FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
# )
# from easydel.modules.deepseek_v2 import (
#     FlaxDeepseekV2Model as FlaxDeepseekV2Model,
# )
# from easydel.modules.easydel_modelling_utils import (
#     EasyDeLFlaxPretrainedModel as EasyDeLFlaxPretrainedModel,
# )
# from easydel.modules.easydel_modelling_utils import (
#     EasyDeLPretrainedConfig as EasyDeLPretrainedConfig,
# )
# from easydel.modules.falcon import (
#     FalconConfig as FalconConfig,
# )
# from easydel.modules.falcon import (
#     FlaxFalconForCausalLM as FlaxFalconForCausalLM,
# )
# from easydel.modules.falcon import (
#     FlaxFalconModel as FlaxFalconModel,
# )
# from easydel.modules.gemma import (
#     FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
# )
# from easydel.modules.gemma import (
#     FlaxGemmaModel as FlaxGemmaModel,
# )
# from easydel.modules.gemma import (
#     GemmaConfig as GemmaConfig,
# )
# from easydel.modules.gemma2 import (
#     FlaxGemma2ForCausalLM as FlaxGemma2ForCausalLM,
# )
# from easydel.modules.gemma2 import (
#     FlaxGemma2Model as FlaxGemma2Model,
# )
# from easydel.modules.gemma2 import (
#     Gemma2Config as Gemma2Config,
# )
# from easydel.modules.gpt2 import (
#     FlaxGPT2LMHeadModel as FlaxGPT2LMHeadModel,
# )
# from easydel.modules.gpt2 import (
#     FlaxGPT2Model as FlaxGPT2Model,
# )
# from easydel.modules.gpt2 import (
#     GPT2Config as GPT2Config,
# )
# from easydel.modules.gpt_j import (
#     FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
# )
# from easydel.modules.gpt_j import (
#     FlaxGPTJModel as FlaxGPTJModel,
# )
# from easydel.modules.gpt_j import (
#     GPTJConfig as GPTJConfig,
# )
# from easydel.modules.gpt_neo_x import (
#     FlaxGPTNeoXForCausalLM as FlaxGPTNeoXForCausalLM,
# )
# from easydel.modules.gpt_neo_x import (
#     FlaxGPTNeoXModel as FlaxGPTNeoXModel,
# )
# from easydel.modules.gpt_neo_x import (
#     GPTNeoXConfig as GPTNeoXConfig,
# )
# from easydel.modules.grok_1 import (
#     FlaxGrok1ForCausalLM as FlaxGrok1ForCausalLM,
# )
# from easydel.modules.grok_1 import (
#     FlaxGrok1Model as FlaxGrok1Model,
# )
# from easydel.modules.grok_1 import (
#     Grok1Config as Grok1Config,
# )
# from easydel.modules.llama import (
#     FlaxLlamaForCausalLM as FlaxLlamaForCausalLM,
# )
# from easydel.modules.llama import (
#     FlaxLlamaForSequenceClassification as FlaxLlamaForSequenceClassification,
# )

# # INFERENCE IMPORT END HERE
# # MODULES IMPORT START HERE
# from easydel.modules.llama import (
#     FlaxLlamaModel as FlaxLlamaModel,
# )
# from easydel.modules.llama import (
#     FlaxVisionLlamaForCausalLM as FlaxVisionLlamaForCausalLM,
# )
# from easydel.modules.llama import (
#     LlamaConfig as LlamaConfig,
# )
# from easydel.modules.llama import (
#     VisionLlamaConfig as VisionLlamaConfig,
# )
# from easydel.modules.lucid_transformer import (
#     FlaxLTConfig as FlaxLTConfig,
# )
# from easydel.modules.lucid_transformer import (
#     FlaxLTForCausalLM as FlaxLTForCausalLM,
# )
# from easydel.modules.lucid_transformer import (
#     FlaxLTModel as FlaxLTModel,
# )
# from easydel.modules.mamba import (
#     FlaxMambaForCausalLM as FlaxMambaForCausalLM,
# )
# from easydel.modules.mamba import (
#     FlaxMambaModel as FlaxMambaModel,
# )
# from easydel.modules.mamba import (
#     MambaConfig as MambaConfig,
# )
# from easydel.modules.mistral import (
#     FlaxMistralForCausalLM as FlaxMistralForCausalLM,
# )
# from easydel.modules.mistral import (
#     FlaxMistralModel as FlaxMistralModel,
# )
# from easydel.modules.mistral import (
#     FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
# )
# from easydel.modules.mistral import (
#     MistralConfig as MistralConfig,
# )
# from easydel.modules.mistral import (
#     VisionMistralConfig as VisionMistralConfig,
# )
# from easydel.modules.mixtral import (
#     FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
# )
# from easydel.modules.mixtral import (
#     FlaxMixtralModel as FlaxMixtralModel,
# )
# from easydel.modules.mixtral import (
#     MixtralConfig as MixtralConfig,
# )
# from easydel.modules.mosaic_mpt import (
#     FlaxMptForCausalLM as FlaxMptForCausalLM,
# )
# from easydel.modules.mosaic_mpt import (
#     FlaxMptModel as FlaxMptModel,
# )
# from easydel.modules.mosaic_mpt import (
#     MptAttentionConfig as MptAttentionConfig,
# )
# from easydel.modules.mosaic_mpt import (
#     MptConfig as MptConfig,
# )
# from easydel.modules.olmo import (
#     FlaxOlmoForCausalLM as FlaxOlmoForCausalLM,
# )
# from easydel.modules.olmo import (
#     FlaxOlmoModel as FlaxOlmoModel,
# )
# from easydel.modules.olmo import (
#     OlmoConfig as OlmoConfig,
# )
# from easydel.modules.openelm import (
#     FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
# )
# from easydel.modules.openelm import (
#     FlaxOpenELMModel as FlaxOpenELMModel,
# )
# from easydel.modules.openelm import (
#     OpenELMConfig as OpenELMConfig,
# )
# from easydel.modules.opt import (
#     FlaxOPTForCausalLM as FlaxOPTForCausalLM,
# )
# from easydel.modules.opt import (
#     FlaxOPTModel as FlaxOPTModel,
# )
# from easydel.modules.opt import (
#     OPTConfig as OPTConfig,
# )
# from easydel.modules.palm import (
#     FlaxPalmForCausalLM as FlaxPalmForCausalLM,
# )
# from easydel.modules.palm import (
#     FlaxPalmModel as FlaxPalmModel,
# )
# from easydel.modules.palm import (
#     PalmConfig as PalmConfig,
# )
# from easydel.modules.phi import (
#     FlaxPhiForCausalLM as FlaxPhiForCausalLM,
# )
# from easydel.modules.phi import (
#     FlaxPhiModel as FlaxPhiModel,
# )
# from easydel.modules.phi import (
#     PhiConfig as PhiConfig,
# )
# from easydel.modules.phi3 import (
#     FlaxPhi3ForCausalLM as FlaxPhi3ForCausalLM,
# )
# from easydel.modules.phi3 import (
#     FlaxPhi3Model as FlaxPhi3Model,
# )
# from easydel.modules.phi3 import (
#     Phi3Config as Phi3Config,
# )
# from easydel.modules.qwen1 import (
#     FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
# )
# from easydel.modules.qwen1 import (
#     FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
# )
# from easydel.modules.qwen1 import (
#     FlaxQwen1Model as FlaxQwen1Model,
# )
# from easydel.modules.qwen1 import (
#     Qwen1Config as Qwen1Config,
# )
# from easydel.modules.qwen2 import (
#     FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
# )
# from easydel.modules.qwen2 import (
#     FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
# )
# from easydel.modules.qwen2 import (
#     FlaxQwen2Model as FlaxQwen2Model,
# )
# from easydel.modules.qwen2 import (
#     Qwen2Config as Qwen2Config,
# )
# from easydel.modules.qwen2_moe import (
#     FlaxQwen2MoeForCausalLM as FlaxQwen2MoeForCausalLM,
# )
# from easydel.modules.qwen2_moe import (
#     FlaxQwen2MoeModel as FlaxQwen2MoeModel,
# )
# from easydel.modules.qwen2_moe import (
#     Qwen2MoeConfig as Qwen2MoeConfig,
# )
# from easydel.modules.stablelm import (
#     FlaxStableLmForCausalLM as FlaxStableLmForCausalLM,
# )
# from easydel.modules.stablelm import (
#     FlaxStableLmModel as FlaxStableLmModel,
# )
# from easydel.modules.stablelm import (
#     StableLmConfig as StableLmConfig,
# )
# from easydel.modules.t5 import (
#     FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
# )
# from easydel.modules.t5 import (
#     FlaxT5Model as FlaxT5Model,
# )
# from easydel.modules.t5 import (
#     T5Config as T5Config,
# )
# from easydel.modules.whisper import (
#     FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
# )
# from easydel.modules.whisper import (
#     FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
# )
# from easydel.modules.whisper import (
#     FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
# )
# from easydel.modules.whisper import (
#     WhisperConfig as WhisperConfig,
# )
# from easydel.smi import (
#     get_mem as get_mem,
# )
# from easydel.smi import (
#     initialise_tracking as initialise_tracking,
# )

# # UTILS IMPORT ENDS HERE
# # SMI IMPORT START HERE
# from easydel.smi import (
#     run as run,
# )
# from easydel.trainers import (
#     BaseTrainer as BaseTrainer,
# )
# from easydel.trainers import (
#     CausalLanguageModelTrainer as CausalLanguageModelTrainer,
# )
# from easydel.trainers import (
#     CausalLMTrainerOutput as CausalLMTrainerOutput,
# )
# from easydel.trainers import (
#     DPOTrainer as DPOTrainer,
# )
# from easydel.trainers import (
#     DPOTrainerOutput as DPOTrainerOutput,
# )
# from easydel.trainers import (
#     EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
# )
# from easydel.trainers import (
#     JaxDistributedConfig as JaxDistributedConfig,
# )
# from easydel.trainers import (
#     ORPOTrainer as ORPOTrainer,
# )
# from easydel.trainers import (
#     ORPOTrainerOutput as ORPOTrainerOutput,
# )
# from easydel.trainers import (
#     SFTTrainer as SFTTrainer,
# )
# from easydel.trainers import (
#     TrainArguments as TrainArguments,
# )

# # MODULES IMPORT END HERE
# # TRAINER IMPORT START HERE
# from easydel.trainers import (
#     VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
# )
# from easydel.trainers import (
#     VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
# )
# from easydel.trainers import (
#     conversations_formatting_function as conversations_formatting_function,
# )
# from easydel.trainers import (
#     create_constant_length_dataset as create_constant_length_dataset,
# )
# from easydel.trainers import (
#     get_formatting_func_from_dataset as get_formatting_func_from_dataset,
# )
# from easydel.trainers import (
#     instructions_formatting_function as instructions_formatting_function,
# )
# from easydel.transform import (
#     easystate_to_huggingface_model as easystate_to_huggingface_model,
# )
# from easydel.transform import (
#     easystate_to_torch as easystate_to_torch,
# )

# # SMI IMPORT ENDS HERE
# # TRANSFORM IMPORT START HERE
# from easydel.transform import (
#     toch_dict_to_flatten_dict as toch_dict_to_flatten_dict,
# )
# from easydel.utils.helpers import RNG as RNG

# # TRAINER IMPORT ENDS HERE
# # UTILS IMPORT START HERE
# from easydel.utils.helpers import get_mesh as get_mesh

# # ETILS IMPORT ENDS HERE
# __version__ = "0.0.69"
