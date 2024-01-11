from .serve.torch_serve import (
    PyTorchServer as PyTorchServer,
    PytorchServerConfig as PytorchServerConfig
)
from .serve.jax_serve import (
    JAXServer as JAXServer,
    JAXServerConfig as JAXServerConfig
)
from .serve.gradio_user_interface_base import (
    GradioUserInference as GradioUserInference
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

from .modules.auto_easydel_model import (
    AutoEasyDelModelForCausalLM as AutoEasyDelModelForCausalLM,
    AutoEasyDelConfig as AutoEasyDelConfig,
    get_modules_by_type as get_modules_by_type
)

from .utils.utils import (
    get_mesh as get_mesh,
    names_in_mesh as names_in_mesh,
    get_names_from_partition_spec as get_names_from_partition_spec,
    make_shard_and_gather_fns as make_shard_and_gather_fns,
    with_sharding_constraint as with_sharding_constraint,
    RNG as RNG
)

from .trainer import (
    CausalLanguageModelTrainer,
    TrainArguments,
    create_casual_language_model_evaluation_step,
    create_casual_language_model_train_step,
)

from .linen import (
    from_8bit as from_8bit,
    Dense8Bit as Dense8Bit,
    array_from_8bit as array_from_8bit,
    array_to_bit8 as array_to_bit8,
    to_8bit as to_8bit
)
from .smi import (
    run as run,
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

__version__ = "0.0.42"
