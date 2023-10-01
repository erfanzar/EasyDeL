from .utils import make_shard_and_gather_fns, get_mesh
from .modules import FlaxLlamaModel, FlaxLlamaForCausalLM, LlamaConfig, \
    FlaxLTModelModule, FlaxLTConfig, FlaxLTForCausalLM, FlaxLTModel, GPTJConfig, FlaxGPTJModule, \
    FlaxGPTJForCausalLMModule, FlaxGPTJModel, FlaxGPTJForCausalLM, FlaxMptForCausalLM, MptConfig, FlaxMptModel, \
    FlaxFalconForCausalLM, FlaxFalconModel, FalconConfig, FlaxGPTNeoXForCausalLM, GPTNeoXConfig, FlaxGPTNeoXModel, \
    FlaxT5ForConditionalGeneration, FlaxT5Model, FlaxPalmForCausalLM, PalmModel, PalmConfig, T5Config, \
    FlaxOPTForCausalLM, FlaxOPTModel, OPTConfig
from .trainer import TrainArguments, fsdp_train_step, get_training_modules, CausalLMTrainer

try:
    from .serve import JAXServer, PyTorchServer, JaxServerConfig
except ValueError as vr:
    print(f"\033[1;31mWarning\033[1;0m : JAXServer Wont be Imported Be Cause {vr}")
    JAXServer = None
__version__ = "0.0.30"

# import os
#
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet
__all__ = "TrainArguments", "fsdp_train_step", "get_training_modules", \
    "FlaxLlamaForCausalLM", \
    "FlaxLlamaModel", "FlaxGPTJModule", "FlaxGPTJForCausalLMModule", \
    "FlaxGPTJModel", "FlaxGPTJForCausalLM", "GPTJConfig", \
    "FlaxLTModel", "FlaxLTConfig", "FlaxLTModelModule", "FlaxLTForCausalLM", \
    "FlaxMptForCausalLM", "MptConfig", "FlaxMptModel", \
    "FlaxFalconForCausalLM", "FlaxFalconModel", "FalconConfig", \
    "FlaxGPTNeoXForCausalLM", "GPTNeoXConfig", "FlaxGPTNeoXModel", \
    "FlaxT5ForConditionalGeneration", "FlaxT5Model", \
    "FlaxPalmForCausalLM", "PalmModel", "PalmConfig", "T5Config", \
    "FlaxOPTForCausalLM", "FlaxOPTModel", "OPTConfig", "CausalLMTrainer", "LlamaConfig", "__version__", "JAXServer", \
    "get_mesh", "PyTorchServer", "JaxServerConfig"
