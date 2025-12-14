"""FalconH1 modules for EasyDeL.

This package provides an EasyDeL-native JAX/Flax implementation of the
HuggingFace FalconH1 architecture.

HuggingFace references:
    - https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/configuration_falcon_h1.py
    - https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/modeling_falcon_h1.py
"""

from .falcon_h1_configuration import FalconH1Config
from .modeling_falcon_h1 import FalconH1ForCausalLM, FalconH1Model

__all__ = ("FalconH1Config", "FalconH1ForCausalLM", "FalconH1Model")
