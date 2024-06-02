from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "llama_configuration": ["LlamaConfig"],
    "modelling_llama_flax": [
        "FlaxLlamaForSequenceClassification",
        "FlaxLlamaForSequenceClassificationModule",
        "FlaxLlamaForCausalLM",
        "FlaxLlamaForCausalLMModule",
        "FlaxLlamaModel",
        "FlaxLlamaModule"
    ],
    "modelling_vision_llama_flax": [
        "FlaxVisionLlamaForCausalLM"
    ],
    "vision_llama_configuration": ["VisionLlamaConfig"]
}

if TYPE_CHECKING:
    from .llama_configuration import LlamaConfig
    from .modelling_llama_flax import (
        FlaxLlamaForSequenceClassification,
        FlaxLlamaForSequenceClassificationModule,
        FlaxLlamaForCausalLM,
        FlaxLlamaForCausalLMModule,
        FlaxLlamaModel,
        FlaxLlamaModule
    )
    from .modelling_vision_llama_flax import FlaxVisionLlamaForCausalLM
    from .vision_llama_configuration import VisionLlamaConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
