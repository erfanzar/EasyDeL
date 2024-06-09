from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "mistral_configuration": ["MistralConfig"],
    "modelling_mistral_flax": [
        "FlaxMistralForCausalLM",
        "FlaxMistralForCausalLMModule",
        "FlaxMistralModel",
        "FlaxMistralModule",
    ],
    "modelling_vision_mistral_flax": ["FlaxVisionMistralForCausalLM"],
    "vision_mistral_configuration": ["VisionMistralConfig"],
}

if TYPE_CHECKING:
    from .mistral_configuration import MistralConfig as MistralConfig
    from .modelling_mistral_flax import (
        FlaxMistralForCausalLM as FlaxMistralForCausalLM,
        FlaxMistralForCausalLMModule as FlaxMistralForCausalLMModule,
        FlaxMistralModel as FlaxMistralModel,
        FlaxMistralModule as FlaxMistralModule,
    )
    from .modelling_vision_mistral_flax import (
        FlaxVisionMistralForCausalLM as FlaxVisionMistralForCausalLM,
    )
    from .vision_mistral_configuration import VisionMistralConfig as VisionMistralConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
