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
    "modelling_vision_mistral_flax": [
        "FlaxVisionMistralForCausalLM"
    ],
    "vision_mistral_configuration": ["VisionMistralConfig"]
}

if TYPE_CHECKING:
    from .mistral_configuration import MistralConfig
    from .modelling_mistral_flax import (
        FlaxMistralForCausalLM,
        FlaxMistralForCausalLMModule,
        FlaxMistralModel,
        FlaxMistralModule,
    )
    from .modelling_vision_mistral_flax import (
        FlaxVisionMistralForCausalLM
    )
    from .vision_mistral_configuration import VisionMistralConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)