from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "vision_causal_language_model_trainer": [
        "VisionCausalLanguageModelTrainer",
        "VisionCausalLMTrainerOutput",
    ],
}

if TYPE_CHECKING:
    from .vision_causal_language_model_trainer import (
        VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
        VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
