from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "modelling_output": ["VisionCausalLMTrainerOutput"],
    "vision_causal_language_model_trainer": ["VisionCausalLanguageModelTrainer"],
}

if TYPE_CHECKING:
    from .modelling_output import (
        VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    )
    from .vision_causal_language_model_trainer import (
        VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
