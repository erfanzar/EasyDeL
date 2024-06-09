from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "causal_language_model_trainer": [
        "CausalLanguageModelTrainer",
        "CausalLMTrainerOutput",
    ],
}

if TYPE_CHECKING:
    from .causal_language_model_trainer import (
        CausalLanguageModelTrainer as CausalLanguageModelTrainer,
        CausalLMTrainerOutput as CausalLMTrainerOutput,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
