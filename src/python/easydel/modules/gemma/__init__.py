from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "gemma_configuration": ["GemmaConfig"],
    "modelling_gemma_flax": ["FlaxGemmaForCausalLM", "FlaxGemmaModel"],
}

if TYPE_CHECKING:
    from .gemma_configuration import GemmaConfig as GemmaConfig
    from .modelling_gemma_flax import (
        FlaxGemmaForCausalLM as FlaxGemmaForCausalLM,
        FlaxGemmaModel as FlaxGemmaModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
