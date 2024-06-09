from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "gpt_j_configuration": ["GPTJConfig"],
    "modelling_gpt_j_flax": [
        "FlaxGPTJForCausalLM",
        "FlaxGPTJForCausalLMModule",
        "FlaxGPTJModel",
        "FlaxGPTJModule",
    ],
}

if TYPE_CHECKING:
    from .gpt_j_configuration import GPTJConfig as GPTJConfig
    from .modelling_gpt_j_flax import (
        FlaxGPTJForCausalLM as FlaxGPTJForCausalLM,
        FlaxGPTJForCausalLMModule as FlaxGPTJForCausalLMModule,
        FlaxGPTJModel as FlaxGPTJModel,
        FlaxGPTJModule as FlaxGPTJModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
