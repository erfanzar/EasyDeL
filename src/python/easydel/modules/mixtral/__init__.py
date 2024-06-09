from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "mixtral_configuration": ["MixtralConfig"],
    "modelling_mixtral_flax": [
        "FlaxMixtralForCausalLM",
        "FlaxMixtralForCausalLMModule",
        "FlaxMixtralModel",
        "FlaxMixtralModule",
    ],
}

if TYPE_CHECKING:
    from .mixtral_configuration import MixtralConfig as MixtralConfig
    from .modelling_mixtral_flax import (
        FlaxMixtralForCausalLM as FlaxMixtralForCausalLM,
        FlaxMixtralForCausalLMModule as FlaxMixtralForCausalLMModule,
        FlaxMixtralModel as FlaxMixtralModel,
        FlaxMixtralModule as FlaxMixtralModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
