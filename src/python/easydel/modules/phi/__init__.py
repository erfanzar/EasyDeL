from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "phi_configuration": ["PhiConfig"],
    "modelling_phi_flax": [
        "FlaxPhiForCausalLM",
        "FlaxPhiForCausalLMModule",
        "FlaxPhiModel",
        "FlaxPhiModule",
    ],
}

if TYPE_CHECKING:
    from .phi_configuration import PhiConfig as PhiConfig
    from .modelling_phi_flax import (
        FlaxPhiForCausalLM as FlaxPhiForCausalLM,
        FlaxPhiForCausalLMModule as FlaxPhiForCausalLMModule,
        FlaxPhiModel as FlaxPhiModel,
        FlaxPhiModule as FlaxPhiModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
