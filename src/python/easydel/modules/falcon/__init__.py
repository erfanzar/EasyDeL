from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "falcon_configuration": ["FalconConfig"],
    "modelling_falcon_flax": [
        "FlaxFalconForCausalLM",
        "FlaxFalconForCausalLMModule",
        "FlaxFalconModel",
        "FlaxFalconModule",
    ],
}

if TYPE_CHECKING:
    from .falcon_configuration import FalconConfig as FalconConfig
    from .modelling_falcon_flax import (
        FlaxFalconForCausalLM as FlaxFalconForCausalLM,
        FlaxFalconForCausalLMModule as FlaxFalconForCausalLMModule,
        FlaxFalconModel as FlaxFalconModel,
        FlaxFalconModule as FlaxFalconModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
