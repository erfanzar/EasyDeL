from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "palm_configuration": ["PalmConfig"],
    "modelling_palm_flax": [
        "FlaxPalmForCausalLM",
        "FlaxPalmForCausalLMModule",
        "FlaxPalmModel",
        "FlaxPalmModule"
    ],
}

if TYPE_CHECKING:
    from .palm_configuration import PalmConfig
    from .modelling_palm_flax import (
        FlaxPalmForCausalLM,
        FlaxPalmForCausalLMModule,
        FlaxPalmModel,
        FlaxPalmModule
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)