from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "lt_configuration": ["FlaxLTConfig"],
    "modelling_lt_flax": [
        "FlaxLTForCausalLM",
        "FlaxLTModel",
        "FlaxLTModule",
    ],
}

if TYPE_CHECKING:
    from .lt_configuration import FlaxLTConfig
    from .modelling_lt_flax import (
        FlaxLTForCausalLM,
        FlaxLTModel,
        FlaxLTModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)