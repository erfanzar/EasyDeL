from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "opt_configuration": ["OPTConfig"],
    "modelling_opt_flax": [
        "FlaxOPTForCausalLM",
        "FlaxOPTForCausalLMModule",
        "FlaxOPTModel",
        "FlaxOPTModule"
    ],
}

if TYPE_CHECKING:
    from .opt_configuration import OPTConfig
    from .modelling_opt_flax import (
        FlaxOPTForCausalLM,
        FlaxOPTForCausalLMModule,
        FlaxOPTModel,
        FlaxOPTModule
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
