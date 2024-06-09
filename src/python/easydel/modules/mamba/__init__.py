from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "mamba_configuration": ["MambaConfig"],
    "modelling_mamba_flax": [
        "FlaxMambaModule",
        "FlaxMambaCache",
        "FlaxMambaForCausalLMModule",
        "FlaxMambaForCausalLM",
        "FlaxMambaModel",
    ],
}

if TYPE_CHECKING:
    from .mamba_configuration import MambaConfig as MambaConfig
    from .modelling_mamba_flax import (
        FlaxMambaModule as FlaxMambaModule,
        FlaxMambaCache as FlaxMambaCache,
        FlaxMambaForCausalLMModule as FlaxMambaForCausalLMModule,
        FlaxMambaForCausalLM as FlaxMambaForCausalLM,
        FlaxMambaModel as FlaxMambaModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
