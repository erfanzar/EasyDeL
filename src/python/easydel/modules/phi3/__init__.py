from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "phi3_configuration": ["Phi3Config"],
    "modelling_phi3_flax": [
        "FlaxPhi3ForCausalLM",
        "FlaxPhi3ForCausalLMModule",
        "FlaxPhi3Model",
        "FlaxPhi3Module"
    ]
}
if TYPE_CHECKING:
    from .phi3_configuration import Phi3Config
    from .modelling_phi3_flax import (
        FlaxPhi3ForCausalLM,
        FlaxPhi3ForCausalLMModule,
        FlaxPhi3Model,
        FlaxPhi3Module
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
