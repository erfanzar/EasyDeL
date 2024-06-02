from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "grok_1_configuration": ["Grok1Config"],
    "modelling_grok_1_flax": [
        "FlaxGrok1ForCausalLM",
        "FlaxGrok1Model",
    ],
}

if TYPE_CHECKING:
    from .grok_1_configuration import Grok1Config
    from .modelling_grok_1_flax import (
        FlaxGrok1ForCausalLM,
        FlaxGrok1Model,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
