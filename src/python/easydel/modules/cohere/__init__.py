from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "cohere_configuration": ["CohereConfig"],
    "modelling_cohere_flax": [
        "FlaxCohereModel",
        "FlaxCohereForCausalLM",
    ],
}

if TYPE_CHECKING:
    from .cohere_configuration import CohereConfig
    from .modelling_cohere_flax import (
        FlaxCohereModel,
        FlaxCohereForCausalLM
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
