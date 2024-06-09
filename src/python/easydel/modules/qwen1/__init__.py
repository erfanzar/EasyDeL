from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "qwen1_configuration": ["Qwen1Config"],
    "modelling_qwen1_flax": [
        "FlaxQwen1ForCausalLM",
        "FlaxQwen1Model",
        "FlaxQwen1ForSequenceClassification",
    ],
}

if TYPE_CHECKING:
    from .qwen1_configuration import Qwen1Config as Qwen1Config
    from .modelling_qwen1_flax import (
        FlaxQwen1ForCausalLM as FlaxQwen1ForCausalLM,
        FlaxQwen1Model as FlaxQwen1Model,
        FlaxQwen1ForSequenceClassification as FlaxQwen1ForSequenceClassification,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
