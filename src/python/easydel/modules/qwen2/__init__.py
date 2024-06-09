from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "qwen_configuration": ["Qwen2Config"],
    "modelling_qwen_flax": [
        "FlaxQwen2ForCausalLM",
        "FlaxQwen2ForSequenceClassification",
        "FlaxQwen2Model",
    ],
}

if TYPE_CHECKING:
    from .qwen_configuration import Qwen2Config as Qwen2Config
    from .modelling_qwen_flax import (
        FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLM,
        FlaxQwen2ForSequenceClassification as FlaxQwen2ForSequenceClassification,
        FlaxQwen2Model as FlaxQwen2Model,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
