# ChatGLM is not fully supported by EasyDeL and it's added only as an available model. and Block


from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "chatglm_configuration": ["ChatGLMConfig"],
    "modelling_chatglm_flax": ["FlaxChatGLMModel", "FlaxChatGLMTransformer"],
}

if TYPE_CHECKING:
    from .chatglm_configuration import ChatGLMConfig as ChatGLMConfig
    from .modelling_chatglm_flax import (
        FlaxChatGLMModel as FlaxChatGLMModel,
        FlaxChatGLMTransformer as FlaxChatGLMTransformer,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
