from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {"pipeline": ["GenerationPipeline", "GenerationPipelineConfig"]}

if TYPE_CHECKING:
    from .pipeline import (
        GenerationPipeline as GenerationPipeline,
        GenerationPipelineConfig as GenerationPipelineConfig,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
