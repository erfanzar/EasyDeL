from ..utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {"data_processor": ["DataProcessorArguments", "DataProcessor"]}
if TYPE_CHECKING:
    from .data_processor import (
        DataProcessorArguments as DataProcessorArguments,
        DataProcessor as DataProcessor,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
