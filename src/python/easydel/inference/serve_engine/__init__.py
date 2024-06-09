from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "serve": ["EasyDeLServeEngine", "EasyDeLServeEngineConfig"],
    "client": ["EngineClient"],
}

if TYPE_CHECKING:
    from .serve import (
        EasyDeLServeEngine as EasyDeLServeEngine,
        EasyDeLServeEngineConfig as EasyDeLServeEngineConfig,
    )
    from .client import EngineClient as EngineClient
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
