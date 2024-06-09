from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "whisper_configuration": ["WhisperConfig"],
    "modelling_whisper_flax": [
        "FlaxWhisperForConditionalGeneration",
        "FlaxWhisperForAudioClassification",
        "FlaxWhisperTimeStampLogitsProcessor",
    ],
}

if TYPE_CHECKING:
    from .whisper_configuration import WhisperConfig as WhisperConfig
    from .modelling_whisper_flax import (
        FlaxWhisperForConditionalGeneration as FlaxWhisperForConditionalGeneration,
        FlaxWhisperForAudioClassification as FlaxWhisperForAudioClassification,
        FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
