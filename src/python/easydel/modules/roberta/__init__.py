from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "roberta_configuration": ["RobertaConfig"],
    "modelling_roberta_flax": [
        "FlaxRobertaForCausalLM",
        "FlaxRobertaForMultipleChoice",
        "FlaxRobertaForMaskedLMModule",
        "FlaxRobertaForQuestionAnswering",
        "FlaxRobertaForSequenceClassification",
        "FlaxRobertaForTokenClassification",
    ],
}

if TYPE_CHECKING:
    from .roberta_configuration import RobertaConfig as RobertaConfig
    from .modelling_roberta_flax import (
        FlaxRobertaForCausalLM as FlaxRobertaForCausalLM,
        FlaxRobertaForMultipleChoice as FlaxRobertaForMultipleChoice,
        FlaxRobertaForMaskedLMModule as FlaxRobertaForMaskedLMModule,
        FlaxRobertaForQuestionAnswering as FlaxRobertaForQuestionAnswering,
        FlaxRobertaForSequenceClassification as FlaxRobertaForSequenceClassification,
        FlaxRobertaForTokenClassification as FlaxRobertaForTokenClassification,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
