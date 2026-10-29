"""Public HF task dispatch and legacy filename compatibility contracts."""

import pytest
import transformers
from easydel.infra.factory import TaskType
from easydel.infra.mixins import bridge


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "task,loader_name",
    [
        (TaskType.TOKEN_CLASSIFICATION, "AutoModelForTokenClassification"),
        (TaskType.QUESTION_ANSWERING, "AutoModelForQuestionAnswering"),
    ],
)
def test_public_task_dispatch_uses_standard_hf_loader(task, loader_name):
    class Model(bridge.EasyBridgeMixin):
        _model_task = task

    assert Model.get_torch_loader() is getattr(transformers, loader_name)


@pytest.mark.cpu_ok
def test_legacy_filename_constants_come_from_base_config():
    from easydel.infra import base_config

    for name in (
        "TF2_WEIGHTS_NAME",
        "TF2_WEIGHTS_INDEX_NAME",
        "TF_WEIGHTS_NAME",
        "MODEL_CARD_NAME",
        "FEATURE_EXTRACTOR_NAME",
        "IMAGE_PROCESSOR_NAME",
        "PROCESSOR_NAME",
        "CHAT_TEMPLATE_NAME",
    ):
        assert getattr(bridge, name) == getattr(base_config, name)
