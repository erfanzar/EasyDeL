# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-safe regression tests for ``easydel.modules._base`` / ``_drafting`` / Arctic config (no array math)."""

from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType
from easydel.modules._base import (
    AUTO_MODEL_FACTORY_REGISTRY,
    BaseConditionalGenerationModule,
    BaseTokenClassificationModule,
    EncoderDecoderProtocol,
    VisionLanguageProtocol,
    VisionModelProtocol,
    create_question_answering_class,
    create_token_classification_class,
)
from easydel.modules._drafting import validate_target_layer_ids

pytestmark = pytest.mark.cpu_ok


def test_embedding_layer_rejected_when_not_allowed():
    with pytest.raises(ValueError, match="only valid for DSpark/DFlash"):
        validate_target_layer_ids([-1, 0], allow_embedding=False)


def test_embedding_layer_accepted_when_allowed():
    assert validate_target_layer_ids([-1, 2, 5], allow_embedding=True) == (-1, 2, 5)


def test_layer_below_embedding_is_invalid():
    with pytest.raises(ValueError, match="Invalid target layer id"):
        validate_target_layer_ids([-2], allow_embedding=True)


def test_token_classification_and_qa_factories_preserve_task_wrappers():
    class _Cfg:
        pass

    class _Model:
        pass

    tok = create_token_classification_class("CpuContractTok", _Model, _Cfg, model_type="cpu_contract_tok")
    qa = create_question_answering_class("CpuContractQa", _Model, _Cfg, model_type="cpu_contract_qa")
    assert tok.__name__ == "CpuContractTokForTokenClassification"
    assert qa.__name__ == "CpuContractQaForQuestionAnswering"
    assert issubclass(tok, BaseTokenClassificationModule)
    # Registry still contains the task types TaskType really defines.
    assert AUTO_MODEL_FACTORY_REGISTRY[TaskType.CAUSAL_LM] is not None


def test_generated_class_can_be_subclassed_without_recursion(monkeypatch):
    calls = []

    def _fake_base_init(self, *args, **kwargs):
        calls.append(kwargs["base_model_name"])

    monkeypatch.setattr(BaseTokenClassificationModule, "__init__", _fake_base_init)

    class _Cfg:
        pass

    class _Model:
        pass

    generated = create_token_classification_class("CpuContractSub", _Model, _Cfg, model_type="cpu_contract_sub")
    sub = type("SubOfGenerated", (generated,), {})
    instance = sub.__new__(sub)
    generated.__init__(instance, SimpleNamespace(), rngs=None)
    sub.__init__(instance, SimpleNamespace(), rngs=None)
    assert calls == ["model", "model"]


def test_conditional_generation_accepts_router_aux_loss_coef():
    import inspect

    sig = inspect.signature(BaseConditionalGenerationModule.__init__)
    assert "router_aux_loss_coef" in sig.parameters
    assert sig.parameters["router_aux_loss_coef"].default is None


def test_arctic_enable_expert_tensor_parallelism_sets_expert_tensor_mode():
    from easydel.modules.arctic import ArcticConfig

    on = ArcticConfig(enable_expert_tensor_parallelism=True)
    off = ArcticConfig(enable_expert_tensor_parallelism=False)
    assert on.use_expert_tensor_mode is True
    assert off.use_expert_tensor_mode is False


@pytest.mark.parametrize("protocol", [EncoderDecoderProtocol, VisionModelProtocol, VisionLanguageProtocol])
def test_public_protocols_support_structural_runtime_checks(protocol):
    class StructuralModel:
        config = None
        dtype = "float32"
        param_dtype = "float32"
        precision = None

        def __call__(self, **kwargs):
            return kwargs

        def get_embedding(self):
            return "embedding"

        def get_decoder(self):
            return "decoder"

        def get_encoder(self):
            return "encoder"

        def get_vision_tower(self):
            return "vision"

        def get_language_model(self):
            return "language"

    assert isinstance(StructuralModel(), protocol)
    assert not isinstance(object(), protocol)
