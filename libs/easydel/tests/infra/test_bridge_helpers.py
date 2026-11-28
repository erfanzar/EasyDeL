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


@pytest.mark.cpu_ok
def test_qwen35_causallm_pre_remaps_vlm_nested_keys_before_fused_retp():
    from types import SimpleNamespace

    import numpy as np
    from easydel.layers.layouts import retp_fused_state
    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    expected_key = ("parameters", "model", "layers", "0", "mlp", "gate_up_proj", "weight")
    nested_key = ("parameters", "model", "language_model", "layers", "0", "mlp", "gate_up_proj", "weight")
    visual_key = ("parameters", "model", "visual", "patch_embed", "weight")

    # Canonical tp=1 weight: [gate_0..7 | up_0..7] along last dim (size 16 = 8 + 8).
    canonical = np.arange(32, dtype=np.float32).reshape(2, 16)
    state = {nested_key: canonical.copy(), visual_key: np.zeros((2, 2), dtype=np.float32)}
    expected_leaves = {expected_key: object()}

    dummy = SimpleNamespace(fused_params={"model.layers.0.mlp.gate_up_proj": (8, 8)})
    remapped = Qwen3_5ForCausalLM._remap_legacy_native_keys(dummy, state, expected_leaves)
    assert expected_key in remapped
    assert nested_key not in remapped
    assert visual_key in remapped

    # After pre-remapping, retp_fused_state(tp=1 -> tp=4) must interleave each TP rank's [gate_r, up_r].
    interleaved = retp_fused_state(dummy, remapped, saved_tp=1, target_tp=4)
    got = np.asarray(interleaved[expected_key])
    # Rank 0 gets gate[0:2], up[0:2]; rank 1 gets gate[2:4], up[2:4]; etc.
    expected_order = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    np.testing.assert_array_equal(got[0], canonical[0, expected_order])
    np.testing.assert_array_equal(got[1], canonical[1, expected_order])
