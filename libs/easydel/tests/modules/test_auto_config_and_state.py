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

"""Config / auto-class regressions (no array computation)."""

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.modules.auto import auto_configuration
from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig, normalize_task
from easydel.modules.auto.auto_modeling import (
    AutoEasyDeLAnyToAnyModel,
    AutoStateAnyToAnyModel,
)
from easydel.modules.cohere2.cohere2_configuration import Cohere2Config
from easydel.modules.dbrx.dbrx_configuration import DbrxFFNConfig


def _tiny_cohere2() -> Cohere2Config:
    return Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        sliding_window=8,
        sliding_window_pattern=2,
        max_position_embeddings=64,
    )


@pytest.mark.cpu_ok
def test_cohere2_config_serializes_its_own_model_type():
    assert Cohere2Config().to_dict()["model_type"] == "cohere2"


@pytest.mark.cpu_ok
def test_cohere2_config_roundtrips_through_auto_config(tmp_path):
    config = _tiny_cohere2()
    config.save_pretrained(tmp_path)

    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path))

    assert isinstance(loaded, Cohere2Config)
    assert loaded.layer_types == config.layer_types
    assert "sliding_attention" in loaded.layer_types


@pytest.mark.cpu_ok
@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("from_torch", [False, True])
def test_auto_config_requires_consent_before_executing_remote_code(tmp_path, requested, from_torch):
    _tiny_cohere2().save_pretrained(tmp_path)
    config_file = tmp_path / "config.json"
    config_dict = json.loads(config_file.read_text())
    config_dict["auto_map"] = {"AutoConfig": "configuration_review.RemoteConfig"}
    config_file.write_text(json.dumps(config_dict))
    marker = tmp_path / "remote_code_executed"
    (tmp_path / "configuration_review.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PretrainedConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class RemoteConfig(PretrainedConfig):\n"
        "    model_type = 'cohere2'\n"
    )
    kwargs = {} if requested is None else {"trust_remote_code": requested}

    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path), from_torch=from_torch, **kwargs)

    assert isinstance(loaded, Cohere2Config)
    assert marker.exists() is bool(requested)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("architectures", [None, ["Cohere2Model"]])
def test_auto_config_falls_back_to_causal_lm_when_task_inference_fails(tmp_path, architectures):
    _tiny_cohere2().save_pretrained(tmp_path)
    config_file = tmp_path / "config.json"
    config_dict = json.loads(config_file.read_text())
    config_dict["architectures"] = architectures
    config_file.write_text(json.dumps(config_dict))
    assert auto_configuration.infer_task_from_hf_config(str(tmp_path)) is None

    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path))

    assert isinstance(loaded, Cohere2Config)


@pytest.mark.cpu_ok
def test_auto_config_accepts_loose_task_strings(tmp_path):
    _tiny_cohere2().save_pretrained(tmp_path)

    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path), model_task="causal_lm")

    assert isinstance(loaded, Cohere2Config)
    assert normalize_task("sequence-classification") == TaskType.SEQUENCE_CLASSIFICATION


@pytest.mark.cpu_ok
def test_auto_state_any_to_any_returns_the_model_state(monkeypatch):
    state = SimpleNamespace(step=0, model="any_to_any_model")
    model = SimpleNamespace(to_state=lambda: state)
    monkeypatch.setattr(AutoEasyDeLAnyToAnyModel, "from_config", lambda **_: model)

    loaded = AutoStateAnyToAnyModel.from_config(SimpleNamespace())

    assert loaded is state
    assert loaded.model == "any_to_any_model"


@pytest.mark.cpu_ok
def test_dbrx_ffn_config_absorbs_extra_kwargs():
    config = DbrxFFNConfig(moe_num_experts=4, pad_token_id=7)

    assert config.moe_num_experts == 4
    assert config.pad_token_id == 7
    assert DbrxFFNConfig.from_dict(config.to_dict()).pad_token_id == 7


@pytest.mark.cpu_ok
def test_clip_text_projection_resolves_through_embedding_registry():
    from easydel.modules.clip.clip_configuration import CLIPTextConfig
    from easydel.modules.clip.modeling_clip import CLIPTextModelWithProjection
    from transformers import AutoModel
    from transformers import CLIPTextConfig as TorchCLIPTextConfig
    from transformers import CLIPTextModelWithProjection as TorchCLIPTextModelWithProjection

    original_mapping = AutoModel._model_mapping[TorchCLIPTextConfig]
    registration = registry.get_module_registration(TaskType.EMBEDDING, "clip_text_model")
    loader = registration.module.get_torch_loader()

    assert registration.module is CLIPTextModelWithProjection
    assert loader._model_mapping[CLIPTextConfig] is TorchCLIPTextModelWithProjection
    assert loader._model_mapping[TorchCLIPTextConfig] is TorchCLIPTextModelWithProjection
    assert AutoModel._model_mapping[TorchCLIPTextConfig] is original_mapping


@pytest.mark.cpu_ok
def test_clip_projection_loader_reads_local_checkpoint_config(tmp_path, monkeypatch):
    from easydel.modules.clip.modeling_clip import CLIPTextModelWithProjection
    from transformers import CLIPTextConfig as TorchCLIPTextConfig
    from transformers import CLIPTextModelWithProjection as TorchCLIPTextModelWithProjection

    config = TorchCLIPTextConfig(projection_dim=17)
    config.save_pretrained(tmp_path)
    loaded_model = SimpleNamespace(text_projection="projected")

    def load_reference(cls, path, *args, config, **kwargs):
        assert isinstance(config, TorchCLIPTextConfig)
        assert config.projection_dim == 17
        return loaded_model

    monkeypatch.setattr(TorchCLIPTextModelWithProjection, "from_pretrained", classmethod(load_reference))

    assert (
        CLIPTextModelWithProjection.get_torch_loader().from_pretrained(str(tmp_path), local_files_only=True)
        is loaded_model
    )


@pytest.mark.cpu_ok
def test_clip_projection_to_torch_uses_projected_model_class(monkeypatch):
    from easydel.infra.base_module import EasyDeLBaseModule
    from easydel.layers.layouts import _runtime
    from easydel.modules.clip.clip_configuration import CLIPTextConfig
    from easydel.modules.clip.modeling_clip import CLIPTextModelWithProjection
    from easydel.utils.parameters_transformation import ModelConverter
    from transformers import CLIPTextModelWithProjection as TorchCLIPTextModelWithProjection

    config = CLIPTextConfig(projection_dim=17)
    exported_model = SimpleNamespace(text_projection="exported")
    probe = SimpleNamespace(
        config=config,
        param_dtype="float32",
        get_torch_loader=CLIPTextModelWithProjection.get_torch_loader,
        _get_reform_param=lambda: {},
    )

    def convert(*, module, base_huggingface_module, config, dtype, reform_param, **kwargs):
        assert module is probe
        assert base_huggingface_module is TorchCLIPTextModelWithProjection
        assert config.projection_dim == 17
        return exported_model

    # Exercise the inherited export control flow without allocating model parameters
    # or executing JAX computation. Tensor conversion is covered by hardware tests.
    monkeypatch.setattr(_runtime, "tensor_parallel_size", lambda config: 1)
    monkeypatch.setattr(ModelConverter, "easydel_to_huggingface", staticmethod(convert))

    assert EasyDeLBaseModule.to_torch(probe) is exported_model


@pytest.mark.cpu_ok
def test_dbrx_import_does_not_modify_transformers_auxiliary_loss():
    from easydel.modules.dbrx import modeling_dbrx

    original_class = modeling_dbrx.DbrxForCausalLM
    original_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "dbrx")
    # Reload creates new model classes and re-registers them. Isolate both the
    # registry and module namespace rather than restoring only the HF function.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """\
                import importlib

                from easydel.modules.dbrx import modeling_dbrx
                from transformers.models.dbrx import modeling_dbrx as hf_dbrx

                def reference_loss(*args, **kwargs):
                    return 0

                hf_dbrx.load_balancing_loss_func = reference_loss
                importlib.reload(modeling_dbrx)

                assert hf_dbrx.load_balancing_loss_func is reference_loss
                assert hf_dbrx.load_balancing_loss_func(None) == 0
                """
            ),
        ],
        env={**os.environ, "ENABLE_DISTRIBUTED_INIT": "0", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert modeling_dbrx.DbrxForCausalLM is original_class
    assert registry.get_module_registration(TaskType.CAUSAL_LM, "dbrx") is original_registration
    assert original_registration.module is original_class
