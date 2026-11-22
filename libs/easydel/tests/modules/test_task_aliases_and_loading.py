"""Public task aliases and subfolder loading without JAX array computation."""

import json
from functools import partial

import numpy as np
import pytest
from easydel.infra.factory import Registry, TaskType
from easydel.infra.mixins import bridge
from easydel.modules.auto import auto_configuration
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForCausalLM, PreTrainedLoading
from easydel.utils.parameters_transformation import TensorConverter
from transformers import MixtralConfig

pytestmark = pytest.mark.cpu_ok


def test_task_normalization_is_shared_across_public_entry_points():
    from easydel.infra import factory
    from easydel.infra.elarge import processing

    for module in (auto_configuration, processing):
        assert module.TASK_ALIASES is factory.TASK_ALIASES
        assert module.normalize_task is factory.normalize_task
        for task in TaskType:
            for value in (task, task.value, task.value.replace("-", "_"), f" {task.value.upper()} "):
                assert module.normalize_task(t=value) is task
        for alias, task in factory.TASK_ALIASES.items():
            assert module.normalize_task(alias) is task
            assert module.normalize_task(f" {alias.upper().replace('_', '-')} ") is task
        for value in (None, "", "unknown_task"):
            assert module.normalize_task(value) is None


def test_pretrained_loading_subfolder_roundtrip():
    config = PreTrainedLoading.from_dict(pretrained_model_name_or_path="local-model", subfolder="nested")
    assert config.to_dict()["subfolder"] == "nested"
    assert PreTrainedLoading.from_dict(config.to_dict()).subfolder == "nested"
    assert PreTrainedLoading.from_dict(pretrained_model_name_or_path="local-model").subfolder == ""


@pytest.mark.parametrize("from_torch", [False, True])
@pytest.mark.parametrize("conflicting_root", [False, True])
def test_auto_configuration_reads_selected_subfolder(tmp_path, from_torch, conflicting_root):
    nested = tmp_path / "nested"
    nested.mkdir()
    config_dict = {
        "model_type": "cohere",
        "architectures": ["CohereForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "vocab_size": 32,
    }
    (nested / "config.json").write_text(json.dumps(config_dict))
    if conflicting_root:
        (tmp_path / "config.json").write_text(json.dumps({**config_dict, "model_type": "cohere2", "hidden_size": 24}))
    config = auto_configuration.AutoEasyDeLConfig.from_pretrained(
        str(tmp_path), subfolder="nested", local_files_only=True, from_torch=from_torch
    )
    assert config.model_type == "cohere"
    assert config.hidden_size == 16
    assert config.architectures == ["CohereForCausalLM"]


@pytest.mark.parametrize("sharded", [False, True])
def test_auto_streaming_conversion_reads_checkpoint_subfolder(tmp_path, monkeypatch, sharded):
    import torch

    class Config(MixtralConfig):
        def add_basic_configurations(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Model(bridge.EasyBridgeMixin):
        @classmethod
        def lazy_init(cls, *, config, param_dtype, **kwargs):
            model = cls()
            model.config, model.param_dtype = config, param_dtype
            return model

        def assert_parameters_materialized(self, **kwargs):
            return None

        def materialize_meta_state(self):
            return self

        def quantize(self, **kwargs):
            return self

        def wire_distributed_matmul(self):
            return self

        def eval(self):
            return self

        pure_transform_fn = staticmethod(partial(dict, embedding_layer_names=["embed_tokens"]))

    registry = Registry()
    registry.register_config("mixtral")(Config)
    registry.register_module(TaskType.CAUSAL_LM, Config, "mixtral")(Model)
    monkeypatch.setattr(auto_configuration, "registry", registry)
    monkeypatch.setattr(bridge.spx, "Rngs", lambda seed: None)
    monkeypatch.setattr(
        TensorConverter,
        "convert_pytorch_to_jnp",
        staticmethod(lambda tensor, dtype: tensor.detach().cpu().numpy().astype(dtype)),
    )

    def merge(model, tree):
        model.params = tree
        return model

    monkeypatch.setattr(bridge, "merge_model_and_tree", merge)
    checkpoint = tmp_path / "nested"
    checkpoint.mkdir()
    MixtralConfig(hidden_size=3, num_attention_heads=1, num_key_value_heads=1).save_pretrained(checkpoint)
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    state = {"model.embed_tokens.weight": torch.from_numpy(expected.copy())}
    if sharded:
        torch.save(state, checkpoint / "part-0.bin")
        (checkpoint / bridge.WEIGHTS_INDEX_NAME).write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "nested/part-0.bin"}})
        )
    else:
        torch.save(state, checkpoint / bridge.WEIGHTS_NAME)

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path),
        subfolder="nested",
        local_files_only=True,
        auto_shard_model=False,
        torch_load_mode="streaming",
        torch_streaming_cache="hf_cache",
        dtype=np.float32,
        param_dtype=np.float32,
        verbose=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    actual = {".".join(key): value for key, value in bridge.flatten_dict(model.params).items()}
    assert isinstance(model, Model)
    assert set(actual) == {"model.embed_tokens.weight"}
    np.testing.assert_array_equal(actual["model.embed_tokens.weight"], expected)


@pytest.mark.parametrize("native", [False, True])
def test_auto_format_detection_uses_selected_subfolder(tmp_path, monkeypatch, native):
    """A conflicting root format must not select the wrong bridge loader."""
    nested = tmp_path / "nested"
    nested.mkdir()
    # Use the actual marker name so this stays aligned with the checkpoint format.
    from easydel.modules.auto.auto_modeling import TENSORSTORE_INDEX_NAME

    native_marker = TENSORSTORE_INDEX_NAME
    (nested / (native_marker if native else "pytorch_model.bin")).write_text("{}")
    (tmp_path / ("pytorch_model.bin" if native else native_marker)).write_text("{}")
    monkeypatch.setattr(
        AutoEasyDeLModelForCausalLM,
        "_from_easydel_params",
        classmethod(lambda cls, config: ("native", config.subfolder)),
    )
    monkeypatch.setattr(
        AutoEasyDeLModelForCausalLM,
        "_from_torch_pretrained",
        classmethod(lambda cls, config: ("torch", config.subfolder)),
    )
    assert AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path), subfolder="nested", local_files_only=True
    ) == ("native" if native else "torch", "nested")
