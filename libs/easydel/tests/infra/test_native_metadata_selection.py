"""Native public loader metadata selection using real, string-only checkpoints.

No JAX arrays or numerical computation: only the RNG initialization boundary
is replaced. Registry/config resolution, native loading, state merge and
checkpoint sidecar reads all run normally.
"""

import json
import typing as tp

import jax
import pytest
import spectrax as spx
from easydel.infra.factory import Registry, TaskType
from easydel.infra.mixins import bridge
from easydel.modules.auto import auto_configuration
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForCausalLM

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def native_model(monkeypatch):
    class Config(bridge.EasyDeLBaseConfig):
        model_type = "gpt2"
        _legacy_native_defaults: tp.ClassVar = {"modern_behavior": False}

        def __init__(self, modern_behavior=True, **kwargs):
            super().__init__(**kwargs)
            self.modern_behavior = modern_behavior

    class Model(spx.Module, bridge.EasyBridgeMixin):
        _model_task = TaskType.CAUSAL_LM

        def __init__(self, config):
            self.config = config
            self.label = spx.Parameter("not loaded")
            self.mesh = jax.sharding.Mesh([jax.devices("cpu")[0]], ("single",))

        @classmethod
        def lazy_init(cls, *, config, **kwargs):
            return cls(config)

        def assert_parameters_materialized(self, **kwargs):
            assert isinstance(self.label.value, str)

        def materialize_meta_state(self):
            return self

        def quantize(self, **kwargs):
            return self

        def wire_distributed_matmul(self):
            return self

    registry = Registry()
    registry.register_config("gpt2")(Config)
    registry.register_module(TaskType.CAUSAL_LM, Config, "gpt2")(Model)
    monkeypatch.setattr(auto_configuration, "registry", registry)

    class NoArrayRngs:
        def __init__(self, seed):
            pass

    monkeypatch.setattr(bridge.spx, "Rngs", NoArrayRngs)
    return Model


def _write_checkpoint(directory, native_model, *, modern_behavior=None):
    directory.mkdir(parents=True, exist_ok=True)
    raw = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "n_embd": 12, "n_head": 1, "n_layer": 1}
    if modern_behavior is not None:
        raw["modern_behavior"] = modern_behavior
    (directory / "config.json").write_text(json.dumps(raw))
    (directory / "generation_config.json").write_text(json.dumps({"max_new_tokens": 37, "do_sample": False}))
    mesh = jax.sharding.Mesh([jax.devices("cpu")[0]], ("single",))
    checkpointer = bridge.Checkpointer(base_path=str(directory), save_interval=None, step_policies=[])
    checkpointer.save_pytree(tree={"parameters": {"label": "nested checkpoint"}}, prefix="model", mesh=mesh)
    checkpointer.wait_until_finished()
    # Pure-Python checkpoints need only their structure sidecar, but the public
    # native format detector also needs its usual TensorStore index marker.
    (directory / bridge.TENSORSTORE_INDEX_NAME).write_text(json.dumps({"prefixes": {"model": []}}))


def _load(loader, native_model, path, **kwargs):
    entry = native_model if loader == "direct" else AutoEasyDeLModelForCausalLM
    return entry.from_pretrained(
        pretrained_model_name_or_path=str(path),
        auto_shard_model=False,
        verbose=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


@pytest.mark.parametrize("loader", ["direct", "auto"])
@pytest.mark.parametrize("root_config", [False, True])
@pytest.mark.parametrize("modern_behavior", [None, True])
def test_native_loader_uses_selected_metadata_and_restores_checkpoint(
    tmp_path, native_model, loader, root_config, modern_behavior
):
    _write_checkpoint(tmp_path / "nested", native_model, modern_behavior=modern_behavior)
    if root_config:
        # Both wrong family selection and wrong raw legacy-default decisions
        # must be observable, not hidden by a duplicate root config.
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "root_must_not_be_selected", "modern_behavior": modern_behavior is None})
        )
        (tmp_path / "generation_config.json").write_text(json.dumps({"max_new_tokens": 99}))
    model = _load(loader, native_model, tmp_path, subfolder="nested", local_files_only=True)
    assert isinstance(model, native_model)
    assert model.config.n_embd == 12
    assert model.config.modern_behavior is (modern_behavior is not None)
    assert model.label.value == "nested checkpoint"
    assert model.generation_config.max_new_tokens == 37
    assert not model.training


@pytest.mark.parametrize("root_config", [False, True])
@pytest.mark.parametrize("loader", ["direct", "auto"])
def test_public_versioned_save_load_roundtrip_selects_child_metadata(tmp_path, native_model, root_config, loader):
    from transformers import GenerationConfig

    # Resolve the registered config without constructing any array parameters.
    config_class = auto_configuration.registry.get_config("gpt2")
    model = native_model(config_class(n_embd=12, n_head=1, n_layer=1, modern_behavior=True))
    model.label.value = "versioned checkpoint"
    model.generation_config = GenerationConfig(max_new_tokens=53)
    model.save_pretrained(tmp_path, step=17)
    if root_config:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "wrong_root_family"}))
    loaded = _load(
        loader, native_model, tmp_path, local_files_only=True, **({"from_torch": False} if loader == "auto" else {})
    )
    assert loaded.label.value == "versioned checkpoint"
    assert loaded.config.modern_behavior is True
    assert loaded.config.n_embd == 12
    assert loaded.generation_config.max_new_tokens == 53


@pytest.mark.parametrize("loader", ["direct", "auto"])
def test_native_loader_forwards_hub_options_to_selected_metadata(tmp_path, native_model, monkeypatch, loader):
    import huggingface_hub
    from easydel.infra import base_config
    from transformers import configuration_utils
    from transformers.generation import configuration_utils as generation_configuration_utils

    _write_checkpoint(tmp_path / "remote" / "nested", native_model, modern_behavior=True)
    cache_dir = str(tmp_path / "cache")
    token = "test-token-not-a-secret"
    # The direct loader exposes proxies; the typed Auto API does not currently
    # accept that argument. Both paths support token authentication.
    proxies = {"https": "https://review-user:review-proxy-password@proxy.invalid:443"} if loader == "direct" else None

    def download(repo_id, filename, *, subfolder="", **kwargs):
        # This transport emulates a gated, revision-specific cached snapshot.
        # Wrong/missing access or location options cannot read its metadata.
        assert repo_id == "review-private/native-model"
        assert kwargs["token"] == token
        assert kwargs.get("proxies") == proxies
        assert kwargs["revision"] == "selected-revision"
        assert kwargs["cache_dir"] == cache_dir
        assert kwargs["local_files_only"] is True
        path = tmp_path / "remote" / subfolder / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    monkeypatch.setattr(base_config, "cached_file", download)
    monkeypatch.setattr(configuration_utils, "cached_file", download)
    monkeypatch.setattr(generation_configuration_utils, "cached_file", download)
    monkeypatch.setattr(
        huggingface_hub.HfApi, "hf_hub_download", lambda self, *args, **kwargs: download(*args, **kwargs)
    )
    model = _load(
        loader,
        native_model,
        "review-private/native-model",
        subfolder="nested",
        cache_dir=cache_dir,
        revision="selected-revision",
        token=token,
        local_files_only=True,
        **({"from_torch": False} if loader == "auto" else {"proxies": proxies}),
    )
    assert isinstance(model, native_model)
    assert model.config.modern_behavior is True
    assert model.label.value == "nested checkpoint"
    assert model.generation_config.max_new_tokens == 37

    # Loading credentials belong to transport, not to persisted model settings.
    # Exercise both the serialization API and a real load-then-save round trip.
    diff = model.config.to_diff_dict()
    saved_directory = tmp_path / "exported-config"
    model.config.save_pretrained(saved_directory)
    saved = json.loads((saved_directory / "config.json").read_text())
    serialized = json.dumps({"diff": diff, "saved": saved})
    assert {
        "diff_has_token": "token" in diff,
        "diff_has_proxies": "proxies" in diff,
        "saved_has_token": "token" in saved,
        "saved_has_proxies": "proxies" in saved,
        "contains_token_value": token in serialized,
        "contains_proxy_credentials": "review-proxy-password" in serialized,
    } == dict.fromkeys(
        (
            "diff_has_token",
            "diff_has_proxies",
            "saved_has_token",
            "saved_has_proxies",
            "contains_token_value",
            "contains_proxy_credentials",
        ),
        False,
    )
