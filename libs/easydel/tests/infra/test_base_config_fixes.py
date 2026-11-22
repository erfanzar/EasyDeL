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

"""Regression tests for ``EasyDeLBaseConfig`` behavior."""

import json
import warnings

import pytest
from easydel.infra.base_config import EasyDeLBaseConfig


@pytest.mark.cpu_ok
def test_read_basics_from_config_propagates_mlp_and_matmul_settings():
    operation_configs = {"review_test": object()}
    parent = EasyDeLBaseConfig(
        use_fused_mlp=False,
        collective_matmul_impl="xla",
        fcm_max_ratio=0.25,
        fcm_min_ratio=0.1,
        operation_configs=operation_configs,
    )
    child = EasyDeLBaseConfig()
    assert child.use_fused_mlp is True
    assert child.collective_matmul_impl == "auto"

    child.read_basics_from_config(parent)

    assert child.use_fused_mlp is False
    assert child.collective_matmul_impl == "xla"
    assert child.fcm_max_ratio == 0.25
    assert child.fcm_min_ratio == 0.1
    assert child.operation_configs is operation_configs


@pytest.mark.cpu_ok
def test_jax_mesh_deprecated_accessor_returns_config_mesh():
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = config.jax_mesh()
    assert mesh is config.mesh
    assert config.get_mesh() is config.mesh
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("explicit_repo_id", [None, "user/explicit"])
@pytest.mark.parametrize("as_epath", [False, True])
def test_save_pretrained_push_to_hub_accepts_pathlike(tmp_path, monkeypatch, explicit_repo_id, as_epath):
    uploads: list[tuple[str, str]] = []

    monkeypatch.setattr(EasyDeLBaseConfig, "_create_repo", lambda self, repo_id, **kw: repo_id, raising=False)
    monkeypatch.setattr(
        EasyDeLBaseConfig,
        "_upload_modified_files",
        lambda self, working_dir, repo_id, files_timestamps, **kw: uploads.append((str(working_dir), repo_id)),
    )

    target = tmp_path / "my-config"
    from eformer.paths import ePath

    kwargs = {} if explicit_repo_id is None else {"repo_id": explicit_repo_id}
    EasyDeLBaseConfig().save_pretrained(ePath(target) if as_epath else target, push_to_hub=True, **kwargs)

    assert (target / "config.json").is_file()
    assert uploads == [(str(target), explicit_repo_id or "my-config")]


@pytest.mark.cpu_ok
def test_save_pretrained_supports_hub_api_when_transformers_has_no_create_repo(tmp_path, monkeypatch):
    import types

    import huggingface_hub

    requested = []
    uploaded = []

    def create_repo(repo_id, **kwargs):
        requested.append((repo_id, kwargs))
        return types.SimpleNamespace(repo_id=f"user/{repo_id}")

    monkeypatch.setattr(EasyDeLBaseConfig, "_create_repo", None, raising=False)
    monkeypatch.setattr(huggingface_hub, "create_repo", create_repo)
    monkeypatch.setattr(
        EasyDeLBaseConfig,
        "_upload_modified_files",
        lambda self, directory, repo_id, timestamps, **kwargs: uploaded.append(repo_id),
    )
    EasyDeLBaseConfig().save_pretrained(tmp_path / "my-config", push_to_hub=True, private=True, token="test-token")
    assert requested == [("my-config", {"token": "test-token", "private": True, "exist_ok": True})]
    assert uploaded == ["user/my-config"]
    assert (tmp_path / "my-config" / "config.json").is_file()


@pytest.mark.cpu_ok
@pytest.mark.parametrize("as_epath", [False, True])
@pytest.mark.parametrize("filename", ["config.json", "custom_config.json"])
def test_from_pretrained_respects_subfolder_when_root_config_also_exists(tmp_path, as_epath, filename):
    from eformer.paths import ePath

    nested = tmp_path / "nested" / "text"
    nested.mkdir(parents=True)
    (tmp_path / filename).write_text(json.dumps({"hidden_size": 16}), encoding="utf-8")
    (nested / filename).write_text(json.dumps({"hidden_size": 32}), encoding="utf-8")
    path = ePath(tmp_path) if as_epath else tmp_path

    config = EasyDeLBaseConfig.from_pretrained(
        path, subfolder="nested/text", _configuration_file=filename, local_files_only=True
    )

    assert config.hidden_size == 32
    assert EasyDeLBaseConfig.from_pretrained(path, _configuration_file=filename).hidden_size == 16


@pytest.mark.cpu_ok
def test_from_pretrained_missing_subfolder_does_not_fall_back_to_root_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 16}), encoding="utf-8")

    # Transformers deliberately returns an empty config when config.json is absent.
    # Preserve that contract without accidentally consuming the root config.
    config_dict, _ = EasyDeLBaseConfig.get_config_dict(tmp_path, subfolder="missing", local_files_only=True)
    assert config_dict == {}
    config = EasyDeLBaseConfig.from_pretrained(tmp_path, subfolder="missing", local_files_only=True)
    assert not hasattr(config, "hidden_size")


@pytest.mark.cpu_ok
def test_from_pretrained_still_loads_direct_config_file(tmp_path):
    path = tmp_path / "direct.json"
    path.write_text(json.dumps({"hidden_size": 64}), encoding="utf-8")

    assert EasyDeLBaseConfig.from_pretrained(path, local_files_only=True).hidden_size == 64


@pytest.mark.cpu_ok
def test_get_fcm_mask_requires_rng_when_active_and_is_none_when_disabled():
    config = EasyDeLBaseConfig(fcm_max_ratio=0.5, fcm_min_ratio=0.1)
    with pytest.raises(ValueError, match="rng"):
        config.get_fcm_mask(2, 4, deterministic=False)
    assert config.get_fcm_mask(2, 4, deterministic=True) is None
    assert EasyDeLBaseConfig().get_fcm_mask(2, 4, deterministic=False) is None


@pytest.mark.cpu_ok
@pytest.mark.parametrize("flat_model_type", ["flat_rope_test_model", "gemma3_text"])
def test_per_layer_rope_parameters_declared_by_family_config(flat_model_type):
    class _PerLayerConfig(EasyDeLBaseConfig):
        model_type = "per_layer_rope_test_model"
        per_layer_rope_parameters = True

    layer_types = ["sliding_attention", "full_attention"]
    config = _PerLayerConfig(layer_types=layer_types, rope_theta=10000.0)
    config.layer_types = layer_types
    assert set(config.rope_parameters) == set(layer_types)

    class _FlatConfig(EasyDeLBaseConfig):
        model_type = flat_model_type

    flat = _FlatConfig(layer_types=layer_types, rope_theta=10000.0)
    flat.layer_types = layer_types
    assert "rope_type" in flat.rope_parameters


@pytest.mark.cpu_ok
def test_hf_filename_constants_still_importable_from_base_config():
    from easydel.infra import base_config

    assert base_config.SAFE_WEIGHTS_NAME == "model.safetensors"
    assert base_config.GENERATION_CONFIG_NAME == "generation_config.json"
    with pytest.raises(AttributeError):
        base_config.NOT_A_REAL_CONSTANT  # noqa: B018


@pytest.mark.cpu_ok
@pytest.mark.parametrize("backend", [None, "", "cpu"])
def test_public_config_accessors_return_axis_settings_and_resolve_backend(backend, monkeypatch):
    import types

    import jax.extend.backend

    # None resolves the backend at construction, whereas an empty string keeps
    # get_backend's dynamic fallback. Patch before either resolution can occur.
    monkeypatch.setattr("jax.default_backend", lambda: "cpu")
    monkeypatch.setattr(jax.extend.backend, "get_backend", lambda: types.SimpleNamespace(platform="cpu"))
    config = EasyDeLBaseConfig(backend=backend)
    assert config.get_axis_dims() == config.sharding_axis_dims
    assert config.get_axis_names() == config.sharding_axis_names
    assert config.get_backend() == "cpu"


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (2, 2, 4, 4)}, "shape"),
        ({"shape": (2, 1, 4, 3)}, "shape"),
        ({"shape": (3, 1, 4, 4), "batch_size": 2}, "batch_size"),
        ({"shape": (2, 1, 4, 4), "seq_length": 3}, "seq_length"),
        ({"batch_size": 0, "seq_length": 4}, "Positive"),
        ({"shape": (2, 1, 4, 4), "fcm_ratio": 1.1}, "fcm_ratio"),
    ],
)
def test_get_fcm_mask_rejects_invalid_shape_or_ratio_without_computation(kwargs, message):
    # Validation precedes random sampling, so a sentinel key suffices.
    with pytest.raises(ValueError, match=message):
        EasyDeLBaseConfig(fcm_max_ratio=0.5).get_fcm_mask(key=object(), **kwargs)
