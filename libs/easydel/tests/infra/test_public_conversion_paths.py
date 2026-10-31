"""Public loader integration without JAX numerics: only the array/init boundaries are substituted.

Real checkpoint discovery, key normalization, expert grouping, reform fusers,
PyTorch shard reads and TensorStore writes run in both public entry points.
This is control-flow/layout coverage, not accelerator/numerical JAX validation.
"""

import json
from functools import partial

import numpy as np
import pytest
from easydel.infra.factory import Registry, TaskType
from easydel.infra.mixins import bridge
from easydel.modules.auto import auto_configuration
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForCausalLM
from easydel.utils.parameters_transformation import TensorConverter
from transformers import MixtralConfig

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def conversion_model(monkeypatch):
    """Register a metadata-only model; leave loader/matcher/fuser implementations real."""

    class Config(MixtralConfig):
        def add_basic_configurations(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Model(bridge.EasyBridgeMixin):
        @classmethod
        def lazy_init(cls, *, config, param_dtype, **kwargs):
            instance = cls()
            instance.config = config
            instance.param_dtype = param_dtype
            return instance

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

    registry = Registry()
    registry.register_config("mixtral")(Config)
    registry.register_module(TaskType.CAUSAL_LM, Config, "mixtral")(Model)
    monkeypatch.setattr(auto_configuration, "registry", registry)
    # Do not execute a model initializer or the torch->JAX array boundary on CPU.
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
    return Model


def _normalizer(key):
    # Synthetic fusion outputs must not be run through this source-only mapping again.
    return key.removeprefix("vendor.") if key.startswith("vendor.") else None


def _fuse(torch, gate, up):
    return torch.cat((gate, up), dim=-2).transpose(-1, -2).contiguous()


def _unfuse(torch, fused):
    return torch.chunk(fused.transpose(-1, -2), 2, dim=-2)


@pytest.mark.parametrize(
    ("loader", "checkpoint_layout"),
    [
        ("streaming", "single"),
        ("streaming", "sharded"),
        ("sequential", "single"),
        ("sequential", "subfolder_single"),
        ("sequential", "subfolder_sharded"),
    ],
)
@pytest.mark.parametrize("expert_layout", ["canonical_parts", "reform_fusion"])
def test_public_conversion_preserves_experts_and_reform_fusion(
    tmp_path, conversion_model, loader, expert_layout, checkpoint_layout
):
    import tensorstore as ts
    import torch
    from jax.experimental.array_serialization import serialization as jax_ser

    subfolder = "nested" if checkpoint_layout.startswith("subfolder") else ""
    source = tmp_path / "source"
    checkpoint = source / subfolder
    checkpoint.mkdir(parents=True)
    MixtralConfig(
        hidden_size=3,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_local_experts=2,
        tie_word_embeddings=False,
    ).save_pretrained(checkpoint)
    block = "model.layers.0.mlp"
    group = f"{block}.experts"
    parts = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    state = {}
    expected = {}
    # Explicit, asymmetric values make expert order, transpose and fusion order observable.
    for part_index, (part, leaf) in enumerate(parts.items()):
        values = []
        for expert in range(2):
            value = np.arange(6, dtype=np.float32).reshape(2, 3) + 100 * part_index + 10 * expert
            name = part if expert_layout == "canonical_parts" else leaf
            state[f"vendor.{group}.{expert}.{name}.weight"] = torch.from_numpy(value.copy())
            values.append(value.T)
        expected[f"{group}.{leaf}.weight"] = np.stack(values)

    # Normalizers explicitly reject optional/unowned payloads. They must not
    # reach conversion or appear in the resulting model/checkpoint.
    state["mtp.0.weight"] = torch.ones((2, 3), dtype=torch.float32)
    reform = {}
    moe_names = list(parts.values())
    if expert_layout == "reform_fusion":
        reform[f"{group}.gate_up_proj.weight$"] = {
            "sources": [f"{group}.gate_proj.weight", f"{group}.up_proj.weight"],
            "fuser": _fuse,
            "inverse_fuser": _unfuse,
            "already_converted": True,
        }
        expected[f"{group}.gate_up_proj.weight"] = np.concatenate(
            (expected.pop(f"{group}.gate_proj.weight"), expected.pop(f"{group}.up_proj.weight")), axis=-1
        )
        moe_names = ["gate_up_proj", "down_proj"]

    conversion_model.pure_transform_fn = staticmethod(
        partial(
            dict,
            moe_block_path=[block],
            moe_path=[f"{group}.{name}" for name in moe_names],
            moe_names=moe_names,
            reform_param=reform,
            checkpoint_key_normalizer=_normalizer,
            hf_model_type="mixtral",
        )
    )
    # Only shape metadata is needed by the incremental TensorStore writer.
    conversion_model.parameters_shape = {key: value for key, value in expected.items()}

    if checkpoint_layout.endswith("sharded"):
        shards = [{}, {}]
        weight_map = {}
        for index, (key, value) in enumerate(state.items()):
            shard = index % 2
            shards[shard][key] = value
            # Indexes in the wild may already prefix the configured subfolder.
            weight_map[key] = f"{subfolder + '/' if subfolder else ''}part-{shard}.bin"
        for shard, values in enumerate(shards):
            torch.save(values, checkpoint / f"part-{shard}.bin")
        (checkpoint / bridge.WEIGHTS_INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    else:
        torch.save(state, checkpoint / bridge.WEIGHTS_NAME)

    common = dict(
        pretrained_model_name_or_path=str(source),
        subfolder=subfolder,
        local_files_only=True,
        dtype=np.float32,
        param_dtype=np.float32,
        verbose=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    if loader == "streaming":
        # The Auto loader's typed configuration does not currently expose subfolder.
        # Sequential conversion below exercises the public subfolder option.
        common.pop("subfolder")
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            **common,
            from_torch=True,
            auto_shard_model=False,
            torch_load_mode="streaming",
            torch_streaming_cache="hf_cache",
        )
        actual = bridge.flatten_dict(model.params)
        actual = {".".join(map(str, key)): value for key, value in actual.items()}
    else:
        destination = tmp_path / "converted"
        result = conversion_model.huggingface_to_easydel_sequential(**common, save_directory=destination)
        assert result == str(destination)
        entries = json.loads((destination / bridge.TENSORSTORE_INDEX_NAME).read_text())["prefixes"]["model"]
        actual = {}
        for entry in entries:
            key = entry["path"].removeprefix("model/").replace("/", ".")
            spec = jax_ser.get_tensorstore_spec(str(destination / entry["path"]))
            actual[key] = ts.open(spec).result().read().result()
            assert entry["shape"] == list(expected[key].shape)
            assert entry["dtype"] == "float32"

    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
        assert actual[key].dtype == np.float32


@pytest.mark.parametrize("loader", ["streaming", "sequential"])
def test_remote_sharded_safetensors_win_over_legacy_bin(tmp_path, conversion_model, monkeypatch, loader):
    import huggingface_hub
    import torch
    from easydel.infra import base_config
    from safetensors.torch import save_file
    from transformers import configuration_utils
    from transformers.generation import configuration_utils as generation_configuration_utils

    source = tmp_path / "remote"
    source.mkdir()
    MixtralConfig(tie_word_embeddings=False).save_pretrained(source)
    # Asymmetric matrix: embedding/layernorm options preserve layout while a
    # projection transposes. This exercises the consumed positional contract.
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = {
        "vendor.model.embed.weight": values,
        "vendor.model.norm.weight": values + 10,
        "vendor.model.proj.weight": values + 20,
    }
    save_file(state, source / "part.safetensors")
    torch.save({"wrong.weight": torch.ones((1, 1))}, source / bridge.WEIGHTS_NAME)
    (source / bridge.SAFE_WEIGHTS_INDEX_NAME).write_text(
        json.dumps({"weight_map": dict.fromkeys(state, "part.safetensors")})
    )
    conversion_model.pure_transform_fn = staticmethod(
        partial(
            dict,
            checkpoint_key_normalizer=_normalizer,
            embedding_layer_names=["model.embed"],
            layernorm_names=["model.norm"],
        )
    )
    expected = {
        "model.embed.weight": values.numpy(),
        "model.norm.weight": (values + 10).numpy(),
        "model.proj.weight": (values + 20).numpy().T,
    }
    conversion_model.parameters_shape = expected
    downloads = []

    def download(repo_id, filename, **kwargs):
        assert repo_id == "review-remote/model"
        downloads.append(filename)
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(base_config, "cached_file", download)
    monkeypatch.setattr(configuration_utils, "cached_file", download)
    monkeypatch.setattr(generation_configuration_utils, "cached_file", download)
    common = dict(
        pretrained_model_name_or_path="review-remote/model",
        local_files_only=True,
        dtype=np.float32,
        param_dtype=np.float32,
        verbose=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    if loader == "streaming":
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            **common, from_torch=True, auto_shard_model=False, torch_load_mode="streaming"
        )
        actual = {".".join(map(str, key)): value for key, value in bridge.flatten_dict(model.params).items()}
    else:
        import tensorstore as ts
        from jax.experimental.array_serialization import serialization as jax_ser

        destination = tmp_path / "converted"
        conversion_model.huggingface_to_easydel_sequential(**common, save_directory=destination)
        entries = json.loads((destination / bridge.TENSORSTORE_INDEX_NAME).read_text())["prefixes"]["model"]
        actual = {
            entry["path"].removeprefix("model/").replace("/", "."): ts.open(
                jax_ser.get_tensorstore_spec(str(destination / entry["path"]))
            )
            .result()
            .read()
            .result()
            for entry in entries
        }
    assert bridge.WEIGHTS_NAME not in downloads
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        np.testing.assert_array_equal(actual[key], expected_value)


@pytest.mark.parametrize("partial_expert", [False, True])
def test_sequential_resume_keeps_completed_outputs_after_interruption(
    tmp_path, conversion_model, monkeypatch, partial_expert
):
    import safetensors.torch
    import torch

    source = tmp_path / "source"
    source.mkdir()
    MixtralConfig(tie_word_embeddings=False).save_pretrained(source)
    state = {
        "vendor.model.zfirst.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "vendor.model.second.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10,
    }
    files = ["a.safetensors", "b.safetensors"]
    weight_map = dict(zip(state, files, strict=True))
    group = "model.layers.0.mlp.experts.down_proj"
    if partial_expert:
        for index, filename in enumerate(files):
            key = f"vendor.model.layers.0.mlp.experts.{index}.down_proj.weight"
            state[key] = torch.full((2, 3), index + 1, dtype=torch.float32)
            weight_map[key] = filename
    for filename in files:
        safetensors.torch.save_file(
            {key: value for key, value in state.items() if weight_map[key] == filename}, source / filename
        )
    (source / bridge.SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    options = {"checkpoint_key_normalizer": _normalizer}
    if partial_expert:
        options.update(moe_block_path=["model.layers.0.mlp"], moe_path=[group], moe_names=["down_proj"])
    conversion_model.pure_transform_fn = staticmethod(partial(dict, **options))
    conversion_model.parameters_shape = {
        key.removeprefix("vendor."): value.numpy().T for key, value in state.items() if ".experts." not in key
    }
    if partial_expert:
        conversion_model.parameters_shape[f"{group}.weight"] = np.empty((2, 3, 2), dtype=np.float32)
    original_open = safetensors.torch.safe_open
    resumed = False
    reads = []

    def interrupted_open(path, *args, **kwargs):
        name = str(path).split("/")[-1]
        reads.append(name)
        if name == "b.safetensors" and not resumed:
            raise RuntimeError("interrupted checkpoint read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "safe_open", interrupted_open)
    destination = tmp_path / "converted"
    common = dict(
        pretrained_model_name_or_path=str(source),
        save_directory=destination,
        local_files_only=True,
        dtype=np.float32,
        param_dtype=np.float32,
        verbose=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    with pytest.raises(RuntimeError, match="interrupted checkpoint read"):
        conversion_model.huggingface_to_easydel_sequential(**common)
    entries = json.loads((destination / bridge.TENSORSTORE_INDEX_NAME).read_text())["prefixes"]["model"]
    assert [entry["path"] for entry in entries] == ["model/model/zfirst/weight"]
    resumed = True
    reads.clear()
    conversion_model.huggingface_to_easydel_sequential(**common, resume=True)
    assert reads == (["a.safetensors", "b.safetensors"] if partial_expert else ["b.safetensors"])
    entries = json.loads((destination / bridge.TENSORSTORE_INDEX_NAME).read_text())["prefixes"]["model"]
    expected_paths = {"model/model/zfirst/weight", "model/model/second/weight"}
    if partial_expert:
        expected_paths.add("model/model/layers/0/mlp/experts/down_proj/weight")
        import tensorstore as ts
        from jax.experimental.array_serialization import serialization as jax_ser

        expert_path = destination / "model/model/layers/0/mlp/experts/down_proj/weight"
        restored = ts.open(jax_ser.get_tensorstore_spec(str(expert_path))).result().read().result()
        np.testing.assert_array_equal(restored, np.stack([np.full((3, 2), 1), np.full((3, 2), 2)]))
    assert {entry["path"] for entry in entries} == expected_paths
