"""Public loader integration without JAX numerics: only the array/init boundaries are substituted.

Real checkpoint discovery, key normalization, expert grouping, reform fusers,
PyTorch shard reads and TensorStore writes run in both public entry points.
This is control-flow/layout coverage, not accelerator/numerical JAX validation.
"""

import collections
import contextlib
import json
from functools import partial
from types import SimpleNamespace

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


# --- gs:// sources and converted_save_directory ---------------------------------------------------------------

_MOE_BLOCK = "model.layers.0.mlp"
_MOE_GROUP = f"{_MOE_BLOCK}.experts"
_LOAD_OPTIONS = dict(
    local_files_only=True,
    auto_shard_model=False,
    dtype=np.float32,
    param_dtype=np.float32,
    verbose=False,
    sharding_axis_dims=(1, 1, 1, 1, 1, 1),
)


class _LocalStorageClient:
    """``google.cloud.storage.Client`` stand-in serving ``gs://<bucket>/<name>`` from ``root/<bucket>/<name>``.

    Only the storage client is replaced: ePath, the bridge's staging helpers and both loaders run unchanged.
    Every object download is recorded, and ``on_download(name)`` runs before the bytes are returned, so tests
    can observe network reads and what the loader had staged on local disk at that moment.
    """

    def __init__(self, root):
        self.root = root
        self.downloads = []
        self.on_download = None
        # Object name -> exception every download of that object raises (a denied or failing request).
        self.errors = {}

    def bucket(self, name):
        return _LocalStorageBucket(self, name)


class _LocalStorageBucket:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.root = client.root / name

    def exists(self):
        return self.root.is_dir()

    def blob(self, name):
        return _LocalStorageBlob(self, name)

    def list_blobs(self, prefix="", max_results=None, **kwargs):
        names = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        hits = [SimpleNamespace(name=name) for name in names if name.startswith(prefix)]
        return hits if max_results is None else hits[:max_results]


class _LocalStorageBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.path = bucket.root / name

    def exists(self):
        return self.path.is_file()

    @property
    def size(self):
        return self.path.stat().st_size if self.path.is_file() else None

    def reload(self):
        # Metadata read, as google.cloud.storage.Blob.reload: a missing object raises NotFound; nothing is downloaded.
        from google.api_core.exceptions import NotFound

        if not self.path.is_file():
            raise NotFound(f"gs://{self.bucket.name}/{self.name}")

    def download_as_bytes(self):
        from google.api_core.exceptions import NotFound

        error = self.bucket.client.errors.get(self.name)
        if error is not None:
            raise error
        if not self.path.is_file():
            raise NotFound(f"gs://{self.bucket.name}/{self.name}")
        client = self.bucket.client
        if client.on_download is not None:
            client.on_download(self.name)
        client.downloads.append(self.name)
        return self.path.read_bytes()

    def download_as_text(self, encoding="utf-8"):
        return self.download_as_bytes().decode(encoding)

    def upload_from_string(self, data, content_type=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


@pytest.fixture
def gcs_bucket(tmp_path, monkeypatch):
    """Serve ``gs://`` URIs from ``tmp_path/gcs`` without network access."""
    import shutil

    import eformer.paths as paths

    client = _LocalStorageClient(tmp_path / "gcs")
    monkeypatch.setattr(paths.ePath, "_gcs_client", client)
    real_which = shutil.which

    def which(cmd, *args, **kwargs):
        # `gcloud storage cp` would reach the real bucket; without the CLI every transfer goes through ePath.
        return None if cmd == "gcloud" else real_which(cmd, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)
    return client


def _write_moe_checkpoint(directory, *, layout="sharded", weight_format="safetensors", index_prefix=""):
    """Write a small Hugging Face-format MoE checkpoint into ``directory``.

    The down_proj experts and the two sources of a reform fusion are each split across two of the four
    shards; the last shard also holds an unowned tensor that the normalizer rejects.

    Returns:
        ``(expected, weight_files)``: converted values keyed by dotted parameter path, and the weight files
        written (the shards in index order, or the single payload).
    """
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    MixtralConfig(
        hidden_size=3,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_local_experts=4,
        tie_word_embeddings=False,
    ).save_pretrained(directory)

    def value(offset, shape=(2, 3)):
        return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + offset

    down = [value(100 + 10 * expert) for expert in range(4)]
    gate = [value(200 + 10 * expert) for expert in range(4)]
    q, k = value(300), value(400)
    embed, proj, norm = value(0), value(10), value(20, (3,))
    shards = [
        {"vendor.model.embed.weight": embed, "vendor.model.attn.q.weight": q}
        | {f"vendor.{_MOE_GROUP}.{expert}.down_proj.weight": down[expert] for expert in (0, 1)},
        {"vendor.model.proj.weight": proj}
        | {f"vendor.{_MOE_GROUP}.{expert}.down_proj.weight": down[expert] for expert in (2, 3)},
        {"vendor.model.attn.k.weight": k}
        | {f"vendor.{_MOE_GROUP}.{expert}.gate_proj.weight": gate[expert] for expert in range(4)},
        {"vendor.model.norm.weight": norm, "mtp.0.weight": value(900)},
    ]
    # Independent reference: embeddings/norms keep their layout, projections transpose, experts stack
    # transposed per expert, and the fusion concatenates q and k before its own transpose.
    expected = {
        "model.embed.weight": embed,
        "model.proj.weight": proj.T,
        "model.norm.weight": norm,
        "model.attn.qk.weight": np.concatenate((q, k)).T,
        f"{_MOE_GROUP}.down_proj.weight": np.stack([tensor.T for tensor in down]),
        f"{_MOE_GROUP}.gate_proj.weight": np.stack([tensor.T for tensor in gate]),
    }
    safetensors = weight_format == "safetensors"

    def save(tensors, path):
        tensors = {key: torch.from_numpy(array.copy()) for key, array in tensors.items()}
        if safetensors:
            save_file(tensors, path)
        else:
            torch.save(tensors, path)

    if layout == "single":
        name = bridge.SAFE_WEIGHTS_NAME if safetensors else bridge.WEIGHTS_NAME
        save({key: array for shard in shards for key, array in shard.items()}, directory / name)
        return expected, [name]
    stem, suffix = ("model", "safetensors") if safetensors else ("pytorch_model", "bin")
    weight_files, weight_map = [], {}
    for index, tensors in enumerate(shards, start=1):
        name = f"{stem}-{index:05d}-of-{len(shards):05d}.{suffix}"
        save(tensors, directory / name)
        weight_files.append(name)
        weight_map.update(dict.fromkeys(tensors, index_prefix + name))
    index_name = bridge.SAFE_WEIGHTS_INDEX_NAME if safetensors else bridge.WEIGHTS_INDEX_NAME
    (directory / index_name).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return expected, weight_files


def _use_moe_transform(model):
    """Give the metadata-only model the conversion options matching :func:`_write_moe_checkpoint`."""
    model.pure_transform_fn = staticmethod(
        partial(
            dict,
            embedding_layer_names=["model.embed"],
            layernorm_names=["model.norm"],
            moe_block_path=[_MOE_BLOCK],
            moe_path=[f"{_MOE_GROUP}.down_proj", f"{_MOE_GROUP}.gate_proj"],
            moe_names=["down_proj", "gate_proj"],
            reform_param={
                "model.attn.qk.weight$": {
                    "sources": ["model.attn.q.weight", "model.attn.k.weight"],
                    "fuser": _fuse,
                    "inverse_fuser": _unfuse,
                    "already_converted": True,
                }
            },
            checkpoint_key_normalizer=_normalizer,
            hf_model_type="mixtral",
        )
    )


def _converted_params(model):
    return {".".join(map(str, key)): value for key, value in bridge.flatten_dict(model.params).items()}


@pytest.mark.parametrize(
    ("layout", "weight_format", "subfolder", "options"),
    [
        ("sharded", "safetensors", "", {}),
        # The 'temp' cache used to refuse expert groups that span shard files.
        ("sharded", "bin", "", {"torch_streaming_cache": "temp"}),
        # transformers' full loader cannot read gs://: the request warns and streams instead.
        ("sharded", "safetensors", "nested", {"torch_load_mode": "full"}),
        ("single", "safetensors", "", {}),
    ],
)
def test_gcs_hf_checkpoint_loads_through_public_from_pretrained(
    tmp_path, conversion_model, gcs_bucket, layout, weight_format, subfolder, options
):
    checkpoint = gcs_bucket.root / "bucket" / "models" / "hf" / subfolder
    expected, weight_files = _write_moe_checkpoint(
        checkpoint, layout=layout, weight_format=weight_format, index_prefix=f"{subfolder}/" if subfolder else ""
    )
    _use_moe_transform(conversion_model)
    staging = tmp_path / "staging"
    shards = set(weight_files) if layout == "sharded" else set()
    staged_at_fetch = []

    def count_staged_shards(name):
        if name.rsplit("/", 1)[-1] in shards:
            staged_at_fetch.append(sum(path.name in shards for path in staging.rglob("*")))

    gcs_bucket.on_download = count_staged_shards
    expect_warning = options.get("torch_load_mode") == "full"
    with pytest.warns(UserWarning, match="streaming") if expect_warning else contextlib.nullcontext():
        # Positional location and no from_torch: the format is detected through ePath on the bucket.
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            "gs://bucket/models/hf",
            subfolder=subfolder,
            torch_streaming_tmp_dir=str(staging),
            **_LOAD_OPTIONS,
            **options,
        )

    actual = _converted_params(model)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
        assert actual[key].dtype == np.float32
    # Each weight file crossed the network once, including the shards an expert group or a fusion spans ...
    fetched = collections.Counter(name.rsplit("/", 1)[-1] for name in gcs_bucket.downloads)
    assert {name: fetched[name] for name in weight_files} == dict.fromkeys(weight_files, 1)
    # ... no more than two shards were ever on local disk together, and nothing is left behind.
    if shards:
        assert len(staged_at_fetch) == len(shards)
        assert max(staged_at_fetch) + 1 <= 2 < len(shards)
    assert list(staging.iterdir()) == []


_TOKENIZER_FILES = {
    "tokenizer.json": b'{"model": {"type": "BPE"}}',
    "tokenizer_config.json": b'{"model_max_length": 8}',
    "special_tokens_map.json": b'{"eos_token": "</s>"}',
    "chat_template.jinja": b"{% for message in messages %}{{ message.content }}{% endfor %}",
    "preprocessor_config.json": b'{"size": 4}',
}


@pytest.mark.parametrize(
    ("source_kind", "destination_kind", "process_index"),
    [
        ("gcs", "gcs", 0),
        ("gcs", "local", 0),
        ("local", "local", 0),
        ("local", "gcs", 0),
        ("hub", "local", 0),
        ("hub", "gcs", 0),
        # The save is collective: every process writes it; only process 0 copies tokenizer files.
        ("gcs", "gcs", 1),
    ],
)
def test_converted_save_directory_holds_plain_weights_and_tokenizer_files(
    tmp_path, conversion_model, gcs_bucket, monkeypatch, source_kind, destination_kind, process_index
):
    import eformer.paths as paths
    import huggingface_hub
    from easydel.infra import base_config
    from transformers import configuration_utils
    from transformers.generation import configuration_utils as generation_configuration_utils

    if source_kind == "gcs":
        checkpoint = gcs_bucket.root / "bucket" / "models" / "hf"
        source, source_options = "gs://bucket/models/hf", {}
    elif source_kind == "local":
        checkpoint = tmp_path / "hf"
        source, source_options = str(checkpoint), {"torch_load_mode": "streaming"}
    else:
        # Only revision "v1" exists, so every Hub read must carry the caller's revision.
        checkpoint = tmp_path / "hub" / "v1"
        source = "org/model"
        source_options = {"torch_load_mode": "streaming", "from_torch": True, "revision": "v1"}

        def download(repo_id, filename, *args, revision=None, subfolder="", **kwargs):
            assert repo_id == source
            path = tmp_path / "hub" / (revision or "main") / (subfolder or "") / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        for module in (base_config, configuration_utils, generation_configuration_utils):
            monkeypatch.setattr(module, "cached_file", download)
    expected, _ = _write_moe_checkpoint(checkpoint)
    for name, payload in _TOKENIZER_FILES.items():
        (checkpoint / name).write_bytes(payload)
    if destination_kind == "gcs":
        destination = "gs://bucket/out/converted"
        destination_view = gcs_bucket.root / "bucket" / "out" / "converted"
    else:
        destination_view = tmp_path / "out" / "converted"
        destination = str(destination_view)

    saves = []

    def save_pretrained(self, save_directory, **kwargs):
        # Stands in for the TensorStore save: persists the parameters the model holds at save time.
        saves.append(save_directory)
        flat = {key: np.asarray(value).tolist() for key, value in _converted_params(self).items()}
        (paths.ePath(str(save_directory)) / "saved_parameters.json").write_text(json.dumps(flat))

    def quantize(self, *, apply_quantization=False, **kwargs):
        # Runtime quantization visibly replaces every parameter.
        if apply_quantization:
            flat = bridge.flatten_dict(self.params)
            self.params = bridge.unflatten_dict({key: np.zeros(np.shape(v), dtype=np.int8) for key, v in flat.items()})
        return self

    conversion_model.save_pretrained = save_pretrained
    conversion_model.quantize = quantize
    _use_moe_transform(conversion_model)
    monkeypatch.setattr(bridge.jax, "process_index", lambda: process_index)

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        source,
        converted_save_directory=destination,
        apply_quantization=True,
        **_LOAD_OPTIONS,
        **source_options,
    )

    # One save, into the requested destination ...
    assert saves == [destination]
    # ... holding the plain converted weights: runtime quantization ran only afterwards.
    saved = json.loads((destination_view / "saved_parameters.json").read_text())
    assert set(saved) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(np.asarray(saved[key], dtype=np.float32), value)
    returned = _converted_params(model)
    assert set(returned) == set(expected)
    assert all(value.dtype == np.int8 and not value.any() for value in returned.values())
    # Tokenizer files present at the source are copied verbatim; absent ones are not created.
    copied = {path.name: path.read_bytes() for path in destination_view.iterdir()}
    copied.pop("saved_parameters.json")
    assert copied == (_TOKENIZER_FILES if process_index == 0 else {})


@pytest.mark.parametrize("case", ["native_checkpoint", "from_torch_false", "file_destination"])
def test_converted_save_directory_rejects_requests_it_cannot_honor(tmp_path, case):
    source = tmp_path / "source"
    source.mkdir()
    MixtralConfig(tie_word_embeddings=False).save_pretrained(source)
    destination = tmp_path / "converted"
    options = {}
    if case == "native_checkpoint":
        # An EasyDeL checkpoint selects the native loader, which converts nothing.
        (source / bridge.TENSORSTORE_INDEX_NAME).write_text("{}")
    elif case == "from_torch_false":
        options["from_torch"] = False
    else:
        destination.write_text("occupied")

    with pytest.raises(ValueError, match="converted_save_directory"):
        AutoEasyDeLModelForCausalLM.from_pretrained(
            str(source), converted_save_directory=str(destination), local_files_only=True, **options
        )

    # Refused before anything was loaded or written.
    if case == "file_destination":
        assert destination.read_text() == "occupied"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("failing_object", [bridge.CONFIG_NAME, bridge.SAFE_WEIGHTS_INDEX_NAME])
@pytest.mark.parametrize("status", ["403", "503"])
def test_gcs_read_failure_propagates_and_leaves_nothing_staged(
    tmp_path, conversion_model, gcs_bucket, monkeypatch, failing_object, status
):
    import tempfile

    from google.api_core import exceptions

    _write_moe_checkpoint(gcs_bucket.root / "bucket" / "models" / "hf")
    _use_moe_transform(conversion_model)
    error = exceptions.Forbidden("denied") if status == "403" else exceptions.ServiceUnavailable("unavailable")
    # The object exists but cannot be read. Only a 404 may mean "absent": a missing index falls through to
    # model.safetensors, which this sharded checkpoint lacks, so a misread failure surfaced as "not found".
    gcs_bucket.errors[f"models/hf/{failing_object}"] = error
    system_tmp = tmp_path / "system_tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    staging = tmp_path / "staging"

    with pytest.raises(type(error)) as raised:
        AutoEasyDeLModelForCausalLM.from_pretrained(
            "gs://bucket/models/hf", torch_streaming_tmp_dir=str(staging), **_LOAD_OPTIONS
        )

    # The storage error itself, not a FileNotFoundError, and no staged index, shard or partial file is left.
    assert raised.value is error
    assert not staging.exists() or list(staging.iterdir()) == []
    if failing_object == bridge.CONFIG_NAME:
        # The metadata stager removed its temp dir before the error left it.
        assert list(system_tmp.iterdir()) == []


def test_gcs_checkpoint_missing_a_shard_fails_clearly_and_cleans_up(tmp_path, conversion_model, gcs_bucket):
    checkpoint = gcs_bucket.root / "bucket" / "models" / "hf"
    _, weight_files = _write_moe_checkpoint(checkpoint)
    # A partial upload: the index lists a shard the bucket does not hold (a genuine 404).
    (checkpoint / weight_files[1]).unlink()
    _use_moe_transform(conversion_model)
    staging = tmp_path / "staging"

    with pytest.raises(FileNotFoundError, match="checkpoint is incomplete"):
        AutoEasyDeLModelForCausalLM.from_pretrained(
            "gs://bucket/models/hf", torch_streaming_tmp_dir=str(staging), **_LOAD_OPTIONS
        )

    # The shard staged before the failure is gone too.
    assert list(staging.iterdir()) == []


def test_gcs_fetch_maps_only_404_to_file_not_found_and_drops_partial_files(tmp_path, gcs_bucket, monkeypatch):
    import shutil
    import subprocess

    from google.api_core import exceptions

    objects = gcs_bucket.root / "bucket" / "models"
    objects.mkdir(parents=True)
    (objects / "shard.safetensors").write_bytes(b"0123456789")
    staged = tmp_path / "staged"
    staged.mkdir()

    # 404 is the one failure that means "absent".
    with pytest.raises(FileNotFoundError):
        bridge._fetch_gcs_object("gs://bucket/models/missing.safetensors", str(staged / "missing"), use_cli=False)

    # The CLI transfer dies midway and leaves a partial file; the ePath fallback is then denied (403).
    denied = exceptions.Forbidden("denied")
    gcs_bucket.errors["models/shard.safetensors"] = denied
    target = staged / "shard.safetensors"

    def interrupted_cp(argv, **kwargs):
        target.write_bytes(b"01234")
        return subprocess.CompletedProcess(argv, 1, b"", b"connection reset")

    monkeypatch.setattr(shutil, "which", lambda cmd, *args, **kwargs: "gcloud" if cmd == "gcloud" else None)
    monkeypatch.setattr(subprocess, "run", interrupted_cp)
    with pytest.raises(exceptions.Forbidden) as raised:
        bridge._fetch_gcs_object("gs://bucket/models/shard.safetensors", str(target))

    # Propagated unchanged, and no truncated copy is left behind to be read as the shard.
    assert raised.value is denied
    assert list(staged.iterdir()) == []
