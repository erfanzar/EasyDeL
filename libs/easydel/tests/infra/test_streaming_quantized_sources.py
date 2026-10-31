"""Accelerator regressions for original-key FP8/MXFP4 decoding before fusion."""

import json
from functools import partial

import numpy as np
import pytest
from easydel.infra.mixins import bridge
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForCausalLM
from transformers import MixtralConfig

from .test_public_conversion_paths import _fuse, _normalizer, _unfuse
from .test_public_conversion_paths import conversion_model as conversion_model_fixture  # noqa: F401


@pytest.mark.parametrize("loader", ["streaming", "sequential"])
@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
@pytest.mark.parametrize("fusion", [False, True])
def test_quantized_sources_decode_across_shards_and_preserve_learned_scales(
    tmp_path,
    conversion_model_fixture,  # noqa: F811
    loader,
    method,
    fusion,
):
    import tensorstore as ts
    import torch
    from jax.experimental.array_serialization import serialization as jax_ser

    conversion_model = conversion_model_fixture
    source = tmp_path / "source"
    source.mkdir()
    MixtralConfig(tie_word_embeddings=False, quantization_config={"quant_method": method}).save_pretrained(source)
    names = ["gate_proj", "up_proj"] if fusion else ["proj"]
    sources = {}
    scales = {}
    for index, name in enumerate(names):
        stem = f"vendor.model.{name}"
        if method == "fp8":
            sources[f"{stem}.weight"] = torch.full((2, 64), index + 1, dtype=torch.float8_e4m3fn)
            scales[f"{stem}.weight_scale"] = torch.full((2, 1), 2.0)
        else:
            # E2M1 nibble 2 is exactly 1; E8M0 byte 128 is exactly 2.
            sources[f"{stem}.weight"] = torch.full((2, 32), 0x22, dtype=torch.uint8)
            scales[f"{stem}.weight_scale"] = torch.full((2, 2), 128 + index, dtype=torch.uint8)
    sources["vendor.model.attn_hc.scale"] = torch.tensor([3.0, 7.0])
    sources["mtp.0.weight"] = torch.ones((1, 1))
    weight_map = {key: "weights.bin" for key in sources} | {key: "scales.bin" for key in scales}
    torch.save(sources, source / "weights.bin")
    torch.save(scales, source / "scales.bin")
    (source / bridge.WEIGHTS_INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    reform = {}
    if fusion:
        reform["model.gate_up_proj.weight$"] = {
            "sources": ["model.gate_proj.weight", "model.up_proj.weight"],
            "fuser": _fuse,
            "inverse_fuser": _unfuse,
            "already_converted": True,
        }
    conversion_model.pure_transform_fn = staticmethod(
        partial(dict, checkpoint_key_normalizer=_normalizer, reform_param=reform)
    )
    expected = {
        f"model.{name}.weight": np.full((64, 2), 2.0 * (index + 1), dtype=np.float32) for index, name in enumerate(names)
    }
    if fusion:
        expected = {"model.gate_up_proj.weight": np.concatenate(list(expected.values()), axis=-1)}
    expected["model.attn_hc.scale"] = np.array([3.0, 7.0], dtype=np.float32)
    conversion_model.parameters_shape = expected
    common = dict(
        pretrained_model_name_or_path=str(source),
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
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
