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

"""``load_deepseek_v41_dspark``: native ``mtp.*`` shards -> DSpark draft model (computation test).

Random DSpark parameters are mapped to native ``mtp.N.*`` tensors by the reference oracle's own mapping
(``tests/modules/_deepseek_v41_reference.py``, independent of the loader), written to a safetensors shard and loaded
back. Checks: every loaded leaf equals the oracle value at tp=1; greedy ids / logits / confidence of two decode steps
match DeepSeek's reference ``forward_spec`` at tp=1 and tp=4 (tp=4 exercises the shared-expert ``gate_up`` interleave
and the expert-over-tp placement); a shard missing a tensor is rejected.
"""

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v41 import DeepseekV41Config, load_deepseek_v41_dspark

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

RTOL = ATOL = 1e-3
PREFILL = ref.DSPARK_PREFILL_LENS[1]  # ring wraps at seeding


def _config(tp_size: int) -> DeepseekV41Config:
    kwargs = ref.dspark_config_kwargs()
    kwargs["sharding_axis_dims"] = (1, 1, 1, 1, tp_size, 1)
    config = DeepseekV41Config(**kwargs)
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


class Oracle(tp.NamedTuple):
    backbone: dict[str, np.ndarray]
    dspark: dict[str, np.ndarray]
    inputs: ref.DSparkInputs
    outputs: list[ref.DSparkOutputs]


@functools.cache
def _oracle() -> Oracle:
    config = _config(1)
    backbone = ref.random_easydel_params(config, ref.CASE_SEEDS["plain"])
    dspark = ref.random_dspark_params(config, ref.DSPARK_SEED)
    ref_model = ref.build_reference(config, dspark=True)
    ref.load_easydel_params(ref_model, backbone)
    ref.load_dspark_params(ref_model, dspark)
    inputs = ref.dspark_inputs(config, PREFILL)
    return Oracle(backbone, dspark, inputs, ref.reference_dspark(ref_model, inputs))


def _native_tensors(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    native: dict[str, np.ndarray] = {}
    for name in sorted(params):
        for native_name, value in ref.dspark_to_reference(name, params[name]):
            assert native_name not in native, native_name
            native[native_name] = np.ascontiguousarray(value, dtype=np.float32)
    return native


@pytest.fixture(scope="module")
def shard(tmp_path_factory) -> str:
    from safetensors.numpy import save_file

    path = tmp_path_factory.mktemp("dspark") / "mtp.safetensors"
    save_file(_native_tensors(_oracle().dspark), str(path))
    return str(path)


def _load(config: DeepseekV41Config, paths: list[str]):
    return load_deepseek_v41_dspark(
        config, paths, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST
    )


def _run(config: DeepseekV41Config, model, oracle: Oracle) -> list[ref.DSparkOutputs]:
    embed_table = jnp.asarray(oracle.backbone["model.embed_tokens.weight"])
    head_kernel = jnp.asarray(oracle.backbone["lm_head.weight"])

    def embed(ids):
        return embed_table[ids]

    def lm_head(hidden):
        return jnp.einsum("...d,dv->...v", hidden.astype(jnp.float32), head_kernel, precision=jax.lax.Precision.HIGHEST)

    graphdef, state = spx.export(model)

    @jax.jit
    def step(state, tokens, hidden, draft_state):
        return spx.bind(graphdef, state).draft(tokens, hidden, draft_state, embed, lm_head)

    outputs = []
    with config.mesh:
        draft_state = model.prefill(jnp.asarray(oracle.inputs.prefill_hidden))
        for tokens, hidden in zip(oracle.inputs.tokens, oracle.inputs.step_hidden, strict=True):
            ids, logits, confidence, draft_state = step(
                state, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(hidden), draft_state
            )
            outputs.append(
                ref.DSparkOutputs(np.asarray(ids).astype(np.int64), np.asarray(logits), np.asarray(confidence))
            )
    return outputs


def test_loaded_parameters_equal_oracle_at_tp1(shard):
    oracle = _oracle()
    model = _load(_config(1), [shard])
    _, state = spx.export(model)
    seen = set()
    for _collection, path, leaf in state.items():
        value = np.asarray(leaf.value if hasattr(leaf, "value") else leaf)
        assert path in oracle.dspark, f"loaded leaf {path} unknown to the oracle"
        np.testing.assert_array_equal(value, oracle.dspark[path], err_msg=path)
        seen.add(path)
    assert seen == set(oracle.dspark)


@pytest.mark.parametrize("tp_size", [1, 4])
def test_loaded_draft_matches_reference(shard, tp_size):
    if jax.device_count() < tp_size:
        pytest.skip(f"needs {tp_size} devices")
    oracle = _oracle()
    config = _config(tp_size)
    outputs = _run(config, _load(config, [shard]), oracle)
    for step, (got, want) in enumerate(zip(outputs, oracle.outputs, strict=True)):
        np.testing.assert_array_equal(got.ids, want.ids, err_msg=f"step {step}")
        np.testing.assert_allclose(got.logits, want.logits, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")
        np.testing.assert_allclose(got.confidence, want.confidence, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")


def test_missing_tensor_is_rejected(tmp_path):
    from safetensors.numpy import save_file

    native = _native_tensors(_oracle().dspark)
    dropped = next(k for k in sorted(native) if k.endswith("attn.wq_a.weight"))
    del native[dropped]
    path = tmp_path / "partial.safetensors"
    save_file(native, str(path))
    with pytest.raises(ValueError, match="missing"):
        _load(_config(1), [str(path)])
