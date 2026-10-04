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

"""DeepSeek-V4.1 DSpark main-hidden hook (``output_dspark_hidden``).

The hook returns, for every ``dspark_target_layer_ids`` entry, ``hidden_streams.mean(axis=2)`` taken after that
layer's engram and before its attention, concatenated in ascending layer order. Checks:

- parity with the reference ``Transformer.forward`` ``main_hidden`` (fp32, ``precision=HIGHEST``);
- equality with a manual capture built from ``output_hidden_states`` plus the layer's own engram (proves the capture
  point is AFTER the engram);
- cached prefill + one-token decode reproduce the stateless capture;
- the default (flag off) leaves logits unchanged and returns no ``main_hidden``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
import torch
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM
from easydel.modules.deepseek_v41._engram_hash import EngramHasher

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

RTOL = ATOL = 1e-4
DECODE_ATOL = 5e-4


def _make_config(case: str, targets: tuple[int, ...] | None = None) -> DeepseekV41Config:
    kwargs = ref.dspark_config_kwargs() if case == "plain" else ref.case_config_kwargs(case)
    if targets is not None:
        kwargs["dspark_target_layer_ids"] = list(targets)
    config = DeepseekV41Config(**kwargs)
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _build_easydel(config: DeepseekV41Config, params: dict[str, np.ndarray]):
    """fp32 / HIGHEST EasyDeL model whose every parameter is overwritten with ``params``."""
    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        graphdef, state = spx.export(model)
        for collection, path, leaf in list(state.items()):
            if path not in params:
                continue
            old = _leaf_array(leaf)
            assert tuple(old.shape) == params[path].shape, (path, old.shape, params[path].shape)
            new = jnp.asarray(params[path], dtype=jnp.float32)
            sharding = getattr(old, "sharding", None)
            if sharding is not None:
                new = jax.device_put(new, sharding)
            state.set(collection, path, new)
        model = spx.bind(graphdef, state)
        model.eval()
    return model


@functools.cache
def _plain_oracle():
    config = _make_config("plain")  # targets (6, 8)
    params = ref.random_easydel_params(config, ref.CASE_SEEDS["plain"])
    ids = ref.case_input_ids(config, "plain")
    ref_model = ref.build_reference(config, dspark=True)
    ref.load_easydel_params(ref_model, params)
    ref._reset_state(ref_model, ref.load_reference_module())
    _, _, main_hidden = ref_model(torch.as_tensor(ids, dtype=torch.long), 0)
    return config, params, ids, main_hidden.detach().float().numpy()


def test_main_hidden_matches_reference():
    """``main_hidden`` equals the reference ``Transformer.forward`` ``main_hidden`` (targets 6 and 8)."""
    config, params, ids, expected = _plain_oracle()
    model = _build_easydel(config, params)
    with config.mesh:
        out = model(input_ids=jnp.asarray(ids, dtype=jnp.int32), output_dspark_hidden=True)
        got = np.asarray(out.main_hidden)
    width = len(config.dspark_target_layer_ids) * config.hidden_size
    assert got.shape == expected.shape == (ref.CASE_BATCH, ref.CASE_SEQ_LEN["plain"], width)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_main_hidden_matches_manual_capture_after_engram():
    """Capture point is after the layer's engram: rebuild it from ``hidden_states`` + the layer's own engram."""
    targets = (1, 5, 8)  # layers 1 and 5 carry an engram, layer 8 does not
    config = _make_config("engram", targets)
    params = ref.random_easydel_params(config, ref.CASE_SEEDS["engram"])
    ids = ref.case_input_ids(config, "engram")
    hashes = jnp.asarray(
        EngramHasher.from_config(config, np.arange(ref.VOCAB, dtype=np.int64)).hash(ids), dtype=jnp.int32
    )
    model = _build_easydel(config, params)
    layers = list(model.model.layers)
    with config.mesh:
        out = model(
            input_ids=jnp.asarray(ids, dtype=jnp.int32),
            engram_hash_ids=hashes,
            output_hidden_states=True,
            output_dspark_hidden=True,
        )
        manual, before_engram = [], []
        for i in targets:
            streams = out.hidden_states[i]  # streams entering layer i, before its engram
            before_engram.append(np.asarray(streams.mean(axis=2)))
            layer = layers[i]
            if layer.engram is not None:
                streams = layer.engram(streams, hashes[:, :, layer.engram.layer_hash_index], None)
            manual.append(streams.mean(axis=2))
        manual = np.asarray(jnp.concatenate(manual, axis=-1))
        got = np.asarray(out.main_hidden)
    np.testing.assert_allclose(got, manual, rtol=RTOL, atol=ATOL)
    hidden = config.hidden_size
    for slot, i in enumerate(targets):
        part = slice(slot * hidden, (slot + 1) * hidden)
        gap = float(np.abs(got[..., part] - before_engram[slot]).max())
        if layers[i].engram is not None:
            assert gap > 1e-5, f"layer {i}: capture equals the pre-engram streams"
        else:
            assert gap < 1e-6, f"layer {i} has no engram but the capture moved by {gap:.3e}"


def test_flag_off_is_unchanged_and_returns_nothing():
    """Default forward: ``main_hidden`` is ``None`` and the logits are bit-identical to the flag-on logits."""
    config, params, ids, _ = _plain_oracle()
    model = _build_easydel(config, params)
    batch = jnp.asarray(ids, dtype=jnp.int32)
    with config.mesh:
        off = model(input_ids=batch)
        on = model(input_ids=batch, output_dspark_hidden=True)
    assert off.main_hidden is None
    assert on.main_hidden is not None
    np.testing.assert_array_equal(np.asarray(off.logits), np.asarray(on.logits))


def test_cached_prefill_and_decode_reproduce_stateless_capture():
    """Prefill + single-token decode carry the capture through the cached paths (jitted)."""
    config, params, ids, _ = _plain_oracle()
    model = _build_easydel(config, params)
    batch_ids = jnp.asarray(ids, dtype=jnp.int32)
    batch, total = batch_ids.shape
    prefill_len = 13
    graphdef, state = spx.export(model)

    @jax.jit
    def cached_step(state, input_ids, cache):
        out = spx.bind(graphdef, state)(input_ids=input_ids, past_key_values=cache, output_dspark_hidden=True)
        return out.main_hidden, out.past_key_values

    with config.mesh:
        stateless = np.asarray(model(input_ids=batch_ids, output_dspark_hidden=True).main_hidden)
        cache = model.init_cache(batch_size=batch, max_length=total + 8)
        hidden, cache = cached_step(state, batch_ids[:, :prefill_len], cache)
        np.testing.assert_allclose(np.asarray(hidden), stateless[:, :prefill_len], rtol=RTOL, atol=DECODE_ATOL)
        for pos in range(prefill_len, total):
            hidden, cache = cached_step(state, batch_ids[:, pos : pos + 1], cache)
            assert hidden.shape == (batch, 1, stateless.shape[-1])
            np.testing.assert_allclose(np.asarray(hidden), stateless[:, pos : pos + 1], rtol=RTOL, atol=DECODE_ATOL)
