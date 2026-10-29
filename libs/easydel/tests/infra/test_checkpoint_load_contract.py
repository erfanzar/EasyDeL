"""Tiny accelerator regressions for streaming quantization and strict checkpoint compatibility."""

from functools import wraps

import easydel as ed
import pytest
import spectrax as spx
from easydel.infra.mixins import bridge
from easydel.layers.quantization import QuantizationConfig
from easydel.utils import set_inference_mode
from jax import numpy as jnp


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
    )
    config.sharding_axis_dims = (1, 1, 1, 1, -1, 1)
    config.use_qmm_best_config = False
    with set_inference_mode():
        model = ed.LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.save_pretrained(tmp_path)
    return tmp_path


def _load(path, **kwargs):
    with set_inference_mode():
        return ed.LlamaForCausalLM.from_pretrained(
            str(path),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            sharding_axis_dims=(1, 1, 1, 1, -1, 1),
            auto_shard_model=False,
            verbose=False,
            **kwargs,
        )


def test_affine_quantization_replaces_dense_kernel_before_loader_returns(tiny_checkpoint, monkeypatch):
    original_load = bridge.Checkpointer.load_pytree
    streamed_dtypes = []

    @wraps(original_load)
    def observe_stream(self, *args, **kwargs):
        callback = kwargs["callback"]

        def observe_tensor(value, path):
            result = callback(value, path)
            name = ".".join(map(str, path)) if not isinstance(path, str) else path
            if name.endswith("o_proj.weight"):
                # Assert the result at the deserialization boundary, not a post-load
                # quantized model that could hide a dense streaming fallback.
                streamed_dtypes.append(result.dtype)
            return result

        kwargs["callback"] = observe_tensor
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(bridge.Checkpointer, "load_pytree", observe_stream)
    loaded = _load(
        tiny_checkpoint,
        apply_quantization=True,
        quantization_config=QuantizationConfig(dtype="int8", group_size=16, pattern=r".*o_proj$"),
        checkpoint_load_chunk_size=1,
    )
    assert streamed_dtypes
    assert all(jnp.issubdtype(dtype, jnp.integer) for dtype in streamed_dtypes)
    loaded.assert_parameters_materialized()


@pytest.mark.parametrize("mismatch", ["unexpected", "missing", "shape"])
def test_strict_loading_rejects_incompatible_checkpoint(tiny_checkpoint, monkeypatch, mismatch):
    original_load = bridge.Checkpointer.load_pytree

    @wraps(original_load)
    def corrupt_checkpoint(self, *args, **kwargs):
        state, extras = original_load(self, *args, **kwargs)
        state = dict(state)
        key = next(key for key in state if "o_proj" in str(key) and "weight" in str(key))
        if mismatch == "unexpected":
            extra_key = ("parameters", "extra", "weight") if isinstance(key, tuple) else "parameters.extra.weight"
            state[extra_key] = jnp.zeros((1,), dtype=jnp.float32)
        elif mismatch == "missing":
            state.pop(key)
        else:
            state[key] = jnp.zeros((1, 1), dtype=jnp.float32)
        return state, extras

    monkeypatch.setattr(bridge.Checkpointer, "load_pytree", corrupt_checkpoint)
    with pytest.raises(ValueError, match="mismatch_allowed=False"):
        _load(tiny_checkpoint, mismatch_allowed=False)


def test_permissive_loading_still_accepts_extra_checkpoint_leaves(tiny_checkpoint, monkeypatch):
    original_load = bridge.Checkpointer.load_pytree

    @wraps(original_load)
    def add_extra_leaf(self, *args, **kwargs):
        state, extras = original_load(self, *args, **kwargs)
        return {**state, "parameters.unused.weight": jnp.zeros((1,))}, extras

    monkeypatch.setattr(bridge.Checkpointer, "load_pytree", add_extra_leaf)
    loaded = _load(tiny_checkpoint, mismatch_allowed=True)
    assert not hasattr(loaded, "unused")
    assert "unused" not in str(spx.export(loaded)[1].raw())
