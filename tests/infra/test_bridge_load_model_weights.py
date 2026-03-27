import jax
import numpy as np
from jax import numpy as jnp

from easydel.infra.mixins.bridge import EasyBridgeMixin


class _LoadConfigStub:
    quantization_config = None


class _LoadModelStub:
    def __init__(self):
        self.config = _LoadConfigStub()
        self.graphtree_params_shape = {"layer": {"weight": jax.ShapeDtypeStruct((2, 2), jnp.float32)}}

    def _get_partition_rules(self, _):
        return None


def test_load_model_weights_normalizes_numpy_arrays_before_merge(monkeypatch):
    device_put_calls: list[tuple[object, object]] = []
    merge_calls: list[tuple[dict[str, object], bool]] = []

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def load_pytree(self, **kwargs):
            return {"params": {"layer": {"weight": np.ones((2, 2), dtype=np.float32)}}}, None

    cpu_device = object()

    def _fake_device_put(value, device):
        device_put_calls.append((value, device))
        return jnp.asarray(value)

    def _fake_merge_model_and_tree(*, model, tree, silence):
        merge_calls.append((tree, silence))
        return model

    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)
    monkeypatch.setattr(
        "easydel.infra.mixins.bridge.jax.devices",
        lambda platform=None: [cpu_device] if platform == "cpu" else [],
    )
    monkeypatch.setattr("easydel.infra.mixins.bridge.jax.device_put", _fake_device_put)
    monkeypatch.setattr("easydel.infra.mixins.bridge.merge_model_and_tree", _fake_merge_model_and_tree)

    model = _LoadModelStub()
    loaded = EasyBridgeMixin._load_model_weights(
        resolved_archive_file="/tmp/fake-checkpoint",
        model=model,
        param_dtype=jnp.float32,
        mesh="mesh",
        shard_fns=None,
        quantization_config=None,
        apply_quantization=False,
        verbose=False,
    )

    assert loaded is model
    assert len(device_put_calls) == 1
    assert isinstance(device_put_calls[0][0], np.ndarray)
    assert device_put_calls[0][1] is cpu_device
    assert len(merge_calls) == 1
    assert isinstance(merge_calls[0][0]["layer"]["weight"], jax.Array)
    assert merge_calls[0][1] is False
