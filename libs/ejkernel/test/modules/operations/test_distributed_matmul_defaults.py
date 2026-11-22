from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax_sharding = importlib.import_module("jax.sharding")
Mesh = jax_sharding.Mesh
PartitionSpec = jax_sharding.PartitionSpec

ag_mod = importlib.import_module("ejkernel.modules.operations.all_gather_matmul")
rs_mod = importlib.import_module("ejkernel.modules.operations.reduce_scatter_matmul")
configs_mod = importlib.import_module("ejkernel.modules.operations.configs")
AllGatherMatmulConfig = configs_mod.AllGatherMatmulConfig
ReduceScatterMatmulConfig = configs_mod.ReduceScatterMatmulConfig


def _single_device_mesh() -> Mesh:
    devices = np.array(jax.devices()[:1])
    return Mesh(devices, axis_names=("tp",))


@pytest.mark.parametrize("rhs_transpose", [False, True])
def test_all_gather_mesh_defaults_specs(monkeypatch, rhs_transpose: bool):
    captured: dict[str, object] = {}

    def _fake_executor(_kernel, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(ag_mod, "_all_gather_matmul_executor", _fake_executor)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    if rhs_transpose:
        y = jnp.ones((3, 4), dtype=jnp.float32)
        expected_in_specs = (PartitionSpec("tp", None), PartitionSpec("tp", None))
    else:
        y = jnp.ones((4, 3), dtype=jnp.float32)
        expected_in_specs = (PartitionSpec("tp", None), PartitionSpec(None, "tp"))

    _ = ag_mod.all_gather_matmul(
        x,
        y,
        "tp",
        rhs_transpose=rhs_transpose,
        mesh=_single_device_mesh(),
    )

    assert captured["method"] == "shard_map"
    assert captured["in_specs"] == expected_in_specs
    assert captured["out_specs"] == PartitionSpec(None, "tp")


def test_reduce_scatter_mesh_defaults_specs(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_executor(_kernel, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(rs_mod, "_reduce_scatter_matmul_executor", _fake_executor)

    x = jnp.ones((4, 3), dtype=jnp.float32)
    y = jnp.ones((2, 3), dtype=jnp.float32)

    _ = rs_mod.reduce_scatter_matmul(
        x,
        y,
        "tp",
        mesh=_single_device_mesh(),
    )

    assert captured["method"] == "shard_map"
    assert captured["in_specs"] == (PartitionSpec(None, "tp"), PartitionSpec(None, "tp"))
    assert captured["out_specs"] == PartitionSpec("tp", None)


def test_reduce_scatter_run_resolves_tp_size_when_omitted(monkeypatch):
    kernel = rs_mod.ReduceScatterMatmul()
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs):
        captured.update(kwargs)
        return jnp.zeros((1, 1), dtype=jnp.float32)

    monkeypatch.setattr(kernel, "get_impl", lambda _cfg: _fake_impl)

    world = int(jax.device_count())
    x = jnp.ones((max(2, world), 8), dtype=jnp.float32)
    y = jnp.ones((4, 8), dtype=jnp.float32)

    _ = kernel.run(
        x=x,
        y=y,
        axis_name="tp",
        tp_size=None,
        cfg=ReduceScatterMatmulConfig(platform="xla"),
    )

    assert captured["tp_size"] == world


def test_reduce_scatter_run_prefers_global_device_count_over_local(monkeypatch):
    kernel = rs_mod.ReduceScatterMatmul()
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs):
        captured.update(kwargs)
        return jnp.zeros((1, 1), dtype=jnp.float32)

    monkeypatch.setattr(kernel, "get_impl", lambda _cfg: _fake_impl)
    monkeypatch.setattr(rs_mod.jax, "device_count", lambda: 4)
    monkeypatch.setattr(rs_mod.jax, "local_device_count", lambda: 1)

    x = jnp.ones((4, 8), dtype=jnp.float32)
    y = jnp.ones((4, 8), dtype=jnp.float32)

    _ = kernel.run(
        x=x,
        y=y,
        axis_name="tp",
        tp_size=None,
        cfg=ReduceScatterMatmulConfig(platform="xla"),
    )

    assert captured["tp_size"] == 4


@pytest.mark.parametrize(
    ("x_shape", "y_shape", "rhs_transpose", "cfg_block_n", "cfg_block_k", "expected_bn", "expected_bk"),
    [
        ((64, 96), (96, 80), False, 128, 128, 16, 32),
        ((64, 128), (72, 128), True, 128, 128, 8, 128),
    ],
)
def test_all_gather_run_clamps_block_sizes(
    monkeypatch,
    x_shape: tuple[int, int],
    y_shape: tuple[int, int],
    rhs_transpose: bool,
    cfg_block_n: int,
    cfg_block_k: int,
    expected_bn: int,
    expected_bk: int,
):
    kernel = ag_mod.AllGatherMatmul()
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs):
        captured.update(kwargs)
        m = int(kwargs["x"].shape[0]) * int(kwargs["tp_size"])
        n_local = int(kwargs["y"].shape[0] if kwargs["rhs_transpose"] else kwargs["y"].shape[1])
        return jnp.zeros((m, n_local), dtype=kwargs["x"].dtype)

    monkeypatch.setattr(kernel, "get_impl", lambda _cfg: _fake_impl)
    x = jnp.ones(x_shape, dtype=jnp.float32)
    y = jnp.ones(y_shape, dtype=jnp.float32)
    cfg = AllGatherMatmulConfig(block_n=cfg_block_n, block_k=cfg_block_k, platform="xla")

    _ = kernel.run(
        x=x,
        y=y,
        axis_name="tp",
        rhs_transpose=rhs_transpose,
        tp_size=2,
        cfg=cfg,
    )

    assert captured["bn"] == expected_bn
    assert captured["bk"] == expected_bk
    assert captured["tp_size"] == 2


def test_reduce_scatter_run_clamps_block_sizes(monkeypatch):
    kernel = rs_mod.ReduceScatterMatmul()
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs):
        captured.update(kwargs)
        m_total = int(kwargs["x"].shape[0])
        m_local = m_total // int(kwargs["tp_size"])
        n = int(kwargs["y"].shape[0])
        return jnp.zeros((m_local, n), dtype=kwargs["x"].dtype)

    monkeypatch.setattr(kernel, "get_impl", lambda _cfg: _fake_impl)

    x = jnp.ones((256, 96), dtype=jnp.float32)
    y = jnp.ones((80, 96), dtype=jnp.float32)
    cfg = ReduceScatterMatmulConfig(block_m=128, block_n=128, block_k=128, platform="xla")

    _ = kernel.run(
        x=x,
        y=y,
        axis_name="tp",
        tp_size=4,
        cfg=cfg,
    )

    assert captured["bm"] == 32
    assert captured["bn"] == 16
    assert captured["bk"] == 32
    assert captured["tp_size"] == 4


def test_all_gather_run_platform_override_sets_xla_backend_any(monkeypatch):
    kernel = ag_mod.AllGatherMatmul()
    captured_cfg: dict[str, object] = {}

    def _fake_get_impl(cfg):
        captured_cfg["cfg"] = cfg

        def _fake_impl(**kwargs):
            m = int(kwargs["x"].shape[0]) * int(kwargs["tp_size"])
            n_local = int(kwargs["y"].shape[1])
            return jnp.zeros((m, n_local), dtype=kwargs["x"].dtype)

        return _fake_impl

    monkeypatch.setattr(kernel, "get_impl", _fake_get_impl)

    x = jnp.ones((32, 64), dtype=jnp.float32)
    y = jnp.ones((64, 32), dtype=jnp.float32)
    cfg = AllGatherMatmulConfig(block_n=128, block_k=128, platform="pallas", backend="tpu")

    _ = kernel.run(
        x=x,
        y=y,
        axis_name="tp",
        tp_size=2,
        platform="xla",
        cfg=cfg,
    )

    rewritten = captured_cfg["cfg"]
    assert rewritten.platform == "xla"
    assert rewritten.backend == "any"


def test_all_gather_shard_map_wrapper_rejects_mismatched_tp_size():
    class _FakeMesh:
        def __init__(self, size: int):
            self.shape = {"tp": size}

    kernel = ag_mod.AllGatherMatmul()
    x = jnp.ones((2, 4), dtype=jnp.float32)
    y = jnp.ones((4, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="must match mesh axis"):
        _ = kernel.create_shard_map_wrapper(
            x=x,
            y=y,
            axis_name="tp",
            tp_size=1,
            cfg=AllGatherMatmulConfig(platform="xla"),
            mesh=_FakeMesh(2),
        )


def test_reduce_scatter_shard_map_wrapper_rejects_mismatched_tp_size():
    class _FakeMesh:
        def __init__(self, size: int):
            self.shape = {"tp": size}

    kernel = rs_mod.ReduceScatterMatmul()
    x = jnp.ones((4, 3), dtype=jnp.float32)
    y = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="must match mesh axis"):
        _ = kernel.create_shard_map_wrapper(
            x=x,
            y=y,
            axis_name="tp",
            tp_size=1,
            cfg=ReduceScatterMatmulConfig(platform="xla"),
            mesh=_FakeMesh(2),
        )
