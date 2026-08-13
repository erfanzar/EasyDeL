# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Registry and dispatch contract for the fused-MLP / quantized-matmul ops.

Asserts what the ``add-ejkernel-kernel`` skill requires of every operation:
each op id resolves through ``kernel_registry`` for every intended
(platform, backend) pair, cross-implementation signatures match exactly, and
the public module-level wrapper produces the same numbers as the registered
XLA implementation it dispatches to.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules.operations import fused_mlp
from jax import numpy as jnp

from ._utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

QUANT_OP_IDS = [
    "fused_mlp",
    "fused_mlp_w4a4",
    "channelwise_quantized_matmul",
    "packed_int4_gemv",
    "w4a4_gemv",
]

EXPECTED_REGISTRATIONS = {
    "fused_mlp": [(Platform.XLA, Backend.ANY)],
    "fused_mlp_w4a4": [(Platform.XLA, Backend.ANY), (Platform.PALLAS, Backend.TPU)],
    "channelwise_quantized_matmul": [(Platform.XLA, Backend.ANY)],
    "packed_int4_gemv": [(Platform.XLA, Backend.ANY), (Platform.PALLAS, Backend.TPU)],
    "w4a4_gemv": [(Platform.XLA, Backend.ANY), (Platform.PALLAS, Backend.TPU)],
}


def _register_all() -> None:
    """Import every module that carries a registration decorator."""
    import ejkernel.kernels._pallas.tpu.fused_mlp
    import ejkernel.kernels._pallas.tpu.quantized_matmul
    import ejkernel.kernels._xla.fused_mlp
    import ejkernel.kernels._xla.quantized_matmul  # noqa: F401


@pytest.mark.parametrize("op_id", QUANT_OP_IDS)
def test_signatures_match_across_implementations(op_id):
    _register_all()
    assert kernel_registry.validate_signatures(op_id) is True


@pytest.mark.parametrize(
    ("op_id", "platform", "backend"),
    [(op, p, b) for op, pairs in EXPECTED_REGISTRATIONS.items() for p, b in pairs],
)
def test_registry_resolves_every_intended_pair(op_id, platform, backend):
    _register_all()
    impl = kernel_registry.get(op_id, platform=platform, backend=backend)
    assert callable(impl)


def test_xla_fallback_exists_for_every_op():
    """The mandatory Platform.XLA fallback (golden rule of the registry)."""
    _register_all()
    for op_id in QUANT_OP_IDS:
        impl = kernel_registry.get(op_id, platform=Platform.XLA, backend=Backend.ANY)
        assert callable(impl)


def _quantize_channelwise(w: np.ndarray, dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric per-output-channel quantization of ``[k, n]`` to int codes."""
    max_code = 7.0 if dtype == jnp.int4 else 127.0
    scale = np.abs(w).max(axis=0, keepdims=True) / max_code
    codes = np.clip(np.round(w / scale), -max_code, max_code)
    return jnp.asarray(codes, dtype), jnp.asarray(scale, jnp.float32)


class TestPublicWrapperDispatch:
    """The public ``fused_mlp`` matches the registered XLA impl it routes to."""

    def _inputs(self, seed=0, m=8, k=128, i=256):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
        w_gate = rng.normal(size=(k, i)).astype(np.float32)
        w_up = rng.normal(size=(k, i)).astype(np.float32)
        w_down = rng.normal(size=(i, k)).astype(np.float32)
        return x, w_gate, w_up, w_down

    def test_dense_matches_naive_composition_bitwise(self):
        x, w_gate, w_up, w_down = self._inputs()
        wg, wu, wd = (jnp.asarray(w, jnp.bfloat16) for w in (w_gate, w_up, w_down))
        got = fused_mlp(x, wg, wu, wd)
        want = (jax.nn.silu(x @ wg) * (x @ wu)) @ wd
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def test_int8_matches_registered_xla_impl(self):
        _register_all()
        x, w_gate, w_up, w_down = self._inputs(seed=1)
        gq, gs = _quantize_channelwise(w_gate, jnp.int8)
        uq, us = _quantize_channelwise(w_up, jnp.int8)
        dq, ds = _quantize_channelwise(w_down, jnp.int8)

        impl = kernel_registry.get("fused_mlp", platform=Platform.XLA, backend=Backend.ANY)
        want = impl(x, gq, uq, dq, gs, us, ds)
        got = fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds)
        assert_allclose(got, want, atol=0, rtol=0)

    def test_platform_xla_opt_out_accepted_and_equal_on_cpu(self):
        """platform='xla' is the vanilla-XLA opt-out; on CPU it is a no-op."""
        _register_all()
        x, w_gate, w_up, w_down = self._inputs(seed=3)
        gq, gs = _quantize_channelwise(w_gate, jnp.int8)
        uq, us = _quantize_channelwise(w_up, jnp.int8)
        dq, ds = _quantize_channelwise(w_down, jnp.int8)

        default = fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds)
        forced = fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds, platform="xla")
        np.testing.assert_array_equal(np.asarray(default, np.float32), np.asarray(forced, np.float32))

    def test_int8_frozen_gradients_flow_to_x_only(self):
        x, w_gate, w_up, w_down = self._inputs(seed=2)
        gq, gs = _quantize_channelwise(w_gate, jnp.int8)
        uq, us = _quantize_channelwise(w_up, jnp.int8)
        dq, ds = _quantize_channelwise(w_down, jnp.int8)

        def loss(x):
            out = fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        dx = jax.grad(loss)(x)
        assert dx.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(dx.astype(jnp.float32))))
        assert float(jnp.abs(dx.astype(jnp.float32)).max()) > 0.0

    def test_quantized_activation_bwd_linearizes_at_forward_preactivations(self):
        """With ``quantize_activations`` at prefill sizes the backward must
        recompute the pre-activations through the SAME int-dot regime the
        forward ran — dropping the knobs makes the adjoint linearize at
        fused-upcast pre-activations the forward never produced (~7e-3 rel
        drift on dx).

        The reference below re-derives dx from the documented gradient math
        (module docstring): pre-activations via the public channelwise op
        with the same knobs, cotangent-side scale fold, per-token int8
        quantization of the transposed products.
        """
        from ejkernel.modules import channelwise_quantized_matmul

        rng = np.random.default_rng(5)
        m, k, i, pt = 64, 128, 256, 32
        x = jnp.asarray(rng.normal(size=(m, k)), jnp.float32)
        gq, gs = _quantize_channelwise(rng.normal(size=(k, i)).astype(np.float32), jnp.int8)
        uq, us = _quantize_channelwise(rng.normal(size=(k, i)).astype(np.float32), jnp.int8)
        dq, ds = _quantize_channelwise(rng.normal(size=(i, k)).astype(np.float32), jnp.int8)

        def f(x):
            return fused_mlp(
                x,
                gq,
                uq,
                dq,
                gate_scale=gs,
                up_scale=us,
                down_scale=ds,
                quantize_activations=True,
                prefill_threshold=pt,
            )

        out, vjp = jax.vjp(f, x)
        gy = jnp.asarray(rng.normal(size=out.shape), out.dtype)
        (dx,) = vjp(gy)

        def tdot(cot, codes, scale):
            scaled = cot * scale.reshape(1, codes.shape[-1])
            c_abs = jnp.max(jnp.abs(scaled), axis=1, keepdims=True)
            c_scale = c_abs / 127.0
            c_q = jnp.clip(jnp.round(scaled / jnp.where(c_scale == 0, 1, c_scale)), -127, 127).astype(jnp.int8)
            prod = jax.lax.dot_general(
                c_q, codes, dimension_numbers=(((1,), (1,)), ((), ())), preferred_element_type=jnp.int32
            )
            return prod.astype(jnp.float32) * c_scale

        a = channelwise_quantized_matmul(x, gq, gs, quantize_activations=True, prefill_threshold=pt).astype(jnp.float32)
        b = channelwise_quantized_matmul(x, uq, us, quantize_activations=True, prefill_threshold=pt).astype(jnp.float32)
        assert m >= pt, "the test must exercise the int-dot prefill regime"
        _, combine_vjp = jax.vjp(lambda a, b: jax.nn.silu(a) * b, a, b)
        dh = tdot(gy.astype(jnp.float32), dq, ds)
        da, db = combine_vjp(dh)
        dx_ref = (tdot(da, gq, gs) + tdot(db, uq, us)).astype(x.dtype)

        rel = float(jnp.abs(dx - dx_ref).max() / (jnp.abs(dx_ref).max() + 1e-9))
        assert rel < 1e-4, f"bwd recompute diverged from the forward composition: rel {rel}"

    def test_packed_weights_without_scales_raises_clear_error(self):
        """A missing channel scale must fail eagerly on the host, not as an
        opaque AttributeError inside the (TPU-only) packed kernel."""
        x, w_gate, w_up, w_down = self._inputs(seed=4)
        gq, gs = _quantize_channelwise(w_gate, jnp.int4)
        uq, us = _quantize_channelwise(w_up, jnp.int4)
        dq, ds = _quantize_channelwise(w_down, jnp.int4)
        packed = (
            jnp.zeros((w_gate.shape[0] // 2, w_gate.shape[1]), jnp.uint8),
            jnp.zeros((w_up.shape[0] // 2, w_up.shape[1]), jnp.uint8),
            jnp.zeros((w_down.shape[0] // 2, w_down.shape[1]), jnp.uint8),
        )
        with pytest.raises(ValueError, match="down_scale"):
            fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, packed_weights=packed)
        with pytest.raises(ValueError, match=r"gate_scale.*up_scale.*down_scale"):
            fused_mlp(x, gq, uq, dq, packed_weights=packed)
        # platform="xla" documents that packed weights are ignored entirely,
        # so the scale requirement does not apply there.
        out = fused_mlp(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds, packed_weights=packed, platform="xla")
        assert out.shape == x.shape


class TestShardedRouting:
    """Sharding-aware dense routing: plan policy + sharded-path numerics."""

    def _mesh(self, shape):
        import numpy as _np
        from jax.sharding import Mesh

        devs = _np.array(jax.devices()[: shape[0] * shape[1]]).reshape(shape)
        return Mesh(devs, ("fsdp", "tp"))

    def _resolve(self, monkeypatch, x, wf, wd, mesh, specs, **kw):
        from ejkernel.modules.operations.fused_mlp import resolve_dense_sharded_plan

        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        return resolve_dense_sharded_plan(x, wf, wd, mesh=mesh, in_specs=specs, activation="silu", **kw)

    def test_rows_accepted(self, monkeypatch):
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        x = jnp.zeros((4096, 256), jnp.bfloat16)
        wf = jnp.zeros((256, 1024), jnp.bfloat16)
        wd = jnp.zeros((512, 256), jnp.bfloat16)
        plan = self._resolve(
            monkeypatch,
            x,
            wf,
            wd,
            mesh,
            (P("fsdp", None), P(None, None), P(None, None)),
        )
        assert plan is not None
        assert plan.tp_axes == ()
        assert plan.rows_axes == ("fsdp",)

    def test_tp_currently_routes_to_composition(self, monkeypatch):
        """tp measured a tie/loss at 4B — resolver rejects (never-slower)."""
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        x = jnp.zeros((4096, 256), jnp.bfloat16)
        wf = jnp.zeros((256, 1024), jnp.bfloat16)
        wd = jnp.zeros((512, 256), jnp.bfloat16)
        plan = self._resolve(
            monkeypatch,
            x,
            wf,
            wd,
            mesh,
            (P("fsdp", None), P(None, "tp"), P("tp", None)),
            layout="interleaved",
            segments=2,
        )
        assert plan is None

    def test_concat_layout_with_tp_sharding_rejected(self, monkeypatch):
        """Global-concat [gate|up] sharded on I is locally garbage — reject."""
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        x = jnp.zeros((4096, 256), jnp.bfloat16)
        wf = jnp.zeros((256, 1024), jnp.bfloat16)
        wd = jnp.zeros((512, 256), jnp.bfloat16)
        plan = self._resolve(monkeypatch, x, wf, wd, mesh, (P("fsdp", None), P(None, "tp"), P("tp", None)))
        assert plan is None

    def test_k_sharded_weights_rejected(self, monkeypatch):
        """fsdp storage sharding on K must fall back to the XLA composition."""
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        x = jnp.zeros((4096, 256), jnp.bfloat16)
        wf = jnp.zeros((256, 1024), jnp.bfloat16)
        wd = jnp.zeros((512, 256), jnp.bfloat16)
        plan = self._resolve(monkeypatch, x, wf, wd, mesh, (P("fsdp", None), P("fsdp", "tp"), P("tp", "fsdp")))
        assert plan is None

    def test_small_rows_rejected(self, monkeypatch):
        """Sub-prefill row counts stay on the XLA composition (decode roofline)."""
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        x = jnp.zeros((64, 256), jnp.bfloat16)
        wf = jnp.zeros((256, 1024), jnp.bfloat16)
        wd = jnp.zeros((512, 256), jnp.bfloat16)
        plan = self._resolve(monkeypatch, x, wf, wd, mesh, (P("fsdp", None), P(None, "tp"), P("tp", None)))
        assert plan is None

    def test_sharded_path_matches_composition_and_grads(self):
        """dp rows x tp columns: sharded path == composition, fwd and bwd."""
        from ejkernel.modules.operations.fused_mlp import _dense_pallas_sharded, _DenseShardedPlan
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        rng = np.random.default_rng(0)
        m, k, i, tp = 512, 128, 256, 2
        seg = i // tp
        x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
        gate = jnp.asarray(rng.normal(size=(k, i)) * 0.05, jnp.bfloat16)
        up = jnp.asarray(rng.normal(size=(k, i)) * 0.05, jnp.bfloat16)
        # TP-interleaved storage: rank r's column shard is [gate_r | up_r].
        wf = jnp.concatenate(
            [
                jnp.concatenate([gate[:, r * seg : (r + 1) * seg], up[:, r * seg : (r + 1) * seg]], axis=1)
                for r in range(tp)
            ],
            axis=1,
        )
        wd = jnp.asarray(rng.normal(size=(i, k)) * 0.05, jnp.bfloat16)
        x = jax.device_put(x, NamedSharding(mesh, P("fsdp", None)))
        wf = jax.device_put(wf, NamedSharding(mesh, P(None, "tp")))
        wd = jax.device_put(wd, NamedSharding(mesh, P("tp", None)))

        plan = _DenseShardedPlan(
            mesh=mesh,
            x_spec=P("fsdp", None),
            wf_spec=P(None, "tp"),
            wd_spec=P("tp", None),
            rows_axes=("fsdp",),
            tp_axes=("tp",),
            activation="silu",
            act_params=(),
            tile_m=128,
            tile_i=128,
            layout="interleaved",
            segments=tp,
        )

        def loss_pallas(x, wf, wd):
            return jnp.sum(_dense_pallas_sharded(plan, x, wf, wd).astype(jnp.float32) ** 2)

        def loss_ref(x, wf, wd):
            from ejkernel.kernels._xla.fused_mlp import split_gate_up

            g, u = split_gate_up(wf, layout="interleaved", segments=tp)
            out = (jax.nn.silu((x @ g).astype(jnp.float32)) * (x @ u).astype(jnp.float32)).astype(x.dtype) @ wd
            return jnp.sum(out.astype(jnp.float32) ** 2)

        lp, gp = jax.value_and_grad(loss_pallas, argnums=(0, 1, 2))(x, wf, wd)
        lr, gr = jax.value_and_grad(loss_ref, argnums=(0, 1, 2))(x, wf, wd)
        assert abs(float(lp) - float(lr)) / abs(float(lr)) < 2e-2
        for a, b in zip(gp, gr, strict=True):
            na, nb = np.asarray(a, np.float32), np.asarray(b, np.float32)
            cos = (na * nb).sum() / (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-9)
            assert cos > 0.99, f"grad cosine {cos}"

    def test_output_sharding_mirrors_input_never_replicated(self):
        """Default out spec keeps the input rows sharding; explicit out_spec wins."""
        from ejkernel.modules.operations.fused_mlp import _dense_pallas_sharded, _DenseShardedPlan
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        mesh = self._mesh((4, 2))
        rng = np.random.default_rng(1)
        m, k, i = 512, 128, 256
        x = jax.device_put(jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16), NamedSharding(mesh, P("fsdp", None)))
        wf = jax.device_put(
            jnp.asarray(rng.normal(size=(k, 2 * i)) * 0.05, jnp.bfloat16), NamedSharding(mesh, P(None, None))
        )
        wd = jax.device_put(
            jnp.asarray(rng.normal(size=(i, k)) * 0.05, jnp.bfloat16), NamedSharding(mesh, P(None, None))
        )

        def plan(out_spec=None):
            return _DenseShardedPlan(
                mesh=mesh,
                x_spec=P("fsdp", None),
                wf_spec=P(None, None),
                wd_spec=P(None, None),
                rows_axes=("fsdp",),
                tp_axes=(),
                activation="silu",
                act_params=(),
                tile_m=128,
                tile_i=128,
                layout="concat",
                segments=1,
                out_spec=out_spec,
            )

        out = jax.jit(lambda x, wf, wd: _dense_pallas_sharded(plan(), x, wf, wd))(x, wf, wd)
        assert out.sharding.spec == P("fsdp", None), f"output got {out.sharding.spec}, not input rows sharding"

        out2 = jax.jit(lambda x, wf, wd: _dense_pallas_sharded(plan(P(("fsdp", "tp"), None)), x, wf, wd))(x, wf, wd)
        got_rows = out2.sharding.spec[0] if len(out2.sharding.spec) else None
        assert got_rows == ("fsdp", "tp"), f"explicit out_spec ignored: {out2.sharding.spec}"


class TestParameterizedCombines:
    """Each parameterized combine through the public dense path vs hand math."""

    def _xu(self, seed=0, m=16, k=64, i=128):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
        wg = jnp.asarray(rng.normal(size=(k, i)) * 0.3, jnp.bfloat16)
        wu = jnp.asarray(rng.normal(size=(k, i)) * 0.3, jnp.bfloat16)
        wd = jnp.asarray(rng.normal(size=(i, k)) * 0.3, jnp.bfloat16)
        return x, wg, wu, wd

    def _check(self, activation, act_params, combine_np):
        x, wg, wu, wd = self._xu()
        got = np.asarray(fused_mlp(x, wg, wu, wd, activation=activation, act_params=act_params), np.float32)
        g = np.asarray(x, np.float32) @ np.asarray(wg, np.float32)
        u = np.asarray(x, np.float32) @ np.asarray(wu, np.float32)
        want = combine_np(g, u) @ np.asarray(wd, np.float32)
        rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 5e-2, f"{activation}{act_params}: relerr {rel}"

    def test_clamped_swiglu(self):
        def ref(g, u):
            g = np.minimum(g, 2.0)
            u = np.clip(u, -2.0, 2.0)
            return (g / (1 + np.exp(-g))) * u

        self._check("clamped_swiglu", (2.0,), ref)

    def test_swiglu_oai(self):
        def ref(g, u):
            g = np.minimum(g, 3.0)
            u = np.clip(u, -3.0, 3.0)
            return (u + 1.0) * (g * (1 / (1 + np.exp(-g * 1.7))))

        self._check("swiglu_oai", (1.7, 3.0), ref)

    def test_scaled_silu(self):
        def ref(g, u):
            gm = 1.25
            return ((g * gm) / (1 + np.exp(-(g * gm)))) * u

        self._check("scaled_silu", (1.25,), ref)

    def test_situ(self):
        def ref(g, u):
            beta, lb = 4.0, 2.0
            gate = beta * np.tanh(g / beta) * (1 / (1 + np.exp(-g)))
            return gate * (lb * np.tanh(u / lb))

        self._check("situ", (4.0, 2.0), ref)

    def test_situ_no_linear_squash(self):
        def ref(g, u):
            beta = 4.0
            return beta * np.tanh(g / beta) * (1 / (1 + np.exp(-g))) * u

        self._check("situ", (4.0, 0.0), ref)

    def test_combine_gradients_flow(self):
        """Parameterized combines differentiate through the vjp path (int8)."""
        rng = np.random.default_rng(7)
        k, i = 64, 128
        x = jnp.asarray(rng.normal(size=(16, k)), jnp.bfloat16)
        gq, gs = _quantize_channelwise(rng.normal(size=(k, i)).astype(np.float32), jnp.int8)
        uq, us = _quantize_channelwise(rng.normal(size=(k, i)).astype(np.float32), jnp.int8)
        dq, ds = _quantize_channelwise(rng.normal(size=(i, k)).astype(np.float32), jnp.int8)

        def loss(x):
            out = fused_mlp(
                x,
                gq,
                uq,
                dq,
                gate_scale=gs,
                up_scale=us,
                down_scale=ds,
                activation="clamped_swiglu",
                act_params=(2.0,),
            )
            return jnp.sum(out.astype(jnp.float32) ** 2)

        dx = jax.grad(loss)(x)
        assert bool(jnp.all(jnp.isfinite(dx.astype(jnp.float32))))
        assert float(jnp.abs(dx.astype(jnp.float32)).max()) > 0.0

    def test_bad_arity_raises(self):
        x, wg, wu, wd = self._xu()
        with pytest.raises(ValueError, match="act_params"):
            fused_mlp(x, wg, wu, wd, activation="clamped_swiglu", act_params=())
