"""Slow dynamic transform matrix covering broader user-facing combinations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx

from ._dynamic_helpers import (
    DTYPES,
    SHAPE_KINDS,
    VMAP_SHAPE_KINDS,
    Affine,
    StatefulAffine,
    assert_only_collections_changed,
    assert_state_allclose,
    assert_tree_allclose,
    make_input,
    make_state_tangent,
    make_tangent,
    make_target,
    mse,
    snapshot_state,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("transform", ["eval_shape", "jit", "remat"])
@pytest.mark.parametrize("placement", ["arg", "kw", "multi"])
@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("form", ["direct", "decorator"])
def test_forward_transform_matrix(transform: str, placement: str, shape_kind: str, dtype: jnp.dtype, form: str):
    """Forward-style transforms should preserve numerics across placements."""
    if transform == "eval_shape" and form == "decorator":
        pytest.skip("eval_shape has direct-call form only")

    x = make_input(shape_kind, dtype)

    if placement == "arg":
        model = Affine(dtype=dtype)
        if transform == "eval_shape":
            out = spx.eval_shape(lambda mod, xb: mod(xb), model, jax.ShapeDtypeStruct(x.shape, x.dtype))
            ref = jax.eval_shape(lambda xb: model(xb), jax.ShapeDtypeStruct(x.shape, x.dtype))
            assert out.shape == ref.shape
            assert out.dtype == ref.dtype
            return
        if transform == "jit":
            if form == "direct":
                fn = spx.jit(lambda mod, xb: mod(xb))
            else:

                @spx.jit
                def fn(mod, xb):
                    """Helper function."""
                    return mod(xb)

        else:
            if form == "direct":
                fn = spx.remat(lambda mod, xb: mod(xb))
            else:

                @spx.remat
                def fn(mod, xb):
                    """Helper function."""
                    return mod(xb)

        out = fn(model, x)
        ref = model(x)
    elif placement == "kw":
        model = Affine(dtype=dtype)
        if transform == "eval_shape":
            out = spx.eval_shape(
                lambda xb, *, model: model(xb),
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                model=model,
            )
            ref = jax.eval_shape(lambda xb: model(xb), jax.ShapeDtypeStruct(x.shape, x.dtype))
            assert out.shape == ref.shape
            assert out.dtype == ref.dtype
            return
        if transform == "jit":
            if form == "direct":
                fn = spx.jit(lambda xb, *, model: model(xb))
            else:

                @spx.jit
                def fn(xb, *, model):
                    """Helper function."""
                    return model(xb)

        else:
            if form == "direct":
                fn = spx.remat(lambda xb, *, model: model(xb))
            else:

                @spx.remat
                def fn(xb, *, model):
                    """Helper function."""
                    return model(xb)

        out = fn(x, model=model)
        ref = model(x)
    else:
        left = Affine(dtype=dtype, scale=1.0)
        right = Affine(dtype=dtype, scale=0.5, bias_shift=0.125)
        if transform == "eval_shape":
            out = spx.eval_shape(
                lambda lhs, rhs, xb: lhs(xb) + rhs(xb),
                left,
                right,
                jax.ShapeDtypeStruct(x.shape, x.dtype),
            )
            ref = jax.eval_shape(lambda xb: left(xb) + right(xb), jax.ShapeDtypeStruct(x.shape, x.dtype))
            assert out.shape == ref.shape
            assert out.dtype == ref.dtype
            return
        if transform == "jit":
            if form == "direct":
                fn = spx.jit(lambda lhs, rhs, xb: lhs(xb) + rhs(xb))
            else:

                @spx.jit
                def fn(lhs, rhs, xb):
                    """Helper function."""
                    return lhs(xb) + rhs(xb)

        else:
            if form == "direct":
                fn = spx.remat(lambda lhs, rhs, xb: lhs(xb) + rhs(xb))
            else:

                @spx.remat
                def fn(lhs, rhs, xb):
                    """Helper function."""
                    return lhs(xb) + rhs(xb)

        out = fn(left, right, x)
        ref = left(x) + right(x)

    assert_tree_allclose(out, ref, dtype=dtype)


@pytest.mark.parametrize("placement", ["arg", "kw", "multi"])
@pytest.mark.parametrize("shape_kind", VMAP_SHAPE_KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_vmap_matrix(shape_kind: str, placement: str, dtype: jnp.dtype):
    """vmap should preserve numerics across placements and dtypes."""
    x = make_input(shape_kind, dtype)

    if placement == "arg":
        model = Affine(dtype=dtype)
        out = spx.vmap(lambda mod, xb: mod(xb), in_axes=(None, 0))(model, x)
        ref = jnp.stack([model(x[i]) for i in range(x.shape[0])], axis=0)
    elif placement == "kw":
        model = Affine(dtype=dtype)
        out = spx.vmap(lambda xb, *, model: model(xb), in_axes=0)(x, model=model)
        ref = jnp.stack([model(x[i]) for i in range(x.shape[0])], axis=0)
    else:
        left = Affine(dtype=dtype, scale=1.0)
        right = Affine(dtype=dtype, scale=0.5, bias_shift=0.125)
        out = spx.vmap(lambda lhs, rhs, xb: lhs(xb) + rhs(xb), in_axes=(None, None, 0))(left, right, x)
        ref = jnp.stack([left(x[i]) + right(x[i]) for i in range(x.shape[0])], axis=0)

    assert_tree_allclose(out, ref, dtype=dtype)


@pytest.mark.parametrize("transform", ["grad", "value_and_grad"])
@pytest.mark.parametrize("placement", ["arg0", "arg1", "multi"])
@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_autodiff_value_matrix(transform: str, placement: str, shape_kind: str, dtype: jnp.dtype):
    """grad-style transforms should match state-based JAX references."""
    x = make_input(shape_kind, dtype)
    target = make_target(shape_kind, dtype)

    if placement == "arg0":
        model = Affine(dtype=dtype)
        gdef, state = spx.export(model)

        def loss(mod, xb, yb):
            """Compute the loss."""
            return mse(mod(xb), yb)

        def ref_loss(state_, xb, yb):
            """ref_loss helper."""
            return mse(spx.bind(gdef, state_)(xb), yb)

        if transform == "grad":
            out = spx.grad(loss)(model, x, target)
            ref = jax.grad(ref_loss)(state, x, target)
            assert_state_allclose(out, ref, dtype=dtype)
        else:
            value, grads = spx.value_and_grad(loss)(model, x, target)
            ref_value, ref_grads = jax.value_and_grad(ref_loss)(state, x, target)
            assert_tree_allclose(value, ref_value, dtype=dtype)
            assert_state_allclose(grads, ref_grads, dtype=dtype)
    elif placement == "arg1":
        model = Affine(dtype=dtype)
        gdef, state = spx.export(model)

        def loss(xb, mod, yb):
            """Compute the loss."""
            return mse(mod(xb), yb)

        def ref_loss(state_, xb, yb):
            """ref_loss helper."""
            return mse(spx.bind(gdef, state_)(xb), yb)

        if transform == "grad":
            out = spx.grad(loss, argnum=1)(x, model, target)
            ref = jax.grad(ref_loss)(state, x, target)
            assert_state_allclose(out, ref, dtype=dtype)
        else:
            value, grads = spx.value_and_grad(loss, argnum=1)(x, model, target)
            ref_value, ref_grads = jax.value_and_grad(ref_loss)(state, x, target)
            assert_tree_allclose(value, ref_value, dtype=dtype)
            assert_state_allclose(grads, ref_grads, dtype=dtype)
    else:
        left = Affine(dtype=dtype, scale=1.0)
        right = Affine(dtype=dtype, scale=0.5, bias_shift=0.125)
        left_gdef, left_state = spx.export(left)
        right_gdef, right_state = spx.export(right)

        def loss(lhs, rhs, xb, yb):
            """Compute the loss."""
            return mse(lhs(xb) + rhs(xb), yb)

        def ref_loss(left_state_, xb, yb):
            """ref_loss helper."""
            lhs = spx.bind(left_gdef, left_state_)
            rhs = spx.bind(right_gdef, right_state)
            return mse(lhs(xb) + rhs(xb), yb)

        if transform == "grad":
            out = spx.grad(loss)(left, right, x, target)
            ref = jax.grad(ref_loss)(left_state, x, target)
            assert_state_allclose(out, ref, dtype=dtype)
        else:
            value, grads = spx.value_and_grad(loss)(left, right, x, target)
            ref_value, ref_grads = jax.value_and_grad(ref_loss)(left_state, x, target)
            assert_tree_allclose(value, ref_value, dtype=dtype)
            assert_state_allclose(grads, ref_grads, dtype=dtype)


@pytest.mark.parametrize("transform", ["jvp_state", "jvp_module", "vjp"])
@pytest.mark.parametrize("placement", ["arg0", "arg1", "multi"])
@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_autodiff_linearization_matrix(transform: str, placement: str, shape_kind: str, dtype: jnp.dtype):
    """jvp/vjp should match state-based JAX references across placements."""
    x = make_input(shape_kind, dtype)
    x_tangent = make_tangent(shape_kind, dtype)

    if placement == "arg0":
        model = Affine(dtype=dtype)
        gdef, state = spx.export(model)
        state_tangent = make_state_tangent(model)
        module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), model)

        def ref_apply(state_, xb):
            """ref_apply helper."""
            return spx.bind(gdef, state_)(xb)

        if transform == "jvp_state":
            out, tangent_out = spx.jvp(lambda mod, xb: mod(xb), (model, x), (state_tangent, x_tangent))
            ref_out, ref_tangent = jax.jvp(ref_apply, (state, x), (state_tangent, x_tangent))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        elif transform == "jvp_module":
            out, tangent_out = spx.jvp(lambda mod, xb: mod(xb), (model, x), (module_tangent, x_tangent))
            ref_out, ref_tangent = jax.jvp(ref_apply, (state, x), (state_tangent, x_tangent))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        else:
            out, pullback = spx.vjp(lambda mod, xb: mod(xb), model, x)
            ref_out, ref_pullback = jax.vjp(ref_apply, state, x)
            state_cts, x_cts = pullback(jnp.ones_like(out))
            ref_state_cts, ref_x_cts = ref_pullback(jnp.ones_like(ref_out))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_state_allclose(state_cts, ref_state_cts, dtype=dtype)
            assert_tree_allclose(x_cts, ref_x_cts, dtype=dtype)
    elif placement == "arg1":
        model = Affine(dtype=dtype)
        gdef, state = spx.export(model)
        state_tangent = make_state_tangent(model)
        module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), model)

        def ref_apply(xb, state_):
            """ref_apply helper."""
            return spx.bind(gdef, state_)(xb)

        if transform == "jvp_state":
            out, tangent_out = spx.jvp(lambda xb, mod: mod(xb), (x, model), (x_tangent, state_tangent))
            ref_out, ref_tangent = jax.jvp(ref_apply, (x, state), (x_tangent, state_tangent))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        elif transform == "jvp_module":
            out, tangent_out = spx.jvp(lambda xb, mod: mod(xb), (x, model), (x_tangent, module_tangent))
            ref_out, ref_tangent = jax.jvp(ref_apply, (x, state), (x_tangent, state_tangent))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        else:
            out, pullback = spx.vjp(lambda xb, mod: mod(xb), x, model)
            ref_out, ref_pullback = jax.vjp(ref_apply, x, state)
            x_cts, state_cts = pullback(jnp.ones_like(out))
            ref_x_cts, ref_state_cts = ref_pullback(jnp.ones_like(ref_out))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(x_cts, ref_x_cts, dtype=dtype)
            assert_state_allclose(state_cts, ref_state_cts, dtype=dtype)
    else:
        left = Affine(dtype=dtype, scale=1.0)
        right = Affine(dtype=dtype, scale=0.5, bias_shift=0.125)
        left_gdef, left_state = spx.export(left)
        right_gdef, right_state = spx.export(right)
        left_tangent = make_state_tangent(left)
        right_tangent = make_state_tangent(right)
        left_module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), left)
        right_module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), right)

        def ref_apply(left_state_, right_state_, xb):
            """ref_apply helper."""
            lhs = spx.bind(left_gdef, left_state_)
            rhs = spx.bind(right_gdef, right_state_)
            return lhs(xb) + rhs(xb)

        if transform == "jvp_state":
            out, tangent_out = spx.jvp(
                lambda lhs, rhs, xb: lhs(xb) + rhs(xb),
                (left, right, x),
                (left_tangent, right_tangent, x_tangent),
            )
            ref_out, ref_tangent = jax.jvp(
                ref_apply,
                (left_state, right_state, x),
                (left_tangent, right_tangent, x_tangent),
            )
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        elif transform == "jvp_module":
            out, tangent_out = spx.jvp(
                lambda lhs, rhs, xb: lhs(xb) + rhs(xb),
                (left, right, x),
                (left_module_tangent, right_module_tangent, x_tangent),
            )
            ref_out, ref_tangent = jax.jvp(
                ref_apply,
                (left_state, right_state, x),
                (left_tangent, right_tangent, x_tangent),
            )
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_tree_allclose(tangent_out, ref_tangent, dtype=dtype)
        else:
            out, pullback = spx.vjp(lambda lhs, rhs, xb: lhs(xb) + rhs(xb), left, right, x)
            ref_out, ref_pullback = jax.vjp(ref_apply, left_state, right_state, x)
            left_cts, right_cts, x_cts = pullback(jnp.ones_like(out))
            ref_left_cts, ref_right_cts, ref_x_cts = ref_pullback(jnp.ones_like(ref_out))
            assert_tree_allclose(out, ref_out, dtype=dtype)
            assert_state_allclose(left_cts, ref_left_cts, dtype=dtype)
            assert_state_allclose(right_cts, ref_right_cts, dtype=dtype)
            assert_tree_allclose(x_cts, ref_x_cts, dtype=dtype)


@pytest.mark.parametrize(
    "transform",
    ["jit", "cond", "switch", "fori_loop", "while_loop", "scan", "remat_scan", "jvp", "vjp"],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_stateful_transform_matrix(transform: str, dtype: jnp.dtype):
    """Stateful transforms should only write back declared mutable collections."""
    model = StatefulAffine(dtype=dtype)
    before = snapshot_state(model)
    x = make_input("batch", dtype)

    if transform == "jit":
        spx.jit(lambda mod, xb: mod(xb, mutate=True, amount=1.5), mutable="batch_stats")(model, x)
        expected = 1.5
    elif transform == "cond":
        spx.cond(
            jnp.bool_(True),
            lambda mod, xb: mod(xb, mutate=True, amount=2.0),
            lambda mod, xb: mod(xb),
            model,
            x,
            mutable="batch_stats",
        )
        expected = 2.0
    elif transform == "switch":
        spx.switch(
            jnp.int32(1),
            [
                lambda mod, xb: mod(xb),
                lambda mod, xb: mod(xb, mutate=True, amount=2.5),
                lambda mod, xb: mod(xb),
            ],
            model,
            x,
            mutable="batch_stats",
        )
        expected = 2.5
    elif transform == "fori_loop":
        spx.fori_loop(
            0,
            3,
            lambda _i, mod, carry: mod(carry, mutate=True, amount=1.0),
            model,
            x,
            mutable="batch_stats",
        )
        expected = 3.0
    elif transform == "while_loop":
        spx.while_loop(
            lambda _mod, carry: carry[0] < 2,
            lambda mod, carry: (carry[0] + 1, mod(carry[1], mutate=True, amount=1.25)),
            model,
            (jnp.int32(0), x),
            mutable="batch_stats",
        )
        expected = 2.5
    elif transform == "scan":
        spx.scan(lambda mod, xb: mod(xb, mutate=True, amount=1.0), model, x, mutable="batch_stats")
        expected = float(x.shape[0])
    elif transform == "remat_scan":
        spx.remat_scan(lambda mod, xb: mod(xb, mutate=True, amount=1.0), model, x, mutable="batch_stats")
        expected = float(x.shape[0])
    elif transform == "jvp":
        spx.jvp(
            lambda mod, xb: mod(xb, mutate=True, amount=1.75),
            (model, x),
            (make_state_tangent(model), make_tangent("batch", dtype)),
            mutable="batch_stats",
        )
        expected = 1.75
    else:
        out, pullback = spx.vjp(lambda mod, xb: mod(xb, mutate=True, amount=1.25), model, x, mutable="batch_stats")
        pullback(jnp.ones_like(out))
        expected = 1.25

    after = snapshot_state(model)
    assert float(model.acc.value.astype(jnp.float32)) == pytest.approx(expected, abs=1e-5)
    assert_only_collections_changed(before, after, allowed=("batch_stats",), dtype=dtype)
