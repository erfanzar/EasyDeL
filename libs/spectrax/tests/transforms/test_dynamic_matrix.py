"""Dynamic regression coverage for module-aware transforms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx

from ._dynamic_helpers import (
    SHAPE_KINDS,
    VMAP_SHAPE_KINDS,
    Affine,
    ScaleModule,
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


def _state_forward(gdef, state, x):
    """State-aware forward helper."""
    model = spx.bind(gdef, state)
    return model(x)


def _manual_loop(model, x, steps: int) -> jax.Array:
    """Manual loop implementation."""
    out = x
    for _ in range(steps):
        out = model(out)
    return out


def _mutating_scan_impl(transform: str, model: StatefulAffine, xs: jax.Array, *, mutable=()) -> Any:
    """Mutating scan implementation."""
    if transform == "scan":
        return spx.scan(lambda mod, x: mod(x, mutate=True, amount=1.0), model, xs, mutable=mutable)
    if transform == "remat_scan":
        return spx.remat_scan(lambda mod, x: mod(x, mutate=True, amount=1.0), model, xs, mutable=mutable)
    raise AssertionError(transform)


def _key_drift_scan_impl(transform: str, model: StatefulAffine, xs: jax.Array) -> Any:
    """Key-drift scan implementation."""

    def body(mod, x):
        """Loop body function."""
        mod.extra = spx.Buffer(jnp.array(1.0, dtype=x.dtype), kind="batch_stats")
        return mod(x)

    if transform == "scan":
        return spx.scan(body, model, xs)
    if transform == "remat_scan":
        return spx.remat_scan(body, model, xs)
    raise AssertionError(transform)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("placement", ["arg", "kw", "multi"])
def test_eval_shape_dynamic_matches_reference(shape_kind: str, placement: str):
    """eval_shape should preserve the user's calling style."""
    x_spec = jax.ShapeDtypeStruct(make_input(shape_kind).shape, jnp.float32)

    if placement == "arg":
        model = Affine()
        out = spx.eval_shape(lambda mod, x: mod(x), model, x_spec)
        ref = jax.eval_shape(lambda x: model(x), x_spec)
    elif placement == "kw":
        model = Affine()
        out = spx.eval_shape(lambda x, *, model: model(x), x_spec, model=model)
        ref = jax.eval_shape(lambda x: model(x), x_spec)
    else:
        left = Affine(scale=1.0)
        right = Affine(scale=0.5, bias_shift=0.1)
        out = spx.eval_shape(lambda lhs, rhs, x: lhs(x) + rhs(x), left, right, x_spec)
        ref = jax.eval_shape(lambda x: left(x) + right(x), x_spec)

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


def test_eval_shape_does_not_write_back_dynamic_mutations():
    """Abstract evaluation keeps live module state untouched."""
    model = StatefulAffine()
    before = snapshot_state(model)
    x_spec = jax.ShapeDtypeStruct(make_input("batch").shape, jnp.float32)
    out = spx.eval_shape(lambda mod, x: mod(x, mutate=True, amount=2.0), model, x_spec)
    after = snapshot_state(model)
    assert out.shape == (3, 4)
    assert_state_allclose(before, after)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
@pytest.mark.parametrize("form", ["direct", "decorator"])
@pytest.mark.parametrize("placement", ["arg", "kw", "multi"])
def test_jit_dynamic_matches_eager(shape_kind: str, form: str, placement: str):
    """jit should match eager numerics across call styles."""
    x = make_input(shape_kind)

    if placement == "arg":
        if form == "direct":
            fn = spx.jit(lambda mod, x: mod(x))
        else:

            @spx.jit
            def fn(mod, x):
                """Helper function."""
                return mod(x)

        model = Affine()
        out = fn(model, x)
        ref = model(x)
    elif placement == "kw":
        if form == "direct":
            fn = spx.jit(lambda x, *, model: model(x))
        else:

            @spx.jit
            def fn(x, *, model):
                """Helper function."""
                return model(x)

        model = Affine()
        out = fn(x, model=model)
        ref = model(x)
    else:
        left = Affine(scale=1.0)
        right = Affine(scale=0.5, bias_shift=0.1)
        if form == "direct":
            fn = spx.jit(lambda lhs, rhs, x: lhs(x) + rhs(x))
        else:

            @spx.jit
            def fn(lhs, rhs, x):
                """Helper function."""
                return lhs(x) + rhs(x)

        out = fn(left, right, x)
        ref = left(x) + right(x)

    assert_tree_allclose(out, ref)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
def test_grad_dynamic_matches_state_reference(shape_kind: str):
    """grad should agree with a pure state-based JAX reference."""
    model = Affine()
    x = make_input(shape_kind)
    target = make_target(shape_kind)
    gdef, state = spx.export(model)

    def loss(mod, xb, yb):
        """Compute the loss."""
        return mse(mod(xb), yb)

    def ref_loss(state_, xb, yb):
        """ref_loss helper."""
        return mse(_state_forward(gdef, state_, xb), yb)

    grads = spx.grad(loss)(model, x, target)
    ref_grads = jax.grad(ref_loss)(state, x, target)
    assert_state_allclose(grads, ref_grads)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
def test_value_and_grad_dynamic_has_aux_matches_reference(shape_kind: str):
    """value_and_grad should preserve both value and aux."""
    model = Affine()
    x = make_input(shape_kind)
    target = make_target(shape_kind)
    gdef, state = spx.export(model)

    def loss(mod, xb, yb):
        """Compute the loss."""
        pred = mod(xb)
        return mse(pred, yb), pred.shape

    def ref_loss(state_, xb, yb):
        """ref_loss helper."""
        pred = _state_forward(gdef, state_, xb)
        return mse(pred, yb)

    (value, aux), grads = spx.value_and_grad(loss, has_aux=True)(model, x, target)
    ref_value, ref_grads = jax.value_and_grad(ref_loss)(state, x, target)
    assert aux == make_input(shape_kind).shape
    assert_tree_allclose(value, ref_value)
    assert_state_allclose(grads, ref_grads)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
def test_jvp_dynamic_matches_state_reference(shape_kind: str):
    """jvp should match JAX for both State and Module tangents."""
    model = Affine()
    x = make_input(shape_kind)
    x_tangent = make_tangent(shape_kind)
    state_tangent = make_state_tangent(model)
    module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), model)
    gdef, state = spx.export(model)

    def ref_apply(state_, xb):
        """ref_apply helper."""
        return _state_forward(gdef, state_, xb)

    ref_out, ref_tangent = jax.jvp(ref_apply, (state, x), (state_tangent, x_tangent))
    out_state, tangent_state = spx.jvp(lambda mod, xb: mod(xb), (model, x), (state_tangent, x_tangent))
    out_module, tangent_module = spx.jvp(lambda mod, xb: mod(xb), (model, x), (module_tangent, x_tangent))

    assert_tree_allclose(out_state, ref_out)
    assert_tree_allclose(tangent_state, ref_tangent)
    assert_tree_allclose(out_module, ref_out)
    assert_tree_allclose(tangent_module, ref_tangent)


@pytest.mark.parametrize("shape_kind", SHAPE_KINDS)
def test_vjp_dynamic_matches_state_reference(shape_kind: str):
    """vjp pullbacks should match a pure state-based JAX reference."""
    model = Affine()
    x = make_input(shape_kind)
    gdef, state = spx.export(model)

    def ref_apply(state_, xb):
        """ref_apply helper."""
        return _state_forward(gdef, state_, xb)

    out, pullback = spx.vjp(lambda mod, xb: mod(xb), model, x)
    ref_out, ref_pullback = jax.vjp(ref_apply, state, x)
    cotangent = jnp.ones_like(out)
    state_cts, x_cts = pullback(cotangent)
    ref_state_cts, ref_x_cts = ref_pullback(cotangent)

    assert_tree_allclose(out, ref_out)
    assert_state_allclose(state_cts, ref_state_cts)
    assert_tree_allclose(x_cts, ref_x_cts)


@pytest.mark.parametrize("shape_kind", VMAP_SHAPE_KINDS)
def test_vmap_dynamic_matches_manual(shape_kind: str):
    """vmap should match an explicit Python stack across leading-axis slices."""
    model = Affine()
    x = make_input(shape_kind)
    vmapped = spx.vmap(lambda mod, xb: mod(xb), in_axes=(None, 0))(model, x)
    manual = jnp.stack([model(x[i]) for i in range(x.shape[0])], axis=0)
    assert_tree_allclose(vmapped, manual)


@pytest.mark.parametrize("op", ["cond", "switch", "fori_loop", "while_loop"])
def test_control_flow_dynamic_matches_reference(op: str):
    """Control-flow wrappers should match equivalent eager logic."""
    model = Affine()
    x = make_input("batch", offset=2.0)

    if op == "cond":
        out = spx.cond(
            jnp.bool_(True),
            lambda mod, xb: mod(xb) + 1.0,
            lambda mod, xb: mod(xb) - 1.0,
            model,
            x,
        )
        ref = model(x) + 1.0
    elif op == "switch":
        out = spx.switch(
            jnp.int32(2),
            [
                lambda mod, xb: mod(xb) * 0.0,
                lambda mod, xb: mod(xb) * 1.0,
                lambda mod, xb: mod(xb) * 2.0,
            ],
            model,
            x,
        )
        ref = model(x) * 2.0
    elif op == "fori_loop":
        out = spx.fori_loop(0, 3, lambda _i, mod, carry: mod(carry), model, x)
        ref = _manual_loop(model, x, 3)
    else:

        def cond_fn(_mod, carry):
            """Condition function."""
            i, _x = carry
            return i < 3

        def body_fn(mod, carry):
            """Body function for loop/cond."""
            i, xb = carry
            return i + 1, mod(xb)

        out = spx.while_loop(cond_fn, body_fn, model, (jnp.int32(0), x))[1]
        ref = _manual_loop(model, x, 3)

    assert_tree_allclose(out, ref)


@pytest.mark.parametrize("transform", ["scan", "remat_scan"])
@pytest.mark.parametrize("shape_kind", VMAP_SHAPE_KINDS)
def test_scan_family_dynamic_matches_manual(transform: str, shape_kind: str):
    """scan wrappers should match explicit per-step evaluation when pure."""
    model = Affine()
    xs = make_input(shape_kind)
    if transform == "scan":
        ys = spx.scan(lambda mod, x: mod(x), model, xs)
    else:
        ys = spx.remat_scan(lambda mod, x: mod(x), model, xs)
    ref = jnp.stack([model(xs[i]) for i in range(xs.shape[0])], axis=0)
    assert_tree_allclose(ys, ref)


def test_remat_dynamic_matches_eager():
    """remat should preserve pure forward numerics."""
    model = Affine()
    x = make_input("sequence")
    out = spx.remat(lambda mod, xb: mod(xb))(model, x)
    ref = model(x)
    assert_tree_allclose(out, ref)


def test_associative_scan_dynamic_matches_jax():
    """associative_scan should agree with JAX for a pure combine."""
    model = ScaleModule()
    xs = make_input("batch")
    out = spx.associative_scan(lambda mod, a, b: mod.scale.value * (a + b), model, xs, axis=0)
    ref = jax.lax.associative_scan(lambda a, b: a + b, xs, axis=0)
    assert_tree_allclose(out, ref)


@pytest.mark.parametrize("op", ["cond", "switch", "fori_loop", "while_loop"])
def test_jit_wrapped_control_flow_matches_eager(op: str):
    """jit around control-flow wrappers should preserve numerics."""
    model = Affine()
    x = make_input("batch", offset=2.0)

    if op == "cond":

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.cond(
                jnp.bool_(True),
                lambda inner, v: inner(v) + 1.0,
                lambda inner, v: inner(v) - 1.0,
                mod,
                xb,
            )
    elif op == "switch":

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.switch(
                jnp.int32(1),
                [
                    lambda inner, v: inner(v) * 0.0,
                    lambda inner, v: inner(v) * 1.0,
                    lambda inner, v: inner(v) * 2.0,
                ],
                mod,
                xb,
            )
    elif op == "fori_loop":

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.fori_loop(0, 3, lambda _i, inner, carry: inner(carry), mod, xb)
    else:

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.while_loop(
                lambda _inner, carry: carry[0] < 3,
                lambda inner, carry: (carry[0] + 1, inner(carry[1])),
                mod,
                (jnp.int32(0), xb),
            )[1]

    compiled = spx.jit(eager)
    assert_tree_allclose(compiled(model, x), eager(model, x))


@pytest.mark.parametrize("transform", ["grad", "value_and_grad", "jvp_state", "jvp_module", "vjp"])
def test_jit_wrapped_autodiff_matches_eager(transform: str):
    """jit around autodiff wrappers should preserve nested-transform numerics."""
    model = Affine()
    x = make_input("batch")
    target = make_target("batch")
    state_tangent = make_state_tangent(model)
    module_tangent = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.125), model)
    x_tangent = make_tangent("batch")

    def loss(mod, xb, yb):
        """Compute the loss."""
        return mse(mod(xb), yb)

    if transform == "grad":

        def eager(mod, xb, yb):
            """Eager execution helper."""
            return spx.grad(loss)(mod, xb, yb)

        compiled = spx.jit(eager)
        assert_state_allclose(compiled(model, x, target), eager(model, x, target))
    elif transform == "value_and_grad":

        def eager(mod, xb, yb):
            """Eager execution helper."""
            return spx.value_and_grad(loss)(mod, xb, yb)

        compiled = spx.jit(eager)
        value_out, grads_out = compiled(model, x, target)
        value_ref, grads_ref = eager(model, x, target)
        assert_tree_allclose(value_out, value_ref)
        assert_state_allclose(grads_out, grads_ref)
    elif transform == "jvp_state":

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.jvp(lambda inner, val: inner(val), (mod, xb), (state_tangent, x_tangent))

        compiled = spx.jit(eager)
        out, tangent_out = compiled(model, x)
        ref_out, ref_tangent = eager(model, x)
        assert_tree_allclose(out, ref_out)
        assert_tree_allclose(tangent_out, ref_tangent)
    elif transform == "jvp_module":

        def eager(mod, xb):
            """Eager execution helper."""
            return spx.jvp(lambda inner, val: inner(val), (mod, xb), (module_tangent, x_tangent))

        compiled = spx.jit(eager)
        out, tangent_out = compiled(model, x)
        ref_out, ref_tangent = eager(model, x)
        assert_tree_allclose(out, ref_out)
        assert_tree_allclose(tangent_out, ref_tangent)
    else:

        def eager(mod, xb):
            """Eager execution helper."""
            out, pullback = spx.vjp(lambda inner, val: inner(val), mod, xb)
            grad_mod, grad_x = pullback(jnp.ones_like(out))
            return out, grad_mod, grad_x

        compiled = spx.jit(eager)
        out, grad_mod, grad_x = compiled(model, x)
        ref_out, ref_grad_mod, ref_grad_x = eager(model, x)
        assert_tree_allclose(out, ref_out)
        assert_state_allclose(grad_mod, ref_grad_mod)
        assert_tree_allclose(grad_x, ref_grad_x)


@pytest.mark.parametrize("transform", ["scan", "remat_scan"])
def test_grad_through_scan_family_matches_state_reference(transform: str):
    """grad through scan-based forwards should agree with a pure reference."""
    model = Affine()
    xs = make_input("batch")
    target = make_target("batch")
    gdef, state = spx.export(model)

    def loss(mod, xb, yb):
        """Compute the loss."""
        if transform == "scan":
            ys = spx.scan(lambda inner, x: inner(x), mod, xb)
        else:
            ys = spx.remat_scan(lambda inner, x: inner(x), mod, xb)
        return mse(ys, yb)

    def ref_loss(state_, xb, yb):
        """ref_loss helper."""
        rebound = spx.bind(gdef, state_)
        ys = jnp.stack([rebound(xb[i]) for i in range(xb.shape[0])], axis=0)
        return mse(ys, yb)

    grads = spx.grad(loss)(model, xs, target)
    ref_grads = jax.grad(ref_loss)(state, xs, target)
    assert_state_allclose(grads, ref_grads)


def test_vmap_over_jitted_forward_matches_manual():
    """Composing vmap over jit should preserve batched numerics."""
    model = Affine()
    x = make_input("batch")
    compiled = spx.jit(lambda mod, xb: mod(xb))
    vmapped = spx.vmap(lambda mod, xb: compiled(mod, xb), in_axes=(None, 0))(model, x)
    manual = jnp.stack([compiled(model, x[i]) for i in range(x.shape[0])], axis=0)
    assert_tree_allclose(vmapped, manual)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "vjp"])
def test_pure_transforms_reject_undeclared_mutation(transform: str):
    """Pure transforms should reject value writes without ``mutable=``."""
    model = StatefulAffine()
    x = make_input("batch")

    def fn(mod, xb):
        """Helper function."""
        return mod(xb, mutate=True, amount=1.0)

    with pytest.raises(spx.IllegalMutationError):
        if transform == "jit":
            spx.jit(fn)(model, x)
        elif transform == "vmap":
            spx.vmap(lambda mod, xb: fn(mod, xb), in_axes=(None, 0))(model, x)
        elif transform == "jvp":
            spx.jvp(fn, (model, x), (make_state_tangent(model), make_tangent("batch")))
        else:
            out, pullback = spx.vjp(fn, model, x)
            pullback(jnp.ones_like(out))


@pytest.mark.parametrize("transform", ["cond", "switch", "fori_loop", "while_loop"])
def test_control_flow_rejects_undeclared_mutation(transform: str):
    """Control-flow wrappers should reject invariant value writes."""
    model = StatefulAffine()
    x = make_input("batch")

    with pytest.raises(spx.IllegalMutationError):
        if transform == "cond":
            spx.cond(
                jnp.bool_(True),
                lambda mod, xb: mod(xb, mutate=True, amount=1.0),
                lambda mod, xb: mod(xb),
                model,
                x,
            )
        elif transform == "switch":
            spx.switch(
                jnp.int32(1),
                [
                    lambda mod, xb: mod(xb),
                    lambda mod, xb: mod(xb, mutate=True, amount=1.0),
                    lambda mod, xb: mod(xb),
                ],
                model,
                x,
            )
        elif transform == "fori_loop":
            spx.fori_loop(0, 2, lambda _i, mod, carry: mod(carry, mutate=True, amount=1.0), model, x)
        else:
            spx.while_loop(
                lambda _mod, carry: carry[0] < 2,
                lambda mod, carry: (carry[0] + 1, mod(carry[1], mutate=True, amount=1.0)),
                model,
                (jnp.int32(0), x),
            )


@pytest.mark.parametrize("transform", ["scan", "remat_scan"])
def test_scan_family_rejects_undeclared_mutation(transform: str):
    """scan wrappers should reject invariant value writes."""
    model = StatefulAffine()
    xs = make_input("batch")
    with pytest.raises(spx.IllegalMutationError):
        _mutating_scan_impl(transform, model, xs)


@pytest.mark.parametrize("transform", ["cond", "switch", "fori_loop", "while_loop"])
def test_control_flow_rejects_invariant_key_drift(transform: str):
    """Changing the invariant key set should raise."""
    model = StatefulAffine()
    x = make_input("batch")

    with pytest.raises(ValueError, match="invariant state changed"):
        if transform == "cond":
            spx.cond(
                jnp.bool_(True),
                lambda mod, xb: setattr(mod, "extra", spx.Buffer(jnp.array(1.0), kind="batch_stats")) or mod(xb),
                lambda mod, xb: mod(xb),
                model,
                x,
            )
        elif transform == "switch":
            spx.switch(
                jnp.int32(2),
                [
                    lambda mod, xb: mod(xb),
                    lambda mod, xb: mod(xb),
                    lambda mod, xb: setattr(mod, "extra", spx.Buffer(jnp.array(1.0), kind="batch_stats")) or mod(xb),
                ],
                model,
                x,
            )
        elif transform == "fori_loop":

            def body(_i, mod, carry):
                """Loop body function."""
                mod.extra = spx.Buffer(jnp.array(1.0), kind="batch_stats")
                return carry

            spx.fori_loop(0, 1, body, model, x)
        else:

            def body(mod, carry):
                """Loop body function."""
                mod.extra = spx.Buffer(jnp.array(1.0), kind="batch_stats")
                return carry[0] + 1, carry[1]

            spx.while_loop(lambda _mod, carry: carry[0] < 1, body, model, (jnp.int32(0), x))


@pytest.mark.parametrize("transform", ["scan", "remat_scan"])
def test_scan_family_rejects_invariant_key_drift(transform: str):
    """scan wrappers should detect invariant key drift."""
    model = StatefulAffine()
    xs = make_input("batch")
    with pytest.raises(ValueError, match="invariant state changed"):
        _key_drift_scan_impl(transform, model, xs)


@pytest.mark.parametrize("transform", ["cond", "switch", "fori_loop", "while_loop", "scan", "remat_scan"])
def test_stateful_transforms_only_write_declared_mutable_collections(transform: str):
    """Mutable collections should write back and only those collections should change."""
    model = StatefulAffine()
    before = snapshot_state(model)
    x = make_input("batch")

    if transform == "cond":
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
                lambda mod, xb: mod(xb, mutate=True, amount=3.0),
                lambda mod, xb: mod(xb),
            ],
            model,
            x,
            mutable="batch_stats",
        )
        expected = 3.0
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
            lambda _mod, carry: carry[0] < 4,
            lambda mod, carry: (carry[0] + 1, mod(carry[1], mutate=True, amount=1.0)),
            model,
            (jnp.int32(0), x),
            mutable="batch_stats",
        )
        expected = 4.0
    else:
        xs = make_input("batch")
        _mutating_scan_impl(transform, model, xs, mutable="batch_stats")
        expected = float(xs.shape[0])

    after = snapshot_state(model)
    assert float(model.acc.value) == expected
    assert_only_collections_changed(before, after, allowed=("batch_stats",))
