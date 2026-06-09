"""Quick smoke test for sxjit / sxcall static_argnums / donate_argnums."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import sxcall, sxjit, sxstage_iter
from spectrax.runtime.schedules import GPipe
from spectrax.runtime.types import MpMdMesh


class _Block(spx.Module):
    """Helper block module for testing."""

    def __init__(self, d, *, rngs):
        """Initialize with fc."""
        super().__init__()
        self.fc = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        return jax.nn.relu(self.fc(x))


def _make_mesh(n=2):
    """Create a test mesh."""
    devices = jax.devices()[:n] if len(jax.devices()) >= n else jax.devices() * n
    devices = devices[:n]
    return MpMdMesh(Mesh(np.array(devices), ("mpmd",)), "mpmd")


def test_static_argnums_basic():
    """Test that static_argnums embeds the arg as a constant."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    blocks = [_Block(4, rngs=rngs) for _ in range(2)]

    @sxjit(mesh=mesh, static_argnums=(2,))
    def forward(model, x, scale):
        """Run the forward pass."""
        x = model[0].forward(x)
        x = sxstage_iter(x)
        x = model[1].forward(x)
        return x * scale

    x = jnp.ones((2, 4))
    out = forward(blocks, x, 3.0)
    expected = 3.0 * blocks[1].forward(blocks[0].forward(x))
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    print("test_static_argnums_basic PASSED")


def test_donate_argnums_basic():
    """Test that donate_argnums is accepted and execution works."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    blocks = [_Block(4, rngs=rngs) for _ in range(2)]

    @sxjit(mesh=mesh, donate_argnums=(1,))
    def forward(model, x):
        """Run the forward pass."""
        x = model[0].forward(x)
        x = sxstage_iter(x)
        x = model[1].forward(x)
        return x

    x = jnp.ones((2, 4))
    expected = blocks[1].forward(blocks[0].forward(x))
    out = forward(blocks, x)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    print("test_donate_argnums_basic PASSED")


def test_static_and_donate_combined():
    """Test static_argnums + donate_argnums together."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    blocks = [_Block(4, rngs=rngs) for _ in range(2)]

    @sxjit(mesh=mesh, static_argnums=(2,), donate_argnums=(1,))
    def forward(model, x, scale):
        """Run the forward pass."""
        x = model[0].forward(x)
        x = sxstage_iter(x)
        x = model[1].forward(x)
        return x * scale

    x = jnp.ones((2, 4))
    expected = 2.0 * blocks[1].forward(blocks[0].forward(x))
    out = forward(blocks, x, 2.0)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    print("test_static_and_donate_combined PASSED")


def test_legacy_path_still_works():
    """Ensure backward compatibility when no static/donate args are given."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    blocks = [_Block(4, rngs=rngs) for _ in range(2)]

    @sxjit(mesh=mesh)
    def forward(model, x):
        """Run the forward pass."""
        x = model[0].forward(x)
        x = sxstage_iter(x)
        x = model[1].forward(x)
        return x

    x = jnp.ones((2, 4))
    out = forward(blocks, x)
    np.testing.assert_allclose(out, blocks[1].forward(blocks[0].forward(x)), rtol=1e-5)
    print("test_legacy_path_still_works PASSED")


def test_mpmd_call_static_targets():
    """Test sxcall with static target args."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    model = PipelineSequential(*[_Block(4, rngs=rngs) for _ in range(2)])

    def loss_fn(y, target):
        """Compute the loss."""
        return ((y - target) ** 2).mean()

    x = jnp.ones((4, 4))
    target = jnp.ones((2, 4)) * 2.0

    loss, grads = sxcall(
        model,
        (x, target),
        mesh=mesh,
        schedule=GPipe(microbatches=2),
        loss_fn=loss_fn,
        static_argnums=(1,),
    )

    assert loss.shape == ()
    assert len(grads) == 2
    print("test_mpmd_call_static_targets PASSED")


def test_mpmd_call_donate_targets():
    """Test sxcall with donated target args."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    model = PipelineSequential(*[_Block(4, rngs=rngs) for _ in range(2)])

    def loss_fn(y, target):
        """Compute the loss."""
        return ((y - target) ** 2).mean()

    x = jnp.ones((4, 4))
    target = jnp.ones((2, 4)) * 2.0

    loss, grads = sxcall(
        model,
        (x, target),
        mesh=mesh,
        schedule=GPipe(microbatches=2),
        loss_fn=loss_fn,
        donate_argnums=(1,),
    )

    assert loss.shape == ()
    assert len(grads) == 2
    print("test_mpmd_call_donate_targets PASSED")


def test_mpmd_call_static_and_donate():
    """Test sxcall with static + donate combined."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    model = PipelineSequential(*[_Block(4, rngs=rngs) for _ in range(2)])

    def loss_fn(y, target, scale):
        """Compute the loss."""
        return ((y - target) ** 2).mean() * scale

    x = jnp.ones((4, 4))
    target = jnp.ones((2, 4)) * 2.0

    loss, grads = sxcall(
        model,
        (x, target, 3.0),
        mesh=mesh,
        schedule=GPipe(microbatches=2),
        loss_fn=loss_fn,
        static_argnums=(2,),
        donate_argnums=(1,),
    )

    assert loss.shape == ()
    assert len(grads) == 2
    print("test_mpmd_call_static_and_donate PASSED")


def test_mpmd_call_forward_donate_input():
    """Test sxcall forward-only mode with donated input."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    stages = [_Block(4, rngs=rngs) for _ in range(2)]
    model = PipelineSequential(*stages)

    x = jnp.ones((4, 4))
    expected = stages[1].forward(stages[0].forward(x))

    out = sxcall(
        model,
        (x,),
        mesh=mesh,
        schedule=GPipe(microbatches=2),
        mode="forward",
        donate_argnums=(0,),
    )

    np.testing.assert_allclose(out, expected, rtol=1e-5)
    print("test_mpmd_call_forward_donate_input PASSED")


def test_mpmd_call_train_donate_input_raises():
    """Test that donating input in train mode raises an error."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    model = PipelineSequential(*[_Block(4, rngs=rngs) for _ in range(2)])

    def loss_fn(y, target):
        """Compute the loss."""
        return ((y - target) ** 2).mean()

    x = jnp.ones((4, 4))
    target = jnp.ones((2, 4)) * 2.0

    try:
        sxcall(
            model,
            (x, target),
            mesh=mesh,
            schedule=GPipe(microbatches=2),
            loss_fn=loss_fn,
            donate_argnums=(0,),
        )
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "cannot donate batch[0]" in str(e)
    print("test_mpmd_call_train_donate_input_raises PASSED")


def test_mpmd_call_static_input_raises():
    """Test that static input raises an error."""
    mesh = _make_mesh(2)
    rngs = spx.Rngs(0)
    model = PipelineSequential(*[_Block(4, rngs=rngs) for _ in range(2)])

    def loss_fn(y, target):
        """Compute the loss."""
        return ((y - target) ** 2).mean()

    x = jnp.ones((4, 4))
    target = jnp.ones((2, 4)) * 2.0

    try:
        sxcall(
            model,
            (x, target),
            mesh=mesh,
            schedule=GPipe(microbatches=2),
            loss_fn=loss_fn,
            static_argnums=(0,),
        )
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "cannot be static" in str(e)
    print("test_mpmd_call_static_input_raises PASSED")


if __name__ == "__main__":
    test_legacy_path_still_works()
    test_static_argnums_basic()
    test_donate_argnums_basic()
    test_static_and_donate_combined()
    test_mpmd_call_static_targets()
    test_mpmd_call_donate_targets()
    test_mpmd_call_static_and_donate()
    test_mpmd_call_forward_donate_input()
    test_mpmd_call_train_donate_input_raises()
    test_mpmd_call_static_input_raises()
    print("\nAll tests passed!")
