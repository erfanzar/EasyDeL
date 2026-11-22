# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`spectrax.run` — one entry point for SPMD or MPMD execution.

Write the model once as a normal :class:`spectrax.Module`. Pass it
to ``spx.run(model, inputs=..., mesh=mesh, mode=...)`` along with a
:class:`SpxMesh`; ``mesh.is_mpmd`` decides whether the model is
executed under pure SPMD (a single :func:`jax.jit` call running
across the whole mesh) or MPMD (auto-split into per-rank pipeline
stages dispatched via :func:`spectrax.runtime.sxcall`).

Examples::

    # SPMD only — pjit, no pipeline parallelism
    mesh = spx.create_mesh(axis_dims=(1, 1, -1, 1, 1, 1))
    out  = spx.run(model, inputs=ids, mesh=mesh, mode='forward')

    # Add pipeline parallelism — same call site, mesh changes
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis='pp')
    loss, grads = spx.run(model, inputs=ids, targets=labels,
                          mesh=mesh, mode='train', loss_fn=ce,
                          microbatches=4)

``inputs`` and ``targets`` accept three forms (Option C):

* a single array  -> forwarded as-is
* a tuple/list    -> unpacked as positional args
* a dict          -> unpacked as kwargs

So ``inputs=ids`` calls ``model.forward(ids)``;
``inputs=(ids, mask)`` calls ``model.forward(ids, mask)``;
``inputs=dict(ids=ids, mask=mask)`` calls ``model.forward(ids=ids, mask=mask)``.
Same shape rules for ``targets`` against ``loss_fn``.

The SPMD path keeps a small per-``GraphDef`` cache for the placed
state and the jitted forward / train step, so repeated calls
(notably autoregressive decode loops) reuse the compiled program.
The MPMD path requires exactly one positional input (microbatched
along its leading axis) and routes train-mode calls through
:func:`spectrax.runtime.sxcall`, whose train wrapper uses the same
schedule-faithful ``sxjit`` / ``sxvalue_and_grad`` dispatcher as the
lower-level MPMD API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import jax

from .core._weakcache import weak_invalidate
from .core.graph import bind, export
from .core.module import Module
from .core.paths import str_to_path
from .core.state import State, _nested_set
from .runtime.mpmd.runtime import sxcall
from .runtime.schedules import GPipe as _DefaultGPipe
from .sharding.mesh import SpxMesh
from .sharding.partition import get_named_sharding

__all__ = ["run"]

_SPMD_FWD_CACHE: dict[int, Callable] = {}
_SPMD_TRAIN_CACHE: dict[tuple, Callable] = {}
_SPMD_STATE_CACHE: dict[tuple[object, ...], State] = {}


def _as_call(payload: object) -> tuple[tuple[object, ...], dict[str, object]]:
    """Normalize an ``inputs`` / ``targets`` payload into ``(args, kwargs)``.

    Implements the three accepted call shapes documented on
    :func:`run` (Option C):

    * ``None``                -> ``((), {})``
    * single array            -> ``((array,), {})``
    * tuple / list of values  -> ``(tuple(...), {})``
    * dict                    -> ``((), dict(...))``

    Args:
        payload: The raw value passed as ``inputs=`` or ``targets=``.

    Returns:
        A ``(args, kwargs)`` pair ready to splat into ``model(...)``
        or ``loss_fn(out, ...)``.
    """
    if payload is None:
        return (), {}
    if isinstance(payload, Mapping):
        return (), dict(payload)
    if isinstance(payload, (tuple, list)):
        return tuple(payload), {}
    return (payload,), {}


def _place_state(state: State, model: Module, mesh: object) -> State:
    """Place every leaf of a :class:`State` onto its target device sharding.

    Asks the model for a ``{collection: {path: NamedSharding}}`` map
    via :func:`spectrax.sharding.partition.get_named_sharding`, then
    walks the state and calls :func:`jax.device_put` on each leaf for
    which a sharding is registered. Leaves with no registered sharding
    pass through untouched.

    Args:
        state: The :class:`State` returned by :func:`spectrax.export`.
        model: The live module used to resolve logical axis-name
            annotations against the active logical-axis rules context.
        mesh: A :class:`jax.sharding.Mesh` to bind the per-leaf
            ``NamedSharding`` s to.

    Returns:
        A new :class:`State` of the same type and shape as ``state``,
        with each leaf placed onto its requested sharding.
    """
    shards = get_named_sharding(model, mesh)
    out: dict[str, dict[str, object]] = {}
    for col, path, leaf in state.items():
        sh = shards.get(col, {}).get(path)
        placed = jax.device_put(leaf, sh) if sh is not None else leaf
        _nested_set(out.setdefault(col, {}), str_to_path(path), placed)
    return type(state)._from_raw(out, writers=state._writers)


def _mpmd_dummy_loss(y: object, *_a: object) -> jax.Array:
    """Stable dummy loss used when MPMD ``mode='forward'`` needs a loss arg.

    :func:`spectrax.runtime.sxcall` always takes a ``loss_fn`` parameter;
    in forward mode we pass this fixed zero-returning function so that
    the cache key for the compiled program stays constant across
    repeated forward calls (it is defined at module scope so ``id()``
    does not vary between calls).

    Args:
        y: The model output (ignored).
        *_a: Any additional positional args (ignored).

    Returns:
        A scalar ``float32`` zero.
    """
    return jax.numpy.zeros((), dtype=jax.numpy.float32)


def _run_spmd(
    model: Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    mesh: SpxMesh,
    mode: str,
    loss_args: tuple[object, ...],
    loss_kwargs: dict[str, object],
    loss_fn: Callable[..., object] | None,
):
    """Run ``model`` under pure SPMD (``pjit``) — forward-only or train.

    Exports the model, places the resulting :class:`State` onto the
    mesh, and dispatches either a jitted forward or a jitted
    :func:`jax.value_and_grad` ``(loss, grads)`` step. Jit-compiled
    functions and the placed state are cached per-``GraphDef`` so
    repeat callers (e.g. an autoregressive decode loop or a training
    step in a Python ``for`` loop) reuse the compiled program and
    avoid redundant :func:`_place_state` walks. Cache entries are
    weakly invalidated when the underlying ``GraphDef``, mesh, or
    ``loss_fn`` go out of scope.

    Args:
        model: The :class:`Module` to run.
        args: Positional arguments for ``model.forward``.
        kwargs: Keyword arguments for ``model.forward``.
        mesh: The :class:`SpxMesh` whose JAX mesh to enter for
            execution.
        mode: ``"forward"`` or ``"train"``.
        loss_args: Positional arguments forwarded to ``loss_fn``
            after the model output.
        loss_kwargs: Keyword arguments forwarded to ``loss_fn``.
        loss_fn: Required for ``mode='train'``; called as
            ``loss_fn(out, *loss_args, **loss_kwargs)``.

    Returns:
        The model output (forward) or ``(loss, grads)`` (train).

    Raises:
        ValueError: ``mode='train'`` was requested but no ``loss_fn``
            was provided.
    """
    gdef, state = export(model)
    jax_mesh = mesh.jax_mesh

    state_key = (id(gdef), id(jax_mesh))
    cached_state = _SPMD_STATE_CACHE.get(state_key)
    if cached_state is not None:
        state = cached_state
    else:
        state = _place_state(state, model, jax_mesh)
        _SPMD_STATE_CACHE[state_key] = state
        weak_invalidate(gdef, _SPMD_STATE_CACHE, state_key)
        weak_invalidate(jax_mesh, _SPMD_STATE_CACHE, state_key)

    if mode == "forward":
        cache_key = id(gdef)
        _fwd = _SPMD_FWD_CACHE.get(cache_key)
        if _fwd is None:

            @jax.jit
            def _fwd(state, *a, **kw):
                """Jitted forward: rebind ``gdef`` to ``state`` and call the module.

                Cached per-``GraphDef`` so the same compiled program is
                reused across calls.

                Args:
                    state: SpectraX state tree or transform state passed into the operation.
                    *a: Additional positional arguments forwarded to the wrapped callable or backend.
                    **kw: Additional keyword arguments forwarded to the wrapped callable or backend.
                """
                return bind(gdef, state)(*a, **kw)

            _SPMD_FWD_CACHE[cache_key] = _fwd
            weak_invalidate(gdef, _SPMD_FWD_CACHE, cache_key)

        with jax_mesh:
            return _fwd(state, *args, **kwargs)

    if loss_fn is None:
        raise ValueError("loss_fn required for mode='train'.")

    train_key = (id(gdef), id(loss_fn))
    _step = _SPMD_TRAIN_CACHE.get(train_key)
    if _step is None:

        @jax.jit
        def _step(state, args, kwargs, l_args, l_kwargs):
            """One jitted train step: ``(loss, grads)`` via :func:`jax.value_and_grad`.

            Differentiates with respect to the full :class:`State`
            tree; partition out trainable subsets at the call site if
            you want narrower gradients.

            Args:
                state: SpectraX state tree or transform state passed into the operation.
                args: Positional arguments forwarded to the wrapped callable.
                kwargs: Keyword arguments forwarded to the wrapped callable.
                l_args: L args value consumed by this operation.
                l_kwargs: L kwargs value consumed by this operation.
            """

            def loss(state):
                """Forward + loss closure used as the differentiation target.

                Captures ``gdef``, the model call args, and the
                supplied ``loss_fn`` from the enclosing scope so the
                resulting function depends only on ``state``.

                Args:
                    state: SpectraX state tree or transform state passed into the operation.
                """
                out = bind(gdef, state)(*args, **kwargs)
                return loss_fn(out, *l_args, **l_kwargs)

            return jax.value_and_grad(loss)(state)

        _SPMD_TRAIN_CACHE[train_key] = _step
        weak_invalidate(gdef, _SPMD_TRAIN_CACHE, train_key)
        weak_invalidate(loss_fn, _SPMD_TRAIN_CACHE, train_key)

    with jax_mesh:
        loss_val, grads = _step(state, args, kwargs, loss_args, loss_kwargs)
    return loss_val, grads


def _run_mpmd(
    model: Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    mesh: SpxMesh,
    mode: str,
    loss_args: tuple[object, ...],
    loss_kwargs: dict[str, object],
    loss_fn: Callable[..., object] | None,
    microbatches: int,
    schedule: object = None,
    fuse_1f1b: bool | None = None,
    fuse_zb: bool | None = None,
    has_aux: bool = False,
):
    """Auto-split ``model`` into per-rank stages and dispatch via :func:`sxcall`.

    Whether or not ``schedule`` is supplied the call routes through
    :func:`spectrax.runtime.sxcall`. In train mode that public wrapper
    lowers the model to scheduled ``sxjit`` and uses the true
    schedule-faithful MPMD forward/backward dispatcher. When
    ``schedule`` is ``None``, a default :class:`GPipe` schedule with
    ``microbatches`` microbatches is used.

    ``fuse_1f1b`` / ``fuse_zb`` are legacy schedule-walker knobs.  The
    true scheduled MPMD path rejects explicit ``True`` values; use a
    schedule that emits the desired fused cells directly.

    For ``mode='train'`` with keyword loss targets, the ``loss_fn`` is
    rewrapped to take its target arguments positionally so they can
    flow through the pipeline batch tuple.

    Args:
        model: The :class:`Module` to split and run.
        args: Positional inputs; exactly one positional input is
            required (microbatched along its leading axis).
        kwargs: Keyword inputs (currently unused under MPMD).
        mesh: An MPMD-flavoured :class:`SpxMesh`.
        mode: ``"forward"`` or ``"train"``.
        loss_args: Positional loss-target arguments.
        loss_kwargs: Keyword loss-target arguments.
        loss_fn: Required for ``mode='train'``.
        microbatches: Microbatch count for the default :class:`GPipe`
            schedule when ``schedule`` is ``None``. Forced to be at
            least 1.
        schedule: Optional pipeline-parallel schedule.
        fuse_1f1b: Legacy schedule-walker knob. Explicit ``True`` is
            rejected by the true scheduled MPMD path.
        fuse_zb: Legacy schedule-walker knob. Explicit ``True`` is
            rejected by the true scheduled MPMD path.
        has_aux: Whether ``loss_fn`` returns ``(loss, aux)``; forwarded
            to :func:`sxcall`.

    Returns:
        Whatever :func:`sxcall` returns for the chosen mode.

    Raises:
        ValueError: ``mode`` is not ``"forward"`` or ``"train"``,
            ``loss_fn`` is missing in train mode, or no positional
            input was supplied.
    """
    if mode not in {"forward", "train"}:
        raise ValueError(f"mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "train" and loss_fn is None:
        raise ValueError("loss_fn required for mode='train'.")
    if not args:
        raise ValueError("MPMD execution requires at least one positional input batch.")

    if loss_fn is not None and loss_kwargs:
        target_keys = tuple(loss_kwargs.keys())
        target_vals = tuple(loss_kwargs.values())
        original_loss = loss_fn

        def _wrapped_loss(out: object, *vals: object) -> object:
            """Re-key positional pipeline targets back to keyword args for ``loss_fn``.

            Pairs each value in ``vals`` with the corresponding key in
            the captured ``target_keys`` and calls the original
            ``loss_fn`` with keyword targets, preserving its kwargs
            interface even though the pipeline batch is positional.

            Args:
                out: Output value from an earlier call or transform.
                *vals: Additional positional arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            return original_loss(out, **dict(zip(target_keys, vals, strict=True)))

        loss_fn = _wrapped_loss
        loss_args = loss_args + target_vals

    if schedule is not None:
        batch = (args[0], *tuple(loss_args)) if mode == "train" else (args[0],)
        return sxcall(
            model,
            batch,
            mesh=mesh.mpmd_mesh,
            schedule=schedule,
            loss_fn=loss_fn,
            fuse_1f1b=fuse_1f1b,
            fuse_zb=fuse_zb,
            mode=mode,
            has_aux=has_aux,
        )

    batch = (args[0], *tuple(loss_args)) if mode == "train" else (args[0],)
    return sxcall(
        model,
        batch,
        mesh=mesh.mpmd_mesh,
        schedule=_DefaultGPipe(microbatches=max(microbatches, 1)),
        loss_fn=loss_fn if mode == "train" else _mpmd_dummy_loss,
        fuse_1f1b=fuse_1f1b,
        fuse_zb=fuse_zb,
        mode=mode,
        has_aux=has_aux,
    )


def run(
    model: Module,
    *,
    inputs: object,
    targets: object = None,
    mesh: SpxMesh,
    mode: Literal["train", "forward"] = "forward",
    loss_fn: Callable[..., object] | None = None,
    microbatches: int = 1,
    schedule: object = None,
    fuse_1f1b: bool | None = None,
    fuse_zb: bool | None = None,
    has_aux: bool = False,
) -> object:
    """Run a model under SPMD or MPMD — the mesh decides which.

    Two modes, because that's all there is:

    * ``"forward"`` — run the model, no autograd. This is both
      inference and the decode primitive: to generate, call
      ``"forward"`` in a loop and feed the returned per-stage state
      back in next step (KV cache, beam records, whatever lives in
      stage state).
    * ``"train"`` — forward + ``loss_fn`` + backward. Returns
      ``(loss, grads)``.

    Args:
        model: An :class:`spectrax.Module`. For MPMD, the model is
            auto-split into per-rank stages — either via
            ``model.pipeline_split(n_pp)`` if defined or by detecting
            a ``blocks: ModuleList`` attribute.
        inputs: Forward arguments for ``model.forward``. Accepts:

            * a single array -> ``forward(array)``
            * tuple/list      -> ``forward(*payload)``
            * dict            -> ``forward(**payload)``

        targets: Loss targets passed to ``loss_fn`` after the output.
            Same shape rules as ``inputs``. Required for ``mode="train"``;
            forbidden in ``mode="forward"``.
        mesh: An :class:`SpxMesh` (built via :func:`spectrax.create_mesh`).
            ``mesh.is_mpmd`` decides the path:

            * ``False`` -> pjit (pure SPMD, FSDP/TP via the model's
              :func:`logical_axis_rules` annotations).
            * ``True``  -> split the model and call
              :func:`sxcall` (PP x FSDP x TP).

        mode: ``"forward"`` or ``"train"``.
        loss_fn: Required for ``mode="train"``. Called as
            ``loss_fn(output, *target_args, **target_kwargs)``.
        microbatches: Pipeline microbatch count. Ignored for SPMD.
        schedule: Pipeline schedule for MPMD execution. Pass ``None``
            to use a default :class:`GPipe`. Must be ``None`` under
            an SPMD mesh.
        fuse_1f1b: Legacy schedule-walker knob. Explicit ``True`` is
            rejected on the true scheduled MPMD path.
        fuse_zb: Legacy schedule-walker knob. Explicit ``True`` is
            rejected on the true scheduled MPMD path.
        has_aux: Whether ``loss_fn`` returns ``(loss, aux)`` (MPMD only).

    Returns:
        * ``mode="forward"``  -> ``output`` (same shape under SPMD and MPMD).
        * ``mode="train"``    -> ``(loss_scalar, grads)``. Under SPMD ``grads``
          is a single State; under MPMD it's a ``tuple[per_rank_State]``.

    Raises:
        TypeError: ``mesh`` is not an :class:`SpxMesh`.
        ValueError: ``mode`` is invalid; ``targets`` was supplied with
            ``mode="forward"``; an MPMD mesh was given more than one
            positional input or any keyword inputs; or ``schedule=`` was
            supplied alongside an SPMD mesh.

    Note:
        For MPMD decode that needs the per-rank stage state out (KV cache
        threading), drop to :func:`sxcall` directly — its
        ``mode='forward'`` returns ``(output, tuple[per_rank_state])``.
        ``spx.run`` hides that to keep its return shape uniform across
        SPMD and MPMD.
    """
    if not isinstance(mesh, SpxMesh):
        raise TypeError(f"mesh must be an SpxMesh (build via spx.create_mesh); got {type(mesh).__name__}.")
    if mode not in {"forward", "train"}:
        raise ValueError(f"mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "forward" and targets is not None:
        raise ValueError("targets are only valid for mode='train'; forward mode does not consume loss targets.")

    args, kwargs = _as_call(inputs)
    loss_args, loss_kwargs = _as_call(targets)

    if mesh.is_mpmd:
        if len(args) != 1 or kwargs:
            raise ValueError(
                "MPMD (pipeline) execution requires exactly one positional "
                "model input, microbatched along its leading axis; got "
                f"{len(args)} positional and {len(kwargs)} keyword inputs. "
                "Pass `inputs=<single_array>` (a bare array or a 1-tuple). "
                "Multi-input / dict-shaped inputs are not supported under "
                "MPMD yet — use an SPMD mesh or fold the inputs into one "
                "batched array."
            )
        return _run_mpmd(
            model,
            args,
            kwargs,
            mesh=mesh,
            mode=mode,
            loss_args=loss_args,
            loss_kwargs=loss_kwargs,
            loss_fn=loss_fn,
            microbatches=microbatches,
            schedule=schedule,
            fuse_1f1b=fuse_1f1b,
            fuse_zb=fuse_zb,
            has_aux=has_aux,
        )
    if schedule is not None:
        raise ValueError(
            "schedule= is only meaningful with an MPMD mesh "
            "(create_mesh(mpmd_axis=...)); got a pure SPMD mesh. "
            "SPMD has no pipeline schedule — drop schedule= or use "
            "an MPMD mesh."
        )
    return _run_spmd(
        model,
        args,
        kwargs,
        mesh=mesh,
        mode=mode,
        loss_args=loss_args,
        loss_kwargs=loss_kwargs,
        loss_fn=loss_fn,
    )
