import importlib.util
import sys
from pathlib import Path
from typing import get_args

import jax
import jax.numpy as jnp
import optax
import pytest
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from eformer.optimizers._base import _OPTIMIZER_BUILDER_REGISTRY

from easydel.infra.base_state import EasyDeLState
from easydel.infra.etils import AVAILABLE_OPTIMIZERS, EasyDeLOptimizers


def _assert_tree_allclose(actual, expected, *, atol=1e-6, rtol=1e-6):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert jnp.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def _load_fused_optimizers_module():
    module_name = "easydel.trainers.fused_optimizers"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    module_path = Path(__file__).parents[2] / "easydel" / "trainers" / "fused_optimizers.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _small_params_and_grads():
    params = {
        "layers": {
            "0": {
                "mlp": {
                    "weight": jnp.array(
                        [[0.2, -0.7, 0.4, 1.1], [1.3, 0.4, -0.6, 0.8]],
                        dtype=jnp.float32,
                    ),
                    "bias": jnp.array([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32),
                }
            }
        }
    }
    grads = jax.tree_util.tree_map(lambda param: jnp.sin(param) + 0.1, params)
    return params, grads


def _optimizer_smoke_config(module, optimizer_name):
    if optimizer_name in {"skew", "quad"}:
        from eformer.optimizers import WhiteKronConfig

        return WhiteKronConfig(dtype=jnp.float32, block_size=2, noise_scale=0.0)
    if optimizer_name == "prism":
        return module.PrismConfig(
            group_size=2,
            norm_group_size=2,
            single_sided_prism_group_size=2,
            single_sided_prism_dtype=jnp.float32,
            momentum_dtype=jnp.float32,
            newton_schulz_dtype=jnp.float32,
        )
    if optimizer_name == "prism_hyperball":
        return module.HyperballConfig(
            prism_group_size=2,
            prism_norm_group_size=2,
            prism_grouped_size=2,
            prism_dtype=jnp.float32,
            prism_momentum_dtype=jnp.float32,
            prism_newton_schulz_dtype=jnp.float32,
        )
    return None


def test_research_optimizer_names_are_easy_del_options():
    expected_names = {
        "normuon",
        "novograd",
        "yogi",
        "prism",
        "schedule_free_plus",
        "hyperball",
        "adamw_hyperball",
        "muon_hyperball",
        "normuon_hyperball",
        "novograd_hyperball",
        "yogi_hyperball",
        "mars_hyperball",
        "prism_hyperball",
        "schedule_free_plus_hyperball",
    }

    literal_names = set(get_args(AVAILABLE_OPTIMIZERS))
    enum_names = {optimizer.value for optimizer in EasyDeLOptimizers}

    assert expected_names <= literal_names
    assert expected_names <= enum_names


def test_all_easy_del_optimizer_names_are_available_options():
    assert {optimizer.value for optimizer in EasyDeLOptimizers} <= set(get_args(AVAILABLE_OPTIMIZERS))


@pytest.mark.parametrize("optimizer_name", [optimizer.value for optimizer in EasyDeLOptimizers])
def test_every_easy_del_optimizer_builds_and_applies_one_finite_step(optimizer_name):
    module = _load_fused_optimizers_module()
    params, grads = _small_params_and_grads()
    optimizer_config = _optimizer_smoke_config(module, optimizer_name)

    assert optimizer_name in _OPTIMIZER_BUILDER_REGISTRY
    tx, scheduler = OptimizerFactory.create(
        optimizer_name,
        SchedulerConfig(learning_rate=1e-3),
        optimizer_config,
    )
    state = tx.init(params)
    updates, next_state = tx.update(grads, state, params)
    next_params = optax.apply_updates(params, updates)

    assert isinstance(tx, optax.GradientTransformation)
    assert scheduler(0) == 1e-3
    assert next_state is not None
    assert jnp.isfinite(optax.global_norm(updates))
    assert jnp.isfinite(optax.global_norm(next_params))


def test_research_optimizers_register_and_build():
    _load_fused_optimizers_module()

    scheduler_config = SchedulerConfig(learning_rate=1e-3)
    optimizer_names = [
        "normuon",
        "novograd",
        "yogi",
        "prism",
        "schedule_free_plus",
        "hyperball",
        "adamw_hyperball",
        "muon_hyperball",
        "normuon_hyperball",
        "novograd_hyperball",
        "yogi_hyperball",
        "mars_hyperball",
        "prism_hyperball",
        "schedule_free_plus_hyperball",
    ]

    for optimizer_name in optimizer_names:
        assert optimizer_name in _OPTIMIZER_BUILDER_REGISTRY
        tx, scheduler = OptimizerFactory.create(optimizer_name, scheduler_config)
        assert isinstance(tx, optax.GradientTransformation)
        assert scheduler(0) == 1e-3


def test_novograd_and_yogi_match_optax_when_extras_disabled():
    module = _load_fused_optimizers_module()

    novograd_scheduler = optax.constant_schedule(0.003)
    novograd_params = {"w": jnp.array([[0.2, -0.7], [1.3, 0.4]], dtype=jnp.float32)}
    novograd_grads = {"w": jnp.sin(novograd_params["w"]) + 0.1}
    novograd_config = module.NovoGradConfig(b1=0.83, b2=0.91, eps=1e-7, eps_root=1e-12, grad_averaging=False)
    novograd_ref = optax.novograd(
        learning_rate=novograd_scheduler,
        b1=novograd_config.b1,
        b2=novograd_config.b2,
        eps=novograd_config.eps,
        eps_root=novograd_config.eps_root,
    )
    novograd_new = module.NovoGradOptimizer(config=novograd_config).build(novograd_scheduler)
    novograd_ref_state = novograd_ref.init(novograd_params)
    novograd_new_state = novograd_new.init(novograd_params)

    novograd_ref_updates, _novograd_ref_state = novograd_ref.update(
        novograd_grads,
        novograd_ref_state,
        novograd_params,
    )
    novograd_new_updates, _novograd_new_state = novograd_new.update(
        novograd_grads,
        novograd_new_state,
        novograd_params,
    )

    _assert_tree_allclose(novograd_new_updates, novograd_ref_updates)

    yogi_scheduler = optax.constant_schedule(0.002)
    yogi_params = {"w": jnp.array([0.2, -0.7, 1.3], dtype=jnp.float32)}
    yogi_grads = {"w": jnp.sin(yogi_params["w"]) + 0.1}
    yogi_config = module.YogiConfig(b1=0.84, b2=0.96, eps=1e-4)
    yogi_ref = optax.yogi(learning_rate=yogi_scheduler, b1=yogi_config.b1, b2=yogi_config.b2, eps=yogi_config.eps)
    yogi_new = module.YogiOptimizer(config=yogi_config).build(yogi_scheduler)
    yogi_ref_state = yogi_ref.init(yogi_params)
    yogi_new_state = yogi_new.init(yogi_params)

    yogi_ref_updates, _yogi_ref_state = yogi_ref.update(yogi_grads, yogi_ref_state, yogi_params)
    yogi_new_updates, _yogi_new_state = yogi_new.update(yogi_grads, yogi_new_state, yogi_params)

    _assert_tree_allclose(yogi_new_updates, yogi_ref_updates)


def test_prism_optimizer_updates_matrix_and_tracks_group_stats():
    module = _load_fused_optimizers_module()
    scheduler = optax.constant_schedule(0.01)
    params = {
        "layers": {
            "0": {
                "mlp": {
                    "weight": jnp.array(
                        [[0.2, -0.7, 0.4, 1.1], [1.3, 0.4, -0.6, 0.8]],
                        dtype=jnp.float32,
                    )
                }
            }
        }
    }
    grads = jax.tree_util.tree_map(lambda param: jnp.sin(param) + 0.1, params)
    tx = module.PrismOptimizer(
        config=module.PrismConfig(
            group_size=2,
            norm_group_size=2,
            single_sided_prism_group_size=2,
            single_sided_prism_dtype=jnp.float32,
            momentum_dtype=jnp.float32,
            newton_schulz_dtype=jnp.float32,
        )
    ).build(scheduler)

    state = tx.init(params)
    updates, next_state = tx.update(grads, state, params)

    assert updates["layers"]["0"]["mlp"]["weight"].shape == params["layers"]["0"]["mlp"]["weight"].shape
    assert not jnp.allclose(updates["layers"]["0"]["mlp"]["weight"], jnp.zeros_like(updates["layers"]["0"]["mlp"]["weight"]))
    assert next_state.norm_second_moment["layers"]["0"]["mlp"]["weight"].shape == (4, 1)


def test_research_optimizers_converge_on_small_linear_regression():
    module = _load_fused_optimizers_module()
    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (128, 4))
    target_weight = jnp.array(
        [[0.8, -0.3, 0.4], [-0.7, 0.2, 0.9], [0.1, -0.5, 0.6], [0.4, 0.3, -0.2]],
        dtype=jnp.float32,
    )
    targets = inputs @ target_weight
    initial_params = {"dense": {"weight": jnp.zeros_like(target_weight)}}

    def loss_fn(params):
        predictions = inputs @ params["dense"]["weight"]
        return 0.5 * jnp.mean((predictions - targets) ** 2)

    def final_loss(optimizer_name, learning_rate, steps, optimizer_config=None):
        tx, _scheduler = OptimizerFactory.create(
            optimizer_name,
            SchedulerConfig(learning_rate=learning_rate),
            optimizer_config,
        )
        initial_state = tx.init(initial_params)

        def body(carry, _):
            params, state = carry
            _loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = tx.update(grads, state, params)
            return (optax.apply_updates(params, updates), state), None

        (params, _state), _ = jax.jit(lambda: jax.lax.scan(body, (initial_params, initial_state), None, length=steps))()
        return loss_fn(params)

    start_loss = loss_fn(initial_params)
    prism_config = module.PrismConfig(
        group_size=4,
        norm_group_size=4,
        single_sided_prism_group_size=4,
        momentum_dtype=jnp.float32,
        newton_schulz_dtype=jnp.float32,
        single_sided_prism_dtype=jnp.float32,
    )
    runs = [
        ("novograd", 3e-2, 400, None, 1e-2),
        ("yogi", 3e-2, 400, None, 1e-2),
        ("normuon", 1e-2, 400, None, 1e-2),
        ("schedule_free_plus", 3e-2, 400, module.ScheduleFreePlusConfig(beta_final=None, r=0.0), 1e-3),
        ("prism", 3e-3, 1000, prism_config, 1e-3),
    ]

    for optimizer_name, learning_rate, steps, optimizer_config, max_ratio in runs:
        end_loss = final_loss(optimizer_name, learning_rate, steps, optimizer_config)
        assert end_loss < start_loss * max_ratio, optimizer_name


def test_schedule_free_plus_polyak_converges_and_exposes_eval_params():
    module = _load_fused_optimizers_module()
    key = jax.random.PRNGKey(11)
    inputs = jax.random.normal(key, (128, 4))
    target_weight = jnp.array(
        [[0.8, -0.3, 0.4], [-0.7, 0.2, 0.9], [0.1, -0.5, 0.6], [0.4, 0.3, -0.2]],
        dtype=jnp.float32,
    )
    targets = inputs @ target_weight
    initial_params = {"dense": {"weight": jnp.zeros_like(target_weight)}}

    def loss_fn(params):
        predictions = inputs @ params["dense"]["weight"]
        return 0.5 * jnp.mean((predictions - targets) ** 2)

    config = module.ScheduleFreePlusConfig(polyak=True, beta_final=None, r=0.0)
    tx, _scheduler = OptimizerFactory.create("schedule_free_plus", SchedulerConfig(learning_rate=1.0), config)
    initial_state = tx.init(initial_params)

    def body(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = tx.update(grads, state, params, value=loss)
        return (optax.apply_updates(params, updates), state), None

    (params, state), _ = jax.jit(lambda: jax.lax.scan(body, (initial_params, initial_state), None, length=300))()
    eval_params = module.schedule_free_plus_eval_params(state)
    train_params = module.schedule_free_plus_train_params(
        state,
        beta=config.beta,
        beta_final=config.beta_final,
        beta_anneal_steps=config.beta_anneal_steps,
    )
    start_loss = loss_fn(initial_params)

    _assert_tree_allclose(train_params, params, atol=1e-5, rtol=1e-5)
    assert loss_fn(params) < start_loss * 1e-2
    assert loss_fn(eval_params) < start_loss * 1e-2
    assert jnp.isfinite(optax.global_norm(eval_params))


def test_schedule_free_plus_beta_anneal_is_zero_based():
    module = _load_fused_optimizers_module()

    assert jnp.allclose(
        module._schedule_free_beta(jnp.asarray(0), beta=0.9, beta_final=0.965, beta_anneal_steps=10),
        0.9,
    )
    assert jnp.allclose(
        module._schedule_free_beta(jnp.asarray(10), beta=0.9, beta_final=0.965, beta_anneal_steps=10),
        0.965,
    )


def test_easy_del_state_apply_gradients_ignores_extra_args_for_plain_optax():
    params = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    tx = optax.sgd(0.1)
    state = EasyDeLState(
        step=0,
        graphdef=None,
        graphstate=params,
        graphother={},
        tx=tx,
        opt_state=tx.init(params),
    )

    state = state.apply_gradients(
        grads={"w": jnp.asarray([0.5], dtype=jnp.float32)},
        optimizer_extra_args={"loss": jnp.asarray(1.0, dtype=jnp.float32)},
    )

    assert state.step == 1
    assert jnp.allclose(state.graphstate["w"], jnp.asarray([0.95], dtype=jnp.float32))


_HYPERBALL_BASES = ("adamw", "muon", "normuon", "novograd", "yogi", "mars", "prism", "schedule_free_plus")
_HYPERBALL_ALIAS_CASES = tuple((f"{base}_hyperball", None) for base in _HYPERBALL_BASES)
_HYPERBALL_GENERIC_CASES = tuple(("hyperball", base) for base in _HYPERBALL_BASES)


def _hyperball_config(module, base_optimizer):
    config_kwargs = {
        "min_param_ndim": 2,
        "prism_group_size": 2,
        "prism_norm_group_size": 2,
        "prism_grouped_size": 2,
        "prism_dtype": jnp.float32,
        "prism_momentum_dtype": jnp.float32,
        "prism_newton_schulz_dtype": jnp.float32,
    }
    if base_optimizer is not None:
        config_kwargs["base_optimizer"] = base_optimizer
    if base_optimizer == "schedule_free_plus":
        config_kwargs.update(
            schedule_free_beta=0.0,
            schedule_free_beta_final=None,
            schedule_free_r=0.0,
            schedule_free_b2=0.999,
        )
    return module.HyperballConfig(**config_kwargs)


def _schedule_free_plus_hyperball_config(module):
    return module.HyperballConfig(
        min_param_ndim=2,
        schedule_free_beta=0.0,
        schedule_free_beta_final=None,
        schedule_free_r=0.0,
        schedule_free_b2=0.999,
    )


@pytest.mark.parametrize(
    ("optimizer_name", "base_optimizer"),
    [*_HYPERBALL_ALIAS_CASES, *_HYPERBALL_GENERIC_CASES],
    ids=[
        *(f"alias-{optimizer_name}" for optimizer_name, _base_optimizer in _HYPERBALL_ALIAS_CASES),
        *(f"generic-{base_optimizer}" for _optimizer_name, base_optimizer in _HYPERBALL_GENERIC_CASES),
    ],
)
def test_hyperball_variants_project_selected_matrices_only(optimizer_name, base_optimizer):
    module = _load_fused_optimizers_module()
    scheduler_config = SchedulerConfig(learning_rate=0.1)
    params = {
        "matrix": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        "vector": jnp.array([1.0, -1.0], dtype=jnp.float32),
    }
    grads = {
        "matrix": jnp.array([[0.3, -0.2], [0.1, 0.4]], dtype=jnp.float32),
        "vector": jnp.array([0.25, -0.5], dtype=jnp.float32),
    }

    tx, _scheduler = OptimizerFactory.create(
        optimizer_name,
        scheduler_config,
        _hyperball_config(module, base_optimizer),
        weight_decay=0.1,
    )
    state = tx.init(params)
    updates, _state = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.allclose(jnp.linalg.norm(new_params["matrix"]), jnp.linalg.norm(params["matrix"]), rtol=1e-6)
    assert jnp.isfinite(optax.global_norm(updates))
    assert not jnp.allclose(new_params["vector"], params["vector"])


def test_hyperball_default_routing_excludes_embedding_and_routes_lm_head_to_adamh_for_muon():
    module = _load_fused_optimizers_module()
    params = {
        "token_embedding": {"weight": jnp.ones((2, 2), dtype=jnp.float32)},
        "layers": {"0": {"mlp": {"weight": jnp.ones((2, 2), dtype=jnp.float32)}}},
        "lm_head": {"weight": jnp.ones((2, 2), dtype=jnp.float32)},
    }

    hyperball_mask = module._default_hyperball_mask(params, min_param_ndim=2)
    muon_dims = module._default_muon_hyperball_dimension_numbers(params)

    assert hyperball_mask["token_embedding"]["weight"] is False
    assert hyperball_mask["layers"]["0"]["mlp"]["weight"] is True
    assert hyperball_mask["lm_head"]["weight"] is True
    assert muon_dims["token_embedding"]["weight"] is None
    assert muon_dims["lm_head"]["weight"] is None
    assert isinstance(muon_dims["layers"]["0"]["mlp"]["weight"], optax.contrib.MuonDimensionNumbers)


def test_hyperball_default_mask_leaves_embedding_norm_unconstrained():
    params = {
        "token_embedding": {"weight": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)},
        "layers": {"0": {"mlp": {"weight": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)}}},
    }
    grads = jax.tree_util.tree_map(lambda param: param, params)

    tx, _scheduler = OptimizerFactory.create("adamw_hyperball", SchedulerConfig(learning_rate=0.1))
    state = tx.init(params)
    updates, _state = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert not jnp.allclose(
        jnp.linalg.norm(new_params["token_embedding"]["weight"]),
        jnp.linalg.norm(params["token_embedding"]["weight"]),
    )
    assert jnp.allclose(
        jnp.linalg.norm(new_params["layers"]["0"]["mlp"]["weight"]),
        jnp.linalg.norm(params["layers"]["0"]["mlp"]["weight"]),
        rtol=1e-6,
    )


def test_hyperball_preserves_stacked_matrix_norms_per_slice():
    params = {
        "layers": {
            "mlp": {
                "weight": jnp.arange(1, 9, dtype=jnp.float32).reshape(2, 2, 2),
            }
        }
    }
    grads = jax.tree_util.tree_map(lambda param: jnp.sin(param) + 0.1, params)

    tx, _scheduler = OptimizerFactory.create("adamw_hyperball", SchedulerConfig(learning_rate=0.1))
    state = tx.init(params)
    updates, _state = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    old_norms = jnp.linalg.norm(params["layers"]["mlp"]["weight"], axis=(1, 2))
    new_norms = jnp.linalg.norm(new_params["layers"]["mlp"]["weight"], axis=(1, 2))

    assert jnp.allclose(new_norms, old_norms, rtol=1e-6)


def test_hyperball_updates_descent_space_then_maps_to_weight_space():
    module = _load_fused_optimizers_module()
    params = {"matrix": jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)}
    grads = {"matrix": jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)}

    tx = module.hyperball(optax.sgd(1.0), 0.1, mask=True, min_param_ndim=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    updates, state = tx.update(grads, state, params)
    params = optax.apply_updates(params, updates)

    first_trial = jnp.array([[1.0, -0.1], [0.0, 0.0]], dtype=jnp.float32)
    first_z = first_trial / jnp.linalg.norm(first_trial)
    second_trial = first_z + jnp.array([[0.0, -0.1], [0.0, 0.0]], dtype=jnp.float32)
    expected_z = {"matrix": second_trial / jnp.linalg.norm(second_trial)}

    _assert_tree_allclose(state.z, expected_z)
    _assert_tree_allclose(params, expected_z)
    assert jnp.allclose(jnp.linalg.norm(state.z["matrix"]), jnp.asarray(1.0, dtype=jnp.float32), rtol=1e-6)


def test_schedule_free_plus_hyperball_accepts_loss_extra_args():
    module = _load_fused_optimizers_module()
    params = {"matrix": jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)}
    target = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    def loss_fn(params):
        matrix_direction = params["matrix"] / jnp.linalg.norm(params["matrix"])
        target_direction = target / jnp.linalg.norm(target)
        return 0.5 * jnp.sum((matrix_direction - target_direction) ** 2)

    config = module.HyperballConfig(
        schedule_free_polyak=True,
        schedule_free_beta_final=None,
        schedule_free_r=0.0,
    )
    tx, _scheduler = OptimizerFactory.create("schedule_free_plus_hyperball", SchedulerConfig(learning_rate=1.0), config)
    state = tx.init(params)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, state = tx.update(grads, state, params, value=loss)
    params = optax.apply_updates(params, updates)

    assert state is not None
    assert jnp.isfinite(loss_fn(params))
    assert jnp.allclose(jnp.linalg.norm(params["matrix"]), jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32)), rtol=1e-6)


def test_schedule_free_plus_hyperball_projects_x_y_and_z_lerps():
    module = _load_fused_optimizers_module()
    params = {"matrix": jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)}
    grads = {"matrix": jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)}
    radius = jnp.linalg.norm(params["matrix"])

    tx = module.schedule_free_plus_hyperball_adamw(
        0.1,
        mask=True,
        beta=0.5,
        beta_final=None,
        r=0.0,
        b2=0.999,
    )
    state = tx.init(params)

    for _ in range(3):
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        train_params = module.schedule_free_plus_train_params(state, beta=0.5, beta_final=None)

        assert jnp.allclose(jnp.linalg.norm(state.z["matrix"]), radius, rtol=1e-6)
        assert jnp.allclose(jnp.linalg.norm(state.x["matrix"]), radius, rtol=1e-6)
        assert jnp.allclose(jnp.linalg.norm(train_params["matrix"]), radius, rtol=1e-6)
        assert jnp.allclose(jnp.linalg.norm(params["matrix"]), radius, rtol=1e-6)


def test_schedule_free_plus_hyperball_normalizes_adam_direction_by_default():
    module = _load_fused_optimizers_module()
    params = {"matrix": jnp.array([[3.0, 4.0], [0.0, 0.0]], dtype=jnp.float32)}
    grads = {"matrix": jnp.array([[1.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)}
    radius = jnp.linalg.norm(params["matrix"])

    def one_step(normalize_update=None):
        optimizer_kwargs = {
            "mask": True,
            "beta": 0.0,
            "beta_final": None,
            "r": 0.0,
            "b1": 0.0,
            "b2": 0.0,
        }
        if normalize_update is not None:
            optimizer_kwargs["normalize_update"] = normalize_update
        tx = module.schedule_free_plus_hyperball_adamw(
            0.1,
            **optimizer_kwargs,
        )
        state = tx.init(params)
        updates, _state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates)["matrix"]

    default_params = one_step()
    raw_params = one_step(normalize_update=False)
    normalized_params = one_step(normalize_update=True)
    direction = jnp.array([[1.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)
    expected_raw_trial = params["matrix"] - 0.1 * direction
    expected_normalized_trial = params["matrix"] - 0.1 * radius * direction / jnp.linalg.norm(direction)
    expected_raw = radius * expected_raw_trial / jnp.linalg.norm(expected_raw_trial)
    expected_normalized = radius * expected_normalized_trial / jnp.linalg.norm(expected_normalized_trial)

    assert jnp.allclose(default_params, expected_normalized, rtol=1e-6)
    assert jnp.allclose(raw_params, expected_raw, rtol=1e-6)
    assert jnp.allclose(normalized_params, expected_normalized, rtol=1e-6)
    assert not jnp.allclose(raw_params, normalized_params, rtol=1e-6)


@pytest.mark.parametrize(
    ("optimizer_name", "base_optimizer"),
    [*_HYPERBALL_ALIAS_CASES, *_HYPERBALL_GENERIC_CASES],
    ids=[
        *(f"alias-{optimizer_name}" for optimizer_name, _base_optimizer in _HYPERBALL_ALIAS_CASES),
        *(f"generic-{base_optimizer}" for _optimizer_name, base_optimizer in _HYPERBALL_GENERIC_CASES),
    ],
)
def test_hyperball_variants_converge_on_scale_invariant_direction_loss(optimizer_name, base_optimizer):
    module = _load_fused_optimizers_module()
    key = jax.random.PRNGKey(42)
    initial_weight = jax.random.normal(key, (8, 8), dtype=jnp.float32)
    target_weight = jax.random.normal(jax.random.PRNGKey(1), (8, 8), dtype=jnp.float32)
    target_weight = target_weight / jnp.linalg.norm(target_weight)
    initial_params = {"matrix": initial_weight}

    def loss_fn(params):
        matrix_direction = params["matrix"] / jnp.linalg.norm(params["matrix"])
        return 0.5 * jnp.sum((matrix_direction - target_weight) ** 2)

    optimizer_config = (
        _schedule_free_plus_hyperball_config(module)
        if optimizer_name == "schedule_free_plus_hyperball" and base_optimizer is None
        else _hyperball_config(module, base_optimizer)
    )
    is_schedule_free_hyperball = optimizer_name == "schedule_free_plus_hyperball" or base_optimizer == "schedule_free_plus"
    learning_rate = 1e-2
    steps = 1500 if is_schedule_free_hyperball else 300

    tx, _scheduler = OptimizerFactory.create(
        optimizer_name,
        SchedulerConfig(learning_rate=learning_rate),
        optimizer_config,
    )
    initial_state = tx.init(initial_params)

    def body(carry, _):
        params, state = carry
        _loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = tx.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), None

    (params, _state), _ = jax.jit(lambda: jax.lax.scan(body, (initial_params, initial_state), None, length=steps))()

    assert loss_fn(params) < 1e-3
    assert jnp.allclose(jnp.linalg.norm(params["matrix"]), jnp.linalg.norm(initial_weight), rtol=1e-6)


@pytest.mark.parametrize("b1", [0.9, 0.5])
def test_novograd_grad_averaging_changes_updates(b1):
    """grad_averaging=True (paper/Ginsburg) must differ from False (optax default)."""
    module = sys.modules["easydel.trainers.fused_optimizers"]
    scheduler = optax.constant_schedule(0.01)
    params = {"w": jnp.array([[0.2, -0.7], [1.3, 0.4]], dtype=jnp.float32)}
    grads = {"w": jnp.sin(params["w"]) + 0.1}

    cfg_off = module.NovoGradConfig(b1=b1, b2=0.25, eps=1e-6, grad_averaging=False)
    cfg_on = module.NovoGradConfig(b1=b1, b2=0.25, eps=1e-6, grad_averaging=True)

    tx_off = module.NovoGradOptimizer(config=cfg_off).build(scheduler)
    tx_on = module.NovoGradOptimizer(config=cfg_on).build(scheduler)

    state_off = tx_off.init(params)
    state_on = tx_on.init(params)

    updates_off, state_off2 = tx_off.update(grads, state_off, params)
    updates_on, state_on2 = tx_on.update(grads, state_on, params)

    # updates must diverge on the 2nd step (first step is identical for both)
    updates_off2, _ = tx_off.update(grads, state_off2, params)
    updates_on2, _ = tx_on.update(grads, state_on2, params)

    assert not jnp.allclose(updates_off2["w"], updates_on2["w"]), (
        f"grad_averaging=True/False should diverge at step 2 for b1={b1}"
    )

    # Also verify that grad_averaging=True is exactly (1-b1) times the normalized
    # grad on the first step (since step 1 momentum == normalized for both)
    # At step 2, the momentum in the grad_averaging=True case should be
    # b1 * normalized + (1-b1) * normalized = normalized, so after step 1
    # both have momentum == normalized. But step 2 onwards diverges because
    # the new normalized input is scaled differently.
    # Let's just verify the first step explicitly is identical:
    assert jnp.allclose(updates_off["w"], updates_on["w"]), "step 1 must be identical"

