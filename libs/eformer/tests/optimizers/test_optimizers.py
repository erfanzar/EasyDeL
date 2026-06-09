# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

import json

import jax
import jax.numpy as jnp
import optax
import pytest

from eformer.optimizers import (
    AdafactorConfig,
    AdafactorOptimizer,
    AdamWConfig,
    AdamWOptimizer,
    ConstantSchedulerBuilder,
    CosineSchedulerBuilder,
    LinearSchedulerBuilder,
    LionConfig,
    LionOptimizer,
    MarsConfig,
    MarsOptimizer,
    MuonConfig,
    MuonOptimizer,
    OptimizerBuilder,
    OptimizerFactory,
    QuadOptimizer,
    RMSPropConfig,
    RMSPropOptimizer,
    SchedulerBuilder,
    SchedulerConfig,
    SchedulerFactory,
    SkewOptimizer,
    WhiteKronConfig,
    make_stage_local_gradient_transformation,
    register_optimizer,
    register_scheduler,
)
from eformer.optimizers._base import _OPTIMIZER_BUILDER_REGISTRY, _SCHEDULER_BUILDER_REGISTRY
from eformer.optimizers._tx import create_cosine_scheduler, create_linear_scheduler


def _assert_tree_allclose(actual, expected, *, atol=1e-6, rtol=1e-6):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert jnp.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


class TestSerializationMixin:
    """Test the SerializationMixin functionality."""

    def test_to_dict_filters_private_fields(self):
        config = AdamWConfig()
        config._private_field = "should_not_appear"
        result = config.to_dict()
        assert "_private_field" not in result
        assert "b1" in result
        assert "b2" in result

    def test_from_dict_with_valid_data(self):
        data = {"b1": 0.95, "b2": 0.998, "eps": 1e-7}
        config = AdamWConfig.from_dict(data)
        assert config.b1 == 0.95
        assert config.b2 == 0.998
        assert config.eps == 1e-7

    def test_from_dict_with_extra_keys(self):
        data = {"b1": 0.95, "unknown_key": "value"}
        with pytest.warns(UserWarning, match="Ignoring unexpected keys"):
            config = AdamWConfig.from_dict(data)
        assert config.b1 == 0.95

    def test_to_json_serialization(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["b1"] == 0.95
        assert data["b2"] == 0.998

    def test_from_json_deserialization(self):
        json_str = '{"b1": 0.95, "b2": 0.998, "eps": 1e-7}'
        config = AdamWConfig.from_json(json_str)
        assert config.b1 == 0.95
        assert config.b2 == 0.998
        assert config.eps == 1e-7


class TestSchedulerConfig:
    """Test SchedulerConfig functionality."""

    def test_default_initialization(self):
        config = SchedulerConfig()
        assert config.scheduler_type is None
        assert config.learning_rate == 5e-5
        assert config.exponent == 1.0

    def test_linear_scheduler_validation(self):
        with pytest.raises(ValueError, match="Linear scheduler requires learning_rate_end"):
            SchedulerConfig(scheduler_type="linear", steps=1000)

    def test_scheduler_requires_steps(self):
        with pytest.raises(ValueError, match="Steps must be specified for non-constant schedulers"):
            SchedulerConfig(scheduler_type="cosine")

    def test_warmup_validation(self):
        with pytest.raises(ValueError, match="Steps required when using warmup"):
            SchedulerConfig(warmup_steps=100)

        with pytest.raises(ValueError, match="Warmup steps must be less than total steps"):
            SchedulerConfig(warmup_steps=1000, steps=500, scheduler_type="cosine")

    def test_valid_linear_config(self):
        config = SchedulerConfig(scheduler_type="linear", steps=1000, learning_rate_end=1e-6, warmup_steps=100)
        assert config.scheduler_type == "linear"
        assert config.steps == 1000
        assert config.learning_rate_end == 1e-6
        assert config.warmup_steps == 100

    def test_valid_cosine_config(self):
        config = SchedulerConfig(scheduler_type="cosine", steps=1000, warmup_steps=100)
        assert config.scheduler_type == "cosine"
        assert config.steps == 1000
        assert config.warmup_steps == 100


class TestOptimizerConfigs:
    """Test individual optimizer configuration classes."""

    def test_adafactor_config_defaults(self):
        config = AdafactorConfig()
        assert config.min_dim_size_to_factor == 128
        assert config.decay_rate == 0.8
        assert config.multiply_by_parameter_scale is True
        assert config.factored is True

    def test_adamw_config_defaults(self):
        config = AdamWConfig()
        assert config.b1 == 0.9
        assert config.b2 == 0.999
        assert config.eps == 1e-8
        assert config.eps_root == 0.0

    def test_lion_config_defaults(self):
        config = LionConfig()
        assert config.b1 == 0.9
        assert config.b2 == 0.99
        assert config.mu_dtype is None

    def test_rmsprop_config_defaults(self):
        config = RMSPropConfig()
        assert config.decay == 0.9
        assert config.initial_scale == 0.0
        assert config.momentum is None
        assert config.nesterov is False

    def test_muon_config_defaults(self):
        config = MuonConfig()
        assert config.ns_coeffs == (3.4445, -4.775, 2.0315)
        assert config.ns_steps == 5
        assert config.beta == 0.95
        assert config.nesterov is True

    def test_mars_config_defaults(self):
        config = MarsConfig()
        assert config.weight_decay == 0.1
        assert config.beta1 == 0.95
        assert config.beta2 == 0.99
        assert config.gamma == 0.025

    def test_white_kron_config_defaults(self):
        config = WhiteKronConfig()
        assert config.lr_style == "adam"
        assert config.b1 == 0.95
        assert config.normalize_grads is False
        assert config.max_size_dense == 16384
        assert config.preconditioner_lr == 0.7
        assert config.preconditioner_init_scale == 1.0
        assert config.dtype == jnp.bfloat16
        assert config.block_size == 256
        assert config.pipeline_axis_size == 1
        assert config.noise_scale == 1e-9
        assert config.weight_decay == 0.1


class TestSchedulerFactory:
    """Test SchedulerFactory functionality."""

    def test_constant_scheduler(self):
        config = SchedulerConfig(learning_rate=0.001)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == 0.001
        assert scheduler(1000) == 0.001

    def test_linear_scheduler_without_warmup(self):
        config = SchedulerConfig(scheduler_type="linear", learning_rate=0.001, learning_rate_end=0.0001, steps=1000)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(0.001, rel=1e-3)
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_linear_scheduler_with_warmup(self):
        config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.001, learning_rate_end=0.0001, steps=1000, warmup_steps=100
        )
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(1e-8, rel=1e-2)
        assert scheduler(100) == pytest.approx(0.001, rel=1e-3)
        # Linear scheduler with warmup has slight precision differences
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-2)

    def test_linear_scheduler_with_warmup_uses_total_steps(self):
        config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.001, learning_rate_end=0.0001, steps=1000, warmup_steps=100
        )
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(config.warmup_steps) == pytest.approx(0.001, rel=1e-3)
        assert scheduler(config.steps) == pytest.approx(config.learning_rate_end, rel=1e-3)
        assert scheduler(config.steps - 1) > scheduler(config.steps)

    def test_cosine_scheduler_without_warmup(self):
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.001, steps=1000)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == 0.001
        assert scheduler(1000) < 0.001

    def test_cosine_scheduler_without_warmup_with_learning_rate_end(self):
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.001, learning_rate_end=0.0001, steps=1000)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_cosine_scheduler_with_warmup(self):
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.001, steps=1000, warmup_steps=100)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(1e-8, rel=1e-2)
        assert scheduler(100) == pytest.approx(0.001, rel=1e-3)
        assert scheduler(1000) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_scheduler_with_warmup_uses_total_steps_and_end_value(self):
        config = SchedulerConfig(
            scheduler_type="cosine",
            learning_rate=0.001,
            learning_rate_end=0.0001,
            steps=1000,
            warmup_steps=100,
            exponent=1.0,
        )
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_custom_scheduler(self):
        config = SchedulerConfig(steps=1000)

        def custom_fn(steps):
            return optax.constant_schedule(0.005)

        scheduler = SchedulerFactory.create_scheduler(config, custom_scheduler=custom_fn)
        assert scheduler(0) == 0.005

    def test_custom_scheduler_requires_steps(self):
        config = SchedulerConfig()

        def custom_fn(steps):
            return optax.constant_schedule(0.005)

        with pytest.raises(ValueError, match="Custom schedulers require steps configuration"):
            SchedulerFactory.create_scheduler(config, custom_scheduler=custom_fn)

    def test_unsupported_scheduler_type(self):
        config = SchedulerConfig(scheduler_type="unknown", steps=1000)
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            SchedulerFactory.create_scheduler(config)


class TestOptimizerFactory:
    """Test OptimizerFactory functionality."""

    def test_optimizer_registry(self):
        """Test that all expected optimizers are registered in the builder registry."""
        assert "adafactor" in _OPTIMIZER_BUILDER_REGISTRY
        assert "adamw" in _OPTIMIZER_BUILDER_REGISTRY
        assert "lion" in _OPTIMIZER_BUILDER_REGISTRY
        assert "rmsprop" in _OPTIMIZER_BUILDER_REGISTRY
        assert "muon" in _OPTIMIZER_BUILDER_REGISTRY
        assert "mars" in _OPTIMIZER_BUILDER_REGISTRY
        assert "skew" in _OPTIMIZER_BUILDER_REGISTRY
        assert "quad" in _OPTIMIZER_BUILDER_REGISTRY

    def test_unsupported_optimizer_type(self):
        scheduler_config = SchedulerConfig()
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.create("unknown", scheduler_config)

    def test_create_adamw_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig(b1=0.95, b2=0.998)

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_optimizer_with_defaults(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_optimizer_with_weight_decay(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, _scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config, weight_decay=0.01)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_with_gradient_clipping(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, _scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config, clip_grad=1.0)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_with_gradient_accumulation(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, _scheduler = OptimizerFactory.create(
            "adamw",
            scheduler_config,
            optimizer_config,
            gradient_accumulation_steps=4,
        )

        assert isinstance(optimizer, optax.GradientTransformation)
        assert hasattr(optimizer, "apply_gradients_stage_local")

    def test_create_skew_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(b1=0.9, preconditioner_lr=0.5)

        optimizer, scheduler = OptimizerFactory.create("skew", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_quad_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(normalize_grads=True, block_size=128)

        optimizer, scheduler = OptimizerFactory.create("quad", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_white_kron_with_defaults(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)

        optimizer, scheduler = OptimizerFactory.create("skew", scheduler_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_invalid_config_type(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        wrong_config = LionConfig()  # Wrong config for AdamW

        with pytest.raises(TypeError, match="Invalid config type"):
            OptimizerFactory.create("adamw", scheduler_config, wrong_config)

    def test_warmup_without_scheduler_type(self):
        with pytest.raises(ValueError, match="Steps required when using warmup"):
            SchedulerConfig(learning_rate=0.001, warmup_steps=100)

    def test_dtype_conversion(self):
        config = AdamWConfig()
        config.mu_dtype = "float16"
        OptimizerFactory._convert_dtypes(config)
        assert config.mu_dtype == jnp.float16

    def test_invalid_dtype_conversion(self):
        config = AdamWConfig()
        config.mu_dtype = "invalid_dtype"
        with pytest.raises(ValueError, match="Invalid dtype specified"):
            OptimizerFactory._convert_dtypes(config)

    def test_generate_template(self):
        template = OptimizerFactory.generate_template("adamw")
        assert "AdamWConfig" in template
        assert "b1" in template
        assert "b2" in template

    def test_generate_template_unknown_optimizer(self):
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.generate_template("unknown")

    def test_serialize_config_dict(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        result = OptimizerFactory.serialize_config(config, format="dict")
        assert isinstance(result, dict)
        assert result["b1"] == 0.95

    def test_serialize_config_json(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        result = OptimizerFactory.serialize_config(config, format="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["b1"] == 0.95

    def test_serialize_config_invalid_format(self):
        config = AdamWConfig()
        with pytest.raises(ValueError, match="Supported formats"):
            OptimizerFactory.serialize_config(config, format="xml")

    def test_deserialize_config_dict(self):
        data = {"b1": 0.95, "b2": 0.998}
        config = OptimizerFactory.deserialize_config("adamw", data, format="dict")
        assert isinstance(config, AdamWConfig)
        assert config.b1 == 0.95

    def test_deserialize_config_json(self):
        json_str = '{"b1": 0.95, "b2": 0.998}'
        config = OptimizerFactory.deserialize_config("adamw", json_str, format="json")
        assert isinstance(config, AdamWConfig)
        assert config.b1 == 0.95

    def test_deserialize_config_unknown_optimizer(self):
        data = {"b1": 0.95}
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.deserialize_config("unknown", data)

    def test_deserialize_config_invalid_data_type_json(self):
        with pytest.raises(TypeError, match="Expected string input for JSON format"):
            OptimizerFactory.deserialize_config("adamw", {"b1": 0.95}, format="json")

    def test_deserialize_config_invalid_data_type_dict(self):
        with pytest.raises(TypeError, match="Expected dictionary input for dict format"):
            OptimizerFactory.deserialize_config("adamw", "invalid", format="dict")

    def test_deserialize_config_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            OptimizerFactory.deserialize_config("adamw", {}, format="xml")


class TestBuilderPattern:
    """Test the builder pattern and registration system."""

    def test_optimizer_builder_abstract_base(self):
        """Test that OptimizerBuilder is abstract and requires build method."""
        from abc import ABC

        assert issubclass(OptimizerBuilder, ABC)

    def test_scheduler_builder_abstract_base(self):
        """Test that SchedulerBuilder is abstract and requires build method."""
        from abc import ABC

        assert issubclass(SchedulerBuilder, ABC)

    def test_adamw_builder_instantiation(self):
        """Test AdamWOptimizer builder can be instantiated with config."""
        config = AdamWConfig(b1=0.95, b2=0.998)
        builder = AdamWOptimizer(config=config)
        assert builder.config == config
        assert builder.config.b1 == 0.95

    def test_adamw_builder_build(self):
        """Test AdamWOptimizer builder builds correct transformation."""
        config = AdamWConfig(b1=0.95, b2=0.998, eps=1e-7)
        builder = AdamWOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

        # Test that it can initialize and update
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        state = tx.init(params)
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}
        updates, _new_state = tx.update(grads, state, params)
        assert "w" in updates

    def test_adamw_builder_build_has_no_hidden_weight_decay(self):
        """Test AdamWOptimizer builder does not inherit optax's default weight decay."""
        builder = AdamWOptimizer(config=AdamWConfig())
        scheduler = optax.constant_schedule(1.0)

        tx = builder.build(scheduler)
        params = {"w": jnp.array([1.0])}
        state = tx.init(params)
        zero_grads = {"w": jnp.array([0.0])}
        updates, _new_state = tx.update(zero_grads, state, params)

        assert jnp.allclose(updates["w"], jnp.array([0.0]))

    def test_factory_adamw_weight_decay_controls_updates(self):
        """Test factory weight_decay is the only default decay applied for AdamW."""
        scheduler_config = SchedulerConfig(learning_rate=1.0)
        params = {"w": jnp.array([1.0])}
        zero_grads = {"w": jnp.array([0.0])}

        no_decay_tx, _ = OptimizerFactory.create(
            "adamw",
            scheduler_config,
            AdamWConfig(),
            weight_decay=0.0,
        )
        no_decay_state = no_decay_tx.init(params)
        no_decay_updates, _ = no_decay_tx.update(zero_grads, no_decay_state, params)

        decay_tx, _ = OptimizerFactory.create(
            "adamw",
            scheduler_config,
            AdamWConfig(),
            weight_decay=0.5,
        )
        decay_state = decay_tx.init(params)
        decay_updates, _ = decay_tx.update(zero_grads, decay_state, params)

        assert jnp.allclose(no_decay_updates["w"], jnp.array([0.0]))
        assert jnp.allclose(decay_updates["w"], jnp.array([-0.5]))

    @pytest.mark.parametrize(
        ("optimizer_type", "optimizer_config"),
        [
            ("adamw", AdamWConfig()),
            ("lion", LionConfig()),
            ("rmsprop", RMSPropConfig()),
            ("adafactor", AdafactorConfig()),
            ("mars", MarsConfig(max_grad_norm=None)),
            ("muon", MuonConfig()),
            ("quad", WhiteKronConfig(dtype=jnp.float32, block_size=4, noise_scale=0.0)),
            ("skew", WhiteKronConfig(dtype=jnp.float32, block_size=4, noise_scale=0.0)),
        ],
    )
    def test_factory_stage_local_update_matches_optax_update(self, optimizer_type, optimizer_config):
        """The explicit PP-safe optimizer API matches the normal Optax update."""
        scheduler_config = SchedulerConfig(learning_rate=0.1)
        params = {
            "w": jnp.array([[1.0, -2.0], [0.5, -0.25]], dtype=jnp.float32),
            "b": jnp.array([0.25, -0.5], dtype=jnp.float32),
        }
        grads = {
            "w": jnp.array([[0.25, -0.5], [0.125, -0.25]], dtype=jnp.float32),
            "b": jnp.array([0.05, -0.1], dtype=jnp.float32),
        }

        tx, scheduler = OptimizerFactory.create(
            optimizer_type,
            scheduler_config,
            optimizer_config,
            weight_decay=0.01,
        )
        assert hasattr(tx, "apply_gradients_stage_local")
        expected_state_in = tx.init(params)
        actual_state_in = tx.init(params)
        expected_grads = jax.tree_util.tree_map(lambda x: x + jnp.asarray(0, x.dtype), grads)
        actual_grads = jax.tree_util.tree_map(lambda x: x + jnp.asarray(0, x.dtype), grads)

        updates, expected_state = tx.update(expected_grads, expected_state_in, params)
        expected_params = optax.apply_updates(params, updates)
        actual_params, actual_state = tx.apply_gradients_stage_local(
            params=params,
            grads=actual_grads,
            opt_state=actual_state_in,
            learning_rate_fn=scheduler,
        )

        _assert_tree_allclose(actual_params, expected_params, atol=1e-6, rtol=1e-6)
        _assert_tree_allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6)

    def test_factory_stage_local_global_clip_matches_optax_update(self):
        """Global clipping is applied explicitly before leafwise PP updates."""
        scheduler_config = SchedulerConfig(learning_rate=0.1)
        params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
        grads = {"w": jnp.array([10.0, -20.0], dtype=jnp.float32)}

        tx, scheduler = OptimizerFactory.create(
            "adamw",
            scheduler_config,
            AdamWConfig(),
            clip_grad=1.0,
        )
        state = tx.init(params)
        expected_grads = jax.tree_util.tree_map(lambda x: x + jnp.asarray(0, x.dtype), grads)
        actual_grads = jax.tree_util.tree_map(lambda x: x + jnp.asarray(0, x.dtype), grads)
        updates, expected_state = tx.update(expected_grads, state, params)
        expected_params = optax.apply_updates(params, updates)
        actual_params, actual_state = tx.apply_gradients_stage_local(
            params=params,
            grads=actual_grads,
            opt_state=state,
            learning_rate_fn=scheduler,
        )

        _assert_tree_allclose(actual_params, expected_params, atol=1e-6, rtol=1e-6)
        _assert_tree_allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6)

    def test_stage_local_metadata_keeps_extra_kwargs(self):
        """Unknown MPMD builder kwargs are preserved for optimizer extensions."""
        scheduler = optax.constant_schedule(0.1)
        builder = AdamWOptimizer(AdamWConfig())
        optimizer = optax.chain(builder.build(scheduler))

        tx = builder.build_mpmd(
            scheduler,
            optimizer=optimizer,
            weight_decay=0.01,
            custom_mpmd_option="enabled",
        )
        metadata = tx.update._eformer_stage_local_metadata

        assert metadata.weight_decay == 0.01
        assert metadata.extra_kwargs == {"custom_mpmd_option": "enabled"}

    def test_registered_optimizer_build_mpmd_hook_is_used(self):
        """Custom registered optimizers can provide their own PP-safe apply path."""
        import dataclasses

        @dataclasses.dataclass
        class CustomConfig:
            pass

        @register_optimizer("test_mpmd_custom")
        @dataclasses.dataclass
        class CustomOptimizer(OptimizerBuilder):
            config: CustomConfig

            def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                return optax.sgd(learning_rate=scheduler)

            def build_mpmd(
                self,
                scheduler: optax.Schedule,
                *,
                optimizer: optax.GradientTransformation,
                **tx_kwargs,
            ) -> optax.GradientTransformation:
                del scheduler, tx_kwargs

                def apply_fn(
                    *,
                    params,
                    grads,
                    opt_state,
                    learning_rate_fn=None,
                    delete_grads: bool = False,
                ):
                    del learning_rate_fn, delete_grads
                    updates, new_state = optimizer.update(grads, opt_state, params)
                    return optax.apply_updates(params, updates), new_state

                return make_stage_local_gradient_transformation(optimizer, apply_fn=apply_fn)

        try:
            scheduler_config = SchedulerConfig(learning_rate=0.1)
            params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
            grads = {"w": jnp.array([0.25, -0.5], dtype=jnp.float32)}

            tx, _scheduler = OptimizerFactory.create(
                "test_mpmd_custom",
                scheduler_config,
                CustomConfig(),
            )
            state = tx.init(params)
            updates, expected_state = tx.update(grads, state, params)
            expected_params = optax.apply_updates(params, updates)
            actual_params, actual_state = tx.apply_gradients_stage_local(
                params=params,
                grads=grads,
                opt_state=state,
            )

            _assert_tree_allclose(actual_params, expected_params, atol=1e-6, rtol=1e-6)
            _assert_tree_allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6)
        finally:
            del _OPTIMIZER_BUILDER_REGISTRY["test_mpmd_custom"]

    def test_registered_optimizer_without_build_mpmd_gets_clear_pp_error(self):
        """Normal custom optimizers remain usable but fail loudly in PP mode."""
        import dataclasses

        @dataclasses.dataclass
        class NoMpmdConfig:
            pass

        @register_optimizer("test_no_mpmd")
        @dataclasses.dataclass
        class NoMpmdOptimizer(OptimizerBuilder):
            config: NoMpmdConfig

            def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                return optax.sgd(learning_rate=scheduler)

        try:
            params = {"w": jnp.array([1.0], dtype=jnp.float32)}
            grads = {"w": jnp.array([0.25], dtype=jnp.float32)}
            tx, _scheduler = OptimizerFactory.create(
                "test_no_mpmd",
                SchedulerConfig(learning_rate=0.1),
                NoMpmdConfig(),
            )
            state = tx.init(params)
            updates, _ = tx.update(grads, state, params)
            updated = optax.apply_updates(params, updates)
            assert jnp.allclose(updated["w"], jnp.array([0.975], dtype=jnp.float32))
            with pytest.raises(NotImplementedError, match="build_mpmd"):
                tx.apply_gradients_stage_local(params=params, grads=grads, opt_state=state)
        finally:
            del _OPTIMIZER_BUILDER_REGISTRY["test_no_mpmd"]

    def test_lion_builder_build(self):
        """Test LionOptimizer builder builds correct transformation."""
        config = LionConfig(b1=0.9, b2=0.99)
        builder = LionOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_adafactor_builder_build(self):
        """Test AdafactorOptimizer builder builds correct transformation."""
        config = AdafactorConfig(min_dim_size_to_factor=128, decay_rate=0.8)
        builder = AdafactorOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_rmsprop_builder_build(self):
        """Test RMSPropOptimizer builder builds correct transformation."""
        config = RMSPropConfig(decay=0.9, eps=1e-8)
        builder = RMSPropOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_muon_builder_build(self):
        """Test MuonOptimizer builder builds correct transformation."""
        config = MuonConfig(beta=0.95, ns_steps=5)
        builder = MuonOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_mars_builder_build(self):
        """Test MarsOptimizer builder builds correct transformation."""
        config = MarsConfig(beta1=0.95, beta2=0.99, gamma=0.025)
        builder = MarsOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_skew_builder_build(self):
        """Test SkewOptimizer builder builds correct transformation."""
        config = WhiteKronConfig(b1=0.9, preconditioner_lr=0.7, block_size=128)
        builder = SkewOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_quad_builder_build(self):
        """Test QuadOptimizer builder builds correct transformation."""
        config = WhiteKronConfig(b1=0.9, preconditioner_lr=0.7, max_size_dense=8192)
        builder = QuadOptimizer(config=config)
        scheduler = optax.constant_schedule(0.001)

        tx = builder.build(scheduler)
        assert isinstance(tx, optax.GradientTransformation)

    def test_constant_scheduler_builder_build(self):
        """Test ConstantSchedulerBuilder builds correct schedule."""
        config = SchedulerConfig(learning_rate=0.001)
        builder = ConstantSchedulerBuilder(config=config)

        schedule = builder.build()
        assert schedule(0) == 0.001
        assert schedule(1000) == 0.001

    def test_linear_scheduler_builder_build(self):
        """Test LinearSchedulerBuilder builds correct schedule."""
        config = SchedulerConfig(scheduler_type="linear", learning_rate=0.01, learning_rate_end=0.001, steps=1000)
        builder = LinearSchedulerBuilder(config=config)

        schedule = builder.build()
        assert schedule(0) == pytest.approx(0.01, rel=1e-3)
        assert schedule(1000) == pytest.approx(0.001, rel=1e-3)

    def test_linear_scheduler_builder_with_warmup_reaches_step_count(self):
        config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.01, learning_rate_end=0.001, steps=1000, warmup_steps=100
        )
        builder = LinearSchedulerBuilder(config=config)
        schedule = builder.build()
        assert schedule(config.steps) == pytest.approx(0.001, rel=1e-3)

    def test_cosine_scheduler_builder_build(self):
        """Test CosineSchedulerBuilder builds correct schedule."""
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.01, learning_rate_end=0.001, steps=1000)
        builder = CosineSchedulerBuilder(config=config)

        schedule = builder.build()
        assert schedule(0) == 0.01
        assert schedule(1000) == pytest.approx(0.001, rel=1e-3)

    def test_cosine_scheduler_builder_with_warmup_reaches_end(self):
        config = SchedulerConfig(
            scheduler_type="cosine",
            learning_rate=0.01,
            learning_rate_end=0.001,
            steps=1000,
            warmup_steps=100,
        )
        builder = CosineSchedulerBuilder(config=config)
        schedule = builder.build()
        assert schedule(1000) == pytest.approx(0.001, rel=1e-3)


class TestTxSchedulerHelpers:
    """Test tx scheduler helper functions."""

    def test_create_linear_scheduler_with_warmup_reaches_end(self):
        scheduler = create_linear_scheduler(
            steps=1000,
            learning_rate_start=0.001,
            learning_rate_end=0.0001,
            warmup_steps=100,
        )
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-2)

    def test_create_cosine_scheduler_without_warmup_with_end(self):
        scheduler = create_cosine_scheduler(
            steps=1000,
            learning_rate=0.001,
            learning_rate_end=0.0001,
        )
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_create_cosine_scheduler_with_warmup_reaches_end(self):
        scheduler = create_cosine_scheduler(
            steps=1000,
            learning_rate=0.001,
            learning_rate_end=0.0001,
            warmup_steps=100,
        )
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_builder_validate_hook_called(self):
        """Test that validate() hook is called during factory creation."""
        import dataclasses

        @dataclasses.dataclass
        class TestConfig:
            value: int = 10

        @register_optimizer("test_validate")
        @dataclasses.dataclass
        class TestOptimizer(OptimizerBuilder):
            config: TestConfig
            validate_called: bool = dataclasses.field(default=False, init=False)

            def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                return optax.sgd(learning_rate=scheduler)

            def validate(self) -> None:
                self.validate_called = True
                if self.config.value < 0:
                    raise ValueError("value must be non-negative")

        # Create using factory
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        config = TestConfig(value=5)

        # Factory should call validate
        tx, _ = OptimizerFactory.create("test_validate", scheduler_config, config)
        assert isinstance(tx, optax.GradientTransformation)

        # Clean up registry
        del _OPTIMIZER_BUILDER_REGISTRY["test_validate"]


class TestDecoratorRegistration:
    """Test the decorator-based registration system."""

    def test_register_optimizer_decorator(self):
        """Test @register_optimizer decorator registers a builder."""
        import dataclasses

        @dataclasses.dataclass
        class DummyConfig:
            lr: float = 0.001

        # Store original registry size
        original_size = len(_OPTIMIZER_BUILDER_REGISTRY)

        @register_optimizer("dummy_test")
        @dataclasses.dataclass
        class DummyOptimizer(OptimizerBuilder):
            config: DummyConfig

            def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                return optax.sgd(learning_rate=scheduler)

        # Check registration
        assert "dummy_test" in _OPTIMIZER_BUILDER_REGISTRY
        assert _OPTIMIZER_BUILDER_REGISTRY["dummy_test"] == DummyOptimizer
        assert len(_OPTIMIZER_BUILDER_REGISTRY) == original_size + 1

        # Clean up
        del _OPTIMIZER_BUILDER_REGISTRY["dummy_test"]

    def test_register_optimizer_duplicate_raises_error(self):
        """Test that registering duplicate optimizer name raises ValueError."""
        import dataclasses

        @dataclasses.dataclass
        class Config1:
            pass

        @register_optimizer("dup_test")
        @dataclasses.dataclass
        class Optimizer1(OptimizerBuilder):
            config: Config1

            def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                return optax.sgd(learning_rate=scheduler)

        # Try to register again with same name
        with pytest.raises(ValueError, match="already registered"):

            @register_optimizer("dup_test")
            @dataclasses.dataclass
            class Optimizer2(OptimizerBuilder):
                config: Config1

                def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
                    return optax.sgd(learning_rate=scheduler)

        # Clean up
        del _OPTIMIZER_BUILDER_REGISTRY["dup_test"]

    def test_register_scheduler_decorator(self):
        """Test @register_scheduler decorator registers a builder."""
        import dataclasses

        # Store original registry size
        original_size = len(_SCHEDULER_BUILDER_REGISTRY)

        @register_scheduler("dummy_scheduler")
        @dataclasses.dataclass
        class DummyScheduler(SchedulerBuilder):
            config: SchedulerConfig

            def build(self) -> optax.Schedule:
                return optax.constant_schedule(self.config.learning_rate)

        # Check registration
        assert "dummy_scheduler" in _SCHEDULER_BUILDER_REGISTRY
        assert _SCHEDULER_BUILDER_REGISTRY["dummy_scheduler"] == DummyScheduler
        assert len(_SCHEDULER_BUILDER_REGISTRY) == original_size + 1

        # Clean up
        del _SCHEDULER_BUILDER_REGISTRY["dummy_scheduler"]

    def test_register_scheduler_duplicate_raises_error(self):
        """Test that registering duplicate scheduler name raises ValueError."""
        import dataclasses

        @register_scheduler("dup_sched_test")
        @dataclasses.dataclass
        class Scheduler1(SchedulerBuilder):
            config: SchedulerConfig

            def build(self) -> optax.Schedule:
                return optax.constant_schedule(0.001)

        # Try to register again with same name
        with pytest.raises(ValueError, match="already registered"):

            @register_scheduler("dup_sched_test")
            @dataclasses.dataclass
            class Scheduler2(SchedulerBuilder):
                config: SchedulerConfig

                def build(self) -> optax.Schedule:
                    return optax.constant_schedule(0.002)

        # Clean up
        del _SCHEDULER_BUILDER_REGISTRY["dup_sched_test"]

    def test_all_builtin_optimizers_registered(self):
        """Test that all built-in optimizers are registered."""
        expected_optimizers = ["adamw", "adafactor", "lion", "rmsprop", "muon"]
        for name in expected_optimizers:
            assert name in _OPTIMIZER_BUILDER_REGISTRY, f"Optimizer '{name}' not registered"

    def test_all_custom_optimizers_registered(self):
        """Test that all custom optimizers are registered."""
        expected_custom = ["mars", "skew", "quad"]
        for name in expected_custom:
            assert name in _OPTIMIZER_BUILDER_REGISTRY, f"Custom optimizer '{name}' not registered"

    def test_all_schedulers_registered(self):
        """Test that all schedulers are registered."""
        expected_schedulers = ["constant", "linear", "cosine"]
        for name in expected_schedulers:
            assert name in _SCHEDULER_BUILDER_REGISTRY, f"Scheduler '{name}' not registered"

    def test_builder_registry_contains_correct_types(self):
        """Test that registry contains OptimizerBuilder subclasses."""
        for name, builder_cls in _OPTIMIZER_BUILDER_REGISTRY.items():
            assert issubclass(builder_cls, OptimizerBuilder), f"'{name}' is not an OptimizerBuilder subclass"

    def test_scheduler_registry_contains_correct_types(self):
        """Test that registry contains SchedulerBuilder subclasses."""
        for name, builder_cls in _SCHEDULER_BUILDER_REGISTRY.items():
            assert issubclass(builder_cls, SchedulerBuilder), f"'{name}' is not a SchedulerBuilder subclass"


class TestOptimizerIntegration:
    """Integration tests for optimizers with actual JAX operations."""

    def test_adamw_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (10, 10))}
        grads = {"w": jax.random.normal(key, (10, 10))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, _ = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (10, 10)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_lion_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (5, 5))}
        grads = {"w": jax.random.normal(key, (5, 5))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = LionConfig()

        optimizer, _ = OptimizerFactory.create("lion", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (5, 5)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_cosine_schedule_progression(self):
        scheduler_config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.01, steps=1000)
        optimizer_config = AdamWConfig()

        _, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        lr_start = scheduler(0)
        lr_mid = scheduler(500)
        lr_end = scheduler(1000)

        assert lr_start == 0.01
        assert lr_mid < lr_start
        assert lr_end < lr_mid

    def test_linear_schedule_progression(self):
        scheduler_config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.01, learning_rate_end=0.001, steps=1000
        )
        optimizer_config = AdamWConfig()

        _, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        lr_start = scheduler(0)
        lr_mid = scheduler(500)
        lr_end = scheduler(1000)

        assert lr_start == pytest.approx(0.01, rel=1e-3)
        assert lr_mid == pytest.approx(0.0055, rel=1e-3)
        assert lr_end == pytest.approx(0.001, rel=1e-3)

    def test_skew_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (8, 8))}
        grads = {"w": jax.random.normal(key, (8, 8))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(b1=0.9, block_size=4)

        optimizer, _ = OptimizerFactory.create("skew", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (8, 8)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_quad_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (6, 6))}
        grads = {"w": jax.random.normal(key, (6, 6))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(preconditioner_lr=0.5, max_size_dense=1000)

        optimizer, _ = OptimizerFactory.create("quad", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (6, 6)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_white_kron_with_different_shapes(self):
        """Test white_kron optimizers with different parameter shapes."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        params = {
            "dense": jax.random.normal(keys[0], (10, 20)),  # 2D matrix
            "bias": jax.random.normal(keys[1], (20,)),  # 1D vector
            "small": jax.random.normal(keys[2], (3, 3)),  # Small matrix
            "large": jax.random.normal(keys[3], (50, 100)),  # Larger matrix
        }
        grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(123), x.shape), params)

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(max_size_dense=1000, block_size=8)

        optimizer, _ = OptimizerFactory.create("skew", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Check all parameters were updated
        for key in params:
            assert key in new_params
            assert new_params[key].shape == params[key].shape
            assert not jnp.allclose(params[key], new_params[key])


if __name__ == "__main__":
    pytest.main([__file__])
