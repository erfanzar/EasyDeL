from functools import partial

import chex
import flax.core
import jax
from jax.experimental import pjit
from transformers import PretrainedConfig, GenerationConfig
from flax import linen as nn
from typing import Sequence, Optional, Type, Tuple, Any, Callable, Mapping
from jax import numpy as jnp
from ...modules.auto_easydel_model import AutoEasyDelModelForCausalLM
from ...modules.easydel_modelling_utils import EasyDelPretrainedConfig, EasyDelFlaxPretrainedModel
from fjformer import GenerateRNG, with_sharding_constraint, make_shard_and_gather_fns, match_partition_rules
from dataclasses import dataclass


@dataclass
class ValueHeadModuleOutput:
    summery: chex.Array
    logits: chex.Array


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


class ValueHead(nn.Module):
    summary_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")
    kernel_init: Callable = nn.initializers.orthogonal()

    def setup(self):
        """
        The setup function is called by the model's constructor.
        It initializes all the layers in your model, and assigns them to member variables.
        The setup function should be used for any initialization that needs to happen before running forward().
        This includes things like loading weights from a file, or setting up an optimizer.
        :param self: Represent the instance of the class
        """
        self.dropout = nn.Dropout(self.summary_dropout_prob)

        self.summary = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            use_bias=False
        )

    def __call__(self, hidden_states: chex.Array, deterministic: bool = True):
        """
        The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, e.g., x(arg).
        The __call__ method enables instances of a class to be called like standard Python functions.

        :param self: Represent the instance of the class
        :param hidden_states: chex.Array: Pass the hidden states of the previous layer
        :param deterministic: bool: Determine whether to use dropout
        :return: A tensor of shape (batch_size, num_classes)

        """
        return self.summary(self.dropout(hidden_states, deterministic=deterministic))


class AutoRLModelForCasualLMWithValueHead:
    def __init__(
            self,
            module: EasyDelFlaxPretrainedModel,
            config: EasyDelPretrainedConfig | PretrainedConfig,
            module_params: dict | flax.core.FrozenDict,
            summary_dropout_prob: float = 0.0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            kernel_init: Callable = nn.initializers.orthogonal(),
            generation_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp"),
            generation_config: GenerationConfig = GenerationConfig(),
            seed: int = 42
    ):
        self.module = module
        self.module_config = config
        self.generation_partition_spec = generation_partition_spec
        self.v_head = ValueHead(
            summary_dropout_prob=summary_dropout_prob,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init
        )
        self.seed = seed
        self._rng_generator = GenerateRNG(seed)

        self.module_params = self._check_parameters(module_params)

        self.generation_function = self.create_generation_function(generation_config)

    def _get_rng(self):
        return next(self._rng_generator)

    def _check_parameters(self, params):
        try:
            params = params["params"]
        except KeyError:
            ...

        resume_training = "v_head" in params.keys()
        if not resume_training:
            hidden_size = None
            if hasattr(self.module_config, "hidden_size"):
                hidden_size = self.module_config.hidden_size
            elif hasattr(self.module_config, "word_embed_proj_dim"):
                hidden_size = self.module_config.word_embed_proj_dim
            elif hasattr(self.module_config, "is_encoder_decoder"):
                if self.module_config.is_encoder_decoder and hasattr(self.module_config, "decoder"):
                    if hasattr(self.module_config, "hidden_size"):
                        hidden_size = self.module_config.decoder.hidden_size

            assert hidden_size is not None, "Seems like the models doesn't have any hidden_size"
            params = self.v_head.init(
                {
                    "params": self._get_rng()
                },
                jnp.ones(
                    (
                        1, hidden_size
                    )
                )
            )["params"] | params
        return params

    def create_generation_function(self, generation_config):
        @partial(
            pjit.pjit,
            in_shardings=(
                    match_partition_rules(
                        self.module.config.get_partition_rules(True),
                        self.module_params
                    ),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec()
            ),
            out_shardings=(
                    jax.sharding.PartitionSpec()
            )
        )
        def generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, self.generation_partition_spec)
            attention_mask = with_sharding_constraint(attention_mask, self.generation_partition_spec)
            predict = self.module.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=generation_config
            ).sequences[:, input_ids.shape[1]:]
            return predict

        return generate

    def get_mesh(self):
        return self.module.config.jax_mesh()

    def generate(self, input_id, attention_mask, params: dict | flax.core.FrozenDict = None):
        params = self.module_params if params is None else params
        return self.generation_function(
            params,
            input_id,
            attention_mask
        )

    def shard_parameters(self, params, partition_rules=None):
        partition_rules = self.module.config.get_partition_rules(True) if partition_rules is None else partition_rules
        with self.get_mesh():
            return jax.tree_util.tree_map(
                lambda f, p: f(p),
                make_shard_and_gather_fns(
                    match_partition_rules(
                        partition_rules,
                        params
                    )
                )[0],
                params
            )

    def __str__(self):
        padded_model = "\t" + "\n\t".join(self.module.__str__().split("\n"))
        string = f"{self.__class__.__name__}(\n{padded_model}\n)"
        return string

    def __call__(
            self,
            input_ids: chex.Array = None,
            past_key_values: chex.Array | dict = None,
            attention_mask: chex.Array = None,
            params: Optional[dict | flax.core.FrozenDict] = None,
            **kwargs,
    ) -> ValueHeadModuleOutput:
        params = self.module_params if params is None else params
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True
        kwargs["dropout_rng"] = self._get_rng()
        base_model_output = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            params=params,
            **kwargs
        )
        logits = base_model_output.logits
        last_hidden_state = base_model_output.hidden_states[-1]
        summery = self.v_head.apply(
            params,
            last_hidden_state,
            True
        )
        return ValueHeadModuleOutput(
            summery=summery,
            logits=logits
        )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            query_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            key_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Tuple[int, int] = (1, 1),
            backend: Optional[str] = None,
            kernel_init: Callable = nn.initializers.orthogonal(),
            generation_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec("dp", "fsdp"),
            generation_config: GenerationConfig = GenerationConfig(),
            shard_fns: Optional[Mapping[tuple, Callable]] = None,
            seed: int = 42,
            summary_dropout_prob: float = 0.0,
            **kwargs
    ):
        model, params = AutoEasyDelModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            query_partition_spec=query_partition_spec,
            key_partition_spec=key_partition_spec,
            value_partition_spec=value_partition_spec,
            bias_partition_spec=bias_partition_spec,
            attention_partition_spec=attention_partition_spec,
            use_shard_map=use_shard_map,
            input_shape=input_shape,
            shard_fns=shard_fns,
            backend=backend,
            **kwargs
        )
        rl_model = cls(
            module=model,
            dtype=dtype,
            module_params=params,
            precision=precision,
            param_dtype=param_dtype,
            config=model.config,
            kernel_init=kernel_init,
            summary_dropout_prob=summary_dropout_prob,
            seed=seed,
            generation_config=generation_config,
            generation_partition_spec=generation_partition_spec
        )

        return rl_model, rl_model.module_params
