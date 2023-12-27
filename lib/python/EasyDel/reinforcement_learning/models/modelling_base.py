import json
import logging
import os

import chex
import flax.core
import jax
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError
from transformers import FlaxPreTrainedModel
from flax import linen as nn
from flax.serialization import from_bytes
import msgpack
from typing import Sequence, Optional, Type, Tuple

from ...modules.auto_easydel_model import AutoEasyDelModelForCausalLM


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


class FlaxPreTrainedModelWrapper(nn.Module):
    pretrained_model: Type[FlaxPreTrainedModel]
    supported_modules = ("v_head",)
    supported_rm_modules = ("score",)
    supported_pretrained_model_architectures = FlaxPreTrainedModel

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: jax.lax.Precision = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Tuple[int, int] = (1, 1),
            backend: Optional[str] = None,
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
            q_ps=q_ps,
            k_ps=k_ps,
            v_ps=v_ps,
            b_ps=b_ps,
            a_ps=a_ps,
            use_shard_map=use_shard_map,
            input_shape=input_shape,
            backend=backend,
            **kwargs
        )
        model = cls(pretrained_model=model)
        is_resuming_training = "v_head" in params.keys()
        if not is_resuming_training:
            params = model.post_init(
                params=params,
                input_shape=input_shape,
                head_name="v_head"
            )
        return model, params

    def post_init(
            self,
            params: dict | flax.core.FrozenDict,
            input_shape: Tuple[int, int],
            head_name: str = "v_head"
    ):
        r"""
        Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def compute_reward_score(
            self,
            pretrained_params: dict | flax.core.FrozenDict,
            input_ids: chex.Array = None,
            attention_mask: chex.Array = None,
            ppo_adapter_name="default",
            **kwargs
    ):

        """
        The compute_reward_score function is used to compute the reward score for a given input.
        The function takes in an input_ids tensor and returns a tensor of scores. The shape of the returned
        tensor will be (batch_size, sequence_length). The higher the score, the more likely that token should be kept.

        :param self: Represent the instance of the class
        :param pretrained_params: dict | flax.core.FrozenDict: parameters to be passed to the model
        :param input_ids: Pass the input tokens to the model
        :param attention_mask: Indicate which tokens are padding
        :param ppo_adapter_name: Set the adapter back to its original state
        :param kwargs: Pass a variable number of arguments to a function
        :return: The scores for the given input_ids
        
        """
        if not self.supports_rm_adapter:
            raise ValueError("This model does not support reward modeling adapter.")

        base_model_output = self.pretrained_model(
            params=pretrained_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        last_hidden_states = base_model_output.hidden_states[-1]
        scores = self.score(last_hidden_states)

        return scores
